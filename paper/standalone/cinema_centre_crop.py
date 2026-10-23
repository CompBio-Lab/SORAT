#!/usr/bin/env python3
"""CineMA standalone run on the ACDC test set, without ground truth at inference.

This is the CineMA notebook (notebooks/segmentation_acdc_cinema.ipynb) with the
three changes that make its masks comparable, voxel for voxel, with SORAT's:

  1. Crop: the notebook's input comes from CineMA's official preprocess_pid,
     which crops each image around the ground-truth LV. Here the same official
     steps run (resample_spacing_4d to 1 x 1 x 10 mm, crop_4d to 192 x 192,
     clip_and_normalise_intensity_4d, cast_to_uint8) with a centre crop instead.
  2. Grid: the notebook scores masks on CineMA's resampled, cropped grid. Here
     each mask is resampled (nearest neighbour) back onto the original frame's
     grid, using the cropped image's physical geometry.
  3. Frames: the notebook takes frame 0 as ED and the smallest predicted LV as
     ES. Here ED and ES come from Info.cfg (frame numbers count from 1).

Model loading and inference are the notebook's code (convunetr.get_model,
SpatialPadd to 192 x 192 x 16 at the end, intensity / 255, argmax), with the
checkpoints in the CineMA container (/models/cinema), the ones SORAT uses.
Nothing here imports SORAT code.

Output: <run_dir>/predictions/{pid}_{ED,ES}_{dataset}_seed{seed}.nii.gz in the
original geometry, and <run_dir>/timing.json.

Usage (in containers/sorat-cinema.sif, on a GPU node):
    python standalone/cinema_centre_crop.py --run_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import Compose, SpatialPadd
from omegaconf import OmegaConf
from safetensors.torch import safe_open

from cinema.data.acdc import ACDC_SAX_SLICE_SIZE, ACDC_SPACING
from cinema.data.acdc.preprocess import load_config
from cinema.data.sitk import (
    cast_to_uint8,
    clip_and_normalise_intensity_4d,
    crop_4d,
    get_center_crop_size_from_bbox,
    resample_spacing_4d,
)
from cinema.segmentation.convunetr import get_model

ACDC_TEST = Path(os.environ["SORAT_DATA_ROOT"]) / "ACDC" / "database" / "testing"
MODEL_DIR = Path("/models/cinema/finetuned/segmentation")
DATASETS = ("acdc", "mnms", "mnms2")
SEEDS = (0, 1, 2)
VIEW = "sax"


def load(dataset: str, seed: int, device: torch.device):
    state = {}
    with safe_open(str(MODEL_DIR / f"{dataset}_{VIEW}" / f"{dataset}_{VIEW}_{seed}.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    model = get_model(OmegaConf.load(str(MODEL_DIR / f"{dataset}_{VIEW}" / "config.yaml")))
    model.load_state_dict(state)
    return model.eval().to(device)


def preprocess(pid_dir: Path) -> tuple[sitk.Image, dict]:
    """Official CineMA ACDC preprocessing with a centre crop in place of the ground-truth crop."""
    cfg = load_config(pid_dir / "Info.cfg")
    image = resample_spacing_4d(image=sitk.ReadImage(str(pid_dir / f"{pid_dir.name}_4d.nii.gz")),
                                is_label=False, target_spacing=ACDC_SPACING)
    size = image.GetSize()[:3]
    crop_lower, crop_upper = get_center_crop_size_from_bbox(
        bbox_min=(0, 0, 0), bbox_max=size, current_size=size, target_size=(*ACDC_SAX_SLICE_SIZE, size[2]))
    image = crop_4d(image, crop_lower, crop_upper)
    image = clip_and_normalise_intensity_4d(image, intensity_range=None)
    return cast_to_uint8(image), cfg


def frame_geometry(image_4d: sitk.Image, t: int) -> sitk.Image:
    """Frame t of a 4-D image as a 3-D image with the spatial geometry."""
    return sitk.Extract(image_4d, (*image_4d.GetSize()[:3], 0), (0, 0, 0, t))


def to_original(labels: np.ndarray, cropped_frame: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Labels (x, y, z) on the cropped grid, resampled onto the original frame grid."""
    pred = sitk.GetImageFromArray(np.transpose(labels).astype(np.uint8))
    pred.CopyInformation(cropped_frame)
    return sitk.Resample(pred, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.run_dir / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    transform = Compose([SpatialPadd(keys=VIEW, spatial_size=(192, 192, 16), method="end")])
    t0, timing = time.time(), {"device": str(device), "dtype": str(dtype)}

    patients = sorted(d for d in ACDC_TEST.iterdir() if d.is_dir())
    pre = {}
    for d in patients:
        pre[d.name] = preprocess(d)
    timing["preprocess_s"] = time.time() - t0

    for dataset in DATASETS:
        for seed in SEEDS:
            model = load(dataset, seed, device)
            for d in patients:
                pid = d.name
                image, cfg = pre[pid]
                images = np.transpose(sitk.GetArrayFromImage(image))  # (x, y, z, t)
                n_slices = images.shape[2]
                if n_slices > 16:
                    raise SystemExit(f"{pid}: {n_slices} slices after resampling; the model grid holds 16")
                for tag, frame in (("ED", int(cfg["ed_frame"])), ("ES", int(cfg["es_frame"]))):
                    t = frame - 1  # Info.cfg counts frames from 1
                    batch = transform({VIEW: torch.from_numpy(images[None, ..., t].astype(np.float32) / 255.0)})
                    batch = {k: v[None, ...].to(device=device, dtype=torch.float32) for k, v in batch.items()}
                    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                        logits = model(batch)[VIEW]
                    labels = torch.argmax(logits, dim=1)[0, ..., :n_slices].cpu().numpy()
                    reference = sitk.ReadImage(str(d / f"{pid}_frame{frame:02d}.nii.gz"))
                    mask = to_original(labels, frame_geometry(image, t), reference)
                    sitk.WriteImage(mask, str(out / f"{pid}_{tag}_{dataset}_seed{seed}.nii.gz"), useCompression=True)
            print(f"{dataset} seed {seed}: done ({time.time() - t0:.0f} s)", flush=True)
    timing["total_s"] = time.time() - t0
    (args.run_dir / "timing.json").write_text(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
