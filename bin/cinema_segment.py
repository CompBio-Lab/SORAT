#!/usr/bin/env python3
"""
CineMA Segmentation Script for SORAT Pipeline

Runs cardiac segmentation inference using the CineMA model.
"""

import argparse
import json
import re
from pathlib import Path
try:
    from frame_manifest import read_manifest, write_manifest
except ImportError:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.getcwd())
    from frame_manifest import read_manifest, write_manifest  # noqa: F401

import numpy as np
import SimpleITK as sitk
import torch


def load_model(model_dir: Path, trained_dataset: str, seed: int, device: torch.device):
    """Load CineMA segmentation model."""
    from safetensors.torch import safe_open
    from omegaconf import OmegaConf
    
    # Try to import cinema module
    try:
        from cinema.segmentation.convunetr import get_model
    except ImportError:
        # Fallback: add parent directory to path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from cinema.segmentation.convunetr import get_model
    
    view = "sax"
    model_path = model_dir / f"finetuned/segmentation/{trained_dataset}_{view}/{trained_dataset}_{view}_{seed}.safetensors"
    config_path = model_dir / f"finetuned/segmentation/{trained_dataset}_{view}/config.yaml"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # Load state dict
    state_dict = {}
    with safe_open(str(model_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    
    # Load config and initialize model
    config = OmegaConf.load(str(config_path))
    model = get_model(config)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model, config


def _fit_frame_to_patch(frame: np.ndarray, patch_size: tuple) -> tuple:
    """Crop/pad one ``(x, y, z)`` frame to the model patch size.

    CineMA checkpoints are configured for a fixed patch grid (currently
    192x192x16).  MONAI's ``SpatialPadd`` only padded smaller volumes, so
    patients with more than 16 slices reached the model as 17+ slices and
    crashed when the decoder reshaped tokens back to the fixed grid.  This
    helper normalizes the frame before inference and returns slices that map
    predictions back into the original preprocessed grid.
    """
    fitted = np.zeros(tuple(int(x) for x in patch_size), dtype=frame.dtype)

    src_slices = []
    dst_slices = []
    for current, target in zip(frame.shape, patch_size):
        current = int(current)
        target = int(target)
        if current > target:
            start = (current - target) // 2
            src_slices.append(slice(start, start + target))
            dst_slices.append(slice(0, target))
        else:
            src_slices.append(slice(0, current))
            dst_slices.append(slice(0, current))

    fitted[tuple(dst_slices)] = frame[tuple(src_slices)]
    return fitted, tuple(src_slices), tuple(dst_slices)


def _restore_patch_labels(
    labels_patch: np.ndarray,
    output_shape: tuple,
    src_slices: tuple,
    dst_slices: tuple,
) -> np.ndarray:
    """Place patch-size labels back into the original preprocessed grid."""
    restored = np.zeros(tuple(int(x) for x in output_shape), dtype=labels_patch.dtype)
    patch_slices = []
    restore_slices = []

    for src_slice, dst_slice, current, target in zip(src_slices, dst_slices, output_shape, labels_patch.shape):
        current = int(current)
        target = int(target)
        if current > target:
            restore_slices.append(src_slice)
            patch_slices.append(slice(0, target))
        else:
            restore_slices.append(slice(0, current))
            patch_slices.append(dst_slice)

    restored[tuple(restore_slices)] = labels_patch[tuple(patch_slices)]
    return restored


def run_inference(
    model,
    images: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    patch_size: tuple = (192, 192, 16)
) -> np.ndarray:
    """
    Run segmentation inference on 4D volume.
    
    Args:
        model: Loaded CineMA model
        images: Input 4D array (x, y, z, t)
        device: Torch device
        dtype: Data type for inference
        patch_size: Spatial patch size
    
    Returns:
        Segmentation array (x, y, z, t)
    """
    view = "sax"
    
    n_frames = images.shape[-1]
    labels_list = []
    
    for t in range(n_frames):
        frame = images[..., t].astype(np.float32)
        frame_patch, src_slices, dst_slices = _fit_frame_to_patch(frame, patch_size)

        # Prepare input. Shape is (batch, channel, x, y, z).
        batch = {view: torch.from_numpy(frame_patch[None, None, ...] / 255.0)}
        batch = {k: v.to(device=device, dtype=torch.float32) for k, v in batch.items()}
        
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.autocast("cuda", dtype=dtype, enabled=True):
                    logits = model(batch)[view]
            else:
                logits = model(batch)[view]
        
        labels_patch = torch.argmax(logits, dim=1)[0].detach().to(torch.uint8).cpu().numpy()
        labels_list.append(_restore_patch_labels(labels_patch, images.shape[:3], src_slices, dst_slices))
    
    labels = np.stack(labels_list, axis=-1).astype(np.float32)
    return labels


def _sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def _spatial_direction_from_any(direction: tuple) -> tuple:
    """Extract a valid 3x3 spatial direction from 3D or 4D direction cosines."""
    identity = (
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    )

    if len(direction) == 9:
        return tuple(direction)

    dim = int(round(len(direction) ** 0.5))
    if dim * dim != len(direction) or dim < 3:
        return identity

    try:
        mat = np.asarray(direction, dtype=np.float64).reshape(dim, dim)
        spatial = mat[:3, :3]
        if np.linalg.matrix_rank(spatial) < 3:
            return identity
        return tuple(float(x) for x in spatial.reshape(-1))
    except Exception:
        return identity


def _load_original_geometry(input_dir: Path, patient_id: str, fallback_sitk: sitk.Image) -> dict:
    """Load original-image geometry from preprocessed metadata.json.

    Returns a dict with ``size``, ``spacing``, ``origin``, ``direction`` (3x3
    spatial) of the *original* image, plus ``crop_origin`` and ``crop_spacing``
    describing the 192x192 preprocessed grid.  Falls back to the preprocessed
    4D volume's own geometry (origin 0, preprocessed spacing) when the new
    metadata fields are absent, preserving backward compatibility with caches
    produced by older pipeline versions.
    """
    import json

    metadata_path = Path(input_dir) / "metadata.json"
    meta = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}

    if all(k in meta for k in ("original_size_3d", "original_spacing_3d", "original_origin_3d", "crop_origin_3d")):
        return {
            "has_original_geometry": True,
            "size": [int(x) for x in meta["original_size_3d"]],
            "spacing": [float(x) for x in meta["original_spacing_3d"]],
            "origin": [float(x) for x in meta["original_origin_3d"]],
            "direction": [float(x) for x in meta.get("original_direction_3d", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])],
            "crop_origin": [float(x) for x in meta["crop_origin_3d"]],
            "crop_spacing": [float(x) for x in meta.get("target_spacing", [1.0, 1.0, 10.0])],
        }

    # Backward compat: no original-geometry metadata -> stay in preprocessed space.
    spacing = list(fallback_sitk.GetSpacing()[:3])
    origin = list(fallback_sitk.GetOrigin()[:3])
    direction = list(_spatial_direction_from_any(fallback_sitk.GetDirection()))
    return {
        "has_original_geometry": False,
        "size": list(fallback_sitk.GetSize()[:3]),
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
        "crop_origin": origin,
        "crop_spacing": spacing,
    }


def _resample_pred_to_original(
    pred_arr: np.ndarray,
    geometry: dict,
) -> sitk.Image:
    """Resample a 192x192 prediction array back into the original image space.

    The prediction array is in (x, y, z) order with the preprocessed grid's
    spacing/origin/direction.  We build a SimpleITK image with the *cropped*
    geometry (so physical-space mapping is correct) and resample it into a
    reference grid described by the original image geometry.  Nearest-neighbor
    interpolation preserves label values; the default fill is 0 (background).
    The returned image carries the original geometry so it aligns with the
    ground truth and the other architectures' segmentations.
    """
    pred_xyz = pred_arr  # (x, y, z)
    pred_zyx = np.transpose(pred_xyz, (2, 1, 0)).astype(np.uint8)

    crop_spacing = tuple(geometry["crop_spacing"][:3])
    crop_origin = tuple(geometry["crop_origin"][:3])
    crop_direction = tuple(geometry["direction"][:9])

    pred_sitk = sitk.GetImageFromArray(pred_zyx)
    pred_sitk.SetSpacing(crop_spacing)
    pred_sitk.SetOrigin(crop_origin)
    pred_sitk.SetDirection(crop_direction)

    ref_sitk = sitk.Image(
        [int(x) for x in geometry["size"][:3]],
        sitk.sitkUInt8,
    )
    ref_sitk.SetSpacing(tuple(float(x) for x in geometry["spacing"][:3]))
    ref_sitk.SetOrigin(tuple(float(x) for x in geometry["origin"][:3]))
    ref_sitk.SetDirection(tuple(float(x) for x in geometry["direction"][:9]))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    out = resampler.Execute(pred_sitk)
    out.CopyInformation(ref_sitk)
    return out


def segment_patient(
    input_dir: Path,
    patient_id: str,
    output_prefix: str,
    model_dir: Path,
    trained_dataset: str = 'acdc',
    seeds: list = [0],
    ensemble: bool = False,
    device: torch.device = None,
    model_tag: str = None
) -> dict:
    """
    Segment a single patient's cardiac MRI.
    
    Args:
        input_dir: Directory with preprocessed data
        patient_id: Patient identifier
        output_prefix: Prefix for output files
        model_dir: Directory containing model weights
        trained_dataset: Dataset the model was trained on
        seeds: List of random seeds to use
        ensemble: Whether to ensemble predictions across seeds
        device: Torch device
    
    Returns:
        Dictionary with segmentation results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dtype = torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    
    # Load preprocessed 4D volume
    image_path = input_dir / f"{patient_id}_sax_t.nii.gz"
    if not image_path.exists():
        raise FileNotFoundError(f"Preprocessed image not found: {image_path}")
    
    image_sitk = sitk.ReadImage(str(image_path))
    images = np.transpose(sitk.GetArrayFromImage(image_sitk))  # (x, y, z, t)

    # Load original-image geometry so predictions can be mapped back into the
    # original coordinate space (matching the ground truth + other models).
    geometry = _load_original_geometry(input_dir, patient_id, image_sitk)

    # Load frame manifest from preprocessed directory
    manifest_path = input_dir / f"{patient_id}_manifest.json"
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
    else:
        # Fallback: infer ED/ES from volumes (backward compat)
        n_frames = images.shape[-1]
        manifest = {
            "patient_id": patient_id,
            "has_info_cfg": False,
            "num_frames": n_frames,
            "frames": [
                {"tag": "ED", "idx": 0},
                {"tag": "ES", "idx": n_frames - 1 if n_frames > 1 else 0},
            ]
        }
    
    # Load ground truth references for spacing
    ed_gt_path = input_dir / f"{patient_id}_sax_ed_gt.nii.gz"
    es_gt_path = input_dir / f"{patient_id}_sax_es_gt.nii.gz"
    
    spacing = image_sitk.GetSpacing()[:3]
    origin = image_sitk.GetOrigin()[:3]
    direction = image_sitk.GetDirection()
    spatial_direction = _spatial_direction_from_any(direction)
    
    # Build model tag for filenames/metadata
    if model_tag is None:
        if ensemble and len(seeds) > 1:
            model_tag = f"cinema__{trained_dataset}_ensemble"
        elif len(seeds) == 1:
            model_tag = f"cinema__{trained_dataset}_seed{seeds[0]}"
        else:
            model_tag = f"cinema__{trained_dataset}_seeds{len(seeds)}"
    model_tag = _sanitize_tag(model_tag)

    # Run inference for each seed
    all_predictions = []
    for seed in seeds:
        print(f"  Running inference with seed {seed}...")
        model, config = load_model(model_dir, trained_dataset, seed, device)
        predictions = run_inference(model, images, device, dtype)
        all_predictions.append(predictions)
        del model
        torch.cuda.empty_cache()
    
    # Ensemble predictions if requested
    if ensemble and len(all_predictions) > 1:
        # Majority voting
        stacked = np.stack(all_predictions, axis=0)
        from scipy import stats
        labels, _ = stats.mode(stacked, axis=0, keepdims=False)
        labels = labels.squeeze()
    else:
        labels = all_predictions[0]
    
    # Extract and save frames according to manifest
    for frame in manifest["frames"]:
        tag = frame["tag"]
        idx = frame["idx"]
        if idx >= labels.shape[-1]:
            idx = labels.shape[-1] - 1
        pred = labels[..., idx].astype(np.uint8)
        if geometry["has_original_geometry"]:
            pred_sitk = _resample_pred_to_original(pred, geometry)
        else:
            pred_sitk = sitk.GetImageFromArray(np.transpose(pred, (2, 1, 0)))
            pred_sitk.SetSpacing(spacing)
            pred_sitk.SetOrigin(origin)
            pred_sitk.SetDirection(spatial_direction)
        sitk.WriteImage(pred_sitk, f"{output_prefix}_{tag}_{model_tag}.nii.gz", useCompression=True)
    
    # Write segment-level manifest confirming output files
    segment_manifest = {
        "patient_id": patient_id,
        "has_info_cfg": manifest.get("has_info_cfg", False),
        "num_frames": manifest.get("num_frames", len(manifest["frames"])),
        "frames": manifest["frames"],
    }
    write_manifest(segment_manifest, f"{output_prefix}_{model_tag}_manifest.json")

    results = {
        'patient_id': patient_id,
        'architecture': 'cinema',
        'model_tag': model_tag,
        'trained_dataset': trained_dataset,
        'seeds': seeds,
        'ensemble': ensemble,
        'frame_tags': [f["tag"] for f in manifest["frames"]],
        'frame_count': len(manifest["frames"]),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='CineMA segmentation for SORAT pipeline')
    parser.add_argument('--input_dir', required=True, help='Directory with preprocessed data')
    parser.add_argument('--patient_id', required=True, help='Patient identifier')
    parser.add_argument('--output_prefix', required=True, help='Output file prefix')
    parser.add_argument('--model_dir', required=True, help='Directory containing model weights')
    parser.add_argument('--trained_dataset', default='acdc', help='Training dataset')
    parser.add_argument('--seeds', default='0', help='Comma-separated list of seeds')
    parser.add_argument('--ensemble', action='store_true', help='Ensemble predictions')
    parser.add_argument('--model_tag', default=None, help='Model tag for output naming')
    
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    results = segment_patient(
        input_dir=Path(args.input_dir),
        patient_id=args.patient_id,
        output_prefix=args.output_prefix,
        model_dir=Path(args.model_dir),
        trained_dataset=args.trained_dataset,
        seeds=seeds,
        ensemble=args.ensemble,
        model_tag=args.model_tag
    )
    
    print(f"Segmentation complete for {args.patient_id}")
    print(f"Output frames: {results.get('frame_tags', [])}")


if __name__ == '__main__':
    main()
