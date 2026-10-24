"""Per-slice Dice for the main models, with each slice's position base→apex.

Reads SORAT's raw segmentations and the original ground truth for every
held-out test subject and frame, and writes one row per (patient, frame,
model, slice) to tables/slice_dice.csv.gz, plus the predicted foreground on
slices without any ground-truth heart to tables/off_heart_predictions.csv. This is the slow step (a few
thousand volumes); run it as a CPU batch job.

Slice position: the GT slices containing LV cavity or myocardium are numbered
from base (0) to apex (1). The basal end is the end of that range with the
larger LV cross-section, since the LV cavity narrows toward the apex.

Run: python analysis/slice_metrics.py
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd

import common as C

ARCH_DIR = {"cinema": "cinema", "nnformer": "nnformer", "vsa3l": "vsa3l"}
# Raw GT value -> SORAT label (1=RV, 2=MYO, 3=LV)
GT_LABEL_MAP = {"ACDC": {1: 1, 2: 2, 3: 3}, "MMS1": {1: 3, 2: 2, 3: 1}, "MMS2": {1: 3, 2: 2, 3: 1}}


def load_gt(dataset: str, pid: str, tag: str, frame_idx: int) -> np.ndarray:
    root = C.DATA_ROOT
    if dataset == "ACDC":
        arr = nib.load(root / "ACDC" / "database" / "testing" / pid / f"{pid}_frame{frame_idx + 1:02d}_gt.nii.gz").get_fdata()
    elif dataset == "MMS1":
        arr = np.asanyarray(nib.load(root / "M&Ms" / "Testing" / pid / f"{pid}_sa_gt.nii.gz").dataobj)[..., frame_idx]
    else:
        arr = nib.load(root / "M&Ms_2" / "MnM2" / "SA" / pid / f"{pid}_SA_{tag}_gt.nii.gz").get_fdata()
    arr = np.rint(arr).astype(np.int16)
    out = np.zeros_like(arr)
    for raw, canonical in GT_LABEL_MAP[dataset].items():
        out[arr == raw] = canonical
    return out


def slice_positions(gt: np.ndarray) -> dict[int, float]:
    """Map slice index (last axis) to position 0 = base ... 1 = apex."""
    heart = [z for z in range(gt.shape[-1]) if np.isin(gt[..., z], (2, 3)).any()]
    if len(heart) < 2:
        return {z: 0.5 for z in heart}
    first, last = heart[0], heart[-1]
    lv_first, lv_last = (gt[..., first] == 3).sum(), (gt[..., last] == 3).sum()
    base, apex = (first, last) if lv_first >= lv_last else (last, first)
    return {z: abs(z - base) / abs(apex - base) for z in heart}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else np.nan


def main() -> None:
    rows = []
    off_heart = []
    for ds in C.DATASETS:
        metrics = C.primary_subset(C.load_metrics(ds))
        metrics = metrics[metrics["model_name"].notna()]
        frames = metrics[["patient_id", "frame_tag", "frame_idx"]].drop_duplicates()
        folder = C.RESULTS / C.DATASETS[ds]["folder"]
        for _, fr in frames.iterrows():
            pid, tag, idx = fr["patient_id"], fr["frame_tag"], int(fr["frame_idx"])
            gt = load_gt(ds, pid, tag, idx)
            pos = slice_positions(gt)
            for model, name in C.MAIN_MODELS.items():
                arch = model.split("__")[0]
                seg_pid = str(int(pid)) if ds == "MMS2" and not (folder / ARCH_DIR[arch] / "segmentations" / f"{pid}_{tag}_{model}.nii.gz").exists() else pid
                seg_path = folder / ARCH_DIR[arch] / "segmentations" / f"{seg_pid}_{tag}_{model}.nii.gz"
                seg = np.rint(nib.load(seg_path).get_fdata()).astype(np.int16)
                if seg.shape != gt.shape:
                    raise ValueError(f"{seg_path}: shape {seg.shape} != GT {gt.shape}")
                outside = [z for z in range(gt.shape[-1]) if z not in pos]
                off_heart.append({"dataset": ds, "patient_id": pid, "frame_tag": tag, "model_name": name,
                                  "pred_voxels_total": int((seg > 0).sum()),
                                  "pred_voxels_off_heart_slices": int((seg[..., outside] > 0).sum()) if outside else 0,
                                  "off_heart_slices_with_prediction": int(sum((seg[..., z] > 0).any() for z in outside))})
                for z, p in pos.items():
                    row = {"dataset": ds, "patient_id": pid, "frame_tag": tag, "model_name": name,
                           "slice": z, "position": p}
                    for key, struct in C.STRUCTURES:
                        label = {"rv": 1, "myo": 2, "lv": 3}[key]
                        g, s = gt[..., z] == label, seg[..., z] == label
                        row[f"dice_{struct}"] = dice(s, g)
                        row[f"gt_{struct}"] = int(g.sum())
                        row[f"pred_{struct}"] = int(s.sum())
                    rows.append(row)
        print(ds, "done", flush=True)
    C.TABLES.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(C.TABLES / "slice_dice.csv.gz", index=False)
    pd.DataFrame(off_heart).to_csv(C.TABLES / "off_heart_predictions.csv", index=False)


if __name__ == "__main__":
    main()
