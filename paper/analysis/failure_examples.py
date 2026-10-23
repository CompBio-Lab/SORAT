"""Qualitative failure examples (Supplementary Figure; CLAIM item 39).

Picks, from the held-out test sets, three representative error types:
  A  apical RV under-segmentation (CineMA-ACDC on M&Ms-2): the M&Ms-2 frame
     with the lowest apical RV slice Dice among frames whose mid-ventricular
     RV Dice is >= 0.85 (so the error is local, not a global failure);
  B  prediction on a slice without heart (VSA-3L on M&Ms-1): the frame with
     the largest share of predicted foreground off the reference heart;
  C  a typical external-data result (nnFormer on M&Ms-2): the frame closest
     to that model's median mean Dice.
Each panel shows the cine image, the manual contour and the prediction on the
relevant slice. Selection is deterministic from the saved tables.

Needs tables/slice_dice.csv.gz and tables/off_heart_predictions.csv.
Output: figures/fig_failure_examples.{png,pdf}, tables/failure_examples.csv
Run: python analysis/failure_examples.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

import common as C
from slice_metrics import load_gt

LABEL_CMAP = ListedColormap([(0, 0, 0, 0), (0.92, 0.41, 0.20, 0.55), (0.11, 0.69, 0.48, 0.55), (0.16, 0.47, 0.84, 0.55)])
FOLDER = {"ACDC": "ACDC", "MMS1": "MMS", "MMS2": "MMS_2"}
ARCH = {"CineMA-ACDC": ("cinema", "cinema__acdc_ensemble"), "VSA-3L": ("vsa3l", "vsa3l__model"), "nnFormer": ("nnformer", "nnformer__fold0")}


def image_frame(ds: str, pid: str, idx: int) -> np.ndarray:
    root = C.DATA_ROOT
    path = {"ACDC": root / "ACDC" / "database" / "testing" / pid / f"{pid}_4d.nii.gz",
            "MMS1": root / "M&Ms" / "Testing" / pid / f"{pid}_sa.nii.gz",
            "MMS2": root / "M&Ms_2" / "MnM2" / "SA" / pid / f"{pid}_SA_CINE.nii.gz"}[ds]
    return np.asanyarray(nib.load(path).dataobj)[..., idx].astype(float)


def prediction(ds: str, pid: str, tag: str, model: str) -> np.ndarray:
    arch, tagname = ARCH[model]
    f = C.RESULTS / FOLDER[ds] / arch / "segmentations" / f"{pid}_{tag}_{tagname}.nii.gz"
    return np.rint(np.asanyarray(nib.load(f).dataobj)).astype(np.int16)


def frame_index(ds: str, pid: str, tag: str) -> int:
    m = C.load_metrics(ds)
    return int(m[(m["patient_id"] == pid) & (m["frame_tag"] == tag)]["frame_idx"].iloc[0])


def pick() -> list[dict]:
    sl = pd.read_csv(C.TABLES / "slice_dice.csv.gz", dtype={"patient_id": str})
    sl["third"] = pd.cut(sl["position"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=["Base", "Mid", "Apex"])
    a = sl[(sl["dataset"] == "MMS2") & (sl["model_name"] == "CineMA-ACDC")]
    mid = a[a["third"] == "Mid"].groupby(["patient_id", "frame_tag"])["dice_RV"].mean()
    apex = a[(a["third"] == "Apex") & (a["gt_RV"] > 0)].sort_values("dice_RV")
    apex = apex[apex.set_index(["patient_id", "frame_tag"]).index.isin(mid[mid >= 0.85].index)]
    ra = apex.iloc[0]
    off = pd.read_csv(C.TABLES / "off_heart_predictions.csv", dtype={"patient_id": str})
    off["frac"] = off["pred_voxels_off_heart_slices"] / off["pred_voxels_total"]
    rb = off[(off["dataset"] == "MMS1") & (off["model_name"] == "VSA-3L")].sort_values("frac").iloc[-1]
    m = C.primary_subset(C.load_metrics("MMS2"))
    m = m[m["model_name"] == "nnFormer"].assign(d=lambda x: x[["dice_rv", "dice_myo", "dice_lv"]].mean(axis=1))
    rc = m.iloc[(m["d"] - m["d"].median()).abs().argsort().iloc[0]]
    return [
        {"panel": "A", "dataset": "MMS2", "model": "CineMA-ACDC", "patient_id": ra["patient_id"], "frame_tag": ra["frame_tag"],
         "slice": int(ra["slice"]), "note": f"apical slice, RV Dice {ra['dice_RV']:.2f}"},
        {"panel": "B", "dataset": "MMS1", "model": "VSA-3L", "patient_id": rb["patient_id"], "frame_tag": rb["frame_tag"],
         "slice": None, "note": f"{100 * rb['frac']:.0f}% of predicted foreground off the heart"},
        {"panel": "C", "dataset": "MMS2", "model": "nnFormer", "patient_id": rc["patient_id"], "frame_tag": rc["frame_tag"],
         "slice": None, "note": f"median case, mean Dice {rc['d']:.2f}"},
    ]


def main() -> None:
    cases = pick()
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(len(cases), 3, figsize=(7.2, 7.4))
    for row, case in zip(axes, cases):
        ds, pid, tag = case["dataset"], case["patient_id"], case["frame_tag"]
        idx = frame_index(ds, pid, tag)
        img, gt, pred = image_frame(ds, pid, idx), load_gt(ds, pid, tag, idx), prediction(ds, pid, tag, case["model"])
        if case["slice"] is None:
            if case["panel"] == "B":  # slice with most predicted foreground and no reference heart
                heart = np.isin(gt, (2, 3)).any(axis=(0, 1))
                counts = (pred > 0).sum(axis=(0, 1)) * (~heart)
                case["slice"] = int(np.argmax(counts))
            else:
                case["slice"] = int(np.argmax((gt == 3).sum(axis=(0, 1))))
        z = case["slice"]
        im = np.rot90(img[..., z])
        lo, hi = np.percentile(im, [1, 99.5])
        for ax, (title, lab) in zip(row, (("Cine image", None), ("Manual contour", gt), (case["model"], pred))):
            ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
            if lab is not None:
                ax.imshow(np.rot90(lab[..., z]), cmap=LABEL_CMAP, vmin=0, vmax=3, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=8)
        row[0].set_ylabel(f"{case['panel']}  {C.DATASETS[ds]['label']} {pid} {tag}\n{case['note']}", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LABEL_CMAP(i)) for i in (1, 2, 3)]
    fig.legend(handles, ["RV", "MYO", "LV"], loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_failure_examples.{ext}", dpi=300)
    pd.DataFrame(cases).to_csv(C.TABLES / "failure_examples.csv", index=False)
    print(pd.DataFrame(cases).to_string(index=False))


if __name__ == "__main__":
    main()
