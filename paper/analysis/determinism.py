"""Run-to-run reproducibility: two independent full SORAT runs on ACDC (Table 2).

Compares every mask of the first run ($SORAT_RESULTS/ACDC) with a second run
started from a cold cache ($SORAT_RUNS/sorat_repeat/ACDC), and the DSC of each
frame between the two runs.

Outputs: tables/determinism.csv, tables/determinism_per_mask.csv
Run: python analysis/determinism.py
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd

import common as C

RUNS = C.RUNS


def determinism() -> pd.DataFrame:
    first, second = C.RESULTS / "ACDC", RUNS / "sorat_repeat" / "ACDC"
    rows = []
    for arch in ("cinema", "nnformer", "vsa3l"):
        for f in sorted((first / arch / "segmentations").glob("*.nii.gz")):
            g = second / arch / "segmentations" / f.name
            a, b = np.asanyarray(nib.load(f).dataobj), np.asanyarray(nib.load(g).dataobj)
            model = f.name.split("_", 2)[2].replace(".nii.gz", "")
            rows.append({"architecture": arch, "model": model, "file": f.name, "identical": bool(np.array_equal(a, b)),
                         "voxel_agreement": float((a == b).mean())})
    per_file = pd.DataFrame(rows)
    per_file.to_csv(C.TABLES / "determinism_per_mask.csv", index=False)
    m1 = pd.read_csv(first / "comparison" / "aggregated_metrics.csv", dtype={"patient_id": str})
    m2 = pd.read_csv(second / "comparison" / "aggregated_metrics.csv", dtype={"patient_id": str})
    keys = ["patient_id", "model", "frame_tag"]
    m = m1.merge(m2, on=keys, suffixes=("_1", "_2"))
    out = per_file.groupby("architecture").agg(masks=("identical", "size"), identical=("identical", "sum"),
                                                min_voxel_agreement=("voxel_agreement", "min")).reset_index()
    out["max_abs_dice_difference"] = [
        max((m.loc[m["model"].str.startswith(a), f"dice_{s}_1"] - m.loc[m["model"].str.startswith(a), f"dice_{s}_2"]).abs().max()
            for s in ("rv", "myo", "lv")) for a in out["architecture"]]
    return out



def main() -> None:
    C.TABLES.mkdir(exist_ok=True)
    det = determinism()
    det.to_csv(C.TABLES / "determinism.csv", index=False)
    print(det.to_string(index=False))


if __name__ == "__main__":
    main()
