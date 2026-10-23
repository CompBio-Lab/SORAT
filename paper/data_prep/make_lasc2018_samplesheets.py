#!/usr/bin/env python3
"""Write SORAT samplesheets for the 2018 Atrial Segmentation Challenge.

Rows point at the original .nrrd files; SORAT's PREPARE_INPUT step converts
them to NIfTI. The .nrrd headers store a placeholder spacing of 1 x 1 x 1, so
each row sets image_spacing to the true 0.625x0.625x1.25 mm (the same
Utah scans in the MBAS release store 44 slices at 2.5 mm against 88 here).
A column `in_mbas_training` marks the scans that are also in the atrial
nnU-Net's MBAS training data (data_prep/mbas_overlap.csv, correlation >= 0.95).

Usage: python3 data_prep/make_lasc2018_samplesheets.py --root <Atria_2018> --out <SORAT/data>
"""

import argparse
import csv
from pathlib import Path

SPACING = "0.625x0.625x1.25"
SPLITS = {"Training Set": "train", "Testing Set": "test"}
OVERLAP = Path(__file__).with_name("mbas_overlap.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    overlap = {r["case"]: float(r["correlation"]) >= 0.95 for r in csv.DictReader(open(OVERLAP))
               if r["source"].startswith("LASC2018")}
    for folder, split in SPLITS.items():
        rows = []
        for case in sorted(p for p in (args.root / folder).iterdir() if p.is_dir()):
            rows.append({"patient_id": case.name, "image": str(case / "lgemri.nrrd"),
                         "ground_truth": str(case / "laendo.nrrd"), "info_cfg": "",
                         "image_spacing": SPACING, "in_mbas_training": int(overlap[case.name])})
        out = args.out / f"lasc2018_{split}_samplesheet.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{out}: {len(rows)} rows, {sum(r['in_mbas_training'] for r in rows)} in MBAS training")


if __name__ == "__main__":
    main()
