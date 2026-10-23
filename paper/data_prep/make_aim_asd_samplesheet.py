#!/usr/bin/env python3
"""Samplesheet for the AIM-ASD atrial septal defect scans.

Each case folder <root>/<case>/ holds the LGE image <case>-lge.nii and the
biatrial reference <case>-biatrial.nii. Cases without both files are skipped.

Usage: python paper/data_prep/make_aim_asd_samplesheet.py --root "$SORAT_DATA_ROOT/AIM-ASD/dicoms/AIM-ASD/ASD" \
           --out data/aim_asd_samplesheet.csv
"""

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for case in sorted(p for p in args.root.iterdir() if p.is_dir()):
        image, ref = case / f"{case.name}-lge.nii", case / f"{case.name}-biatrial.nii"
        if image.exists() and ref.exists():
            rows.append([case.name, str(image), str(ref), ""])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["patient_id", "image", "ground_truth", "info_cfg"])
        w.writerows(rows)
    print(f"{len(rows)} cases -> {args.out}")


if __name__ == "__main__":
    main()
