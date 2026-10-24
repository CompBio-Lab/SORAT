#!/usr/bin/env python3
"""
Generate ACDC-style samplesheets for the reorganized M&Ms-2 dataset.

Produces two CSVs in SORAT/data/:
    mms2_sa_samplesheet.csv
    mms2_la_samplesheet.csv

Header (mirrors data/acdc_testing_samplesheet.csv):
    patient_id,image,ground_truth,info_cfg

Each row:
    NNN, <abs>/M&Ms_2/MnM2/{SA,LA}/NNN/{NNN}_{SA,LA}_CINE.nii.gz,
         <abs>/M&Ms_2/MnM2/{SA,LA}/NNN, <abs>/M&Ms_2/MnM2/{SA,LA}/NNN/Info.cfg
"""

import os
import csv
import sys
from pathlib import Path

ROOT = Path(os.environ["SORAT_DATA_ROOT"]) / "M&Ms_2" / "MnM2"
DATA_DIR = Path(os.environ.get("SORAT_SAMPLESHEETS", Path(__file__).resolve().parents[2] / "data"))
AXES = {"SA": "mms2_sa_samplesheet.csv", "LA": "mms2_la_samplesheet.csv"}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for axis, fname in AXES.items():
        axis_root = ROOT / axis
        if not axis_root.exists():
            sys.exit(f"ERROR: {axis_root} not found; run the reorganization first.")

        patient_dirs = sorted(
            d for d in axis_root.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        if not patient_dirs:
            sys.exit(f"ERROR: no patient folders under {axis_root}")

        out_csv = DATA_DIR / fname
        rows = 0
        with open(out_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["patient_id", "image", "ground_truth", "info_cfg"])
            for pdir in patient_dirs:
                pid = pdir.name
                cine = pdir / f"{pid}_{axis}_CINE.nii.gz"
                cfg = pdir / "Info.cfg"
                if not cine.exists():
                    print(f"WARN: missing CINE for {pid} {axis}: {cine}", file=sys.stderr)
                    continue
                if not cfg.exists():
                    print(f"WARN: missing Info.cfg for {pid} {axis}: {cfg}", file=sys.stderr)
                writer.writerow(
                    [
                        pid,
                        str(cine.resolve()),
                        str(pdir.resolve()),
                        str(cfg.resolve()) if cfg.exists() else "",
                    ]
                )
                rows += 1
        print(f"{fname}: {rows} rows -> {out_csv}")
        if rows == 0:
            sys.exit(f"ERROR: no rows written for {axis}")


if __name__ == "__main__":
    main()