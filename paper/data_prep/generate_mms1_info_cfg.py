#!/usr/bin/env python3
"""
Write ACDC-style Info.cfg files for the M&Ms-1 test set and point the
samplesheet at them, without touching the dataset itself.

M&Ms-1 ships one 4-D ground-truth file per subject with labels on two frames
only, and gives the ED/ES frames (0-based) in the dataset information CSV.
SORAT reads Info.cfg as ACDC does (frames counted from 1), so each file gets
ED/ES = CSV index + 1.

For every subject the script checks that the CSV ED/ES frames are exactly the
annotated frames of the 4-D ground truth, and stops if any subject disagrees.

Run inside the sorat-cinema.sif container (needs nibabel + numpy):
    python paper/data_prep/generate_mms1_info_cfg.py \
        --mms_root "$SORAT_DATA_ROOT/M&Ms" \
        --samplesheet data/mms_testing_samplesheet.csv \
        --out_dir data/mms1_info
"""

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

INFO_CSV = "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"


def annotated_frames(gt_path: Path) -> list:
    gt = np.asanyarray(nib.load(str(gt_path)).dataobj)
    if gt.ndim != 4:
        raise ValueError(f"{gt_path} is not 4-D")
    return [t for t in range(gt.shape[-1]) if np.count_nonzero(gt[..., t]) > 10]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mms_root", required=True, type=Path)
    parser.add_argument("--samplesheet", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    info = {row["External code"]: row for row in csv.DictReader(open(args.mms_root / INFO_CSV))}
    with open(args.samplesheet) as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    out_dir = args.out_dir.resolve()
    problems = 0
    for row in rows:
        pid = row["patient_id"]
        meta = info.get(pid)
        if meta is None:
            print(f"{pid}: not in {INFO_CSV}")
            problems += 1
            continue
        ed, es = int(meta["ED"]), int(meta["ES"])
        image = nib.load(row["image"])
        n_frames = int(image.shape[-1])
        annotated = annotated_frames(Path(row["ground_truth"]))
        if sorted(annotated) != sorted({ed, es}):
            print(f"{pid}: CSV ED={ed} ES={es} but GT is annotated on frames {annotated}")
            problems += 1
            continue
        cfg_dir = out_dir / pid
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg = cfg_dir / "Info.cfg"
        cfg.write_text(
            f"ED: {ed + 1}\nES: {es + 1}\nGroup: {meta['Pathology']}\n"
            f"Vendor: {meta['VendorName']}\nCentre: {meta['Centre']}\nNbFrame: {n_frames}\n"
        )
        row["info_cfg"] = str(cfg)

    if problems:
        print(f"FAILED: {problems} subjects; samplesheet not updated")
        return 1

    with open(args.samplesheet, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Info.cfg files under {out_dir} and updated {args.samplesheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
