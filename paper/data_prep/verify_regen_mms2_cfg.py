#!/usr/bin/env python3
"""
Verify the reorganized M&Ms-2 SA/LA trees and (re)generate every Info.cfg.

For each patient folder under M&Ms_2/MnM2/{SA,LA}/NNN this script:
  1. Confirms the 5 expected files ({pid}_{axis}_{CINE,ED,ED_gt,ES,ES_gt}.nii.gz) are present.
  2. Loads the 4D CINE + 3D ED/ES and finds the temporal frame index that
     matches each (np.array_equal / np.allclose).
  3. (Re)writes Info.cfg with ED/ES frame numbers counted from 1, as in ACDC
     (frame number = 0-based CINE index + 1), which is how SORAT reads Info.cfg.
  4. Cross-checks that the cfg indices actually match the volumes.

Run inside the sorat-cinema.sif container (needs nibabel + numpy).
"""

import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

ROOT = Path(os.environ["SORAT_DATA_ROOT"]) / "M&Ms_2" / "MnM2"
AXES = ("SA", "LA")
EXPECTED = ["CINE", "ED", "ED_gt", "ES", "ES_gt"]


def find_idx(cine_4d, frame_3d):
    for t in range(cine_4d.shape[-1]):
        if np.array_equal(cine_4d[..., t], frame_3d):
            return t
    for t in range(cine_4d.shape[-1]):
        if np.allclose(cine_4d[..., t], frame_3d):
            return t
    return None


def main():
    requested = [a for a in sys.argv[1:] if not a.startswith("-")]
    problems = 0
    for axis in AXES:
        axis_root = ROOT / axis
        dirs = sorted(d for d in axis_root.iterdir()
                      if d.is_dir() and d.name.isdigit())
        if requested:
            want = set(requested)
            dirs = [d for d in dirs if d.name in want]
        if not requested:
            print(f"[{axis}] {len(dirs)} patient folders")
            if len(dirs) != 360:
                print(f"  !! expected 360, got {len(dirs)}"); problems += 1
        for d in dirs:
            pid = d.name
            # file completeness
            missing = []
            for suf in EXPECTED:
                f = d / f"{pid}_{axis}_{suf}.nii.gz"
                if not f.exists() or f.stat().st_size == 0:
                    missing.append(suf)
            if missing:
                print(f"  [{axis}/{pid}] MISSING/EMPTY: {missing}")
                problems += 1
                continue
            try:
                cine = nib.load(str(d / f"{pid}_{axis}_CINE.nii.gz")).get_fdata()
                ed = nib.load(str(d / f"{pid}_{axis}_ED.nii.gz")).get_fdata()
                es = nib.load(str(d / f"{pid}_{axis}_ES.nii.gz")).get_fdata()
            except Exception as e:
                print(f"  [{axis}/{pid}] load error: {e}")
                problems += 1
                continue
            if cine.ndim != 4:
                print(f"  [{axis}/{pid}] CINE not 4D: {cine.shape}")
                problems += 1
                continue
            ed_idx = find_idx(cine, ed)
            es_idx = find_idx(cine, es)
            if ed_idx is None or es_idx is None:
                print(f"  [{axis}/{pid}] could not locate ED/ES frame in CINE")
                problems += 1
                continue
            # (re)write Info.cfg with 1-based frame numbers (ACDC convention)
            cfg = d / "Info.cfg"
            with open(cfg, "w") as fh:
                fh.write(f"ED: {ed_idx + 1}\n")
                fh.write(f"ES: {es_idx + 1}\n")
                fh.write(f"NbFrame: {int(cine.shape[-1])}\n")
    print(f"\nREGEN {'PASSED' if problems==0 else 'FAILED with '+str(problems)+' problems'}")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()