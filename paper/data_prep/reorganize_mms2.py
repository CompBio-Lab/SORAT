#!/usr/bin/env python3
"""
Reorganize the M&Ms-2 dataset into ACDC-style SA / LA trees.

Source layout (per patient folder NNN):
    dataset/NNN/{NNN}_{SA,LA}_{CINE,ED,ED_gt,ES,ES_gt}.nii.gz

Target layout (ACDC-style, axis-separated):
    SA/NNN/{NNN}_SA_CINE.nii.gz
    SA/NNN/{NNN}_SA_ED.nii.gz
    SA/NNN/{NNN}_SA_ED_gt.nii.gz
    SA/NNN/{NNN}_SA_ES.nii.gz
    SA/NNN/{NNN}_SA_ES_gt.nii.gz
    SA/NNN/Info.cfg
    (likewise LA/NNN/ ...)

For each patient * axis the ED / ES frame index inside the 4D CINE volume
is located by exact array comparison (np.allclose) against the supplied
3D ED / ES volumes.  Info.cfg is written with true 0-indexed frame indices,
the convention the SORAT pipeline expects (see bin/frame_manifest.py).

Files are *moved* (not copied) from the source tree, then the emptied
source per-patient folders are removed.  dataset_information.csv and
readme.txt at the dataset root are left untouched.
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

DATASET_ROOT = Path(os.environ["SORAT_DATA_ROOT"]) / "M&Ms_2" / "MnM2"
SRC = DATASET_ROOT / "dataset"
OUT_ROOT = DATASET_ROOT  # -> M&Ms_2/SA  M&Ms_2/LA
AXES = ("SA", "LA")
SUFFIXES = ("CINE", "ED", "ED_gt", "ES", "ES_gt")


def find_frame_idx(cine_4d: np.ndarray, frame_3d: np.ndarray):
    """Return the 0-indexed temporal index of frame_3d within cine_4d."""
    nframes = cine_4d.shape[-1]
    target = frame_3d
    for t in range(nframes):
        if np.array_equal(cine_4d[..., t], target):
            return t
    for t in range(nframes):
        if np.allclose(cine_4d[..., t], target):
            return t
    return None


def process_patient_axis(pid: str, axis: str, src_dir: Path, out_dir: Path, log):
    """Move the 5 axis files for one patient/axis and write Info.cfg."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cine_path = src_dir / f"{pid}_{axis}_CINE.nii.gz"
    ed_path = src_dir / f"{pid}_{axis}_ED.nii.gz"
    es_path = src_dir / f"{pid}_{axis}_ES.nii.gz"

    if not cine_path.exists():
        log(f"  [{axis}] MISSING CINE: {cine_path}")
        return False

    # Determine ED / ES frame indices from the 4D CINE.
    ed_idx = es_idx = None
    try:
        cine_img = nib.load(str(cine_path))
        cine = cine_img.get_fdata()
        if cine.ndim != 4:
            log(f"  [{axis}] CINE not 4D: shape={cine.shape} ({cine_path})")
            ed_idx, es_idx = 0, max(cine.shape[-1] - 1, 0)
        else:
            nframes = cine.shape[-1]
            if ed_path.exists():
                ed = nib.load(str(ed_path)).get_fdata()
                ed_idx = find_frame_idx(cine, ed)
            if es_path.exists():
                es = nib.load(str(es_path)).get_fdata()
                es_idx = find_frame_idx(cine, es)
            if ed_idx is None:
                ed_idx = 0
            if es_idx is None:
                es_idx = nframes - 1
    except Exception as e:
        log(f"  [{axis}] ERROR reading CINE/ED/ES: {e}")
        return False

    # Move the 5 files into the output directory.
    for suf in SUFFIXES:
        src_f = src_dir / f"{pid}_{axis}_{suf}.nii.gz"
        dst_f = out_dir / f"{pid}_{axis}_{suf}.nii.gz"
        if not src_f.exists():
            log(f"  [{axis}] MISSING file: {src_f}")
            return False
        shutil.move(str(src_f), str(dst_f))

    # Write ACDC-style Info.cfg with 0-indexed ED/ES frame indices.
    nframes = int(cine.shape[-1]) if cine.ndim == 4 else 1
    cfg_path = out_dir / "Info.cfg"
    with open(cfg_path, "w") as fh:
        fh.write(f"ED: {ed_idx}\n")
        fh.write(f"ES: {es_idx}\n")
        fh.write(f"NbFrame: {nframes}\n")

    log(f"  [{axis}] OK  ED={ed_idx} ES={es_idx} NbFrame={nframes}")
    return True


def main():
    if not SRC.exists():
        sys.exit(f"Source dataset not found: {SRC}")

    # Optional argv: a list of patient IDs to process (default: all found).
    requested = [a for a in sys.argv[1:] if not a.startswith("-")]

    sa_root = OUT_ROOT / "SA"
    la_root = OUT_ROOT / "LA"
    sa_root.mkdir(parents=True, exist_ok=True)
    la_root.mkdir(parents=True, exist_ok=True)

    all_dirs = sorted(
        d for d in SRC.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    if requested:
        want = set(requested)
        patient_dirs = [d for d in all_dirs if d.name in want]
        missing_ids = sorted(want - {d.name for d in patient_dirs})
        if missing_ids:
            print(f"NOTE: {len(missing_ids)} requested IDs not present in source "
                  f"(already processed?): {missing_ids[:5]}{'...' if len(missing_ids)>5 else ''}")
    else:
        patient_dirs = all_dirs

    if not patient_dirs:
        if requested:
            print("No matching patient folders found; requested set already processed.")
            return
        sys.exit(f"No patient folders found under {SRC}")

    report = {"ok": 0, "fail": 0, "patients": 0}

    def log(msg):
        print(msg, flush=True)

    for src_dir in patient_dirs:
        pid = src_dir.name
        report["patients"] += 1
        for axis, root in (("SA", sa_root), ("LA", la_root)):
            out_dir = root / pid
            if process_patient_axis(pid, axis, src_dir, out_dir, log):
                report["ok"] += 1
            else:
                report["fail"] += 1

        # Remove the source per-patient folder if it is now empty.
        remaining = list(src_dir.iterdir())
        if not remaining:
            try:
                src_dir.rmdir()
            except OSError:
                pass
        else:
            log(f"  NOTE: {src_dir} still has {len(remaining)} item(s): "
                + ", ".join(sorted(p.name for p in remaining[:5])))

    log("")
    log(f"Patients processed: {report['patients']}")
    log(f"Patient-axis folders OK: {report['ok']}")
    log(f"Patient-axis folders FAIL: {report['fail']}")
    if report["fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()