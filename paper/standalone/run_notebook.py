#!/usr/bin/env python3
"""Run one of the original standalone segmentation notebooks as a timed script.

The notebooks in notebooks/ run each model outside SORAT on the ACDC test set,
following the preprocessing and inference code in the model's public
repository. They read CINEMA_ROOT, NNFORMER_ROOT, VSA3L_ROOT (model code and
weights) and ACDC_ROOT (ACDC/database) from the environment. This runner executes their code
cells in order, in one namespace, with a small, explicit list of text
substitutions so that

  * every output (preprocessed data, predictions, debug reports) goes to a
    fresh run directory instead of overwriting the original results, and
  * the VSA-3L notebook selects ED/ES with the ACDC convention (Info.cfg frame
    numbers count from 1).

Nothing else in the notebooks is changed. Cells that only plot or summarise
are skipped (listed per model below). Wall-clock time for the whole run and
per cell is written to <run_dir>/timing.json.

Usage (inside the analysis container, on a GPU node):
    python standalone/run_notebook.py --model nnformer --run_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(os.environ["SORAT_DATA_ROOT"])  # holds ACDC/
NOTEBOOK_DIR = Path(__file__).resolve().parent / "notebooks"

NOTEBOOKS = {
    "nnformer": NOTEBOOK_DIR / "segmentation_acdc_nnformer.ipynb",
    "cinema": NOTEBOOK_DIR / "segmentation_acdc_cinema.ipynb",
    "vsa3l": NOTEBOOK_DIR / "segmentation_acdc_monai.ipynb",
}

# (old, new[, cell_marker]) applied to each cell's source; with a marker, only
# to cells containing it. Each must match at least once, so a changed notebook
# fails loudly.
SUBSTITUTIONS = {
    "nnformer": [
        ('RESULTS_DIR = NNFORMER_ROOT / "results"', 'RESULTS_DIR = Path(os.environ["RUN_DIR"])'),
    ],
    "cinema": [
        ('PREPROCESSED_DIR = CINEMA_ROOT / "preprocessed"', 'PREPROCESSED_DIR = Path(os.environ["RUN_DIR"]) / "preprocessed"'),
        ('RESULTS_DIR = CINEMA_ROOT / "results"', 'RESULTS_DIR = Path(os.environ["RUN_DIR"])'),
    ],
    "vsa3l": [
        ('RESULTS_DIR = Path(BUNDLE_ROOT) / "results" / f"vsa3l_acdc_{DATASET_MODE}"', 'RESULTS_DIR = Path(os.environ["RUN_DIR"])'),
        ('DEBUG_DIR = Path(BUNDLE_ROOT) / "results" / "debug"', 'DEBUG_DIR = Path(os.environ["RUN_DIR"]) / "debug"'),
        # Info.cfg frame numbers count from 1 (ACDC); convert to 0-based array indices,
        # only where they index the 4-D image (the GT cell matches frameNN by number).
        ("ed_frame = int(line.split(':')[1].strip())", "ed_frame = int(line.split(':')[1].strip()) - 1",
         "Extracting 2D slices for ED and ES frames"),
        ("es_frame = int(line.split(':')[1].strip())", "es_frame = int(line.split(':')[1].strip()) - 1",
         "Extracting 2D slices for ED and ES frames"),
    ],
}

# Cells that draw per-patient comparison figures (not part of the timed pipeline).
SKIP_MARKERS = ("visualize_comparison(",)


def load_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def assemble_vsa3l(run_dir: Path) -> int:
    """Stack VSA-3L's 2-D slice predictions into 3-D masks in the original geometry.

    The notebook saves one 256x256 prediction per slice; slices it skipped
    (fewer than 100 non-zero pixels) are left as background. Each slice is
    resized back to the original in-plane size with nearest-neighbour
    interpolation and written with the 4-D image's affine, so the masks can be
    scored with the same code as SORAT's output. Not part of the timed run.
    """
    import re

    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk
    from skimage.transform import resize

    data_dir = ROOT / "ACDC" / "database" / "testing"
    out_dir = run_dir / "volumes"
    out_dir.mkdir(exist_ok=True)
    slices = {}
    for f in (run_dir / "predictions").glob("*_slice_*_seg.nii.gz"):
        m = re.match(r"(patient\d+)_(ED|ES)_slice_(\d+)_seg", f.name)
        if m:
            slices.setdefault((m[1], m[2]), {})[int(m[3])] = f
    for (pid, tag), files in slices.items():
        img = nib.load(data_dir / pid / f"{pid}_4d.nii.gz")
        nx, ny, nz = img.shape[:3]
        vol = np.zeros((nx, ny, nz), dtype=np.uint8)
        for z, f in files.items():
            seg = sitk.GetArrayFromImage(sitk.ReadImage(str(f)))
            vol[:, :, z] = resize(seg, (nx, ny), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        nib.save(nib.Nifti1Image(vol, img.affine), out_dir / f"{pid}_{tag}.nii.gz")
    return len(slices)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=sorted(NOTEBOOKS))
    parser.add_argument("--run_dir", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    os.environ["RUN_DIR"] = str(run_dir)

    cells = load_cells(NOTEBOOKS[args.model])
    for old, new, *marker in SUBSTITUTIONS[args.model]:
        targets = [i for i, c in enumerate(cells) if old in c and (not marker or marker[0] in c)]
        if not targets:
            raise SystemExit(f"Substitution target not found in {args.model} notebook: {old!r}")
        for i in targets:
            cells[i] = cells[i].replace(old, new)

    import matplotlib

    matplotlib.use("Agg")
    namespace = {"__name__": "__main__"}
    timing = {"model": args.model, "host": socket.gethostname(), "notebook": str(NOTEBOOKS[args.model]),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cells": []}
    os.chdir(NOTEBOOKS[args.model].parent)
    sys.path.insert(0, str(NOTEBOOKS[args.model].parent))
    t0 = time.perf_counter()
    for i, src in enumerate(cells):
        if any(m in src for m in SKIP_MARKERS):
            timing["cells"].append({"cell": i, "skipped": True})
            continue
        start = time.perf_counter()
        exec(compile(src, f"<{args.model} cell {i}>", "exec"), namespace)
        timing["cells"].append({"cell": i, "seconds": time.perf_counter() - start})
    timing["total_seconds"] = time.perf_counter() - t0
    if args.model == "vsa3l":
        timing["assembled_volumes"] = assemble_vsa3l(run_dir)
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    print(f"{args.model}: {timing['total_seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
