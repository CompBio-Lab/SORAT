#!/usr/bin/env python3
"""Check whether any evaluation scan is also in the atrial nnU-Net training set.

The atrial nnU-Net was trained on the 70 MBAS 2024 training cases. MBAS and
the 2018 Atrial Segmentation Challenge both come from University of Utah
LGE-MRI, so the same scan may appear in both. MBAS stores half as many
slices as the 2018 release (e.g. 44 vs 88), so each evaluation image is also
compared with every second slice (both phases). For each evaluation image
(2018 challenge training and testing sets, AIM-ASD) this finds MBAS images of
the same (or slice-halved) shape and reports the best intensity correlation
after the axis flips and transposes that format conversion can introduce.
A correlation near 1 means the same scan. On these data the values are
clearly bimodal (>= 0.985 for shared scans, <= 0.71 otherwise), so 0.95 is
used as the threshold in the analysis.

Output: data_prep/mbas_overlap.csv
Run inside a container with SimpleITK + numpy:
  python data_prep/check_mbas_overlap.py
"""

import os
import csv
import itertools
from pathlib import Path

import numpy as np
import SimpleITK as sitk

MBAS = Path(os.environ["MBAS_TRAINING_ROOT"])  # MBAS 2024 training set (70 cases)
LASC = Path(os.environ["SORAT_DATA_ROOT"]) / "Atria_2018"
AIM = Path(os.environ["SORAT_DATA_ROOT"]) / "AIM-ASD" / "dicoms" / "AIM-ASD" / "ASD"


def load(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def variants(a):
    """Arrays equivalent to `a` up to axis flips/permutations (shape-preserving only)."""
    seen = set()
    for perm in itertools.permutations(range(3)):
        b = np.transpose(a, perm)
        for flips in itertools.product((False, True), repeat=3):
            c = b
            for ax, f in enumerate(flips):
                if f:
                    c = np.flip(c, ax)
            key = (perm, flips)
            if key not in seen:
                seen.add(key)
                yield key, c


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0


def main():
    mbas = {}
    for case in sorted(MBAS.glob("MBAS_*")):
        img = case / f"{case.name}_gt.nii.gz"
        a = load(img)
        mbas.setdefault(tuple(sorted(a.shape)), []).append((case.name, a[::4, ::4, ::4]))
    evals = [("LASC2018-train", p / "lgemri.nrrd") for p in sorted((LASC / "Training Set").iterdir())]
    evals += [("LASC2018-test", p / "lgemri.nrrd") for p in sorted((LASC / "Testing Set").iterdir())]
    evals += [("AIM-ASD", p / f"{p.name}-lge.nii") for p in sorted(AIM.iterdir())]
    rows = []
    for source, path in evals:
        full = load(path)
        candidates = [full, full[::2], full[1::2]]  # slice axis is axis 0 in SimpleITK arrays
        best = ("", 0.0)
        for a in candidates:
            for name, m in mbas.get(tuple(sorted(a.shape)), []):
                for _, v in variants(a):
                    small = v[::4, ::4, ::4]
                    if small.shape != m.shape:
                        continue
                    c = corr(small, m)
                    if c > best[1]:
                        best = (name, c)
        a = full
        rows.append({"source": source, "case": path.parent.name, "shape": "x".join(map(str, a.shape)),
                     "best_mbas_match": best[0], "correlation": round(best[1], 4)})
        print(rows[-1], flush=True)
    with open(Path(__file__).with_name("mbas_overlap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
