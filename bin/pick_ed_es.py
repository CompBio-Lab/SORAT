#!/usr/bin/env python3
"""
Pick ED and ES from all-frame segmentations (--frames_mode all) and write Info.cfg.

ED is the frame with the largest LV cavity volume and ES the smallest. Writes
<patient_id>/Info.cfg (frames counted from 1), volume_curves.csv and ed_es.csv,
which flags ED more than 15% of the cycle from the first frame, frames without
an LV, and extra local minima in the LV curve.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk

DEFAULT_LABELS = "rv=1,myo=2,lv=3"


def parse_labels(text: str) -> Dict[str, int]:
    labels = {}
    for part in text.split(","):
        name, value = part.split("=")
        labels[name.strip()] = int(value)
    return labels


def frame_volumes(path: Path, labels: Dict[str, int]) -> Dict[str, float]:
    image = sitk.ReadImage(str(path))
    voxel_ml = float(np.prod(image.GetSpacing()[:3])) / 1000.0
    arr = sitk.GetArrayViewFromImage(image)
    return {f"{name}_ml": float(np.count_nonzero(arr == value)) * voxel_ml for name, value in labels.items()}


def local_minima(values: np.ndarray) -> List[int]:
    """Indices of strict local minima of a cyclic curve."""
    n = len(values)
    return [i for i in range(n) if values[i] < values[(i - 1) % n] and values[i] < values[(i + 1) % n]]


def pick(volumes: List[Dict[str, float]], indices: List[int], n_frames: int) -> dict:
    lv = np.array([v["lv_ml"] for v in volumes])
    valid = lv > 0
    if valid.sum() < 2:
        raise ValueError("fewer than two frames with an LV")
    ed = int(np.argmax(np.where(valid, lv, -np.inf)))
    es = int(np.argmin(np.where(valid, lv, np.inf)))
    ed_frame, es_frame = indices[ed], indices[es]
    cyclic = min(ed_frame, n_frames - ed_frame) / n_frames
    minima = [i for i in local_minima(lv[valid]) if lv[valid][i] != lv[es]]
    return {
        "ed_index": ed_frame, "es_index": es_frame, "ed_1based": ed_frame + 1, "es_1based": es_frame + 1,
        "lvedv_ml": round(float(lv[ed]), 2), "lvesv_ml": round(float(lv[es]), 2),
        "lvef_pct": round(100.0 * (lv[ed] - lv[es]) / lv[ed], 2),
        "ed_far_from_start": cyclic > 0.15, "empty_lv_frames": int((~valid).sum()), "extra_minima": len(minima),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick ED/ES from all-frame segmentations and write Info.cfg")
    parser.add_argument("--segmentations", required=True, type=Path,
                        help="Folder with <patient>_frameNN_<model_tag>.nii.gz and <patient>_<model_tag>_manifest.json")
    parser.add_argument("--model_tag", required=True, help="e.g. nnformer__fold0")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Label values of the masks (default ACDC: rv=1,myo=2,lv=3)")
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    if "lv" not in labels:
        parser.error("--labels must define lv")
    manifests = sorted(args.segmentations.glob(f"*_{args.model_tag}_manifest.json"))
    if not manifests:
        parser.error(f"no manifests for {args.model_tag} in {args.segmentations}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves, summary = [], []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        pid = manifest["patient_id"]
        frames = sorted((f for f in manifest["frames"] if re.fullmatch(r"frame\d+", f["tag"])), key=lambda f: f["idx"])
        n_frames = int(manifest.get("num_frames") or len(frames))
        try:
            if len(frames) < n_frames:
                raise ValueError(f"{len(frames)} of {n_frames} frames segmented; run with --frames_mode all")
            volumes = [frame_volumes(args.segmentations / f"{pid}_{f['tag']}_{args.model_tag}.nii.gz", labels)
                       for f in frames]
            result = pick(volumes, [f["idx"] for f in frames], n_frames)
        except (ValueError, RuntimeError) as err:
            summary.append({"patient_id": pid, "n_frames": n_frames, "error": str(err)})
            print(f"{pid}: {err}", file=sys.stderr)
            continue
        for f, v in zip(frames, volumes):
            curves.append({"patient_id": pid, "frame_index": f["idx"], **{k: round(x, 3) for k, x in v.items()}})
        summary.append({"patient_id": pid, "n_frames": n_frames, **result})
        cfg = args.out_dir / pid / "Info.cfg"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(f"ED: {result['ed_1based']}\nES: {result['es_1based']}\nNbFrame: {n_frames}\n"
                       f"Source: {args.model_tag} LV volume curve (pick_ed_es.py)\n")

    with open(args.out_dir / "volume_curves.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["patient_id", "frame_index"] + [f"{n}_ml" for n in labels])
        w.writeheader()
        w.writerows(curves)
    fields = ["patient_id", "n_frames", "ed_index", "es_index", "ed_1based", "es_1based", "lvedv_ml", "lvesv_ml",
              "lvef_pct", "ed_far_from_start", "empty_lv_frames", "extra_minima", "error"]
    with open(args.out_dir / "ed_es.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    n_ok = sum("error" not in s for s in summary)
    print(f"{n_ok} of {len(summary)} patients: Info.cfg written to {args.out_dir}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
