#!/usr/bin/env python3
"""
Discover existing segmentation frames for postprocess/feature extraction.
"""

import argparse
import csv
import json
import re
from pathlib import Path


def infer_architecture(model_tag: str) -> str:
    if model_tag.startswith("cinema"):
        return "cinema"
    if model_tag.startswith("nnformer"):
        return "nnformer"
    if model_tag.startswith("vsa3l"):
        return "vsa3l"
    if model_tag.startswith("atrial_nnunet"):
        return "atrial_nnunet"
    return "unknown"


def parse_samplesheet(path: Path) -> dict:
    rows = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("patient_id") or "").strip()
            if not pid:
                continue
            rows[pid] = {
                "image": (row.get("image") or "").strip(),
                "info_cfg": (row.get("info_cfg") or "").strip(),
            }
    return rows


def collect_frames(results_dir: Path, allowed_models: set[str]) -> list[dict]:
    """Collect per-frame segmentation files using a unified regex pattern."""
    frame_pattern = re.compile(r"^(.*)_(ED|ES|frame\d{2,})_(.+)\.nii\.gz$")

    rows = []
    for seg in results_dir.rglob("*_*.nii.gz"):
        if "postprocess" in {part.lower() for part in seg.parts}:
            continue

        name = seg.name
        match = frame_pattern.match(name)
        if not match:
            continue

        patient_id = match.group(1)
        frame_tag = match.group(2)
        model = match.group(3)

        if model.endswith("_pp"):
            continue

        arch = infer_architecture(model)
        if arch == "atrial_nnunet":
            continue
        if allowed_models and arch not in allowed_models and "all" not in allowed_models:
            continue

        # Try to read frame index from manifest if available
        frame_idx = _frame_index_from_tag(frame_tag)
        manifest_path = seg.parent / f"{patient_id}_{model}_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                for frm in manifest.get("frames", []):
                    if frm.get("tag") == frame_tag:
                        frame_idx = frm.get("idx", frame_idx)
                        break
            except Exception:
                pass

        rows.append({
            "patient_id": patient_id,
            "model": model,
            "architecture": arch,
            "frame_tag": frame_tag,
            "frame_idx": frame_idx,
            "seg": str(seg),
        })

    # Deduplicate: keep one row per (patient, model, frame_tag)
    seen = set()
    unique = []
    for row in rows:
        key = (row["patient_id"], row["model"], row["frame_tag"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _frame_index_from_tag(tag: str) -> int:
    """Extract integer frame index from a tag like 'frame05'."""
    if tag.isdigit():
        return int(tag)
    match = re.match(r"frame(\d+)", tag)
    if match:
        return int(match.group(1))
    if tag == "ED":
        return 0
    if tag == "ES":
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover postprocess inputs from segmentation frames")
    parser.add_argument("--samplesheet", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--models", default="all", help="comma-separated architecture filters")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    samplesheet_map = parse_samplesheet(Path(args.samplesheet))
    allowed = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    results_dir = Path(args.results_dir)
    found = collect_frames(results_dir, allowed)

    out_rows = []
    for row in found:
        meta = samplesheet_map.get(row["patient_id"], {})
        image = meta.get("image", "")

        if not image:
            continue
        out_rows.append({
            "patient_id": row["patient_id"],
            "model": row["model"],
            "architecture": row["architecture"],
            "frame_tag": row["frame_tag"],
            "frame_idx": str(row["frame_idx"]),
            "seg": row["seg"],
            "image": image,
            "info_cfg": meta.get("info_cfg", ""),
        })

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["patient_id", "model", "architecture", "frame_tag", "frame_idx", "seg", "image", "info_cfg"],
        )
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"Discovered {len(out_rows)} per-frame segmentation inputs")


if __name__ == "__main__":
    main()
