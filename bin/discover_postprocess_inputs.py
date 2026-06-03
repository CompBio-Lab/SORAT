#!/usr/bin/env python3
"""
Discover existing segmentation pairs (ED/ES) for postprocess-only execution.
"""

import argparse
import csv
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


def collect_pairs(results_dir: Path, allowed_models: set[str]) -> list[dict]:
    pairs = {}
    for seg in results_dir.rglob("*_ED_*.nii.gz"):
        # Ignore outputs from previous postprocess runs to avoid recursive _pp -> _pp_pp processing.
        if "postprocess" in {part.lower() for part in seg.parts}:
            continue

        name = seg.name
        stem = name[:-7] if name.endswith(".nii.gz") else seg.stem
        if "_ED_" not in stem:
            continue
        patient_id, model = stem.split("_ED_", 1)
        if model.endswith("_pp"):
            continue

        arch = infer_architecture(model)
        if arch == "atrial_nnunet":
            continue
        if allowed_models and arch not in allowed_models and "all" not in allowed_models:
            continue

        key = (patient_id, model)
        pairs.setdefault(key, {})["seg_ed"] = str(seg)

    for seg in results_dir.rglob("*_ES_*.nii.gz"):
        if "postprocess" in {part.lower() for part in seg.parts}:
            continue

        name = seg.name
        stem = name[:-7] if name.endswith(".nii.gz") else seg.stem
        if "_ES_" not in stem:
            continue
        patient_id, model = stem.split("_ES_", 1)
        if model.endswith("_pp"):
            continue

        arch = infer_architecture(model)
        if arch == "atrial_nnunet":
            continue
        if allowed_models and arch not in allowed_models and "all" not in allowed_models:
            continue

        key = (patient_id, model)
        pairs.setdefault(key, {})["seg_es"] = str(seg)

    rows = []
    for (patient_id, model), files in sorted(pairs.items()):
        if "seg_ed" not in files or "seg_es" not in files:
            continue
        rows.append(
            {
                "patient_id": patient_id,
                "model": model,
                "architecture": infer_architecture(model),
                "seg_ed": files["seg_ed"],
                "seg_es": files["seg_es"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover postprocess inputs")
    parser.add_argument("--samplesheet", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--models", default="all", help="comma-separated architecture filters")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    samplesheet_map = parse_samplesheet(Path(args.samplesheet))
    allowed = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    results_dir = Path(args.results_dir)
    found = collect_pairs(results_dir, allowed)

    out_rows = []
    for row in found:
        meta = samplesheet_map.get(row["patient_id"], {})
        image = meta.get("image", "")

        if not image:
            continue
        out_rows.append(
            {
                "patient_id": row["patient_id"],
                "model": row["model"],
                "architecture": row["architecture"],
                "seg_ed": row["seg_ed"],
                "seg_es": row["seg_es"],
                "image": image,
                "info_cfg": meta.get("info_cfg", ""),
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["patient_id", "model", "architecture", "seg_ed", "seg_es", "image", "info_cfg"],
        )
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"Discovered {len(out_rows)} postprocess pairs")


if __name__ == "__main__":
    main()
