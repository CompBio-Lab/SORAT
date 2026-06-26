#!/usr/bin/env python3
"""
Universal Dataset Samplesheet Generator for SORAT Pipeline.

Auto-detects dataset structure and generates a CSV samplesheet with columns:
  patient_id, image, ground_truth, info_cfg

Supports:
  - ACDC layout: patient###/{id}_4d.nii.gz, per-frame GT, Info.cfg
  - M&Ms layout: {CASE_ID}/{CASE_ID}_sa.nii.gz, {CASE_ID}_sa_gt.nii.gz,
    ED/ES from metadata CSV
  - Generic layout: any case directories with .nii.gz files
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional


def _is_gt_candidate(name: str) -> bool:
    low = name.lower()
    return "_gt" in low or "_label" in low or "ground_truth" in low


def _find_image(case_dir: Path, case_id: str) -> Optional[Path]:
    nii_files = sorted(
        [f for f in case_dir.iterdir() if f.is_file() and f.name.endswith(".nii.gz")]
    )
    if not nii_files:
        return None

    gt_files = [f for f in nii_files if _is_gt_candidate(f.name)]
    non_gt = [f for f in nii_files if f not in gt_files]

    if not non_gt:
        if len(nii_files) == 1:
            return nii_files[0]
        return None

    candidates = [
        f"{case_id}_4d.nii.gz",
        f"{case_id}_sa.nii.gz",
        f"{case_id}.nii.gz",
    ]
    for cand in candidates:
        match = case_dir / cand
        if match in non_gt:
            return match

    if len(non_gt) == 1:
        return non_gt[0]

    return non_gt[0]


def _find_ground_truth(case_dir: Path, case_id: str) -> Optional[str]:
    has_frame_gt = any(
        f.name.startswith(case_id)
        and "_frame" in f.name
        and "_gt" in f.name
        and f.name.endswith(".nii.gz")
        for f in case_dir.iterdir()
        if f.is_file()
    )
    if has_frame_gt:
        return str(case_dir)

    candidates = [
        f"{case_id}_sa_gt.nii.gz",
        f"{case_id}_gt.nii.gz",
    ]
    for cand in candidates:
        match = case_dir / cand
        if match.exists():
            return str(match)

    for f in sorted(case_dir.iterdir()):
        if f.is_file() and f.name.endswith(".nii.gz") and "_gt" in f.name.lower():
            return str(f)

    return None


def _find_native_info_cfg(case_dir: Path) -> Optional[Path]:
    cfg = case_dir / "Info.cfg"
    return cfg if cfg.exists() else None


def _parse_metadata_csv(
    csv_path: Path,
    case_id_col: str,
    ed_col: str,
    es_col: str,
) -> dict[str, dict[str, Optional[int]]]:
    lookup: dict[str, dict[str, Optional[int]]] = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            print(f"Warning: metadata CSV {csv_path} has no header; skipping.", file=sys.stderr)
            return lookup

        header = [h.strip() for h in reader.fieldnames]

        if case_id_col not in header:
            alt_candidates = ["External code", "ExternalCode", "case_id", "PatientID", "ID"]
            found_alt = next((c for c in alt_candidates if c in header), None)
            if found_alt:
                case_id_col = found_alt
            else:
                print(
                    f"Warning: case ID column '{case_id_col}' not found in {csv_path}. "
                    f"Available columns: {header}. Skipping ED/ES mapping.",
                    file=sys.stderr,
                )
                return lookup

        if ed_col not in header or es_col not in header:
            ed_alt = next((c for c in header if c.upper() == "ED"), None)
            es_alt = next((c for c in header if c.upper() == "ES"), None)
            if ed_alt:
                ed_col = ed_alt
            if es_alt:
                es_col = es_alt

        for row in reader:
            cid = (row.get(case_id_col, "") or "").strip()
            if not cid:
                continue
            entry: dict[str, Optional[int]] = {"ed": None, "es": None}
            try:
                entry["ed"] = int(row[ed_col])
            except (KeyError, ValueError, TypeError):
                pass
            try:
                entry["es"] = int(row[es_col])
            except (KeyError, ValueError, TypeError):
                pass
            lookup[cid] = entry

    return lookup


def _discover_split_dir(data_root: Path, data_split: Optional[str]) -> Path:
    if data_split:
        split_dir = data_root / data_split
        if split_dir.is_dir():
            return split_dir

    subdirs = [d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {data_root}")

    for known in ["Testing", "testing", "test", "Training", "training", "Validation", "validation"]:
        candidate = data_root / known
        if candidate.is_dir():
            print(f"Auto-detected data split: {candidate.name}")
            return candidate

    raise FileNotFoundError(
        f"No recognizable data split found in {data_root}. "
        f"Subdirectories: {[d.name for d in subdirs]}. "
        f"Specify --data_split explicitly."
    )


def generate_samplesheet(
    data_root: Path,
    data_split: Optional[str],
    output_path: Path,
    metadata_csv: Optional[Path] = None,
    case_id_col: str = "External code",
    ed_col: str = "ED",
    es_col: str = "ES",
) -> int:
    data_dir = _discover_split_dir(data_root, data_split)

    if metadata_csv is None:
        csv_candidates = sorted(
            [f for f in data_root.iterdir()
             if f.is_file() and f.name.endswith('.csv') and not f.name.startswith('.')]
        )
        if csv_candidates:
            metadata_csv = csv_candidates[0]

    metadata_lookup: dict[str, dict[str, Optional[int]]] = {}
    if metadata_csv and metadata_csv.exists():
        metadata_lookup = _parse_metadata_csv(metadata_csv, case_id_col, ed_col, es_col)
        if metadata_lookup:
            print(f"Loaded ED/ES metadata for {len(metadata_lookup)} cases from {metadata_csv}")

    case_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    if not case_dirs:
        raise FileNotFoundError(f"No case directories found in {data_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    skipped = 0

    for case_dir in case_dirs:
        case_id = case_dir.name
        image = _find_image(case_dir, case_id)
        if image is None:
            print(f"Warning: no image found for {case_id}, skipping", file=sys.stderr)
            skipped += 1
            continue

        gt = _find_ground_truth(case_dir, case_id)
        native_info = _find_native_info_cfg(case_dir)
        meta = metadata_lookup.get(case_id, {})

        if native_info:
            info_cfg_path = str(native_info)
        else:
            info_cfg_path = ""

        rows.append(
            f"{case_id},{image.absolutePath() if hasattr(image, 'absolutePath') else image},{gt or ''},{info_cfg_path}"
        )

    if not rows:
        raise RuntimeError("No valid cases found to write to samplesheet.")

    header = "patient_id,image,ground_truth,info_cfg"
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for row in rows:
            fh.write(row + "\n")

    print(f"Generated samplesheet with {len(rows)} cases: {output_path}")
    if skipped:
        print(f"Skipped {skipped} case(s) due to missing image files.")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a SORAT samplesheet CSV from a dataset directory"
    )
    parser.add_argument(
        "--data_root", required=True, help="Root path of the dataset"
    )
    parser.add_argument(
        "--data_split",
        default=None,
        help="Subdirectory name within data_root (e.g., Testing, testing). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output", required=True, help="Output CSV path"
    )
    parser.add_argument(
        "--metadata_csv",
        default=None,
        help="Optional metadata CSV with ED/ES per case (e.g., M&Ms information CSV)",
    )
    parser.add_argument(
        "--case_id_col",
        default="External code",
        help="Column in metadata CSV for case identifier",
    )
    parser.add_argument(
        "--ed_col",
        default="ED",
        help="Column in metadata CSV for ED frame index",
    )
    parser.add_argument(
        "--es_col",
        default="ES",
        help="Column in metadata CSV for ES frame index",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        print(f"ERROR: data_root is not a directory: {data_root}", file=sys.stderr)
        sys.exit(1)

    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else None
    if metadata_csv and not metadata_csv.exists():
        print(f"Warning: metadata_csv not found: {metadata_csv}", file=sys.stderr)
        metadata_csv = None

    try:
        count = generate_samplesheet(
            data_root=data_root,
            data_split=args.data_split,
            output_path=Path(args.output),
            metadata_csv=metadata_csv,
            case_id_col=args.case_id_col,
            ed_col=args.ed_col,
            es_col=args.es_col,
        )
        sys.exit(0 if count > 0 else 1)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
