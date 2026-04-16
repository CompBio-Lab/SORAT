#!/usr/bin/env python3
"""
Discover available model variants inside CASC containers.

Outputs a CSV of available models for a given architecture.
"""

import argparse
from pathlib import Path
import csv
import re


def _sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def discover_cinema(models_root: Path):
    seg_root = models_root / "cinema" / "finetuned" / "segmentation"
    results = []
    if not seg_root.exists():
        return results

    for model_dir in sorted(seg_root.glob("*_sax")):
        trained_dataset = model_dir.name.replace("_sax", "")
        config_path = model_dir / "config.yaml"
        if not config_path.exists():
            # Skip incomplete model directories
            continue
        seed_files = sorted(model_dir.glob(f"{model_dir.name}_*.safetensors"))
        for sf in seed_files:
            try:
                seed = int(sf.stem.split("_")[-1])
            except ValueError:
                continue
            model_tag = _sanitize_tag(f"cinema__{trained_dataset}_seed{seed}")
            results.append({
                "architecture": "cinema",
                "trained_dataset": trained_dataset,
                "seed": str(seed),
                "fold": "",
                "model_path": "",
                "model_tag": model_tag
            })
    return results


def discover_nnformer(models_root: Path):
    trainer_dir = models_root / "nnformer" / "nnFormer_trained_models" / "nnFormer" / "3d_fullres" / "Task001_ACDC" / "nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1"
    results = []
    if not trainer_dir.exists():
        return results

    fold_dirs = sorted([p for p in trainer_dir.glob("fold_*") if p.is_dir()])
    if fold_dirs:
        for fold_dir in fold_dirs:
            try:
                fold = int(fold_dir.name.split("_")[-1])
            except ValueError:
                continue
            model_tag = _sanitize_tag(f"nnformer__fold{fold}")
            results.append({
                "architecture": "nnformer",
                "trained_dataset": "",
                "seed": "",
                "fold": str(fold),
                "model_path": "",
                "model_tag": model_tag
            })
    else:
        # Fallback: single model in trainer_dir
        model_tag = _sanitize_tag("nnformer__fold0")
        results.append({
            "architecture": "nnformer",
            "trained_dataset": "",
            "seed": "",
            "fold": "0",
            "model_path": "",
            "model_tag": model_tag
        })

    return results


def discover_vsa3l(models_root: Path):
    vsa_root = models_root / "vsa3l"
    results = []
    if not vsa_root.exists():
        return results

    weight_files = sorted([p for p in vsa_root.glob("*.pt") if p.is_file()])
    for wf in weight_files:
        model_tag = _sanitize_tag(f"vsa3l__{wf.stem}")
        results.append({
            "architecture": "vsa3l",
            "trained_dataset": "",
            "seed": "",
            "fold": "",
            "model_path": str(wf),
            "model_tag": model_tag
        })

    return results


def discover_atrial_nnunet(models_root: Path):
    """Discover atrial nnUNetv2 model variants."""
    base = models_root / "atrial_nnunet"
    results = []

    if not base.exists():
        return results

    candidate_roots = [
        base / "nnUNet_results" / "Dataset001_LGE" / "nnUNetTrainer__nnUNetPlans__2d",
        base / "nnUNet_results" / "Dataset001_LGE" / "Dataset001_LGE" / "nnUNetTrainer__nnUNetPlans__2d",
        base / "Dataset001_LGE" / "nnUNetTrainer__nnUNetPlans__2d",
    ]

    trainer_dir = None
    for candidate in candidate_roots:
        if candidate.exists():
            trainer_dir = candidate
            break

    if trainer_dir is None:
        return results

    fold_dirs = sorted([p for p in trainer_dir.glob("fold_*") if p.is_dir()])
    folds = []
    for fold_dir in fold_dirs:
        suffix = fold_dir.name.split("_", 1)[-1]
        if suffix.isdigit():
            folds.append(int(suffix))

    if not folds:
        folds = [0]

    folds_sorted = sorted(folds)
    folds_csv = str(folds_sorted[0]) if len(folds_sorted) == 1 else "all"
    model_tag = _sanitize_tag(f"atrial_nnunet__dataset001_lge_2d_folds{len(folds)}")
    results.append({
        "architecture": "atrial_nnunet",
        "trained_dataset": "Dataset001_LGE",
        "seed": "",
        "fold": folds_csv,
        "model_path": str(trainer_dir),
        "model_tag": model_tag
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="Discover available models inside container")
    parser.add_argument("--architecture", required=True, choices=["cinema", "nnformer", "vsa3l", "atrial_nnunet"], help="Architecture to discover")
    parser.add_argument("--models_root", default="/models", help="Root path for models inside container")
    parser.add_argument("--output", required=True, help="Output CSV path")

    args = parser.parse_args()

    models_root = Path(args.models_root)

    if args.architecture == "cinema":
        rows = discover_cinema(models_root)
    elif args.architecture == "nnformer":
        rows = discover_nnformer(models_root)
    elif args.architecture == "atrial_nnunet":
        rows = discover_atrial_nnunet(models_root)
    else:
        rows = discover_vsa3l(models_root)

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["architecture", "trained_dataset", "seed", "fold", "model_path", "model_tag"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Discovered {len(rows)} models for {args.architecture}")


if __name__ == "__main__":
    main()
