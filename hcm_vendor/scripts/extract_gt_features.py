#!/usr/bin/env python3
"""T12: extract SORAT features from ground-truth masks for the study cohort.

Runs inside containers/sorat-cinema.sif with the PyRadiomics venv on PYTHONPATH
(see extract_gt_features.sbatch). Ground-truth labels are canonicalized to the
SORAT convention (1=RV, 2=MYO, 3=LV) with bin/frame_manifest.py's anatomy-based
remapper, because M&Ms-2 ground truth uses 1=LV, 3=RV.

Output mirrors the pipeline's per-phase CSVs:
  <out_root>/<dataset>/<config>/<pid>_gt_<PHASE>_features.csv
plus <out_root>/label_mapping.csv recording how each mask was relabelled.
"""

import argparse
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "bin"))

from extract_features import (  # noqa: E402
    LABEL_LV,
    LABEL_MYO,
    LABEL_RV,
    build_radiomics_settings,
    compute_phase_features,
    flatten_feature_dicts,
    parse_spacing,
)
from frame_manifest import _remap_cardiac_labels  # noqa: E402
from geometry_utils import read_nifti_with_sitk_fallback  # noqa: E402


def read_info_cfg(path: Path) -> dict:
    values = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def phase_files(dataset: str, row: pd.Series, phase: str):
    """Return (image, gt_mask) paths for one 3-D phase of a subject."""
    folder = Path(row["ground_truth"])
    pid = row["patient_id"]
    if dataset == "mms2":
        return folder / f"{pid}_SA_{phase}.nii.gz", folder / f"{pid}_SA_{phase}_gt.nii.gz"
    frame = int(read_info_cfg(row["info_cfg"])[phase])  # ACDC Info.cfg frames are 1-indexed like filenames
    return folder / f"{pid}_frame{frame:02d}.nii.gz", folder / f"{pid}_frame{frame:02d}_gt.nii.gz"


def canonical_mask(gt_path: Path, out_path: Path) -> dict:
    """Write a SORAT-convention copy of a GT mask; return the observed label mapping."""
    image = read_nifti_with_sitk_fallback(gt_path)  # some M&Ms-2 sforms are non-orthonormal
    raw = sitk.GetArrayFromImage(image)
    canon = np.asarray(_remap_cardiac_labels(raw.copy(), np, "ventricular")).astype(np.uint8)

    mapping = {}
    for label in (LABEL_RV, LABEL_MYO, LABEL_LV):
        sources = np.unique(raw[canon == label])
        mapping[label] = [int(v) for v in sources]

    out = sitk.GetImageFromArray(canon)
    out.CopyInformation(image)
    sitk.WriteImage(out, str(out_path))
    return mapping


def run_one(task: dict) -> dict:
    image_path, gt_path = Path(task["image"]), Path(task["gt"])
    with tempfile.TemporaryDirectory() as tmp:
        mask_path = Path(tmp) / gt_path.name
        mapping = canonical_mask(gt_path, mask_path)
        record = {
            "dataset": task["dataset"], "patient_id": task["pid"], "phase": task["phase"],
            "gt_file": str(gt_path), "rv_from": mapping[LABEL_RV],
            "myo_from": mapping[LABEL_MYO], "lv_from": mapping[LABEL_LV],
        }
        for config_name, settings in task["configs"].items():
            out_csv = Path(task["out_root"]) / task["dataset"] / config_name / (
                f"{task['pid']}_gt_{task['phase']}_features.csv"
            )
            if out_csv.exists() and not task["overwrite"]:
                continue
            features = compute_phase_features(
                patient_id=f"{task['pid']}_gt_{task['phase']}",
                image_path=image_path,
                mask_path=mask_path,
                frame_tag=task["phase"],
                frame_idx=0,  # per-phase images are 3-D
                mask_source="ground_truth",
                radiomics_settings=settings,
            )
            features["mask_file"] = str(gt_path)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            flatten_feature_dicts(features["patient_id"], None, None, features).to_csv(out_csv, index=False)
    return record


def config_to_settings(options: dict) -> dict:
    return build_radiomics_settings(
        normalize=bool(options.get("normalize", False)),
        normalize_scale=options.get("normalize_scale"),
        bin_count=options.get("bin_count"),
        bin_width=options.get("bin_width"),
        resample_spacing=parse_spacing(options.get("resample_spacing")),
        force2d=bool(options.get("force2d", False)),
        force2d_dimension=int(options.get("force2d_dimension", 0)),
        remove_outliers=options.get("remove_outliers"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs_dir", default=str(REPO_ROOT / "results_hcm_vendor" / "inputs"))
    parser.add_argument("--out_root", default=str(REPO_ROOT / "results_hcm_vendor" / "features" / "gt"))
    parser.add_argument("--study_yaml", default=str(REPO_ROOT / "hcm_vendor" / "configs" / "study.yaml"),
                        help="Study config; its feature_configs section defines the radiomics configs")
    parser.add_argument("--configs", default=None, help="Comma-separated subset of feature_configs (default: all)")
    parser.add_argument("--datasets", default="mms2,acdc")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N subjects per dataset")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.study_yaml) as handle:
        feature_configs = yaml.safe_load(handle)["feature_configs"]
    wanted = args.configs.split(",") if args.configs else list(feature_configs)
    configs = {name: config_to_settings(feature_configs[name] or {}) for name in wanted}
    tasks = []
    for dataset in args.datasets.split(","):
        sheet = pd.read_csv(Path(args.inputs_dir) / f"{dataset}_nor_hcm.csv", dtype=str)
        if args.limit:
            sheet = sheet.head(args.limit)
        for _, row in sheet.iterrows():
            for phase in ("ED", "ES"):
                image, gt = phase_files(dataset, row, phase)
                if not image.exists() or not gt.exists():
                    raise FileNotFoundError(f"{dataset} {row['patient_id']} {phase}: {image} / {gt}")
                tasks.append({
                    "dataset": dataset, "pid": row["patient_id"], "phase": phase,
                    "image": str(image), "gt": str(gt), "configs": configs,
                    "out_root": args.out_root, "overwrite": args.overwrite,
                })

    print(f"{len(tasks)} phase masks x {len(configs)} configs, {args.workers} workers", flush=True)
    records, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # report every failure, then exit non-zero
                failures.append(f"{task['dataset']} {task['pid']} {task['phase']}: {exc!r}")
            if i % 50 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} done, {len(failures)} failed", flush=True)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values(["dataset", "patient_id", "phase"]).to_csv(
        out_root / "label_mapping.csv", index=False
    )
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)
    print("GT FEATURE EXTRACTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
