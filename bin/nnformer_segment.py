#!/usr/bin/env python3
"""
nnFormer Segmentation Script for SORAT Pipeline

Runs cardiac segmentation inference using the nnFormer model.

NOTE: Environment variables must be set BEFORE importing nnformer modules
because paths.py runs at import time.
"""

import argparse
import json
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Dict
try:
    from frame_manifest import read_manifest, write_manifest
except ImportError:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.getcwd())
    from frame_manifest import read_manifest, write_manifest  # noqa: F401

# CRITICAL: Set environment variables to /tmp BEFORE importing nnformer
# This MUST happen before any nnformer import because paths.py runs at import time
# Always use /tmp which is writable inside containers
_TMP_BASE = '/tmp/nnformer_tmp'
os.environ['nnFormer_raw_data_base'] = f'{_TMP_BASE}/raw'
os.environ['nnFormer_preprocessed'] = f'{_TMP_BASE}/preprocessed'
os.environ['RESULTS_FOLDER'] = f'{_TMP_BASE}/results'
os.environ['MPLCONFIGDIR'] = _TMP_BASE

# Create the directories before importing nnformer (since paths.py calls maybe_mkdir_p)
os.makedirs(f'{_TMP_BASE}/raw/nnFormer_raw_data', exist_ok=True)
os.makedirs(f'{_TMP_BASE}/raw/nnFormer_cropped_data', exist_ok=True)
os.makedirs(f'{_TMP_BASE}/preprocessed', exist_ok=True)
os.makedirs(f'{_TMP_BASE}/results', exist_ok=True)

import numpy as np
import SimpleITK as sitk
import torch


def setup_nnformer_env(model_dir: Path):
    """Set up nnFormer environment - DO NOT override temp directories.
    
    Note: The raw_data_base, preprocessed, and RESULTS_FOLDER env vars
    are set at module load time to writable temp directories. We only
    need to add nnformer to the path here.
    """
    # Add nnFormer to path if needed
    nnformer_path = model_dir.parent
    if str(nnformer_path) not in sys.path:
        sys.path.insert(0, str(nnformer_path))


def ensure_plans_file(model_dir: Path):
    """Ensure nnFormer plans file exists in nnFormer_preprocessed."""
    plans_src = model_dir / "nnFormer_trained_models" / "nnFormer" / "3d_fullres" / "Task001_ACDC" / "nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1" / "plans.pkl"
    preprocessed_root = Path(os.environ.get("nnFormer_preprocessed", f"{_TMP_BASE}/preprocessed"))
    target_dir = preprocessed_root / "Task001_ACDC"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_plans = target_dir / "nnFormerPlansv2.1_plans_3D.pkl"

    if not target_plans.exists():
        if not plans_src.exists():
            raise FileNotFoundError(f"Missing plans file at {plans_src}")
        shutil.copy(plans_src, target_plans)


def load_nnformer_model(model_dir: Path, fold: int = 0):
    """Load nnFormer model."""
    from nnformer.training.model_restore import restore_model
    
    trainer_dir = model_dir / "nnFormer_trained_models" / "nnFormer" / "3d_fullres" / "Task001_ACDC" / "nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1"
    
    trainer, params = restore_model(
        str(trainer_dir),
        checkpoint_name=f"model_best"
    )
    
    return trainer


def segment_files(
    input_dir: Path,
    output_dir: Path,
    model_dir: Path,
    fold: int = 0,
    tta: bool = True,
    mixed_precision: bool = True
) -> list:
    """
    Run nnFormer segmentation on preprocessed files.
    
    Args:
        input_dir: Directory with preprocessed _0000.nii.gz files
        output_dir: Directory for output segmentations
        model_dir: Directory containing nnFormer model
        fold: Model fold to use
        tta: Test-time augmentation
        mixed_precision: Use mixed precision inference
    
    Returns:
        List of output file paths
    """
    from nnformer.inference.predict import predict_from_folder

    ensure_plans_file(model_dir)
    
    trainer_dir = model_dir / "nnFormer_trained_models" / "nnFormer" / "3d_fullres" / "Task001_ACDC" / "nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predict_from_folder(
        model=str(trainer_dir),
        input_folder=str(input_dir),
        output_folder=str(output_dir),
        folds=[fold],
        save_npz=False,
        num_threads_preprocessing=4,
        num_threads_nifti_save=2,
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
        tta=tta,
        mixed_precision=mixed_precision,
        overwrite_existing=True,
        mode='normal',
        step_size=0.5,
        checkpoint_name="model_best"
    )
    
    return list(output_dir.glob("*.nii.gz"))


def _sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def segment_patient(
    input_dir: Path,
    patient_id: str,
    output_prefix: str,
    model_dir: Path,
    fold: int = 0,
    tta: bool = True,
    mixed_precision: bool = True,
    model_tag: str = None
) -> dict:
    """
    Segment a single patient's cardiac MRI using nnFormer.
    
    Args:
        input_dir: Directory with preprocessed data
        patient_id: Patient identifier
        output_prefix: Prefix for output files
        model_dir: Directory containing nnFormer model
        fold: Model fold to use
        tta: Test-time augmentation
        mixed_precision: Use mixed precision inference
    
    Returns:
        Dictionary with segmentation results
    """
    # Setup environment
    setup_nnformer_env(model_dir)
    
    # Create temporary output directory
    temp_output = Path(f"temp_nnformer_{patient_id}")
    temp_output.mkdir(exist_ok=True)
    
    try:
        # Run segmentation
        output_files = segment_files(
            input_dir=input_dir,
            output_dir=temp_output,
            model_dir=model_dir,
            fold=fold,
            tta=tta,
            mixed_precision=mixed_precision
        )

        # Load frame manifest to know expected frame tags.
        # When the manifest is missing (e.g. stale cache from a previous
        # pipeline version), fall back to deriving tags from the nnUNet
        # output filenames so that *every* output file is preserved rather
        # than silently dropping all but the first frame.
        manifest_path = input_dir / f"{patient_id}_manifest.json"
        if manifest_path.exists():
            manifest = read_manifest(manifest_path)
        else:
            manifest = {}

        manifest_frames = manifest.get("frames", []) if isinstance(manifest, dict) else []
        expected_tags = [f["tag"] for f in manifest_frames]
        idx_by_tag = {f["tag"]: f["idx"] for f in manifest_frames}

        # nnUNet strips the _0000 suffix: input patient101_ED_0000.nii.gz
        # -> output patient101_ED.nii.gz, so the frame tag is recoverable
        # from the output filename.
        tag_re = re.compile(r"_(ED|ES|frame\d{2,})\.nii\.gz$", re.IGNORECASE)

        def _tag_from_name(path: Path) -> str:
            m = tag_re.search(path.name)
            return m.group(1) if m else None

        # Pass 1 — match each output to an expected manifest tag (case-insensitive).
        frame_outputs: Dict[str, Path] = {}
        used_files = set()
        for output_file in sorted(output_files):
            name_lower = output_file.name.lower()
            for frame_tag in expected_tags:
                if frame_tag in frame_outputs:
                    continue
                if f"_{frame_tag.lower()}" in name_lower:
                    frame_outputs[frame_tag] = output_file
                    used_files.add(output_file)
                    break

        # Pass 2 — for unmatched outputs, derive a tag from the filename.
        for output_file in sorted(output_files):
            if output_file in used_files:
                continue
            derived = _tag_from_name(output_file)
            if derived and derived not in frame_outputs:
                frame_outputs[derived] = output_file
                used_files.add(output_file)

        # Pass 3 — any still-unmatched outputs get a positional frameNN tag
        # so no segmentation file is ever silently dropped.
        frame_counter = 0
        for output_file in sorted(output_files):
            if output_file in used_files:
                continue
            while f"frame{frame_counter:02d}" in frame_outputs:
                frame_counter += 1
            positional_tag = f"frame{frame_counter:02d}"
            frame_outputs[positional_tag] = output_file
            used_files.add(output_file)
            frame_counter += 1

        # Final safety net: if everything above failed, keep the first output.
        if not frame_outputs and output_files:
            frame_outputs["frame00"] = output_files[0]

        # Rename and move to final location
        if model_tag is None:
            model_tag = f"nnformer__fold{fold}"
        model_tag = _sanitize_tag(model_tag)

        # Preserve a stable output order (ED, ES, then frameNN ascending).
        def _sort_key(tag: str):
            if tag == "ED":
                return (0, 0)
            if tag == "ES":
                return (0, 1)
            m = re.match(r"frame(\d+)", tag)
            return (1, int(m.group(1)) if m else 9999) if m else (1, 9999)

        ordered_tags = sorted(frame_outputs.keys(), key=_sort_key)
        saved_tags = []
        for i, tag in enumerate(ordered_tags):
            output_file = frame_outputs[tag]
            final_path = f"{output_prefix}_{tag}_{model_tag}.nii.gz"
            shutil.copy(output_file, final_path)
            saved_tags.append((tag, idx_by_tag.get(tag, i)))

        # Write segment manifest
        segment_manifest = {
            "patient_id": patient_id,
            "has_info_cfg": manifest.get("has_info_cfg", False) if isinstance(manifest, dict) else False,
            "num_frames": manifest.get("num_frames", len(saved_tags)) if isinstance(manifest, dict) else len(saved_tags),
            "frames": [{"tag": tag, "idx": idx} for tag, idx in saved_tags],
        }
        write_manifest(segment_manifest, f"{output_prefix}_{model_tag}_manifest.json")
        
        results = {
            'patient_id': patient_id,
            'architecture': 'nnformer',
            'model_tag': model_tag,
            'fold': fold,
            'tta': tta,
            'frame_tags': [tag for tag, _ in saved_tags],
            'frame_count': len(saved_tags),
        }
        
    finally:
        # Cleanup
        if temp_output.exists():
            shutil.rmtree(temp_output)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='nnFormer segmentation for SORAT pipeline')
    parser.add_argument('--input_dir', required=True, help='Directory with preprocessed data')
    parser.add_argument('--patient_id', required=True, help='Patient identifier')
    parser.add_argument('--output_prefix', required=True, help='Output file prefix')
    parser.add_argument('--model_dir', required=True, help='Directory containing nnFormer model')
    parser.add_argument('--fold', type=int, default=0, help='Model fold')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--model_tag', default=None, help='Model tag for output naming')
    
    args = parser.parse_args()
    
    results = segment_patient(
        input_dir=Path(args.input_dir),
        patient_id=args.patient_id,
        output_prefix=args.output_prefix,
        model_dir=Path(args.model_dir),
        fold=args.fold,
        tta=args.tta,
        mixed_precision=args.mixed_precision,
        model_tag=args.model_tag
    )
    
    print(f"Segmentation complete for {args.patient_id}")
    print(f"Output frames: {results.get('frame_tags', [])}")


if __name__ == '__main__':
    main()
