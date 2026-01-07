#!/usr/bin/env python3
"""
nnFormer Segmentation Script for CASC Pipeline

Runs cardiac segmentation inference using the nnFormer model.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch


def setup_nnformer_env(model_dir: Path):
    """Set up nnFormer environment variables."""
    os.environ['nnFormer_raw_data_base'] = str(model_dir / "nnFormer_raw")
    os.environ['nnFormer_preprocessed'] = str(model_dir / "nnFormer_preprocessed")
    os.environ['RESULTS_FOLDER'] = str(model_dir / "nnFormer_trained_models")
    
    # Add nnFormer to path
    nnformer_path = model_dir.parent
    if str(nnformer_path) not in sys.path:
        sys.path.insert(0, str(nnformer_path))


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


def segment_patient(
    input_dir: Path,
    patient_id: str,
    output_prefix: str,
    model_dir: Path,
    fold: int = 0,
    tta: bool = True,
    mixed_precision: bool = True
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
        
        # Find ED and ES outputs
        ed_output = None
        es_output = None
        
        for output_file in output_files:
            if 'frame01' in output_file.name or 'frame1' in output_file.name:
                ed_output = output_file
            else:
                es_output = output_file
        
        # If we only have one output, it's both ED and ES
        if len(output_files) == 1:
            ed_output = output_files[0]
            es_output = output_files[0]
        
        # Rename and move to final location
        final_ed = f"{output_prefix}_ED_nnformer.nii.gz"
        final_es = f"{output_prefix}_ES_nnformer.nii.gz"
        
        if ed_output:
            import shutil
            shutil.copy(ed_output, final_ed)
        
        if es_output and es_output != ed_output:
            import shutil
            shutil.copy(es_output, final_es)
        elif ed_output:
            import shutil
            shutil.copy(ed_output, final_es)
        
        results = {
            'patient_id': patient_id,
            'model': 'nnformer',
            'fold': fold,
            'tta': tta,
            'ed_output': final_ed,
            'es_output': final_es
        }
        
    finally:
        # Cleanup
        import shutil
        if temp_output.exists():
            shutil.rmtree(temp_output)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='nnFormer segmentation for CASC pipeline')
    parser.add_argument('--input_dir', required=True, help='Directory with preprocessed data')
    parser.add_argument('--patient_id', required=True, help='Patient identifier')
    parser.add_argument('--output_prefix', required=True, help='Output file prefix')
    parser.add_argument('--model_dir', required=True, help='Directory containing nnFormer model')
    parser.add_argument('--fold', type=int, default=0, help='Model fold')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    
    args = parser.parse_args()
    
    results = segment_patient(
        input_dir=Path(args.input_dir),
        patient_id=args.patient_id,
        output_prefix=args.output_prefix,
        model_dir=Path(args.model_dir),
        fold=args.fold,
        tta=args.tta,
        mixed_precision=args.mixed_precision
    )
    
    print(f"Segmentation complete for {args.patient_id}")
    print(f"ED output: {results['ed_output']}")
    print(f"ES output: {results['es_output']}")


if __name__ == '__main__':
    main()
