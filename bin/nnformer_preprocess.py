#!/usr/bin/env python3
"""
nnFormer Preprocessing Script for CASC Pipeline

Preprocesses cardiac MRI data for nnFormer model:
- Extract ED and ES frames from 4D volume
- Convert to nnFormer naming convention (_0000 suffix)
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    return obj


def parse_info_cfg(info_path: Path) -> dict:
    """Parse ACDC Info.cfg file to get ED/ES frame indices."""
    info = {'ed_frame': 0, 'es_frame': None}
    
    if info_path and info_path.exists():
        with open(info_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'ed':
                        info['ed_frame'] = int(value)
                    elif key == 'es':
                        info['es_frame'] = int(value)
    
    return info


def extract_frame(image_4d: sitk.Image, frame_idx: int) -> sitk.Image:
    """Extract a single 3D frame from 4D image."""
    array_4d = sitk.GetArrayFromImage(image_4d)  # (t, z, y, x)
    
    if frame_idx >= array_4d.shape[0]:
        frame_idx = array_4d.shape[0] - 1
    
    array_3d = array_4d[frame_idx]  # (z, y, x)
    
    image_3d = sitk.GetImageFromArray(array_3d)
    image_3d.SetSpacing(image_4d.GetSpacing()[:3])
    image_3d.SetOrigin(image_4d.GetOrigin()[:3])
    
    return image_3d


def preprocess_patient(
    input_path: Path,
    output_dir: Path,
    patient_id: str,
    info_cfg: Path = None,
    ground_truth: Path = None
) -> dict:
    """
    Preprocess a single patient's data for nnFormer.
    
    Args:
        input_path: Path to input NIfTI file (3D or 4D)
        output_dir: Directory to save preprocessed data
        patient_id: Patient identifier
        info_cfg: Path to Info.cfg file (optional)
        ground_truth: Path to ground truth directory (optional)
    
    Returns:
        Dictionary with preprocessing metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse info config
    info = parse_info_cfg(info_cfg) if info_cfg else {'ed_frame': 0, 'es_frame': None}
    
    # Load image
    image = sitk.ReadImage(str(input_path))
    array = sitk.GetArrayFromImage(image)
    
    is_4d = len(array.shape) == 4
    
    if is_4d:
        # 4D volume: extract ED and ES frames
        ed_frame_idx = info['ed_frame']
        es_frame_idx = info['es_frame'] if info['es_frame'] is not None else array.shape[0] - 1
        
        ed_image = extract_frame(image, ed_frame_idx)
        es_image = extract_frame(image, es_frame_idx)
        
        # Save with nnFormer naming convention
        # Use ED/ES naming instead of frame numbers for cleaner case IDs
        ed_output = output_dir / f"{patient_id}_ED_0000.nii.gz"
        es_output = output_dir / f"{patient_id}_ES_0000.nii.gz"
        
        sitk.WriteImage(ed_image, str(ed_output), useCompression=True)
        sitk.WriteImage(es_image, str(es_output), useCompression=True)
        
        output_files = [str(ed_output), str(es_output)]
    else:
        # 3D volume: assume it's already a single frame
        # Just add the _0000 suffix
        output_path = output_dir / f"{patient_id}_0000.nii.gz"
        sitk.WriteImage(image, str(output_path), useCompression=True)
        output_files = [str(output_path)]
        ed_frame_idx = 0
        es_frame_idx = 0
    
    # NOTE: Ground truth files are NOT copied to the preprocessing directory
    # because nnFormer's predict_from_folder gets confused by non-input files.
    # Ground truth is passed separately for metric computation.
    
    # Save metadata
    metadata = {
        'patient_id': patient_id,
        'input_path': str(input_path),
        'is_4d': is_4d,
        'ed_frame': ed_frame_idx if is_4d else 0,
        'es_frame': es_frame_idx if is_4d else 0,
        'output_files': output_files,
        'spacing': list(image.GetSpacing()[:3])
    }
    
    # Convert numpy types to native Python types for JSON serialization
    metadata = convert_to_native(metadata)
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='nnFormer preprocessing for CASC pipeline')
    parser.add_argument('--input', required=True, help='Input NIfTI file')
    parser.add_argument('--patient_id', required=True, help='Patient identifier')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--info_cfg', help='Path to Info.cfg file')
    parser.add_argument('--ground_truth', help='Path to ground truth file or directory')
    
    args = parser.parse_args()
    
    metadata = preprocess_patient(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        patient_id=args.patient_id,
        info_cfg=Path(args.info_cfg) if args.info_cfg else None,
        ground_truth=Path(args.ground_truth) if args.ground_truth else None
    )
    
    print(f"Preprocessing complete for {args.patient_id}")
    print(f"Output files: {metadata['output_files']}")


if __name__ == '__main__':
    main()
