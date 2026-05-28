#!/usr/bin/env python3
"""
VSA-3L (MONAI) Preprocessing Script for SORAT Pipeline

Preprocesses cardiac MRI data for MONAI VSA-3L model:
- Extract 2D slices from 4D volume
- Resize to model input size (256x256)
- Normalize intensities
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
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


def preprocess_patient(
    input_path: Path,
    output_dir: Path,
    patient_id: str,
    info_cfg: Path = None,
    ground_truth: Path = None,
    input_size: tuple = (256, 256)
) -> dict:
    """
    Preprocess a single patient's data for VSA-3L.
    
    Extracts 2D slices from ED and ES frames and saves as numpy arrays.
    
    Args:
        input_path: Path to input 4D NIfTI file
        output_dir: Directory to save preprocessed data
        patient_id: Patient identifier
        info_cfg: Path to Info.cfg file (optional)
        ground_truth: Path to ground truth directory (optional)
        input_size: Target input size for the model
    
    Returns:
        Dictionary with preprocessing metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    slices_dir = output_dir / "slices"
    slices_dir.mkdir(exist_ok=True)
    
    # Parse info config
    info = parse_info_cfg(info_cfg) if info_cfg else {'ed_frame': 0, 'es_frame': None}
    
    # Load 4D image
    img = nib.load(str(input_path))
    data_4d = img.get_fdata()
    voxelspacing = img.header.get_zooms()[:3]
    
    if len(data_4d.shape) != 4:
        raise ValueError(f"Expected 4D image, got shape {data_4d.shape}")
    
    # Get frame indices
    ed_frame = info['ed_frame']
    es_frame = info['es_frame'] if info['es_frame'] is not None else data_4d.shape[-1] - 1
    
    slice_items = []
    
    for frame_type, frame_idx in {"ED": ed_frame, "ES": es_frame}.items():
        if frame_idx < 0 or frame_idx >= data_4d.shape[-1]:
            continue
        
        data_3d = data_4d[:, :, :, frame_idx]
        
        for slice_idx in range(data_3d.shape[2]):
            slice_2d = data_3d[:, :, slice_idx]
            
            # Skip nearly empty slices
            if np.sum(slice_2d > 0) < 100:
                continue
            
            # Save slice as numpy array
            slice_filename = f"{patient_id}_{frame_type}_slice_{slice_idx:02d}.npy"
            slice_path = slices_dir / slice_filename
            np.save(slice_path, slice_2d.astype(np.float32))
            
            slice_items.append({
                'patient': patient_id,
                'frame_type': frame_type,
                'frame_idx': frame_idx,
                'slice_idx': int(slice_idx),
                'npy_path': str(slice_path),
                'original_shape': list(slice_2d.shape),
                'voxelspacing_2d': [float(voxelspacing[1]), float(voxelspacing[0])]
            })
    
    # Process ground truth if available
    gt_items = []
    if ground_truth:
        gt_path = Path(ground_truth)
        gt_dir = gt_path.parent if gt_path.is_file() else gt_path
        
        gt_slices_dir = output_dir / "gt_slices"
        gt_slices_dir.mkdir(exist_ok=True)
        
        for frame_type, frame_idx in {"ED": ed_frame, "ES": es_frame}.items():
            # Look for ground truth file
            gt_candidates = [
                gt_dir / f"{patient_id}_frame{frame_idx:02d}_gt.nii.gz",
                gt_dir / f"{patient_id}_frame{frame_idx + 1:02d}_gt.nii.gz",
                gt_dir / f"{patient_id}_{frame_type.lower()}_gt.nii.gz"
            ]
            
            gt_file = None
            for candidate in gt_candidates:
                if candidate.exists():
                    gt_file = candidate
                    break
            
            if gt_file:
                gt_img = nib.load(str(gt_file))
                gt_data = gt_img.get_fdata()
                
                for slice_idx in range(gt_data.shape[2]):
                    gt_slice = gt_data[:, :, slice_idx]
                    
                    gt_filename = f"{patient_id}_{frame_type}_slice_{slice_idx:02d}_gt.npy"
                    gt_slice_path = gt_slices_dir / gt_filename
                    np.save(gt_slice_path, gt_slice.astype(np.uint8))
                    
                    gt_items.append({
                        'patient': patient_id,
                        'frame_type': frame_type,
                        'slice_idx': int(slice_idx),
                        'npy_path': str(gt_slice_path)
                    })
    
    # Save metadata
    metadata = {
        'patient_id': patient_id,
        'original_path': str(input_path),
        'ed_frame': ed_frame,
        'es_frame': es_frame,
        'num_frames': data_4d.shape[-1],
        'num_slices': data_4d.shape[2],
        'original_shape': list(data_4d.shape),
        'voxelspacing': list(voxelspacing),
        'input_size': list(input_size),
        'slice_items': slice_items,
        'gt_items': gt_items
    }
    
    # Convert numpy types to native Python types for JSON serialization
    metadata = convert_to_native(metadata)
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Extracted {len(slice_items)} slices for {patient_id}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='VSA-3L preprocessing for SORAT pipeline')
    parser.add_argument('--input', required=True, help='Input 4D NIfTI file')
    parser.add_argument('--patient_id', required=True, help='Patient identifier')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--info_cfg', help='Path to Info.cfg file')
    parser.add_argument('--ground_truth', help='Path to ground truth directory')
    parser.add_argument('--input_size', nargs=2, type=int, default=[256, 256],
                        help='Target input size')
    
    args = parser.parse_args()
    
    metadata = preprocess_patient(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        patient_id=args.patient_id,
        info_cfg=Path(args.info_cfg) if args.info_cfg else None,
        ground_truth=Path(args.ground_truth) if args.ground_truth else None,
        input_size=tuple(args.input_size)
    )
    
    print(f"Preprocessing complete for {args.patient_id}")
    print(f"Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
