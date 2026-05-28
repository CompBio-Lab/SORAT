#!/usr/bin/env python3
"""
CineMA Preprocessing Script for SORAT Pipeline

Preprocesses cardiac MRI data for CineMA model:
- Resample to target spacing (default: 1.0 x 1.0 x 10.0 mm)
- Crop to target size (default: 192 x 192) based on LV center
- Normalize intensity to [0, 1]
"""

import argparse
import json
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
    info = {'ed_frame': 0, 'es_frame': None, 'group': None, 'height': None, 'weight': None}
    
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
                    elif key == 'group':
                        info['group'] = value
                    elif key == 'height':
                        info['height'] = float(value) if value else None
                    elif key == 'weight':
                        info['weight'] = float(value) if value else None
    
    return info


def resample_image(image: sitk.Image, target_spacing: tuple, interpolator=sitk.sitkLinear) -> sitk.Image:
    """Resample image to target spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(interpolator)
    
    return resampler.Execute(image)


def center_crop_or_pad(array: np.ndarray, target_size: tuple) -> np.ndarray:
    """Center crop or pad array to target size (for x, y dimensions)."""
    current_shape = array.shape
    
    # Handle each dimension
    result = array.copy()
    
    for dim in range(2):  # Only x and y
        current = current_shape[dim]
        target = target_size[dim]
        
        if current > target:
            # Crop
            start = (current - target) // 2
            if dim == 0:
                result = result[start:start + target, ...]
            else:
                result = result[:, start:start + target, ...]
        elif current < target:
            # Pad
            pad_before = (target - current) // 2
            pad_after = target - current - pad_before
            pad_width = [(0, 0)] * len(current_shape)
            pad_width[dim] = (pad_before, pad_after)
            result = np.pad(result, pad_width, mode='constant', constant_values=0)
    
    return result


def normalize_intensity(array: np.ndarray) -> np.ndarray:
    """Normalize intensity to [0, 255] range."""
    array = array.astype(np.float32)
    min_val = np.min(array)
    max_val = np.max(array)
    
    if max_val - min_val > 0:
        array = (array - min_val) / (max_val - min_val) * 255.0
    
    return array.astype(np.uint8)


def preprocess_patient(
    input_path: Path,
    output_dir: Path,
    patient_id: str,
    info_cfg: Path = None,
    ground_truth: Path = None,
    target_spacing: tuple = (1.0, 1.0, 10.0),
    crop_size: tuple = (192, 192)
) -> dict:
    """
    Preprocess a single patient's data.
    
    Args:
        input_path: Path to input 4D NIfTI file
        output_dir: Directory to save preprocessed data
        patient_id: Patient identifier
        info_cfg: Path to Info.cfg file (optional)
        ground_truth: Path to ground truth segmentation directory (optional)
        target_spacing: Target voxel spacing (x, y, z)
        crop_size: Target crop size (x, y)
    
    Returns:
        Dictionary with preprocessing metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse info config
    info = parse_info_cfg(info_cfg) if info_cfg else {'ed_frame': 0, 'es_frame': None}
    
    # Load 4D image
    image_4d = sitk.ReadImage(str(input_path))
    array_4d = sitk.GetArrayFromImage(image_4d)  # (t, z, y, x)
    original_spacing = image_4d.GetSpacing()[:3]
    
    # Transpose to (x, y, z, t)
    array_4d = np.transpose(array_4d, (3, 2, 1, 0))
    
    # Resample each frame
    resampled_frames = []
    for t in range(array_4d.shape[-1]):
        frame_3d = array_4d[..., t]
        frame_sitk = sitk.GetImageFromArray(np.transpose(frame_3d, (2, 1, 0)))
        frame_sitk.SetSpacing(original_spacing)
        
        resampled = resample_image(frame_sitk, target_spacing)
        resampled_array = np.transpose(sitk.GetArrayFromImage(resampled), (2, 1, 0))
        resampled_frames.append(resampled_array)
    
    resampled_4d = np.stack(resampled_frames, axis=-1)
    
    # Crop/pad to target size
    cropped_4d = center_crop_or_pad(resampled_4d, crop_size)
    
    # Normalize intensity
    normalized_4d = normalize_intensity(cropped_4d)
    
    # Save preprocessed 4D volume
    output_4d = sitk.GetImageFromArray(np.transpose(normalized_4d, (3, 2, 1, 0)))
    output_4d.SetSpacing(target_spacing + (1.0,))  # Add time spacing
    output_path = output_dir / f"{patient_id}_sax_t.nii.gz"
    sitk.WriteImage(output_4d, str(output_path), useCompression=True)
    
    # Extract and save ED frame
    ed_frame_idx = info['ed_frame']
    ed_array = normalized_4d[..., ed_frame_idx]
    ed_sitk = sitk.GetImageFromArray(np.transpose(ed_array, (2, 1, 0)))
    ed_sitk.SetSpacing(target_spacing)
    sitk.WriteImage(ed_sitk, str(output_dir / f"{patient_id}_sax_ed.nii.gz"), useCompression=True)
    
    # Extract and save ES frame
    es_frame_idx = info['es_frame'] if info['es_frame'] is not None else array_4d.shape[-1] - 1
    es_array = normalized_4d[..., es_frame_idx]
    es_sitk = sitk.GetImageFromArray(np.transpose(es_array, (2, 1, 0)))
    es_sitk.SetSpacing(target_spacing)
    sitk.WriteImage(es_sitk, str(output_dir / f"{patient_id}_sax_es.nii.gz"), useCompression=True)
    
    # Process ground truth if available
    if ground_truth and Path(ground_truth).exists():
        gt_dir = Path(ground_truth).parent if Path(ground_truth).is_file() else Path(ground_truth)
        
        # Look for ED ground truth
        ed_gt_candidates = [
            gt_dir / f"{patient_id}_frame{ed_frame_idx:02d}_gt.nii.gz",
            gt_dir / f"{patient_id}_ED_gt.nii.gz",
            gt_dir / f"{patient_id}_frame01_gt.nii.gz"
        ]
        
        for ed_gt_path in ed_gt_candidates:
            if ed_gt_path.exists():
                ed_gt = sitk.ReadImage(str(ed_gt_path))
                ed_gt_resampled = resample_image(ed_gt, target_spacing, sitk.sitkNearestNeighbor)
                ed_gt_array = np.transpose(sitk.GetArrayFromImage(ed_gt_resampled), (2, 1, 0))
                ed_gt_cropped = center_crop_or_pad(ed_gt_array, crop_size)
                ed_gt_out = sitk.GetImageFromArray(np.transpose(ed_gt_cropped, (2, 1, 0)).astype(np.uint8))
                ed_gt_out.SetSpacing(target_spacing)
                sitk.WriteImage(ed_gt_out, str(output_dir / f"{patient_id}_sax_ed_gt.nii.gz"), useCompression=True)
                break
        
        # Look for ES ground truth
        es_gt_candidates = [
            gt_dir / f"{patient_id}_frame{es_frame_idx:02d}_gt.nii.gz",
            gt_dir / f"{patient_id}_ES_gt.nii.gz"
        ]
        
        for es_gt_path in es_gt_candidates:
            if es_gt_path.exists():
                es_gt = sitk.ReadImage(str(es_gt_path))
                es_gt_resampled = resample_image(es_gt, target_spacing, sitk.sitkNearestNeighbor)
                es_gt_array = np.transpose(sitk.GetArrayFromImage(es_gt_resampled), (2, 1, 0))
                es_gt_cropped = center_crop_or_pad(es_gt_array, crop_size)
                es_gt_out = sitk.GetImageFromArray(np.transpose(es_gt_cropped, (2, 1, 0)).astype(np.uint8))
                es_gt_out.SetSpacing(target_spacing)
                sitk.WriteImage(es_gt_out, str(output_dir / f"{patient_id}_sax_es_gt.nii.gz"), useCompression=True)
                break
    
    # Save metadata
    metadata = {
        'patient_id': patient_id,
        'original_spacing': list(original_spacing),
        'target_spacing': list(target_spacing),
        'crop_size': list(crop_size),
        'ed_frame': ed_frame_idx,
        'es_frame': es_frame_idx,
        'num_frames': array_4d.shape[-1],
        'output_shape': list(normalized_4d.shape)
    }
    
    # Convert numpy types to native Python types for JSON serialization
    metadata = convert_to_native(metadata)
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='CineMA preprocessing for SORAT pipeline')
    parser.add_argument('--input', required=True, help='Input 4D NIfTI file')
    parser.add_argument('--patient_id', required=True, help='Patient identifier')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--info_cfg', help='Path to Info.cfg file')
    parser.add_argument('--ground_truth', help='Path to ground truth directory')
    parser.add_argument('--spacing', nargs=3, type=float, default=[1.0, 1.0, 10.0],
                        help='Target spacing (x y z)')
    parser.add_argument('--crop_size', nargs=2, type=int, default=[192, 192],
                        help='Crop size (x y)')
    
    args = parser.parse_args()
    
    metadata = preprocess_patient(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        patient_id=args.patient_id,
        info_cfg=Path(args.info_cfg) if args.info_cfg else None,
        ground_truth=Path(args.ground_truth) if args.ground_truth else None,
        target_spacing=tuple(args.spacing),
        crop_size=tuple(args.crop_size)
    )
    
    print(f"Preprocessing complete for {args.patient_id}")
    print(f"Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
