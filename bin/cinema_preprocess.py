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
try:
    from frame_manifest import build_frame_manifest, write_manifest
except ImportError:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.getcwd())
    from frame_manifest import build_frame_manifest, write_manifest  # noqa: F401

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


def _spatial_direction_3d(direction: tuple) -> list:
    """Extract a flat 9-element 3x3 spatial direction from 3D or 4D cosines.

    For a 4D image SimpleITK stores a 4x4 (16-element) direction matrix; the
    spatial 3x3 block is rows/columns 0..2 (indices 0,1,2,4,5,6,8,9,10).
    """
    if len(direction) == 9:
        return [float(x) for x in direction]
    dim = int(round(len(direction) ** 0.5))
    if dim * dim != len(direction) or dim < 3:
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    mat = np.asarray(direction, dtype=np.float64).reshape(dim, dim)
    spatial = mat[:3, :3]
    return [float(x) for x in spatial.reshape(-1)]


def _crop_origin_3d(
    original_origin: list,
    direction_3d: list,
    target_spacing: tuple,
    offsets: tuple,
) -> list:
    """Physical origin of the cropped 192x192 grid in the original space.

    The resampled (pre-crop) frame inherits the original image's origin and
    direction, so voxel (offset_x, offset_y, 0) of that grid maps to the
    physical point::

        O + offset_x*tx*col0 + offset_y*ty*col1

    where ``colN`` are the direction-matrix columns.  Negative offsets
    (padding) shift the origin backwards, correctly placing the padded grid
    relative to the original anatomy.
    """
    ox, oy = offsets
    tx, ty, _tz = target_spacing
    d = direction_3d  # row-major 3x3: [d0,d1,d2, d3,d4,d5, d6,d7,d8]
    return [
        float(original_origin[0] + ox * tx * d[0] + oy * ty * d[1]),
        float(original_origin[1] + ox * tx * d[3] + oy * ty * d[4]),
        float(original_origin[2] + ox * tx * d[6] + oy * ty * d[7]),
    ]


def center_crop_or_pad(array: np.ndarray, target_size: tuple) -> tuple:
    """Center crop or pad array to target size (for x, y dimensions).

    Returns a ``(cropped_array, offsets)`` tuple where ``offsets`` is
    ``(offset_x, offset_y)`` describing where the cropped grid starts in the
    pre-crop grid: a non-negative value when cropping (start voxel), or a
    negative value when padding (negative of the pad-before count).  The
    offsets let downstream code reconstruct the physical origin of the
    cropped 192x192 grid so predictions can be mapped back into the original
    image coordinate space.
    """
    current_shape = array.shape

    # Handle each dimension
    result = array.copy()
    offsets = [0, 0]

    for dim in range(2):  # Only x and y
        current = current_shape[dim]
        target = target_size[dim]

        if current > target:
            # Crop
            start = (current - target) // 2
            offsets[dim] = start
            if dim == 0:
                result = result[start:start + target, ...]
            else:
                result = result[:, start:start + target, ...]
        elif current < target:
            # Pad
            pad_before = (target - current) // 2
            pad_after = target - current - pad_before
            offsets[dim] = -pad_before
            pad_width = [(0, 0)] * len(current_shape)
            pad_width[dim] = (pad_before, pad_after)
            result = np.pad(result, pad_width, mode='constant', constant_values=0)

    return result, (offsets[0], offsets[1])


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
    crop_size: tuple = (192, 192),
    frames_mode: str = "auto",
    max_frames: int = None
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
    
    # Load 4D image
    image_4d = sitk.ReadImage(str(input_path))
    array_4d = sitk.GetArrayFromImage(image_4d)  # (t, z, y, x)
    
    # Build frame manifest (handles ED/ES vs all-frames logic)
    manifest = build_frame_manifest(
        info_cfg_path=info_cfg,
        num_frames=array_4d.shape[-1],
        patient_id=patient_id,
        frames_mode=frames_mode,
        max_frames=max_frames,
        ground_truth_path=ground_truth,
    )
    original_spacing = image_4d.GetSpacing()[:3]
    # Capture the original image geometry so segmentation predictions can be
    # resampled back into the original coordinate space downstream (matching
    # the ground truth and the other architectures).  All frames share one
    # spatial geometry, so a single copy is sufficient.
    original_size_3d = list(image_4d.GetSize()[:3])
    original_origin_3d = [float(x) for x in image_4d.GetOrigin()[:3]]
    original_direction_3d = _spatial_direction_3d(image_4d.GetDirection())
    
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
    cropped_4d, crop_offsets = center_crop_or_pad(resampled_4d, crop_size)
    
    # Physical origin of the 192x192 cropped grid in the original space.
    crop_origin_3d = _crop_origin_3d(
        original_origin_3d, original_direction_3d, target_spacing, crop_offsets
    )
    
    # Normalize intensity
    normalized_4d = normalize_intensity(cropped_4d)
    
    # Save preprocessed 4D volume
    output_4d = sitk.GetImageFromArray(np.transpose(normalized_4d, (3, 2, 1, 0)))
    output_4d.SetSpacing(target_spacing + (1.0,))  # Add time spacing
    output_path = output_dir / f"{patient_id}_sax_t.nii.gz"
    sitk.WriteImage(output_4d, str(output_path), useCompression=True)
    
    # Extract and save frames according to manifest
    for frame in manifest["frames"]:
        tag = frame["tag"]
        idx = frame["idx"]
        frame_array = normalized_4d[..., idx]
        frame_sitk = sitk.GetImageFromArray(np.transpose(frame_array, (2, 1, 0)))
        frame_sitk.SetSpacing(target_spacing)
        sitk.WriteImage(frame_sitk, str(output_dir / f"{patient_id}_sax_{tag.lower()}.nii.gz"), useCompression=True)
    
    # Process ground truth if available
    if ground_truth and Path(ground_truth).exists():
        gt_dir = Path(ground_truth).parent if Path(ground_truth).is_file() else Path(ground_truth)

        for frame in manifest["frames"]:
            tag = frame["tag"]
            idx = frame["idx"]
            gt_candidates = [
                gt_dir / f"{patient_id}_frame{idx + 1:02d}_gt.nii.gz",
                gt_dir / f"{patient_id}_frame{idx:02d}_gt.nii.gz",
                gt_dir / f"{patient_id}_{tag}_gt.nii.gz",
            ]
            if tag == "ED":
                gt_candidates.append(gt_dir / f"{patient_id}_frame01_gt.nii.gz")

            for gt_path in gt_candidates:
                if gt_path.exists():
                    gt = sitk.ReadImage(str(gt_path))
                    gt_resampled = resample_image(gt, target_spacing, sitk.sitkNearestNeighbor)
                    gt_array = np.transpose(sitk.GetArrayFromImage(gt_resampled), (2, 1, 0))
                    gt_cropped, _ = center_crop_or_pad(gt_array, crop_size)
                    gt_out = sitk.GetImageFromArray(np.transpose(gt_cropped, (2, 1, 0)).astype(np.uint8))
                    gt_out.SetSpacing(target_spacing)
                    sitk.WriteImage(gt_out, str(output_dir / f"{patient_id}_sax_{tag.lower()}_gt.nii.gz"), useCompression=True)
                    break
    
    # Save frame manifest
    write_manifest(manifest, output_dir / f"{patient_id}_manifest.json")
    
    # Save metadata
    metadata = {
        'patient_id': patient_id,
        'original_spacing': list(original_spacing),
        'target_spacing': list(target_spacing),
        'crop_size': list(crop_size),
        'ed_frame': manifest["frames"][0]["idx"] if manifest["frames"] else 0,
        'es_frame': manifest["frames"][-1]["idx"] if len(manifest["frames"]) > 1 else None,
        'num_frames': array_4d.shape[-1],
        'output_shape': list(normalized_4d.shape),
        'frame_count': len(manifest["frames"]),
        'frame_tags': [f["tag"] for f in manifest["frames"]],
        # Original-image geometry used to map predictions back into the
        # original coordinate space (matches ground truth + other models).
        'original_size_3d': [int(x) for x in original_size_3d],
        'original_spacing_3d': [float(x) for x in original_spacing],
        'original_origin_3d': [float(x) for x in original_origin_3d],
        'original_direction_3d': [float(x) for x in original_direction_3d],
        'crop_origin_3d': [float(x) for x in crop_origin_3d],
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
    parser.add_argument('--frames_mode', default='auto',
                        choices=['auto', 'ed_es', 'all'],
                        help='Frame extraction mode')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames when in all-frames mode')
    
    args = parser.parse_args()
    
    metadata = preprocess_patient(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        patient_id=args.patient_id,
        info_cfg=Path(args.info_cfg) if args.info_cfg else None,
        ground_truth=Path(args.ground_truth) if args.ground_truth else None,
        target_spacing=tuple(args.spacing),
        crop_size=tuple(args.crop_size),
        frames_mode=args.frames_mode,
        max_frames=args.max_frames
    )
    
    print(f"Preprocessing complete for {args.patient_id}")
    print(f"Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
