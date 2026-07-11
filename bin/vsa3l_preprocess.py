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
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def _add_helper_import_paths() -> None:
    """Add common helper locations for copied Nextflow task scripts."""
    for path in (Path(__file__).resolve().parent, Path.cwd(), Path("/app/bin")):
        value = str(path)
        if path.exists() and value not in sys.path:
            sys.path.insert(0, value)


try:
    from frame_manifest import build_frame_manifest, write_manifest
except ImportError:
    _add_helper_import_paths()
    from frame_manifest import build_frame_manifest, write_manifest  # noqa: F401

try:
    from geometry_utils import _spatial_direction_3d, read_nifti_with_sitk_fallback
except ImportError:
    _add_helper_import_paths()
    from geometry_utils import _spatial_direction_3d, read_nifti_with_sitk_fallback  # noqa: F401


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


def preprocess_patient(
    input_path: Path,
    output_dir: Path,
    patient_id: str,
    info_cfg: Path = None,
    ground_truth: Path = None,
    input_size: tuple = (256, 256),
    frames_mode: str = "auto",
    max_frames: int = None
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
    
    # Load 4D image
    img = nib.load(str(input_path))
    data_4d = img.get_fdata()
    voxelspacing = img.header.get_zooms()[:3]
    
    if len(data_4d.shape) != 4:
        raise ValueError(f"Expected 4D image, got shape {data_4d.shape}")

    # Capture the original image geometry via SimpleITK so segmentation
    # predictions can be written back into the original coordinate space
    # (matching the ground truth + other architectures).  nibabel is kept for
    # array loading; SimpleITK gives us LPS-convention origin / direction
    # cosines directly, avoiding error-prone RAS->LPS affine conversion.  All
    # frames share one spatial geometry, so a single copy is sufficient.
    img_sitk = read_nifti_with_sitk_fallback(input_path)
    original_size_3d = [int(x) for x in img_sitk.GetSize()[:3]]
    original_spacing_3d = [float(x) for x in img_sitk.GetSpacing()[:3]]
    original_origin_3d = [float(x) for x in img_sitk.GetOrigin()[:3]]
    original_direction_3d = list(_spatial_direction_3d(img_sitk.GetDirection()))
    
    # Build frame manifest (handles ED/ES vs all-frames logic)
    num_frames = data_4d.shape[-1]
    manifest = build_frame_manifest(
        info_cfg_path=info_cfg,
        num_frames=num_frames,
        patient_id=patient_id,
        frames_mode=frames_mode,
        max_frames=max_frames,
        ground_truth_path=ground_truth,
    )
    
    slice_items = []

    for frame in manifest["frames"]:
        tag = frame["tag"]
        idx = frame["idx"]
        if idx < 0 or idx >= data_4d.shape[-1]:
            continue

        data_3d = data_4d[:, :, :, idx]

        for slice_idx in range(data_3d.shape[2]):
            slice_2d = data_3d[:, :, slice_idx]

            # Skip nearly empty slices
            if np.sum(slice_2d > 0) < 100:
                continue

            # Save slice as numpy array
            slice_filename = f"{patient_id}_{tag}_slice_{slice_idx:02d}.npy"
            slice_path = slices_dir / slice_filename
            np.save(slice_path, slice_2d.astype(np.float32))

            slice_items.append({
                'patient': patient_id,
                'frame_tag': tag,
                'frame_idx': idx,
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

        for frame in manifest["frames"]:
            tag = frame["tag"]
            idx = frame["idx"]
            gt_candidates = [
                gt_dir / f"{patient_id}_frame{idx + 1:02d}_gt.nii.gz",
                gt_dir / f"{patient_id}_frame{idx:02d}_gt.nii.gz",
                gt_dir / f"{patient_id}_{tag}_gt.nii.gz",
                gt_dir / f"{patient_id}_{tag.lower()}_gt.nii.gz",
            ]
            # M&Ms-2 axis-tagged layout: {pid}_{SA,LA}_{ED,ES}_gt.nii.gz
            for _axis in ("SA", "LA"):
                gt_candidates.append(gt_dir / f"{patient_id}_{_axis}_{tag}_gt.nii.gz")
                gt_candidates.append(gt_dir / f"{patient_id}_{_axis}_{tag.lower()}_gt.nii.gz")

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

                    gt_filename = f"{patient_id}_{tag}_slice_{slice_idx:02d}_gt.npy"
                    gt_slice_path = gt_slices_dir / gt_filename
                    np.save(gt_slice_path, gt_slice.astype(np.uint8))

                    gt_items.append({
                        'patient': patient_id,
                        'frame_tag': tag,
                        'slice_idx': int(slice_idx),
                        'npy_path': str(gt_slice_path)
                    })
    
    # Save frame manifest
    write_manifest(manifest, output_dir / f"{patient_id}_manifest.json")

    metadata = {
        'patient_id': patient_id,
        'original_path': str(input_path),
        'frame_tags': [f["tag"] for f in manifest["frames"]],
        'frame_count': len(manifest["frames"]),
        'ed_frame': manifest["frames"][0]["idx"] if manifest["frames"] else 0,
        'es_frame': manifest["frames"][-1]["idx"] if len(manifest["frames"]) > 1 else None,
        'num_frames': data_4d.shape[-1],
        'num_slices': data_4d.shape[2],
        'original_shape': list(data_4d.shape),
        'voxelspacing': list(voxelspacing),
        'input_size': list(input_size),
        'slice_items': slice_items,
        'gt_items': gt_items,
        # Original-image geometry used to map predictions back into the
        # original coordinate space (matches ground truth + other models).
        'original_size_3d': original_size_3d,
        'original_spacing_3d': original_spacing_3d,
        'original_origin_3d': original_origin_3d,
        'original_direction_3d': original_direction_3d,
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
        input_size=tuple(args.input_size),
        frames_mode=args.frames_mode,
        max_frames=args.max_frames
    )
    
    print(f"Preprocessing complete for {args.patient_id}")
    print(f"Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
