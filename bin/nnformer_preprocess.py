#!/usr/bin/env python3
"""
nnFormer Preprocessing Script for SORAT Pipeline

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
try:
    from frame_manifest import build_frame_manifest, write_manifest
except ImportError:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.getcwd())
    from frame_manifest import build_frame_manifest, write_manifest  # noqa: F401

try:
    from geometry_utils import read_nifti_with_sitk_fallback
except ImportError:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.getcwd())
    from geometry_utils import read_nifti_with_sitk_fallback  # noqa: F401


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
    ground_truth: Path = None,
    frames_mode: str = "auto",
    max_frames: int = None
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
    
    # Load image
    image = read_nifti_with_sitk_fallback(input_path)
    array = sitk.GetArrayFromImage(image)
    
    is_4d = len(array.shape) == 4

    # Build frame manifest (handles ED/ES vs all-frames logic)
    num_frames = array.shape[0] if is_4d else 1

    manifest = build_frame_manifest(
        info_cfg_path=info_cfg,
        num_frames=num_frames,
        patient_id=patient_id,
        frames_mode=frames_mode,
        max_frames=max_frames,
        ground_truth_path=ground_truth,
    )
    
    if is_4d:
        # 4D volume: extract frames according to manifest
        output_files = []
        for frame in manifest["frames"]:
            tag = frame["tag"]
            idx = frame["idx"]
            frame_image = extract_frame(image, idx)
            output_path = output_dir / f"{patient_id}_{tag}_0000.nii.gz"
            sitk.WriteImage(frame_image, str(output_path), useCompression=True)
            output_files.append(str(output_path))
    else:
        # 3D volume: single frame
        output_path = output_dir / f"{patient_id}_0000.nii.gz"
        sitk.WriteImage(image, str(output_path), useCompression=True)
        output_files = [str(output_path)]
    
    # Save frame manifest
    write_manifest(manifest, output_dir / f"{patient_id}_manifest.json")

    metadata = {
        'patient_id': patient_id,
        'input_path': str(input_path),
        'is_4d': is_4d,
        'frame_tags': [f["tag"] for f in manifest["frames"]],
        'frame_count': len(manifest["frames"]),
        'output_files': output_files,
        'spacing': list(image.GetSpacing()[:3])
    }
    
    # Convert numpy types to native Python types for JSON serialization
    metadata = convert_to_native(metadata)
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='nnFormer preprocessing for SORAT pipeline')
    parser.add_argument('--input', required=True, help='Input NIfTI file')
    parser.add_argument('--patient_id', required=True, help='Patient identifier')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--info_cfg', help='Path to Info.cfg file')
    parser.add_argument('--ground_truth', help='Path to ground truth file or directory')
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
        frames_mode=args.frames_mode,
        max_frames=args.max_frames,
    )
    
    print(f"Preprocessing complete for {args.patient_id}")
    print(f"Output files: {metadata['output_files']}")


if __name__ == '__main__':
    main()
