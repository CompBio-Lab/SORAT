#!/usr/bin/env python3
"""
VSA-3L (MONAI) Segmentation Script for SORAT Pipeline

Runs cardiac segmentation inference using the MONAI VSA-3L model.
"""

import argparse
import json
import re
from pathlib import Path

import monai
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from skimage.transform import resize


def load_model(model_path: Path, bundle_root: Path, device: torch.device):
    """Load MONAI VSA-3L model."""
    # Load bundle configuration
    parser = monai.bundle.load_bundle_config(str(bundle_root), "inference.json")
    net = parser.get_parsed_content("network_def")
    
    # Load model weights
    net.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=False))
    net = net.to(device)
    net.eval()
    
    return net


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """
    Remap model labels to ACDC convention.
    
    Model output: 1=LV, 2=Myo, 3=RV
    ACDC convention: 1=RV, 2=Myo, 3=LV
    """
    seg_remapped = np.zeros_like(seg)
    seg_remapped[seg == 1] = 3  # LV -> 3
    seg_remapped[seg == 2] = 2  # Myo -> 2
    seg_remapped[seg == 3] = 1  # RV -> 1
    return seg_remapped


def segment_slice(
    model,
    slice_data: np.ndarray,
    device: torch.device,
    input_size: tuple = (256, 256)
) -> np.ndarray:
    """
    Segment a single 2D slice.
    
    Args:
        model: Loaded MONAI model
        slice_data: 2D input slice
        device: Torch device
        input_size: Expected input size for model
    
    Returns:
        Segmentation mask
    """
    original_shape = slice_data.shape
    
    # Resize to model input size
    if slice_data.shape != input_size:
        slice_resized = resize(slice_data, input_size, preserve_range=True, anti_aliasing=True)
    else:
        slice_resized = slice_data
    
    # Normalize
    denom = float(slice_resized.max())
    if denom == 0.0:
        return np.zeros(original_shape, dtype=np.uint8)
    
    input_tensor = torch.from_numpy(slice_resized / denom).float().to(device)
    input_tensor = input_tensor[None, None, :, :]  # Add batch and channel dims
    
    # Run inference
    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.softmax(pred[0], dim=0)
        seg = torch.argmax(pred, dim=0).cpu().numpy()
    
    # Remap labels
    seg_remapped = remap_labels(seg)
    
    # Resize back to original shape
    if original_shape != input_size:
        seg_remapped = resize(
            seg_remapped.astype(np.float32),
            original_shape,
            preserve_range=True,
            anti_aliasing=False,
            order=0
        ).astype(np.uint8)
    
    return seg_remapped


def _sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def segment_patient(
    input_dir: Path,
    patient_id: str,
    output_prefix: str,
    model_path: Path,
    bundle_root: Path,
    device: torch.device = None,
    model_tag: str = None
) -> dict:
    """
    Segment a single patient's cardiac MRI using VSA-3L.
    
    Args:
        input_dir: Directory with preprocessed data
        patient_id: Patient identifier
        output_prefix: Prefix for output files
        model_path: Path to model weights
        bundle_root: Path to MONAI bundle root
        device: Torch device
    
    Returns:
        Dictionary with segmentation results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model tag for filenames/metadata
    if model_tag is None:
        model_tag = f"vsa3l__{model_path.stem}"
    model_tag = _sanitize_tag(model_tag)

    # Load model
    print(f"Loading VSA-3L model from {model_path}...")
    model = load_model(model_path, bundle_root, device)
    
    # Load metadata
    metadata_path = input_dir / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Organize slices by frame type
    slices_by_frame = {'ED': {}, 'ES': {}}
    for item in metadata['slice_items']:
        frame_type = item['frame_type']
        slice_idx = item['slice_idx']
        slices_by_frame[frame_type][slice_idx] = item
    
    # Process each frame
    results = {}
    for frame_type in ['ED', 'ES']:
        if not slices_by_frame[frame_type]:
            continue
        
        slice_indices = sorted(slices_by_frame[frame_type].keys())
        
        # Segment each slice
        segmentations = {}
        original_shape = None
        
        for slice_idx in slice_indices:
            item = slices_by_frame[frame_type][slice_idx]
            slice_data = np.load(item['npy_path'])
            
            if original_shape is None:
                original_shape = slice_data.shape
            
            seg = segment_slice(model, slice_data, device)
            segmentations[slice_idx] = seg
        
        # Stack into 3D volume
        num_slices = max(slice_indices) + 1
        volume_shape = (original_shape[0], original_shape[1], num_slices)
        volume = np.zeros(volume_shape, dtype=np.uint8)
        
        for slice_idx, seg in segmentations.items():
            if seg.shape[:2] == volume_shape[:2]:
                volume[:, :, slice_idx] = seg
            else:
                # Resize if needed
                seg_resized = resize(
                    seg.astype(np.float32),
                    volume_shape[:2],
                    preserve_range=True,
                    order=0
                ).astype(np.uint8)
                volume[:, :, slice_idx] = seg_resized
        
        # Save as NIfTI
        output_path = f"{output_prefix}_{frame_type}_{model_tag}.nii.gz"
        
        # Get voxel spacing from metadata
        spacing = metadata.get('voxelspacing', [1.0, 1.0, 1.0])[:3]
        
        # Create SimpleITK image
        seg_sitk = sitk.GetImageFromArray(np.transpose(volume, (2, 1, 0)))
        seg_sitk.SetSpacing(spacing)
        sitk.WriteImage(seg_sitk, output_path, useCompression=True)
        
        results[f'{frame_type.lower()}_output'] = output_path
    
    return {
        'patient_id': patient_id,
        'architecture': 'vsa3l',
        'model_tag': model_tag,
        'ed_output': results.get('ed_output', f"{output_prefix}_ED_{model_tag}.nii.gz"),
        'es_output': results.get('es_output', f"{output_prefix}_ES_{model_tag}.nii.gz")
    }


def main():
    parser = argparse.ArgumentParser(description='VSA-3L segmentation for SORAT pipeline')
    parser.add_argument('--input_dir', required=True, help='Directory with preprocessed data')
    parser.add_argument('--patient_id', required=True, help='Patient identifier')
    parser.add_argument('--output_prefix', required=True, help='Output file prefix')
    parser.add_argument('--model_path', required=True, help='Path to model weights')
    parser.add_argument('--bundle_root', required=True, help='Path to MONAI bundle root')
    parser.add_argument('--model_tag', default=None, help='Model tag for output naming')
    
    args = parser.parse_args()
    
    results = segment_patient(
        input_dir=Path(args.input_dir),
        patient_id=args.patient_id,
        output_prefix=args.output_prefix,
        model_path=Path(args.model_path),
        bundle_root=Path(args.bundle_root),
        model_tag=args.model_tag
    )
    
    print(f"Segmentation complete for {args.patient_id}")
    print(f"ED output: {results['ed_output']}")
    print(f"ES output: {results['es_output']}")


if __name__ == '__main__':
    main()
