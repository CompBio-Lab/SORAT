#!/usr/bin/env python3
"""
Compute Metrics Script for SORAT Pipeline

Computes segmentation metrics (Dice, HD95) against ground truth.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from medpy.metric import binary

try:
    from frame_manifest import resolve_frame_ground_truth
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.getcwd())
    from frame_manifest import resolve_frame_ground_truth

VENTRICULAR_LABELS = [(1, "rv"), (2, "myo"), (3, "lv")]
ATRIAL_LABELS = [(1, "wall"), (2, "ra"), (3, "la")]


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate Dice score for binary masks."""
    if (pred.sum() + gt.sum()) == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum())


def hd95_score(pred: np.ndarray, gt: np.ndarray, voxelspacing=None) -> float:
    """Calculate 95th percentile Hausdorff distance."""
    if pred.sum() > 0 and gt.sum() > 0:
        return binary.hd95(pred, gt, voxelspacing=voxelspacing)
    elif pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    else:
        return np.inf


def resample_to_reference(image: sitk.Image, reference: sitk.Image, is_label: bool = True) -> sitk.Image:
    """
    Resample an image to match the reference image's size, spacing, and orientation.
    
    Args:
        image: Image to resample
        reference: Reference image to match
        is_label: If True, use nearest neighbor interpolation (for segmentations)
    
    Returns:
        Resampled image
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    
    return resampler.Execute(image)


def get_label_spec(architecture: str) -> list[tuple[int, str]]:
    """Return label ids and names for a given model architecture."""
    if architecture == "atrial_nnunet":
        return ATRIAL_LABELS
    return VENTRICULAR_LABELS


def compute_metrics_for_volume(pred_path: Path, gt_path: Path, label_spec: list[tuple[int, str]]) -> dict:
    """
    Compute metrics for a single prediction-ground truth pair.
    
    Args:
        pred_path: Path to prediction NIfTI
        gt_path: Path to ground truth NIfTI
    
    Returns:
        Dictionary with metrics for each structure
    """
    # Load volumes
    pred_sitk = sitk.ReadImage(str(pred_path))
    gt_sitk = sitk.ReadImage(str(gt_path))
    
    # Check if shapes match, if not resample prediction to ground truth space
    pred_size = pred_sitk.GetSize()
    gt_size = gt_sitk.GetSize()
    
    if pred_size != gt_size:
        print(f"  Resampling prediction from {pred_size} to {gt_size}")
        pred_sitk = resample_to_reference(pred_sitk, gt_sitk, is_label=True)
    
    pred = sitk.GetArrayFromImage(pred_sitk)
    gt = sitk.GetArrayFromImage(gt_sitk)
    
    # Get voxel spacing from ground truth for HD95
    spacing = gt_sitk.GetSpacing()
    
    metrics = {}

    for label, name in label_spec:
        pred_mask = (pred == label).astype(np.uint8)
        gt_mask = (gt == label).astype(np.uint8)

        metrics[f'dice_{name}'] = dice_score(pred_mask, gt_mask)
        metrics[f'hd95_{name}'] = hd95_score(pred_mask, gt_mask, voxelspacing=spacing)

    # Compute mean metrics
    dice_keys = [f"dice_{name}" for _, name in label_spec]
    hd95_keys = [f"hd95_{name}" for _, name in label_spec]

    metrics['dice_mean'] = np.mean([metrics[key] for key in dice_keys])

    hd95_vals = [metrics[key] for key in hd95_keys]
    finite_hd95 = [v for v in hd95_vals if np.isfinite(v)]
    metrics['hd95_mean'] = np.mean(finite_hd95) if finite_hd95 else np.inf

    return metrics


def _infer_architecture(model: str) -> str:
    if '__' in model:
        return model.split('__', 1)[0]
    for prefix in ['cinema', 'nnformer', 'vsa3l', 'atrial_nnunet']:
        if model.startswith(prefix):
            return prefix
    return 'unknown'


def compute_patient_metrics(
    patient_id: str,
    model: str,
    seg: Path,
    frame_tag: str,
    frame_idx: int,
    ground_truth: Path,
    output_path: Path,
    architecture: str = None
) -> pd.DataFrame:
    """
    Compute metrics for a patient's single-frame segmentation.
    
    Args:
        patient_id: Patient identifier
        model: Model name
        seg: Path to segmentation file
        frame_tag: Frame tag (e.g. "ED", "ES", or raw frame number)
        frame_idx: Frame index (0-based)
        ground_truth: Path to ground truth directory or file
        output_path: Path to save metrics CSV
    
    Returns:
        DataFrame with metrics
    """
    gt_path = Path(ground_truth)
    if architecture is None:
        architecture = _infer_architecture(model)

    label_spec = get_label_spec(architecture)
    
    gt = resolve_frame_ground_truth(gt_path, frame_tag, frame_idx, patient_id, architecture=architecture)
    
    has_gt = gt is not None and gt.exists()
    
    if has_gt and Path(seg).exists():
        metrics = compute_metrics_for_volume(seg, gt, label_spec)
        result = {
            'patient_id': patient_id,
            'model': model,
            'architecture': architecture,
            'frame_tag': frame_tag,
            'frame_idx': frame_idx,
            'has_gt': True,
            **metrics
        }
    else:
        result = {
            'patient_id': patient_id,
            'model': model,
            'architecture': architecture,
            'frame_tag': frame_tag,
            'frame_idx': frame_idx,
            'has_gt': False,
        }
        for _, name in label_spec:
            result[f'dice_{name}'] = float('nan')
            result[f'hd95_{name}'] = float('nan')
        result['dice_mean'] = float('nan')
        result['hd95_mean'] = float('nan')
    
    df = pd.DataFrame([result])
    df.to_csv(output_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description='Compute segmentation metrics')
    parser.add_argument('--patient_id', required=True, help='Patient identifier')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--architecture', default=None, help='Architecture name')
    parser.add_argument('--seg', required=True, help='Path to segmentation')
    parser.add_argument('--frame_tag', required=True, type=str, help='Frame tag (e.g., ED, ES)')
    parser.add_argument('--frame_idx', required=True, type=int, help='Frame index (0-based)')
    parser.add_argument('--ground_truth', required=True, help='Path to ground truth')
    parser.add_argument('--output', required=True, help='Output CSV path')
    
    args = parser.parse_args()
    
    df = compute_patient_metrics(
        patient_id=args.patient_id,
        model=args.model,
        seg=Path(args.seg),
        frame_tag=args.frame_tag,
        frame_idx=args.frame_idx,
        ground_truth=Path(args.ground_truth),
        output_path=Path(args.output),
        architecture=args.architecture
    )
    
    print(f"Metrics computed for {args.patient_id} frame {args.frame_tag} using {args.model}")
    if not df.empty:
        print(f"Dice Mean: {df['dice_mean'].values[0]:.4f}")


if __name__ == '__main__':
    main()
