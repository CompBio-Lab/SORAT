#!/usr/bin/env python3
"""
Compute Metrics Script for CASC Pipeline

Computes segmentation metrics (Dice, HD95) against ground truth.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from medpy.metric import binary


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


def compute_metrics_for_volume(pred_path: Path, gt_path: Path) -> dict:
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
    
    pred = sitk.GetArrayFromImage(pred_sitk)
    gt = sitk.GetArrayFromImage(gt_sitk)
    
    # Get voxel spacing for HD95
    spacing = pred_sitk.GetSpacing()
    
    # ACDC labels: 1=RV, 2=MYO, 3=LV
    metrics = {}
    
    for label, name in [(1, 'rv'), (2, 'myo'), (3, 'lv')]:
        pred_mask = (pred == label).astype(np.uint8)
        gt_mask = (gt == label).astype(np.uint8)
        
        metrics[f'dice_{name}'] = dice_score(pred_mask, gt_mask)
        metrics[f'hd95_{name}'] = hd95_score(pred_mask, gt_mask, voxelspacing=spacing)
    
    # Compute mean metrics
    metrics['dice_mean'] = np.mean([metrics['dice_rv'], metrics['dice_myo'], metrics['dice_lv']])
    
    hd95_vals = [metrics['hd95_rv'], metrics['hd95_myo'], metrics['hd95_lv']]
    finite_hd95 = [v for v in hd95_vals if np.isfinite(v)]
    metrics['hd95_mean'] = np.mean(finite_hd95) if finite_hd95 else np.inf
    
    return metrics


def compute_patient_metrics(
    patient_id: str,
    model: str,
    seg_ed: Path,
    seg_es: Path,
    ground_truth: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Compute metrics for a patient's ED and ES segmentations.
    
    Args:
        patient_id: Patient identifier
        model: Model name
        seg_ed: Path to ED segmentation
        seg_es: Path to ES segmentation
        ground_truth: Path to ground truth directory or file
        output_path: Path to save metrics CSV
    
    Returns:
        DataFrame with metrics
    """
    gt_path = Path(ground_truth)
    
    results = []
    
    # Find ground truth files
    if gt_path.is_dir():
        ed_gt_candidates = [
            gt_path / f"{patient_id}_frame01_gt.nii.gz",
            gt_path / f"{patient_id}_ED_gt.nii.gz",
            gt_path / f"{patient_id}_sax_ed_gt.nii.gz"
        ]
        es_gt_candidates = [
            gt_path / f"{patient_id}_ES_gt.nii.gz",
            gt_path / f"{patient_id}_sax_es_gt.nii.gz"
        ]
        
        # Add frame-based naming for ES
        for i in range(2, 30):
            es_gt_candidates.append(gt_path / f"{patient_id}_frame{i:02d}_gt.nii.gz")
        
        ed_gt = None
        es_gt = None
        
        for candidate in ed_gt_candidates:
            if candidate.exists():
                ed_gt = candidate
                break
        
        for candidate in es_gt_candidates:
            if candidate.exists():
                es_gt = candidate
                break
    else:
        # Single ground truth file (assume it's for ED)
        ed_gt = gt_path
        es_gt = None
    
    # Compute ED metrics
    if ed_gt and ed_gt.exists() and Path(seg_ed).exists():
        ed_metrics = compute_metrics_for_volume(seg_ed, ed_gt)
        ed_result = {
            'patient_id': patient_id,
            'model': model,
            'frame_type': 'ED',
            **{f'ed_{k}': v for k, v in ed_metrics.items()}
        }
        results.append(ed_result)
    
    # Compute ES metrics
    if es_gt and es_gt.exists() and Path(seg_es).exists():
        es_metrics = compute_metrics_for_volume(seg_es, es_gt)
        es_result = {
            'patient_id': patient_id,
            'model': model,
            'frame_type': 'ES',
            **{f'es_{k}': v for k, v in es_metrics.items()}
        }
        results.append(es_result)
    
    # Create combined result
    if results:
        combined = {
            'patient_id': patient_id,
            'model': model
        }
        
        for result in results:
            for k, v in result.items():
                if k not in ['patient_id', 'model', 'frame_type']:
                    combined[k] = v
        
        # Compute overall means
        if 'ed_dice_mean' in combined and 'es_dice_mean' in combined:
            combined['overall_dice_mean'] = np.mean([combined['ed_dice_mean'], combined['es_dice_mean']])
        elif 'ed_dice_mean' in combined:
            combined['overall_dice_mean'] = combined['ed_dice_mean']
        elif 'es_dice_mean' in combined:
            combined['overall_dice_mean'] = combined['es_dice_mean']
        
        df = pd.DataFrame([combined])
    else:
        df = pd.DataFrame()
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Compute segmentation metrics')
    parser.add_argument('--patient_id', required=True, help='Patient identifier')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--seg_ed', required=True, help='Path to ED segmentation')
    parser.add_argument('--seg_es', required=True, help='Path to ES segmentation')
    parser.add_argument('--ground_truth', required=True, help='Path to ground truth')
    parser.add_argument('--output', required=True, help='Output CSV path')
    
    args = parser.parse_args()
    
    df = compute_patient_metrics(
        patient_id=args.patient_id,
        model=args.model,
        seg_ed=Path(args.seg_ed),
        seg_es=Path(args.seg_es),
        ground_truth=Path(args.ground_truth),
        output_path=Path(args.output)
    )
    
    print(f"Metrics computed for {args.patient_id} using {args.model}")
    if not df.empty:
        print(f"Overall Dice: {df['overall_dice_mean'].values[0]:.4f}")


if __name__ == '__main__':
    main()
