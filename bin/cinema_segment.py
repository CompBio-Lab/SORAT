#!/usr/bin/env python3
"""
CineMA Segmentation Script for SORAT Pipeline

Runs cardiac segmentation inference using the CineMA model.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from monai.transforms import Compose, SpatialPadd


def load_model(model_dir: Path, trained_dataset: str, seed: int, device: torch.device):
    """Load CineMA segmentation model."""
    from safetensors.torch import safe_open
    from omegaconf import OmegaConf
    
    # Try to import cinema module
    try:
        from cinema.segmentation.convunetr import get_model
    except ImportError:
        # Fallback: add parent directory to path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from cinema.segmentation.convunetr import get_model
    
    view = "sax"
    model_path = model_dir / f"finetuned/segmentation/{trained_dataset}_{view}/{trained_dataset}_{view}_{seed}.safetensors"
    config_path = model_dir / f"finetuned/segmentation/{trained_dataset}_{view}/config.yaml"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # Load state dict
    state_dict = {}
    with safe_open(str(model_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    
    # Load config and initialize model
    config = OmegaConf.load(str(config_path))
    model = get_model(config)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model, config


def run_inference(
    model,
    images: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    patch_size: tuple = (192, 192, 16)
) -> np.ndarray:
    """
    Run segmentation inference on 4D volume.
    
    Args:
        model: Loaded CineMA model
        images: Input 4D array (x, y, z, t)
        device: Torch device
        dtype: Data type for inference
        patch_size: Spatial patch size
    
    Returns:
        Segmentation array (x, y, z, t)
    """
    view = "sax"
    transform = Compose([
        SpatialPadd(keys=view, spatial_size=patch_size, method="end"),
    ])
    
    n_slices, n_frames = images.shape[-2:]
    labels_list = []
    
    for t in range(n_frames):
        # Prepare input
        batch = {view: torch.from_numpy(images[None, ..., t].astype(np.float32) / 255.0)}
        batch = transform(batch)
        batch = {k: v[None, ...].to(device=device, dtype=torch.float32) for k, v in batch.items()}
        
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.autocast("cuda", dtype=dtype, enabled=True):
                    logits = model(batch)[view]
            else:
                logits = model(batch)[view]
        
        labels_list.append(torch.argmax(logits, dim=1)[0, ..., :n_slices])
    
    labels = torch.stack(labels_list, dim=-1).detach().to(torch.float32).cpu().numpy()
    return labels


def _sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def _spatial_direction_from_any(direction: tuple) -> tuple:
    """Extract a valid 3x3 spatial direction from 3D or 4D direction cosines."""
    identity = (
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    )

    if len(direction) == 9:
        return tuple(direction)

    dim = int(round(len(direction) ** 0.5))
    if dim * dim != len(direction) or dim < 3:
        return identity

    try:
        mat = np.asarray(direction, dtype=np.float64).reshape(dim, dim)
        spatial = mat[:3, :3]
        if np.linalg.matrix_rank(spatial) < 3:
            return identity
        return tuple(float(x) for x in spatial.reshape(-1))
    except Exception:
        return identity


def segment_patient(
    input_dir: Path,
    patient_id: str,
    output_prefix: str,
    model_dir: Path,
    trained_dataset: str = 'acdc',
    seeds: list = [0],
    ensemble: bool = False,
    device: torch.device = None,
    model_tag: str = None
) -> dict:
    """
    Segment a single patient's cardiac MRI.
    
    Args:
        input_dir: Directory with preprocessed data
        patient_id: Patient identifier
        output_prefix: Prefix for output files
        model_dir: Directory containing model weights
        trained_dataset: Dataset the model was trained on
        seeds: List of random seeds to use
        ensemble: Whether to ensemble predictions across seeds
        device: Torch device
    
    Returns:
        Dictionary with segmentation results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dtype = torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    
    # Load preprocessed 4D volume
    image_path = input_dir / f"{patient_id}_sax_t.nii.gz"
    if not image_path.exists():
        raise FileNotFoundError(f"Preprocessed image not found: {image_path}")
    
    image_sitk = sitk.ReadImage(str(image_path))
    images = np.transpose(sitk.GetArrayFromImage(image_sitk))  # (x, y, z, t)
    
    # Load ground truth references for spacing
    ed_gt_path = input_dir / f"{patient_id}_sax_ed_gt.nii.gz"
    es_gt_path = input_dir / f"{patient_id}_sax_es_gt.nii.gz"
    
    spacing = image_sitk.GetSpacing()[:3]
    origin = image_sitk.GetOrigin()[:3]
    direction = image_sitk.GetDirection()
    spatial_direction = _spatial_direction_from_any(direction)
    
    # Build model tag for filenames/metadata
    if model_tag is None:
        if ensemble and len(seeds) > 1:
            model_tag = f"cinema__{trained_dataset}_ensemble"
        elif len(seeds) == 1:
            model_tag = f"cinema__{trained_dataset}_seed{seeds[0]}"
        else:
            model_tag = f"cinema__{trained_dataset}_seeds{len(seeds)}"
    model_tag = _sanitize_tag(model_tag)

    # Run inference for each seed
    all_predictions = []
    for seed in seeds:
        print(f"  Running inference with seed {seed}...")
        model, config = load_model(model_dir, trained_dataset, seed, device)
        predictions = run_inference(model, images, device, dtype)
        all_predictions.append(predictions)
        del model
        torch.cuda.empty_cache()
    
    # Ensemble predictions if requested
    if ensemble and len(all_predictions) > 1:
        # Majority voting
        stacked = np.stack(all_predictions, axis=0)
        from scipy import stats
        labels, _ = stats.mode(stacked, axis=0, keepdims=False)
        labels = labels.squeeze()
    else:
        labels = all_predictions[0]
    
    # Determine ED and ES frames
    n_frames = labels.shape[-1]
    lv_volumes = [np.sum(labels[..., t] == 3) for t in range(n_frames)]
    ed_frame_idx = 0  # First frame is typically ED
    es_frame_idx = np.argmin(lv_volumes)  # Minimum LV volume is ES
    
    # Extract ED and ES predictions
    ed_pred = labels[..., ed_frame_idx].astype(np.uint8)
    es_pred = labels[..., es_frame_idx].astype(np.uint8)
    
    # Save ED prediction
    ed_sitk = sitk.GetImageFromArray(np.transpose(ed_pred, (2, 1, 0)))
    ed_sitk.SetSpacing(spacing)
    ed_sitk.SetOrigin(origin)
    ed_sitk.SetDirection(spatial_direction)
    sitk.WriteImage(ed_sitk, f"{output_prefix}_ED_{model_tag}.nii.gz", useCompression=True)
    
    # Save ES prediction
    es_sitk = sitk.GetImageFromArray(np.transpose(es_pred, (2, 1, 0)))
    es_sitk.SetSpacing(spacing)
    es_sitk.SetOrigin(origin)
    es_sitk.SetDirection(spatial_direction)
    sitk.WriteImage(es_sitk, f"{output_prefix}_ES_{model_tag}.nii.gz", useCompression=True)
    
    results = {
        'patient_id': patient_id,
        'architecture': 'cinema',
        'model_tag': model_tag,
        'trained_dataset': trained_dataset,
        'seeds': seeds,
        'ensemble': ensemble,
        'ed_frame': ed_frame_idx,
        'es_frame': es_frame_idx,
        'ed_output': f"{output_prefix}_ED_{model_tag}.nii.gz",
        'es_output': f"{output_prefix}_ES_{model_tag}.nii.gz"
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='CineMA segmentation for SORAT pipeline')
    parser.add_argument('--input_dir', required=True, help='Directory with preprocessed data')
    parser.add_argument('--patient_id', required=True, help='Patient identifier')
    parser.add_argument('--output_prefix', required=True, help='Output file prefix')
    parser.add_argument('--model_dir', required=True, help='Directory containing model weights')
    parser.add_argument('--trained_dataset', default='acdc', help='Training dataset')
    parser.add_argument('--seeds', default='0', help='Comma-separated list of seeds')
    parser.add_argument('--ensemble', action='store_true', help='Ensemble predictions')
    parser.add_argument('--model_tag', default=None, help='Model tag for output naming')
    
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    results = segment_patient(
        input_dir=Path(args.input_dir),
        patient_id=args.patient_id,
        output_prefix=args.output_prefix,
        model_dir=Path(args.model_dir),
        trained_dataset=args.trained_dataset,
        seeds=seeds,
        ensemble=args.ensemble,
        model_tag=args.model_tag
    )
    
    print(f"Segmentation complete for {args.patient_id}")
    print(f"ED output: {results['ed_output']}")
    print(f"ES output: {results['es_output']}")


if __name__ == '__main__':
    main()
