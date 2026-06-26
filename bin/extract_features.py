#!/usr/bin/env python3
"""
Extract interpretable cardiac MRI features from segmentation masks.

Supported labels:
  0 = background
  1 = RV
  2 = MYO
  3 = LV
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from scipy import ndimage


LABEL_BG = 0
LABEL_RV = 1
LABEL_MYO = 2
LABEL_LV = 3
MYOCARDIUM_DENSITY_G_PER_ML = 1.05


def _spatial_direction_from_any(direction: Tuple[float, ...]) -> Tuple[float, ...]:
    """Extract a valid 3x3 spatial direction from 3D or 4D direction cosines."""
    identity = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
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


def parse_info_cfg(info_cfg: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parse ED/ES indices from an optional Info.cfg file."""
    if not info_cfg:
        return None, None

    p = Path(info_cfg)
    if not p.exists():
        return None, None

    ed_idx, es_idx = None, None
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("ED:"):
            try:
                ed_idx = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("ES:"):
            try:
                es_idx = int(line.split(":", 1)[1].strip())
            except Exception:
                pass

    return ed_idx, es_idx


def infer_phase_from_path(mask_path: Path) -> str:
    """Infer cardiac phase from a mask filename."""
    name = mask_path.name.lower()
    if "_ed" in name or "ed_" in name:
        return "ED"
    if "_es" in name or "es_" in name:
        return "ES"
    return "FRAME"


def choose_frame_index(phase: str, ed_idx: Optional[int], es_idx: Optional[int]) -> int:
    """Choose 4D frame index for intensity image extraction."""
    phase = phase.upper()
    if phase == "ED":
        return 0 if ed_idx is None else int(ed_idx)
    if phase == "ES":
        return 1 if es_idx is None else int(es_idx)
    return 0 if ed_idx is None else int(ed_idx)


def extract_3d_frame(image: sitk.Image, frame_idx: int) -> sitk.Image:
    """Extract one 3D frame from a 4D image (SimpleITK arrays are [t, z, y, x])."""
    arr = sitk.GetArrayFromImage(image)
    if arr.ndim == 3:
        return image

    n_frames = arr.shape[0]
    idx = max(0, min(int(frame_idx), n_frames - 1))
    frame = arr[idx]

    frame_img = sitk.GetImageFromArray(frame)
    frame_img.SetSpacing(tuple(image.GetSpacing()[:3]))
    frame_img.SetOrigin(tuple(image.GetOrigin()[:3]))
    frame_img.SetDirection(_spatial_direction_from_any(image.GetDirection()))
    return frame_img


def extract_vector_component(image: sitk.Image, component_idx: int) -> sitk.Image:
    """Extract one scalar component from a vector image."""
    n_comp = int(image.GetNumberOfComponentsPerPixel())
    if n_comp <= 1:
        return image

    idx = max(0, min(int(component_idx), n_comp - 1))
    scalar = sitk.VectorIndexSelectionCast(image, idx)
    scalar.CopyInformation(image)
    return scalar


def resample_intensity_to_reference(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample intensity image to segmentation geometry."""
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(reference)
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(0.0)
    return rs.Execute(image)


def load_mask(mask_path: Path) -> Tuple[sitk.Image, np.ndarray]:
    """Load segmentation mask image and array."""
    mask_img = sitk.ReadImage(str(mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_img)
    return mask_img, mask_arr


def prepare_phase_image(image_path: Path, mask_img: sitk.Image, frame_idx: int) -> sitk.Image:
    """Load original image and return a phase-specific 3D image in mask space."""
    image = sitk.ReadImage(str(image_path))

    if image.GetDimension() == 4:
        image_3d = extract_3d_frame(image, frame_idx)
    elif image.GetNumberOfComponentsPerPixel() > 1:
        image_3d = extract_vector_component(image, frame_idx)
    else:
        image_3d = image

    if image_3d.GetNumberOfComponentsPerPixel() > 1:
        image_3d = extract_vector_component(image_3d, 0)

    image_3d = sitk.Cast(image_3d, sitk.sitkFloat32)

    return resample_intensity_to_reference(image_3d, mask_img)


def compute_volumes(mask_arr: np.ndarray, voxel_volume_ml: float) -> dict:
    """Compute ventricular structure volumes in ml."""
    lv_volume_ml = float(np.sum(mask_arr == LABEL_LV) * voxel_volume_ml)
    rv_volume_ml = float(np.sum(mask_arr == LABEL_RV) * voxel_volume_ml)
    myo_volume_ml = float(np.sum(mask_arr == LABEL_MYO) * voxel_volume_ml)

    return {
        "lv_volume_ml": lv_volume_ml,
        "rv_volume_ml": rv_volume_ml,
        "myo_volume_ml": myo_volume_ml,
        "myocardial_mass_g": myo_volume_ml * MYOCARDIUM_DENSITY_G_PER_ML,
    }


def compute_wall_thickness(mask_arr: np.ndarray, spacing_xyz: Tuple[float, ...]) -> dict:
    """
    Estimate myocardial wall thickness from LV-facing to epicardial MYO boundary.

    Steps:
      1) Identify inner MYO boundary voxels touching LV (label 3).
      2) Identify outer MYO boundary voxels touching BG (label 0).
      3) Compute distance transform from outer boundary and sample on inner boundary.
    """
    lv_mask = mask_arr == LABEL_LV
    myo_mask = mask_arr == LABEL_MYO
    bg_mask = mask_arr == LABEL_BG

    if not np.any(lv_mask) or not np.any(myo_mask):
        return {
            "wall_thickness_mean_mm": np.nan,
            "wall_thickness_max_mm": np.nan,
        }

    structure = ndimage.generate_binary_structure(mask_arr.ndim, 1)

    inner_boundary = myo_mask & ndimage.binary_dilation(lv_mask, structure=structure)
    outer_boundary = myo_mask & ndimage.binary_dilation(bg_mask, structure=structure)

    if not np.any(inner_boundary) or not np.any(outer_boundary):
        return {
            "wall_thickness_mean_mm": np.nan,
            "wall_thickness_max_mm": np.nan,
        }

    spacing_zyx = tuple(reversed(tuple(float(s) for s in spacing_xyz[:3])))
    distance_to_outer_mm = ndimage.distance_transform_edt(~outer_boundary, sampling=spacing_zyx)

    thickness_values_mm = distance_to_outer_mm[inner_boundary]
    thickness_values_mm = thickness_values_mm[thickness_values_mm > 0]

    if thickness_values_mm.size == 0:
        return {
            "wall_thickness_mean_mm": np.nan,
            "wall_thickness_max_mm": np.nan,
        }

    return {
        "wall_thickness_mean_mm": float(np.mean(thickness_values_mm)),
        "wall_thickness_max_mm": float(np.max(thickness_values_mm)),
    }


def extract_radiomics_features(image_img: sitk.Image, mask_img: sitk.Image) -> dict:
    """Extract PyRadiomics features for MYO (label 2) using interpretable classes only."""
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")

    result = extractor.execute(image_img, mask_img, label=LABEL_MYO)

    features = {}
    for key, value in result.items():
        if not str(key).startswith("original_"):
            continue

        if isinstance(value, (np.integer, np.floating)):
            features[f"radiomics_{key}"] = float(value)
        elif isinstance(value, (int, float, bool)):
            features[f"radiomics_{key}"] = value
        else:
            try:
                features[f"radiomics_{key}"] = float(value)
            except Exception:
                pass

    return features


def compute_phase_features(
    patient_id: str,
    image_path: Path,
    mask_path: Path,
    frame_tag: Optional[str] = None,
    frame_idx: int = 0,
    mask_source: str = "auto",
) -> dict:
    """Compute all features for one phase mask."""
    mask_img, mask_arr = load_mask(mask_path)

    if mask_arr.ndim != 3:
        raise ValueError(f"Expected 3D mask for {mask_path}, got array ndim={mask_arr.ndim}")

    phase_name = (frame_tag or infer_phase_from_path(mask_path)).upper()
    image_phase = prepare_phase_image(image_path, mask_img, frame_idx)

    voxel_volume_ml = float(np.prod(mask_img.GetSpacing()[:3]) / 1000.0)

    features = {
        "patient_id": patient_id,
        "phase": phase_name,
        "mask_file": str(mask_path),
        "mask_source": mask_source,
        "voxel_volume_ml": voxel_volume_ml,
    }

    features.update(compute_volumes(mask_arr, voxel_volume_ml))
    features.update(compute_wall_thickness(mask_arr, mask_img.GetSpacing()))

    # Radiomics extraction can fail for empty or malformed MYO regions; keep pipeline robust.
    try:
        features.update(extract_radiomics_features(image_phase, mask_img))
    except Exception as exc:
        features["radiomics_error"] = str(exc)

    return features


def flatten_feature_dicts(
    patient_id: str,
    ed_features: Optional[dict],
    es_features: Optional[dict],
    single_features: Optional[dict],
) -> pd.DataFrame:
    """Flatten phase-specific dictionaries into one-row tabular output."""
    row = {"patient_id": patient_id}

    if single_features is not None:
        prefix = single_features.get("phase", "FRAME").lower()
        for key, value in single_features.items():
            if key in {"patient_id", "phase"}:
                continue
            row[f"{prefix}_{key}"] = value

    if ed_features is not None:
        for key, value in ed_features.items():
            if key in {"patient_id", "phase"}:
                continue
            row[f"ed_{key}"] = value

    if es_features is not None:
        for key, value in es_features.items():
            if key in {"patient_id", "phase"}:
                continue
            row[f"es_{key}"] = value

    if ed_features is not None and es_features is not None:
        lv_ed = ed_features.get("lv_volume_ml", np.nan)
        lv_es = es_features.get("lv_volume_ml", np.nan)
        if np.isfinite(lv_ed) and np.isfinite(lv_es) and lv_ed > 0:
            row["lv_ejection_fraction_percent"] = float((lv_ed - lv_es) / lv_ed * 100.0)
        else:
            row["lv_ejection_fraction_percent"] = np.nan

    if ed_features is not None:
        row["myocardial_mass_g"] = ed_features.get("myocardial_mass_g", np.nan)
    elif single_features is not None:
        row["myocardial_mass_g"] = single_features.get("myocardial_mass_g", np.nan)
    elif es_features is not None:
        row["myocardial_mass_g"] = es_features.get("myocardial_mass_g", np.nan)

    return pd.DataFrame([row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract interpretable cardiac MRI features")
    parser.add_argument("--patient_id", required=True, help="Patient identifier")
    parser.add_argument("--image", required=True, help="Path to original NIfTI image (3D/4D)")
    parser.add_argument("--mask", default=None, help="Path to single 3D segmentation mask")
    parser.add_argument("--mask_ed", default=None, help="Path to ED segmentation mask")
    parser.add_argument("--mask_es", default=None, help="Path to ES segmentation mask")
    parser.add_argument("--info_cfg", default=None, help="Optional Info.cfg with ED/ES frame indices")
    parser.add_argument("--frame_tag", type=str, default=None, help="Cardiac phase tag (e.g., ED, ES)")
    parser.add_argument("--frame_idx", type=int, default=0, help="4D frame index for intensity extraction")
    parser.add_argument("--mask_source", default="auto", help="Mask source label for tracking")
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Output CSV path (default: [patient_id]_features.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not any([args.mask, args.mask_ed, args.mask_es]):
        raise ValueError("Provide --mask or at least one of --mask_ed / --mask_es")

    ed_features = None
    es_features = None
    single_features = None

    if args.mask_ed:
        ed_path = Path(args.mask_ed)
        ed_cfg, es_cfg = parse_info_cfg(args.info_cfg)
        ed_feature_idx = choose_frame_index("ED", ed_cfg, es_cfg)
        ed_features = compute_phase_features(
            patient_id=args.patient_id,
            image_path=image_path,
            mask_path=ed_path,
            frame_tag="ED",
            frame_idx=ed_feature_idx,
            mask_source=args.mask_source,
        )

    if args.mask_es:
        es_path = Path(args.mask_es)
        ed_cfg, es_cfg = parse_info_cfg(args.info_cfg)
        es_feature_idx = choose_frame_index("ES", ed_cfg, es_cfg)
        es_features = compute_phase_features(
            patient_id=args.patient_id,
            image_path=image_path,
            mask_path=es_path,
            frame_tag="ES",
            frame_idx=es_feature_idx,
            mask_source=args.mask_source,
        )

    if args.mask:
        mask_path = Path(args.mask)
        frame_tag = args.frame_tag or infer_phase_from_path(mask_path)
        single_features = compute_phase_features(
            patient_id=args.patient_id,
            image_path=image_path,
            mask_path=mask_path,
            frame_tag=frame_tag,
            frame_idx=args.frame_idx,
            mask_source=args.mask_source,
        )

    df = flatten_feature_dicts(
        patient_id=args.patient_id,
        ed_features=ed_features,
        es_features=es_features,
        single_features=single_features,
    )

    if args.output_csv:
        output_csv = Path(args.output_csv)
    elif args.frame_tag:
        output_csv = Path(f"{args.patient_id}_{args.frame_tag}_features.csv")
    else:
        output_csv = Path(f"{args.patient_id}_features.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Feature extraction completed for {args.patient_id}")
    print(f"Output: {output_csv}")


if __name__ == "__main__":
    main()
