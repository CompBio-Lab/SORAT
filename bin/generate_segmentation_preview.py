#!/usr/bin/env python3
"""
Generate quick segmentation preview PNGs for ED/ES outputs.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import SimpleITK as sitk

DEFAULT_LABELS = {
    1: ("RV", np.array([0.95, 0.25, 0.25], dtype=np.float32), "fill"),
    2: ("MYO", np.array([0.95, 0.80, 0.20], dtype=np.float32), "fill"),
    3: ("LV", np.array([0.20, 0.55, 1.0], dtype=np.float32), "fill"),
}

ATRIAL_LABELS = {
    1: ("Wall", np.array([1.0, 0.30, 0.30], dtype=np.float32), "contour"),
    2: ("RA", np.array([0.25, 0.85, 0.30], dtype=np.float32), "fill"),
    3: ("LA", np.array([0.25, 0.50, 1.0], dtype=np.float32), "fill"),
}


def _spatial_direction_from_any(direction: Tuple[float, ...]) -> Tuple[float, ...]:
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


def parse_info_cfg(info_cfg: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not info_cfg:
        return None, None

    p = Path(info_cfg)
    if not p.exists():
        return None, None

    ed, es = None, None
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("ED:"):
            try:
                ed = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("ES:"):
            try:
                es = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
    return ed, es


def extract_frame(image: sitk.Image, frame_idx: int) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image)
    if arr.ndim == 3:
        return image

    idx = max(0, min(frame_idx, arr.shape[0] - 1))
    frame = arr[idx]

    out = sitk.GetImageFromArray(frame)
    out.SetSpacing(tuple(image.GetSpacing()[:3]))
    out.SetOrigin(tuple(image.GetOrigin()[:3]))
    out.SetDirection(_spatial_direction_from_any(image.GetDirection()))
    return out


def resample_image_to_reference(image: sitk.Image, reference: sitk.Image) -> np.ndarray:
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(reference)
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(0)
    return sitk.GetArrayFromImage(rs.Execute(image))


def resample_label_to_reference(label: sitk.Image, reference: sitk.Image) -> sitk.Image:
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(reference)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(0)
    return rs.Execute(label)


def normalize(img: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(img, [1, 99])
    if p99 <= p1:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - p1) / (p99 - p1)
    return np.clip(out, 0.0, 1.0)


def choose_slice(seg: np.ndarray) -> int:
    if seg.ndim == 2:
        return 0
    fg = (seg > 0).sum(axis=(1, 2))
    if np.any(fg > 0):
        return int(np.argmax(fg))
    return int(seg.shape[0] // 2)


def choose_slice_with_gt(pred: np.ndarray, gt: Optional[np.ndarray]) -> int:
    pred_idx = choose_slice(pred)
    if gt is None:
        return pred_idx

    if gt.ndim == 2:
        gt_fg = int((gt > 0).sum())
        pred_fg = int((pred > 0).sum()) if pred.ndim == 2 else int((pred[pred_idx] > 0).sum())
        return 0 if gt_fg > pred_fg else pred_idx

    gt_fg = (gt > 0).sum(axis=(1, 2))
    if np.any(gt_fg > 0):
        gt_idx = int(np.argmax(gt_fg))
        if pred.ndim == 3:
            pred_fg = (pred > 0).sum(axis=(1, 2))
            if np.any(pred_fg > 0) and int(pred_fg[gt_idx]) >= int(pred_fg[pred_idx]):
                return gt_idx
        return gt_idx

    return pred_idx


def infer_architecture(model: str, architecture: str) -> str:
    arch = (architecture or "").strip().lower()
    if arch:
        return arch
    if "__" in model:
        return model.split("__", 1)[0].lower()
    if model.startswith("atrial_nnunet"):
        return "atrial_nnunet"
    return "ventricular"


def get_label_definitions(architecture: str) -> dict[int, tuple[str, np.ndarray, str]]:
    if architecture == "atrial_nnunet":
        return ATRIAL_LABELS
    return DEFAULT_LABELS


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    denom = float(pred.sum() + gt.sum())
    if denom == 0.0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def _frame_candidates(frame_idx: Optional[int]) -> list[int]:
    if frame_idx is None:
        return []

    candidates = []
    for value in [frame_idx, frame_idx - 1, frame_idx + 1]:
        if value is None:
            continue
        if 0 <= value <= 99 and value not in candidates:
            candidates.append(value)
    return candidates


def resolve_phase_ground_truth(
    patient_id: str,
    phase: str,
    ground_truth: str,
    ed_idx: Optional[int],
    es_idx: Optional[int],
) -> Optional[sitk.Image]:
    if not ground_truth:
        return None

    gt_path = Path(ground_truth)
    if not gt_path.exists():
        return None

    if gt_path.is_file():
        return sitk.ReadImage(str(gt_path))

    candidates: list[Path] = []

    if phase == "ED":
        candidates.extend(
            [
                gt_path / f"{patient_id}_frame01_gt.nii.gz",
                gt_path / f"{patient_id}_ED_gt.nii.gz",
                gt_path / f"{patient_id}_sax_ed_gt.nii.gz",
            ]
        )
        for idx in _frame_candidates(ed_idx):
            candidates.append(gt_path / f"{patient_id}_frame{idx:02d}_gt.nii.gz")
    else:
        candidates.extend(
            [
                gt_path / f"{patient_id}_ES_gt.nii.gz",
                gt_path / f"{patient_id}_sax_es_gt.nii.gz",
            ]
        )
        for idx in _frame_candidates(es_idx):
            candidates.append(gt_path / f"{patient_id}_frame{idx:02d}_gt.nii.gz")
        for idx in range(2, 30):
            candidates.append(gt_path / f"{patient_id}_frame{idx:02d}_gt.nii.gz")

    for candidate in candidates:
        if candidate.exists():
            return sitk.ReadImage(str(candidate))

    return None


def to_overlay(gray: np.ndarray, seg: np.ndarray, label_defs: dict[int, tuple[str, np.ndarray, str]], alpha: float = 0.45) -> np.ndarray:
    base = np.stack([gray, gray, gray], axis=-1)
    out = base.copy()

    for label in sorted(np.unique(seg)):
        if label <= 0:
            continue
        _, color, mode = label_defs.get(int(label), (f"Label {label}", np.array([1.0, 1.0, 0.0], dtype=np.float32), "fill"))
        if mode != "fill":
            continue
        mask = seg == label
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color

    return np.clip(out, 0.0, 1.0)


def to_mask_rgb(seg: np.ndarray, label_defs: dict[int, tuple[str, np.ndarray, str]]) -> np.ndarray:
    rgb = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.float32)
    for label in sorted(np.unique(seg)):
        if label <= 0:
            continue
        _, color, mode = label_defs.get(int(label), (f"Label {label}", np.array([1.0, 1.0, 0.0], dtype=np.float32), "fill"))
        if mode != "fill":
            continue
        rgb[seg == label] = color
    return np.clip(rgb, 0.0, 1.0)


def add_contours(ax, seg: np.ndarray, label_defs: dict[int, tuple[str, np.ndarray, str]]) -> None:
    for label in sorted(np.unique(seg)):
        if label <= 0:
            continue
        _, color, mode = label_defs.get(int(label), (f"Label {label}", np.array([1.0, 1.0, 0.0], dtype=np.float32), "fill"))
        if mode == "contour":
            ax.contour(seg == label, levels=[0.5], colors=[color], linewidths=2.0)


def make_difference_map(pred: np.ndarray, gt: np.ndarray, label_defs: dict[int, tuple[str, np.ndarray, str]]) -> np.ndarray:
    diff = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.float32)
    diff[..., 0] = 0.08  # subtle dark red background for context

    labels = sorted({int(x) for x in np.unique(pred) if int(x) > 0} | {int(x) for x in np.unique(gt) if int(x) > 0})
    for label in labels:
        _, color, _ = label_defs.get(label, (f"Label {label}", np.array([1.0, 0.65, 0.1], dtype=np.float32), "fill"))
        mismatch = (pred == label) != (gt == label)
        diff[mismatch] = color

    return np.clip(diff, 0.0, 1.0)


def render_preview(
    patient_id: str,
    model: str,
    architecture: str,
    phase: str,
    image_arr: np.ndarray,
    seg_arr: np.ndarray,
    gt_arr: Optional[np.ndarray],
    output_png: Path,
) -> None:
    z = choose_slice_with_gt(seg_arr, gt_arr)

    if image_arr.ndim == 2:
        image_2d = normalize(image_arr)
    else:
        image_2d = normalize(image_arr[z])

    if seg_arr.ndim == 2:
        seg_2d = seg_arr
    else:
        seg_2d = seg_arr[z]

    label_defs = get_label_definitions(architecture)

    gt_2d = None
    if gt_arr is not None:
        gt_2d = gt_arr if gt_arr.ndim == 2 else gt_arr[z]

    pred_overlay = to_overlay(image_2d, seg_2d, label_defs)

    if gt_2d is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"{patient_id} | {model} | {phase}", fontsize=12)

        axes[0].imshow(image_2d, cmap="gray")
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(pred_overlay)
        axes[1].set_title("Segmentation Overlay")
        axes[1].axis("off")

        present_labels = [int(x) for x in sorted(np.unique(seg_2d)) if int(x) > 0]
        add_contours(axes[1], seg_2d, label_defs)

        legend_handles = []
        for label in present_labels:
            name, color, _ = label_defs.get(label, (f"Label {label}", np.array([1.0, 1.0, 0.0], dtype=np.float32), "fill"))
            legend_handles.append(Patch(facecolor=color, edgecolor="black", label=f"{label}: {name}"))

        if legend_handles:
            fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=max(1, len(legend_handles)), frameon=True)

        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))
        fig.savefig(str(output_png), dpi=170, bbox_inches="tight")
        plt.close(fig)
        return

    gt_overlay = to_overlay(image_2d, gt_2d, label_defs)
    pred_mask = to_mask_rgb(seg_2d, label_defs)
    gt_mask = to_mask_rgb(gt_2d, label_defs)
    diff_map = make_difference_map(seg_2d, gt_2d, label_defs)

    present_labels = sorted({int(x) for x in np.unique(seg_2d) if int(x) > 0} | {int(x) for x in np.unique(gt_2d) if int(x) > 0})
    dice_by_label: dict[int, float] = {}
    for label in present_labels:
        pred_mask_bin = (seg_2d == label).astype(np.uint8)
        gt_mask_bin = (gt_2d == label).astype(np.uint8)
        dice_by_label[label] = dice_score(pred_mask_bin, gt_mask_bin)

    mean_dsc = float(np.mean(list(dice_by_label.values()))) if dice_by_label else float("nan")

    summary_parts = []
    for label in present_labels:
        name, _, _ = label_defs.get(label, (f"Label {label}", np.array([1.0, 1.0, 0.0], dtype=np.float32), "fill"))
        summary_parts.append(f"{name}={dice_by_label[label]:.3f}")
    summary_text = " | ".join(summary_parts) if summary_parts else "No foreground labels"

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"{patient_id} - {phase} Frame - DSC: {summary_text}", fontsize=12, fontweight="bold")

    axes[0, 0].imshow(image_2d, cmap="gray")
    axes[0, 0].set_title(f"Original - {patient_id} ({phase}, Slice {z})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_mask)
    add_contours(axes[0, 1], seg_2d, label_defs)
    axes[0, 1].set_title(f"Prediction ({model})")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(gt_mask)
    add_contours(axes[0, 2], gt_2d, label_defs)
    axes[0, 2].set_title("Ground Truth")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(pred_overlay)
    add_contours(axes[1, 0], seg_2d, label_defs)
    axes[1, 0].set_title("Prediction Overlay")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gt_overlay)
    add_contours(axes[1, 1], gt_2d, label_defs)
    axes[1, 1].set_title("Ground Truth Overlay")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(diff_map)
    mean_dsc_text = f"{mean_dsc:.3f}" if np.isfinite(mean_dsc) else "n/a"
    axes[1, 2].set_title(f"Difference Map\nMean DSC: {mean_dsc_text}")
    axes[1, 2].axis("off")

    legend_handles = []
    for label in present_labels:
        name, color, _ = label_defs.get(label, (f"Label {label}", np.array([1.0, 1.0, 0.0], dtype=np.float32), "fill"))
        legend_handles.append(Patch(facecolor=color, edgecolor="black", label=f"{name} (DSC={dice_by_label[label]:.3f})"))

    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=max(1, len(legend_handles)), frameon=True)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    fig.savefig(str(output_png), dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ED/ES segmentation preview PNGs")
    parser.add_argument("--patient_id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--ground_truth", default="")
    parser.add_argument("--seg_ed", required=True)
    parser.add_argument("--seg_es", required=True)
    parser.add_argument("--output_ed_png", required=True)
    parser.add_argument("--output_es_png", required=True)
    parser.add_argument("--info_cfg", default="")
    parser.add_argument("--architecture", default="", help="Model architecture (e.g., atrial_nnunet)")
    args = parser.parse_args()

    architecture = infer_architecture(args.model, args.architecture)

    image = sitk.ReadImage(args.image)
    seg_ed_img = sitk.ReadImage(args.seg_ed)
    seg_es_img = sitk.ReadImage(args.seg_es)

    ed_cfg, es_cfg = parse_info_cfg(args.info_cfg.strip() or None)

    image_arr = sitk.GetArrayFromImage(image)
    n_frames = image_arr.shape[0] if image_arr.ndim == 4 else 1

    ed_idx = ed_cfg if ed_cfg is not None else 0
    es_idx = es_cfg if es_cfg is not None else (1 if n_frames > 1 else 0)

    ed_frame = extract_frame(image, ed_idx)
    es_frame = extract_frame(image, es_idx)

    seg_ed_arr = sitk.GetArrayFromImage(seg_ed_img)
    seg_es_arr = sitk.GetArrayFromImage(seg_es_img)

    ed_img_arr = resample_image_to_reference(ed_frame, seg_ed_img)
    es_img_arr = resample_image_to_reference(es_frame, seg_es_img)

    gt_arg = args.ground_truth.strip()
    ed_gt_img = resolve_phase_ground_truth(args.patient_id, "ED", gt_arg, ed_cfg, es_cfg)
    es_gt_img = resolve_phase_ground_truth(args.patient_id, "ES", gt_arg, ed_cfg, es_cfg)

    # Single-label datasets (e.g., atrial MBAS) often provide one GT volume.
    if es_gt_img is None and ed_gt_img is not None:
        es_gt_img = ed_gt_img

    ed_gt_arr = None
    if ed_gt_img is not None:
        ed_gt_arr = sitk.GetArrayFromImage(resample_label_to_reference(ed_gt_img, seg_ed_img))

    es_gt_arr = None
    if es_gt_img is not None:
        es_gt_arr = sitk.GetArrayFromImage(resample_label_to_reference(es_gt_img, seg_es_img))

    render_preview(
        patient_id=args.patient_id,
        model=args.model,
        architecture=architecture,
        phase="ED",
        image_arr=ed_img_arr,
        seg_arr=seg_ed_arr,
        gt_arr=ed_gt_arr,
        output_png=Path(args.output_ed_png),
    )

    render_preview(
        patient_id=args.patient_id,
        model=args.model,
        architecture=architecture,
        phase="ES",
        image_arr=es_img_arr,
        seg_arr=seg_es_arr,
        gt_arr=es_gt_arr,
        output_png=Path(args.output_es_png),
    )


if __name__ == "__main__":
    main()
