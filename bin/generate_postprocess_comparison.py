#!/usr/bin/env python3
"""
Generate side-by-side visualization for original vs postprocessed segmentations.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import SimpleITK as sitk


LABEL_COLORS = {
    1: np.array([1.0, 0.0, 0.0], dtype=np.float32),  # RV
    2: np.array([0.0, 0.7, 0.0], dtype=np.float32),  # MYO
    3: np.array([0.1, 0.1, 1.0], dtype=np.float32),  # LV
}


def spatial_direction_from_any(direction: Tuple[float, ...]) -> Tuple[float, ...]:
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


def parse_info_cfg(path: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not path:
        return None, None
    p = Path(path)
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
    frame_img = sitk.GetImageFromArray(frame)
    sp = image.GetSpacing()
    org = image.GetOrigin()
    frame_img.SetSpacing(tuple(sp[:3]) if len(sp) >= 3 else (1.0, 1.0, 1.0))
    frame_img.SetOrigin(tuple(org[:3]) if len(org) >= 3 else (0.0, 0.0, 0.0))
    frame_img.SetDirection(spatial_direction_from_any(image.GetDirection()))
    return frame_img


def resample_to_seg(image: sitk.Image, seg: sitk.Image) -> np.ndarray:
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(seg)
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetTransform(sitk.Transform())
    rf.SetDefaultPixelValue(0)
    return sitk.GetArrayFromImage(rf.Execute(image))


def normalize(img: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(img, [1, 99])
    if p99 <= p1:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - p1) / (p99 - p1)
    return np.clip(out, 0.0, 1.0)


def to_overlay(gray: np.ndarray, seg: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.stack([gray, gray, gray], axis=-1)
    out = base.copy()
    for label, color in LABEL_COLORS.items():
        mask = seg == label
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0.0, 1.0)


def choose_slice(seg_before: np.ndarray, seg_after: np.ndarray, delta: np.ndarray) -> int:
    if np.any(delta > 0):
        changed = delta.sum(axis=(1, 2))
        return int(np.argmax(changed))
    lv_mass = ((seg_before == 3) | (seg_after == 3)).sum(axis=(1, 2))
    return int(np.argmax(lv_mass))


def render(
    patient_id: str,
    model: str,
    frame_name: str,
    image_arr: np.ndarray,
    seg_before: np.ndarray,
    seg_after: np.ndarray,
    delta: np.ndarray,
    output_png: Path,
) -> None:
    z = choose_slice(seg_before, seg_after, delta)
    img2d = normalize(image_arr[z])

    before_ov = to_overlay(img2d, seg_before[z])
    after_ov = to_overlay(img2d, seg_after[z])

    delta2d = delta[z] > 0
    delta_rgb = np.zeros((*delta2d.shape, 3), dtype=np.float32)
    delta_rgb[..., 0] = delta2d.astype(np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{patient_id} - {frame_name} - {model} - Postprocess Comparison", fontsize=13)

    axes[0, 0].imshow(img2d, cmap="gray")
    axes[0, 0].set_title("Original Image")

    axes[0, 1].imshow(before_ov)
    axes[0, 1].set_title("Prediction (Before)")

    axes[1, 0].imshow(after_ov)
    axes[1, 0].set_title("Prediction (After)")

    axes[1, 1].imshow(img2d, cmap="gray")
    axes[1, 1].imshow(delta_rgb, alpha=0.7)
    axes[1, 1].set_title(f"Delta LV->MYO (changed={int(delta.sum())})")

    for ax in axes.ravel():
        ax.axis("off")

    legend_handles = [
        Patch(facecolor=LABEL_COLORS[1], edgecolor="white", label="RV"),
        Patch(facecolor=LABEL_COLORS[2], edgecolor="white", label="MYO"),
        Patch(facecolor=LABEL_COLORS[3], edgecolor="white", label="LV"),
        Patch(facecolor=(1.0, 0.0, 0.0), edgecolor="black", hatch="///", label="Delta: LV->MYO changed"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=True,
    )

    plt.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize postprocess delta")
    parser.add_argument("--patient_id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frame_tag", required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--image", required=True)
    parser.add_argument("--info_cfg", default="")
    parser.add_argument("--seg_before", required=True)
    parser.add_argument("--seg_after", required=True)
    parser.add_argument("--delta", required=True)
    parser.add_argument("--output_png", required=True)
    args = parser.parse_args()

    image_4d = sitk.ReadImage(args.image)
    arr = sitk.GetArrayFromImage(image_4d)
    n_frames = arr.shape[0] if arr.ndim == 4 else 1

    frame_idx = args.frame_idx
    if frame_idx < 0 or frame_idx >= n_frames:
        ed_cfg, es_cfg = parse_info_cfg(args.info_cfg.strip() or None)
        fallback = ed_cfg if ed_cfg is not None else 0
        frame_idx = max(0, min(fallback, n_frames - 1))

    frame_img = extract_frame(image_4d, frame_idx)
    seg_before_img = sitk.ReadImage(args.seg_before)
    seg_after_img = sitk.ReadImage(args.seg_after)
    delta_img = sitk.ReadImage(args.delta)

    image_arr = resample_to_seg(frame_img, seg_before_img)
    before_arr = sitk.GetArrayFromImage(seg_before_img)
    after_arr = sitk.GetArrayFromImage(seg_after_img)
    delta_arr = sitk.GetArrayFromImage(delta_img)

    render(
        patient_id=args.patient_id,
        model=args.model,
        frame_name=args.frame_tag,
        image_arr=image_arr,
        seg_before=before_arr,
        seg_after=after_arr,
        delta=delta_arr,
        output_png=Path(args.output_png),
    )
    print(f"Saved {args.output_png}")


if __name__ == "__main__":
    main()
