#!/usr/bin/env python3
"""
Optional LV -> MYO postprocessing for SORAT segmentations.

This script applies an intensity-aware correction on voxels currently labeled LV.
Dark LV regions can be relabeled to MYO under configurable safety constraints.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk

try:
    from geometry_utils import read_nifti_with_sitk_fallback, resample_image_to_reference_safe
except ImportError:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.getcwd())
    from geometry_utils import read_nifti_with_sitk_fallback, resample_image_to_reference_safe  # noqa: E402


LABEL_BG = 0
LABEL_RV = 1
LABEL_MYO = 2
LABEL_LV = 3


def _spatial_direction_from_any(direction: Tuple[float, ...]) -> tuple[float, ...]:
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


def _read_seg(path: Path) -> tuple[sitk.Image, np.ndarray]:
	img = sitk.ReadImage(str(path))
	arr = sitk.GetArrayFromImage(img)
	return img, arr


def _read_volume(path: Path) -> sitk.Image:
	return read_nifti_with_sitk_fallback(path)


def _parse_info_cfg(info_cfg: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
	if not info_cfg:
		return None, None
	p = Path(info_cfg)
	if not p.exists():
		return None, None

	ed = None
	es = None
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


def _extract_frame(image_4d: sitk.Image, frame_idx: int) -> sitk.Image:
	arr = sitk.GetArrayFromImage(image_4d)
	if arr.ndim == 3:
		return image_4d

	# SimpleITK 4D array layout is typically [t, z, y, x].
	t = arr.shape[0]
	idx = max(0, min(frame_idx, t - 1))
	frame = arr[idx]

	frame_img = sitk.GetImageFromArray(frame)
	spacing = image_4d.GetSpacing()
	origin = image_4d.GetOrigin()
	direction = image_4d.GetDirection()

	if len(spacing) >= 3:
		frame_img.SetSpacing(tuple(spacing[:3]))
	if len(origin) >= 3:
		frame_img.SetOrigin(tuple(origin[:3]))

	frame_img.SetDirection(_spatial_direction_from_any(direction))
	return frame_img


def _resample_to_seg(image: sitk.Image, seg_ref: sitk.Image) -> np.ndarray:
    """Resample ``image`` into the seg grid, robust to geometry gaps.

    Delegates to :func:`geometry_utils.resample_image_to_reference_safe` so the
    resample does not zero out the image when the seg carries a different
    physical origin / direction than the original image (e.g. VSA-3L on M&Ms).
    Returns the resampled image as a numpy array.
    """
    img_rs = resample_image_to_reference_safe(image, seg_ref)
    return sitk.GetArrayFromImage(img_rs)


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
	if not np.any(mask):
		return np.zeros_like(mask, dtype=np.int32), 0
	cc = sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8)))
	cc_arr = sitk.GetArrayFromImage(cc).astype(np.int32)
	return cc_arr, int(cc_arr.max())


def _component_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
	if radius <= 0:
		return mask
	img = sitk.GetImageFromArray(mask.astype(np.uint8))
	if mask.ndim == 2:
		dil = sitk.BinaryDilate(img, [radius, radius], sitk.sitkBall)
	else:
		dil = sitk.BinaryDilate(img, [radius, radius, radius], sitk.sitkBall)
	return sitk.GetArrayFromImage(dil).astype(bool)


def _smooth_binary_mask(mask: np.ndarray, radius: int, iterations: int) -> np.ndarray:
	"""Reduce jagged edges on binary masks with mild morphology."""
	if radius <= 0 or iterations <= 0 or not np.any(mask):
		return mask

	img = sitk.GetImageFromArray(mask.astype(np.uint8))
	for _ in range(iterations):
		img = sitk.BinaryMorphologicalClosing(img, [radius, radius, radius], sitk.sitkBall)
		img = sitk.BinaryMorphologicalOpening(img, [radius, radius, radius], sitk.sitkBall)

	img = sitk.BinaryFillhole(img)
	return sitk.GetArrayFromImage(img).astype(bool)


def _threshold_from_lv(
	lv_values: np.ndarray,
	mode: str,
	percentile: float,
	absolute: float,
) -> float:
	if mode == "absolute":
		return float(absolute)
	p = max(0.0, min(100.0, float(percentile)))
	return float(np.percentile(lv_values, p))


def _otsu_threshold(values: np.ndarray, bins: int = 64) -> Optional[float]:
	if values.size < 8:
		return None
	vmin = float(np.min(values))
	vmax = float(np.max(values))
	if vmax <= vmin:
		return None

	hist, edges = np.histogram(values, bins=bins, range=(vmin, vmax))
	hist = hist.astype(np.float64)
	if hist.sum() <= 0:
		return None

	centers = (edges[:-1] + edges[1:]) / 2.0
	weight_bg = np.cumsum(hist)
	weight_fg = np.cumsum(hist[::-1])[::-1]
	mean_bg = np.cumsum(hist * centers) / np.maximum(weight_bg, 1e-12)
	mean_fg = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight_fg[::-1], 1e-12))[::-1]

	between = weight_bg[:-1] * weight_fg[1:] * (mean_bg[:-1] - mean_fg[1:]) ** 2
	if between.size == 0:
		return None
	idx = int(np.argmax(between))
	return float(centers[idx])


def _frame_postprocess_legacy(
	seg: np.ndarray,
	frame_img_resampled: np.ndarray,
	threshold_mode: str,
	threshold_percentile: float,
	threshold_absolute: float,
	min_component_size: int,
	boundary_band_radius: int,
	adjacency_radius: int,
	max_relabel_fraction: float,
	min_remaining_lv_fraction: float,
	smoothing_radius: int,
	smoothing_iterations: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
	seg_out = seg.copy()
	lv_mask = seg == LABEL_LV
	myo_mask = seg == LABEL_MYO

	frame_stats = {
		"method": "legacy",
		"lv_voxels": int(lv_mask.sum()),
		"myo_voxels_before": int(myo_mask.sum()),
		"candidate_voxels": 0,
		"selected_voxels": 0,
		"changed_voxels": 0,
		"threshold_mode": threshold_mode,
		"threshold_value": None,
		"relabel_fraction_of_lv": 0.0,
	}

	if frame_stats["lv_voxels"] == 0:
		return seg_out, np.zeros_like(seg, dtype=np.uint8), frame_stats

	lv_values = frame_img_resampled[lv_mask]
	if lv_values.size == 0:
		return seg_out, np.zeros_like(seg, dtype=np.uint8), frame_stats

	thr = _threshold_from_lv(lv_values, threshold_mode, threshold_percentile, threshold_absolute)
	frame_stats["threshold_value"] = float(thr)

	candidate = lv_mask & (frame_img_resampled <= thr)

	# Keep candidates near MYO to avoid relabeling the whole LV cavity.
	if boundary_band_radius > 0:
		myo_band = _component_dilate(myo_mask, boundary_band_radius) & lv_mask
		candidate = candidate & myo_band
	frame_stats["candidate_voxels"] = int(candidate.sum())
	if frame_stats["candidate_voxels"] == 0:
		return seg_out, np.zeros_like(seg, dtype=np.uint8), frame_stats

	cc_arr, n_labels = _connected_components(candidate)
	selected = np.zeros_like(candidate, dtype=bool)

	for lab in range(1, n_labels + 1):
		comp = cc_arr == lab
		comp_size = int(comp.sum())
		if comp_size < int(min_component_size):
			continue

		if adjacency_radius > 0:
			comp_dil = _component_dilate(comp, adjacency_radius)
			if not np.any(comp_dil & myo_mask):
				continue

		selected |= comp

	# Smooth selected components while preserving anatomical constraints.
	if np.any(selected):
		selected = _smooth_binary_mask(selected, smoothing_radius, smoothing_iterations)
		selected = selected & lv_mask
		if boundary_band_radius > 0:
			myo_band = _component_dilate(myo_mask, boundary_band_radius) & lv_mask
			selected = selected & myo_band

	# Guardrail: cap relabel size as fraction of current LV.
	lv_vox = int(lv_mask.sum())
	max_fraction = max(0.0, min(1.0, float(max_relabel_fraction)))
	max_allowed = int(np.floor(max_fraction * lv_vox))

	# Additional guardrail: preserve a minimum fraction of LV voxels.
	min_lv_keep = int(np.floor(max(0.0, min(1.0, float(min_remaining_lv_fraction))) * lv_vox))
	max_by_preserve = max(0, lv_vox - min_lv_keep)
	max_allowed = min(max_allowed, max_by_preserve)

	if selected.sum() > max_allowed and max_allowed > 0:
		# Keep darkest selected voxels to preserve intent while respecting cap.
		selected_idx = np.where(selected)
		selected_int = frame_img_resampled[selected_idx]
		keep_order = np.argsort(selected_int)[:max_allowed]
		trimmed = np.zeros_like(selected, dtype=bool)
		trimmed[tuple(axis_idx[keep_order] for axis_idx in selected_idx)] = True
		selected = trimmed
	elif max_allowed == 0:
		selected = np.zeros_like(selected, dtype=bool)

	delta = selected.astype(np.uint8)
	seg_out[selected] = LABEL_MYO

	changed = int(delta.sum())
	frame_stats["selected_voxels"] = changed
	frame_stats["changed_voxels"] = changed
	frame_stats["relabel_fraction_of_lv"] = float(changed / lv_vox) if lv_vox > 0 else 0.0
	frame_stats["boundary_band_radius"] = int(boundary_band_radius)
	frame_stats["min_remaining_lv_fraction"] = float(min_remaining_lv_fraction)
	frame_stats["smoothing_radius"] = int(smoothing_radius)
	frame_stats["smoothing_iterations"] = int(smoothing_iterations)
	frame_stats["myo_voxels_after"] = int((seg_out == LABEL_MYO).sum())
	frame_stats["lv_voxels_after"] = int((seg_out == LABEL_LV).sum())

	return seg_out, delta, frame_stats


def _frame_postprocess_otsu_2d(
	seg: np.ndarray,
	frame_img_resampled: np.ndarray,
	strength: float,
	max_relabel_fraction: float,
	min_remaining_lv_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
	seg_out = seg.copy()
	lv_mask = seg == LABEL_LV
	myo_mask = seg == LABEL_MYO

	strength = float(max(0.0, min(1.0, strength)))
	# Strength increases correction aggressiveness, but explicit config limits still cap behavior.
	effective_max_relabel_fraction = min(
		max(0.0, min(1.0, float(max_relabel_fraction))),
		0.03 + (0.22 * strength),
	)
	effective_min_remaining_lv_fraction = max(
		max(0.0, min(1.0, float(min_remaining_lv_fraction))),
		0.70 - (0.20 * strength),
	)
	ring_radius = 1 if strength <= 0.6 else 2
	min_component_size = 2 if strength >= 0.7 else 3
	confidence_cutoff = 0.78 - (0.20 * strength)

	frame_stats = {
		"method": "otsu_2d",
		"lv_voxels": int(lv_mask.sum()),
		"myo_voxels_before": int(myo_mask.sum()),
		"seed_voxels": 0,
		"candidate_voxels": 0,
		"grown_voxels_added": 0,
		"selected_voxels": 0,
		"changed_voxels": 0,
		"rejected_small_components": 0,
		"rejected_no_adjacency": 0,
		"rejected_low_confidence": 0,
		"accepted_components": 0,
		"slices_with_otsu": 0,
		"threshold_mode": "otsu_2d_boundary_ring",
		"threshold_value": None,
		"relabel_fraction_of_lv": 0.0,
		"strength": strength,
		"effective_max_relabel_fraction": float(effective_max_relabel_fraction),
		"effective_min_remaining_lv_fraction": float(effective_min_remaining_lv_fraction),
	}

	lv_vox = int(lv_mask.sum())
	if lv_vox == 0:
		return seg_out, np.zeros_like(seg, dtype=np.uint8), frame_stats

	if seg.ndim == 2:
		seg3 = seg[np.newaxis, ...]
		img3 = frame_img_resampled[np.newaxis, ...]
		lv3 = lv_mask[np.newaxis, ...]
		myo3 = myo_mask[np.newaxis, ...]
	else:
		seg3 = seg
		img3 = frame_img_resampled
		lv3 = lv_mask
		myo3 = myo_mask

	selected3 = np.zeros_like(lv3, dtype=bool)
	thresholds = []

	for z in range(seg3.shape[0]):
		lv2 = lv3[z]
		myo2 = myo3[z]
		if not np.any(lv2) or not np.any(myo2):
			continue

		ring2 = _component_dilate(myo2, ring_radius) & lv2
		if int(ring2.sum()) < 8:
			continue

		vals = img3[z][ring2]
		thr = _otsu_threshold(vals, bins=48)
		if thr is None:
			continue

		frame_stats["slices_with_otsu"] += 1
		thresholds.append(float(thr))

		# Border-dark seeds anchor the relabeling to anatomically plausible MYO-adjacent zones.
		seed_candidates = ring2 & (img3[z] <= thr)
		seed_count = int(seed_candidates.sum())
		frame_stats["seed_voxels"] += seed_count
		if seed_count == 0:
			continue

		# Grow seeds through connected dark LV so interior dark zones can be corrected when
		# they are reachable from the boundary through dark tissue (avoids light-enclosed islands).
		ring_std = float(np.std(vals))
		domain_thr = float(thr + (0.10 + 0.40 * strength) * ring_std)
		domain_thr = min(domain_thr, float(np.percentile(vals, 95.0)))
		dark_domain = lv2 & (img3[z] <= domain_thr)

		candidates = seed_candidates.copy()
		max_growth_steps = int(1 + round(5 * strength))
		for _ in range(max_growth_steps):
			expanded = _component_dilate(candidates, 1) & dark_domain
			if int(expanded.sum()) == int(candidates.sum()):
				break
			candidates = expanded

		cand_count = int(candidates.sum())
		frame_stats["candidate_voxels"] += cand_count
		frame_stats["grown_voxels_added"] += max(0, cand_count - seed_count)
		if not np.any(candidates):
			continue

		cc_arr, n_labels = _connected_components(candidates)
		for lab in range(1, n_labels + 1):
			comp = cc_arr == lab
			comp_size = int(comp.sum())
			if comp_size < min_component_size:
				frame_stats["rejected_small_components"] += 1
				continue

			if not np.any(_component_dilate(comp, 1) & myo2):
				frame_stats["rejected_no_adjacency"] += 1
				continue

			vals_comp = img3[z][comp]
			darkness = 1.0 - float((np.mean(vals_comp) - np.min(vals)) / (np.ptp(vals) + 1e-8))
			darkness = float(max(0.0, min(1.0, darkness)))

			adj_ratio = float(((_component_dilate(comp, 1) & myo2).sum()) / max(comp_size, 1))
			adj_ratio = float(max(0.0, min(1.0, adj_ratio * 2.0)))

			seed_ratio = float(((comp & seed_candidates).sum()) / max(comp_size, 1))
			seed_ratio = float(max(0.0, min(1.0, seed_ratio * 2.0)))

			conf = 0.55 * darkness + 0.25 * adj_ratio + 0.20 * seed_ratio
			if conf < confidence_cutoff:
				frame_stats["rejected_low_confidence"] += 1
				continue

			frame_stats["accepted_components"] += 1
			selected3[z] |= comp

	selected = selected3[0] if seg.ndim == 2 else selected3

	max_allowed = int(np.floor(effective_max_relabel_fraction * lv_vox))
	min_lv_keep = int(np.floor(effective_min_remaining_lv_fraction * lv_vox))
	max_by_preserve = max(0, lv_vox - min_lv_keep)
	max_allowed = min(max_allowed, max_by_preserve)

	if selected.sum() > max_allowed and max_allowed > 0:
		selected_idx = np.where(selected)
		selected_int = frame_img_resampled[selected_idx]
		keep_order = np.argsort(selected_int)[:max_allowed]
		trimmed = np.zeros_like(selected, dtype=bool)
		trimmed[tuple(axis_idx[keep_order] for axis_idx in selected_idx)] = True
		selected = trimmed
	elif max_allowed == 0:
		selected = np.zeros_like(selected, dtype=bool)

	delta = selected.astype(np.uint8)
	seg_out[selected] = LABEL_MYO

	changed = int(delta.sum())
	frame_stats["selected_voxels"] = changed
	frame_stats["changed_voxels"] = changed
	frame_stats["relabel_fraction_of_lv"] = float(changed / lv_vox) if lv_vox > 0 else 0.0
	if thresholds:
		frame_stats["threshold_value"] = float(np.median(np.asarray(thresholds)))
	frame_stats["myo_voxels_after"] = int((seg_out == LABEL_MYO).sum())
	frame_stats["lv_voxels_after"] = int((seg_out == LABEL_LV).sum())

	return seg_out, delta, frame_stats


def process_case(
	patient_id: str,
	model: str,
	seg: Path,
	image_4d_path: Path,
	info_cfg: Optional[str],
	output: Path,
	delta: Path,
	frame_tag: str,
	frame_idx: int,
	summary_json: Path,
	method: str,
	strength: float,
	threshold_mode: str,
	threshold_percentile: float,
	threshold_absolute: float,
	min_component_size: int,
	boundary_band_radius: int,
	adjacency_radius: int,
	max_relabel_fraction: float,
	min_remaining_lv_fraction: float,
	smoothing_radius: int,
	smoothing_iterations: int,
) -> None:
	seg_img, seg_arr = _read_seg(seg)

	image_4d = _read_volume(image_4d_path)

	frame_img = _extract_frame(image_4d, frame_idx)
	frame_rs = _resample_to_seg(frame_img, seg_img)

	if method == "legacy":
		out, delta_arr, stats = _frame_postprocess_legacy(
			seg=seg_arr,
			frame_img_resampled=frame_rs,
			threshold_mode=threshold_mode,
			threshold_percentile=threshold_percentile,
			threshold_absolute=threshold_absolute,
			min_component_size=min_component_size,
			boundary_band_radius=boundary_band_radius,
			adjacency_radius=adjacency_radius,
			max_relabel_fraction=max_relabel_fraction,
			min_remaining_lv_fraction=min_remaining_lv_fraction,
			smoothing_radius=smoothing_radius,
			smoothing_iterations=smoothing_iterations,
		)
	else:
		out, delta_arr, stats = _frame_postprocess_otsu_2d(
			seg=seg_arr,
			frame_img_resampled=frame_rs,
			strength=strength,
			max_relabel_fraction=max_relabel_fraction,
			min_remaining_lv_fraction=min_remaining_lv_fraction,
		)

	out_img = sitk.GetImageFromArray(out.astype(np.uint8))
	out_img.CopyInformation(seg_img)
	sitk.WriteImage(out_img, str(output), useCompression=True)

	delta_img = sitk.GetImageFromArray(delta_arr.astype(np.uint8))
	delta_img.CopyInformation(seg_img)
	sitk.WriteImage(delta_img, str(delta), useCompression=True)

	summary = {
		"patient_id": patient_id,
		"model": model,
		"image": str(image_4d_path),
		"info_cfg": info_cfg or "",
		"frames": {
			frame_tag: {"frame_index": int(frame_idx), **stats},
		},
		"total_changed_voxels": int(stats["changed_voxels"]),
		"params": {
			"method": method,
			"strength": float(strength),
			"threshold_mode": threshold_mode,
			"threshold_percentile": float(threshold_percentile),
			"threshold_absolute": float(threshold_absolute),
			"min_component_size": int(min_component_size),
			"boundary_band_radius": int(boundary_band_radius),
			"adjacency_radius": int(adjacency_radius),
			"max_relabel_fraction": float(max_relabel_fraction),
			"min_remaining_lv_fraction": float(min_remaining_lv_fraction),
			"smoothing_radius": int(smoothing_radius),
			"smoothing_iterations": int(smoothing_iterations),
		},
	}
	summary_json.write_text(json.dumps(summary, indent=2))


def main() -> None:
	parser = argparse.ArgumentParser(description="Optional LV->MYO postprocessing")
	parser.add_argument("--patient_id", required=True)
	parser.add_argument("--model", required=True)
	parser.add_argument("--seg", required=True)
	parser.add_argument("--image", required=True, help="Original patient image (3D/4D)")
	parser.add_argument("--info_cfg", default="", help="Optional Info.cfg for legacy fallback")
	parser.add_argument("--output", required=True)
	parser.add_argument("--delta", required=True)
	parser.add_argument("--frame_tag", required=True)
	parser.add_argument("--frame_idx", type=int, required=True)
	parser.add_argument("--summary_json", required=True)
	parser.add_argument("--method", choices=["otsu", "legacy"], default="otsu")
	parser.add_argument("--strength", type=float, default=0.5, help="Conservative-to-aggressive relabel strength [0,1]")

	parser.add_argument("--threshold_mode", choices=["percentile", "absolute"], default="percentile")
	parser.add_argument("--threshold_percentile", type=float, default=15.0)
	parser.add_argument("--threshold_absolute", type=float, default=0.0)
	parser.add_argument("--min_component_size", type=int, default=20)
	parser.add_argument("--boundary_band_radius", type=int, default=2)
	parser.add_argument("--adjacency_radius", type=int, default=1)
	parser.add_argument("--max_relabel_fraction", type=float, default=0.20)
	parser.add_argument("--min_remaining_lv_fraction", type=float, default=0.55)
	parser.add_argument("--smoothing_radius", type=int, default=1)
	parser.add_argument("--smoothing_iterations", type=int, default=1)

	args = parser.parse_args()

	process_case(
		patient_id=args.patient_id,
		model=args.model,
		seg=Path(args.seg),
		image_4d_path=Path(args.image),
		info_cfg=args.info_cfg.strip() or None,
		output=Path(args.output),
		delta=Path(args.delta),
		frame_tag=args.frame_tag,
		frame_idx=args.frame_idx,
		summary_json=Path(args.summary_json),
		method=args.method,
		strength=args.strength,
		threshold_mode=args.threshold_mode,
		threshold_percentile=args.threshold_percentile,
		threshold_absolute=args.threshold_absolute,
		min_component_size=args.min_component_size,
		boundary_band_radius=args.boundary_band_radius,
		adjacency_radius=args.adjacency_radius,
		max_relabel_fraction=args.max_relabel_fraction,
		min_remaining_lv_fraction=args.min_remaining_lv_fraction,
		smoothing_radius=args.smoothing_radius,
		smoothing_iterations=args.smoothing_iterations,
	)

	print(f"Postprocessing complete for {args.patient_id} / {args.model}")


if __name__ == "__main__":
	main()
