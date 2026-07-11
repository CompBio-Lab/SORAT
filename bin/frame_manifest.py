"""
Shared frame-manifest helpers for the SORAT pipeline.
Provides a single source of truth for Info.cfg parsing, frame-list construction,
and frame-level ground-truth resolution.

Replaces the duplicated parse_info_cfg functions previously scattered across
preprocess, postprocess, features, visualization, and comparison scripts.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _read_nifti(path: Union[str, Path], dtype=None):
    try:
        from geometry_utils import read_nifti_with_sitk_fallback

        return read_nifti_with_sitk_fallback(path, dtype=dtype)
    except ImportError:
        import SimpleITK as sitk

        return sitk.ReadImage(str(path))


# ---------------------------------------------------------------------------
# Info.cfg parsing
# ---------------------------------------------------------------------------

def parse_info_cfg(info_cfg_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Parse an ACDC-style Info.cfg file to extract ED / ES frame indices.

    Returns ``{"ed_frame": <int>, "es_frame": <int|None>}``.
    When *info_cfg_path* is ``None`` or the file does not exist the defaults
    ``ed_frame=0`` and ``es_frame=None`` are returned.
    """
    info: Dict[str, Any] = {"ed_frame": 0, "es_frame": None}
    if not info_cfg_path:
        return info

    cfg = Path(info_cfg_path)
    if not cfg.exists():
        return info

    try:
        with open(cfg, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key == "ed":
                    info["ed_frame"] = int(value)
                elif key == "es":
                    info["es_frame"] = int(value)
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Frame-manifest construction
# ---------------------------------------------------------------------------

def discover_annotated_frames(
    gt_path: Optional[Union[str, Path]],
    num_frames: int,
    min_voxels: int = 10,
) -> List[int]:
    """Scan a 4-D ground-truth file and return indices of labelled frames.

    Datasets such as M&Ms store a single 4-D GT volume in which only a few
    temporal frames carry manual annotations.  This helper reads the file,
    counts non-zero voxels per frame, and returns the sorted list of frame
    indices that exceed *min_voxels*.

    Returns an empty list when *gt_path* is missing, is a directory
    (ACDC-style per-frame files), or contains no annotated frames.
    """
    if not gt_path:
        return []

    gt = Path(gt_path)
    if not gt.exists() or not gt.is_file():
        return []

    try:
        import numpy as np
        import SimpleITK as sitk

        gt_img = _read_nifti(gt)
        gt_array = sitk.GetArrayFromImage(gt_img)

        if len(gt_array.shape) < 4:
            return []  # 3-D file — not a multi-frame GT

        annotated: List[int] = []
        for idx in range(min(gt_array.shape[0], num_frames)):
            if np.count_nonzero(gt_array[idx]) > min_voxels:
                annotated.append(idx)
        return annotated
    except Exception:
        return []


def build_frame_manifest(
    info_cfg_path: Optional[Union[str, Path]],
    num_frames: int,
    patient_id: Optional[str] = None,
    frames_mode: str = "auto",
    max_frames: Optional[int] = None,
    ground_truth_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Build a frame manifest describing which frames to process.

    Parameters
    ----------
    info_cfg_path:
        Path to an ACDC ``Info.cfg`` (may be ``None`` / empty).
    num_frames:
        Total number of temporal frames in the 4D volume.
    patient_id:
        Optional patient identifier written into the manifest.
    frames_mode:
        ``"auto"`` — use ED/ES when *info_cfg_path* provides them, else
        discover annotated frames from *ground_truth_path* (M&Ms), else
        all frames.
        ``"ed_es"`` — force 2-frame mode regardless of *info_cfg_path*.
        ``"all"`` — always produce all frames.
    max_frames:
        When *frames_mode* results in all-frames, cap to at most *max_frames*
        by evenly sub-sampling the cardiac cycle (default: no cap).
    ground_truth_path:
        Optional path to a ground-truth file or directory.  When no
        *info_cfg_path* is available, a 4-D GT file is scanned for annotated
        frames; the first and last annotated frames are treated as ED/ES.

    Returns
    -------
    dict
        .. code-block:: json

           {
             "patient_id": "<id>",
             "has_info_cfg": true,
             "num_frames": 30,
             "frames": [{"tag": "ED", "idx": 0}, {"tag": "ES", "idx": 12}]
           }
    """
    has_info_cfg = bool(info_cfg_path and Path(info_cfg_path).exists())
    info = parse_info_cfg(info_cfg_path)

    frames: List[Dict[str, Any]] = []

    if has_info_cfg and frames_mode in ("auto", "ed_es"):
        ed_idx = info.get("ed_frame", 0)
        es_idx = info.get("es_frame")
        if es_idx is None or es_idx >= num_frames:
            es_idx = num_frames - 1
        frames = [
            {"tag": "ED", "idx": ed_idx},
            {"tag": "ES", "idx": es_idx},
        ]
    elif frames_mode == "ed_es":
        frames = [
            {"tag": "ED", "idx": 0},
            {"tag": "ES", "idx": num_frames - 1 if num_frames > 1 else 0},
        ]
    else:
        # All-frames mode.  Datasets without an Info.cfg (e.g. M&Ms) have no
        # ED/ES annotations, so processing every frame is rarely desired and
        # is very slow.  When a 4-D GT file is available, scan it for
        # annotated frames and use the first/last as ED/ES approximations.
        # Otherwise default to a 2-frame (first + last) approximation when
        # ``frames_mode='auto'`` and no cap was requested.
        # Pass ``--frames_mode all`` to force every frame, or ``--max_frames 0``
        # to disable the cap (both yield all frames).

        # GT-aware frame discovery (M&Ms: only some frames have annotations).
        gt_annotated = discover_annotated_frames(ground_truth_path, num_frames)
        if (
            frames_mode == "auto"
            and not has_info_cfg
            and gt_annotated
        ):
            if len(gt_annotated) >= 2:
                frames = [
                    {"tag": "ED", "idx": gt_annotated[0]},
                    {"tag": "ES", "idx": gt_annotated[-1]},
                ]
            else:
                frames = [{"tag": "ED", "idx": gt_annotated[0]}]
            return {
                "patient_id": patient_id,
                "has_info_cfg": has_info_cfg,
                "num_frames": num_frames,
                "frames": frames,
            }

        effective_max = max_frames
        if (
            frames_mode == "auto"
            and not has_info_cfg
            and effective_max is None
        ):
            effective_max = 2

        indices = list(range(num_frames))
        if effective_max is not None and effective_max > 0 and effective_max < num_frames:
            if effective_max == 1:
                indices = [0]
            elif effective_max == 2:
                indices = [0, num_frames - 1]
            else:
                step = (num_frames - 1) / (effective_max - 1)
                indices = sorted(
                    set(
                        [0]
                        + [int(round(i * step)) for i in range(1, effective_max - 1)]
                        + [num_frames - 1]
                    )
                )
                indices = sorted(set(i for i in indices if 0 <= i < num_frames))
        frames = [{"tag": f"frame{idx:02d}", "idx": idx} for idx in indices]

    return {
        "patient_id": patient_id,
        "has_info_cfg": has_info_cfg,
        "num_frames": num_frames,
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def write_manifest(manifest: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write *manifest* dict to a JSON file at *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def read_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a JSON frame manifest from *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(path: Optional[Union[str, Path]]) -> Optional[Dict[str, Any]]:
    """Load a manifest if *path* exists, otherwise return ``None``."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return read_manifest(p)


# ---------------------------------------------------------------------------
# Frame-level ground-truth resolution
# ---------------------------------------------------------------------------

_ATRIAL_ARCHITECTURES = {"atrial_nnunet", "atrial"}
_VENTRICULAR_ARCHITECTURES = {"cinema", "nnformer", "vsa3l", "ventricular", "sax"}


def _architecture_family(architecture: str) -> str:
    """Return the anatomical label family for an architecture string."""
    arch = (architecture or "").strip().lower()
    if arch in _ATRIAL_ARCHITECTURES or arch.startswith("atrial_nnunet"):
        return "atrial"
    if not arch or arch in _VENTRICULAR_ARCHITECTURES:
        return "ventricular"
    if arch.split("__", 1)[0] in _VENTRICULAR_ARCHITECTURES:
        return "ventricular"
    # Metrics default unknown architectures to ventricular labels, so keep GT
    # canonicalization aligned with that downstream contract.
    return "ventricular"


def _is_atrial_architecture(architecture: str) -> bool:
    return _architecture_family(architecture) == "atrial"


def _canonicalize_3d_gt(gt_path: Path, architecture: str) -> Path:
    """Canonicalize a per-frame 3-D GT file to the SORAT
    convention (1=RV, 2=MYO, 3=LV) when the architecture is ventricular.

    Datasets with a raw LV/RV ordering different from SORAT (e.g. M&Ms-2,
    whose per-frame files use 1=LV, 2=MYO, 3=RV) are normalized before
    metrics and previews. ACDC (1=RV, 2=MYO, 3=LV) is left unchanged by the
    anatomy-based canonicalizer, so existing datasets are unaffected.

    Atrial architectures keep their own label semantics and the raw file is
    returned as-is.  On any error (e.g. SimpleITK unavailable) the raw file is
    returned so behaviour never degrades below the previous logic.
    """
    if _is_atrial_architecture(architecture):
        return gt_path

    try:
        import numpy as np
        import SimpleITK as sitk

        gt_img = _read_nifti(gt_path, dtype=np.int16)
        gt_array = sitk.GetArrayFromImage(gt_img)

        if gt_array.ndim != 3:
            return gt_path  # only 3-D per-frame files are canonicalized here

        canonical = _remap_cardiac_labels(gt_array.astype(np.int32), np, architecture)
        if np.array_equal(canonical, gt_array.astype(np.int32)):
            return gt_path  # already canonical (e.g. ACDC) — return raw, no copy

        out_img = sitk.GetImageFromArray(canonical.astype(np.int16))
        out_img.SetSpacing(gt_img.GetSpacing())
        out_img.SetOrigin(gt_img.GetOrigin())
        out_img.SetDirection(gt_img.GetDirection())

        fd, tmp_path = tempfile.mkstemp(
            suffix=".nii.gz",
            prefix=f"{gt_path.stem}_canon_",
        )
        os.close(fd)
        sitk.WriteImage(out_img, tmp_path, useCompression=True)
        return Path(tmp_path)
    except Exception:
        return gt_path


def resolve_frame_ground_truth(
    gt_path: Optional[Union[str, Path]],
    frame_tag: str,
    frame_idx: int,
    patient_id: str,
    architecture: str = "ventricular",
) -> Optional[Path]:
    """Find the ground-truth NIfTI for a concrete frame.

    Handles two common dataset layouts:

    * **ACDC-style** — ``gt_path`` is a **directory** containing per-frame
      ``{patient}_frameNN_gt.nii.gz`` files.
    * **MMS-style** — ``gt_path`` is a **single multi-frame file**.  The
      frame at *frame_idx* is extracted; if it contains no labels (all zero)
      ``None`` is returned.

    Returns
    -------
    Path or None
        A usable GT NIfTI path, or ``None`` when no ground truth is available
        for this frame.
    """
    if not gt_path:
        return None

    gt = Path(gt_path)
    if not gt.exists():
        return None

    # --- directory: ACDC-style -------------------------------------------
    if gt.is_dir():
        candidates: List[Path] = []
        candidates.append(gt / f"{patient_id}_frame{frame_idx + 1:02d}_gt.nii.gz")
        candidates.append(gt / f"{patient_id}_frame{frame_idx:02d}_gt.nii.gz")
        candidates.append(gt / f"{patient_id}_{frame_tag}_gt.nii.gz")
        candidates.append(gt / f"{patient_id}_sax_{frame_tag.lower()}_gt.nii.gz")
        # M&Ms-2 axis-tagged layout ({pid}_SA_{tag}_gt.nii.gz / {pid}_LA_{tag}_gt.nii.gz):
        # try both SA/LA prefixes (patient_id does not carry the axis here).
        for _axis in ("SA", "LA"):
            candidates.append(gt / f"{patient_id}_{_axis}_{frame_tag}_gt.nii.gz")
            candidates.append(gt / f"{patient_id}_{_axis}_{frame_tag.lower()}_gt.nii.gz")
        if frame_tag == "ED":
            candidates.append(gt / f"{patient_id}_ED_gt.nii.gz")
            candidates.append(gt / f"{patient_id}_sax_ed_gt.nii.gz")
            candidates.append(gt / f"{patient_id}_frame01_gt.nii.gz")
            for _axis in ("SA", "LA"):
                candidates.append(gt / f"{patient_id}_{_axis}_ED_gt.nii.gz")
        elif frame_tag == "ES":
            candidates.append(gt / f"{patient_id}_ES_gt.nii.gz")
            candidates.append(gt / f"{patient_id}_sax_es_gt.nii.gz")
            for _axis in ("SA", "LA"):
                candidates.append(gt / f"{patient_id}_{_axis}_ES_gt.nii.gz")

        for cand in candidates:
            if cand.exists():
                return _canonicalize_3d_gt(cand, architecture)
        return None

    # --- file: MMS-style multi-frame GT ----------------------------------
    if gt.is_file():
        import numpy as np
        import SimpleITK as sitk

        gt_img = _read_nifti(gt, dtype=np.int16)
        gt_array = sitk.GetArrayFromImage(gt_img)

        if len(gt_array.shape) >= 4:
            num_gt_frames = gt_array.shape[0]
            if frame_idx < num_gt_frames:
                frame_data = gt_array[int(frame_idx)]
                if np.count_nonzero(frame_data) <= 10:
                    return None  # unlabelled frame

                # Canonicalize GT labels to the SORAT convention (1=RV, 2=MYO,
                # 3=LV) expected by downstream metric and visualization code.
                # The mapping is derived from anatomy (LV = cavity enclosed by
                # the myocardial ring) so it is robust to per-dataset label
                # conventions and ordering (ACDC, M&Ms, future datasets).
                frame_data = _remap_cardiac_labels(frame_data, np, architecture=architecture)

                frame_img = sitk.GetImageFromArray(frame_data)
                _copy_spatial_metadata(gt_img, frame_img)

                fd, tmp_path = tempfile.mkstemp(
                    suffix=".nii.gz",
                    prefix=f"{patient_id}_{frame_tag}_gt_",
                )
                os.close(fd)
                sitk.WriteImage(frame_img, tmp_path, useCompression=True)
                return Path(tmp_path)
            return None

        # 3-D file -> canonicalize like directory-resolved per-frame GT.
        return _canonicalize_3d_gt(gt, architecture)

    return None


def _apply_label_remap(frame_data, remap, np):
    """Apply a ``{old: new}`` label remap (unmapped values become 0)."""
    out = np.zeros_like(frame_data)
    for old, new in remap.items():
        out[frame_data == old] = new
    return out


def _label_containment_scores(frame_data, np, labels):
    """Score each label as a possible myocardial ring."""
    try:
        from scipy import ndimage
    except Exception:
        return []

    data3 = frame_data[np.newaxis, ...] if frame_data.ndim == 2 else frame_data
    scores = []

    for myo_label in labels:
        enclosed = np.zeros_like(data3, dtype=bool)
        myo_mask = data3 == myo_label
        for z in range(data3.shape[0]):
            myo_slice = myo_mask[z]
            if not myo_slice.any():
                continue
            filled = ndimage.binary_fill_holes(myo_slice)
            enclosed[z] = filled & ~myo_slice

        if not enclosed.any():
            continue

        cavity_scores = {}
        for label in labels:
            if label == myo_label:
                continue
            label_mask = data3 == label
            total = int(label_mask.sum())
            if total == 0:
                cavity_scores[label] = 0.0
            else:
                cavity_scores[label] = float((label_mask & enclosed).sum()) / float(total)

        if not cavity_scores:
            continue

        lv_label, lv_score = max(cavity_scores.items(), key=lambda item: item[1])
        scores.append((float(lv_score), myo_label, lv_label))

    return sorted(scores, reverse=True)


def _canonicalize_with_known_myo(frame_data, np, labels, myo_label):
    """Map labels when the myocardium label is known or assumed."""
    cavity_labels = [label for label in labels if label != myo_label]
    if len(cavity_labels) != 2:
        return None

    low, high = sorted(cavity_labels)

    # Prefer containment to identify the LV cavity.
    try:
        from scipy import ndimage

        data3 = frame_data[np.newaxis, ...] if frame_data.ndim == 2 else frame_data
        myo3 = data3 == myo_label
        enclosed = np.zeros_like(data3, dtype=bool)
        for z in range(data3.shape[0]):
            if myo3[z].any():
                enclosed[z] = ndimage.binary_fill_holes(myo3[z]) & ~myo3[z]

        if enclosed.any():
            scores = {}
            for label in cavity_labels:
                mask = data3 == label
                total = int(mask.sum())
                scores[label] = float((mask & enclosed).sum()) / float(total) if total else 0.0
            lv_label, lv_score = max(scores.items(), key=lambda item: item[1])
            if lv_score > 0.1:
                rv_label = next(label for label in cavity_labels if label != lv_label)
                return _apply_label_remap(frame_data, {rv_label: 1, myo_label: 2, lv_label: 3}, np)
    except Exception:
        pass

    # Fallback: LV centroid should sit closer to MYO than RV.
    myo_coords = np.argwhere(frame_data == myo_label)
    if len(myo_coords) > 0:
        myo_centroid = myo_coords.mean(axis=0)
        distances = {}
        for label in cavity_labels:
            coords = np.argwhere(frame_data == label)
            distances[label] = np.linalg.norm(coords.mean(axis=0) - myo_centroid) if len(coords) else float("inf")
        lv_label, lv_dist = min(distances.items(), key=lambda item: item[1])
        rv_label = next(label for label in cavity_labels if label != lv_label)
        if np.isfinite(lv_dist):
            return _apply_label_remap(frame_data, {rv_label: 1, myo_label: 2, lv_label: 3}, np)

    # Last resort: sorted cavity identity around the assumed myocardium.
    return _apply_label_remap(frame_data, {low: 1, myo_label: 2, high: 3}, np)


def _canonicalize_ventricular_labels(frame_data, np, nonzero_labels):
    """Map three ventricular labels onto 1=RV, 2=MYO, 3=LV by anatomy.

    Primary inference chooses the myocardium as the label whose filled ring
    encloses another foreground label, then maps that enclosed cavity to LV.
    If the ring is broken or ambiguous, the fallback keeps the previous
    practical assumption that the middle-valued label is MYO and resolves the
    two cavity labels by containment/centroid.
    """
    labels = list(nonzero_labels)

    scores = _label_containment_scores(frame_data, np, labels)
    if scores:
        best_score, myo_label, lv_label = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else 0.0
        if best_score > 0.1 and best_score >= (second_score + 0.05):
            rv_candidates = [label for label in labels if label not in {myo_label, lv_label}]
            if len(rv_candidates) == 1:
                return _apply_label_remap(frame_data, {rv_candidates[0]: 1, myo_label: 2, lv_label: 3}, np)

    # Silent fallback for ambiguous anatomy: preserve the previous robust
    # behavior for known SAX datasets while still allowing LV/RV swapping.
    assumed_myo = sorted(labels)[1]
    fallback = _canonicalize_with_known_myo(frame_data, np, labels, assumed_myo)
    if fallback is not None:
        return fallback

    remap = {old: new for new, old in enumerate(sorted(labels), start=1)}
    return _apply_label_remap(frame_data, remap, np)


def _remap_cardiac_labels(frame_data, np, architecture: str = "ventricular"):
    """Canonicalize GT labels to the SORAT convention (1=RV, 2=MYO, 3=LV).

    For ventricular architectures the mapping is derived from anatomy (see
    ``_canonicalize_ventricular_labels``) so it is robust to each dataset's
    raw label values *and* RV/LV ordering -- ACDC (1=RV,2=MYO,3=LV), M&Ms
    (1=LV,2=MYO,3=RV), and 85/170/255-style encodings all map correctly.

    Atrial architectures keep their own label semantics and are returned
    unchanged (or sorted-order remapped when raw values exceed 3).
    """
    nonzero_labels = sorted(int(v) for v in np.unique(frame_data) if v != 0)

    if not nonzero_labels:
        return frame_data

    if _is_atrial_architecture(architecture):
        # No LV/RV containment concept; only normalize non-standard values.
        if nonzero_labels[-1] > 3:
            remap = {old: new for new, old in enumerate(nonzero_labels, start=1)}
            return _apply_label_remap(frame_data, remap, np)
        return frame_data

    # Ventricular: three labels -> anatomy-based canonicalization.
    if len(nonzero_labels) == 3:
        return _canonicalize_ventricular_labels(frame_data, np, nonzero_labels)

    # Any other ventricular label count: normalize only non-standard values.
    if nonzero_labels[-1] > 3:
        remap = {old: new for new, old in enumerate(nonzero_labels, start=1)}
        return _apply_label_remap(frame_data, remap, np)
    return frame_data


def _copy_spatial_metadata(src, dst):
    """Copy spatial (3‑D) metadata from a multi‑dim *src* image to *dst*."""
    spacing = src.GetSpacing()
    origin = src.GetOrigin()
    direction = src.GetDirection()
    dst.SetSpacing(spacing[:3])
    dst.SetOrigin(origin[:3])
    # Build a valid 3×3 direction cosine matrix from the N‑D direction.
    ndim = int(len(direction) ** 0.5)
    if ndim >= 4:
        dir_3d = [
            direction[0], direction[1], direction[2],
            direction[4], direction[5], direction[6],
            direction[8], direction[9], direction[10],
        ]
    elif ndim == 3:
        dir_3d = list(direction)
    else:
        dir_3d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    dst.SetDirection(dir_3d)
