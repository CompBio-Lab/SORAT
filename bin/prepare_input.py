#!/usr/bin/env python3
"""
Convert a samplesheet image (and optional ground truth) to NIfTI.

SORAT's preprocessing reads NIfTI. This step lets a samplesheet point at other
formats instead, without converting datasets by hand:

* NIfTI (``.nii``, ``.nii.gz``) is passed through unchanged.
* NRRD, MetaImage (``.mha``/``.mhd``) and other files SimpleITK can read are
  converted with their geometry.
* A directory is read as DICOM, together with its immediate subdirectories.
  It must contain exactly one image series unless ``--dicom_series`` names
  one, or several separated by ``;`` that form one stack (scanners that store
  each short-axis slice as its own series). A series with one image per slice
  position becomes a 3-D volume; a cine series with several images per
  position becomes a 4-D volume, slices ordered along the slice normal and
  frames ordered by trigger time (then acquisition time, then instance number).

Some releases store a placeholder voxel spacing (the 2018 Atrial Segmentation
Challenge stores 1 x 1 x 1). ``--image_spacing XxYxZ`` (mm, e.g. 0.625x0.625x1.25;
commas also accepted) replaces the
spatial spacing of the image and its ground truth; it is never inferred. When
no override is given and every spatial spacing is exactly 1.0, a warning is
printed so a missing value cannot pass unnoticed.

Each call writes ``<patient_id>_input_provenance.json`` recording the source
paths and SHA-256 checksums, the reader, any DICOM series UID, the output
geometry and any override.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

NIFTI_SUFFIXES = (".nii", ".nii.gz")

# DICOM tags (group|element) used to order cine images.
TAG_POSITION = "0020|0032"
TAG_ORIENTATION = "0020|0037"
TAG_TRIGGER_TIME = "0018|1060"
TAG_ACQUISITION_TIME = "0008|0032"
TAG_INSTANCE = "0020|0013"
TAG_SERIES_TIME = "0008|0031"
TAG_ROWS = "0028|0010"
TAG_COLUMNS = "0028|0011"
TAG_PIXEL_SPACING = "0028|0030"


def is_nifti(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith(NIFTI_SUFFIXES)


def parse_spacing(text: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if text is None or not str(text).strip():
        return None
    normalized = str(text).lower().replace("x", ",").replace(";", ",")
    try:
        parts = [float(v) for v in normalized.split(",") if v.strip()]
    except ValueError:
        parts = []
    if len(parts) != 3 or any(v <= 0 for v in parts):
        raise ValueError(f"image_spacing must be three positive numbers in mm (e.g. 0.625x0.625x1.25), got '{text}'")
    return tuple(parts)


def sha256_of(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------

def _series_files(directory: Path) -> Dict[str, List[str]]:
    """Map each DICOM series UID to its files, in ``directory`` and its immediate subdirectories.

    Scanners that store one series per short-axis slice (Siemens) leave the
    stack spread over several folders; looking one level down lets a
    samplesheet point at a folder that holds (or links to) all of them.
    """
    found: Dict[str, List[str]] = {}
    for folder in [directory] + sorted(p for p in directory.iterdir() if p.is_dir()):
        if not any(f.is_file() for f in folder.iterdir()):
            continue
        for uid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or []:
            found.setdefault(uid, []).extend(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), uid))
    return found


def _read_tags(path: str) -> Dict[str, str]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOff()
    reader.ReadImageInformation()
    return {k: reader.GetMetaData(k).strip() for k in reader.GetMetaDataKeys()}


def _float_list(text: str) -> List[float]:
    return [float(v) for v in text.replace(" ", "").split("\\") if v]


def _frame_key(tags: Dict[str, str]) -> Tuple[float, str, int]:
    trigger = float(tags[TAG_TRIGGER_TIME]) if tags.get(TAG_TRIGGER_TIME) else float("inf")
    return trigger, tags.get(TAG_ACQUISITION_TIME, ""), int(float(tags.get(TAG_INSTANCE, "0") or 0))


def _series_time(tags: Dict[str, Dict[str, str]], files: List[str]) -> str:
    return min(tags[f].get(TAG_SERIES_TIME) or tags[f].get(TAG_ACQUISITION_TIME, "") for f in files)


def _drop_repeated_positions(by_series: Dict[str, List[str]], tags: Dict[str, Dict[str, str]],
                             normal: np.ndarray) -> Tuple[List[str], List[str]]:
    """Keep one series per slice position: the latest acquired, as a repeat replaces a failed take."""
    def positions(files):
        return {round(float(np.dot(normal, _float_list(tags[f][TAG_POSITION]))), 3) for f in files}

    order = sorted(by_series, key=lambda uid: _series_time(tags, by_series[uid]), reverse=True)
    taken, kept, dropped = set(), [], []
    for uid in order:
        pos = positions(by_series[uid])
        if pos & taken:
            dropped.append(uid)
            continue
        taken |= pos
        kept.extend(by_series[uid])
    return kept, dropped


def read_dicom_series(directory: Path, series_uid: Optional[str] = None) -> Tuple[sitk.Image, dict]:
    """Read one DICOM series as a 3-D or 4-D image."""
    image, info, _ = _read_dicom(directory, series_uid)
    return image, info


def _read_dicom(directory: Path, series_uid: Optional[str]) -> Tuple[sitk.Image, dict, List[str]]:
    """Read one series, or several (``uid1;uid2;...``) that together form one stack.

    Several series must share orientation, matrix and pixel spacing; they are
    combined by slice position. When two series cover the same position, the
    later one is used and the other is listed under ``dropped_series``.
    """
    available = _series_files(directory)
    ids = list(available)
    if not ids:
        raise ValueError(f"{directory}: no DICOM series found")
    if series_uid is None:
        if len(ids) > 1:
            raise ValueError(
                f"{directory}: {len(ids)} DICOM series found ({', '.join(ids)}); "
                "set dicom_series in the samplesheet to choose one"
            )
        wanted = ids
    else:
        wanted = [u.strip() for u in series_uid.split(";") if u.strip()]
        missing = [u for u in wanted if u not in available]
        if missing:
            raise ValueError(f"{directory}: series {', '.join(missing)} not found (available: {', '.join(ids)})")

    by_series = {uid: available[uid] for uid in wanted}
    tags = {f: _read_tags(f) for files in by_series.values() for f in files}
    dropped: List[str] = []
    if len(wanted) == 1:
        files = by_series[wanted[0]]
    else:
        first = {k: tags[by_series[wanted[0]][0]].get(k, "") for k in (TAG_ROWS, TAG_COLUMNS, TAG_PIXEL_SPACING)}
        reference = _float_list(tags[by_series[wanted[0]][0]][TAG_ORIENTATION])
        for uid in wanted[1:]:
            t = tags[by_series[uid][0]]
            if any(t.get(k, "") != v for k, v in first.items()):
                raise ValueError(f"{directory}: series {uid} differs in matrix or pixel spacing from {wanted[0]}")
            if not np.allclose(_float_list(t[TAG_ORIENTATION]), reference, atol=1e-3):
                raise ValueError(f"{directory}: series {uid} has a different orientation from {wanted[0]}")
        files, dropped = _drop_repeated_positions(by_series, tags, np.cross(reference[:3], reference[3:]))

    orientation = _float_list(tags[files[0]][TAG_ORIENTATION])
    normal = np.cross(orientation[:3], orientation[3:])
    by_position: Dict[float, List[str]] = {}
    for f in files:
        distance = float(np.dot(normal, _float_list(tags[f][TAG_POSITION])))
        by_position.setdefault(round(distance, 3), []).append(f)

    positions = sorted(by_position)
    counts = {len(v) for v in by_position.values()}
    if len(counts) != 1:
        raise ValueError(f"{directory}: slice positions have unequal numbers of images {sorted(counts)}")
    n_frames = counts.pop()
    info = {"reader": "dicom", "series_uid": ";".join(u for u in wanted if u not in dropped),
            "n_files": len(files), "n_slices": len(positions), "n_frames": n_frames}
    if dropped:
        info["dropped_series"] = dropped

    if n_frames == 1:
        ordered = [by_position[p][0] for p in positions]
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(ordered)
        return reader.Execute(), info, files

    # Cine: one 3-D volume per frame, all sharing the slice ordering.
    volumes = []
    frame_lists = [sorted(by_position[p], key=lambda f: _frame_key(tags[f])) for p in positions]
    for t in range(n_frames):
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([frames[t] for frames in frame_lists])
        volumes.append(reader.Execute())
    return sitk.JoinSeries(volumes), info, files


# ---------------------------------------------------------------------------
# Generic reading and writing
# ---------------------------------------------------------------------------

def read_any(path: Path, series_uid: Optional[str] = None) -> Tuple[sitk.Image, dict, List[Path]]:
    if path.is_dir():
        image, info, files = _read_dicom(path, series_uid)
        return image, info, [Path(f) for f in files]
    if not path.is_file():
        raise FileNotFoundError(path)
    return sitk.ReadImage(str(path)), {"reader": "simpleitk", "format": "".join(path.suffixes)}, [path]


def set_spatial_spacing(image: sitk.Image, spacing: Tuple[float, float, float]) -> sitk.Image:
    full = list(image.GetSpacing())
    full[:3] = spacing
    image.SetSpacing(full)
    return image


def geometry(image: sitk.Image) -> dict:
    return {"size": list(image.GetSize()), "spacing": [float(v) for v in image.GetSpacing()],
            "origin": [float(v) for v in image.GetOrigin()], "direction": [float(v) for v in image.GetDirection()]}


def prepare(
    path: Path,
    out_path: Path,
    spacing: Optional[Tuple[float, float, float]],
    series_uid: Optional[str],
    role: str,
) -> dict:
    """Write ``path`` as NIfTI at ``out_path`` (or pass NIfTI through) and describe it."""
    # Nextflow stages inputs as symlinks; record the real file so provenance
    # points at the dataset, not the work directory.
    source = str(path.resolve())
    if is_nifti(path) and spacing is None:
        return {"role": role, "source": source, "converted": False, "output": str(path)}

    image, info, sources = read_any(path, series_uid)
    if spacing is not None:
        image = set_spatial_spacing(image, spacing)
    elif all(abs(s - 1.0) < 1e-9 for s in image.GetSpacing()[:3]):
        print(f"WARNING: {path} has spacing 1 x 1 x 1; if that is a placeholder, set image_spacing "
              "in the samplesheet", file=sys.stderr)
    sitk.WriteImage(image, str(out_path), useCompression=True)
    return {"role": role, "source": source, "converted": True, "output": str(out_path),
            "source_sha256": sha256_of(sources), "spacing_override_mm": list(spacing) if spacing else None,
            **info, "geometry": geometry(image)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert samplesheet inputs to NIfTI with provenance")
    parser.add_argument("--patient_id", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--ground_truth", default="", help="Optional ground truth (file or DICOM directory)")
    parser.add_argument("--image_spacing", default="", help="Spatial spacing override in mm, 'x,y,z'")
    parser.add_argument("--dicom_series", default="", help="DICOM SeriesInstanceUID of the image; several separated by ';' are stacked")
    parser.add_argument("--out_dir", default=".", type=Path)
    args = parser.parse_args()

    spacing = parse_spacing(args.image_spacing)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_out = args.out_dir / f"{args.patient_id}_image.nii.gz"
    records = [prepare(args.image, image_out, spacing, args.dicom_series or None, "image")]
    if not records[0]["converted"]:
        # The row is here because its ground truth needs converting; keep the
        # NIfTI image unchanged but expose it under the expected name.
        source = args.image.resolve()
        if source.name.lower().endswith(".nii.gz"):
            image_out.unlink(missing_ok=True)
            image_out.symlink_to(source)
        else:
            sitk.WriteImage(sitk.ReadImage(str(source)), str(image_out), useCompression=True)
        records[0]["output"] = str(image_out)
    if args.ground_truth:
        gt = prepare(Path(args.ground_truth), args.out_dir / f"{args.patient_id}_gt.nii.gz", spacing, None,
                     "ground_truth")
        records.append(gt)
    (args.out_dir / f"{args.patient_id}_input_provenance.json").write_text(
        json.dumps({"patient_id": args.patient_id, "files": records}, indent=2))
    for r in records:
        print(f"{args.patient_id} {r['role']}: {'converted' if r['converted'] else 'NIfTI, unchanged'} -> {r['output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
