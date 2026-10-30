#!/usr/bin/env python3
"""
Build a SORAT samplesheet from DICOM exams stored as <root>/<subject>/<date>/<series>/.

The short-axis cine of each exam is chosen among series folders matching
--include. A single multi-slice series is used directly. Several series that
share geometry (one series per slice, or a stack split in parts) are linked
into a staging folder and listed in dicom_series for prepare_input.py to stack.
Writes samplesheet.csv and exams.csv (the choice made for every exam).
"""

import argparse
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk

TAGS = {
    "series_uid": "0020|000e", "manufacturer": "0008|0070", "model": "0008|1090", "field_T": "0018|0087",
    "orientation": "0020|0037", "position": "0020|0032", "rows": "0028|0010", "cols": "0028|0011",
    "pixel_spacing": "0028|0030", "frames": "0018|1090", "series_time": "0008|0031",
}
EXAM_FIELDS = ["patient_id", "status", "reason", "layout", "manufacturer", "model", "field_T", "rows", "cols",
               "pixel_spacing", "frames", "n_slices", "n_series", "series_dirs", "series_uids"]


def first_header(folder):
    """Tags of the first readable image in a series folder, plus its file count."""
    files = sorted(p for p in folder.iterdir() if p.is_file())
    for f in files[:5]:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(f))
        try:
            reader.ReadImageInformation()
        except RuntimeError:
            continue
        tags = {k: reader.GetMetaData(t).strip() if reader.HasMetaDataKey(t) else "" for k, t in TAGS.items()}
        tags["n_files"] = len(files)
        return tags
    return None


def floats(text):
    return [float(v) for v in text.split("\\") if v.strip()]


def n_positions(tags):
    return max(1, round(tags["n_files"] / max(1, int(float(tags["frames"] or 1)))))


def slice_distance(tags):
    o = floats(tags["orientation"])
    return float(np.dot(np.cross(o[:3], o[3:]), floats(tags["position"])))


def same_geometry(a, b):
    keys = ("rows", "cols", "pixel_spacing", "frames")
    return all(a[k] == b[k] for k in keys) and np.allclose(floats(a["orientation"]), floats(b["orientation"]), atol=1e-3)


def exam_id(subject, date):
    try:
        date = datetime.strptime(date, "%d%b%Y").strftime("%Y%m%d")
    except ValueError:
        pass
    return re.sub(r"[^A-Za-z0-9]", "", subject) + "_" + re.sub(r"[^A-Za-z0-9]", "", date)


def choose(exam, include, exclude, min_slices):
    row = {"patient_id": exam_id(exam.parent.name, exam.name)}
    folders = [p for p in sorted(exam.iterdir())
               if p.is_dir() and include.search(p.name) and not (exclude and exclude.search(p.name))]
    headers = {f: t for f in folders if (t := first_header(f)) and t["orientation"] and t["position"]}
    if not headers:
        return {**row, "status": "skipped", "reason": "no matching series"}

    stacks = {f: t for f, t in headers.items() if n_positions(t) > 1}
    if stacks:
        ref = next(iter(stacks.values()))
        if len(stacks) == 1:
            chosen, layout, reason = stacks, "stack", "one stack series"
        elif all(same_geometry(ref, t) for t in stacks.values()):
            chosen, layout, reason = stacks, "split_stack", f"{len(stacks)} stack series combined"
        else:
            best = max(stacks, key=lambda f: (stacks[f]["n_files"], stacks[f]["series_time"]))
            chosen, layout, reason = {best: stacks[best]}, "stack", f"largest of {len(stacks)} stack series"
        n_slices = sum(n_positions(t) for t in chosen.values())
    else:
        ref = max(headers.values(), key=lambda t: sum(same_geometry(t, u) for u in headers.values()))
        chosen = {f: t for f, t in headers.items() if same_geometry(ref, t)}
        layout, reason = "slice_series", f"{len(chosen)} single-slice series combined"
        n_slices = len({round(slice_distance(t), 1) for t in chosen.values()})

    if n_slices < min_slices:
        return {**row, "status": "skipped", "reason": f"only {n_slices} slices"}
    first = next(iter(chosen.values()))
    return {**row, "status": "ok", "reason": reason, "layout": layout,
            **{k: first[k] for k in ("manufacturer", "model", "field_T", "rows", "cols", "pixel_spacing", "frames")},
            "n_slices": n_slices, "n_series": len(chosen), "series_dirs": ";".join(map(str, chosen)),
            "series_uids": ";".join(t["series_uid"] for t in chosen.values())}


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build a SORAT samplesheet from a folder of DICOM exams")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--include", required=True, help="Regex matching short-axis cine series folder names")
    parser.add_argument("--exclude", default="", help="Regex for folder names to leave out")
    parser.add_argument("--min_slices", type=int, default=6)
    parser.add_argument("--subjects", nargs="*", help="Only these subject folders")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    include, exclude = re.compile(args.include), re.compile(args.exclude) if args.exclude else None
    subjects = args.subjects or sorted(p.name for p in args.root.iterdir() if p.is_dir())
    exams = [e for s in subjects for e in sorted((args.root / s).iterdir()) if e.is_dir()]
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda e: choose(e, include, exclude, args.min_slices), exams))

    sheet = []
    for row in (r for r in rows if r["status"] == "ok"):
        dirs = row["series_dirs"].split(";")
        if len(dirs) == 1:
            sheet.append({"patient_id": row["patient_id"], "image": dirs[0], "dicom_series": ""})
            continue
        stage = args.out_dir / "staging" / row["patient_id"]
        stage.mkdir(parents=True, exist_ok=True)
        for d in map(Path, dirs):
            if not (stage / d.name).is_symlink():
                (stage / d.name).symlink_to(d, target_is_directory=True)
        sheet.append({"patient_id": row["patient_id"], "image": str(stage), "dicom_series": row["series_uids"]})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "exams.csv", EXAM_FIELDS, rows)
    write_csv(args.out_dir / "samplesheet.csv", ["patient_id", "image", "dicom_series"], sheet)
    print(f"{len(rows)} exams: {len(sheet)} in the samplesheet, {len(rows) - len(sheet)} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
