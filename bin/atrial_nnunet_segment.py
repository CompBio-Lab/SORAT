#!/usr/bin/env python3
"""
Atrial nnUNet Segmentation Script for SORAT Pipeline.

Runs atrial segmentation inference using nnUNetv2 model.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

try:
    from frame_manifest import read_manifest, write_manifest
except ImportError:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.getcwd())
    from frame_manifest import read_manifest, write_manifest  # noqa: F401


def _sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def parse_folds_arg(folds: str) -> list[str]:
    """Parse fold argument into a list suitable for nnUNetv2 CLI."""
    folds = (folds or "all").strip().lower()
    if folds in {"", "all"}:
        return []

    parsed = []
    for token in folds.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"Invalid fold value '{token}'. Expected comma-separated integers or 'all'.")
        parsed.append(token)

    if not parsed:
        return []
    return parsed


def run_nnunet_predict(
    input_dir: Path,
    output_dir: Path,
    model_dir: Path,
    dataset_id: str,
    configuration: str,
    folds: str,
    save_probabilities: bool,
) -> None:
    """Execute nnUNetv2_predict using subprocess."""
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_name = f"nnUNetTrainer__nnUNetPlans__{configuration}"

    # Resolve from concrete trainer locations so nnUNet_results points to the right root.
    # Supported examples:
    # 1) /models/atrial_nnunet/nnUNet_results/Dataset001_LGE/nnUNetTrainer__nnUNetPlans__2d
    # 2) /models/atrial_nnunet/nnUNet_results/Dataset001_LGE/Dataset001_LGE/nnUNetTrainer__nnUNetPlans__2d
    # 3) /models/atrial_nnunet/Dataset001_LGE/nnUNetTrainer__nnUNetPlans__2d
    candidate_trainer_dirs = [
        model_dir / "nnUNet_results" / dataset_id / trainer_name,
        model_dir / "nnUNet_results" / dataset_id / dataset_id / trainer_name,
        model_dir / dataset_id / trainer_name,
    ]

    trainer_dir = next(
        (p for p in candidate_trainer_dirs if (p / "dataset.json").exists()),
        None,
    )

    results_root = trainer_dir.parent.parent if trainer_dir else None

    if results_root is None:
        raise FileNotFoundError(
            f"Could not find a valid trainer folder with dataset.json for '{dataset_id}' and configuration '{configuration}' under {model_dir}"
        )

    os.environ["nnUNet_raw"] = str(model_dir / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(model_dir / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(results_root)
    os.environ.setdefault("NNUNET_COMPILE", "0")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    for key in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    # Temporary compatibility entrypoint. This wrapper is a no-op on runtimes
    # that already expose numpy._core and only patches legacy NumPy runtimes
    # needed to unpickle some checkpoints.
    command = [
        "python",
        "/app/bin/nnunet_predict_compat.py",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        dataset_id,
        "-c",
        configuration,
    ]

    fold_list = parse_folds_arg(folds)
    if fold_list:
        command += ["-f", *fold_list]

    if save_probabilities:
        command.append("--save_probabilities")

    run_env = os.environ.copy()
    # Ensure host user-site packages do not override container NumPy/SciPy.
    run_env["PYTHONNOUSERSITE"] = "1"

    subprocess.run(command, check=True, env=run_env)


def map_outputs_to_frames(output_files, expected_tags):
    """Map nnUNet output filenames back to frame tags using the manifest.

    Guarantees that *every* output file is preserved.  When the manifest is
    missing or doesn't cover all outputs, tags are derived from filenames
    (``_ED_``/``_ES_``/``_frameNN_``) and any remainder is assigned a
    positional ``frameNN`` tag so no segmentation is silently dropped.
    """
    tag_re = re.compile(r"_(ED|ES|frame\d{2,})\.nii\.gz$", re.IGNORECASE)

    def _tag_from_name(path):
        m = tag_re.search(path.name)
        return m.group(1) if m else None

    frame_outputs = {}
    used = set()

    # Pass 1 — match each output to an expected manifest tag (case-insensitive).
    for output_file in sorted(output_files):
        name_lower = output_file.name.lower()
        for tag in expected_tags:
            if tag in frame_outputs:
                continue
            if f"_{tag.lower()}" in name_lower:
                frame_outputs[tag] = output_file
                used.add(output_file)
                break

    # Pass 2 — assign remaining outputs in order to remaining expected tags.
    remaining_tags = [t for t in expected_tags if t not in frame_outputs]
    unmatched = [f for f in sorted(output_files) if f not in used]
    for i, tag in enumerate(remaining_tags):
        if i < len(unmatched):
            frame_outputs[tag] = unmatched[i]
            used.add(unmatched[i])

    # Pass 3 — derive a tag from the filename for any still-unmatched output.
    for output_file in sorted(output_files):
        if output_file in used:
            continue
        derived = _tag_from_name(output_file)
        if derived and derived not in frame_outputs:
            frame_outputs[derived] = output_file
            used.add(output_file)

    # Pass 4 — any leftover outputs get a positional frameNN tag.
    frame_counter = 0
    for output_file in sorted(output_files):
        if output_file in used:
            continue
        while f"frame{frame_counter:02d}" in frame_outputs:
            frame_counter += 1
        frame_outputs[f"frame{frame_counter:02d}"] = output_file
        used.add(output_file)
        frame_counter += 1

    # Final safety net.
    if not frame_outputs and output_files:
        frame_outputs["frame00"] = sorted(output_files)[0]

    return frame_outputs


def segment_patient(
    input_dir: Path,
    patient_id: str,
    output_prefix: str,
    model_dir: Path,
    folds: str,
    model_tag: str,
    dataset_id: str,
    configuration: str,
    save_probabilities: bool = False,
) -> dict:
    """Run atrial nnUNet inference and write standardized SORAT outputs."""
    temp_output = Path(f"temp_atrial_nnunet_{patient_id}")
    temp_output.mkdir(parents=True, exist_ok=True)

    # Load frame manifest from preprocessed directory.  When missing (e.g.
    # stale cache), default to an empty frame list so map_outputs_to_frames
    # derives tags from the output filenames instead of forcing ED/ES (which
    # would silently drop the single LGE output for an absent ES frame).
    manifest_path = input_dir / f"{patient_id}_manifest.json"
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
    else:
        manifest = {"frames": []}

    manifest_frames = manifest.get("frames", [])
    expected_tags = [f["tag"] for f in manifest_frames]
    idx_by_tag = {f["tag"]: f["idx"] for f in manifest_frames}

    try:
        run_nnunet_predict(
            input_dir=input_dir,
            output_dir=temp_output,
            model_dir=model_dir,
            dataset_id=dataset_id,
            configuration=configuration,
            folds=folds,
            save_probabilities=save_probabilities,
        )

        pred_files = list(temp_output.glob("*.nii.gz"))
        frame_outputs = map_outputs_to_frames(pred_files, expected_tags)

        safe_tag = _sanitize_tag(model_tag)

        # Preserve a stable output order (ED, ES, then frameNN ascending).
        def _sort_key(tag):
            if tag == "ED":
                return (0, 0)
            if tag == "ES":
                return (0, 1)
            m = re.match(r"frame(\d+)", tag)
            return (1, int(m.group(1)) if m else 9999)

        ordered_tags = sorted(frame_outputs.keys(), key=_sort_key)
        saved_tags = []
        for i, tag in enumerate(ordered_tags):
            output_file = frame_outputs[tag]
            final_path = Path(f"{output_prefix}_{tag}_{safe_tag}.nii.gz")
            shutil.copy(output_file, final_path)
            saved_tags.append((tag, idx_by_tag.get(tag, i)))

        # Write segment manifest
        segment_manifest = {
            "patient_id": patient_id,
            "has_info_cfg": manifest.get("has_info_cfg", False),
            "num_frames": manifest.get("num_frames", len(saved_tags)),
            "frames": [{"tag": tag, "idx": idx} for tag, idx in saved_tags],
        }
        write_manifest(segment_manifest, f"{output_prefix}_{safe_tag}_manifest.json")

        results = {
            "patient_id": patient_id,
            "architecture": "atrial_nnunet",
            "model_tag": safe_tag,
            "dataset_id": dataset_id,
            "configuration": configuration,
            "folds": folds,
            "frame_tags": [tag for tag, _ in saved_tags],
            "frame_count": len(saved_tags),
        }

        with open(f"{output_prefix}_{safe_tag}_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

        return results
    finally:
        if temp_output.exists():
            shutil.rmtree(temp_output)


def main():
    parser = argparse.ArgumentParser(description="Atrial nnUNet segmentation for SORAT pipeline")
    parser.add_argument("--input_dir", required=True, help="Directory with preprocessed data")
    parser.add_argument("--patient_id", required=True, help="Patient identifier")
    parser.add_argument("--output_prefix", required=True, help="Output file prefix")
    parser.add_argument("--model_dir", required=True, help="Directory containing atrial nnUNet model")
    parser.add_argument("--folds", default="all", help="Comma-separated folds or 'all'")
    parser.add_argument("--model_tag", required=True, help="Model tag for output naming")
    parser.add_argument("--dataset_id", default="Dataset001_LGE", help="nnUNet dataset id")
    parser.add_argument("--configuration", default="2d", help="nnUNet configuration")
    parser.add_argument("--save_probabilities", action="store_true", help="Save probability maps")

    args = parser.parse_args()

    results = segment_patient(
        input_dir=Path(args.input_dir),
        patient_id=args.patient_id,
        output_prefix=args.output_prefix,
        model_dir=Path(args.model_dir),
        folds=args.folds,
        model_tag=args.model_tag,
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        save_probabilities=args.save_probabilities,
    )

    print(f"Segmentation complete for {args.patient_id}")
    print(f"Output frames: {results.get('frame_tags', [])}")


if __name__ == "__main__":
    main()
