#!/usr/bin/env python3
"""
Atrial nnUNet Segmentation Script for CASC Pipeline.

Runs atrial segmentation inference using nnUNetv2 model.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


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


def map_outputs_to_ed_es(output_files: list[Path]) -> tuple[Path, Path]:
    """Map nnUNet output filenames back to ED/ES outputs."""
    ed_output = None
    es_output = None

    for output_file in sorted(output_files):
        name = output_file.name.lower()
        if "_ed" in name:
            ed_output = output_file
        elif "_es" in name:
            es_output = output_file

    if not ed_output and output_files:
        ed_output = sorted(output_files)[0]
    if not es_output and len(output_files) > 1:
        es_output = sorted(output_files)[1]
    if not es_output and ed_output:
        es_output = ed_output

    if not ed_output or not es_output:
        raise RuntimeError("Could not determine ED/ES outputs from nnUNet predictions")

    return ed_output, es_output


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
    """Run atrial nnUNet inference and write standardized CASC outputs."""
    temp_output = Path(f"temp_atrial_nnunet_{patient_id}")
    temp_output.mkdir(parents=True, exist_ok=True)

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
        ed_file, es_file = map_outputs_to_ed_es(pred_files)

        safe_tag = _sanitize_tag(model_tag)
        final_ed = Path(f"{output_prefix}_ED_{safe_tag}.nii.gz")
        final_es = Path(f"{output_prefix}_ES_{safe_tag}.nii.gz")

        shutil.copy(ed_file, final_ed)
        shutil.copy(es_file, final_es)

        results = {
            "patient_id": patient_id,
            "architecture": "atrial_nnunet",
            "model_tag": safe_tag,
            "dataset_id": dataset_id,
            "configuration": configuration,
            "folds": folds,
            "ed_output": str(final_ed),
            "es_output": str(final_es),
        }

        with open(f"{output_prefix}_{safe_tag}_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

        return results
    finally:
        if temp_output.exists():
            shutil.rmtree(temp_output)


def main():
    parser = argparse.ArgumentParser(description="Atrial nnUNet segmentation for CASC pipeline")
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
    print(f"ED output: {results['ed_output']}")
    print(f"ES output: {results['es_output']}")


if __name__ == "__main__":
    main()
