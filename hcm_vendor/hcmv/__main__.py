"""Command-line entry point: ``python -m hcmv <command> [--config ...] [--set key=value ...]``.

Commands are registered in ``COMMANDS`` as tickets land; each receives the loaded
config and the parsed arguments.
"""

import argparse
import json
import sys

from . import __version__
from .config import config_hash, load_config, output_dir
from .manifest import write_manifest


def cmd_show_config(config: dict, args: argparse.Namespace) -> int:
    print(json.dumps({k: v for k, v in config.items()}, indent=2, default=str))
    print(f"config_hash: {config_hash(config)}")
    return 0


def cmd_manifest(config: dict, args: argparse.Namespace) -> int:
    path = write_manifest(output_dir(config, "manifests"), config, "manifest")
    print(path)
    return 0


def cmd_cohort(config: dict, args: argparse.Namespace) -> int:
    from .cohort import write_cohort

    cohort = write_cohort(config)
    write_manifest(output_dir(config, "tables"), config, "cohort")
    print(cohort.groupby(["dataset", "role", "vendor", "disease"]).size().to_string())
    print(f"{len(cohort)} subjects -> {output_dir(config, 'tables') / 'cohort.parquet'}")
    return 0


def cmd_make_samplesheets(config: dict, args: argparse.Namespace) -> int:
    from .cohort import write_cohort, write_samplesheets

    written = write_samplesheets(config, write_cohort(config))
    write_manifest(output_dir(config, "inputs"), config, "make-samplesheets")
    for dataset, path in written.items():
        print(f"{dataset}: {path}")
    return 0


def cmd_validate_gt(config: dict, args: argparse.Namespace) -> int:
    from .qc import validate_gt

    pred_root = config.get("validation", {}).get("pred_root")
    print(validate_gt(config, pred_root=pred_root))
    write_manifest(output_dir(config, "qc", "t12_validation"), config, "validate-gt")
    return 0


def cmd_check_features(config: dict, args: argparse.Namespace) -> int:
    from .qc import check_features

    checks = config.get("checks", {})
    report = check_features(
        config,
        root=checks.get("root", "results_hcm_vendor/features/nnformer"),
        source=checks.get("source", config["segmentation_model_tag"]),
    )
    print(report.to_string(index=False))
    return 0 if report["ok"].all() else 1


def cmd_feature_tables(config: dict, args: argparse.Namespace) -> int:
    import pandas as pd

    from .features import write_feature_tables

    cohort = pd.read_parquet(output_dir(config, "tables") / "cohort.parquet")
    sources = {
        config["segmentation_model_tag"]: "results_hcm_vendor/features/nnformer",
        "gt": "results_hcm_vendor/features/gt",
    }
    for source, root in sources.items():
        for (dataset, cfg), path in write_feature_tables(config, cohort, source, root).items():
            table = pd.read_parquet(path)
            print(f"{source} {dataset} {cfg}: {len(table)} subjects -> {path}")
    write_manifest(output_dir(config, "tables"), config, "feature-tables")
    return 0


def cmd_qc_report(config: dict, args: argparse.Namespace) -> int:
    from .explore import qc_report

    print(qc_report(config))
    write_manifest(output_dir(config, "qc", "t22"), config, "qc-report")
    return 0


def radiomics_cli_flags(options: dict) -> list:
    """Nextflow --feature_extraction.radiomics.* flags for one named feature config."""
    flags = []
    for key, value in (options or {}).items():
        if value is None or value is False:
            continue
        value = "true" if value is True else value
        flags.append(f"--feature_extraction.radiomics.{key} {value}")
    return flags


def cmd_radiomics_flags(config: dict, args: argparse.Namespace) -> int:
    name = config.get("feature_config", "norm")
    print(" ".join(radiomics_cli_flags(config["feature_configs"][name])))
    return 0


COMMANDS = {
    "show-config": (cmd_show_config, "Print the resolved study config and its hash"),
    "manifest": (cmd_manifest, "Write a run manifest (git, config hash, packages)"),
    "cohort": (cmd_cohort, "Build the study cohort table (T20)"),
    "make-samplesheets": (cmd_make_samplesheets, "Write SORAT samplesheets for the study cohort (T13)"),
    "check-features": (cmd_check_features, "T14 checks; --set checks.root=... checks.source=gt for GT"),
    "radiomics-flags": (cmd_radiomics_flags, "Print Nextflow radiomics flags; --set feature_config=<name>"),
    "feature-tables": (cmd_feature_tables, "Build nnFormer and GT feature tables (T21)"),
    "qc-report": (cmd_qc_report, "T22 data QC + exploratory report -> results_hcm_vendor/qc/t22"),
    "validate-gt": (cmd_validate_gt, "T12 report; --set validation.pred_root=<dir> adds nnFormer-vs-GT"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hcmv", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"hcmv {__version__}")
    parser.add_argument("--config", default=None, help="Study YAML (default: configs/study.yaml)")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config value with a dotted key, e.g. --set cv.outer_repeats=1",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        subparsers.add_parser(name, help=help_text)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.overrides)
    handler, _ = COMMANDS[args.command]
    return handler(config, args)


if __name__ == "__main__":
    sys.exit(main())
