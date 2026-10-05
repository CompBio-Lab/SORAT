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


def cmd_run_experiment(config: dict, args: argparse.Namespace) -> int:
    from .models import MODEL_NAMES
    from .runner import run_experiment, smoke_config

    if args.smoke:
        config = smoke_config(config)
    models = args.models.split(",") if args.models else list(MODEL_NAMES)
    family_sets = args.family_sets.split(",") if args.family_sets else None
    units = args.units.split(",") if args.units else None
    summary = run_experiment(config, args.experiment, models, family_sets, units, n_jobs=args.n_jobs,
                             force=args.force)
    if len(summary):
        columns = ["unit", "family_set", "model", "n", "auc", "auc_ci_low", "auc_ci_high", "wall_s"]
        print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def add_run_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", required=True, help="Experiment key in study.yaml, e.g. E1 or E2")
    parser.add_argument("--models", default=None, help="Comma-separated models (default: all five)")
    parser.add_argument("--family-sets", default=None, help="Comma-separated family sets (default: config)")
    parser.add_argument("--units", default=None, help="Comma-separated cohorts or directions (default: all)")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel workers (default: SLURM_CPUS_PER_TASK or 1)")
    parser.add_argument("--smoke", action="store_true", help="Tiny CV and grids, written to runs-smoke/")
    parser.add_argument("--force", action="store_true", help="Rerun even if a matching result exists")


def _smoke(config: dict, args: argparse.Namespace) -> dict:
    from .runner import smoke_config

    return smoke_config(config) if getattr(args, "smoke", False) else config


def cmd_e1_report(config: dict, args: argparse.Namespace) -> int:
    from .experiments.e1 import e1_report
    from .runner import runs_root

    config = _smoke(config, args)
    print(e1_report(config))
    write_manifest(runs_root(config) / "E1" / "analysis", config, "e1-report")
    return 0


def cmd_e2_report(config: dict, args: argparse.Namespace) -> int:
    from .experiments.e2 import e2_report
    from .runner import runs_root

    config = _smoke(config, args)
    print(e2_report(config))
    write_manifest(runs_root(config) / "E2" / "analysis", config, "e2-report")
    return 0


def cmd_e4_report(config: dict, args: argparse.Namespace) -> int:
    from .experiments.e4 import e4_report
    from .runner import runs_root

    config = _smoke(config, args)
    print(e4_report(config))
    write_manifest(runs_root(config) / "E4" / "analysis", config, "e4-report")
    return 0


def cmd_e5_report(config: dict, args: argparse.Namespace) -> int:
    from .experiments.e5 import e5_report, ge_report
    from .runner import runs_root

    config = _smoke(config, args)
    print(e5_report(config))
    print(ge_report(config))
    for name in ("E5", "GE"):
        write_manifest(runs_root(config) / name / "analysis", config, "e5-report")
    return 0


def cmd_texture_check(config: dict, args: argparse.Namespace) -> int:
    from .texture_check import texture_check

    print(texture_check(config))
    write_manifest(output_dir(config, "qc", "texture_check"), config, "texture-check")
    return 0


def cmd_shap(config: dict, args: argparse.Namespace) -> int:
    from .runner import default_n_jobs, smoke_config
    from .shap_analysis import compute, shap_report

    if args.smoke:
        config = smoke_config(config)
    compute(config, n_jobs=args.n_jobs or default_n_jobs(), force=args.force,
            nsamples=200 if args.smoke else None, max_seeds=1 if args.smoke else None)
    print(shap_report(config, n_boot=100 if args.smoke else None))
    write_manifest(output_dir(config, "runs-smoke" if args.smoke else "runs", "SHAP", "analysis"), config, "shap")
    return 0


def cmd_summary(config: dict, args: argparse.Namespace) -> int:
    from .experiments.summary import summary_report

    print(summary_report(config))
    write_manifest(output_dir(config, "tables"), config, "summary")
    return 0


def cmd_e3_probe(config: dict, args: argparse.Namespace) -> int:
    from .experiments.e3 import e3_report, run_e3, smoke_e3
    from .runner import default_n_jobs, runs_root

    config = _smoke(config, args)
    if args.smoke:
        config = smoke_e3(config)
    table = run_e3(config, n_jobs=args.n_jobs or default_n_jobs(), force=args.force)
    print(e3_report(config, table))
    write_manifest(runs_root(config) / "E3" / "analysis", config, "e3-probe")
    return 0


def add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke", action="store_true", help="Read the smoke result store (runs-smoke/)")


def add_e3_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke", action="store_true", help="Tiny CV and 20 permutations, into runs-smoke/")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel workers (default: SLURM_CPUS_PER_TASK or 1)")
    parser.add_argument("--force", action="store_true", help="Recompute even if a matching result exists")


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
    "run-experiment": (cmd_run_experiment, "Nested CV or transfer runs into the result store (T34)"),
    "e1-report": (cmd_e1_report, "E1 tables, ROC curves, model comparisons, grid edge check (T40)"),
    "e2-report": (cmd_e2_report, "E2 transfer metrics, generalization gap, ROC/calibration overlays (T41)"),
    "e4-report": (cmd_e4_report, "E4 family-set ablation of the cross-vendor gap, raw vs normalized texture (T43)"),
    "e5-report": (cmd_e5_report, "E5 external ACDC test and GE specificity check (T44, T45)"),
    "texture-check": (cmd_texture_check, "Raw vs normalized texture: vendor shift, HCM effect agreement, reliance"),
    "shap": (cmd_shap, "SHAP of the E2 models and Siemens-vs-Philips stability (T50)"),
    "summary": (cmd_summary, "Consolidated results table and hypothesis verdicts -> tables/summary.md (T51)"),
    "e3-probe": (cmd_e3_probe, "E3 vendor probe on NOR with permutation test, plus its report (T42)"),
}
COMMAND_ARGS = {"run-experiment": add_run_experiment_args, "e1-report": add_report_args,
                "e2-report": add_report_args, "e3-probe": add_e3_args,
                "e4-report": add_report_args, "shap": add_e3_args, "e5-report": add_report_args}


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
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("--set", dest="sub_overrides", action="append", default=[], metavar="KEY=VALUE",
                               help="Same as the top-level --set (allowed after the command)")
        if name in COMMAND_ARGS:
            COMMAND_ARGS[name](subparser)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.overrides + args.sub_overrides)
    handler, _ = COMMANDS[args.command]
    return handler(config, args)


if __name__ == "__main__":
    sys.exit(main())
