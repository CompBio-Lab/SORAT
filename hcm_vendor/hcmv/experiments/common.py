"""Loading runs from the result store and formatting metric tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..runner import oof_matrix, runs_root
from ..stats import METRICS

METRIC_NAMES = {"auc": "AUC", "balanced_accuracy": "Balanced accuracy", "sensitivity": "Sensitivity",
                "specificity": "Specificity", "brier": "Brier score"}


def load_run(directory) -> dict:
    """Predictions, metrics, hyperparameters and manifest of one completed run (or None)."""
    directory = Path(directory)
    if not (directory / "manifest.json").exists():
        return None
    run = {"dir": directory, "predictions": pd.read_parquet(directory / "predictions.parquet")}
    for name in ("metrics", "hyperparams", "manifest"):
        run[name] = json.loads((directory / f"{name}.json").read_text())
    return run


def load_runs(config: dict, experiment: str) -> dict:
    """``{(unit, family_set, model): run}`` for every completed run of an experiment."""
    feature_config = config["experiments"]["feature_config"]
    root = runs_root(config) / experiment
    runs = {}
    for manifest in sorted(root.glob(f"*/*/*/{feature_config}/manifest.json")):
        unit, family_set, model = manifest.parent.relative_to(root).parts[:3]
        runs[(unit, family_set, model)] = load_run(manifest.parent)
    return runs


def analysis_dir(config: dict, experiment: str) -> Path:
    path = runs_root(config) / experiment / "analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def oof(run: dict):
    """(subject_ids, y, vendor, P) for a nested-CV run; P is (subjects, repeats)."""
    return oof_matrix(run["predictions"])


def ci_cell(values: dict, digits: int = 3) -> str:
    return f"{values['estimate']:.{digits}f} [{values['ci_low']:.{digits}f}, {values['ci_high']:.{digits}f}]"


def metric_rows(block: dict, **keys) -> dict:
    """Flatten one metrics block into estimate / CI columns plus a formatted cell per metric."""
    row = dict(keys)
    for metric in METRICS:
        if metric not in block:
            continue
        row[metric] = block[metric]["estimate"]
        row[f"{metric}_ci_low"] = block[metric]["ci_low"]
        row[f"{metric}_ci_high"] = block[metric]["ci_high"]
    return row


def formatted(table: pd.DataFrame, index_columns, digits: int = 3) -> pd.DataFrame:
    """Display version: one 'estimate [low, high]' cell per metric."""
    out = table[list(index_columns)].copy()
    for metric in METRICS:
        if metric in table:
            out[METRIC_NAMES[metric]] = [
                "" if np.isnan(e) else f"{e:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
                for e, lo, hi in zip(table[metric], table[f"{metric}_ci_low"], table[f"{metric}_ci_high"])]
    return out


def format_p(p: float, config: dict) -> str:
    """Bootstrap p-values cannot resolve below 1 / resamples, so report them as '< 1/B'."""
    floor = 1 / config["bootstrap_resamples"]
    return f"< {floor:.3g}" if p < floor else f"{p:.3f}"
