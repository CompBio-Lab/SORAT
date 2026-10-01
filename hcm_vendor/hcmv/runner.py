"""Experiment runner and result store (T34).

Two primitives:

* :func:`run_nested_cv` -- repeated stratified outer CV on one cohort; inside each
  outer training fold, ``GridSearchCV`` over the model grid (inner CV) refits the
  full pipeline, which then predicts the outer test fold. Gives out-of-fold (OOF)
  probabilities for every subject in every repeat.
* :func:`run_transfer` -- tune by inner CV on a training cohort, refit on all of
  it, predict a separate test cohort. Stochastic models are refitted with several
  seeds and their probabilities averaged; every fitted pipeline is saved for SHAP.

Each run writes one directory::

    <output_root>/runs/<experiment>/<unit>/<family_set>/<model>/<feature_config>/
        predictions.parquet   subject_id, [repeat, fold | seed-averaged], y, prob, vendor
        hyperparams.json      chosen grid point, inner-CV AUC, fit time per fold or seed
        metrics.json          estimates + bootstrap CIs (overall and per vendor)
        manifest.json         git SHA, packages, run hash -- written last
        models/seed_<s>.joblib            (transfer only)
        predictions_by_seed.parquet       (transfer only)

``unit`` is the cohort (E1: pooled / siemens / philips) or the direction (E2:
siemens_to_philips / philips_to_siemens). A run whose ``manifest.json`` holds the
same run hash is skipped, so an interrupted job can simply be resubmitted.
Smoke mode (:func:`smoke_config`) shrinks CV, grids and resamples and writes to
``runs-smoke/`` instead. ``experiments.runs_dir`` renames ``runs`` (e.g. for timing runs).
"""

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV

from .config import output_dir, repo_path
from .manifest import write_manifest
from .models import STOCHASTIC_MODELS, get_model
from .preprocessing import build_pipeline, select_features
from .splits import inner_cv, outer_splits
from .stats import METRICS, metric_value, summarize

PREDICTION_COLUMNS = ["subject_id", "repeat", "fold", "y", "prob", "vendor"]


# ----------------------------------------------------------------------------- config
def smoke_config(config: dict) -> dict:
    """Copy of ``config`` with tiny CV, grids, resamples and model budgets for quick checks."""
    smoke = config.get("smoke", {})
    out = copy.deepcopy(config)
    out["cv"] = {**out["cv"], **smoke.get("cv", {})}
    for key in ("bootstrap_resamples",):
        if key in smoke:
            out[key] = smoke[key]
    out["experiments"]["transfer_seeds"] = smoke.get("transfer_seeds", out["experiments"]["transfer_seeds"])
    for name, grid in out["models"].items():
        # Middle of each axis (edge values such as C=0.001 can give a constant model):
        # two values on the first axis, so there is still a choice, and one on the rest.
        out["models"][name] = {}
        for i, (axis, values) in enumerate(grid.items()):
            mid = len(values) // 2
            out["models"][name][axis] = values[max(mid - 1, 0): mid + 1] if i == 0 else values[mid: mid + 1]
    model_params = copy.deepcopy(out["experiments"].get("model_params", {}))
    for name, params in smoke.get("model_params", {}).items():
        model_params[name] = {**model_params.get(name, {}), **params}
    out["experiments"]["model_params"] = model_params
    out["_smoke"] = True
    return out


def runs_root(config: dict) -> Path:
    name = config["experiments"].get("runs_dir", "runs")
    return output_dir(config, f"{name}-smoke" if config.get("_smoke") else name)


def feature_table_path(config: dict, dataset: str = "mms2") -> Path:
    experiments = config["experiments"]
    source = experiments.get("source") or config["segmentation_model_tag"]
    name = f"features_{dataset}_{source.replace('__', '-')}_{experiments['feature_config']}.parquet"
    return repo_path(config, config["paths"]["output_root"]) / "tables" / name


def apply_filter(table: pd.DataFrame, cohort_filter: dict) -> pd.DataFrame:
    """Rows whose columns equal every ``{column: value}`` in the filter (lists mean 'any of')."""
    mask = np.ones(len(table), dtype=bool)
    for column, value in cohort_filter.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        mask &= table[column].isin(values).to_numpy()
    return table[mask]


def _hash(payload: dict, length: int = 16) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:length]


def run_spec(config: dict, kind: str, experiment: str, unit: str, filters: dict, family_set: str,
             model: str, seeds=None) -> dict:
    """Everything that determines a run's result; its hash keys the result store."""
    return {
        "kind": kind, "experiment": experiment, "unit": unit, "filters": filters,
        "family_set": family_set, "model": model,
        "feature_table": str(feature_table_path(config)),
        "feature_config": config["experiments"]["feature_config"],
        "seed": config["seed"], "seeds": seeds, "cv": config["cv"],
        "inner_scoring": config["inner_scoring"], "grid": config["models"][model],
        "model_params": config["experiments"].get("model_params", {}).get(model, {}),
        "bootstrap_resamples": config["bootstrap_resamples"], "smoke": bool(config.get("_smoke")),
    }


def run_dir(config: dict, spec: dict) -> Path:
    return (runs_root(config) / spec["experiment"] / spec["unit"] / spec["family_set"] / spec["model"]
            / spec["feature_config"])


def is_complete(directory: Path, spec_hash: str) -> bool:
    manifest = directory / "manifest.json"
    if not manifest.exists():
        return False
    try:
        return json.loads(manifest.read_text()).get("run_hash") == spec_hash
    except json.JSONDecodeError:
        return False


# ----------------------------------------------------------------------------- fitting
def _search(model: str, config: dict, seed: int, inner_seed: int) -> GridSearchCV:
    estimator, grid = get_model(model, config, seed=seed)
    estimator.set_params(**config["experiments"].get("model_params", {}).get(model, {}))
    return GridSearchCV(build_pipeline(estimator), grid, cv=inner_cv(config["cv"]["inner_splits"], inner_seed),
                        scoring=config["inner_scoring"], n_jobs=1, refit=True, error_score="raise")


def _fit_and_predict(X_train, y_train, X_test, model, config, seed, inner_seed):
    search = _search(model, config, seed, inner_seed)
    start = time.perf_counter()
    search.fit(X_train, y_train)
    fit_s = time.perf_counter() - start
    prob = search.predict_proba(X_test)[:, 1]
    info = {
        "best_params": {k.removeprefix("model__"): v for k, v in search.best_params_.items()},
        "inner_score": float(search.best_score_),
        "fit_s": round(fit_s, 2),
        "n_features_in": int(X_train.shape[1]),
        "n_features_kept": int(len(search.best_estimator_[:-1].get_feature_names_out())),
    }
    return prob, info, search.best_estimator_


def _nested_fold(X, y, split, model, config):
    seed = config["seed"]
    prob, info, _ = _fit_and_predict(X.iloc[split.train], y[split.train], X.iloc[split.test], model, config,
                                     seed=seed, inner_seed=seed + 100 * split.repeat + split.fold)
    return split, prob, info


def _transfer_seed(X_train, y_train, X_test, model, config, seed):
    prob, info, pipeline = _fit_and_predict(X_train, y_train, X_test, model, config, seed=seed, inner_seed=seed)
    return seed, prob, info, pipeline


# ----------------------------------------------------------------------------- metrics
def _metrics_block(y, P, config: dict, seed: int) -> dict:
    rows = summarize(y, P, n_boot=config["bootstrap_resamples"], seed=seed)
    return {row["metric"]: {k: row[k] for k in ("estimate", "ci_low", "ci_high")} for row in rows}


def oof_matrix(predictions: pd.DataFrame):
    """(subject_ids, y, vendor, P) with P of shape (n_subjects, n_repeats)."""
    wide = predictions.pivot(index="subject_id", columns="repeat", values="prob").sort_index()
    meta = predictions.drop_duplicates("subject_id").set_index("subject_id").loc[wide.index]
    return wide.index.to_numpy(), meta["y"].to_numpy(), meta["vendor"].to_numpy(), wide.to_numpy()


def nested_cv_metrics(predictions: pd.DataFrame, config: dict) -> dict:
    """Repeat-averaged metrics with subject-bootstrap CIs, overall and per vendor (reading (b))."""
    _, y, vendor, P = oof_matrix(predictions)
    seed = config["seed"]
    out = {
        "n": int(len(y)), "n_positive": int(y.sum()), "n_repeats": int(P.shape[1]),
        "overall": _metrics_block(y, P, config, seed),
        "per_repeat": {m: [metric_value(m, y, P[:, r]) for r in range(P.shape[1])] for m in METRICS},
        "by_vendor": {},
    }
    if len(np.unique(vendor)) > 1:
        for name in sorted(np.unique(vendor)):
            rows = vendor == name
            out["by_vendor"][name] = {"n": int(rows.sum()), **_metrics_block(y[rows], P[rows], config, seed)}
    return out


# ----------------------------------------------------------------------------- primitives
def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def run_nested_cv(table: pd.DataFrame, cohort_filter: dict, family_set: str, model: str, config: dict,
                  experiment: str = "E1", unit: str = "pooled", n_jobs: int = 1, force: bool = False) -> dict:
    """Nested CV on ``apply_filter(table, cohort_filter)``; returns a status dict."""
    spec = run_spec(config, "nested_cv", experiment, unit, cohort_filter, family_set, model)
    spec_hash = _hash(spec)
    directory = run_dir(config, spec)
    if not force and is_complete(directory, spec_hash):
        return {"status": "skipped", "dir": str(directory)}

    cohort = apply_filter(table, cohort_filter)
    columns = select_features(cohort, family_set)
    X, y = cohort[columns], cohort["y"].to_numpy().astype(int)
    cv = config["cv"]
    splits = list(outer_splits(cohort, cv["outer_splits"], cv["outer_repeats"], config["seed"],
                               stratify_by=("y", "vendor")))
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(_nested_fold)(X, y, s, model, config) for s in splits)
    wall_s = time.perf_counter() - start

    frames, folds = [], []
    for split, prob, info in results:
        test = cohort.iloc[split.test]
        frames.append(pd.DataFrame({
            "subject_id": test.index.to_numpy(), "repeat": split.repeat, "fold": split.fold,
            "y": y[split.test], "prob": prob, "vendor": test["vendor"].to_numpy(),
        }))
        folds.append({"repeat": split.repeat, "fold": split.fold, **info})
    predictions = pd.concat(frames, ignore_index=True)[PREDICTION_COLUMNS]

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").unlink(missing_ok=True)
    predictions.to_parquet(directory / "predictions.parquet", index=False)
    _write_json(directory / "hyperparams.json", {"folds": folds, "features": columns})
    _write_json(directory / "metrics.json", nested_cv_metrics(predictions, config))
    write_manifest(directory, config, f"run_nested_cv {experiment}/{unit}/{family_set}/{model}",
                   {"run_hash": spec_hash, "spec": spec, "wall_s": round(wall_s, 1), "n_jobs": n_jobs})
    return {"status": "done", "dir": str(directory), "wall_s": wall_s}


def run_transfer(table: pd.DataFrame, train_filter: dict, test_filter: dict, family_set: str, model: str,
                 config: dict, seeds=None, experiment: str = "E2", unit: str = "transfer", n_jobs: int = 1,
                 force: bool = False) -> dict:
    """Tune and refit on the training cohort, predict the test cohort; returns a status dict."""
    if seeds is None:
        n_seeds = config["experiments"]["transfer_seeds"] if model in STOCHASTIC_MODELS else 1
        seeds = [config["seed"] + i for i in range(n_seeds)]
    seeds = [int(s) for s in seeds]
    spec = run_spec(config, "transfer", experiment, unit, {"train": train_filter, "test": test_filter},
                    family_set, model, seeds=seeds)
    spec_hash = _hash(spec)
    directory = run_dir(config, spec)
    if not force and is_complete(directory, spec_hash):
        return {"status": "skipped", "dir": str(directory)}

    train, test = apply_filter(table, train_filter), apply_filter(table, test_filter)
    overlap = set(train.index) & set(test.index)
    if overlap:
        raise ValueError(f"{len(overlap)} subjects are in both train and test, e.g. {sorted(overlap)[:3]}")
    columns = select_features(train, family_set)
    y_train = train["y"].to_numpy().astype(int)
    y_test = test["y"].to_numpy().astype(int)
    start = time.perf_counter()
    results = Parallel(n_jobs=min(n_jobs, len(seeds)))(
        delayed(_transfer_seed)(train[columns], y_train, test[columns], model, config, s) for s in seeds)
    wall_s = time.perf_counter() - start

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").unlink(missing_ok=True)
    (directory / "models").mkdir(exist_ok=True)
    by_seed, fits = [], []
    for seed, prob, info, pipeline in results:
        by_seed.append(pd.DataFrame({"subject_id": test.index.to_numpy(), "seed": seed, "y": y_test,
                                     "prob": prob, "vendor": test["vendor"].to_numpy()}))
        fits.append({"seed": seed, **info})
        joblib.dump(pipeline, directory / "models" / f"seed_{seed}.joblib")
    by_seed = pd.concat(by_seed, ignore_index=True)
    predictions = (by_seed.groupby(["subject_id", "y", "vendor"], sort=False)["prob"].mean().reset_index()
                   [["subject_id", "y", "prob", "vendor"]])

    by_seed.to_parquet(directory / "predictions_by_seed.parquet", index=False)
    predictions.to_parquet(directory / "predictions.parquet", index=False)
    _write_json(directory / "hyperparams.json", {"seeds": fits, "features": columns})
    metrics = {"n_train": int(len(train)), "n": int(len(test)), "n_positive": int(y_test.sum()),
               "n_seeds": len(seeds),
               "overall": _metrics_block(predictions["y"].to_numpy(), predictions["prob"].to_numpy(),
                                         config, config["seed"])}
    _write_json(directory / "metrics.json", metrics)
    write_manifest(directory, config, f"run_transfer {experiment}/{unit}/{family_set}/{model}",
                   {"run_hash": spec_hash, "spec": spec, "wall_s": round(wall_s, 1), "n_jobs": n_jobs})
    return {"status": "done", "dir": str(directory), "wall_s": wall_s}


# ----------------------------------------------------------------------------- experiments
def default_n_jobs() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


def run_experiment(config: dict, experiment: str, models, family_sets=None, units=None, n_jobs: int = None,
                   force: bool = False, log=print) -> pd.DataFrame:
    """Run every unit x family set x model of an experiment in ``config['experiments']``."""
    spec = config["experiments"][experiment]
    family_sets = family_sets or config["experiments"]["family_sets"]
    n_jobs = n_jobs or default_n_jobs()
    table = pd.read_parquet(feature_table_path(config))
    unit_specs = spec["cohorts"] if spec["kind"] == "nested_cv" else spec["directions"]
    rows = []
    for unit, unit_spec in unit_specs.items():
        if units and unit not in units:
            continue
        for family_set in family_sets:
            for model in models:
                if spec["kind"] == "nested_cv":
                    result = run_nested_cv(table, unit_spec, family_set, model, config, experiment, unit,
                                           n_jobs=n_jobs, force=force)
                else:
                    result = run_transfer(table, unit_spec["train"], unit_spec["test"], family_set, model,
                                          config, experiment=experiment, unit=unit, n_jobs=n_jobs, force=force)
                wall = f" in {result['wall_s']:.0f} s" if "wall_s" in result else ""
                log(f"{experiment}/{unit}/{family_set}/{model}: {result['status']}{wall}")
                rows.append({"unit": unit, "family_set": family_set, "model": model, **result})
    summary = collect_results(runs_root(config) / experiment)
    if len(summary):
        summary.to_csv(runs_root(config) / experiment / "summary.csv", index=False)
    return summary


def collect_results(root: Path) -> pd.DataFrame:
    """One row per completed run under ``root``: overall metrics with CIs and run timing."""
    rows = []
    for manifest_path in sorted(Path(root).rglob("manifest.json")):
        directory = manifest_path.parent
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            continue
        manifest, metrics = json.loads(manifest_path.read_text()), json.loads(metrics_path.read_text())
        spec = manifest["spec"]
        row = {"experiment": spec["experiment"], "unit": spec["unit"], "family_set": spec["family_set"],
               "model": spec["model"], "feature_config": spec["feature_config"], "n": metrics["n"],
               "wall_s": manifest.get("wall_s")}
        for metric, values in metrics["overall"].items():
            row[metric] = values["estimate"]
            row[f"{metric}_ci_low"] = values["ci_low"]
            row[f"{metric}_ci_high"] = values["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows)
