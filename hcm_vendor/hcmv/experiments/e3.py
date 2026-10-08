"""E3: can the scanner vendor be predicted from features of normal hearts?

Only NOR subjects are used, so disease cannot carry the signal. For each feature
probe (clinical, shape, texture normalized, texture raw, all) and each target
(3-class Siemens/Philips/GE, primary; binary Siemens vs Philips, secondary):

* Classifiers: multinomial L2 logistic regression with C tuned by inner CV
  (primary) and a random forest with fixed settings (secondary), both
  class-balanced and preceded by the usual leakage-safe preprocessing.
* Estimate: balanced accuracy of out-of-fold predictions, averaged over the
  outer repeats, with a class-stratified subject bootstrap CI.
* Permutation test: the vendor labels are shuffled ``permutations`` times and the
  first outer repeat's splits are rerun; p = (k + 1) / (n + 1), where the observed
  statistic is that same repeat's balanced accuracy. Holm adjustment across the
  five probes, within each target and classifier.

Results go to ``runs/E3/<target>/<classifier>/<probe>.json`` (resumable by hash)
and ``runs/E3/analysis/`` (``e3_probe.csv``, ``summary.md``, null-vs-observed figures).
"""

import copy
import hashlib
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from .. import figures
from ..config import repo_path
from ..preprocessing import build_pipeline, select_features
from ..qc import md_table
from ..runner import runs_root
from ..stats import holm, permutation_p_value
from .common import analysis_dir

PROBE_NAMES = {"clinical": "Clinical", "shape": "Shape", "texture_norm": "Texture (normalized)",
               "texture_raw": "Texture (raw)", "texture_ref": "Texture (blood-pool reference)",
               "all": "All features"}
TARGET_NAMES = {"three_vendor": "Siemens vs Philips vs GE", "siemens_vs_philips": "Siemens vs Philips"}
CLASSIFIER_NAMES = {"logreg": "multinomial logistic regression", "rf": "random forest"}


def smoke_e3(config: dict) -> dict:
    out = copy.deepcopy(config)
    out["e3"].update({"outer_repeats": 1, "outer_splits": 3, "inner_splits": 2, "logreg_C": [0.1, 1.0],
                      "rf": {**out["e3"]["rf"], "n_estimators": 50}})
    out["permutations"] = 20
    out["bootstrap_resamples"] = 100
    out["_smoke"] = True
    return out


def load_probe_data(config: dict, target: str, probe: str):
    """(X, y_vendor, subject_ids) for NOR subjects of the target vendors, columns of the probe."""
    e3 = config["e3"]
    feature_config, family_set = e3["probes"][probe]
    source = config["experiments"].get("source") or config["segmentation_model_tag"]
    path = repo_path(config, config["paths"]["output_root"]) / "tables" / \
        f"features_mms2_{source.replace('__', '-')}_{feature_config}.parquet"
    table = pd.read_parquet(path)
    table = table[(table["disease"] == e3["disease"]) & table["vendor"].isin(e3["targets"][target])]
    columns = select_features(table, family_set)
    return table[columns], table["vendor"].to_numpy(), table.index.to_numpy()


def _estimator(classifier: str, config: dict, seed: int, inner_seed: int):
    e3 = config["e3"]
    if classifier == "logreg":
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed)
        cv = StratifiedKFold(e3["inner_splits"], shuffle=True, random_state=inner_seed)
        return GridSearchCV(build_pipeline(model), {"model__C": list(e3["logreg_C"])}, cv=cv,
                            scoring="balanced_accuracy", n_jobs=1)
    if classifier == "rf":
        return build_pipeline(RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=seed,
                                                     **e3["rf"]))
    raise ValueError(f"Unknown classifier {classifier!r}")


def _outer(config: dict):
    e3 = config["e3"]
    return RepeatedStratifiedKFold(n_splits=e3["outer_splits"], n_repeats=e3["outer_repeats"],
                                   random_state=config["seed"])


def oof_predictions(X, y, classifier: str, config: dict, repeats=None) -> np.ndarray:
    """Out-of-fold labels, shape (n_subjects, n_repeats); ``repeats`` limits which repeats run."""
    k = config["e3"]["outer_splits"]
    splits = list(_outer(config).split(np.zeros(len(y)), y))
    n_repeats = len(splits) // k
    repeats = range(n_repeats) if repeats is None else repeats
    pred = np.empty((len(y), len(repeats)), dtype=object)
    for col, r in enumerate(repeats):
        for fold, (train, test) in enumerate(splits[r * k:(r + 1) * k]):
            model = _estimator(classifier, config, config["seed"], config["seed"] + 100 * r + fold)
            model.fit(X.iloc[train], y[train])
            pred[test, col] = model.predict(X.iloc[test])
    return pred


def repeated_balanced_accuracy(y, pred, rows=None) -> float:
    rows = np.arange(len(y)) if rows is None else rows
    return float(np.mean([balanced_accuracy_score(y[rows], pred[rows, r]) for r in range(pred.shape[1])]))


def bootstrap_ci(y, pred, n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in np.unique(y)]
    draws = [repeated_balanced_accuracy(y, pred, np.concatenate([rng.choice(g, len(g)) for g in groups]))
             for _ in range(n_boot)]
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def _null_draw(X, y, classifier, config, index):
    rng = np.random.default_rng([config["seed"], index])
    shuffled = rng.permutation(y)
    pred = oof_predictions(X, shuffled, classifier, config, repeats=[0])
    return balanced_accuracy_score(shuffled, pred[:, 0])


def _spec(config: dict, target: str, classifier: str, probe: str) -> dict:
    keys = ("disease", "outer_splits", "outer_repeats", "inner_splits", "logreg_C", "rf")
    return {"target": target, "classifier": classifier, "probe": probe,
            "probe_def": config["e3"]["probes"][probe], "vendors": config["e3"]["targets"][target],
            **{k: config["e3"][k] for k in keys}, "seed": config["seed"],
            "permutations": config["permutations"], "bootstrap_resamples": config["bootstrap_resamples"],
            "source": config["experiments"].get("source") or config["segmentation_model_tag"]}


def run_probe(config: dict, target: str, classifier: str, probe: str, n_jobs: int = 1, force: bool = False) -> dict:
    spec = _spec(config, target, classifier, probe)
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = runs_root(config) / "E3" / target / classifier / f"{probe}.json"
    if not force and path.exists() and json.loads(path.read_text()).get("run_hash") == spec_hash:
        return json.loads(path.read_text())

    X, y, ids = load_probe_data(config, target, probe)
    start = time.perf_counter()
    pred = oof_predictions(X, y, classifier, config)
    estimate = repeated_balanced_accuracy(y, pred)
    ci = bootstrap_ci(y, pred, config["bootstrap_resamples"], config["seed"])
    observed_first = balanced_accuracy_score(y, pred[:, 0])
    null = Parallel(n_jobs=n_jobs)(delayed(_null_draw)(X, y, classifier, config, i)
                                   for i in range(config["permutations"]))
    vendors, counts = np.unique(y, return_counts=True)
    result = {
        "run_hash": spec_hash, "spec": spec, "n": int(len(y)),
        "counts": dict(zip(vendors.tolist(), counts.tolist())), "n_features": int(X.shape[1]),
        "balanced_accuracy": estimate, "ci_low": ci[0], "ci_high": ci[1],
        "per_repeat": [balanced_accuracy_score(y, pred[:, r]) for r in range(pred.shape[1])],
        "chance": 1 / len(vendors), "observed_first_repeat": float(observed_first),
        "null": [float(v) for v in null], "p_value": permutation_p_value(observed_first, null),
        "wall_s": round(time.perf_counter() - start, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    return result


def run_e3(config: dict, n_jobs: int = 1, force: bool = False, log=print) -> pd.DataFrame:
    rows = []
    e3 = config["e3"]
    for target in e3["targets"]:
        for classifier in e3["classifiers"]:
            for probe in e3["probes"]:
                r = run_probe(config, target, classifier, probe, n_jobs=n_jobs, force=force)
                log(f"E3/{target}/{classifier}/{probe}: BA {r['balanced_accuracy']:.3f} "
                    f"p {r['p_value']:.4f} ({r['wall_s']} s)")
                rows.append({"target": target, "classifier": classifier, "probe": probe,
                             **{k: r[k] for k in ("n", "n_features", "balanced_accuracy", "ci_low", "ci_high",
                                                  "chance", "observed_first_repeat", "p_value")},
                             "null_mean": float(np.mean(r["null"])), "null": r["null"]})
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    for _, index in table.groupby(["target", "classifier"]).groups.items():
        table.loc[index, "p_holm"] = holm(table.loc[index, "p_value"].to_numpy())
    return table


def e3_report(config: dict, table: pd.DataFrame) -> str:
    out = analysis_dir(config, "E3")
    table.drop(columns=["null"]).to_csv(out / "e3_probe.csv", index=False)
    lines = ["# E3: vendor probe on normal subjects", "",
             f"Balanced accuracy of out-of-fold predictions ({config['e3']['outer_repeats']}×"
             f"{config['e3']['outer_splits']} CV) with a {config['bootstrap_resamples']}-resample bootstrap CI; "
             f"permutation p from {config['permutations']} vendor-label shuffles of the first repeat; Holm across "
             "the five probes within each target and classifier.", ""]
    for (target, classifier), group in table.groupby(["target", "classifier"], sort=False):
        show = pd.DataFrame({
            "probe": group["probe"].map(PROBE_NAMES), "features": group["n_features"],
            "balanced accuracy [95% CI]": [f"{e:.3f} [{lo:.3f}, {hi:.3f}]" for e, lo, hi in
                                           zip(group.balanced_accuracy, group.ci_low, group.ci_high)],
            "chance": group["chance"].map("{:.3f}".format), "null mean": group["null_mean"].map("{:.3f}".format),
            "p": group["p_value"].map("{:.4f}".format), "p (Holm)": group["p_holm"].map("{:.4f}".format)})
        n = group["n"].iloc[0]
        lines += [f"## {TARGET_NAMES[target]}, {CLASSIFIER_NAMES[classifier]} (n = {n} NOR)", "",
                  md_table(show.set_index("probe")), ""]
        results = [{"family": r.probe, "label": PROBE_NAMES[r.probe], "observed": r.observed_first_repeat,
                    "null": np.asarray(r.null), "chance": r.chance, "p": r.p_value, "p_holm": r.p_holm}
                   for r in group.itertuples()]
        figures.plot_probe_null(results, out / f"e3_null_{target}_{classifier}.png",
                                f"Vendor prediction from normal hearts: {TARGET_NAMES[target]}, "
                                f"{CLASSIFIER_NAMES[classifier]} (n = {n})")
    (out / "summary.md").write_text("\n".join(lines))
    return "\n".join(lines)
