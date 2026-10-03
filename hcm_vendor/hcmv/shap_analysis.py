"""SHAP attributions of the E2 models and their stability across training vendors (T50).

For each model, the Siemens-trained and the Philips-trained E2 pipelines (feature set
``all``, primary feature config) are explained on the **same 114 pooled subjects**
(D9), in each pipeline's post-preprocessing feature space:

* RF and XGB: exact TreeSHAP (``shap.TreeExplainer``); RF attributions are on the
  probability scale, XGB on the log-odds scale.
* LR-EN, SVM and MLP: KernelSHAP on ``predict_proba[:, 1]`` with background
  ``shap.kmeans(training-vendor data, 20)`` and ``nsamples = 2 d + 2048``.
  (LinearExplainer would be exact for LR-EN; the proposal specifies KernelSHAP.)
* Seeds: RF, XGB and the MLP were refitted with 5 seeds in E2; attributions are
  averaged over the seeds.

Because the correlation filter can keep different representatives of a correlated
group in each model, importances (mean |φ|) are summed within **correlation
clusters** (average-linkage clustering of the pooled features on 1 − |r|, cut at
|r| > 0.95). A feature a model did not keep counts as 0.

Stability between the Siemens- and Philips-trained model: Spearman ρ of cluster
importances (95% CI from a bootstrap over the explained subjects), top-10 Jaccard
overlap, and each family's share of total |φ|.

Per-seed attributions are cached in ``runs/E2/<direction>/<set>/<model>/<cfg>/shap/``;
the analysis goes to ``runs/SHAP/analysis/``.
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from . import figures
from .config import output_dir, repo_path
from .features import feature_family
from .preprocessing import select_features
from .qc import md_table
from .runner import runs_root

DIRECTIONS = {"siemens_to_philips": "Siemens", "philips_to_siemens": "Philips"}  # direction -> training vendor
TREE_MODELS = ("rf", "xgb")


def load_pool(config: dict) -> pd.DataFrame:
    cfg = config["experiments"]["feature_config"]
    source = config["experiments"].get("source") or config["segmentation_model_tag"]
    table = pd.read_parquet(repo_path(config, config["paths"]["output_root"]) / "tables" /
                            f"features_mms2_{source.replace('__', '-')}_{cfg}.parquet")
    return table[table["role"] == "train_pool"]


def run_dir(config: dict, direction: str, family_set: str, model: str) -> Path:
    return runs_root(config) / "E2" / direction / family_set / model / config["experiments"]["feature_config"]


def explain_seed(pipeline_path: str, model: str, X_explain: pd.DataFrame, X_background: pd.DataFrame,
                 n_background: int = 20, nsamples: int = None, seed: int = 0) -> dict:
    """SHAP values of one fitted pipeline: (n_subjects, n_kept_features) on the transformed data."""
    import joblib
    import shap

    pipe = joblib.load(pipeline_path)
    pre, est = pipe[:-1], pipe[-1]
    Z = pre.transform(X_explain)
    names = list(pre.get_feature_names_out())
    if model in TREE_MODELS:
        values = shap.TreeExplainer(est).shap_values(Z)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
    else:
        background = shap.kmeans(pre.transform(X_background).to_numpy(), n_background)
        f = lambda data: est.predict_proba(pd.DataFrame(data, columns=names))[:, 1]  # noqa: E731
        explainer = shap.KernelExplainer(f, background)
        np.random.seed(seed)
        values = explainer.shap_values(Z.to_numpy(), nsamples=nsamples or 2 * len(names) + 2048, silent=True)
        values = np.asarray(values)
    return {"values": values, "names": names, "Z": Z.to_numpy()}


def compute(config: dict, family_set: str = "all", models=None, n_jobs: int = 1, force: bool = False,
            nsamples: int = None, max_seeds: int = None, log=print) -> None:
    """Compute and cache per-seed SHAP values for every direction x model."""
    pool = load_pool(config)
    tasks = []
    for direction, vendor in DIRECTIONS.items():
        for model in models or figures.MODEL_ORDER:
            directory = run_dir(config, direction, family_set, model)
            if not (directory / "manifest.json").exists():
                log(f"missing E2 run: {directory}")
                continue
            features = json.loads((directory / "hyperparams.json").read_text())["features"]
            paths = sorted(glob.glob(str(directory / "models" / "seed_*.joblib")))[:max_seeds]
            for path in paths:
                out = directory / "shap" / (Path(path).stem + ".npz")
                if out.exists() and not force:
                    continue
                tasks.append((path, model, out, features, vendor))
    log(f"{len(tasks)} SHAP tasks")

    def work(path, model, out, features, vendor):
        result = explain_seed(path, model, pool[features], pool.loc[pool["vendor"] == vendor, features],
                              nsamples=nsamples, seed=config["seed"])
        out.parent.mkdir(exist_ok=True)
        np.savez(out, values=result["values"], names=np.array(result["names"], dtype=object),
                 Z=result["Z"], subjects=pool.index.to_numpy().astype(str), allow_pickle=True)
        return str(out)

    for done in Parallel(n_jobs=n_jobs)(delayed(work)(*t) for t in tasks):
        log(f"wrote {done}")


def load_shap(config: dict, direction: str, model: str, family_set: str = "all"):
    """Seed-averaged (values DataFrame subjects x features, transformed data DataFrame)."""
    files = sorted((run_dir(config, direction, family_set, model) / "shap").glob("seed_*.npz"))
    if not files:
        return None
    frames, data = [], None
    for f in files:
        npz = np.load(f, allow_pickle=True)
        names, subjects = list(npz["names"]), list(npz["subjects"])
        frames.append(pd.DataFrame(npz["values"], index=subjects, columns=names))
        data = pd.DataFrame(npz["Z"], index=subjects, columns=names)
    return sum(frames) / len(frames), data


def correlation_clusters(pool: pd.DataFrame, family_set: str = "all", threshold: float = 0.95) -> pd.Series:
    """Feature -> cluster label (named after its highest-priority member)."""
    columns = select_features(pool, family_set)
    X = pool[columns].astype(float)
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]
    corr = np.nan_to_num(np.abs(np.corrcoef(X.to_numpy(), rowvar=False)))
    distance = np.clip(1 - corr, 0, None)
    np.fill_diagonal(distance, 0)
    labels = fcluster(linkage(squareform(distance, checks=False), "average"), t=1 - threshold,
                      criterion="distance")
    clusters = pd.Series(labels, index=X.columns)
    # Name each cluster after its first member in select_features order (clinical first, mass leading).
    first = {}
    for c in X.columns:
        first.setdefault(clusters[c], c)
    return clusters.map(first)


def cluster_importance(values: pd.DataFrame, clusters: pd.Series, rows=None) -> pd.Series:
    v = values if rows is None else values.iloc[rows]
    mean_abs = v.abs().mean()
    return mean_abs.groupby(clusters.reindex(mean_abs.index)).sum().reindex(clusters.unique(), fill_value=0.0)


def stability(si: pd.DataFrame, ph: pd.DataFrame, clusters: pd.Series, n_boot: int, seed: int,
              top: int = 10) -> dict:
    subjects = si.index.intersection(ph.index)
    si, ph = si.loc[subjects], ph.loc[subjects]
    a, b = cluster_importance(si, clusters), cluster_importance(ph, clusters)
    rho = spearmanr(a, b).statistic
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(subjects), len(subjects))
        draws.append(spearmanr(cluster_importance(si, clusters, rows), cluster_importance(ph, clusters, rows)).statistic)
    top_a, top_b = set(a.nlargest(top).index), set(b.nlargest(top).index)
    return {"spearman_rho": float(rho), "rho_ci_low": float(np.nanpercentile(draws, 2.5)),
            "rho_ci_high": float(np.nanpercentile(draws, 97.5)),
            f"top{top}_jaccard": len(top_a & top_b) / len(top_a | top_b),
            f"top{top}_shared": len(top_a & top_b)}


def family_share(values: pd.DataFrame) -> pd.Series:
    mean_abs = values.abs().mean()
    return (mean_abs.groupby(mean_abs.index.map(feature_family)).sum() / mean_abs.sum()).reindex(
        ["clinical", "shape", "texture"], fill_value=0.0)


def shap_report(config: dict, family_set: str = "all", n_boot: int = None) -> str:
    pool = load_pool(config)
    clusters = correlation_clusters(pool, family_set)
    out = output_dir(config, "runs", "SHAP", "analysis") if not config.get("_smoke") else \
        runs_root(config) / "SHAP" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    n_boot = n_boot or config["bootstrap_resamples"]
    rows, shares, importances, panels = [], [], [], {}
    for model in figures.MODEL_ORDER:
        loaded = {d: load_shap(config, d, model, family_set) for d in DIRECTIONS}
        if any(v is None for v in loaded.values()):
            continue
        si, ph = loaded["siemens_to_philips"][0], loaded["philips_to_siemens"][0]
        rows.append({"model": model, **stability(si, ph, clusters, n_boot, config["seed"])})
        for direction, (values, data) in loaded.items():
            vendor = DIRECTIONS[direction]
            shares.append({"model": model, "trained_on": vendor, **family_share(values).to_dict()})
            imp = cluster_importance(values, clusters)
            importances.append(pd.DataFrame({"model": model, "trained_on": vendor, "cluster": imp.index,
                                             "importance": imp.to_numpy(),
                                             "rank": imp.rank(ascending=False, method="average").to_numpy()}))
            panels.setdefault(model, {})[vendor] = (values, data)
    if not rows:
        raise FileNotFoundError("No SHAP values cached; run `hcmv shap` first")
    stab, shares, importances = pd.DataFrame(rows), pd.DataFrame(shares), pd.concat(importances)
    stab.to_csv(out / "shap_stability.csv", index=False)
    shares.to_csv(out / "shap_family_share.csv", index=False)
    importances.to_csv(out / "shap_cluster_importance.csv", index=False)
    clusters.rename("cluster").to_csv(out / "correlation_clusters.csv")

    figures.plot_shap_rank_scatter(importances, stab, out / "shap_rank_scatter.png")
    figures.plot_shap_family_share(shares, out / "shap_family_share.png")
    for model, by_vendor in panels.items():
        figures.plot_shap_beeswarm(by_vendor, out / f"shap_beeswarm_{model}.png", model)

    lines = ["# SHAP stability across training vendors (T50)", "",
             f"E2 pipelines trained on Siemens or on Philips, feature set `{family_set}`, explained on the same "
             f"{len(pool)} pooled subjects. Importance = mean |φ| summed within |r| > 0.95 correlation clusters "
             f"({clusters.nunique()} clusters from {len(clusters)} features); ρ CI from {n_boot} subject bootstraps.",
             "", md_table(stab.round(3).set_index("model")), "",
             "## Share of total |φ| by feature family", "",
             md_table(shares.round(3).set_index(["model", "trained_on"])), "",
             "## Top 5 clusters per model and training vendor", ""]
    top = (importances[importances["importance"] > 0].sort_values("importance", ascending=False)
           .groupby(["model", "trained_on"], sort=False).head(5))
    for (model, vendor), g in top.groupby(["model", "trained_on"], sort=False):
        lines.append(f"- {figures.MODEL_NAMES[model]}, trained on {vendor}: " +
                     ", ".join(figures.feature_label(c) for c in g["cluster"]))
    (out / "summary.md").write_text("\n".join(lines))
    return "\n".join(lines)
