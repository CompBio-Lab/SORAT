"""Metrics and resampling statistics (T33).

Predictions are handled as a matrix ``P`` of shape (n_subjects, n_repeats): one
column per outer-CV repeat (out-of-fold probabilities) or a single column for a
train/test transfer. A metric on ``P`` is the mean over repeats of the metric on
each column, and bootstrap resamples *subjects* (rows), stratified by class.
"""

import numpy as np
from scipy import stats as _stats

THRESHOLD = 0.5  # D7: fixed; models are trained class-balanced


def fast_auc(y, prob) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic (ties get average ranks)."""
    y = np.asarray(y).astype(int)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = _stats.rankdata(prob)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metric_value(metric: str, y, prob, threshold: float = THRESHOLD) -> float:
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    if metric == "auc":
        return fast_auc(y, prob)
    if metric == "brier":
        return float(np.mean((prob - y) ** 2))
    pred = prob >= threshold
    positives, negatives = y == 1, y == 0
    sensitivity = pred[positives].mean() if positives.any() else np.nan
    specificity = 1 - pred[negatives].mean() if negatives.any() else np.nan
    if metric == "sensitivity":
        return float(sensitivity)
    if metric == "specificity":
        return float(specificity)
    if metric == "balanced_accuracy":
        return float(np.nanmean([sensitivity, specificity]))
    raise ValueError(f"Unknown metric {metric!r}")


METRICS = ("auc", "balanced_accuracy", "sensitivity", "specificity", "brier")


def binary_metrics(y, prob, threshold: float = THRESHOLD) -> dict:
    return {m: metric_value(m, y, prob, threshold) for m in METRICS}


def _as_matrix(P) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    return P[:, None] if P.ndim == 1 else P


def repeated_metric(y, P, metric: str, rows=None) -> float:
    """Mean over repeat columns of ``metric`` on the (optionally resampled) rows."""
    y = np.asarray(y)
    P = _as_matrix(P)
    rows = np.arange(len(y)) if rows is None else rows
    values = [metric_value(metric, y[rows], P[rows, r]) for r in range(P.shape[1])]
    return float(np.nanmean(values))


def stratified_bootstrap_indices(y, n_boot: int, seed: int):
    """Yield row-index arrays resampled with replacement within each class."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in np.unique(y)]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])


def summarize(y, P, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05, metrics=METRICS) -> list:
    """Estimates and percentile CIs for several metrics from one bootstrap loop."""
    draws = {m: [] for m in metrics}
    for rows in stratified_bootstrap_indices(y, n_boot, seed):
        for m in metrics:
            draws[m].append(repeated_metric(y, P, m, rows))
    out = []
    for m in metrics:
        low, high = np.nanpercentile(draws[m], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out.append({"metric": m, "estimate": repeated_metric(y, P, m),
                    "ci_low": float(low), "ci_high": float(high)})
    return out


def bootstrap_ci(y, P, metric: str, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Point estimate and percentile CI for a single (repeated) metric."""
    return summarize(y, P, n_boot=n_boot, seed=seed, alpha=alpha, metrics=(metric,))[0]


def paired_bootstrap(y, P_a, P_b, metric: str = "auc", n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Difference metric(A) - metric(B) on the same subjects, with CI and two-sided p."""
    estimate = repeated_metric(y, P_a, metric) - repeated_metric(y, P_b, metric)
    diffs = np.array([
        repeated_metric(y, P_a, metric, rows) - repeated_metric(y, P_b, metric, rows)
        for rows in stratified_bootstrap_indices(y, n_boot, seed)
    ])
    diffs = diffs[~np.isnan(diffs)]
    low, high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"metric": metric, "difference": float(estimate), "ci_low": float(low),
            "ci_high": float(high), "p_value": float(p)}


def permutation_p_value(observed: float, null) -> float:
    """One-sided p = (k + 1) / (n + 1), k = null statistics >= observed."""
    null = np.asarray(null, dtype=float)
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def holm(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple:
    """Wilson score interval for a proportion (e.g. specificity on GE NOR)."""
    if n == 0:
        return (np.nan, np.nan)
    z = _stats.norm.ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))
