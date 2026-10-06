"""ComBat harmonization of features across scanner vendors (Johnson, Li & Rabinovic 2007).

Location-scale empirical Bayes adjustment, without covariates: every feature is
standardized, a per-vendor shift (γ) and scale (δ²) are estimated and shrunk towards
their across-feature priors, then removed. Used transductively in the cross-vendor
setting: it is fitted on the features of the training and the test vendor together,
never on the disease labels.

Caveat: without a disease covariate, ComBat also removes the part of a vendor's mean
that comes from its disease mix (Siemens 58% HCM, Philips 44% in the train pool). A
covariate cannot be used here, because the test vendor's labels are unknown.
"""

import numpy as np
import pandas as pd


def _aprior(delta):
    m, s2 = np.mean(delta), np.var(delta, ddof=1)
    return (2 * s2 + m ** 2) / s2


def _bprior(delta):
    m, s2 = np.mean(delta), np.var(delta, ddof=1)
    return (m * s2 + m ** 3) / s2


def combat(X: pd.DataFrame, batch, tol: float = 1e-4, max_iter: int = 1000) -> pd.DataFrame:
    """Harmonized copy of ``X`` (subjects x features) across the levels of ``batch``.

    Columns with zero variance (overall or within a batch) are returned unchanged.
    Missing values are not allowed.
    """
    if X.isna().any().any():
        raise ValueError("combat needs complete data; impute first")
    batch = np.asarray(batch)
    levels = list(pd.unique(batch))
    if len(levels) < 2:
        return X.copy()
    data = X.to_numpy(dtype=float)
    keep = data.std(axis=0) > 0
    for level in levels:
        rows = batch == level
        if rows.sum() < 2:
            raise ValueError(f"batch {level!r} has fewer than 2 subjects")
        keep &= data[rows].std(axis=0, ddof=1) > 0
    Y = data[:, keep]
    n = len(Y)
    masks = [batch == level for level in levels]
    sizes = np.array([m.sum() for m in masks])
    batch_means = np.vstack([Y[m].mean(axis=0) for m in masks])
    grand_mean = (sizes[:, None] * batch_means).sum(axis=0) / n
    fitted = np.zeros_like(Y)
    for i, m in enumerate(masks):
        fitted[m] = batch_means[i]
    var_pooled = ((Y - fitted) ** 2).sum(axis=0) / n
    S = (Y - grand_mean) / np.sqrt(var_pooled)

    gamma_hat = np.vstack([S[m].mean(axis=0) for m in masks])
    delta_hat = np.vstack([S[m].var(axis=0, ddof=1) for m in masks])
    gamma_star, delta_star = np.empty_like(gamma_hat), np.empty_like(delta_hat)
    for i, m in enumerate(masks):
        g_bar, t2 = gamma_hat[i].mean(), gamma_hat[i].var(ddof=1)
        a, b = _aprior(delta_hat[i]), _bprior(delta_hat[i])
        g_old, d_old = gamma_hat[i].copy(), delta_hat[i].copy()
        k = sizes[i]
        for _ in range(max_iter):
            g_new = (k * t2 * gamma_hat[i] + d_old * g_bar) / (k * t2 + d_old)
            ssr = ((S[m] - g_new) ** 2).sum(axis=0)
            d_new = (0.5 * ssr + b) / (k / 2 + a - 1)
            change = max(np.max(np.abs(g_new - g_old) / np.maximum(np.abs(g_old), 1e-12)),
                         np.max(np.abs(d_new - d_old) / d_old))
            g_old, d_old = g_new, d_new
            if change < tol:
                break
        gamma_star[i], delta_star[i] = g_old, d_old

    adjusted = S.copy()
    for i, m in enumerate(masks):
        adjusted[m] = (S[m] - gamma_star[i]) / np.sqrt(delta_star[i])
    adjusted = adjusted * np.sqrt(var_pooled) + grand_mean
    out = data.copy()
    out[:, keep] = adjusted
    return pd.DataFrame(out, index=X.index, columns=X.columns)
