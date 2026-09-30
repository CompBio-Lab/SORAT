import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from hcmv.stats import (
    binary_metrics,
    bootstrap_ci,
    fast_auc,
    holm,
    paired_bootstrap,
    permutation_p_value,
    repeated_metric,
    stratified_bootstrap_indices,
    summarize,
    wilson_ci,
)


def _sim(n=100, shift=1.0, seed=0):
    """Normals separated by ``shift`` SD; the true AUC for shift 1 is Phi(1/sqrt2) = 0.760."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    score = rng.normal(0, 1, n) + shift * y
    return y, 1 / (1 + np.exp(-score))


def test_fast_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    prob = np.round(rng.random(200), 1)  # heavy ties
    assert fast_auc(y, prob) == pytest.approx(roc_auc_score(y, prob))
    assert np.isnan(fast_auc(np.zeros(5), np.arange(5)))


def test_metrics_at_fixed_threshold_hand_example():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    prob = np.array([0.9, 0.6, 0.4, 0.5, 0.2, 0.1, 0.3])
    m = binary_metrics(y, prob)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(3 / 4)  # 0.5 counts as positive
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 3 / 4) / 2)
    assert m["brier"] == pytest.approx(brier_score_loss(y, prob))
    assert m["auc"] == pytest.approx(roc_auc_score(y, prob))


def test_repeated_metric_averages_over_repeats():
    y, p1 = _sim(seed=1)
    _, p2 = _sim(seed=2)
    P = np.column_stack([p1, p2])
    expected = (roc_auc_score(y, p1) + roc_auc_score(y, p2)) / 2
    assert repeated_metric(y, P, "auc") == pytest.approx(expected)
    assert repeated_metric(y, p1, "auc") == pytest.approx(roc_auc_score(y, p1))


def test_bootstrap_indices_preserve_class_counts_and_are_seeded():
    y = np.array([0] * 7 + [1] * 3)
    a = list(stratified_bootstrap_indices(y, 5, seed=4))
    b = list(stratified_bootstrap_indices(y, 5, seed=4))
    assert all((x == z).all() for x, z in zip(a, b))
    assert all(np.bincount(y[rows]).tolist() == [7, 3] for rows in a)


def test_ci_is_seeded_and_contains_estimate():
    y, p = _sim()
    a = bootstrap_ci(y, p, "auc", n_boot=300, seed=0)
    b = bootstrap_ci(y, p, "auc", n_boot=300, seed=0)
    assert a == b
    assert a["ci_low"] < a["estimate"] < a["ci_high"]
    rows = summarize(y, np.column_stack([p, p]), n_boot=200, seed=0)
    assert [r["metric"] for r in rows] == ["auc", "balanced_accuracy", "sensitivity", "specificity", "brier"]
    assert all(r["ci_low"] <= r["estimate"] <= r["ci_high"] for r in rows)


def test_auc_ci_coverage_is_near_nominal():
    true_auc = 0.7602  # Phi(1 / sqrt(2))
    covered = 0
    n_sims = 200
    for s in range(n_sims):
        y, p = _sim(n=100, seed=1000 + s)
        ci = bootstrap_ci(y, p, "auc", n_boot=300, seed=s)
        covered += ci["ci_low"] <= true_auc <= ci["ci_high"]
    # Percentile bootstrap runs slightly liberal at n=100; accept a band around 95%.
    assert 0.88 <= covered / n_sims <= 0.99


def test_paired_bootstrap_identical_predictions_gives_p_one():
    y, p = _sim()
    out = paired_bootstrap(y, p, p, "auc", n_boot=300, seed=0)
    assert out["difference"] == 0
    assert out["p_value"] == pytest.approx(1.0)
    assert out["ci_low"] == out["ci_high"] == 0


def test_paired_bootstrap_detects_clear_difference():
    y, good = _sim(n=200, shift=2.0, seed=3)
    noise = np.random.default_rng(9).random(len(y))
    out = paired_bootstrap(y, good, noise, "auc", n_boot=500, seed=0)
    assert out["difference"] > 0.3
    assert out["ci_low"] > 0
    assert out["p_value"] < 0.01
    flipped = paired_bootstrap(y, noise, good, "auc", n_boot=500, seed=0)
    assert flipped["difference"] == pytest.approx(-out["difference"])
    assert flipped["p_value"] == pytest.approx(out["p_value"])


def test_permutation_p_value():
    assert permutation_p_value(0.9, [0.1, 0.2, 0.95, 0.3]) == pytest.approx(2 / 5)
    assert permutation_p_value(5.0, np.zeros(999)) == pytest.approx(1 / 1000)


def test_holm_adjustment():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])


def test_wilson_ci():
    low, high = wilson_ci(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3)
    assert high == pytest.approx(0.9433, abs=1e-3)
    low, high = wilson_ci(0, 3)
    assert low == pytest.approx(0.0, abs=1e-12) and 0 < high < 1
    assert all(np.isnan(wilson_ci(0, 0)))
