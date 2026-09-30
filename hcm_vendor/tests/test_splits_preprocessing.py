import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from hcmv.preprocessing import CorrelationFilter, build_pipeline, build_preprocessor, select_features
from hcmv.splits import inner_cv, outer_splits, strata_labels


def _cohort(n_per_cell=12, seed=0):
    """Pooled-style table: 2 vendors x 2 classes, one row per subject."""
    rng = np.random.default_rng(seed)
    rows = []
    for vendor in ("Siemens", "Philips"):
        for y in (0, 1):
            for i in range(n_per_cell):
                rows.append({"subject_id": f"{vendor}_{y}_{i}", "vendor": vendor, "y": y,
                             "x": rng.normal(y, 1.0)})
    return pd.DataFrame(rows).set_index("subject_id")


def _features(n=60, seed=0):
    """Columns named like the real tables, with a clinical/shape duplicate pair."""
    rng = np.random.default_rng(seed)
    myo = rng.normal(100, 20, n)
    return pd.DataFrame({
        "ed_radiomics_original_shape_MeshVolume": myo * 1.05 + rng.normal(0, 0.1, n),
        "ed_radiomics_original_glcm_Contrast": rng.normal(0, 1, n),
        "ed_myo_volume_ml": myo,
        "ed_lv_volume_ml": rng.normal(150, 30, n),
        "ed_radiomics_original_firstorder_Mean": rng.normal(0, 1, n),
    })


# ---------------------------------------------------------------- splits


def test_each_subject_tested_once_per_repeat_and_disjoint():
    table = _cohort()
    splits = list(outer_splits(table, n_splits=5, n_repeats=3, seed=1, stratify_by=("y", "vendor")))
    assert len(splits) == 15
    for repeat in range(3):
        folds = [s for s in splits if s.repeat == repeat]
        assert [s.fold for s in folds] == list(range(5))
        tested = np.concatenate([s.test for s in folds])
        assert sorted(tested) == list(range(len(table)))
        for s in folds:
            assert not set(s.train) & set(s.test)
            assert len(s.train) + len(s.test) == len(table)


def test_pooled_splits_keep_vendor_mix():
    table = _cohort()
    for s in outer_splits(table, 4, 1, seed=0, stratify_by=("y", "vendor")):
        cells = table.iloc[s.test].groupby(["vendor", "y"]).size()
        assert len(cells) == 4 and cells.min() == cells.max() == 3


def test_splits_are_seeded():
    table = _cohort()
    a = [s.test.tolist() for s in outer_splits(table, 5, 2, seed=7)]
    b = [s.test.tolist() for s in outer_splits(table, 5, 2, seed=7)]
    c = [s.test.tolist() for s in outer_splits(table, 5, 2, seed=8)]
    assert a == b and a != c


def test_small_composite_stratum_falls_back_to_label():
    table = _cohort()
    # Only 2 GE subjects: y x vendor has a stratum smaller than k=5.
    extra = pd.DataFrame({"vendor": ["GE", "GE"], "y": [0, 1], "x": [0.0, 1.0]}, index=["ge_0", "ge_1"])
    table = pd.concat([table, extra])
    splits = list(outer_splits(table, 5, 1, seed=0, stratify_by=("y", "vendor")))
    assert sorted(np.concatenate([s.test for s in splits])) == list(range(len(table)))
    assert strata_labels(table.head(1), ("y", "vendor"))[0] == "0|Siemens"


def test_inner_cv_is_seeded_stratified():
    cv = inner_cv(5, seed=3)
    assert cv.n_splits == 5 and cv.shuffle and cv.random_state == 3


# ---------------------------------------------------------- preprocessing


def test_select_features_orders_clinical_first_and_filters_families():
    cols = select_features(_features(), "all")
    assert cols[:2] == ["ed_myo_volume_ml", "ed_lv_volume_ml"]
    assert cols[2] == "ed_radiomics_original_shape_MeshVolume"
    assert set(cols[3:]) == {"ed_radiomics_original_glcm_Contrast", "ed_radiomics_original_firstorder_Mean"}
    assert select_features(_features(), "clinical") == ["ed_myo_volume_ml", "ed_lv_volume_ml"]


def test_myocardial_mass_is_ordered_before_myo_volume():
    X = _features()
    X["ed_myocardial_mass_g"] = X["ed_myo_volume_ml"] * 1.05
    cols = select_features(X, "clinical")
    assert cols[0] == "ed_myocardial_mass_g"
    kept = CorrelationFilter(0.95).fit(X[cols]).get_feature_names_out()
    assert "ed_myocardial_mass_g" in kept and "ed_myo_volume_ml" not in kept


def test_correlation_filter_keeps_clinical_over_shape_duplicate():
    X = _features()[select_features(_features(), "all")]
    kept = CorrelationFilter(0.95).fit(X).get_feature_names_out()
    assert "ed_myo_volume_ml" in kept
    assert "ed_radiomics_original_shape_MeshVolume" not in kept
    assert len(kept) == X.shape[1] - 1


def test_correlation_filter_is_deterministic_and_order_dependent():
    X = _features()[select_features(_features(), "all")]
    a = CorrelationFilter(0.95).fit(X).get_feature_names_out()
    b = CorrelationFilter(0.95).fit(X).get_feature_names_out()
    assert list(a) == list(b)
    reversed_cols = X[X.columns[::-1]]
    kept = CorrelationFilter(0.95).fit(reversed_cols).get_feature_names_out()
    # Greedy in column order: now the shape column comes first and wins.
    assert "ed_radiomics_original_shape_MeshVolume" in kept
    assert "ed_myo_volume_ml" not in kept


def test_correlation_filter_feature_names_round_trip_numpy():
    X = _features()
    f = CorrelationFilter(0.95).fit(X)
    out = f.transform(X.to_numpy())
    assert out.shape[1] == len(f.get_feature_names_out())
    assert list(f.transform(X).columns) == list(f.get_feature_names_out())
    assert list(f.get_feature_names_out([f"c{i}" for i in range(X.shape[1])])) == [
        f"c{i}" for i in f.keep_
    ]


def test_preprocessor_drops_constant_imputes_and_scales():
    X = _features()
    X["constant"] = 3.0
    X.iloc[0, 3] = np.nan
    out = build_preprocessor().fit_transform(X)
    assert "constant" not in out.columns
    assert not out.isna().any().any()
    np.testing.assert_allclose(out.mean(), 0, atol=1e-9)
    np.testing.assert_allclose(out.std(ddof=0), 1, atol=1e-9)


def test_preprocessor_is_fitted_on_train_rows_only():
    X = _features(n=80)
    X.iloc[5, 0] = np.nan
    train, test = X.iloc[:60], X.iloc[60:]
    fitted = build_preprocessor().fit(train)
    before = fitted.transform(train)
    altered = test * 1000 + 7  # wildly different test rows
    refit = build_preprocessor().fit(train)
    # Fitting never saw test rows: changing them changes nothing learned from train.
    pd.testing.assert_frame_equal(refit.transform(train), before)
    np.testing.assert_allclose(refit.named_steps["scale"].mean_, fitted.named_steps["scale"].mean_)
    assert refit.named_steps["impute"].statistics_[0] == pytest.approx(train.iloc[:, 0].median())
    # Transforming test uses the train statistics, so altered rows land far from 0.
    assert refit.transform(altered).abs().to_numpy().mean() > 10


def test_pipeline_clones_and_runs_in_grid_search():
    X = _features(n=80, seed=2)
    y = (X["ed_lv_volume_ml"] + np.random.default_rng(0).normal(0, 10, len(X)) > 150).astype(int)
    pipe = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    clone(pipe)
    search = GridSearchCV(pipe, {"model__C": [0.1, 1.0], "correlation__threshold": [0.9, 0.95]},
                          cv=inner_cv(3, seed=0), scoring="roc_auc")
    search.fit(X, y)
    assert search.best_score_ > 0.7
    proba = search.predict_proba(X)
    assert proba.shape == (len(X), 2)
