import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from hcmv.config import load_config
from hcmv.models import MODEL_NAMES, BalancedXGBClassifier, get_model, grid_size
from hcmv.preprocessing import build_pipeline
from hcmv.splits import inner_cv
from hcmv.stats import fast_auc


@pytest.fixture(scope="module")
def config():
    return load_config()


def _toy(n=80, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n - n // 3), np.ones(n // 3)].astype(int)  # imbalanced, like a vendor subset
    X = pd.DataFrame({
        "ed_lv_volume_ml": rng.normal(150, 20, n),
        "ed_wall_thickness_max_mm": 10 + 5 * y + rng.normal(0, 1.5, n),
        "ed_radiomics_original_glcm_Contrast": rng.normal(0, 1, n),
        "ed_radiomics_original_shape_Sphericity": rng.normal(0.5, 0.05, n) + 0.05 * y,
    })
    return X, y


def test_grid_sizes_match_ticket(config):
    sizes = {name: grid_size(get_model(name, config)[1]) for name in MODEL_NAMES}
    assert sizes == {"lr_en": 24, "svm": 16, "rf": 18, "xgb": 16}


def test_grid_values_parse_from_yaml(config):
    _, rf = get_model("rf", config)
    assert rf["model__max_depth"][0] is None
    _, svm = get_model("svm", config)
    assert svm["model__gamma"][0] == "scale"
    _, lr = get_model("lr_en", config)
    np.testing.assert_allclose(lr["model__C"], np.logspace(-3, 2, 8), rtol=1e-3)


def test_models_are_class_balanced_and_seeded(config):
    for name in MODEL_NAMES:
        est, _ = get_model(name, config, seed=5)
        params = est.get_params()
        assert params["random_state"] == 5
        if name != "xgb":
            assert params["class_weight"] == "balanced"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_each_model_runs_in_grid_search_pipeline(config, name):
    X, y = _toy()
    est, grid = get_model(name, config)
    small = {k: v[:2] for k, v in grid.items()}  # keep the test fast
    if name == "rf":
        est.set_params(n_estimators=50)
    search = GridSearchCV(build_pipeline(clone(est)), small, cv=inner_cv(3, seed=0), scoring="roc_auc")
    search.fit(X, y)
    proba = search.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert fast_auc(y, proba[:, 1]) > 0.85


def test_balanced_xgb_sets_scale_pos_weight_from_fit_data():
    X, y = _toy()
    model = BalancedXGBClassifier(n_estimators=10, max_depth=2, n_jobs=1)
    model.fit(X, y)
    assert model.get_params()["scale_pos_weight"] == pytest.approx((y == 0).sum() / (y == 1).sum())
    model.fit(X.iloc[:40], np.r_[np.zeros(20), np.ones(20)].astype(int))
    assert model.get_params()["scale_pos_weight"] == pytest.approx(1.0)
    assert clone(model).get_params()["max_depth"] == 2
