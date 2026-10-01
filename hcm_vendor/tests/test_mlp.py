import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from hcmv.config import load_config
from hcmv.mlp import TorchMLPClassifier
from hcmv.models import get_model
from hcmv.preprocessing import build_pipeline
from hcmv.splits import inner_cv
from hcmv.stats import fast_auc

FAST = dict(max_epochs=300, patience=30)


def _separable(n=120, d=6, seed=0, shift=2.0):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) < n * 0.4).astype(int)  # 40% positive, imbalanced
    X = rng.normal(size=(n, d))
    X[:, 0] += shift * y
    X[:, 1] -= shift * y
    return X, y


def test_get_params_and_clone_round_trip():
    model = TorchMLPClassifier(hidden=(64, 32), dropout=0.5, weight_decay=1e-2, random_state=3)
    copy = clone(model)
    assert copy.get_params() == model.get_params()
    assert copy.get_params()["hidden"] == (64, 32)
    copy.set_params(dropout=0.2)
    assert model.dropout == 0.5


def test_learns_separable_toy_set():
    X, y = _separable()
    X_test, y_test = _separable(seed=1)
    model = TorchMLPClassifier(**FAST).fit(X, y)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert fast_auc(y_test, proba[:, 1]) > 0.95
    assert set(model.predict(X_test)) <= {0, 1}
    assert model.early_stopping_used_ and 1 <= model.best_epoch_ <= model.n_epochs_


def test_same_seed_same_predictions_and_global_rng_untouched():
    import torch

    X, y = _separable()
    torch.manual_seed(123)
    before = torch.rand(1)
    torch.manual_seed(123)
    a = TorchMLPClassifier(random_state=7, **FAST).fit(X, y).predict_proba(X)
    after = torch.rand(1)
    b = TorchMLPClassifier(random_state=7, **FAST).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)
    assert torch.equal(before, after)  # fit did not consume the global RNG
    c = TorchMLPClassifier(random_state=8, **FAST).fit(X, y).predict_proba(X)
    assert not np.array_equal(a, c)


def test_early_stopping_restores_best_weights():
    X, y = _separable(n=80, shift=0.5)
    model = TorchMLPClassifier(hidden=(64, 32), dropout=0.0, weight_decay=0.0, lr=1e-2,
                               max_epochs=400, patience=15).fit(X, y)
    assert model.n_epochs_ < 400  # stopped early on a noisy problem
    assert model.n_epochs_ == model.best_epoch_ + 15
    assert model.best_validation_loss_ == pytest.approx(min(model.validation_curve_))


def test_string_labels_and_classes():
    X, y = _separable()
    labels = np.where(y == 1, "HCM", "NOR")
    model = TorchMLPClassifier(**FAST).fit(X, labels)
    assert list(model.classes_) == ["HCM", "NOR"]
    assert set(model.predict(X)) <= {"HCM", "NOR"}
    # predict_proba columns follow classes_, so column 1 is "NOR" here
    assert fast_auc(labels == "NOR", model.predict_proba(X)[:, 1]) > 0.95


def test_rejects_multiclass_and_feature_mismatch():
    X, y = _separable()
    with pytest.raises(ValueError):
        TorchMLPClassifier(**FAST).fit(X, np.arange(len(y)) % 3)
    model = TorchMLPClassifier(**FAST).fit(X, y)
    with pytest.raises(ValueError):
        model.predict_proba(X[:, :3])


def test_tiny_class_trains_without_validation_split():
    X, y = _separable(n=12)
    y = np.r_[np.ones(1), np.zeros(11)].astype(int)
    model = TorchMLPClassifier(max_epochs=20).fit(X, y)
    assert not model.early_stopping_used_ and model.n_epochs_ == 20


def test_pickle_round_trip():
    X, y = _separable()
    model = TorchMLPClassifier(**FAST).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(model.predict_proba(X), restored.predict_proba(X))


def test_grid_search_on_pipeline_with_study_grid():
    X, y = _separable()
    X = pd.DataFrame(X, columns=[f"ed_radiomics_original_glcm_F{i}" for i in range(X.shape[1])])
    estimator, grid = get_model("mlp", load_config())
    assert grid["model__hidden"] == [(32, 16), (64, 32)]
    estimator.set_params(**FAST)
    small = {"model__hidden": grid["model__hidden"], "model__dropout": [0.2]}
    search = GridSearchCV(build_pipeline(estimator), small, cv=inner_cv(3, seed=0), scoring="roc_auc")
    search.fit(X, y)
    assert search.best_score_ > 0.9
    assert search.predict_proba(X).shape == (len(y), 2)


def test_sklearn_estimator_checks():
    from sklearn.utils.estimator_checks import check_estimator

    failures = check_estimator(TorchMLPClassifier(max_epochs=50, patience=10), on_fail=None)
    failed = [f for f in failures if f["status"] == "failed"]
    assert not failed, [(f["check_name"], str(f["exception"])[:200]) for f in failed]
