"""Model zoo (T31): class-balanced estimators and their hyperparameter grids.

``get_model`` returns an unfitted estimator and a grid whose keys are prefixed
with ``model__``, ready for ``GridSearchCV(build_pipeline(estimator), grid)``.
Four families: linear (``lr_en``), kernel (``svm``), tree ensembles (``rf``, ``xgb``)
and neural (``mlp``, see :mod:`hcmv.mlp`).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .mlp import TorchMLPClassifier


class BalancedXGBClassifier(XGBClassifier):
    """XGBoost with ``scale_pos_weight`` = n_negative / n_positive of the data it is fitted on,
    so class balancing is recomputed inside every (inner or outer) training fold."""

    def fit(self, X, y, **kwargs):
        y = np.asarray(y)
        self.set_params(scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))
        return super().fit(X, y, **kwargs)


def _estimator(name: str, seed: int):
    if name == "lr_en":
        return LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000,
                                  class_weight="balanced", random_state=seed)
    if name == "svm":
        return SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=1, random_state=seed)
    if name == "xgb":
        return BalancedXGBClassifier(tree_method="hist", n_jobs=1, subsample=0.8, colsample_bytree=0.8,
                                     eval_metric="logloss", random_state=seed)
    if name == "mlp":
        return TorchMLPClassifier(random_state=seed)
    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")


MODEL_NAMES = ("lr_en", "svm", "rf", "xgb", "mlp")
# Models whose fit depends on the seed beyond tie-breaking; transfers average them over seeds.
STOCHASTIC_MODELS = ("rf", "xgb", "mlp")


def get_model(name: str, config: dict, seed: int = None):
    """(unfitted estimator, ``model__``-prefixed param grid) for a model in ``config['models']``."""
    seed = config["seed"] if seed is None else seed
    grid = {f"model__{key}": [tuple(v) if isinstance(v, list) else v for v in values]
            for key, values in config["models"][name].items()}
    return _estimator(name, seed), grid


def grid_size(grid: dict) -> int:
    return int(np.prod([len(v) for v in grid.values()]))
