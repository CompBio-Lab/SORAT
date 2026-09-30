"""Leakage-safe preprocessing (T30): every step is fitted inside the training fold."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .features import feature_family

FAMILY_PRIORITY = {"clinical": 0, "shape": 1, "texture": 2, "sensitivity": 3}
FAMILY_SETS = {
    "clinical": ("clinical",),
    "clinical+shape": ("clinical", "shape"),
    "clinical+texture": ("clinical", "texture"),
    "all": ("clinical", "shape", "texture"),
    "shape": ("shape",),
    "texture": ("texture",),
}


def select_features(table: pd.DataFrame, family_set: str) -> list:
    """Feature columns for a named family set, clinical first so the correlation
    filter keeps interpretable clinical measures over correlated radiomics."""
    families = FAMILY_SETS[family_set]
    columns = [c for c in table.columns if feature_family(c) in families]
    return sorted(columns, key=lambda c: (FAMILY_PRIORITY[feature_family(c)], columns.index(c)))


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop features whose |Pearson r| with an earlier kept feature exceeds ``threshold``.

    Greedy in column order, so earlier (higher-priority) columns win ties.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        corr = np.abs(np.corrcoef(X.to_numpy(dtype=float), rowvar=False))
        corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0)
        keep = []
        for j in range(X.shape[1]):
            if all(corr[j, k] <= self.threshold for k in keep):
                keep.append(j)
        self.keep_ = np.asarray(keep, dtype=int)
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_")
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.keep_]
        return np.asarray(X)[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "keep_")
        names = self.feature_names_in_ if input_features is None else np.asarray(input_features, dtype=object)
        return names[self.keep_]


def build_preprocessor(corr_threshold: float = 0.95) -> Pipeline:
    """Median impute -> drop zero-variance -> correlation filter -> standardize."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(0.0)),
        ("correlation", CorrelationFilter(corr_threshold)),
        ("scale", StandardScaler()),
    ]).set_output(transform="pandas")


def build_pipeline(estimator, corr_threshold: float = 0.95) -> Pipeline:
    """Preprocessing followed by ``estimator`` as step ``model``."""
    steps = list(build_preprocessor(corr_threshold).steps) + [("model", estimator)]
    return Pipeline(steps).set_output(transform="pandas")
