"""Cross-validation splits. One row per subject, so every split is patient-level."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


@dataclass(frozen=True)
class Split:
    repeat: int
    fold: int
    train: np.ndarray  # positional indices
    test: np.ndarray


def strata_labels(table: pd.DataFrame, by=("y",)) -> np.ndarray:
    """Composite stratification labels, e.g. y x vendor for pooled cohorts."""
    return table[list(by)].astype(str).agg("|".join, axis=1).to_numpy()


def outer_splits(table: pd.DataFrame, n_splits: int, n_repeats: int, seed: int, stratify_by=("y",)):
    """Repeated stratified K-fold over subjects; yields :class:`Split` objects."""
    strata = strata_labels(table, stratify_by)
    counts = pd.Series(strata).value_counts()
    if counts.min() < n_splits:
        # Fall back to label-only stratification when a composite stratum is too small.
        strata = strata_labels(table, ("y",))
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for index, (train, test) in enumerate(splitter.split(np.zeros(len(table)), strata)):
        yield Split(repeat=index // n_splits, fold=index % n_splits, train=train, test=test)


def inner_cv(n_splits: int, seed: int) -> StratifiedKFold:
    """Inner CV used for hyperparameter tuning inside a training fold."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
