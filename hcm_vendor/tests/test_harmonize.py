import numpy as np
import pandas as pd
import pytest

from hcmv.harmonize import combat


def _data(seed=0, n=40):
    rng = np.random.default_rng(seed)
    batch = np.array(["A"] * n + ["B"] * n)
    base = rng.normal(size=(2 * n, 6))
    shift = np.where(batch == "B", 3.0, 0.0)[:, None]
    scale = np.where(batch == "B", 2.0, 1.0)[:, None]
    X = pd.DataFrame(base * scale + shift, columns=[f"f{i}" for i in range(6)])
    X["const"] = 1.0
    return X, batch


def test_combat_removes_batch_location_and_scale():
    X, batch = _data()
    H = combat(X, batch)
    a, b = H[batch == "A"], H[batch == "B"]
    before = (X[batch == "B"].mean() - X[batch == "A"].mean()).abs().drop("const").mean()
    after = (b.mean() - a.mean()).abs().drop("const").mean()
    assert before > 2.5 and after < 0.3
    ratio = (b.std() / a.std()).drop("const")
    assert ratio.between(0.7, 1.4).all()
    assert (H["const"] == 1.0).all()


def test_combat_keeps_order_within_batch_and_index():
    X, batch = _data(seed=1)
    H = combat(X, batch)
    assert list(H.index) == list(X.index) and list(H.columns) == list(X.columns)
    for level in ("A", "B"):
        rows = batch == level
        np.testing.assert_array_equal(np.argsort(X.loc[rows, "f0"].to_numpy()),
                                      np.argsort(H.loc[rows, "f0"].to_numpy()))


def test_combat_rejects_missing_values():
    X, batch = _data()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        combat(X, batch)
