import json

import joblib
import numpy as np
import pandas as pd
import pytest

from hcmv.config import load_config
from hcmv.runner import (PREDICTION_COLUMNS, apply_filter, collect_results, run_experiment, run_nested_cv,
                         run_transfer, smoke_config)


def _table(n_per_cell=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for vendor in ("Siemens", "Philips"):
        for y in (0, 1):
            for _ in range(n_per_cell):
                rows.append({"vendor": vendor, "y": y, "role": "train_pool",
                             "disease": "HCM" if y else "NOR"})
    table = pd.DataFrame(rows)
    n = len(table)
    table["ed_wall_thickness_max_mm"] = 10 + 4 * table["y"] + rng.normal(0, 1.5, n)
    table["ed_lv_volume_ml"] = rng.normal(150, 20, n)
    table["ed_myocardial_mass_g"] = 100 + 30 * table["y"] + rng.normal(0, 10, n)
    table["ed_radiomics_original_shape_Sphericity"] = rng.normal(0.5, 0.05, n)
    table["ed_radiomics_original_glcm_Contrast"] = rng.normal(0, 1, n) + 0.5 * (table["vendor"] == "Siemens")
    table.index = pd.Index([f"s{i:03d}" for i in range(n)], name="subject_id")
    return table


@pytest.fixture()
def config(tmp_path):
    cfg = smoke_config(load_config())
    cfg["paths"]["output_root"] = str(tmp_path)
    return cfg


def test_smoke_config_shrinks_everything():
    base = load_config()
    smoke = smoke_config(base)
    assert smoke["cv"] == {"outer_splits": 2, "outer_repeats": 1, "inner_splits": 2}
    assert smoke["models"]["mlp"]["hidden"] == [[32, 16], [64, 32]]
    assert all(len(v) == 1 for k, v in smoke["models"]["mlp"].items() if k != "hidden")
    assert smoke["experiments"]["model_params"]["rf"]["n_estimators"] == 50
    assert base["cv"]["outer_repeats"] == 5  # original untouched


def test_apply_filter():
    table = _table()
    assert len(apply_filter(table, {"vendor": "Siemens"})) == 20
    assert len(apply_filter(table, {"vendor": ["Siemens", "Philips"], "y": 1})) == 20


@pytest.mark.parametrize("model", ["lr_en", "mlp"])
def test_nested_cv_writes_store_and_is_resumable(config, model):
    table = _table()
    first = run_nested_cv(table, {"role": "train_pool"}, "all", model, config, "E1", "pooled")
    assert first["status"] == "done"
    directory = first["dir"]
    predictions = pd.read_parquet(f"{directory}/predictions.parquet")
    assert list(predictions.columns) == PREDICTION_COLUMNS
    # every subject predicted exactly once per repeat, out of fold
    assert predictions.groupby(["subject_id", "repeat"]).size().eq(1).all()
    assert set(predictions["subject_id"]) == set(table.index)
    assert predictions["prob"].between(0, 1).all()
    metrics = json.loads(open(f"{directory}/metrics.json").read())
    assert metrics["n"] == 40 and metrics["overall"]["auc"]["estimate"] > 0.8
    assert set(metrics["by_vendor"]) == {"Siemens", "Philips"}
    hyper = json.loads(open(f"{directory}/hyperparams.json").read())
    assert len(hyper["folds"]) == 2 and "best_params" in hyper["folds"][0]
    manifest = json.loads(open(f"{directory}/manifest.json").read())
    assert manifest["spec"]["smoke"] is True and manifest["run_hash"]

    again = run_nested_cv(table, {"role": "train_pool"}, "all", model, config, "E1", "pooled")
    assert again["status"] == "skipped"
    config["seed"] += 1  # a different config is not skipped
    assert run_nested_cv(table, {"role": "train_pool"}, "all", model, config, "E1", "pooled")["status"] == "done"


def test_nested_cv_is_deterministic(config, tmp_path):
    table = _table()
    a = run_nested_cv(table, {"vendor": "Siemens"}, "all", "mlp", config, "E1", "siemens")
    b = run_nested_cv(table, {"vendor": "Siemens"}, "all", "mlp", config, "E1", "siemens", n_jobs=2, force=True)
    assert a["dir"] == b["dir"]
    pa = pd.read_parquet(f"{a['dir']}/predictions.parquet")
    run_nested_cv(table, {"vendor": "Siemens"}, "all", "mlp", config, "E1", "siemens", force=True)
    pc = pd.read_parquet(f"{a['dir']}/predictions.parquet")
    pd.testing.assert_frame_equal(pa, pc)


def test_transfer_averages_seeds_and_saves_pipelines(config):
    table = _table()
    result = run_transfer(table, {"vendor": "Siemens"}, {"vendor": "Philips"}, "all", "rf", config,
                          experiment="E2", unit="siemens_to_philips")
    directory = result["dir"]
    by_seed = pd.read_parquet(f"{directory}/predictions_by_seed.parquet")
    predictions = pd.read_parquet(f"{directory}/predictions.parquet")
    assert by_seed["seed"].nunique() == 2 and len(predictions) == 20
    assert set(predictions["vendor"]) == {"Philips"}
    mean = by_seed.groupby("subject_id")["prob"].mean()
    np.testing.assert_allclose(predictions.set_index("subject_id")["prob"].loc[mean.index], mean)
    pipelines = sorted((pd.io.common.Path(directory) / "models").glob("seed_*.joblib"))
    assert len(pipelines) == 2
    pipeline = joblib.load(pipelines[0])
    test = apply_filter(table, {"vendor": "Philips"})
    features = json.loads(open(f"{directory}/hyperparams.json").read())["features"]
    assert pipeline.predict_proba(test[features]).shape == (20, 2)
    # deterministic models use a single seed
    lr = run_transfer(table, {"vendor": "Siemens"}, {"vendor": "Philips"}, "all", "lr_en", config,
                      experiment="E2", unit="siemens_to_philips")
    assert json.loads(open(f"{lr['dir']}/metrics.json").read())["n_seeds"] == 1


def test_transfer_rejects_overlap(config):
    table = _table()
    with pytest.raises(ValueError, match="both train and test"):
        run_transfer(table, {"role": "train_pool"}, {"vendor": "Philips"}, "all", "lr_en", config)


def test_run_experiment_and_summary(config, monkeypatch):
    table = _table()
    monkeypatch.setattr("hcmv.runner.pd.read_parquet", lambda path: table)
    messages = []
    summary = run_experiment(config, "E1", ["lr_en", "svm"], ["all", "all-no-wt"], units=["pooled"],
                             log=messages.append)
    assert len(summary) == 4 and summary["auc"].notna().all()
    assert set(summary["family_set"]) == {"all", "all-no-wt"}
    assert collect_results(f"{config['paths']['output_root']}/runs-smoke/E1").shape[0] == 4
    run_experiment(config, "E1", ["lr_en", "svm"], ["all", "all-no-wt"], units=["pooled"], log=messages.append)
    assert sum("skipped" in m for m in messages) == 4
