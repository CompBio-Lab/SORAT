import json

import numpy as np
import pandas as pd
import pytest

from hcmv.config import load_config
from hcmv.experiments import e1, e2, e3
from hcmv.experiments.common import format_p, load_runs
from hcmv.runner import run_nested_cv, run_transfer, smoke_config


def _table(n_per_cell=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"vendor": v, "y": y, "role": "train_pool", "disease": "HCM" if y else "NOR"}
            for v in ("Siemens", "Philips") for y in (0, 1) for _ in range(n_per_cell)]
    table = pd.DataFrame(rows)
    n = len(table)
    table["ed_wall_thickness_max_mm"] = 10 + 4 * table["y"] + rng.normal(0, 1.5, n)
    table["ed_lv_volume_ml"] = rng.normal(150, 20, n)
    table["ed_myocardial_mass_g"] = 100 + 30 * table["y"] + rng.normal(0, 10, n)
    table["ed_radiomics_original_glcm_Contrast"] = rng.normal(0, 1, n) + 2 * (table["vendor"] == "Siemens")
    table.index = pd.Index([f"s{i:03d}" for i in range(n)], name="subject_id")
    return table


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    cfg = smoke_config(load_config())
    cfg["paths"]["output_root"] = str(tmp_path_factory.mktemp("out"))
    table = _table()
    models = ("lr_en", "svm")
    for unit, filt in (("pooled", {"role": "train_pool"}), ("siemens", {"vendor": "Siemens"}),
                       ("philips", {"vendor": "Philips"})):
        for model in models:
            run_nested_cv(table, filt, "all", model, cfg, "E1", unit)
    for direction, a, b in (("siemens_to_philips", "Siemens", "Philips"), ("philips_to_siemens", "Philips", "Siemens")):
        for model in models:
            run_transfer(table, {"vendor": a}, {"vendor": b}, "all", model, cfg, experiment="E2", unit=direction)
    return cfg


def test_e1_report_tables_and_figures(store):
    text = e1.e1_report(store)
    out = e1.analysis_dir(store, "E1")
    metrics = pd.read_csv(out / "e1_metrics.csv")
    # 3 cohorts x 2 models reading rows + pooled per-vendor rows (2 vendors x 2 models)
    assert len(metrics) == 6 + 4
    assert set(metrics["reading"]) == {"pooled", e1.READING_A, e1.READING_B}
    comparisons = pd.read_csv(out / "e1_model_comparisons.csv")
    assert len(comparisons) == 1 and comparisons["p_holm"].iloc[0] >= comparisons["p_value"].iloc[0]
    assert (out / "e1_roc_all.png").exists() and (out / "e1_model_comparison_all.png").exists()
    assert "Grid edge check" in text


def test_grid_edges_ordering():
    config = {"models": {"rf": {"max_depth": [None, 3, 5], "max_features": ["sqrt", 0.3]},
                         "mlp": {"hidden": [[32, 16], [64, 32]]}}}
    runs = {("pooled", "all", "rf"): {"hyperparams": {"folds": [
                {"best_params": {"max_depth": None, "max_features": "sqrt"}},
                {"best_params": {"max_depth": None, "max_features": 0.3}}]}},
            ("pooled", "all", "mlp"): {"hyperparams": {"folds": [
                {"best_params": {"hidden": [64, 32]}}, {"best_params": {"hidden": [64, 32]}}]}}}
    edges = e1.grid_edges(runs, config).set_index(["model", "axis"])
    depth = edges.loc[("rf", "max_depth")]
    assert depth["highest"] == "None" and depth["share_highest"] == 1.0 and depth["at_edge"]
    assert ("rf", "max_features") not in edges.index  # only one ordered value besides 'sqrt'
    assert edges.loc[("mlp", "hidden")]["share_highest"] == 1.0


def test_e2_gap_matches_definition(store):
    e2.e2_report(store)
    out = e1.analysis_dir(store, "E2")
    gap = pd.read_csv(out / "e2_gap.csv")
    assert len(gap) == 2 * 2 * 5  # directions x models x metrics
    row = gap[(gap.direction == "siemens_to_philips") & (gap.model == "lr_en") & (gap.metric == "auc")].iloc[0]
    e1_auc = load_runs(store, "E1")[("philips", "all", "lr_en")]["metrics"]["overall"]["auc"]["estimate"]
    e2_auc = load_runs(store, "E2")[("siemens_to_philips", "all", "lr_en")]["metrics"]["overall"]["auc"]["estimate"]
    assert row["difference"] == pytest.approx(e1_auc - e2_auc)
    for name in ("e2_gap_auc.png", "e2_roc_all.png", "e2_calibration_all.png", "model_comparison_all.png"):
        assert (out / name).exists()


def test_format_p():
    assert format_p(0.0, {"bootstrap_resamples": 2000}) == "< 0.0005"
    assert format_p(0.0312, {"bootstrap_resamples": 2000}) == "0.031"


def test_e3_probe_detects_vendor_signal(tmp_path, monkeypatch):
    cfg = e3.smoke_e3(smoke_config(load_config()))
    cfg["paths"]["output_root"] = str(tmp_path)
    cfg["permutations"] = 19
    cfg["e3"]["probes"] = {"texture_norm": ["norm", "texture"], "clinical": ["norm", "clinical"]}
    cfg["e3"]["classifiers"] = ["logreg"]
    cfg["e3"]["targets"] = {"siemens_vs_philips": ["Siemens", "Philips"]}
    table = _table(n_per_cell=15)
    nor = table[table.disease == "NOR"]

    def fake_load(config, target, probe):
        columns = {"texture_norm": ["ed_radiomics_original_glcm_Contrast"],
                   "clinical": ["ed_lv_volume_ml", "ed_myocardial_mass_g"]}[probe]
        return nor[columns], nor["vendor"].to_numpy(), nor.index.to_numpy()

    monkeypatch.setattr(e3, "load_probe_data", fake_load)
    result = e3.run_e3(cfg, log=lambda *_: None).set_index("probe")
    assert result.loc["texture_norm", "balanced_accuracy"] > 0.8
    assert result.loc["texture_norm", "p_value"] == pytest.approx(1 / 20)  # beats all 19 shuffles
    assert result.loc["clinical", "balanced_accuracy"] < 0.75
    again = e3.run_e3(cfg, log=lambda *_: None).set_index("probe")  # resumable: same numbers from disk
    assert again.loc["texture_norm", "null"] == result.loc["texture_norm", "null"]
    text = e3.e3_report(cfg, again.reset_index())
    assert "Siemens vs Philips" in text
    assert (tmp_path / "runs-smoke" / "E3" / "analysis" / "e3_null_siemens_vs_philips_logreg.png").exists()


def test_grid_edges_ignore_ties():
    config = {"models": {"lr_en": {"C": [0.1, 1.0, 10.0]}}}
    scores = lambda a, b, c: [{"params": {"C": v}, "score": s} for v, s in zip((0.1, 1.0, 10.0), (a, b, c))]  # noqa
    folds = [{"best_params": {"C": 0.1}, "grid_scores": scores(0.99, 0.99, 0.95)},  # tie -> not strict
             {"best_params": {"C": 0.1}, "grid_scores": scores(0.99, 0.98, 0.95)},  # strict
             {"best_params": {"C": 1.0}, "grid_scores": scores(0.90, 0.99, 0.95)}]
    row = e1.grid_edges({("pooled", "all", "lr_en"): {"hyperparams": {"folds": folds}}}, config).iloc[0]
    assert row["share_lowest"] == pytest.approx(2 / 3)
    assert row["share_strict_lowest"] == pytest.approx(1 / 3)
    assert not row["at_edge"]
