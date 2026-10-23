"""Shared loading, labelling and statistics for the SORAT manuscript analyses.

Every analysis reads the SORAT result folders (one per dataset) and the dataset
metadata. Locations come from environment variables; see paper/README.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set {name} (see paper/README.md).")
    return Path(value)


OUT_ROOT = Path(__file__).resolve().parents[1]  # paper/
REPO = OUT_ROOT.parent  # SORAT repository root
DATA_ROOT = _env_path("SORAT_DATA_ROOT")  # holds ACDC/, M&Ms/, M&Ms_2/, Atria_2018/, AIM-ASD/
RESULTS = _env_path("SORAT_RESULTS")  # SORAT output folders, one per dataset
RUNS = Path(os.environ.get("SORAT_RUNS") or RESULTS.parent / "paper_runs")  # standalone and repeat runs
SAMPLESHEETS = Path(os.environ.get("SORAT_SAMPLESHEETS") or REPO / "data")  # samplesheets and M&Ms-1 Info.cfg
TABLES = OUT_ROOT / "tables"
FIGURES = OUT_ROOT / "figures"

DATASETS = {
    "ACDC": {"folder": "ACDC", "label": "ACDC"},
    "MMS1": {"folder": "MMS", "label": "M&Ms-1"},
    "MMS2": {"folder": "MMS_2", "label": "M&Ms-2"},
}

# One representative variant per trained checkpoint is reported in the main
# text; single seeds go to the supplement.
MAIN_MODELS = {
    "cinema__acdc_ensemble": "CineMA-ACDC",
    "cinema__mnms_ensemble": "CineMA-M&Ms",
    "cinema__mnms2_ensemble": "CineMA-M&Ms-2",
    "nnformer__fold0": "nnFormer",
    "vsa3l__model": "VSA-3L",
}
MODEL_ORDER = list(MAIN_MODELS.values())
MODEL_COLORS = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))

STRUCTURES = [("rv", "RV"), ("myo", "MYO"), ("lv", "LV")]

# Data each checkpoint was trained on, used to mark results as in-distribution.
# CineMA checkpoints were fine-tuned on the official challenge training splits;
# VSA-3L was trained on non-public data (MONAI model zoo card).
TRAINED_ON = {
    "CineMA-ACDC": {"ACDC"},
    "CineMA-M&Ms": {"MMS1"},
    "CineMA-M&Ms-2": {"MMS2"},
    "nnFormer": {"ACDC"},
    "VSA-3L": set(),
}

# M&Ms-2 subjects 1-160 are the challenge training set and 161-200 validation;
# 201-360 are the test set used for the primary analysis.
MMS2_TEST = range(201, 361)


def load_metrics(dataset: str, include_postprocessed: bool = False) -> pd.DataFrame:
    """Per-frame metrics for one dataset, with string patient IDs and display names."""
    folder = DATASETS[dataset]["folder"]
    df = pd.read_csv(RESULTS / folder / "comparison" / "aggregated_metrics.csv", dtype={"patient_id": str})
    df["postprocessed"] = df["model"].str.endswith("_pp")
    if not include_postprocessed:
        df = df[~df["postprocessed"]]
    df["base_model"] = df["model"].str.replace(r"_pp$", "", regex=True)
    if dataset == "MMS2":
        df["patient_id"] = df["patient_id"].astype(int).map("{:03d}".format)
    df["dataset"] = dataset
    df["model_name"] = df["base_model"].map(MAIN_MODELS)
    for key, _ in STRUCTURES:
        df[f"hd95_{key}"] = pd.to_numeric(df[f"hd95_{key}"], errors="coerce")
    return df.merge(load_metadata(dataset), on="patient_id", how="left", validate="many_to_one")


def load_metadata(dataset: str) -> pd.DataFrame:
    """Vendor, pathology and split per subject."""
    if dataset == "ACDC":
        rows = []
        for cfg in sorted((DATA_ROOT / "ACDC" / "database" / "testing").glob("patient*/Info.cfg")):
            info = dict(line.strip().split(": ", 1) for line in cfg.read_text().splitlines() if ": " in line)
            rows.append({"patient_id": cfg.parent.name, "pathology": info["Group"], "vendor": "Siemens", "split": "test"})
        return pd.DataFrame(rows)
    if dataset == "MMS1":
        meta = pd.read_csv(DATA_ROOT / "M&Ms" / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
        meta = meta.rename(columns={"External code": "patient_id", "VendorName": "vendor", "Pathology": "pathology"})
        meta["split"] = "test"
        return meta[["patient_id", "vendor", "pathology", "split"]]
    # The released CSV carries ~1M empty spreadsheet rows; keep the 360 subjects.
    meta = pd.read_csv(DATA_ROOT / "M&Ms_2" / "MnM2" / "dataset_information.csv", dtype=str).dropna(subset=["SUBJECT_CODE"])
    meta["SUBJECT_CODE"] = meta["SUBJECT_CODE"].astype(int)
    assert len(meta) == 360, len(meta)
    meta["patient_id"] = meta["SUBJECT_CODE"].map("{:03d}".format)
    meta["vendor"] = meta["VENDOR"].map({"SIEMENS": "Siemens", "Philips Medical Systems": "Philips", "GE MEDICAL SYSTEMS": "GE"})
    meta["pathology"] = meta["DISEASE"]
    meta["split"] = np.where(meta["SUBJECT_CODE"] <= 160, "train", np.where(meta["SUBJECT_CODE"] <= 200, "validation", "test"))
    return meta[["patient_id", "vendor", "pathology", "split"]]


def primary_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Held-out test subjects only (M&Ms-2 201-360; all subjects for ACDC and M&Ms-1)."""
    return df[df["split"] == "test"]


def in_distribution(model_name: str, dataset: str) -> bool:
    return dataset in TRAINED_ON.get(model_name, set())


def patient_level(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean over ED and ES per patient, so each patient counts once in tests."""
    return df.groupby(["model_name", "patient_id"], as_index=False)[metric].mean()


def bootstrap_ci(values, statistic=np.mean, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = statistic(rng.choice(values, size=(n_boot, values.size), replace=True), axis=1)
    return tuple(np.quantile(boots, [alpha / 2, 1 - alpha / 2]))


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_model_tests(wide: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Friedman test across models, then pairwise Wilcoxon signed-rank with Holm.

    `wide` has one row per patient and one column per model (complete cases).
    """
    wide = wide.dropna()
    friedman_p = stats.friedmanchisquare(*[wide[c] for c in wide.columns]).pvalue
    pairs = []
    cols = list(wide.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            diff = wide[a] - wide[b]
            p = stats.wilcoxon(wide[a], wide[b]).pvalue if np.any(diff != 0) else 1.0
            pairs.append({"model_a": a, "model_b": b, "median_diff": float(np.median(diff)), "p": p})
    pairs = pd.DataFrame(pairs)
    pairs["p_holm"] = holm(pairs["p"])
    return friedman_p, pairs


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"
