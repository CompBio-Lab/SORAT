"""Feature validation and QC reports."""

import numpy as np
import pandas as pd
from scipy import stats

from .config import output_dir, repo_path
from .features import build_feature_table, feature_columns
from .figures import VENDOR_ORDER, plot_gt_wall_thickness, plot_texture_vendor_effect

WT_COLUMNS = ["ed_wall_thickness_max_mm", "ed_wall_thickness_p95_mm", "ed_wall_thickness_mean_mm"]
CLINICAL_AGREEMENT = [
    "ed_lv_volume_ml", "es_lv_volume_ml", "ed_rv_volume_ml", "ed_myocardial_mass_g",
    "ed_wall_thickness_max_mm", "ed_wall_thickness_mean_mm", "lvef_pct", "rvef_pct",
]


def md_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (index included)."""
    frame = frame.reset_index()
    header = ["" if str(c).startswith("index") else " / ".join(str(p) for p in c if str(p)) if isinstance(c, tuple) else str(c)
              for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in frame.itertuples(index=False):
        cells = ["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_tables(config: dict, cohort: pd.DataFrame, source: str, root: str, cfg: str) -> pd.DataFrame:
    """Concatenate the mms2 and acdc tables for one source/config."""
    tables = []
    for dataset in ("mms2", "acdc"):
        table, _ = build_feature_table(repo_path(config, root) / dataset / cfg, source, cohort, dataset)
        tables.append(table)
    return pd.concat(tables)


def group_label(table: pd.DataFrame) -> pd.Series:
    dataset = table["dataset"].map({"mms2": "M&Ms-2", "acdc": "ACDC"})
    return dataset + " " + table["vendor"] + " " + table["disease"]


def wall_thickness_summary(table: pd.DataFrame) -> pd.DataFrame:
    grouped = table.assign(group=group_label(table)).groupby("group")[WT_COLUMNS]
    summary = grouped.median().round(1).add_suffix("_median")
    summary.insert(0, "n", grouped.size())
    return summary


def vendor_effect(table: pd.DataFrame, columns) -> pd.DataFrame:
    """Kruskal-Wallis vendor effect on M&Ms-2 NOR subjects, with eta-squared (H-based)."""
    nor = table[(table["dataset"] == "mms2") & (table["disease"] == "NOR")]
    rows = []
    for col in columns:
        samples = [nor.loc[nor["vendor"] == v, col].dropna() for v in VENDOR_ORDER]
        samples = [s for s in samples if len(s) > 1]
        if len(samples) < 2 or all(s.nunique() <= 1 for s in samples):
            continue
        h, p = stats.kruskal(*samples)
        n, k = sum(len(s) for s in samples), len(samples)
        rows.append({"feature": col, "H": h, "p": p, "eta2": max(0.0, (h - k + 1) / (n - k))})
    return pd.DataFrame(rows)


def icc_3_1(a, b) -> float:
    """ICC(3,1), two-way mixed, consistency, single rater (Shrout & Fleiss), for two raters."""
    x = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((x - grand) ** 2).sum() - ss_rows - ss_cols
    ms_rows, ms_err = ss_rows / (n - 1), ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err
    return float((ms_rows - ms_err) / denom) if denom > 0 else float("nan")


def agreement(pred: pd.DataFrame, gt: pd.DataFrame, columns) -> pd.DataFrame:
    """Per-vendor agreement between predicted-mask and GT-mask features."""
    joined = pred[["vendor", "dataset"] + columns].join(gt[columns], rsuffix="_gt", how="inner")
    rows = []
    for (dataset, vendor), group in joined.groupby(["dataset", "vendor"]):
        for col in columns:
            a, b = group[col].astype(float), group[f"{col}_gt"].astype(float)
            ok = a.notna() & b.notna()
            if ok.sum() < 3:
                continue
            diff = a[ok] - b[ok]
            rows.append({
                "dataset": dataset, "vendor": vendor, "feature": col, "n": int(ok.sum()),
                "pearson_r": stats.pearsonr(a[ok], b[ok])[0], "icc_3_1": icc_3_1(a[ok], b[ok]),
                "bias": diff.mean(), "loa_low": diff.mean() - 1.96 * diff.std(),
                "loa_high": diff.mean() + 1.96 * diff.std(),
            })
    return pd.DataFrame(rows)


def validate_gt(config: dict, pred_root: str = None) -> str:
    """Ground-truth validation report. ``pred_root`` (e.g. re-extracted nnFormer features) adds pred-vs-GT agreement."""
    cohort = pd.read_parquet(output_dir(config, "tables") / "cohort.parquet")
    out = output_dir(config, "qc", "t12_validation")
    gt_root = "results_hcm_vendor/features/gt"
    gt = {cfg: load_tables(config, cohort, "gt", gt_root, cfg) for cfg in ("norm", "raw")}

    wt = wall_thickness_summary(gt["norm"])
    wt.to_csv(out / "gt_wall_thickness_by_group.csv")
    plot_gt_wall_thickness(gt["norm"], out / "gt_wall_thickness_by_group.png")

    effects = {cfg: vendor_effect(gt[cfg], feature_columns(gt[cfg], ["texture"])) for cfg in ("raw", "norm")}
    for cfg, frame in effects.items():
        frame.to_csv(out / f"gt_texture_vendor_effect_{cfg}.csv", index=False)
    plot_texture_vendor_effect(effects, out / "gt_texture_vendor_effect.png",
                               masks="Manual (ground-truth) segmentations")

    lines = [
        "# Validation: ground-truth-mask features", "",
        "## ED wall thickness (median mm) by group, `norm` tables", "",
        md_table(wt), "",
        "## Texture vendor effect among M&Ms-2 NOR (ground-truth masks)", "",
        "| config | texture features | median η² | share with p < 0.05 |", "|---|---|---|---|",
    ]
    for cfg, frame in effects.items():
        lines.append(f"| {cfg} | {len(frame)} | {frame['eta2'].median():.2f} | {(frame['p'] < 0.05).mean():.0%} |")

    if pred_root:
        pred = load_tables(config, cohort, config["segmentation_model_tag"], pred_root, "norm")
        agree = agreement(pred, gt["norm"], CLINICAL_AGREEMENT)
        agree.to_csv(out / "pred_vs_gt_agreement.csv", index=False)
        pivot = agree.pivot_table(index="feature", columns=["dataset", "vendor"], values="pearson_r").round(2)
        icc = agree.pivot_table(index="feature", columns=["dataset", "vendor"], values="icc_3_1").round(2)
        bias = agree.pivot_table(index="feature", columns=["dataset", "vendor"], values="bias").round(1)
        lines += ["", "## nnFormer vs ground truth: Pearson r", "", md_table(pivot),
                  "", "## nnFormer vs ground truth: ICC(3,1)", "", md_table(icc),
                  "", "## nnFormer vs ground truth: mean bias (pred − GT)", "", md_table(bias)]

    report = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(report)
    return report


def _expected_settings(options: dict) -> dict:
    """PyRadiomics settings dict a feature config should record (mirrors extract_features)."""
    options = options or {}
    settings = {}
    if options.get("normalize"):
        settings["normalize"] = True
        if options.get("normalize_scale") is not None:
            settings["normalizeScale"] = float(options["normalize_scale"])
    if options.get("remove_outliers") is not None:
        settings["removeOutliers"] = float(options["remove_outliers"])
    if options.get("bin_count") is not None:
        settings["binCount"] = int(options["bin_count"])
    if options.get("bin_width") is not None:
        settings["binWidth"] = float(options["bin_width"])
    if options.get("resample_spacing"):
        settings["resampledPixelSpacing"] = [float(v) for v in str(options["resample_spacing"]).split(",")]
    if options.get("force2d"):
        settings["force2D"] = True
        settings["force2Ddimension"] = int(options.get("force2d_dimension", 0))
    if options.get("intensity_reference"):
        settings["intensityReference"] = options["intensity_reference"]
        settings["intensityReferenceScale"] = float(options.get("intensity_reference_scale") or 100.0)
    return settings


def check_features(config: dict, root: str, source: str) -> pd.DataFrame:
    """Feature-extraction acceptance checks for every dataset x feature config under ``root``."""
    import json

    from .features import read_phase_csvs

    cohort = pd.read_parquet(output_dir(config, "tables") / "cohort.parquet")
    rows = []
    for dataset in ("mms2", "acdc"):
        expected_ids = set(cohort.loc[cohort["dataset"] == dataset, "source_id"])
        for cfg, options in config["feature_configs"].items():
            directory = repo_path(config, root) / dataset / cfg
            files = sorted(directory.glob(f"*_{source}_*_features.csv")) if directory.exists() else []
            row = {"dataset": dataset, "config": cfg, "files": len(files),
                   "expected_files": 2 * len(expected_ids),
                   "empty_files": sum(p.stat().st_size == 0 for p in files)}
            if files and not row["empty_files"]:
                long = read_phase_csvs(directory, source)
                settings = {json.dumps(json.loads(s), sort_keys=True) for s in long["radiomics_settings"].dropna()}
                want = json.dumps(_expected_settings(options), sort_keys=True)
                model_cols = [c for c in long.columns if c.startswith(("radiomics_original", "lv_", "rv_", "myo_", "wall_"))]
                ed = long[long["phase"] == "ED"]
                row.update({
                    "subjects": long["source_id"].nunique(),
                    "missing_subjects": len(expected_ids - set(long["source_id"])),
                    "radiomics_errors": int(long["radiomics_error"].notna().sum()),
                    "all_nan_columns": int(long[model_cols].isna().all().sum()),
                    "settings_match": settings == {want},
                    "ed_wt_max_median": round(float(ed["wall_thickness_max_mm"].median()), 1),
                })
            row["ok"] = bool(
                row["files"] == row["expected_files"] and row["empty_files"] == 0
                and row.get("missing_subjects") == 0 and row.get("radiomics_errors") == 0
                and row.get("all_nan_columns") == 0 and row.get("settings_match")
            )
            rows.append(row)
    return pd.DataFrame(rows)
