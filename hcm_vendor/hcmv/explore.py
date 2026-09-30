"""Data QC and exploratory report (T22): D6 exclusions, Dice QC, missingness,
univariate AUCs, vendor effects, correlation structure and PCA."""

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import output_dir, repo_path
from .features import feature_columns, feature_family
from .figures import (
    VENDOR_ORDER,
    plot_auc_agreement,
    plot_auc_heatmap,
    plot_clinical_distributions,
    plot_feature_clustermap,
    plot_pca,
    plot_texture_vendor_effect,
)
from .qc import md_table, vendor_effect
from .stats import fast_auc

FAMILIES = ("clinical", "shape", "texture")
KEY_CLINICAL = [
    "ed_lv_volume_ml", "ed_myocardial_mass_g", "ed_wall_thickness_max_mm",
    "ed_wall_thickness_mean_mm", "lvef_pct", "mass_to_volume_g_per_ml",
]

# D6: failed cases are defined by automated criteria only (Dice is QC, never a filter).
MIN_LV_EDV_ML = 20.0


def apply_d6(table: pd.DataFrame) -> pd.DataFrame:
    """Return ``subject_id, reason`` for every subject that fails the D6 criteria."""
    features = feature_columns(table, FAMILIES)
    reasons = {}

    def flag(mask, reason):
        for subject in table.index[mask]:
            reasons.setdefault(subject, []).append(reason)

    flag(table[features].isna().any(axis=1), "missing or NaN features")
    for phase in ("ed", "es"):
        for structure in ("lv", "myo"):
            col = f"{phase}_{structure}_volume_ml"
            flag(~(table[col] > 0), f"empty {structure.upper()} at {phase.upper()}")
    flag(table["ed_lv_volume_ml"] < MIN_LV_EDV_ML, f"LV EDV < {MIN_LV_EDV_ML:g} ml")
    for ef in ("lvef_pct", "rvef_pct"):
        flag((table[ef] < 0) | (table[ef] > 100), f"{ef} outside 0-100%")
    rows = [{"subject_id": s, "reason": "; ".join(r)} for s, r in reasons.items()]
    return pd.DataFrame(rows, columns=["subject_id", "reason"])


def load_dice(path, cohort: pd.DataFrame, dataset: str, model_tag: str) -> pd.DataFrame:
    """nnFormer Dice rows from SORAT's long-format aggregated metrics, joined to the cohort.

    SORAT writes M&Ms-2 IDs unpadded (``9``) while the cohort uses ``009``.
    """
    metrics = pd.read_csv(path, dtype={"patient_id": str})
    metrics = metrics[metrics["model"] == model_tag].copy()
    if dataset == "mms2":
        metrics["source_id"] = metrics["patient_id"].str.zfill(3)
    else:
        metrics["source_id"] = metrics["patient_id"]
    subset = cohort[cohort["dataset"] == dataset]
    return metrics.merge(subset[["subject_id", "source_id", "dataset", "vendor", "disease"]],
                         on="source_id", how="inner")


def dice_summary(dice: pd.DataFrame) -> pd.DataFrame:
    grouped = dice.groupby(["dataset", "vendor", "disease", "frame_tag"])
    summary = grouped[["dice_lv", "dice_myo", "dice_rv"]].median().round(3)
    summary.insert(0, "n", grouped["subject_id"].nunique())
    return summary


def univariate_auc(table: pd.DataFrame, columns, by="vendor") -> pd.DataFrame:
    """HCM-vs-NOR AUC of each raw feature within each group. 0.5 = no signal;
    values below 0.5 mean the feature is lower in HCM."""
    rows = []
    for group, frame in table.groupby(by):
        for col in columns:
            ok = frame[col].notna()
            rows.append({by: group, "feature": col, "family": feature_family(col),
                         "auc": fast_auc(frame.loc[ok, "y"], frame.loc[ok, col])})
    out = pd.DataFrame(rows)
    out["separation"] = (out["auc"] - 0.5).abs() + 0.5
    return out


def auc_agreement(aucs: pd.DataFrame, a: str = "Siemens", b: str = "Philips") -> pd.DataFrame:
    """Per family: do features separate HCM from NOR the same way in vendor a and b?"""
    wide = aucs.pivot_table(index=["feature", "family"], columns="vendor", values="auc").reset_index()
    rows = []
    for family, g in wide.groupby("family"):
        flip = (g[a] - 0.5) * (g[b] - 0.5) < 0
        strong = ((g[a] - 0.5).abs() > 0.15) & ((g[b] - 0.5).abs() > 0.15)
        rows.append({"family": family, "features": len(g), f"auc_r_{a}_vs_{b}": np.corrcoef(g[a], g[b])[0, 1],
                     "direction_flips": int(flip.sum()), "strong_flips": int((flip & strong).sum())})
    return pd.DataFrame(rows).set_index("family")


def _standardized(table: pd.DataFrame, columns) -> np.ndarray:
    X = SimpleImputer(strategy="median").fit_transform(table[columns])
    X = X[:, np.nanstd(X, axis=0) > 0]
    return StandardScaler().fit_transform(X)


def correlation_order(table: pd.DataFrame, columns):
    """|Pearson r| matrix of standardized features in hierarchical-clustering order."""
    kept = [c for c in columns if table[c].std() > 0]
    corr = np.clip(pd.DataFrame(_standardized(table, kept)).corr().abs().to_numpy(), 0, 1)
    link = hierarchy.linkage(1 - corr[np.triu_indices(len(corr), 1)], method="average")
    order = hierarchy.leaves_list(link)
    return corr[np.ix_(order, order)], [kept[i] for i in order]


def pca_scores(table: pd.DataFrame, columns):
    X = _standardized(table, columns)
    pca = PCA(n_components=2, random_state=0).fit(X)
    return pca.transform(X), pca.explained_variance_ratio_


def qc_report(config: dict) -> str:
    tables_dir = output_dir(config, "tables")
    out = output_dir(config, "qc", "t22")
    cohort = pd.read_parquet(tables_dir / "cohort.parquet")
    tag = config["segmentation_model_tag"].replace("__", "-")
    load = lambda dataset, cfg: pd.read_parquet(tables_dir / f"features_{dataset}_{tag}_{cfg}.parquet")  # noqa: E731
    tables = {(d, c): load(d, c) for d in ("mms2", "acdc") for c in ("norm", "raw")}
    lines = ["# T22: data QC and exploratory report", "",
             "Source: nnFormer (fold 0) features. Figures and CSVs are in this folder.", ""]

    # --- D6 exclusions
    excluded = []
    for (dataset, cfg), table in tables.items():
        failed = apply_d6(table)
        failed.insert(1, "dataset", dataset)
        failed.insert(2, "config", cfg)
        excluded.append(failed)
    excluded = pd.concat(excluded, ignore_index=True)
    excluded.to_csv(out / "d6_excluded.csv", index=False)
    lines += ["## D6 failed-case exclusions", "",
              "Criteria: missing/NaN features, empty LV or MYO at ED/ES, "
              f"LV EDV < {MIN_LV_EDV_ML:g} ml, LVEF or RVEF outside 0–100%. Dice is QC only.", ""]
    if excluded.empty:
        lines.append("**No subject fails D6** in any dataset or config, so nothing is excluded.")
    else:
        lines.append(md_table(excluded.set_index("subject_id")))
    lines.append("")

    # --- Dice QC
    dice = []
    for dataset in ("mms2", "acdc"):
        path = repo_path(config, config["paths"]["sorat_results"][dataset]) / "comparison" / "aggregated_metrics.csv"
        dice.append(load_dice(path, cohort, dataset, config["segmentation_model_tag"]))
    dice = pd.concat(dice, ignore_index=True)
    dice.to_csv(out / "dice_per_subject.csv", index=False)
    dsum = dice_summary(dice)
    dsum.to_csv(out / "dice_by_vendor.csv")
    missing = sorted(set(cohort["subject_id"]) - set(dice["subject_id"]))
    lines += ["## nnFormer Dice by vendor (QC only, median)", "", md_table(dsum), "",
              f"Subjects without a Dice row: {len(missing)}" + (f" ({', '.join(missing)})" if missing else ""), ""]

    # --- Missingness
    miss_rows = []
    for (dataset, cfg), table in tables.items():
        feats = feature_columns(table, FAMILIES)
        na = table[feats].isna().sum()
        miss_rows.append({"dataset": dataset, "config": cfg, "features": len(feats),
                          "cells_missing": int(na.sum()), "features_with_missing": int((na > 0).sum())})
    miss = pd.DataFrame(miss_rows)
    miss.to_csv(out / "missingness.csv", index=False)
    lines += ["## Missingness", "", md_table(miss.set_index("dataset")), ""]

    # --- Distributions (M&Ms-2 norm)
    mms2 = tables[("mms2", "norm")]
    plot_clinical_distributions(mms2, KEY_CLINICAL, out / "clinical_distributions.png")
    med = mms2.groupby(["vendor", "disease"])[KEY_CLINICAL].median().round(1)
    med.to_csv(out / "clinical_medians.csv")
    lines += ["## Key clinical features, median by vendor × disease (M&Ms-2)", "",
              "Figure: `clinical_distributions.png`.", "", md_table(med), ""]

    # --- Univariate AUCs
    feats = feature_columns(mms2, FAMILIES)
    aucs = univariate_auc(mms2, feats)
    aucs.to_csv(out / "univariate_auc_by_vendor.csv", index=False)
    counts = {v: mms2[mms2.vendor == v]["disease"].value_counts().to_dict() for v in VENDOR_ORDER}
    top = plot_auc_heatmap(aucs, counts, out / "univariate_auc_top25.png")
    plot_auc_agreement(aucs, counts, out / "auc_agreement_siemens_philips.png")
    fam = aucs.groupby(["family", "vendor"])["separation"].median().unstack()[VENDOR_ORDER].round(3)
    wide = aucs.pivot_table(index="feature", columns="vendor", values="auc")
    agreement = auc_agreement(aucs).round(2)
    agreement.to_csv(out / "auc_agreement_siemens_philips.csv")
    lines += ["## Univariate HCM-vs-NOR AUC within vendor (M&Ms-2, norm)", "",
              "`separation` = max(AUC, 1 − AUC). Figures: `univariate_auc_top25.png`, "
              "`auc_agreement_siemens_philips.png`.", "",
              "Median separation by family:", "", md_table(fam), "",
              "Top 10 features by mean Siemens/Philips separation:", "",
              md_table(wide.loc[top[:10], VENDOR_ORDER].round(3)), "",
              "Do features separate HCM from NOR in the same direction on Siemens and Philips? "
              "`strong_flips` = opposite directions with |AUC − 0.5| > 0.15 on both vendors.", "",
              md_table(agreement), "",
              "Scanner models by disease (a vendor's HCM and NOR can come from different scanners):", "",
              md_table(pd.crosstab([mms2["vendor"], mms2["scanner"]], mms2["disease"])), ""]

    # --- Vendor effect among NOR, per feature and family
    effect = vendor_effect(mms2, feats)
    effect["family"] = effect["feature"].map(feature_family)
    effect.to_csv(out / "vendor_effect_nor.csv", index=False)
    fam_eff = effect.groupby("family").agg(features=("eta2", "size"), median_eta2=("eta2", "median"),
                                           share_p05=("p", lambda p: (p < 0.05).mean())).round(3)
    lines += ["## Vendor effect among M&Ms-2 NOR (Kruskal–Wallis η², norm)", "", md_table(fam_eff), "",
              "Largest vendor effects:", "",
              md_table(effect.nlargest(8, "eta2").set_index("feature")[["family", "eta2", "p"]]
                       .assign(eta2=lambda f: f["eta2"].round(3), p=lambda f: f["p"].map("{:.1e}".format))), ""]

    # --- Raw vs norm texture
    tex_effects = {cfg: vendor_effect(tables[("mms2", cfg)], feature_columns(tables[("mms2", cfg)], ["texture"]))
                   for cfg in ("raw", "norm")}
    plot_texture_vendor_effect(tex_effects, out / "texture_vendor_effect_raw_vs_norm.png",
                               masks="nnFormer (automatic) segmentations")
    lines += ["## Raw vs normalized texture (nnFormer masks, M&Ms-2 NOR)", "",
              "| config | median η² | share p < 0.05 |", "|---|---|---|"]
    for cfg, frame in tex_effects.items():
        lines.append(f"| {cfg} | {frame['eta2'].median():.2f} | {(frame['p'] < 0.05).mean():.0%} |")
    lines += ["", "Figure: `texture_vendor_effect_raw_vs_norm.png`.", ""]

    # --- Correlation structure and PCA
    corr, ordered = correlation_order(mms2, feats)
    plot_feature_clustermap(corr, ordered, out / "feature_correlation_clustermap.png", len(mms2))
    raw = tables[("mms2", "raw")]
    panels, pca_rows = [], []
    for title, table, columns in (
        ("All 129 features, normalized", mms2, feats),
        ("84 texture features, normalized", mms2, feature_columns(mms2, ["texture"])),
        ("84 texture features, raw (no normalization)", raw, feature_columns(raw, ["texture"])),
    ):
        Z, var = pca_scores(table, columns)
        panels.append((title, table, Z, var))
        pca_rows.append({"panel": title, "pc1_var": var[0], "pc2_var": var[1]})
    plot_pca(panels, out / "pca_vendor_disease.png",
             "PCA of M&Ms-2 nnFormer features (standardized; n=135)\n"
             "Left: do subjects cluster by scanner vendor? Right: by diagnosis?")
    pca = pd.DataFrame(pca_rows).round(3)
    lines += ["## Correlation structure and PCA (M&Ms-2)", "",
              "Figures: `feature_correlation_clustermap.png`, `pca_vendor_disease.png`.", "",
              md_table(pca.set_index("panel")), ""]

    report = "\n".join(lines) + "\n"
    (out / "qc_summary.md").write_text(report)
    return report
