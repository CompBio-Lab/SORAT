"""Data QC and exploratory report (T22): D6 exclusions, Dice QC, missingness,
univariate AUCs, vendor effects, correlation structure and PCA."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
from scipy.cluster import hierarchy  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .config import output_dir, repo_path  # noqa: E402
from .features import feature_columns, feature_family  # noqa: E402
from .qc import VENDOR_ORDER, md_table, vendor_effect  # noqa: E402
from .stats import fast_auc  # noqa: E402

FAMILIES = ("clinical", "shape", "texture")
KEY_CLINICAL = [
    "ed_lv_volume_ml", "ed_myocardial_mass_g", "ed_wall_thickness_max_mm",
    "ed_wall_thickness_mean_mm", "lvef_pct", "mass_to_volume_g_per_ml",
]
VENDOR_COLOURS = {"Siemens": "#1f77b4", "Philips": "#ff7f0e", "GE": "#2ca02c"}
DISEASE_COLOURS = {"NOR": "#4c72b0", "HCM": "#c44e52"}

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


def plot_distributions(table: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    groups = [(v, d) for v in VENDOR_ORDER for d in ("NOR", "HCM")]
    for ax, col in zip(axes.flat, KEY_CLINICAL):
        data = [table.loc[(table.vendor == v) & (table.disease == d), col].dropna() for v, d in groups]
        box = ax.boxplot(data, patch_artist=True, showfliers=True)
        for patch, (_, d) in zip(box["boxes"], groups):
            patch.set_facecolor(DISEASE_COLOURS[d])
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([f"{v}\n{d}" for v, d in groups], fontsize=8)
        ax.set_title(col, fontsize=10)
    fig.suptitle("M&Ms-2 nnFormer features by vendor and disease (norm)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_auc_heatmap(aucs: pd.DataFrame, path, top: int = 25) -> list:
    pivot = aucs.pivot_table(index="feature", columns="vendor", values="auc")[VENDOR_ORDER]
    order = aucs[aucs.vendor.isin(["Siemens", "Philips"])].groupby("feature")["separation"].mean()
    features = order.sort_values(ascending=False).index[:top].tolist()
    data = pivot.loc[features]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(features) + 1.5))
    im = ax.imshow(data.to_numpy(), cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(VENDOR_ORDER)))
    ax.set_xticklabels([f"{v}" for v in VENDOR_ORDER])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f.replace("radiomics_original_", "") for f in features], fontsize=7)
    for i in range(len(features)):
        for j in range(len(VENDOR_ORDER)):
            ax.text(j, i, f"{data.iat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="HCM-vs-NOR AUC")
    ax.set_title(f"Top {top} features by mean separation (Siemens, Philips)\nGE has only 3 HCM", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return features


def plot_clustermap(table: pd.DataFrame, columns, path) -> None:
    corr = np.clip(pd.DataFrame(_standardized(table, columns)).corr().abs().to_numpy(), 0, 1)
    kept = [c for c in columns if table[c].std() > 0]
    link = hierarchy.linkage(1 - corr[np.triu_indices(len(corr), 1)], method="average")
    order = hierarchy.leaves_list(link)
    colours = {"clinical": "#2ca02c", "shape": "#9467bd", "texture": "#8c564b"}
    fig, ax = plt.subplots(figsize=(10, 9))
    divider = make_axes_locatable(ax)
    ax_bar = divider.append_axes("top", size="3%", pad=0.05)
    ax_cbar = divider.append_axes("right", size="3%", pad=0.1)
    fams = [feature_family(kept[i]) for i in order]
    ax_bar.imshow([[matplotlib.colors.to_rgb(colours[f]) for f in fams]], aspect="auto")
    ax_bar.set_axis_off()
    ax_bar.set_title("|Pearson r| between features, hierarchically ordered "
                     "(bar: green clinical, purple shape, brown texture)", fontsize=9)
    im = ax.imshow(corr[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, cax=ax_cbar)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca(tables: dict, columns_by_panel: dict, path) -> pd.DataFrame:
    """One row per panel (table, columns); left coloured by vendor, right by disease."""
    rows = []
    fig, axes = plt.subplots(len(columns_by_panel), 2, figsize=(10, 4.2 * len(columns_by_panel)), squeeze=False)
    for (title, (key, columns)), row_axes in zip(columns_by_panel.items(), axes):
        table = tables[key]
        scores = PCA(n_components=2, random_state=0).fit(_standardized(table, columns))
        Z = scores.transform(_standardized(table, columns))
        var = scores.explained_variance_ratio_
        for ax, (by, colours) in zip(row_axes, (("vendor", VENDOR_COLOURS), ("disease", DISEASE_COLOURS))):
            for label, colour in colours.items():
                m = (table[by] == label).to_numpy()
                ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.75, c=colour, label=f"{label} (n={m.sum()})")
            ax.set_xlabel(f"PC1 ({var[0]:.0%})")
            ax.set_ylabel(f"PC2 ({var[1]:.0%})")
            ax.set_title(f"{title}: coloured by {by}", fontsize=10)
            ax.legend(fontsize=7)
        rows.append({"panel": title, "pc1_var": var[0], "pc2_var": var[1]})
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_texture_norm_effect(effects: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 21)
    for name, frame in effects.items():
        ax.hist(frame["eta2"], bins=bins, alpha=0.55, label=f"{name} (median {frame['eta2'].median():.2f})")
    ax.set_xlabel("Vendor effect η² among M&Ms-2 NOR (Kruskal–Wallis)")
    ax.set_ylabel("Texture features")
    ax.set_title("nnFormer masks: texture vendor signal, raw vs normalized")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    plot_distributions(mms2, out / "clinical_distributions.png")
    med = mms2.groupby(["vendor", "disease"])[KEY_CLINICAL].median().round(1)
    med.to_csv(out / "clinical_medians.csv")
    lines += ["## Key clinical features, median by vendor × disease (M&Ms-2)", "",
              "Figure: `clinical_distributions.png`.", "", md_table(med), ""]

    # --- Univariate AUCs
    feats = feature_columns(mms2, FAMILIES)
    aucs = univariate_auc(mms2, feats)
    aucs.to_csv(out / "univariate_auc_by_vendor.csv", index=False)
    top = plot_auc_heatmap(aucs, out / "univariate_auc_top25.png")
    fam = aucs.groupby(["family", "vendor"])["separation"].median().unstack()[VENDOR_ORDER].round(3)
    wide = aucs.pivot_table(index="feature", columns="vendor", values="auc")
    agreement = auc_agreement(aucs).round(2)
    agreement.to_csv(out / "auc_agreement_siemens_philips.csv")
    lines += ["## Univariate HCM-vs-NOR AUC within vendor (M&Ms-2, norm)", "",
              "`separation` = max(AUC, 1 − AUC). Figure: `univariate_auc_top25.png`.", "",
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
    plot_texture_norm_effect(tex_effects, out / "texture_vendor_effect_raw_vs_norm.png")
    lines += ["## Raw vs normalized texture (nnFormer masks, M&Ms-2 NOR)", "",
              "| config | median η² | share p < 0.05 |", "|---|---|---|"]
    for cfg, frame in tex_effects.items():
        lines.append(f"| {cfg} | {frame['eta2'].median():.2f} | {(frame['p'] < 0.05).mean():.0%} |")
    lines += ["", "Figure: `texture_vendor_effect_raw_vs_norm.png`.", ""]

    # --- Correlation structure and PCA
    plot_clustermap(mms2, feats, out / "feature_correlation_clustermap.png")
    panels = {
        "All features (norm)": (("mms2", "norm"), feats),
        "Texture only (norm)": (("mms2", "norm"), feature_columns(mms2, ["texture"])),
        "Texture only (raw)": (("mms2", "raw"), feature_columns(tables[("mms2", "raw")], ["texture"])),
    }
    pca = plot_pca(tables, panels, out / "pca_vendor_disease.png").round(3)
    lines += ["## Correlation structure and PCA (M&Ms-2)", "",
              "Figures: `feature_correlation_clustermap.png`, `pca_vendor_disease.png`.", "",
              md_table(pca.set_index("panel")), ""]

    report = "\n".join(lines) + "\n"
    (out / "qc_summary.md").write_text(report)
    return report
