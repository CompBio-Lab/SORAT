"""Feature validation and QC reports (T12, T22)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from .config import output_dir, repo_path  # noqa: E402
from .features import build_feature_table, feature_columns  # noqa: E402

VENDOR_ORDER = ["Siemens", "Philips", "GE"]
WT_COLUMNS = ["ed_wall_thickness_max_mm", "ed_wall_thickness_p95_mm", "ed_wall_thickness_mean_mm"]
CLINICAL_AGREEMENT = [
    "ed_lv_volume_ml", "es_lv_volume_ml", "ed_rv_volume_ml", "ed_myocardial_mass_g",
    "ed_wall_thickness_max_mm", "ed_wall_thickness_mean_mm", "lvef_pct", "rvef_pct",
]


def md_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (index included)."""
    frame = frame.reset_index()
    header = ["" if str(c).startswith("index") else " / ".join(map(str, c)) if isinstance(c, tuple) else str(c)
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
                "pearson_r": stats.pearsonr(a[ok], b[ok])[0],
                "bias": diff.mean(), "loa_low": diff.mean() - 1.96 * diff.std(),
                "loa_high": diff.mean() + 1.96 * diff.std(),
            })
    return pd.DataFrame(rows)


def plot_wall_thickness(table: pd.DataFrame, path) -> None:
    data = table.assign(group=group_label(table))
    groups = sorted(data["group"].unique())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot([data.loc[data["group"] == g, "ed_wall_thickness_max_mm"].dropna() for g in groups],
               labels=groups, showfliers=True)
    ax.axhline(15, color="crimson", linestyle="--", linewidth=1, label="15 mm HCM criterion")
    ax.set_ylabel("ED max wall thickness (mm)")
    ax.set_title("Ground-truth masks: max wall thickness by dataset, vendor and disease")
    ax.legend(loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vendor_effect(effects: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 21)
    for name, frame in effects.items():
        ax.hist(frame["eta2"], bins=bins, alpha=0.55, label=f"{name} (median {frame['eta2'].median():.2f})")
    ax.set_xlabel("Vendor effect size η² among M&Ms-2 NOR (Kruskal–Wallis)")
    ax.set_ylabel("Texture features")
    ax.set_title("Texture vendor signal: raw vs normalized radiomics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def validate_gt(config: dict, pred_root: str = None) -> str:
    """T12 report. ``pred_root`` (e.g. T14 nnFormer features) adds pred-vs-GT agreement."""
    cohort = pd.read_parquet(output_dir(config, "tables") / "cohort.parquet")
    out = output_dir(config, "qc", "t12_validation")
    gt_root = "results_hcm_vendor/features/gt"
    gt = {cfg: load_tables(config, cohort, "gt", gt_root, cfg) for cfg in ("norm", "raw")}

    wt = wall_thickness_summary(gt["norm"])
    wt.to_csv(out / "gt_wall_thickness_by_group.csv")
    plot_wall_thickness(gt["norm"], out / "gt_wall_thickness_by_group.png")

    effects = {cfg: vendor_effect(gt[cfg], feature_columns(gt[cfg], ["texture"])) for cfg in ("raw", "norm")}
    for cfg, frame in effects.items():
        frame.to_csv(out / f"gt_texture_vendor_effect_{cfg}.csv", index=False)
    plot_vendor_effect(effects, out / "gt_texture_vendor_effect.png")

    lines = [
        "# T12 validation: ground-truth-mask features", "",
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
        bias = agree.pivot_table(index="feature", columns=["dataset", "vendor"], values="bias").round(1)
        lines += ["", "## nnFormer vs ground truth: Pearson r", "", md_table(pivot),
                  "", "## nnFormer vs ground truth: mean bias (pred − GT)", "", md_table(bias)]

    report = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(report)
    return report
