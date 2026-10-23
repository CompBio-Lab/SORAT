"""Segmentation accuracy per dataset, model, structure and phase.

Outputs (tables/ and figures/):
  seg_accuracy_main.csv / .md      Dice mean±SD [95% CI] and HD95 median [IQR], held-out test subjects
  seg_accuracy_by_phase.csv        the same split by ED and ES
  seg_accuracy_all_variants.csv    every CineMA seed and ensemble (supplement)
  seg_pairwise_tests.csv           Friedman + Holm-adjusted Wilcoxon signed-rank between models
  seg_accuracy_by_vendor.csv       Dice by vendor (and pathology) for the main models
  fig_accuracy_by_dataset.{png,pdf}  Figure 2: patient-level DSC and HD95 by dataset and structure
  fig_vendor_heatmap.{png,pdf}     mean Dice by model and vendor

Run: python analysis/seg_accuracy.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C


def load_all(include_all_variants: bool = False) -> pd.DataFrame:
    frames = [C.primary_subset(C.load_metrics(ds)) for ds in C.DATASETS]
    df = pd.concat(frames, ignore_index=True)
    if not include_all_variants:
        df = df[df["model_name"].notna()]
    return df


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by dataset and model in presentation order."""
    df = df.copy()
    df["dataset"] = pd.Categorical(df["dataset"], list(C.DATASETS), ordered=True)
    if "model_name" in df:
        df["model_name"] = pd.Categorical(df["model_name"], C.MODEL_ORDER, ordered=True)
    return df


def summarise(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    df = ordered(df)
    rows = []
    for keys, g in df.groupby(group_cols, sort=True, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n_patients"] = g["patient_id"].nunique()
        for key, name in C.STRUCTURES:
            per_patient = g.groupby("patient_id")[f"dice_{key}"].mean()
            lo, hi = C.bootstrap_ci(per_patient)
            row[f"dice_{name}_mean"] = per_patient.mean()
            row[f"dice_{name}_sd"] = per_patient.std()
            row[f"dice_{name}_ci_lo"], row[f"dice_{name}_ci_hi"] = lo, hi
            hd = g.groupby("patient_id")[f"hd95_{key}"].mean()  # inf propagates: failed structure
            row[f"hd95_{name}_median"] = hd.median()
            row[f"hd95_{name}_q1"], row[f"hd95_{name}_q3"] = hd.quantile([0.25, 0.75])
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "| Dataset | Model | n | Dice RV | Dice MYO | Dice LV | HD95 RV (mm) | HD95 MYO (mm) | HD95 LV (mm) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        flag = " †" if C.in_distribution(r["model_name"], r["dataset"]) else ""
        dice = [f"{r[f'dice_{s}_mean']:.3f} ± {r[f'dice_{s}_sd']:.3f}" for _, s in C.STRUCTURES]
        hd = [f"{r[f'hd95_{s}_median']:.1f} [{r[f'hd95_{s}_q1']:.1f}–{r[f'hd95_{s}_q3']:.1f}]" for _, s in C.STRUCTURES]
        lines.append(f"| {C.DATASETS[r['dataset']]['label']} | {r['model_name']}{flag} | {r['n_patients']} | " + " | ".join(dice + hd) + " |")
    lines.append("")
    lines.append("Dice: mean ± SD over patients (ED and ES averaged per patient). HD95: median [IQR]. "
                 "† the checkpoint was trained on this dataset's training split. "
                 "M&Ms-2: official test subjects 201–360 only.")
    return "\n".join(lines)


def pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ds in C.DATASETS:
        sub = df[df["dataset"] == ds]
        for key, name in C.STRUCTURES:
            for metric in (f"dice_{key}", f"hd95_{key}"):
                wide = C.patient_level(sub, metric).pivot(index="patient_id", columns="model_name", values=metric)[C.MODEL_ORDER]
                wide = wide.replace([np.inf, -np.inf], np.nan)
                n_complete = len(wide.dropna())
                friedman_p, pairs = C.paired_model_tests(wide)
                pairs.insert(0, "metric", metric.split("_")[0])
                pairs.insert(0, "structure", name)
                pairs.insert(0, "dataset", ds)
                pairs["friedman_p"] = friedman_p
                pairs["n_patients"] = n_complete
                out.append(pairs)
    return pd.concat(out, ignore_index=True)


def fig_accuracy(df: pd.DataFrame) -> dict:
    """Figure 2: patient-level DSC (A) and HD95 (B, log scale) by dataset, structure and model.

    HD95 is the patient mean of the finite ED and ES values; a frame where the
    structure is missing from the prediction has no HD95 and is left out (the
    count is returned for the legend).
    """
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    n_ds, n_st = len(C.DATASETS), len(C.STRUCTURES)
    fig = plt.figure(figsize=(7.2, 6.2))
    grid = fig.add_gridspec(n_ds, 2 * n_st + 1, width_ratios=[1] * n_st + [0.45] + [1] * n_st,
                            left=0.085, right=0.995, top=0.91, bottom=0.17, wspace=0.12, hspace=0.12)
    axes = np.array([[fig.add_subplot(grid[i, c if c < n_st else c + 1]) for c in range(2 * n_st)] for i in range(n_ds)])
    lowest = min(C.patient_level(df, f"dice_{k}")[f"dice_{k}"].min() for k, _ in C.STRUCTURES)
    y_floor = np.floor(lowest * 10) / 10
    finite = df.copy()
    for key, _ in C.STRUCTURES:
        finite.loc[~np.isfinite(finite[f"hd95_{key}"]), f"hd95_{key}"] = np.nan
    excluded = {key: int(finite[f"hd95_{key}"].isna().sum()) for key, _ in C.STRUCTURES}
    hd_all = pd.concat([C.patient_level(finite, f"hd95_{k}")[f"hd95_{k}"] for k, _ in C.STRUCTURES]).dropna()
    hd_lim = (hd_all.min() * 0.8, hd_all.max() * 1.4)
    for i, ds in enumerate(C.DATASETS):
        sub = finite[finite["dataset"] == ds]
        for g, metric in enumerate(("dice", "hd95")):
            for j, (key, name) in enumerate(C.STRUCTURES):
                ax = axes[i, g * n_st + j]
                pl = C.patient_level(sub, f"{metric}_{key}")
                data = [pl.loc[pl["model_name"] == m, f"{metric}_{key}"].dropna().values for m in C.MODEL_ORDER]
                bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                                flierprops={"marker": "o", "markersize": 1.5, "alpha": 0.5},
                                medianprops={"color": "#1a1a19", "linewidth": 0.8})
                for patch, m in zip(bp["boxes"], C.MODEL_ORDER):
                    patch.set_facecolor(C.MODEL_COLORS[m])
                    patch.set_edgecolor("#1a1a19")
                    patch.set_linewidth(0.5)
                    if C.in_distribution(m, ds):
                        patch.set_hatch("////")
                if metric == "dice":
                    ax.set_ylim(y_floor, 1.0)
                else:
                    ax.set_yscale("log")
                    ax.set_ylim(*hd_lim)
                ax.grid(axis="y", color="#e5e5e1", linewidth=0.5)
                ax.set_axisbelow(True)
                ax.set_xticks(range(1, len(C.MODEL_ORDER) + 1))
                ax.set_xticklabels(C.MODEL_ORDER if i == n_ds - 1 else [], rotation=60, ha="right", fontsize=6)
                if j > 0:
                    ax.set_yticklabels([])
                if i == 0:
                    ax.set_title(name)
                if j == 0:
                    ax.set_ylabel(f"{C.DATASETS[ds]['label']}\n" + ("DSC" if metric == "dice" else "HD95 (mm)"))
    for g, label in enumerate(("A  DSC", "B  HD95 (mm, log scale)")):
        left = axes[0, g * n_st].get_position().x0
        fig.text(left - 0.06, 0.965, label, ha="left", va="center", fontsize=9, fontweight="bold")
    fig.text(0.99, 0.005, "Hatched: checkpoint trained on this dataset", ha="right", va="bottom", fontsize=6.5)
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_accuracy_by_dataset.{ext}", dpi=300)
    plt.close(fig)
    return excluded


def vendor_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(dice_avg=df[[f"dice_{k}" for k, _ in C.STRUCTURES]].mean(axis=1))
    rows = []
    for (ds, vendor, model), g in df.groupby(["dataset", "vendor", "model_name"]):
        per_patient = g.groupby("patient_id")["dice_avg"].mean()
        lo, hi = C.bootstrap_ci(per_patient)
        rows.append({"dataset": ds, "vendor": vendor, "model_name": model, "n_patients": len(per_patient),
                     "dice_avg_mean": per_patient.mean(), "ci_lo": lo, "ci_hi": hi,
                     "in_distribution": C.in_distribution(model, ds)})
    by_vendor = pd.DataFrame(rows)
    rows = []
    for (ds, path, model), g in df.groupby(["dataset", "pathology", "model_name"]):
        per_patient = g.groupby("patient_id")["dice_avg"].mean()
        rows.append({"dataset": ds, "pathology": path, "model_name": model, "n_patients": len(per_patient),
                     "dice_avg_mean": per_patient.mean()})
    pd.DataFrame(rows).to_csv(C.TABLES / "seg_accuracy_by_pathology.csv", index=False)
    return by_vendor


def fig_vendor(by_vendor: pd.DataFrame) -> None:
    by_vendor = by_vendor.assign(column=by_vendor["dataset"] + "|" + by_vendor["vendor"]
                                 + "\n(n=" + by_vendor["n_patients"].astype(str) + ")")
    order = {d: i for i, d in enumerate(C.DATASETS)}
    cols = list(dict.fromkeys(by_vendor.sort_values(["dataset", "vendor"], key=lambda s: s.map(order) if s.name == "dataset" else s)["column"]))
    grid = by_vendor.pivot(index="model_name", columns="column", values="dice_avg_mean").loc[C.MODEL_ORDER, cols]
    ind = by_vendor.pivot(index="model_name", columns="column", values="in_distribution").loc[C.MODEL_ORDER, cols]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    im = ax.imshow(grid.values, cmap="Blues", vmin=0.6, vmax=0.95, aspect="auto")
    for (r, c), v in np.ndenumerate(grid.values):
        ax.text(c, r, f"{v:.2f}{'†' if ind.values[r, c] else ''}", ha="center", va="center", fontsize=7,
                color="white" if v > 0.85 else "#1a1a19")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.split("|", 1)[1] for c in cols], fontsize=7)
    datasets = [c.split("|", 1)[0] for c in cols]
    for ds in dict.fromkeys(datasets):
        idx = [i for i, d in enumerate(datasets) if d == ds]
        ax.text(np.mean(idx), -0.75, C.DATASETS[ds]["label"], ha="center", va="bottom", fontsize=8, fontweight="bold")
        if idx[0] > 0:
            ax.axvline(idx[0] - 0.5, color="white", linewidth=2)
    ax.set_yticks(range(len(C.MODEL_ORDER)))
    ax.set_yticklabels(C.MODEL_ORDER, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Mean Dice (RV, MYO, LV)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.01, 0.01, "† checkpoint trained on this dataset", fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_vendor_heatmap.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    C.TABLES.mkdir(exist_ok=True)
    C.FIGURES.mkdir(exist_ok=True)
    df = load_all()

    main_summary = summarise(df, ["dataset", "model_name"])
    main_summary["in_distribution"] = [C.in_distribution(m, d) for d, m in zip(main_summary["dataset"], main_summary["model_name"])]
    main_summary.to_csv(C.TABLES / "seg_accuracy_main.csv", index=False)
    (C.TABLES / "seg_accuracy_main.md").write_text(to_markdown(main_summary))

    summarise(df, ["dataset", "frame_tag", "model_name"]).to_csv(C.TABLES / "seg_accuracy_by_phase.csv", index=False)
    summarise(load_all(include_all_variants=True), ["dataset", "base_model"]).to_csv(
        C.TABLES / "seg_accuracy_all_variants.csv", index=False)
    pairwise_tests(df).to_csv(C.TABLES / "seg_pairwise_tests.csv", index=False)

    by_vendor = vendor_table(df)
    by_vendor.to_csv(C.TABLES / "seg_accuracy_by_vendor.csv", index=False)
    excluded = fig_accuracy(df)
    print("frames without HD95 (structure missing), by structure:", excluded)
    fig_vendor(by_vendor)
    print(to_markdown(main_summary))


if __name__ == "__main__":
    main()
