"""All study figures, drawn so each one reads on its own: plain-language titles,
axis labels with units, group sizes, and legends for every colour.

Every figure is catalogued in ``hcm_vendor/FIGURES.md`` (what it shows, the command
that makes it, its inputs, how to read it). Keep that file in sync when adding or
changing a figure here.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

from .features import feature_family  # noqa: E402

VENDOR_ORDER = ["Siemens", "Philips", "GE"]
VENDOR_COLOURS = {"Siemens": "#1f77b4", "Philips": "#ff7f0e", "GE": "#2ca02c"}
DISEASE_COLOURS = {"NOR": "#4c72b0", "HCM": "#c44e52"}
FAMILY_COLOURS = {"clinical": "#2ca02c", "shape": "#9467bd", "texture": "#8c564b", "sensitivity": "#7f7f7f"}
DISEASE_NAMES = {"NOR": "Normal (NOR)", "HCM": "Hypertrophic cardiomyopathy (HCM)"}
CONFIG_NAMES = {
    "raw": "Raw: PyRadiomics defaults (no intensity normalization)",
    "norm": "Normalized: intensities normalized, 32 grey-level bins, 2D, 1.25 mm pixels",
}
DATASET_NAMES = {"mms2": "M&Ms-2", "acdc": "ACDC"}

CLINICAL_NAMES = {
    "lv_volume_ml": "LV cavity volume (ml)",
    "rv_volume_ml": "RV cavity volume (ml)",
    "myo_volume_ml": "Myocardial volume (ml)",
    "myocardial_mass_g": "Myocardial mass (g)",
    "wall_thickness_mean_mm": "Mean wall thickness (mm)",
    "wall_thickness_max_mm": "Max wall thickness (mm)",
    "wall_thickness_p95_mm": "95th-percentile wall thickness (mm)",
}
DERIVED_NAMES = {
    "lv_sv_ml": "LV stroke volume (ml)",
    "rv_sv_ml": "RV stroke volume (ml)",
    "lvef_pct": "LV ejection fraction (%)",
    "rvef_pct": "RV ejection fraction (%)",
    "mass_to_volume_g_per_ml": "Mass-to-volume ratio (g/ml, ED mass / LV EDV)",
}
RADIOMICS_CLASSES = {"shape": "shape", "firstorder": "first-order", "glcm": "GLCM"}


def feature_label(column: str) -> str:
    """Human-readable feature name, e.g. ``ed_wall_thickness_max_mm`` -> 'ED max wall thickness (mm)'."""
    if column in DERIVED_NAMES:
        return DERIVED_NAMES[column]
    phase, _, base = column.partition("_")
    if phase in ("ed", "es"):
        if base in CLINICAL_NAMES:
            name = CLINICAL_NAMES[base]
            return f"{phase.upper()} {name[0].lower()}{name[1:]}" if not name.startswith(("LV", "RV")) \
                else f"{phase.upper()} {name}"
        if base.startswith("radiomics_original_"):
            klass, _, feature = base[len("radiomics_original_"):].partition("_")
            return f"{phase.upper()} {RADIOMICS_CLASSES.get(klass, klass)}: {feature}"
    return column


def _save(fig, path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _disease_legend(ax, **kwargs) -> None:
    ax.legend(handles=[Patch(facecolor=DISEASE_COLOURS[d], alpha=0.6, label=DISEASE_NAMES[d]) for d in ("NOR", "HCM")],
              **kwargs)


def _boxes(ax, data, colours, labels) -> None:
    box = ax.boxplot(data, patch_artist=True, showfliers=True, widths=0.6,
                     medianprops={"color": "black"}, flierprops={"markersize": 4})
    for patch, colour in zip(box["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)


# ------------------------------------------------------------------ T12


def plot_gt_wall_thickness(table: pd.DataFrame, path) -> None:
    """Boxplots of ED max wall thickness from manual (ground-truth) masks, per dataset x vendor x disease."""
    groups = [(ds, v, d) for ds in ("acdc", "mms2") for v in VENDOR_ORDER for d in ("NOR", "HCM")]
    groups = [g for g in groups if ((table.dataset == g[0]) & (table.vendor == g[1]) & (table.disease == g[2])).any()]
    data, labels = [], []
    for ds, v, d in groups:
        values = table.loc[(table.dataset == ds) & (table.vendor == v) & (table.disease == d),
                           "ed_wall_thickness_max_mm"].dropna()
        data.append(values)
        labels.append(f"{DATASET_NAMES[ds]}\n{v}\n{d} (n={len(values)})")
    fig, ax = plt.subplots(figsize=(11, 5))
    _boxes(ax, data, [DISEASE_COLOURS[d] for _, _, d in groups], labels)
    ax.axhline(15, color="crimson", linestyle="--", linewidth=1)
    ax.text(0.6, 15.3, "15 mm clinical HCM threshold", color="crimson", fontsize=8, ha="left", va="bottom")
    ax.set_ylabel("Maximum myocardial wall thickness\nat end-diastole (mm)")
    ax.set_title("Wall thickness measured on manual (ground-truth) segmentations\n"
                 "HCM hearts should be thicker; ACDC HCM is defined by ≥ 15 mm", fontsize=11)
    _disease_legend(ax, loc="upper right", fontsize=8)
    _save(fig, path)


def plot_texture_vendor_effect(effects: dict, path, masks: str) -> None:
    """Histogram of per-feature vendor effect sizes (eta^2) for texture, raw vs normalized.

    ``effects`` maps config name ('raw'/'norm') to a frame with an ``eta2`` column.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bins = np.linspace(0, 1, 21)
    colours = {"raw": "#1f77b4", "norm": "#ff7f0e"}
    n = 0
    for name, frame in effects.items():
        n = len(frame)
        med = frame["eta2"].median()
        ax.hist(frame["eta2"], bins=bins, alpha=0.55, color=colours[name],
                label=f"{CONFIG_NAMES[name]}\n(median η² = {med:.2f})")
        ax.axvline(med, color=colours[name], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Vendor effect size η² (Kruskal–Wallis across Siemens, Philips, GE)\n"
                  "0 = feature does not depend on vendor · 1 = vendor explains all of its variation")
    ax.set_ylabel(f"Number of texture features (of {n})")
    ax.set_title(f"How much texture features depend on the scanner vendor\n"
                 f"{masks}; M&Ms-2 normal subjects only (n=75), so disease cannot explain it", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False,
              title="Radiomics settings (dashed line = median)", title_fontsize=8)
    _save(fig, path)


# ------------------------------------------------------------------ T22


def plot_clinical_distributions(table: pd.DataFrame, columns, path) -> None:
    groups = [(v, d) for v in VENDOR_ORDER for d in ("NOR", "HCM")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, columns):
        data = [table.loc[(table.vendor == v) & (table.disease == d), col].dropna() for v, d in groups]
        _boxes(ax, data, [DISEASE_COLOURS[d] for _, d in groups], [f"{v}\n{d}\nn={len(x)}" for (v, d), x in zip(groups, data)])
        ax.set_ylabel(feature_label(col), fontsize=9)
        ax.set_title(feature_label(col), fontsize=10)
    fig.legend(handles=[Patch(facecolor=DISEASE_COLOURS[d], alpha=0.6, label=DISEASE_NAMES[d]) for d in ("NOR", "HCM")],
               loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Key clinical measurements by scanner vendor and diagnosis\n"
                 "M&Ms-2, measured on nnFormer (automatic) segmentations; ED = end-diastole, ES = end-systole",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, path)


def plot_auc_heatmap(aucs: pd.DataFrame, counts: dict, path, top: int = 25) -> list:
    """Per-vendor HCM-vs-NOR AUC for the ``top`` features with the largest mean Siemens/Philips separation."""
    pivot = aucs.pivot_table(index="feature", columns="vendor", values="auc")[VENDOR_ORDER]
    order = aucs[aucs.vendor.isin(["Siemens", "Philips"])].groupby("feature")["separation"].mean()
    features = order.sort_values(ascending=False).index[:top].tolist()
    data = pivot.loc[features]
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(features) + 2.2))
    im = ax.imshow(data.to_numpy(), cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(VENDOR_ORDER)))
    ax.set_xticklabels([f"{v}\n({counts[v]['HCM']} HCM / {counts[v]['NOR']} NOR)" for v in VENDOR_ORDER], fontsize=8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([feature_label(f) for f in features], fontsize=7.5)
    for tick, f in zip(ax.get_yticklabels(), features):
        tick.set_color(FAMILY_COLOURS[feature_family(f)])
    for i in range(len(features)):
        for j in range(len(VENDOR_ORDER)):
            value = data.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(value - 0.5) > 0.35 else "black")
    divider = make_axes_locatable(ax)
    cbar = fig.colorbar(im, cax=divider.append_axes("right", size="4%", pad=0.1))
    cbar.set_label("AUC for HCM vs NOR using this one feature\n"
                   "1 = higher in HCM · 0 = lower in HCM · 0.5 = no difference", fontsize=8)
    ax.legend(handles=[Patch(color=FAMILY_COLOURS[f], label=f"{f} feature") for f in ("clinical", "shape", "texture")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=7.5, frameon=False,
              title="Label colour = feature family", title_fontsize=7.5)
    ax.set_title(f"Single-feature HCM-vs-NOR discrimination within each vendor\n"
                 f"Top {top} features by mean separation on Siemens and Philips (M&Ms-2, nnFormer, normalized)\n"
                 "GE has only 3 HCM subjects, so its column is noisy", fontsize=9)
    _save(fig, path)
    return features


def plot_auc_agreement(aucs: pd.DataFrame, counts: dict, path) -> None:
    """Scatter of Siemens vs Philips single-feature AUC, one point per feature, coloured by family."""
    wide = aucs.pivot_table(index=["feature", "family"], columns="vendor", values="auc").reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.axhspan(0.5, 1, xmin=0.5, xmax=1, color="#eeeeee", zorder=0)
    ax.axhspan(0, 0.5, xmin=0, xmax=0.5, color="#eeeeee", zorder=0)
    ax.axhline(0.5, color="grey", linewidth=0.8)
    ax.axvline(0.5, color="grey", linewidth=0.8)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8)
    for family in ("texture", "shape", "clinical"):
        g = wide[wide.family == family]
        r = np.corrcoef(g["Siemens"], g["Philips"])[0, 1]
        ax.scatter(g["Siemens"], g["Philips"], s=22, alpha=0.8, c=FAMILY_COLOURS[family],
                   label=f"{family} (n={len(g)}, r = {r:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"AUC on Siemens ({counts['Siemens']['HCM']} HCM / {counts['Siemens']['NOR']} NOR)")
    ax.set_ylabel(f"AUC on Philips ({counts['Philips']['HCM']} HCM / {counts['Philips']['NOR']} NOR)")
    ax.text(0.98, 0.02, "features in the white quadrants\nseparate HCM in opposite\ndirections on the two vendors",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="dimgrey")
    ax.set_title("Does each feature separate HCM from NOR the same way on both vendors?\n"
                 "One point per feature; shaded quadrants = same direction (M&Ms-2, nnFormer, normalized)",
                 fontsize=9)
    ax.legend(title="Feature family (r = Siemens–Philips correlation)", fontsize=8, title_fontsize=8, loc="upper left")
    _save(fig, path)


def plot_feature_clustermap(corr: np.ndarray, columns, path, n_subjects: int) -> None:
    """|Pearson r| matrix with hierarchical ordering; ``columns`` are already in that order."""
    fig, ax = plt.subplots(figsize=(10, 9.5))
    divider = make_axes_locatable(ax)
    ax_bar = divider.append_axes("top", size="3%", pad=0.05)
    ax_cbar = divider.append_axes("right", size="3%", pad=0.1)
    families = [feature_family(c) for c in columns]
    ax_bar.imshow([[matplotlib.colors.to_rgb(FAMILY_COLOURS[f]) for f in families]], aspect="auto")
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_title(f"Correlation between the {len(columns)} features (M&Ms-2, nnFormer, normalized, n={n_subjects})\n"
                     "Features are ordered by hierarchical clustering; bright blocks are groups of near-duplicate "
                     "features.\nTop bar = feature family", fontsize=9)
    im = ax.imshow(corr, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("features (same order as rows)")
    ax.set_ylabel("features")
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("|Pearson r| between two features")
    ax.legend(handles=[Patch(color=FAMILY_COLOURS[f], label=f) for f in ("clinical", "shape", "texture")],
              loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=8, frameon=False)
    _save(fig, path)


def plot_pca(panels: list, path, suptitle: str) -> None:
    """``panels`` is a list of (row title, table, Z (n x 2), explained variance ratio)."""
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 4.4 * len(panels)), squeeze=False)
    for (title, table, Z, var), row_axes in zip(panels, axes):
        for ax, (by, colours, names) in zip(row_axes, (("vendor", VENDOR_COLOURS, None),
                                                       ("disease", DISEASE_COLOURS, DISEASE_NAMES))):
            for label, colour in colours.items():
                m = (table[by] == label).to_numpy()
                name = names[label] if names else label
                ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.75, c=colour, label=f"{name} (n={m.sum()})")
            ax.set_xlabel(f"Principal component 1 ({var[0]:.0%} of variance)")
            ax.set_ylabel(f"Principal component 2 ({var[1]:.0%} of variance)")
            ax.set_title(f"{title}\ncoloured by {'scanner vendor' if by == 'vendor' else 'diagnosis'}", fontsize=9.5)
            ax.legend(fontsize=7)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, path)


# ------------------------------------------------------------------ T40-T42 (experiments)

MODEL_ORDER = ["lr_en", "svm", "rf", "xgb", "mlp"]
MODEL_NAMES = {"lr_en": "Elastic-net LR", "svm": "RBF SVM", "rf": "Random forest", "xgb": "XGBoost",
               "mlp": "MLP"}
MODEL_FAMILIES = {"lr_en": "Linear", "svm": "Kernel", "rf": "Tree ensemble", "xgb": "Tree ensemble",
                  "mlp": "Neural"}
MODEL_COLOURS = {"lr_en": "#1b9e77", "svm": "#d95f02", "rf": "#7570b3", "xgb": "#e7298a", "mlp": "#66a61e"}
FAMILY_SET_NAMES = {
    "all": "all features",
    "all-no-wt": "all features except wall thickness",
    "clinical": "clinical features",
    "clinical+shape": "clinical + shape features",
    "clinical+texture": "clinical + texture features",
    "shape": "shape features",
    "texture": "texture features",
}
UNIT_NAMES = {
    "pooled": "Pooled (Siemens + Philips)",
    "siemens": "Siemens only",
    "philips": "Philips only",
    "siemens_to_philips": "Siemens → Philips",
    "philips_to_siemens": "Philips → Siemens",
    "mms2_to_acdc": "M&Ms-2 → ACDC",
    "pool_to_ge": "Siemens + Philips → GE",
}
FPR_GRID = np.linspace(0, 1, 101)


def mean_roc(y, P) -> np.ndarray:
    """TPR on ``FPR_GRID`` averaged over the repeat columns of ``P`` (vertical averaging)."""
    from sklearn.metrics import roc_curve

    P = np.asarray(P, dtype=float)
    P = P[:, None] if P.ndim == 1 else P
    curves = []
    for r in range(P.shape[1]):
        fpr, tpr, _ = roc_curve(y, P[:, r])
        curve = np.interp(FPR_GRID, fpr, tpr)
        curve[0] = 0.0
        curves.append(curve)
    return np.mean(curves, axis=0)


def _ci_text(row) -> str:
    return f"{row['estimate']:.2f} [{row['ci_low']:.2f}–{row['ci_high']:.2f}]"


def _roc_axes(ax, title: str) -> None:
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("1 − specificity")
    ax.set_title(title, fontsize=10)


def plot_e1_roc(curves: dict, path, family_set: str, cv_text: str = "5×5") -> None:
    """``curves[unit][model] = (tpr, auc_row, n_hcm, n_nor)``; one panel per cohort."""
    units = [u for u in ("pooled", "siemens", "philips") if u in curves]
    fig, axes = plt.subplots(1, len(units), figsize=(4.6 * len(units), 4.9), squeeze=False)
    for ax, unit in zip(axes[0], units):
        n_hcm = n_nor = 0
        for model in MODEL_ORDER:
            if model not in curves[unit]:
                continue
            tpr, auc, n_hcm, n_nor = curves[unit][model]
            ax.plot(FPR_GRID, tpr, color=MODEL_COLOURS[model], linewidth=1.6,
                    label=f"{MODEL_NAMES[model]}: {_ci_text(auc)}")
        _roc_axes(ax, f"{UNIT_NAMES[unit]} ({n_hcm} HCM / {n_nor} NOR)")
        ax.legend(title="AUC [95% CI]", fontsize=7.5, title_fontsize=8, loc="lower right")
    axes[0][0].set_ylabel("Sensitivity")
    fig.suptitle(f"HCM vs normal, nested {cv_text} cross-validation, {FAMILY_SET_NAMES[family_set]}", fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def plot_model_comparison(table: pd.DataFrame, path, family_set: str, metric_name: str = "AUC") -> None:
    """Dot-and-whisker plot of a metric per model, grouped by model family, one marker per setting.

    ``table`` columns: setting, model, estimate, ci_low, ci_high (settings keep their row order).
    """
    settings = list(dict.fromkeys(table["setting"]))
    markers = ["o", "s", "D", "^", "v", "P"]
    palette = plt.get_cmap("tab10")
    models = [m for m in MODEL_ORDER if m in set(table["model"])]
    fig, ax = plt.subplots(figsize=(7.5, 0.75 * len(models) * max(1, len(settings) / 2.5) + 1.5))
    step = 0.8 / max(len(settings), 1)
    for i, setting in enumerate(settings):
        rows = table[table["setting"] == setting].set_index("model")
        for j, model in enumerate(models):
            if model not in rows.index:
                continue
            r = rows.loc[model]
            yy = j + (i - (len(settings) - 1) / 2) * step
            ax.errorbar(r["estimate"], yy, xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                        fmt=markers[i % len(markers)], color=palette(i), markersize=5, capsize=2, linewidth=1.2,
                        label=setting if j == 0 else None)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([f"{MODEL_NAMES[m]}\n({MODEL_FAMILIES[m].lower()})" for m in models], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel(f"{metric_name} (95% bootstrap CI)")
    ax.set_title(f"Model comparison, {FAMILY_SET_NAMES[family_set]}", fontsize=11)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(settings), 3), frameon=False)
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    _save(fig, path)


def plot_transfer_roc(curves: dict, path, family_set: str) -> None:
    """``curves[direction][model] = {'within': (tpr, auc), 'cross': (tpr, auc), 'n': (hcm, nor)}``.

    Rows are directions (scored on the target vendor), columns are models.
    """
    directions = [d for d in ("siemens_to_philips", "philips_to_siemens") if d in curves]
    models = [m for m in MODEL_ORDER if any(m in curves[d] for d in directions)]
    fig, axes = plt.subplots(len(directions), len(models), figsize=(3.1 * len(models), 3.4 * len(directions)),
                             squeeze=False)
    for i, direction in enumerate(directions):
        source, target = direction.split("_to_")
        for j, model in enumerate(models):
            ax = axes[i][j]
            entry = curves[direction].get(model)
            if entry is None:
                ax.axis("off")
                continue
            tpr_w, auc_w = entry["within"]
            tpr_c, auc_c = entry["cross"]
            ax.plot(FPR_GRID, tpr_w, color=VENDOR_COLOURS[target.title()], linewidth=1.6,
                    label=f"trained on {target.title()}: {auc_w['estimate']:.2f}")
            ax.plot(FPR_GRID, tpr_c, color=VENDOR_COLOURS[source.title()], linewidth=1.6, linestyle="--",
                    label=f"trained on {source.title()}: {auc_c['estimate']:.2f}")
            n_hcm, n_nor = entry["n"]
            _roc_axes(ax, f"{MODEL_NAMES[model]}, tested on {target.title()}")
            ax.title.set_fontsize(8.5)
            ax.legend(title="AUC", fontsize=7, title_fontsize=7, loc="lower right")
            if j == 0:
                ax.set_ylabel(f"Sensitivity\n({target.title()}: {n_hcm} HCM / {n_nor} NOR)", fontsize=8.5)
            ax.xaxis.label.set_fontsize(8)
    fig.suptitle(f"Within-vendor vs cross-vendor ROC curves, {FAMILY_SET_NAMES[family_set]}\n"
                 "solid: nested CV within the test vendor; dashed: trained on the other vendor", fontsize=10.5)
    fig.tight_layout()
    _save(fig, path)


def plot_transfer_calibration(curves: dict, path, family_set: str) -> None:
    """``curves[direction][model] = {'within': (mean_pred, frac_pos), 'cross': (...), 'brier': (w, c)}``."""
    directions = [d for d in ("siemens_to_philips", "philips_to_siemens") if d in curves]
    models = [m for m in MODEL_ORDER if any(m in curves[d] for d in directions)]
    fig, axes = plt.subplots(len(directions), len(models), figsize=(3.1 * len(models), 3.4 * len(directions)),
                             squeeze=False)
    for i, direction in enumerate(directions):
        source, target = direction.split("_to_")
        for j, model in enumerate(models):
            ax = axes[i][j]
            entry = curves[direction].get(model)
            if entry is None:
                ax.axis("off")
                continue
            brier_w, brier_c = entry["brier"]
            ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8)
            ax.plot(*entry["within"], marker="o", markersize=4, color=VENDOR_COLOURS[target.title()],
                    label=f"trained on {target.title()}: {brier_w:.3f}")
            ax.plot(*entry["cross"], marker="s", markersize=4, linestyle="--", color=VENDOR_COLOURS[source.title()],
                    label=f"trained on {source.title()}: {brier_c:.3f}")
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            ax.set_title(f"{MODEL_NAMES[model]}, tested on {target.title()}", fontsize=8.5)
            ax.set_xlabel("Mean predicted P(HCM)", fontsize=8)
            if j == 0:
                ax.set_ylabel("Observed fraction HCM", fontsize=8.5)
            ax.legend(title="Brier score", fontsize=7, title_fontsize=7, loc="upper left")
    fig.suptitle(f"Calibration within vs across vendors, {FAMILY_SET_NAMES[family_set]} (quintile bins)",
                 fontsize=10.5)
    fig.tight_layout()
    _save(fig, path)


def plot_gap_forest(table: pd.DataFrame, path, metric_name: str = "AUC", labels: dict = None,
                    title: str = None) -> None:
    """Generalization gap Δ (within − cross) with CI per model, one panel per direction.

    ``table`` columns: direction, family_set, model, difference, ci_low, ci_high.
    """
    directions = [d for d in ("siemens_to_philips", "philips_to_siemens") if d in set(table["direction"])]
    family_sets = list(dict.fromkeys(table["family_set"]))
    models = [m for m in MODEL_ORDER if m in set(table["model"])]
    fig, axes = plt.subplots(1, len(directions), figsize=(5.2 * len(directions),
                                                          0.22 * len(models) * len(family_sets) + 2.6),
                             squeeze=False, sharey=True, sharex=True)
    palette = plt.get_cmap("tab10")
    step = 0.6 / max(len(family_sets), 1)
    for ax, direction in zip(axes[0], directions):
        for i, family_set in enumerate(family_sets):
            rows = table[(table.direction == direction) & (table.family_set == family_set)].set_index("model")
            for j, model in enumerate(models):
                if model not in rows.index:
                    continue
                r = rows.loc[model]
                yy = j + (i - (len(family_sets) - 1) / 2) * step
                ax.errorbar(r["difference"], yy,
                            xerr=[[r["difference"] - r["ci_low"]], [r["ci_high"] - r["difference"]]],
                            fmt="o", color=palette(i), markersize=5, capsize=2,
                            label=(labels or FAMILY_SET_NAMES).get(family_set, family_set) if j == 0 else None)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_NAMES[m] for m in models])
        ax.set_xlabel(f"Δ{metric_name} = within-vendor − cross-vendor (95% CI)")
        ax.set_title(UNIT_NAMES[direction], fontsize=10)
        ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    axes[0][0].invert_yaxis()
    fig.suptitle(title or f"Cross-vendor generalization gap in {metric_name}", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=min(len(family_sets), 3), frameon=False)
    _save(fig, path)


def plot_probe_null(results: list, path, title: str) -> None:
    """Observed balanced accuracy vs permutation null, one panel per feature family.

    ``results``: dicts with family, label, observed, null (array), chance, p, p_holm.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.4), squeeze=False, sharey=True)
    for ax, r in zip(axes[0], results):
        ax.hist(r["null"], bins=30, color="#bbbbbb", edgecolor="white", label="label-shuffled null")
        ax.axvline(r["chance"], color="grey", linestyle=":", linewidth=1, label=f"chance = {r['chance']:.2f}")
        ax.axvline(r["observed"], color="crimson", linewidth=2, label=f"observed = {r['observed']:.2f}")
        ax.set_title(f"{r['label']}\np = {r['p']:.3f} (Holm {r['p_holm']:.3f})", fontsize=9)
        ax.set_xlabel("Balanced accuracy")
        ax.legend(fontsize=6.5, loc="upper left")
    axes[0][0].set_ylabel("Permutations")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def plot_external_roc(curves: dict, path, title: str) -> None:
    """``curves[family_set][model] = (tpr, auc_row)``; one panel per family set."""
    family_sets = list(curves)
    fig, axes = plt.subplots(1, len(family_sets), figsize=(4.3 * len(family_sets), 4.6), squeeze=False)
    for ax, family_set in zip(axes[0], family_sets):
        for model in MODEL_ORDER:
            if model in curves[family_set]:
                tpr, auc = curves[family_set][model]
                ax.plot(FPR_GRID, tpr, color=MODEL_COLOURS[model], linewidth=1.6,
                        label=f"{MODEL_NAMES[model]}: {_ci_text(auc)}")
        _roc_axes(ax, FAMILY_SET_NAMES.get(family_set, family_set).capitalize())
        ax.legend(title="AUC [95% CI]", fontsize=7, title_fontsize=7.5, loc="lower right")
    axes[0][0].set_ylabel("Sensitivity")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, path)



def plot_texture_effect_agreement(features: pd.DataFrame, path) -> None:
    """Per texture feature: HCM effect (Cohen's d) on Siemens vs Philips, one panel per radiomics config."""
    configs = [c for c in ("norm", "raw") if c in set(features["feature_config"])]
    fig, axes = plt.subplots(1, len(configs), figsize=(5.2 * len(configs), 5.2), squeeze=False, sharex=True,
                             sharey=True)
    for ax, cfg in zip(axes[0], configs):
        f = features[features.feature_config == cfg]
        same = f["same_sign"].to_numpy()
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.scatter(f.loc[same, "d_hcm_siemens"], f.loc[same, "d_hcm_philips"], s=20, c="#1b9e77", alpha=0.8,
                   label=f"same direction ({same.sum()} of {len(f)})")
        ax.scatter(f.loc[~same, "d_hcm_siemens"], f.loc[~same, "d_hcm_philips"], s=20, c="#d95f02", alpha=0.8,
                   label=f"opposite direction ({(~same).sum()} of {len(f)})")
        shift = f["shift_siemens_to_philips"].abs().median()
        ax.set_title(f"{'Normalized' if cfg == 'norm' else 'Raw'} texture\n"
                     f"median vendor shift {shift:.1f} SD", fontsize=10)
        ax.set_xlabel("HCM effect on Siemens (Cohen's d)")
        ax.legend(fontsize=8, loc="upper left")
    axes[0][0].set_ylabel("HCM effect on Philips (Cohen's d)")
    fig.suptitle("Texture HCM effects on Siemens vs Philips (M&Ms-2, n = 114)", fontsize=11)
    fig.tight_layout()
    _save(fig, path)
