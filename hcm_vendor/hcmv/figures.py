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
