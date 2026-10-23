"""Downstream case study: does one automatic segmenter reduce vendor shift?

Summarises the HCM-vs-normal vendor study that used SORAT's nnFormer masks
and features on M&Ms-2 (Siemens, Philips, GE). The study itself lives on the
SORAT branch `eece568-hcm-vendor` (hcm_vendor/); the result tables used here
were copied into data/case_study/ (provenance in data/case_study/README.md).

Panel A: balanced accuracy of a probe that predicts the scanner vendor of
normal hearts from each feature family (logistic regression, 3 vendors,
chance 0.33), with features from nnFormer masks vs manual contours.
Panel B: cross-vendor change in HCM-classification AUC (train on one vendor,
test on the other; all features) for each classifier, nnFormer vs manual.

Outputs: figures/fig_case_study.{png,pdf}, tables/case_study_summary.csv
Run: python analysis/case_study.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

DATA = C.OUT_ROOT / "data" / "case_study"
MASKS = {"nnFormer": ("nnFormer masks", "#2a78d6"), "ground truth": ("Manual contours", "#eb6834")}
FAMILIES = [("clinical", "Clinical"), ("shape", "Shape"), ("texture_norm", "Texture")]
CLASSIFIERS = [("lr_en", "LR-EN"), ("svm", "SVM"), ("rf", "RF"), ("xgb", "XGBoost"), ("mlp", "MLP"), ("tabpfn", "TabPFN")]
DIRECTIONS = [("siemens_to_philips", "Siemens → Philips"), ("philips_to_siemens", "Philips → Siemens")]


def main() -> None:
    probe = pd.read_csv(DATA / "s1_e3_probe.csv")
    probe = probe[(probe["target"] == "three_vendor") & (probe["classifier"] == "logreg")]
    gap = pd.read_csv(DATA / "s1_gap_nnformer_vs_gt.csv")
    gap = gap[(gap["metric"] == "auc") & (gap["family_set"] == "all")]

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.6]})

    width = 0.36
    x = np.arange(len(FAMILIES))
    for k, (mask, (label, color)) in enumerate(MASKS.items()):
        sub = probe[probe["masks"] == mask].set_index("probe").loc[[f for f, _ in FAMILIES]]
        pos = x + (k - 0.5) * width
        ax_a.bar(pos, sub["balanced_accuracy"], width * 0.92, color=color, label=label)
        ax_a.errorbar(pos, sub["balanced_accuracy"],
                      yerr=[sub["balanced_accuracy"] - sub["ci_low"], sub["ci_high"] - sub["balanced_accuracy"]],
                      fmt="none", ecolor="#1a1a19", elinewidth=0.8, capsize=2)
    ax_a.axhline(1 / 3, color="#52514e", linestyle="--", linewidth=0.8)
    ax_a.text(len(FAMILIES) - 0.5, 1 / 3 + 0.02, "chance", ha="right", fontsize=7, color="#52514e")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([l for _, l in FAMILIES])
    ax_a.set_ylim(0, 1)
    ax_a.set_ylabel("Vendor identified from normal hearts\n(balanced accuracy)")
    ax_a.set_title("A", loc="left", fontweight="bold")
    ax_a.legend(frameon=False, fontsize=7, loc="upper left")

    ypos = np.arange(len(CLASSIFIERS))
    for d, (direction, dlabel) in enumerate(DIRECTIONS):
        sub = gap[gap["direction"] == direction].set_index("model").loc[[c for c, _ in CLASSIFIERS]]
        offset = (d - 0.5) * 0.36
        for mask, prefix in (("nnFormer", "gap_nnformer"), ("ground truth", "gap_gt")):
            label, color = MASKS[mask]
            marker = "o" if d == 0 else "s"
            ax_b.errorbar(sub[prefix], ypos + offset, xerr=[sub[prefix] - sub[f"{prefix}_low"], sub[f"{prefix}_high"] - sub[prefix]],
                          fmt=marker, color=color, markersize=4, elinewidth=0.8, capsize=0,
                          label=f"{label}, {dlabel}")
    ax_b.axvline(0, color="#52514e", linewidth=0.8)
    ax_b.set_xlim(-0.07, 0.8)  # room on the right for the legend
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([l for _, l in CLASSIFIERS])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Cross-vendor AUC loss (within-vendor − cross-vendor)")
    ax_b.set_title("B", loc="left", fontweight="bold")
    ax_b.grid(axis="x", color="#e5e5e1", linewidth=0.5)
    from matplotlib.lines import Line2D
    direction_keys = [Line2D([], [], marker=m, linestyle="none", color="#52514e", markersize=4, label=l)
                      for m, (_, l) in zip(("o", "s"), DIRECTIONS)]
    ax_b.legend(handles=direction_keys, frameon=False, fontsize=7, loc="lower right",
                title="Train → test vendor (colour as in A)", title_fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_case_study.{ext}", dpi=300)
    plt.close(fig)

    summary = gap[["direction", "model", "gap_nnformer", "gap_gt", "difference", "ci_low", "ci_high", "p_holm"]]
    summary.to_csv(C.TABLES / "case_study_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
