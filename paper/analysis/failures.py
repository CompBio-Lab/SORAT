"""Failure rates and where errors occur along the long axis.

A structure in one frame counts as a failure when Dice < 0.5, HD95 > 20 mm,
or the structure is missing from the prediction (HD95 infinite).

Outputs:
  tables/failure_rates.csv            failure rate per dataset, model and structure
  tables/dice_by_slice_position.csv   Dice by long-axis third (base, mid, apex)
  tables/off_heart_summary.csv        predictions on slices with no ground-truth heart
  figures/fig_dice_by_position.{png,pdf}

Needs tables/slice_dice.csv.gz from slice_metrics.py.
Run: python analysis/failures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

DICE_FAIL, HD95_FAIL = 0.5, 20.0
THIRDS = ["Base", "Mid", "Apex"]


def failure_rates() -> pd.DataFrame:
    rows = []
    for ds in C.DATASETS:
        df = C.primary_subset(C.load_metrics(ds))
        df = df[df["model_name"].notna()]
        for model in C.MODEL_ORDER:
            sub = df[df["model_name"] == model]
            for key, name in C.STRUCTURES:
                fail = (sub[f"dice_{key}"] < DICE_FAIL) | (sub[f"hd95_{key}"] > HD95_FAIL) | ~np.isfinite(sub[f"hd95_{key}"])
                rows.append({"dataset": ds, "model_name": model, "structure": name, "n_frames": len(sub),
                             "failures": int(fail.sum()), "failure_rate": fail.mean(),
                             "missing_structure": int((~np.isfinite(sub[f"hd95_{key}"])).sum()),
                             "in_distribution": C.in_distribution(model, ds)})
    return pd.DataFrame(rows)


def by_position(slices: pd.DataFrame) -> pd.DataFrame:
    slices = slices.assign(third=pd.cut(slices["position"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=THIRDS))
    rows = []
    for (ds, model, third), g in slices.groupby(["dataset", "model_name", "third"], observed=True):
        row = {"dataset": ds, "model_name": model, "third": third, "n_slices": len(g)}
        for _, name in C.STRUCTURES:
            # Slices where the structure is absent from both masks have no Dice and are skipped.
            row[f"dice_{name}"] = g[f"dice_{name}"].mean(skipna=True)
        rows.append(row)
    return pd.DataFrame(rows)


def fig_position(pos: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(C.DATASETS), len(C.STRUCTURES), figsize=(7.2, 6.0), sharex=True, sharey=True)
    x = np.arange(len(THIRDS))
    for i, ds in enumerate(C.DATASETS):
        for j, (_, name) in enumerate(C.STRUCTURES):
            ax = axes[i, j]
            for model in C.MODEL_ORDER:
                sub = pos[(pos["dataset"] == ds) & (pos["model_name"] == model)].set_index("third").reindex(THIRDS)
                ax.plot(x, sub[f"dice_{name}"], marker="o", markersize=3, linewidth=1.5,
                        color=C.MODEL_COLORS[model], label=model)
            ax.grid(axis="y", color="#e5e5e1", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(THIRDS)
            if i == 0:
                ax.set_title(name)
            if j == 0:
                ax.set_ylabel(f"{C.DATASETS[ds]['label']}\nMean slice Dice")
    axes[0, -1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_dice_by_position.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    rates = failure_rates()
    rates.to_csv(C.TABLES / "failure_rates.csv", index=False)
    wide = rates.pivot_table(index=["dataset", "model_name"], columns="structure", values="failure_rate", observed=True)
    print((wide * 100).round(1).to_string())
    slices = pd.read_csv(C.TABLES / "slice_dice.csv.gz", dtype={"patient_id": str})
    pos = by_position(slices)
    pos.to_csv(C.TABLES / "dice_by_slice_position.csv", index=False)
    fig_position(pos)
    off = pd.read_csv(C.TABLES / "off_heart_predictions.csv", dtype={"patient_id": str})
    off["fraction_off_heart"] = off["pred_voxels_off_heart_slices"] / off["pred_voxels_total"]
    off_summary = off.groupby(["dataset", "model_name"]).agg(
        n_frames=("fraction_off_heart", "size"),
        frames_with_off_heart_prediction=("off_heart_slices_with_prediction", lambda x: (x > 0).mean()),
        mean_fraction_of_foreground_off_heart=("fraction_off_heart", "mean")).reset_index()
    off_summary.to_csv(C.TABLES / "off_heart_summary.csv", index=False)


if __name__ == "__main__":
    main()
