"""Clinical measurements from automatic vs manual contours.

Predicted measurements come from SORAT's own feature extraction
(`$SORAT_RESULTS/<dataset>/features/raw`). Reference measurements are computed
from the manual contours with the same SORAT functions
(`bin/extract_features.py: compute_volumes, compute_wall_thickness`), so
both sides share one definition: volume = voxel count x voxel volume,
LV mass = myocardial volume x 1.05 g/ml, LV/RV cavities exclude papillary
muscles exactly as the contours do.

Measurements: LVEDV, LVESV, LVEF, RVEDV, RVESV, RVEF, LV mass (ED), and
maximum ED wall thickness. Agreement: Bland-Altman bias and 95% limits of
agreement, ICC(2,1) (absolute agreement, two-way random), and the share of
subjects with an absolute EF error above 5 percentage points.

Outputs:
  tables/clinical_measurements.csv        one row per subject x model (+ reference)
  tables/clinical_agreement.{csv,md}      agreement per dataset x model x measurement
  tables/clinical_agreement_by_vendor.csv
  figures/fig_bland_altman_ef.{png,pdf}   LVEF and RVEF Bland-Altman, best external model per dataset
Run: python analysis/clinical.py   (add --plot-only to redraw the figure from the saved table)
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

import common as C

sys.path.insert(0, str(C.REPO / "bin"))
from extract_features import compute_volumes, compute_wall_thickness  # noqa: E402

from slice_metrics import load_gt  # noqa: E402

MEASURES = [
    ("lvedv", "LVEDV", "ml"), ("lvesv", "LVESV", "ml"), ("lvef", "LVEF", "%"),
    ("rvedv", "RVEDV", "ml"), ("rvesv", "RVESV", "ml"), ("rvef", "RVEF", "%"),
    ("lv_mass", "LV mass", "g"), ("max_wt", "Max ED wall thickness", "mm"),
]


def derive(ed: dict, es: dict) -> dict:
    out = {"lvedv": ed["lv_volume_ml"], "lvesv": es["lv_volume_ml"],
           "rvedv": ed["rv_volume_ml"], "rvesv": es["rv_volume_ml"],
           "lv_mass": ed["myocardial_mass_g"], "max_wt": ed["wall_thickness_max_mm"]}
    out["lvef"] = 100 * (out["lvedv"] - out["lvesv"]) / out["lvedv"] if out["lvedv"] else np.nan
    out["rvef"] = 100 * (out["rvedv"] - out["rvesv"]) / out["rvedv"] if out["rvedv"] else np.nan
    return out


def reference_measures(ds: str, frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in frames.groupby("patient_id"):
        phase = {}
        for _, fr in g.iterrows():
            gt = load_gt(ds, pid, fr["frame_tag"], int(fr["frame_idx"]))
            spacing = reference_spacing(ds, pid, fr["frame_tag"])
            mask_zyx = np.transpose(gt, (2, 1, 0))  # SORAT functions use SimpleITK (z, y, x) order
            vox_ml = float(np.prod(spacing)) / 1000.0
            phase[fr["frame_tag"]] = {**compute_volumes(mask_zyx, vox_ml), **compute_wall_thickness(mask_zyx, spacing)}
        if {"ED", "ES"} <= phase.keys():
            rows.append({"patient_id": pid, "model_name": "Reference", **derive(phase["ED"], phase["ES"])})
    return pd.DataFrame(rows)


def reference_spacing(ds: str, pid: str, tag: str) -> tuple:
    root = C.DATA_ROOT
    if ds == "ACDC":
        img = root / "ACDC" / "database" / "testing" / pid / f"{pid}_4d.nii.gz"
    elif ds == "MMS1":
        img = root / "M&Ms" / "Testing" / pid / f"{pid}_sa_gt.nii.gz"
    else:
        img = root / "M&Ms_2" / "MnM2" / "SA" / pid / f"{pid}_SA_{tag}_gt.nii.gz"
    return tuple(float(z) for z in nib.load(img).header.get_zooms()[:3])


def predicted_measures(ds: str, patients) -> pd.DataFrame:
    folder = C.RESULTS / C.DATASETS[ds]["folder"] / "features" / "raw"
    rows = []
    for model, name in C.MAIN_MODELS.items():
        for pid in patients:
            phase = {}
            for tag in ("ED", "ES"):
                f = pd.read_csv(folder / f"{pid}_{model}_{tag}_features.csv").iloc[0]
                p = tag.lower() + "_"
                phase[tag] = {k: f[p + k] for k in ("lv_volume_ml", "rv_volume_ml", "myocardial_mass_g", "wall_thickness_max_mm")}
            rows.append({"patient_id": pid, "model_name": name, **derive(phase["ED"], phase["ES"])})
    return pd.DataFrame(rows)


def icc_2_1(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement."""
    data = np.column_stack([x, y])
    data = data[np.all(np.isfinite(data), axis=1)]
    n, k = data.shape
    grand = data.mean()
    ss_rows = k * ((data.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((data.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((data - grand) ** 2).sum() - ss_rows - ss_cols
    msr, msc, mse = ss_rows / (n - 1), ss_cols / (k - 1), ss_err / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def agreement(merged: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in merged.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for key, label, unit in MEASURES:
            pred, ref = g[f"{key}_pred"].to_numpy(float), g[f"{key}_ref"].to_numpy(float)
            ok = np.isfinite(pred) & np.isfinite(ref)
            diff = pred[ok] - ref[ok]
            row = dict(zip(group_cols, keys))
            row.update({"measure": label, "unit": unit, "n": int(ok.sum()), "bias": diff.mean(),
                        "loa_lo": diff.mean() - 1.96 * diff.std(ddof=1), "loa_hi": diff.mean() + 1.96 * diff.std(ddof=1),
                        "mean_abs_error": np.abs(diff).mean(), "icc_2_1": icc_2_1(pred[ok], ref[ok])})
            if key in ("lvef", "rvef"):
                row["share_abs_error_gt5"] = (np.abs(diff) > 5).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def fig_bland_altman(merged: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, len(C.DATASETS), figsize=(7.2, 4.6), sharey="row")
    for j, ds in enumerate(C.DATASETS):
        sub = merged[(merged["dataset"] == ds) & (merged["model_name"] == "nnFormer")]
        for i, (key, label) in enumerate((("lvef", "LVEF"), ("rvef", "RVEF"))):
            ax = axes[i, j]
            mean = (sub[f"{key}_pred"] + sub[f"{key}_ref"]) / 2
            diff = sub[f"{key}_pred"] - sub[f"{key}_ref"]
            bias, sd = diff.mean(), diff.std(ddof=1)
            ax.scatter(mean, diff, s=8, color=C.MODEL_COLORS["nnFormer"], edgecolor="white", linewidth=0.3)
            for y, style in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
                ax.axhline(y, color="#52514e", linestyle=style, linewidth=0.8)
            ax.text(0.98, 0.04, f"bias {bias:+.1f}\nLoA {bias - 1.96 * sd:+.1f} to {bias + 1.96 * sd:+.1f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
            ax.grid(color="#e5e5e1", linewidth=0.5)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(f"{C.DATASETS[ds]['label']} (n={len(sub)})")
            if i == 1:
                ax.set_xlabel("Mean of automatic and manual (%)")
            if j == 0:
                ax.set_ylabel(f"{label}: automatic − manual (%)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_bland_altman_ef.{ext}", dpi=300)
    plt.close(fig)


def to_markdown(agg: pd.DataFrame) -> str:
    keep = ["LVEDV", "LVEF", "RVEDV", "RVEF", "LV mass"]
    lines = ["| Dataset | Model | " + " | ".join(f"{m} bias [LoA]; ICC" for m in keep) + " |",
             "|---|---|" + "---|" * len(keep)]
    for (ds, model), g in agg.groupby(["dataset", "model_name"], sort=False):
        g = g.set_index("measure")
        cells = [f"{g.loc[m, 'bias']:+.1f} [{g.loc[m, 'loa_lo']:+.1f}, {g.loc[m, 'loa_hi']:+.1f}]; {g.loc[m, 'icc_2_1']:.2f}" for m in keep]
        flag = " †" if C.in_distribution(model, ds) else ""
        lines.append(f"| {C.DATASETS[ds]['label']} | {model}{flag} | " + " | ".join(cells) + " |")
    lines += ["", "Bias and 95% limits of agreement (automatic − manual; ml, %, g). ICC(2,1), absolute agreement. "
              "† checkpoint trained on this dataset. M&Ms-2: test subjects 201–360."]
    return "\n".join(lines)


def plot_only() -> None:
    """Redraw the figure from tables/clinical_measurements.csv."""
    m = pd.read_csv(C.TABLES / "clinical_measurements.csv", dtype={"patient_id": str})
    ref = m[m["model_name"] == "Reference"].drop(columns="model_name")
    pred = m[m["model_name"] != "Reference"]
    fig_bland_altman(pred.merge(ref, on=["dataset", "patient_id"], suffixes=("_pred", "_ref")))


def main() -> None:
    if "--plot-only" in sys.argv:
        plot_only()
        return
    all_rows, merged_all = [], []
    for ds in C.DATASETS:
        metrics = C.primary_subset(C.load_metrics(ds))
        frames = metrics[["patient_id", "frame_tag", "frame_idx"]].drop_duplicates()
        ref = reference_measures(ds, frames)
        pred = predicted_measures(ds, sorted(ref["patient_id"]))
        both = pd.concat([ref, pred], ignore_index=True).assign(dataset=ds)
        all_rows.append(both)
        merged = pred.merge(ref.drop(columns="model_name"), on="patient_id", suffixes=("_pred", "_ref"))
        meta = C.load_metadata(ds)
        merged_all.append(merged.assign(dataset=ds).merge(meta, on="patient_id", how="left"))
        print(ds, "done", flush=True)
    pd.concat(all_rows).to_csv(C.TABLES / "clinical_measurements.csv", index=False)
    merged = pd.concat(merged_all, ignore_index=True)
    merged["dataset"] = pd.Categorical(merged["dataset"], list(C.DATASETS), ordered=True)
    merged["model_name"] = pd.Categorical(merged["model_name"], C.MODEL_ORDER, ordered=True)
    agg = agreement(merged, ["dataset", "model_name"])
    agg.to_csv(C.TABLES / "clinical_agreement.csv", index=False)
    agreement(merged, ["dataset", "vendor", "model_name"]).to_csv(C.TABLES / "clinical_agreement_by_vendor.csv", index=False)
    (C.TABLES / "clinical_agreement.md").write_text(to_markdown(agg))
    fig_bland_altman(merged)
    print(to_markdown(agg))


if __name__ == "__main__":
    main()
