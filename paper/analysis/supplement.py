"""Supplementary Tables S1-S6 and Supplementary Figure S1.

S1  accuracy of every CineMA seed and ensemble (and the other models), per dataset
S2  Friedman and Holm-adjusted pairwise Wilcoxon tests between the five main models
S3  change in DSC from the LV-to-MYO post-processing; Figure S1 shows one example
S4  frame-level failure rates and predictions on slices without reference heart
S5  mean DSC by pathology
S6  clinical agreement (bias [95% limits of agreement] and ICC) by vendor

Table S7 (CLAIM 2024 checklist) is manuscript/claim_2024_checklist_final.md.
Figure S2 is figures/fig_failure_examples (failure_examples.py).

Inputs: the CSV tables written by seg_accuracy.py, postprocess_effect.py,
failures.py and clinical.py; for Figure S1 the ACDC images, manual contours and
SORAT's raw and post-processed nnFormer masks.
Outputs: tables/supplement/S1-S6_*.{md,csv}, figures/figS1_postprocess_example.{png,pdf},
tables/supplement/figS1_case.csv
Run: ./run_analysis.sh analysis/supplement.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

import common as C
from failure_examples import LABEL_CMAP, frame_index, image_frame
from slice_metrics import load_gt

OUT = C.TABLES / "supplement"
DS_LABEL = {k: v["label"] for k, v in C.DATASETS.items()}
CHECKPOINT = {"acdc": "CineMA-ACDC", "mnms": "CineMA-M&Ms", "mnms2": "CineMA-M&Ms-2"}


def variant_name(base_model: str) -> tuple[str, str]:
    """('CineMA-ACDC', 'seed 0' | 'ensemble'), or the main-model name with an empty variant."""
    if base_model.startswith("cinema__"):
        ckpt, var = base_model.split("__", 1)[1].rsplit("_", 1)
        return CHECKPOINT[ckpt], ("ensemble" if var == "ensemble" else f"seed {var.removeprefix('seed')}")
    return C.MAIN_MODELS[base_model], ""


def model_rank(name: str) -> int:
    return C.MODEL_ORDER.index(name)


def p_text(p: float) -> str:
    if pd.isna(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write(name: str, title: str, df: pd.DataFrame, note: str, csv: pd.DataFrame | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (csv if csv is not None else df).to_csv(OUT / f"{name}.csv", index=False)
    lines = [f"**{title}**", "", "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    lines += ["", note]
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n")


def s1() -> None:
    d = pd.read_csv(C.TABLES / "seg_accuracy_all_variants.csv")
    d[["model", "variant"]] = d["base_model"].apply(lambda b: pd.Series(variant_name(b)))
    d["_v"] = d["variant"].map(lambda v: (2 if v == "ensemble" else 0 if v == "" else 1, v))
    d = d.sort_values(["dataset", "model", "_v"], key=lambda s: s.map(model_rank) if s.name == "model" else s)
    rows = []
    for _, r in d.iterrows():
        row = {"Dataset": DS_LABEL[r["dataset"]], "Model": r["model"], "Variant": r["variant"] or "–", "n": int(r["n_patients"])}
        for s in ("RV", "MYO", "LV"):
            row[f"DSC {s}"] = f"{r[f'dice_{s}_mean']:.3f} ± {r[f'dice_{s}_sd']:.3f}"
        for s in ("RV", "MYO", "LV"):
            row[f"HD95 {s} (mm)"] = f"{r[f'hd95_{s}_median']:.1f} [{r[f'hd95_{s}_q1']:.1f}–{r[f'hd95_{s}_q3']:.1f}]"
        rows.append(row)
    write("S1_all_variants", "Table S1. Segmentation accuracy of every CineMA seed and ensemble on the held-out test sets.",
          pd.DataFrame(rows),
          "DSC mean ± SD; HD95 median [IQR], per patient (mean of ED and ES). Ensemble: majority vote of the three "
          "seeds, reported in the main text. nnFormer and VSA-3L are repeated from Table 4 for comparison.",
          d.drop(columns="_v"))


def s2() -> None:
    d = pd.read_csv(C.TABLES / "seg_pairwise_tests.csv")
    d = d.assign(_a=d["model_a"].map(model_rank), _b=d["model_b"].map(model_rank),
                 _d=d["dataset"].map(list(C.DATASETS).index), _s=d["structure"].map(["RV", "MYO", "LV"].index),
                 _m=d["metric"].map(["dice", "hd95"].index))
    d = d.sort_values(["_d", "_m", "_s", "_a", "_b"])
    unit = {"dice": "DSC", "hd95": "HD95 (mm)"}
    rows = [{"Dataset": DS_LABEL[r["dataset"]], "Metric": unit[r["metric"]], "Structure": r["structure"],
             "Friedman p": p_text(r["friedman_p"]), "Model A": r["model_a"], "Model B": r["model_b"],
             "Median difference (A − B)": f"{r['median_diff']:.3f}" if r["metric"] == "dice" else f"{r['median_diff']:.2f}",
             "Holm-adjusted p": p_text(r["p_holm"])} for _, r in d.iterrows()]
    write("S2_pairwise_tests", "Table S2. Pairwise comparison of the five main models.", pd.DataFrame(rows),
          "Friedman test across the five models per dataset, metric and structure, followed by two-sided Wilcoxon "
          "signed-rank tests on patient-level values with Holm correction over the ten pairs. Median difference: "
          "median over patients of model A minus model B.", d.drop(columns=["_a", "_b", "_d", "_s", "_m"]))


def s3() -> None:
    d = pd.read_csv(C.TABLES / "postprocess_effect.csv")
    d = d[d["structure"] != "RV"]
    d = d.assign(_d=d["dataset"].map(list(C.DATASETS).index), _m=d["model_name"].map(model_rank))
    d = d.sort_values(["_d", "_m", "structure"], ascending=[True, True, False])
    rows = [{"Dataset": DS_LABEL[r["dataset"]], "Model": r["model_name"], "Structure": r["structure"], "n": int(r["n_patients"]),
             "DSC, model output": f"{r['dice_raw']:.3f}", "DSC, post-processed": f"{r['dice_pp']:.3f}",
             "Change [95% CI]": f"{r['mean_change']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]",
             "Holm-adjusted p": p_text(r["p_holm"])} for _, r in d.iterrows()]
    write("S3_postprocess_effect", "Table S3. Effect of the LV-to-MYO post-processing on DSC against the manual contours.",
          pd.DataFrame(rows),
          "The optional post-processing relabels dark voxels inside the predicted LV cavity (papillary muscles and "
          "trabeculae) as myocardium; the reference contours include them in the cavity, so the change measures the "
          "disagreement in convention, not an error of the post-processing. RV is unchanged by construction. 95% CI: "
          "patient bootstrap; p: Wilcoxon signed-rank, Holm-adjusted. Example in Figure S1.",
          d.drop(columns=["_d", "_m"]))


def s4() -> None:
    f = pd.read_csv(C.TABLES / "failure_rates.csv")
    off = pd.read_csv(C.TABLES / "off_heart_summary.csv")
    rows = []
    for ds in C.DATASETS:
        for model in C.MODEL_ORDER:
            g = f[(f["dataset"] == ds) & (f["model_name"] == model)].set_index("structure")
            o = off[(off["dataset"] == ds) & (off["model_name"] == model)].iloc[0]
            row = {"Dataset": DS_LABEL[ds], "Model": model + (" †" if C.in_distribution(model, ds) else ""),
                   "Frames": int(g["n_frames"].iloc[0])}
            for s in ("RV", "MYO", "LV"):
                row[f"Failures {s}, n (%)"] = f"{int(g.loc[s, 'failures'])} ({100 * g.loc[s, 'failure_rate']:.1f})"
            row["Missing structure, n"] = int(g["missing_structure"].sum())
            row["Frames with off-heart prediction (%)"] = f"{100 * o['frames_with_off_heart_prediction']:.0f}"
            row["Foreground off heart (%)"] = f"{100 * o['mean_fraction_of_foreground_off_heart']:.1f}"
            rows.append(row)
    write("S4_failures", "Table S4. Frame-level failures and predictions on slices without reference heart.",
          pd.DataFrame(rows),
          "Failure: DSC < 0.5, HD95 > 20 mm or a structure missing from the prediction, per frame (ED and ES). "
          "Off-heart prediction: predicted foreground on a slice where the manual contours contain no LV or "
          "myocardium; foreground off heart is the mean share of all predicted foreground voxels on such slices. "
          "† in-distribution. Examples in Figure S2.", pd.DataFrame(rows))


def s5() -> None:
    d = pd.read_csv(C.TABLES / "seg_accuracy_by_pathology.csv")
    d["Model"] = d["model_name"]
    wide = d.pivot_table(index=["dataset", "pathology", "n_patients"], columns="Model", values="dice_avg_mean").reset_index()
    wide["_d"] = wide["dataset"].map(list(C.DATASETS).index)
    wide = wide.sort_values(["_d", "n_patients"], ascending=[True, False])
    out = pd.DataFrame({"Dataset": wide["dataset"].map(DS_LABEL), "Pathology": wide["pathology"], "n": wide["n_patients"].astype(int)})
    for m in C.MODEL_ORDER:
        out[m] = wide[m].map("{:.3f}".format).values
    write("S5_by_pathology", "Table S5. Mean DSC (RV, MYO and LV averaged) by pathology.", out,
          "Held-out test subjects; pathology labels as released with each dataset. Small groups (n < 10) are "
          "imprecise.", d)


def s6() -> None:
    d = pd.read_csv(C.TABLES / "clinical_agreement_by_vendor.csv")
    order = ["LVEDV", "LVESV", "LVEF", "RVEDV", "RVESV", "RVEF", "LV mass", "Max ED wall thickness"]
    present = [m for m in order if m in set(d["measure"])] + sorted(set(d["measure"]) - set(order))
    d["cell"] = d.apply(lambda r: f"{r['bias']:.1f} [{r['loa_lo']:.1f}, {r['loa_hi']:.1f}]; {r['icc_2_1']:.2f}", axis=1)
    wide = d.pivot_table(index=["dataset", "vendor", "measure", "unit", "n"], columns="model_name", values="cell", aggfunc="first").reset_index()
    wide = wide.assign(_d=wide["dataset"].map(list(C.DATASETS).index), _m=wide["measure"].map(present.index))
    wide = wide.sort_values(["_d", "vendor", "_m"])
    out = pd.DataFrame({"Dataset": wide["dataset"].map(DS_LABEL), "Vendor": wide["vendor"],
                        "Measure": wide["measure"] + " (" + wide["unit"] + ")", "n": wide["n"].astype(int)})
    for m in C.MODEL_ORDER:
        out[m] = wide[m].fillna("–").values
    write("S6_clinical_by_vendor", "Table S6. Agreement of clinical measurements with the manual contours, by vendor.", out,
          "Each cell: bias (model − reference) [95% limits of agreement]; ICC(2,1). EF in percentage points. "
          "Held-out test subjects.", d.drop(columns="cell"))


def figure_s1() -> None:
    """nnFormer on ACDC: raw and post-processed mask on the case with the median MYO DSC change."""
    m = C.load_metrics("ACDC", include_postprocessed=True)
    m = m[m["base_model"] == "nnformer__fold0"]
    per = m.pivot_table(index=["patient_id", "frame_tag"], columns="postprocessed", values="dice_myo")
    change = (per[True] - per[False]).rename("change").reset_index()
    ed = change[change["frame_tag"] == "ED"]
    case = ed.iloc[(ed["change"] - ed["change"].median()).abs().argsort().iloc[0]]
    pid, tag = case["patient_id"], case["frame_tag"]
    idx = frame_index("ACDC", pid, tag)
    seg = C.RESULTS / "ACDC" / "postprocess" / "nnformer__fold0" / "segmentations"
    raw = np.rint(np.asanyarray(nib.load(seg / f"{pid}_{tag}_nnformer__fold0.nii.gz").dataobj)).astype(np.int16)
    pp = np.rint(np.asanyarray(nib.load(seg / f"{pid}_{tag}_nnformer__fold0_pp.nii.gz").dataobj)).astype(np.int16)
    img, gt = image_frame("ACDC", pid, idx), load_gt("ACDC", pid, tag, idx)
    moved = (raw == 3) & (pp == 2)
    z = int(np.argmax(moved.sum(axis=(0, 1))))
    im = np.rot90(img[..., z])
    # Crop to the heart with a margin so the papillary muscles are visible.
    ys, xs = np.nonzero(np.rot90(np.isin(gt[..., z], (1, 2, 3)) | (raw[..., z] > 0)))
    pad = 12
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, im.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, im.shape[1])
    crop = lambda a: a[y0:y1, x0:x1]
    lo, hi = np.percentile(crop(im), [1, 99.5])
    moved_cmap = ListedColormap([(0, 0, 0, 0), (1.0, 0.85, 0.0, 0.95)])
    panels = [("Cine image", None), ("Manual contour", gt), ("nnFormer output", raw), ("Post-processed", pp)]
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    for ax, (title, lab) in zip(axes, panels):
        ax.imshow(crop(im), cmap="gray", vmin=lo, vmax=hi)
        if lab is not None:
            ax.imshow(crop(np.rot90(lab[..., z])), cmap=LABEL_CMAP, vmin=0, vmax=3, interpolation="nearest")
        if title == "Post-processed":
            ax.imshow(crop(np.rot90(moved[..., z].astype(int))), cmap=moved_cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=LABEL_CMAP(i)) for i in (1, 2, 3)] + [plt.Rectangle((0, 0), 1, 1, color=moved_cmap(1))]
    fig.legend(handles, ["RV", "MYO", "LV", "Relabelled LV → MYO"], loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"figS1_postprocess_example.{ext}", dpi=300)
    rec = {"patient_id": pid, "frame_tag": tag, "frame_idx": idx, "slice": z, "dice_myo_raw": per.loc[(pid, tag), False],
           "dice_myo_pp": per.loc[(pid, tag), True], "change": case["change"], "median_change_ed": ed["change"].median(),
           "voxels_relabelled_on_slice": int(moved[..., z].sum())}
    pd.DataFrame([rec]).to_csv(OUT / "figS1_case.csv", index=False)
    print(rec)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (s1, s2, s3, s4, s5, s6, figure_s1):
        f()
    for p in sorted(OUT.glob("*.md")):
        print(p.name, sum(1 for ln in p.read_text().splitlines() if ln.startswith("| ")) - 1, "rows")


if __name__ == "__main__":
    main()
