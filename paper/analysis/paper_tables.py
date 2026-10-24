"""Tables 1 and 2 of the manuscript.

Table 1: test-set composition (subjects by vendor and pathology) and which
checkpoint was trained on which dataset.
Table 2: reproducibility, SORAT vs each model's standalone pipeline and
between two independent SORAT runs.

Outputs: tables/table1_datasets.md, tables/table4_reproducibility.md
Run: python analysis/paper_tables.py   (after standalone_agreement.py and determinism.py)
"""

from __future__ import annotations

import pandas as pd

import common as C


def demographics(ds: str, ids: set) -> str:
    """Sex, age and body size of the test subjects, where the dataset releases them."""
    import statistics as st
    if ds == "ACDC":
        h, w = [], []
        for cfg in sorted((C.DATA_ROOT / "ACDC" / "database" / "testing").glob("patient*/Info.cfg")):
            info = dict(line.strip().split(": ", 1) for line in cfg.read_text().splitlines() if ": " in line)
            if cfg.parent.name in ids:
                h.append(float(info["Height"]))
                w.append(float(info["Weight"]))
        return (f"Sex and age not released; height {st.mean(h):.0f} ± {st.stdev(h):.0f} cm, "
                f"weight {st.mean(w):.0f} ± {st.stdev(w):.0f} kg")
    if ds == "MMS1":
        meta = pd.read_csv(C.DATA_ROOT / "M&Ms" / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
        meta = meta[meta["External code"].isin(ids)]
        sex = meta["Sex"].value_counts()
        return (f"{sex.get('M', 0)} M / {sex.get('F', 0)} F; age {meta['Age'].mean():.1f} ± {meta['Age'].std():.1f} years; "
                f"weight {meta['Weight'].mean():.0f} ± {meta['Weight'].std():.0f} kg")
    return "Not released"


def table1() -> str:
    lines = ["| Dataset | Test subjects | Vendors (n) / imaging | Sex, age, body size | Pathologies (n) |",
             "|---|---|---|---|---|"]
    for ds in C.DATASETS:
        m = C.primary_subset(C.load_metrics(ds))
        subj = m[["patient_id", "vendor", "pathology"]].drop_duplicates()
        vend = ", ".join(f"{k} {v}" for k, v in subj["vendor"].value_counts().items())
        if ds == "ACDC":
            vend += "; 1.5 T and 3 T"
        if ds == "MMS2":
            info = pd.read_csv(C.DATA_ROOT / "M&Ms_2" / "MnM2" / "dataset_information.csv", dtype=str).dropna(subset=["SUBJECT_CODE"])
            info = info[info["SUBJECT_CODE"].astype(int).map("{:03d}".format).isin(set(subj["patient_id"]))]
            field = info["FIELD"].value_counts()
            vend += "; " + ", ".join(f"{k} T {v}" for k, v in field.items())
        path = ", ".join(f"{k} {v}" for k, v in subj["pathology"].value_counts().items())
        lines.append(f"| {C.DATASETS[ds]['label']} | {len(subj)} | {vend} | {demographics(ds, set(subj['patient_id']))} | {path} |")
    lines.append("| 2018 Atrial Segmentation Challenge | 154 (87 not in MBAS training) | LGE, University of Utah | Not released | Atrial fibrillation (LA cavity reference) |")
    lines.append("| AIM-ASD | 16 | LGE and MR angiography, 1.5 T | Not released | Atrial septal defect (biatrial reference) |")
    lines += ["", "| Model | Training data | In-distribution test set |", "|---|---|---|"]
    train = {"CineMA-ACDC": "ACDC training split (100)", "CineMA-M&Ms": "M&Ms-1 training split",
             "CineMA-M&Ms-2": "M&Ms-2 training split", "nnFormer": "ACDC training split (100)",
             "VSA-3L": "Non-public (MONAI model zoo)"}
    extra = [("Atrial nnU-Net", "MBAS 2024 training set (70 LGE scans)", "2018 challenge scans also in MBAS (67)")]
    for model in C.MODEL_ORDER:
        ind = ", ".join(C.DATASETS[d]["label"] for d in C.TRAINED_ON[model]) or "None"
        lines.append(f"| {model} | {train[model]} | {ind} |")
    for model, data, ind in extra:
        lines.append(f"| {model} | {data} | {ind} |")
    lines.append("")
    lines.append("CineMA checkpoints: three seeds per training dataset, combined by majority vote. "
                 "M&Ms-2 test set: official subjects 201–360.")
    return "\n".join(lines)


def table4() -> str:
    sa = pd.read_csv(C.TABLES / "standalone_agreement_summary.csv")
    det = pd.read_csv(C.TABLES / "determinism.csv")
    lines = ["| Comparison | Model | Masks | Identical | Minimum voxel agreement | Largest DSC difference |",
             "|---|---|---|---|---|---|"]
    full = pd.read_csv(C.TABLES / "standalone_agreement.csv")
    for _, r in sa.iterrows():
        d = full[full["model_name"] == r["model_name"]]
        diff = max((d[f"dice_sorat_vs_gt_{s}"] - d[f"dice_standalone_vs_gt_{s}"]).abs().max() for s in ("RV", "MYO", "LV"))
        label = "CineMA (9 checkpoints)" if r["model_name"] == "CineMA" else r["model_name"]
        lines.append(f"| SORAT vs standalone notebook | {label} | {int(r['n_frames'])} | {int(r['identical_frames'])} | "
                     f"{100 * r['min_voxel_agreement']:.4f}% | {diff:.4f} |")
    names = {"cinema": "CineMA (12 variants)", "nnformer": "nnFormer", "vsa3l": "VSA-3L"}
    for _, r in det.iterrows():
        lines.append(f"| Two independent SORAT runs | {names[r['architecture']]} | {int(r['masks'])} | {int(r['identical'])} | "
                     f"{100 * r['min_voxel_agreement']:.4f}% | {r['max_abs_dice_difference']:.4f} |")
    lines.append("")
    lines.append("ACDC test set, ED and ES frames. DSC difference: largest absolute difference in DSC against the "
                 "reference contours for any structure and frame. CineMA's standalone run uses its notebook with a "
                 "centre crop in place of the ground-truth LV crop (standalone/cinema_centre_crop.py).")
    return "\n".join(lines)


def main() -> None:
    (C.TABLES / "table1_datasets.md").write_text(table1())
    (C.TABLES / "table4_reproducibility.md").write_text(table4())
    print(table1())
    print()
    print(table4())


if __name__ == "__main__":
    main()
