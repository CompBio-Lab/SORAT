# Figure and table catalogue (EECE 568 HCM × vendor study)

This file tracks every figure and results table: what it shows, how to regenerate it, which data it uses, how to read it and the key numbers. It is the reference for whoever writes the report or paper. Update it whenever a figure is added or changed.

**Conventions**
- All plotting code is in `hcm_vendor/hcmv/figures.py`. The report builders call it and also write the CSVs and a `summary.md` / `qc_summary.md`.
- Every output folder has a `manifest.json` with the git commit, the config hash and package versions at generation time.
- Run commands from the repo root after `source hcm_vendor/scripts/env.sh`, as `hcmv_python -m hcmv <command>`.
- **Data:**
  - M&Ms-2 NOR/HCM cohort: n = 135. Siemens 22 NOR / 30 HCM, Philips 35 / 27, GE 18 / 3.
  - ACDC external test set: 10 NOR / 10 HCM, Siemens.
  - Features come from the nnFormer fold-0 segmentations (T14) or the manual ground-truth segmentations (T12). They are merged per subject into 129 features: 17 clinical, 28 shape and 84 texture (T21, `results_hcm_vendor/tables/`).
- **Feature configs:**
  - `norm` (primary, D5): intensity normalization (scale 100), 32 grey-level bins, 2D extraction, 1.25 mm in-plane resampling.
  - `raw`: PyRadiomics defaults.
  - Clinical and shape features are identical in both configs; only texture differs.
- **Glossary:**
  - ED/ES: end-diastole / end-systole.
  - AUC: area under the ROC curve for HCM vs NOR.
  - η²: Kruskal–Wallis effect size, (H − k + 1)/(n − k).
  - ICC(3,1): two-way mixed, consistency, single-rater intraclass correlation.

Key numbers below are as of commit `58bc695` and later (2026-09-30).

---

## T12: validation of the fixed features (`results_hcm_vendor/qc/t12_validation/`)

Command: `hcmv_python -m hcmv --set validation.pred_root=results_hcm_vendor/features/nnformer validate-gt` (code: `hcmv/qc.py::validate_gt`).

### F1. `gt_wall_thickness_by_group.png`
- **Shows:** boxplots of ED maximum myocardial wall thickness (mm) measured on the **manual** segmentations. There is one box per dataset × vendor × diagnosis, with n under each box. The red dashed line is the 15 mm clinical HCM threshold.
- **Purpose:** proves the fixed wall-thickness measurement (T10) is clinically plausible before we trust the nnFormer features.
- **Inputs:** `results_hcm_vendor/features/gt/{mms2,acdc}/norm/`, plus the cohort table.
- **Key numbers** (`gt_wall_thickness_by_group.csv`, medians):
  - ACDC: NOR 11.3 mm, HCM 19.2 mm. Every ACDC HCM is ≥ 15 mm and every NOR is < 15 mm, which matches ACDC's diagnostic rule, so this is an external validation.
  - M&Ms-2 HCM medians are 13.4–14.8 mm. Only 43% of HCM are ≥ 15 mm, which is a property of the M&Ms-2 labels.
- **Caveat:** M&Ms-2 GE HCM has n = 3.

### F2. `gt_texture_vendor_effect.png`
- **Shows:** a histogram, over the 84 texture features, of the vendor effect size η² (Kruskal–Wallis across Siemens/Philips/GE), computed on M&Ms-2 **normal subjects only** (n = 75), from the manual segmentations. Blue is `raw` and orange is `norm`; dashed lines are the medians.
- **How to read:** further right means the feature depends more on scanner vendor. Using NOR subjects only means disease cannot drive the effect.
- **Key numbers:** median η² is 0.66 for raw and 0.47 for norm. The share of features with p < 0.05 is 96% (raw) and 92% (norm). Normalization reduces the vendor signal but does not remove it.
- **Inputs:** GT feature tables for both configs. Code: `qc.vendor_effect`.

### Tables
- `pred_vs_gt_agreement.csv`: agreement between nnFormer and GT for 8 clinical features, per dataset × vendor. Columns are n, Pearson r, ICC(3,1), bias (pred − GT), and 95% limits of agreement.
  - Volumes and mass: ICC 0.91–0.99.
  - Weaker: M&Ms-2 LVEF (ICC 0.59–0.71), RVEF (0.56–0.73), and GE ED max wall thickness (0.56).
  - Mass is overestimated by about 11–12 g on M&Ms-2.
- `gt_texture_vendor_effect_{raw,norm}.csv`: per-feature H, p and η² behind F2.
- `summary.md`: all of the above as markdown.

---

## T22: data QC and exploratory analysis (`results_hcm_vendor/qc/t22/`)

Command: `hcmv_python -m hcmv qc-report` (code: `hcmv/explore.py::qc_report`). All figures use M&Ms-2 nnFormer features. They use `norm` unless labelled raw.

### F3. `clinical_distributions.png`
- **Shows:** boxplots of 6 key clinical measurements (ED LV volume, ED myocardial mass, ED max and mean wall thickness, LVEF, mass-to-volume ratio) per vendor × diagnosis, with n under each box.
- **How to read:** blue is NOR and red is HCM. Compare red vs blue within a vendor for disease signal, and the same colour across vendors for vendor shifts.
- **Key numbers** (`clinical_medians.csv`):
  - ED max wall thickness: NOR 9.8–10.5 mm vs HCM 15.5–15.8 mm on Siemens and Philips.
  - Mass-to-volume ratio: NOR about 0.7 vs HCM 0.9–1.0.
- **Caveat:** LVEF from nnFormer has weak agreement with GT (see the T12 table).

### F4. `univariate_auc_top25.png`
- **Shows:** a heatmap of single-feature HCM-vs-NOR AUC, computed separately within each vendor, for the 25 features with the largest mean separation on Siemens and Philips.
- **How to read:**
  - Red (AUC near 1) means higher in HCM; blue (near 0) means lower in HCM; white (0.5) means no signal.
  - Label colour gives the feature family.
  - A row that is red on one vendor and blue on another reverses direction between vendors.
- **Key numbers** (`univariate_auc_by_vendor.csv`):
  - ED max wall thickness: AUC 0.99 (Siemens) and 1.00 (Philips).
  - Median separation, max(AUC, 1 − AUC), per family:

    | family | Siemens | Philips | GE |
    |---|---|---|---|
    | clinical | 0.81 | 0.89 | 0.74 |
    | shape | 0.61 | 0.75 | 0.60 |
    | texture | 0.67 | 0.65 | 0.59 |

- **Caveat:** GE has only 3 HCM subjects.

### F5. `auc_agreement_siemens_philips.png`
- **Shows:** one point per feature, plotting its HCM-vs-NOR AUC on Siemens (x) against Philips (y), coloured by family. Shaded quadrants mean the feature separates in the same direction on both vendors; white quadrants mean opposite directions.
- **This is the key finding of T22.** Texture features scatter into the opposite-direction quadrants, while clinical and shape features lie near the diagonal.
- **Key numbers** (`auc_agreement_siemens_philips.csv`):
  - Siemens–Philips correlation of per-feature AUC: clinical 0.88, shape 0.85, **texture −0.25**.
  - 59 of 84 texture features flip direction, 14 of them strongly (|AUC − 0.5| > 0.15 on both vendors).
  - The reversal is **not** explained by the Siemens scanner mix. Restricted to Siemens SymphonyTim, the only Siemens model with NOR subjects, r is still −0.39. The scanner × disease table is in `qc_summary.md`.
- **Implication:** texture HCM signatures should not transfer between vendors (E2, H1/H2).

### F6. `texture_vendor_effect_raw_vs_norm.png`
- **Shows:** the same kind of plot as F2, but for the **nnFormer** segmentations.
- **Key numbers:** median η² is 0.64 for raw and 0.42 for norm (`vendor_effect_nor.csv` holds the per-feature norm values).
- **Per-family vendor effect** in norm (median η²): clinical 0.04, shape 0.05, texture 0.43.

### F7. `feature_correlation_clustermap.png`
- **Shows:** the |Pearson r| matrix of the 129 standardized features, ordered by average-linkage hierarchical clustering. The top bar shows the feature family.
- **How to read:** bright square blocks are groups of near-duplicate features. They motivate the |r| > 0.95 correlation filter inside the model pipeline (T30), which keeps about 81 of the 129 features.
- **Related table:** `results_hcm_vendor/qc/t30_features_after_filter.csv` gives the features kept per family set across the 25 training folds.

### F8. `pca_vendor_disease.png`
- **Shows:** the first two principal components for 3 feature sets (rows): all 129 features (norm), the 84 texture features (norm), and the 84 texture features (raw). Left panels are coloured by vendor, right panels by diagnosis. Axis labels give the % variance explained.
- **Key observations:**
  - Raw texture: PC1 (71% of variance) separates GE from Siemens and Philips almost completely. That is a pure scanner effect.
  - After normalization the GE cluster is gone, but Siemens and Philips still separate along PC1.
  - With all features, diagnosis and vendor both structure PC1.

### Tables
- `d6_excluded.csv`: subjects failing D6. It is **empty**; no subject fails in any dataset or config.
- `dice_per_subject.csv` and `dice_by_vendor.csv`: nnFormer Dice vs GT, median per dataset × vendor × diagnosis × phase. This is QC only.
  - M&Ms-2 ED Dice: LV 0.83–0.86, MYO 0.71–0.79, RV 0.92–0.95.
- `missingness.csv`: 0 missing cells in all 4 nnFormer tables.
- `qc_summary.md`: all T22 tables as markdown.

---

## T30/T31 checks (`results_hcm_vendor/qc/`)

- `t30_features_after_filter.csv`: features surviving impute → variance → |r| > 0.95 filter per family set, over the 25 outer training folds of the M&Ms-2 train pool (n = 114).
- `t31_one_fold_timing.csv`: grid-search fit time and fold-0 test AUC per model on `all` features. Fit times are LR-EN 14 s, SVM 3 s, RF 30 s and XGB 8 s. Test AUC is 0.99–1.00.
- `t31_family_set_ceiling_check.csv`: 5-fold (1 repeat) test AUC per family set for LR-EN and SVM on the M&Ms-2 train pool.
  - Mean LR-EN AUC: `all` 0.998, `all-no-wt` 0.978, `clinical` 0.995, `shape` 0.969, `texture` 0.935.
  - Removing wall thickness barely lowers the ceiling, because mass and shape still separate the classes.
  - This is a quick check, not the E1 result.

---

## T34: runner smoke and timing runs

No figures. These outputs check the machinery and are not study results.
- `results_hcm_vendor/runs-smoke/{E1,E2}/summary.csv`: smoke runs (1×2 outer CV, tiny grids, 100 bootstrap resamples) of all five models on `all` and `all-no-wt`. The AUCs are meaningless; they show every model and unit runs end-to-end.
  - Command: `sbatch --account=st-zlaksman-1 --output=results_hcm_vendor/logs/%x-%j.out --cpus-per-task=8 --mem=16G --time=01:00:00 hcm_vendor/scripts/run_experiment.sbatch --experiment E1 --smoke` (and `--experiment E2 --smoke`). Jobs 13163710 and 13163711.
- `results_hcm_vendor/runs-timing/E1/summary.csv` and `.../E1/pooled/all/<model>/norm/hyperparams.json`: one 5-fold repeat of E1 on the pooled cohort with full grids, used for the runtime estimate. Fit time per outer fold: LR-EN 16 s, SVM 2 s, RF 77 s, XGB 9 s, MLP 46–66 s. AUC 0.98–1.00 (one repeat; not the E1 result).
  - Command: the same sbatch with `--cpus-per-task=5 --time=03:00:00`, args `--experiment E1 --units pooled --family-sets all --set cv.outer_repeats=1 --set experiments.runs_dir=runs-timing`. Job 13163712.
- Inputs: `results_hcm_vendor/tables/features_mms2_nnformer-fold0_norm.parquet`. Each run folder's `manifest.json` records the commit (the Batch 4 code, run before it was committed, so `git_dirty` is true).

