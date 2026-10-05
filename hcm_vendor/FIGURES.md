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

---

## E1–E3 preliminary results (T40–T42), commit `a01a2ab`

Run jobs (submitted by the user from the repo root on `a01a2ab`):
- E1: `sbatch --account=st-zlaksman-1 --output=results_hcm_vendor/logs/%x-%j.out --job-name=hcmv-e1 hcm_vendor/scripts/run_experiment.sbatch --experiment E1 --force` (job 13164013).
- E2: same with `--cpus-per-task=16 --mem=32G --time=02:00:00 ... --experiment E2 --force` (job 13164014).
- E3: `sbatch ... --job-name=hcmv-e3 --time=08:00:00 hcm_vendor/scripts/hcmv_job.sbatch e3-probe` (job 13164015).
- Reports: `hcmv_python -m hcmv e1-report` and `e2-report` (E3's report is written by `e3-probe`). Input: `results_hcm_vendor/tables/features_mms2_nnformer-fold0_{norm,raw}.parquet` (M&Ms-2 train pool, n = 114; NOR for E3, n = 75).
- Protocol: nested CV 5×5 outer (stratified on disease × vendor), 5-fold inner grid search on AUC; 2000-resample class-stratified subject bootstrap CIs of the repeat-averaged metric; threshold 0.5; transfers average 5 seeds for RF/XGB/MLP.

### F9. `runs/E1/analysis/e1_roc_{all,all-no-wt}.png`
Repeat-averaged ROC curves (vertical averaging over the 5 outer repeats) for the five models, one panel per cohort (pooled, Siemens only, Philips only). Legend gives AUC [95% CI]. Key numbers: pooled AUC 0.94 (SVM) to 0.997 (RF) on all features.

### F10. `runs/E2/analysis/model_comparison_{all,all-no-wt}.png` (draft of T61 item 9)
AUC with 95% CI per model, grouped by model family, for E1 pooled, E1 within each vendor and both E2 transfer directions. Read across a model's row: trees stay near 1.0 everywhere; LR-EN drops only Siemens→Philips; SVM and MLP drop in both directions. Not yet marked with the paired-test results (planned for T61). E1-only version: `runs/E1/analysis/e1_model_comparison_*.png`.

### F11. `runs/E2/analysis/e2_gap_auc.png`
Generalization gap ΔAUC = AUC(nested CV within the test vendor) − AUC(trained on the other vendor), both on the same test subjects, with paired-bootstrap 95% CIs; colours are family sets. Positive = loss from crossing vendors. Key numbers (all features): RF/XGB ≈ 0; LR-EN +0.09 Siemens→Philips; MLP +0.20/+0.24; SVM +0.16/+0.19.

### F12. `runs/E2/analysis/e2_roc_{all,all-no-wt}.png`
Per model and direction: ROC within the test vendor (solid, vendor colour) vs trained on the other vendor (dashed).

### F13. `runs/E2/analysis/e2_calibration_{all,all-no-wt}.png`
Reliability curves (quintile bins of predicted P(HCM)) within vs across vendors, Brier score in the legend. Cross-vendor LR-EN, SVM and MLP curves bunch at high predicted probability: the models still rank subjects but over-call HCM on the new vendor, which is why specificity at 0.5 collapses (e.g. LR-EN Siemens→Philips 0.95 → 0.26).

### F14. `runs/E3/analysis/e3_null_<target>_<classifier>.png`
For each feature probe: the label-shuffled null distribution of balanced accuracy (1000 permutations, grey), chance (dotted) and the observed value (red), with permutation p and Holm p. Key numbers (3-class, LR): texture normalized 0.94, raw 0.97, all 0.92 (p = 0.001, the floor); clinical 0.39 and shape 0.44 (n.s.).

### Tables
- `runs/E1/analysis/e1_metrics.csv` (+ `summary.md`): AUC, balanced accuracy, sensitivity, specificity, Brier with CIs; readings (a) within-vendor training and (b) pooled-trained scored per vendor.
- `runs/E1/analysis/e1_model_comparisons.csv`: paired ΔAUC between model pairs on the pooled cohort, Holm within family set.
- `runs/E1/analysis/e1_grid_edges.csv`: share of folds choosing the lowest/highest grid value, raw and strict (ties excluded).
- `runs/E2/analysis/e2_transfer_metrics.csv`, `e2_gap.csv` (Δ for all five metrics with CI, p, Holm across models).
- `runs/E3/analysis/e3_probe.csv`: balanced accuracy with CI, null mean, permutation p and Holm p per target × classifier × probe.

---

## E4, E5 and GE (T43–T45), commit `b7ebc58`

Run jobs (submitted from the repo root on `b7ebc58`; `A="--account=st-zlaksman-1 --output=results_hcm_vendor/logs/%x-%j.out"`, `S=hcm_vendor/scripts/run_experiment.sbatch`):
- `sbatch $A --job-name=hcmv-e1 $S --experiment E1 --family-sets clinical,clinical+shape,clinical+texture,all,all-no-wt` (13207585); the same for E2 with `--cpus-per-task=16 --mem=32G --time=03:00:00` (13207586).
- Raw texture: add `--family-sets clinical+texture,all --set experiments.feature_config=raw` (13207587 E1, 13207588 E2).
- `--experiment E5` (13207589) and `--experiment GE` (13207590) with the five family sets.
- Reports: `sbatch ... hcm_vendor/scripts/hcmv_job.sbatch {e1,e2,e4,e5}-report` (13207644–13207647). E1/E2 tables and figures listed above were regenerated by these with the widened SVM grid.

### F15. `runs/E4/analysis/e4_gap_auc.png`
ΔAUC (within the test vendor − trained on the other vendor; paired-bootstrap 95% CI) per model and direction, coloured by feature family set, normalized texture. Read each model's cluster: clinical (blue) and clinical + shape (orange) sit at 0 for all models; adding texture (green, red) moves SVM and MLP to +0.15 to +0.32. Trees stay at 0 for every set.

### F16. `runs/E4/analysis/e4_gap_auc_raw_vs_norm.png`
The same gap for the two texture-containing sets, normalized vs raw texture. Raw texture gives a smaller gap for the MLP (+0.03 to +0.10 vs +0.20 to +0.32) and for the SVM on all features Philips→Siemens.

### F17. `runs/E5/analysis/e5_roc.png`
ROC curves on the 20 ACDC subjects (10 HCM / 10 NOR) for models trained on all M&Ms-2 NOR/HCM (n = 135), one panel per family set; AUC [95% CI] in the legend. AUC 0.84–1.00; CIs are wide at n = 20.

### Tables
- `runs/E4/analysis/e4_e1_auc.csv` (E1 AUC by family set and config), `e4_gap.csv` (Δ AUC, balanced accuracy and specificity per direction × set × model × config), `e4_contrasts.csv` (ΔΔ family vs clinical and raw vs norm, paired bootstrap, Holm within contrast type and direction), `summary.md`.
- `runs/E5/analysis/e5_metrics.csv` (all metrics with bootstrap CIs plus Wilson CIs for sensitivity/specificity), `e5_predictions.csv` (per-subject P(HCM) for every set × model), `summary.md` with caveats.
- `runs/GE/analysis/ge_specificity.csv` (specificity on 18 GE NOR with Wilson CI and median P(HCM)), `ge_hcm_predictions.csv` (the 3 GE HCM), `summary.md`.

---

## Texture check: why normalized texture transfers worse (follow-up to T43)

Command: `hcmv_python -m hcmv texture-check` (login node, ~3 min). Inputs: `results_hcm_vendor/tables/features_mms2_nnformer-fold0_{norm,raw}.parquet` (train pool, n = 114) and the E2 `clinical+texture` LR-EN/MLP pipelines in `runs/E2/*/clinical+texture/{lr_en,mlp}/{norm,raw}/models/`. Outputs: `results_hcm_vendor/qc/texture_check/`.

### F18. `qc/texture_check/texture_effect_agreement.png`
Each point is one texture feature: its HCM effect (Cohen's d, HCM vs NOR) on Siemens (x) against Philips (y). Green = same direction on both vendors, orange = opposite. Normalized texture: 28 of 84 agree (33%) and the median vendor shift is 2.0 Siemens-NOR SDs; raw texture: 64 of 84 agree (76%; 93% among features with |d| ≥ 0.5 on both) with a median shift of 1.0 SD. Clinical features agree 71% and shape 82% in the normalized table.

### Tables
- `summary.csv` / `summary.md`: per config, median vendor shift, median |d| per vendor, same-sign share, texture features kept by the preprocessing per vendor (norm 48 Siemens / 38 Philips; raw 39 / 38).
- `per_feature.csv`: shift and d for every texture feature.
- `reliance.csv`: AUC drop when the texture block is permuted, for the E2 clinical+texture LR-EN and MLP (first two seeds × 10 permutations), on the training and the test vendor. MLP with normalized texture: +0.19 to +0.20 on the training vendor but −0.08 to −0.09 on the test vendor (texture hurts there); with raw texture +0.02 to +0.06 on the test vendor.

---

## SHAP stability (T50), commit after `3b637b1`

Command: `sbatch --account=st-zlaksman-1 --output=results_hcm_vendor/logs/%x-%j.out --job-name=hcmv-shap --time=06:00:00 --mem=64G hcm_vendor/scripts/hcmv_job.sbatch shap` (job 13208257, 2 min on 32 cores). Inputs: the E2 pipelines `runs/E2/{siemens_to_philips,philips_to_siemens}/all/<model>/norm/models/seed_*.joblib` and the M&Ms-2 train pool (n = 114, explained for both models). Method: TreeSHAP (RF, probability; XGB, log-odds), KernelSHAP for LR-EN/SVM/MLP (background `shap.kmeans` of the training vendor, 20 centres; nsamples 2d + 2048; `l1_reg=False`); seeds averaged; mean |φ| summed within |r| > 0.95 correlation clusters. Outputs: `results_hcm_vendor/runs/SHAP/analysis/`.

### F19. `shap_rank_scatter.png`
One panel per model: each point is a feature cluster, its importance rank when trained on Siemens (x) vs Philips (y), coloured by family; 1 = most important (top right). Title gives Spearman ρ [95% CI]. Points on the diagonal = same importance on both vendors. Ties at the bottom/left are clusters a model barely uses. ρ = 0.17 (LR-EN) to 0.43 (RF).

### F20. `shap_family_share.png`
Stacked bars: share of total mean |SHAP| from clinical, shape and texture features, per model and training vendor. Trees are mostly clinical (0.42–0.97); Siemens-trained LR-EN, SVM and MLP are mostly texture (0.53–0.60); Philips-trained LR-EN is mostly clinical (0.69).

### F21. `shap_beeswarm_<model>.png`
For each model, the top 10 features by mean |SHAP| for the Siemens-trained (left) and Philips-trained (right) model: one dot per subject, x = SHAP value, colour = standardized feature value (red high, blue low). E.g. for LR-EN, Philips-trained relies on ED max wall thickness (high → HCM), Siemens-trained on ED first-order Minimum and texture.

### Tables
- `shap_stability.csv`: ρ with CI, top-10 Jaccard and shared count per model.
- `shap_family_share.csv`, `shap_cluster_importance.csv` (importance and rank per cluster × model × vendor), `correlation_clusters.csv` (feature → cluster).

---

## Results summary and verdicts (T51)

Command: `hcmv_python -m hcmv summary` (seconds; reads the E1–E5, GE and SHAP report CSVs). Outputs: `results_hcm_vendor/tables/`.
- `summary.md`: the four hypothesis verdicts with effect sizes and CIs, headline tables (E1 AUC pooled and within vendor; E2 ΔAUC and Δspecificity; E5 ACDC; GE specificity) and the limitations. The verdict rules are in `hcmv/experiments/summary.py`.
- `summary_tests.csv`: every paired test or probe (135 rows) with question, Holm family, estimate, 95% CI, p, the report's own Holm p and a question-wide Holm p.

---

## Final figures and tables (T61)

Commands: `hcmv_python -m hcmv tables` then `hcmv_python -m hcmv figures` (or the last job of `hcm_vendor/scripts/run_all.sh`). They read the experiment reports, run stores and SHAP caches listed above. Outputs: `results_hcm_vendor/figures/` (PNG and PDF) and `results_hcm_vendor/tables/final/` (CSV and Markdown). Each folder has a `manifest.json` with the commit.

| File | Content | Source data |
|---|---|---|
| `fig01_workflow` | Study workflow: data, segmentation, features, models, experiments E1–E5 and SHAP. Counts from the cohort table. | `tables/cohort.parquet` |
| `fig02_cohort` | (a) subjects per dataset, vendor and diagnosis; (b) ED max wall thickness (nnFormer masks) per vendor × diagnosis, 15 mm line. | cohort, `features_mms2_nnformer-fold0_norm` |
| `fig03_e1_roc` | E1 ROC curves, all features, pooled / Siemens only / Philips only (= F9). | `runs/E1` |
| `fig04_gap_by_family` | ΔAUC by feature family and model, both directions, normalized texture (= F15). | `runs/E4/analysis/e4_gap.csv` |
| `fig05_vendor_probe` | Vendor probe, 3-class, LR: null vs observed per feature family (= F14). | `runs/E3` |
| `fig06_acdc_roc` | ACDC ROC curves for clinical and all features (subset of F17). | `runs/E5` |
| `fig07a_shap_ranks`, `fig07b_shap_family_share` | SHAP rank agreement and family shares (= F19, F20). | `runs/SHAP/analysis` |
| `fig08_calibration_shift` | Calibration within vs across vendors, all features (= F13). | `runs/E1`, `runs/E2` |
| `fig09_model_comparison` | AUC [95% CI] per model and setting (E1 pooled, within each vendor, both E2 directions), grouped by model family; * = below the best model in that setting (paired bootstrap, Holm p < 0.05). | `runs/E1`, `runs/E2`, `tables/final/model_pairs.csv` |

Tables `table1_cohort` … `table8_shap_stability`: cohort counts; E1 AUC; E2 gap (Δ AUC, balanced accuracy, specificity, Brier; * Holm p < 0.05); E3 probe; E4 contrasts; E5 ACDC; GE specificity; SHAP stability. `model_pairs.csv`: every model pair's paired ΔAUC with CI, p and Holm p per setting.

Reproducibility check: `hcmv_python -m hcmv compare-runs --other runs-rerun` → `results_hcm_vendor/qc/rerun_check/compare_runs-rerun.csv` (all 1145 values identical to a full rerun on `5a2b1e5`).

---

## TabPFN added (2026-10-05)

TabPFN v2 is now a sixth model in every E1, E2, E4, E5 and GE run (jobs 13233872–13233877 on `fedd358`), so the E1–E5 report tables and figures F9–F17, the summary and the final tables/figures `fig03`, `fig04`, `fig06`, `fig08`, `fig09` include it (regenerated by jobs 13234106–13234109 and the final job). It is not in the SHAP figures (F19–F21, `fig07a/b`). Key numbers: E1 pooled AUC 0.994; E2 ΔAUC −0.005 / +0.005; specificity drop across vendors 0.14–0.19.

