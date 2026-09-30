# EECE 568 project dashboard: vendor-robust HCM detection from SORAT features

**Project:** Can interpretable features extracted from automated nnFormer segmentations (clinical, shape, texture) separate HCM from normal hearts when the MRI vendor changes between training and testing? Proposal: *Moheban, EECE 568 Project Proposal, Oct 1 2026*.

**Branch:** `eece568-hcm-vendor` · **Code:** `hcm_vendor/` · **Outputs:** `results_hcm_vendor/` (gitignored, never committed)
**Deadlines:** midterm update **Sun Nov 1 2026** · project complete **Fri Nov 27 2026**
**Last updated:** 2026-09-29

## ▶ Resume here (checkpoint)

> A new session saying "continue from where we left off" should read this block first, then the ticket it points to. Update this block at the end of **every** ticket or work session. It is the single source of truth for "where are we".

- **Last completed:** T10 (wall thickness, on `main` + merged), T01 (environment), T02 (package skeleton). Decision D5 is recorded.
- **In progress:** T11 (configurable radiomics settings in `bin/extract_features.py` → `modules/features.nf` → `nextflow.config`). This is a SORAT fix, so commit it on `main` and merge.
- **Next up:** T11 → T13 (samplesheets) and T20 (cohort table), which are independent → T12 (ground-truth features + validation) → T14 (SLURM re-extraction).
- **Open questions for the user:** none.
- **Not pushed:** `main` and `eece568-hcm-vendor` are local only. Ask before pushing.
- **How to run things:** `source hcm_vendor/scripts/env.sh`, then `hcmv_python -m pytest hcm_vendor/tests -q -p no:cacheprovider` or `hcmv_python -m hcmv <cmd>`. For SORAT tests, use `module load gcc apptainer && apptainer exec containers/sorat-cinema.sif python -m unittest discover -s tests`. For SLURM, `sbatch --account=st-zlaksman-1 --output=results_hcm_vendor/logs/%x-%j.out <script>`.
- **Gotchas to remember:**
  - Never create venvs. xgboost and shap are in the container home's `.local`. Keep xgboost below 3.1, and never let pip upgrade numpy, sklearn or pandas there.
  - Your uncommitted work on `main` (VSA-3L edits, README, `nextflow.config`, etc.) stays unstaged on both branches. Never `git add -A`.
  - `nextflow.config` has **uncommitted user edits**. When T11 changes it, stage only the T11 hunk (`git add -p` is not available non-interactively, so build a patch).

### Git workflow for fixes (agreed 2026-09-29)

- **SORAT-level fixes** are anything in `bin/`, `modules/`, `main.nf`, `nextflow.config`, repo `tests/` or `CLAUDE.md`. Commit them on **`main`** first: `git checkout main`, then `git add <specific files>` and commit. Then `git checkout eece568-hcm-vendor` and `git merge main`. That way `main` gets the fix and the project branch stays a superset.
- **Project-only work** is `hcm_vendor/` and `.gitignore` entries for `results_hcm_vendor/`. Commit it on `eece568-hcm-vendor` only.
- Always stage explicit paths. Uncommitted user changes ride along across checkouts and must not be committed.

## How to use this file

- **Order.** Work tickets top to bottom in the board below. It is sorted by completion order, which already respects dependencies. Do not start a ticket until everything in its **Depends on** list is DONE.
- **Statuses.**
  - `TODO`: not started.
  - `DOING`: at most one or two tickets at a time.
  - `DONE`: add the date and commit SHA.
  - `BLOCKED`: say what it is waiting on.
  - `SKIP`: say why.
- **Updating a ticket.** Change the status in the board and in the ticket itself, tick its task boxes, and add a line to its **Log**. If a decision is made, record it in the Decisions log.
- **Acceptance criteria are the definition of DONE.** If one turns out to be wrong, change it explicitly and log why. Don't quietly drop it.
- **Priorities.**
  - **P0**: must be done for the result to be valid.
  - **P1**: a core proposal deliverable.
  - **P2**: stretch work, only if time allows after P0/P1.
- **Never write into `results/`.** It holds the original SORAT runs and is read-only for this project. New pipeline outputs go to `results_hcm_vendor/`.

---

## Status board

| # | Ticket | Phase | Pri | Target | Depends on | Status |
|---|---|---|---|---|---|---|
| T00 | Branch + planning dashboard | 0 Setup | P0 | Sep 29 | — | DONE (2026-09-29) |
| T01 | Analysis environment (existing datascience container) | 0 Setup | P0 | Oct 1 | — | DONE (2026-09-29) |
| T02 | `hcm_vendor/` package skeleton, config, run manifest | 0 Setup | P0 | Oct 2 | T01 | DONE (2026-09-29) |
| T10 | Fix wall thickness in `bin/extract_features.py` | 1 Features | P0 | Oct 5 | T01 | DONE (2026-09-29, `3e3972d` on main) |
| T11 | Configurable, normalized radiomics settings | 1 Features | P0 | Oct 6 | T01 | TODO |
| T12 | Ground-truth-mask features + validate T10/T11 | 1 Features | P0 | Oct 8 | T10, T11, T02 | TODO |
| T13 | Study-cohort samplesheets (M&Ms-2 NOR/HCM, ACDC NOR/HCM) | 1 Features | P0 | Oct 7 | T02 | TODO |
| T14 | Re-extract nnFormer features (normalized + raw) on SLURM | 1 Features | P0 | Oct 11 | T12, T13 | TODO |
| T20 | Cohort/metadata table with vendor, disease, role | 2 Dataset | P0 | Oct 9 | T02 | TODO |
| T21 | Feature-table builder (ED+ES merge, derived features, families) | 2 Dataset | P0 | Oct 13 | T14, T20 | TODO |
| T22 | Data QC + exploratory report | 2 Dataset | P1 | Oct 15 | T21 | TODO |
| T30 | CV splitting + leakage-safe preprocessing pipeline | 3 Framework | P0 | Oct 15 | T21 | TODO |
| T31 | Model zoo + hyperparameter grids (LR-EN, SVM, RF, XGB) | 3 Framework | P0 | Oct 17 | T30 | TODO |
| T32 | PyTorch MLP as a scikit-learn estimator | 3 Framework | P0 | Oct 18 | T30 | TODO |
| T33 | Metrics, bootstrap CIs, paired bootstrap tests | 3 Framework | P0 | Oct 18 | T02 | TODO |
| T34 | Experiment runner, result store, SLURM scripts | 3 Framework | P0 | Oct 21 | T31, T32, T33 | TODO |
| T40 | **E1** pooled + within-vendor nested CV | 4 Experiments | P1 | Oct 25 | T34 | TODO |
| T41 | **E2** cross-vendor transfer + generalization gap Δ | 4 Experiments | P1 | Oct 28 | T40 | TODO |
| T42 | **E3** vendor probe on NOR + permutation test | 4 Experiments | P1 | Oct 29 | T34 | TODO |
| T60 | **Midterm update** (due Nov 1) | M Milestone | P0 | Nov 1 | T22, T40, (T41, T42 preliminary) | TODO |
| T43 | **E4** feature-family ablation (+ raw vs normalized texture) | 4 Experiments | P1 | Nov 6 | T40, T41, T42 | TODO |
| T44 | **E5** external test on ACDC | 4 Experiments | P1 | Nov 8 | T34 | TODO |
| T45 | GE specificity check | 4 Experiments | P1 | Nov 8 | T34 | TODO |
| T50 | SHAP attributions + Siemens-vs-Philips stability | 5 Interpret | P1 | Nov 14 | T41 | TODO |
| T51 | Statistical comparison summary, hypothesis verdicts | 5 Interpret | P1 | Nov 16 | T43, T44, T45, T50 | TODO |
| T61 | Final figures + tables | 6 Deliver | P1 | Nov 20 | T51 | TODO |
| T62 | Final report | 6 Deliver | P0 | Nov 26 | T61 | TODO |
| T63 | Reproducibility pass (one-command rerun, README, tests green) | 6 Deliver | P1 | Nov 25 | T51 | TODO |
| S1 | Rerun E1/E2 on ground-truth-mask features (segmentation vs image shift) | Stretch | P2 | — | T12, T41 | TODO |
| S2 | Segmenter sensitivity (CineMA ACDC ensemble) | Stretch | P2 | — | T43 | TODO |
| S3 | Feature harmonization baseline (ComBat) | Stretch | P2 | — | T43 | TODO |
| S4 | Scanner-level (within-vendor) probe | Stretch | P2 | — | T42 | TODO |
| S5 | AHA-segment wall thickness | Stretch | P2 | — | T10 | TODO |
| S6 | Upstream SORAT PR for T10/T11 (separate from analysis code) | Stretch | P2 | — | T14 | TODO |

### Timeline

```
Sep 29 ─ Oct 2   Phase 0  Setup                         T00 T01 T02
Oct 1  ─ Oct 11  Phase 1  Feature correctness in SORAT  T10 T11 T12 T13 T14   <- critical path
Oct 9  ─ Oct 15  Phase 2  Dataset assembly + QC         T20 T21 T22
Oct 13 ─ Oct 21  Phase 3  Modelling framework           T30 T31 T32 T33 T34
Oct 21 ─ Oct 29  Phase 4a E1, E2 (prelim), E3 (prelim)  T40 T41 T42
Oct 29 ─ Nov 1   MIDTERM UPDATE                         T60
Nov 2  ─ Nov 8   Phase 4b E4, E5, GE check, finalize E2/E3   T43 T44 T45
Nov 9  ─ Nov 16  Phase 5  SHAP + statistics             T50 T51
Nov 16 ─ Nov 27  Phase 6  Figures, report, reproducibility   T61 T62 T63
                 Buffer: Nov 24-27. Stretch work only if Phase 5 is done by Nov 16.
```

**Critical path:** T10 → T12 → T14 → T21 → T30 → T34 → T40 → T41 → T50 → T62. Phase 1 slipping moves everything. If T14 is late, Phase 3 can still be built and tested on the *old* feature CSVs (the schema is the same), then rerun once the corrected features land.

---

## Decisions log

| ID | Date | Decision | Source |
|---|---|---|---|
| D1 | 2026-09-29 | Fix the wall-thickness bug in SORAT (`bin/extract_features.py`), validate it on ground-truth masks, and re-extract nnFormer features for M&Ms-2 and ACDC. | User |
| D2 | 2026-09-29 | Texture features: re-extract with intensity normalization and a fixed bin count as the **primary** set. Keep the raw (current-settings) texture as a **sensitivity analysis**. | User |
| D3 | 2026-09-29 | Branch `eece568-hcm-vendor`. Analysis code in `hcm_vendor/`. Dashboard committed. | User |
| D4 | 2026-09-29 | Midterm update due Nov 1; project complete Nov 27. | User |
| D5 | 2026-09-29 | The `norm` config is: normalize (scale 100), binCount 32, `force2D` (dimension 0), and in-plane resample to 1.25×1.25 mm (z unchanged). `raw` = PyRadiomics defaults, as in the original runs. | User |
| D6 | open (T22) | What counts as a "failed case"? **Recommended:** automated criteria only (missing or NaN features, radiomics error, empty LV/MYO, or implausible volumes). Report Dice against ground truth as QC only, so the pipeline stays "fully automated". | Confirm at T22 |
| D7 | open (T33) | Threshold for balanced accuracy, sensitivity and specificity. **Recommended:** 0.5 on predicted probability, with class-balanced training. AUC is threshold-free and is the primary metric. | Confirm at T33 |
| D8 | open (T42) | E3 probe design. **Recommended:** primary is a 3-class (Siemens/Philips/GE) balanced accuracy with chance = 1/3; secondary is binary Siemens-vs-Philips, since those are the two training vendors. | Confirm at T42 |
| D9 | open (T50) | SHAP evaluation set and feature grouping. **Recommended:** explain both E2 models on the same 114 pooled subjects and aggregate |φ| over correlated-feature clusters (|r| > 0.95), so the rankings compare like with like. | Confirm at T50 |
| D10 | open (T60) | Midterm update format: slides, a written report, or a short memo? What length and template? | Ask user at T60 |

---

## Verified facts (checked 2026-09-29, re-check if data changes)

**Cohort (from `/scratch/st-zlaksman-1/pmoheban/M&Ms_2/MnM2/dataset_information.csv`):**
- This file has about 1M empty Excel padding rows. Drop rows that are entirely empty, which leaves 360 subjects.
- NOR/HCM counts match the proposal exactly:

| | Siemens (`SIEMENS`) | Philips (`Philips Medical Systems`) | GE (`GE MEDICAL SYSTEMS`) |
|---|---|---|---|
| HCM | 30 | 27 | 3 |
| NOR | 22 | 35 | 18 |

- All NOR/HCM cases are at 1.5 T, so field strength does not confound the vendor comparison.
- `SUBJECT_CODE` is an unpadded integer. File and folder IDs are zero-padded to three digits (`071`).
- ACDC test split (`/scratch/st-zlaksman-1/pmoheban/ACDC/database/testing/patient101..150/Info.cfg`) has `Group:` with 10 each of DCM, HCM, MINF, NOR and RV. The study uses the 10 HCM and 10 NOR subjects. They are Siemens, at 1.5 T or 3 T, but the per-subject field strength is not in Info.cfg.

**Segmentations:**
- nnFormer (`nnformer__fold0`) was trained on ACDC training data only. So there is no segmentation-training leakage into M&Ms-2 or into ACDC test (101–150).
- Masks are at `results/{MMS_2,ACDC}/nnformer/segmentations/<pid>_{ED,ES}_nnformer__fold0.nii.gz` (ACDC pid is `patientNNN`), in native image space.
- Masks use the SORAT label convention: **1=RV, 2=MYO, 3=LV**.
- **Ground-truth label conventions differ.**
  - M&Ms-2 ground truth (`M&Ms_2/MnM2/SA/NNN/NNN_SA_{ED,ES}_gt.nii.gz`) is **1=LV, 2=MYO, 3=RV**. This was checked by which label the myocardium encloses.
  - ACDC ground truth (`patientNNN_frameXX_gt.nii.gz`) is 1=RV, 2=MYO, 3=LV.
  - Canonicalize with `_remap_cardiac_labels` / `_canonicalize_3d_gt` in `bin/frame_manifest.py`, which decides from anatomy.
- M&Ms-2 also provides 3D per-phase images (`NNN_SA_ED.nii.gz`, `NNN_SA_ES.nii.gz`).
- Each M&Ms-2 `Info.cfg` holds 0-indexed ED/ES indices into `NNN_SA_CINE.nii.gz`. They were regenerated and checked by `scripts/verify_regen_mms2_cfg.py`.

**Current feature CSVs (the old features, to be superseded by T14):**
- Location: `results/MMS_2/features/raw/<pid>_nnformer__fold0_{ED,ES}_features.csv` (230 subjects) and `results/ACDC/features/raw/patientNNN_nnformer__fold0_{ED,ES}_features.csv` (50, created Sep 26).
- One row per file. Columns are prefixed `ed_`/`es_`. The `patient_id` column is *not* a bare ID, so parse the ID from the filename.
- Per phase there are 62 features:
  - 6 clinical: LV, RV and MYO volume, myocardial mass, and mean/max wall thickness.
  - 14 shape.
  - 18 first-order.
  - 24 GLCM.
- There are also meta columns (`mask_file`, `mask_source`, `voxel_volume_ml`) and a trailing duplicate `myocardial_mass_g`.
- Radiomics are computed on the **MYO label only** (label 2).
- **Known defects (the reason for Phase 1):**
  - **Max wall thickness is wrong.** ED median is about 22–24 mm for NOR and HCM alike, on every vendor (expected about 8–11 mm for NOR). The distance is measured in 3D across 10 mm slices, and the septal epicardium borders the RV (label 1), not background (label 0), so it never counts as the outer boundary.
  - **Texture barely varies for Philips and Siemens.** Intensities are unnormalized and PyRadiomics' default bin width of 25 is used. The median first-order Mean is about 276–420 for GE versus about 45–64 for Philips and Siemens, and GLCM Contrast is about 8–20 for GE versus about 0.5–1.4 for Philips and Siemens.

**Environment (decided 2026-09-29: containers only, no venvs):**
- **Analysis Python** runs in `/arc/project/st-singha53-1/pmoheban/jupyter/jupyter-datascience.sif`, with home `/scratch/st-zlaksman-1/pmoheban/my_jupyter`.
  - Interactive: `module load gcc apptainer && launch-apptainer`.
  - Scripts and sbatch: `source hcm_vendor/scripts/env.sh`, then `hcmv_python ...`.
- It has Python 3.11.6, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.15.2, torch 2.8.0 (cu128), matplotlib 3.10.1, seaborn 0.13.0, statsmodels 0.14.0, pytest 8.3.5, SimpleITK 2.5.0, nibabel, pyarrow 13 and PyYAML.
- **xgboost 3.2.0 and shap 0.49.1** were added on 2026-09-29 with `pip install --user` inside the container (user's choice), so they live in `my_jupyter/.local`. They were installed against a constraints file pinning numpy, scikit-learn, pandas, scipy and matplotlib to the container's versions. **Never let pip upgrade those in `.local`**, because that shadows the image.
- The container has no PyRadiomics. SORAT feature extraction still uses `/scratch/st-zlaksman-1/pmoheban/venvs/sorat-features-conda` via `--feature_extraction.virtualenv_path`, an existing SORAT mechanism that we leave alone.
- **Imports are slow on first use from `/arc`.** On the login node, torch took about 260 s and sklearn about 45 s. Budget for this in job time limits.
- **SORAT code tests** (repo `tests/`) run in `containers/sorat-cinema.sif`, which has numpy/scipy/SimpleITK but no radiomics or sklearn.
- **SLURM account:** `st-zlaksman-1` (and `-gpu`). `st-singha53-1` also exists. `env.sh` defaults to `st-zlaksman-1`.
- The login node has internet (PyPI reachable). Assume compute nodes do not.

**FEATURES_ONLY behaviour (from `main.nf`):**
- Features are written to `--feature_extraction.output_dir`, not to `--outdir`. The default is `<results_dir>/features/raw`, which would write into `results/`, so **always set `output_dir`**.
- `--models all` picks ensembles only. Use `--models nnformer --feature_extraction.model_tag nnformer__fold0`.

**Repo gotchas:**
- `.gitignore` ignores `lib/`, `docs/`, `scripts/`, `/data/`, `results/` and `*.nii.gz`. Do not name anything `lib/` under `hcm_vendor/`.
- Samplesheets under `data/` are not tracked. Generate study samplesheets with a tracked script instead.

---

## Planned layout of `hcm_vendor/`

```
hcm_vendor/
  DASHBOARD.md            # this file
  README.md               # how to set up + run (T63)
  requirements.txt        # pinned analysis deps (T01)
  configs/study.yaml      # paths, cohorts, seeds, grids, bootstrap/permutation counts (T02)
  hcmv/                   # python package
    __init__.py
    paths.py config.py manifest.py        # T02
    cohort.py                             # T20
    features.py                           # T21 (table builder, derived features, families)
    qc.py                                 # T22
    splits.py preprocessing.py            # T30
    models.py                             # T31
    mlp.py                                # T32
    stats.py                              # T33
    runner.py                             # T34
    experiments/e1.py e2.py e3.py e4.py e5.py ge_check.py   # T40-T45
    shap_analysis.py                      # T50
    figures.py tables.py                  # T61
    __main__.py                           # CLI: python -m hcmv <command>
  scripts/                # sbatch + helper shell scripts (T13, T14, T34, T63)
  tests/                  # pytest (unittest-compatible), synthetic fixtures only
  reports/                # midterm + final report sources (T60, T62)
results_hcm_vendor/       # gitignored outputs: inputs/, features/, tables/, qc/, runs/, figures/
```

The SORAT pipeline changes (T10, T11) live in `bin/extract_features.py`, `modules/features.nf` and `nextflow.config`, with tests in the repo-level `tests/`.

---

# Tickets

## Phase 0: Setup

### T00: Branch + planning dashboard
**Status:** DONE (2026-09-29) · **Pri:** P0
- [x] Create branch `eece568-hcm-vendor` from `main`. The working-tree changes on main (VSA-3L edits etc.) come along unstaged and must not be committed as part of this project.
- [x] Write this dashboard from the proposal and a full codebase and data review.
- **Log:** 2026-09-29, created. The user answered D1–D4.

### T01: Analysis environment (existing datascience container)
**Status:** DONE (2026-09-29) · **Pri:** P0 · **Target:** Oct 1 · **Depends on:** —

**Goal:** everything needed for modelling, SHAP and plots, usable from SLURM compute nodes. **No venvs.** Use the user's `jupyter-datascience.sif` (see Verified facts → Environment).

**Tasks:**
- [x] Inventory the container's packages. xgboost and shap were missing.
- [x] Install xgboost and shap with `pip install --user` inside the container, under a constraints file so the core packages are not upgraded.
- [x] `hcm_vendor/scripts/env.sh`: `module load gcc apptainer` plus an `hcmv_python` wrapper. It uses the same image and home as `launch-apptainer`, binds `/scratch` and `/arc`, and puts `hcm_vendor/` on `PYTHONPATH`.
- [x] `hcm_vendor/scripts/smoke_env.py` + `smoke_env.sbatch`: import everything, fit a tiny XGB and torch MLP, and run TreeSHAP.
- [x] Smoke test passes on a **compute node** (job 13152950, node se010, 3m40s).
- [x] Record exact versions in `hcm_vendor/requirements.txt` for documentation. It is not used to install anything.

**Acceptance:** the smoke-test job passes on a compute node; the versions are recorded.

**Notes:** GPUs are not needed (n≈135, d≈130). The proposal mentions two GPUs, but CPU jobs are faster to schedule. MLP code stays device-agnostic anyway.

**Log:**
- 2026-09-29: The venv approach was rejected by the user; switched to the container. A stray empty venv I had created was deleted.
- 2026-09-29: The first smoke job (13152911) failed. xgboost 3.2.0's `base_score` format (`'[5E-1]'`) breaks shap 0.49.1 TreeExplainer, so xgboost is pinned to **3.0.5**. Job 13152950 passed.
- On the compute node, torch import took about 52 s, SimpleITK about 24 s and sklearn about 7 s. **Allow about 2 min of import overhead per job.**

### T02: `hcm_vendor/` package skeleton, config, run manifest
**Status:** DONE (2026-09-29) · **Pri:** P0 · **Target:** Oct 2 · **Depends on:** T01

**Tasks:**
- [x] Create the layout above, with an importable `hcmv` package and `python -m hcmv --help`.
- [x] `configs/study.yaml`:
  - Data roots: M&Ms-2 info CSV, ACDC testing root, SORAT results dirs, new feature dirs.
  - Output root `results_hcm_vendor/`.
  - Global seed and repeats: outer 5×5, inner 5.
  - Bootstrap: 2000 resamples. Permutations: 1000.
  - Model grids, filled in by T31/T32.
  - Feature-config names: `norm` (primary) and `raw` (sensitivity).
- [x] `config.py`: load YAML, allow CLI overrides, and compute a config hash.
- [x] `manifest.py`: every run writes `manifest.json` with git SHA, dirty flag, config hash, package versions, host, SLURM job ID and timestamps.
- [x] Add `results_hcm_vendor/` to `.gitignore`.
- [x] Set up pytest (`hcm_vendor/tests/`) and add a trivial test so the harness works: `python -m pytest hcm_vendor/tests -q`.

**Acceptance:** the CLI runs; the test suite runs; the manifest is written by a dummy command.

---

**Log:**
- 2026-09-29: Built:
  - `hcmv/{__init__,__main__,config,manifest}.py`.
  - CLI commands `show-config` and `manifest`. Add new commands to `COMMANDS` in `__main__.py`.
  - `configs/study.yaml`, with cohort expected counts, CV, bootstrap and permutation settings.
  - `results_hcm_vendor/` added to `.gitignore`.
  - 5 pytest tests pass.
- Run tests with `source hcm_vendor/scripts/env.sh && hcmv_python -m pytest hcm_vendor/tests -q -p no:cacheprovider`.
- The manifest reads versions via `importlib.metadata`, because importing torch from /arc on the login node took minutes.
- `feature_configs` in study.yaml are placeholders until T11.

---

## Phase 1: Feature correctness in SORAT (critical path)

### T10: Fix wall thickness in `bin/extract_features.py`
**Status:** DONE (2026-09-29) · **Pri:** P0 · **Target:** Oct 5 · **Depends on:** T01 (container is enough)

**Log:**
- 2026-09-29: Implemented as specified. Also made the PyRadiomics import lazy, with a fail-fast `import radiomics` in `main()` so CLI behaviour is unchanged.
- Added `tests/test_extract_features_wall_thickness.py` (6 phantom tests). All pass in `sorat-cinema.sif`. Against the old implementation, 3 fail and 1 errors (no p95 key), so the tests do catch the bug.
- Spot check on the first 6 ED subjects per vendor × disease (3 for GE HCM), median values:

  | | nnFormer max | GT max | nnFormer p95 | GT p95 |
  |---|---|---|---|---|
  | GE HCM (n=3) | 11.9 | 13.4 | 10.4 | 9.4 |
  | GE NOR | 9.1 | 9.2 | 7.7 | 7.4 |
  | Philips HCM | 15.6 | 15.2 | 13.5 | 12.0 |
  | Philips NOR | 11.0 | 10.1 | 9.4 | 8.3 |
  | Siemens HCM | 16.6 | 15.3 | 13.7 | 13.2 |
  | Siemens NOR | 10.6 | 10.9 | 8.8 | 9.2 |

  Plausible, and nnFormer tracks ground truth. The full-cohort validation is T12.
- Commits: `3e3972d` on `main`, merged into `eece568-hcm-vendor`. `CLAUDE.md` was committed to main as `da371a6`.
- **Checklist (all done):**
  - [x] Algorithm implemented.
  - [x] Phantom tests: annulus, RV-bordered septum, thickest slice, through-plane leakage, NaN, slice axis.
  - [x] Tests run in the container.

**Problem:** `compute_wall_thickness` has two faults.
- It measures a 3D Euclidean distance transform with 3D neighbourhoods on stacks with 8–10 mm slices, so boundaries leak through the slice direction.
- It defines the epicardial boundary as MYO touching **background (0)** only. The septal epicardium touches the **RV (1)**, so septal endocardial pixels measure all the way to the RV insertion points.

As a result, max wall thickness is about 23 mm for normal hearts. This breaks the ≥15 mm HCM criterion the clinical family depends on.

**Algorithm spec (per-slice, in-plane):**
1. Find the slice axis as the axis with the largest spacing (SA stacks: array axis 0 in `[z,y,x]`). Use in-plane spacing `(sy, sx)`.
2. For each slice that has both LV (3) and MYO (2):
   - `endo` = MYO pixels that are 4-connected in-plane to LV.
   - `exterior` = pixels that are neither MYO nor LV (background **and RV**).
3. `edt = distance_transform_edt(~exterior, sampling=(sy, sx))`, the in-plane distance to the nearest exterior pixel. Thickness samples are `edt[endo]`. Document the ±1-pixel bias of pixel-centre distances.
4. Features:
   - `wall_thickness_mean_mm` = mean of all samples across slices.
   - `wall_thickness_max_mm` = max over all samples.
   - Also add `wall_thickness_p95_mm`, a robust maximum used only as a sensitivity feature. Do not add it to the main clinical family, to stay faithful to the proposal's 6 features.
5. Return NaN when there are no valid slices. Keep the function signature and column names so the pipeline and aggregation keep working.

**Tasks:**
- [ ] Implement the algorithm and update the docstring. Keep `mask_arr.ndim == 3`.
- [ ] Add `tests/test_extract_features_wall_thickness.py` with synthetic phantoms and anisotropic spacing (1.25, 1.25, 10):
  - (a) A concentric annulus with inner radius 20 mm and outer radius 28 mm. Expect a mean of about 8 mm within one pixel, and a max of about 8 mm.
  - (b) The same annulus with RV (label 1) directly against the septal side. Expect no inflation compared with (a).
  - (c) A multi-slice stack with one slice 16 mm thick. Expect max ≈ 16.
  - (d) An empty LV. Expect NaN.
  - (e) A 10 mm slice spacing where adjacent slices differ. Expect no through-plane leakage.
- [ ] Tests import `extract_features` via the same `sys.path` trick as the existing tests. Pure numpy/scipy tests must not need radiomics, so split imports or guard them.
- [ ] Run inside a container: `apptainer exec containers/sorat-cinema.sif python -m unittest tests.test_extract_features_wall_thickness -v`. PyRadiomics is not in that container, so put the venv `site-packages` on `PYTHONPATH` or guard the import in the tests.

**Acceptance:** all phantom tests pass. A quick spot check on about five M&Ms-2 NOR and five HCM nnFormer masks gives plausible values (NOR max mostly 8–13 mm, HCM mostly ≥15 mm). The full check is in T12.

### T11: Configurable, normalized radiomics settings
**Status:** TODO · **Pri:** P0 · **Target:** Oct 6 · **Depends on:** T01

**Problem:** the extractor settings are hardcoded to PyRadiomics defaults:
- no intensity normalization;
- `binWidth = 25`;
- 3D GLCM;
- no resampling.

MRI intensities are in arbitrary units that differ by vendor, so the texture features mostly measure the vendor's intensity scale (D2).

**Tasks:**
- [ ] Add CLI flags to `bin/extract_features.py`. All defaults must reproduce today's behaviour exactly.
  - `--radiomics_normalize` (bool) and `--radiomics_normalize_scale` (default 100).
  - `--radiomics_bin_count` (int; mutually exclusive with `--radiomics_bin_width`, default width 25).
  - `--radiomics_resample_spacing sx,sy,sz` (0 keeps the axis unchanged).
  - `--radiomics_force2d` and `--radiomics_force2d_dimension` (default 0).
  - `--radiomics_remove_outliers` (sigma; optional).
- [ ] Pass these into `RadiomicsFeatureExtractor(**settings)`.
- [ ] Plumb them through `nextflow.config` as `params.feature_extraction.radiomics { ... }` (null means default) and through `modules/features.nf` script args. Include the settings in `versions.yml` or a sidecar so every CSV can be traced to its settings.
- [ ] **Decide D5 with the user.** Proposed `norm` config:
  - `normalize=True`, `normalizeScale=100`;
  - `binCount=32`;
  - in-plane resample to `[1.25, 1.25, 0]`;
  - `force2D=True`, `force2Ddimension=0`.

  Shape features are unaffected by intensity settings. Check whether resampling changes shape values, and document the answer.
- [ ] Tests (skipped if PyRadiomics is missing):
  - (a) With the `norm` config, first-order shape-of-distribution features (Skewness, Kurtosis, Entropy) and GLCM features are invariant, within tolerance, to scaling the image by a linear factor k∈{0.2, 5}. Without `norm` they are not.
  - (b) Default flags give byte-identical output to the pre-change code on a fixture.

**Acceptance:** tests pass; default behaviour is unchanged; the two named configs are defined in `configs/study.yaml` and can be passed from the command line to FEATURES_ONLY.

### T12: Ground-truth-mask features + validation of T10/T11
**Status:** TODO · **Pri:** P0 · **Target:** Oct 8 · **Depends on:** T10, T11, T02

**Goal:**
- Prove the fixed features are clinically plausible before spending the SLURM run.
- Produce ground-truth-derived features as a reference. These enable nnFormer-vs-GT agreement by vendor, and stretch S1.

**Tasks:**
- [ ] `hcm_vendor/scripts/extract_gt_features.py`. For each study subject (T13 list) and phase:
  - Load the GT mask.
  - **Canonicalize labels** by importing `_remap_cardiac_labels` from `bin/frame_manifest.py`. M&Ms-2 ground truth is 1=LV/3=RV and must become 1=RV/3=LV.
  - Write a temporary canonical mask.
  - Call `bin/extract_features.py --mask ... --image ...` for both configs (`norm`, `raw`).
- [ ] Image to use:
  - M&Ms-2: the 3D per-phase `NNN_SA_{ED,ES}.nii.gz`, with `--frame_idx 0`.
  - ACDC: `patientNNN_frameXX.nii.gz`, where the frame comes from Info.cfg ED/ES (1-indexed in the filename, so check this).
- [ ] Run as a small SLURM array or a single multi-core job using the container plus the radiomics venv. Output goes to `results_hcm_vendor/features/gt/{mms2,acdc}/{norm,raw}/`.
- [ ] Validation notebook or script output in `results_hcm_vendor/qc/t12_validation/`:
  - Ground-truth max wall thickness by disease and vendor. Expect NOR about 8–12 mm and HCM mostly ≥15 mm.
  - nnFormer (after T14) vs ground truth, per clinical feature and per vendor: Pearson r, ICC(3,1) and Bland–Altman bias/limits of agreement.
  - Texture: vendor medians of first-order Mean and GLCM Contrast under `raw` vs `norm`.

**Acceptance:**
- Ground-truth wall-thickness distributions look clinically plausible. If not, go back to T10 before T14.
- A validation summary with a table and two figures is saved and summarized in this ticket's log.

### T13: Study-cohort samplesheets
**Status:** TODO · **Pri:** P0 · **Target:** Oct 7 · **Depends on:** T02

**Tasks:**
- [ ] `python -m hcmv make-samplesheets` writes `results_hcm_vendor/inputs/`:
  - `mms2_nor_hcm.csv`: 135 rows, all vendors. Same columns as `data/mms2_sa_samplesheet.csv` (`patient_id,image,ground_truth,info_cfg`) and the same zero-padded IDs.
  - `acdc_nor_hcm.csv`: 20 rows, taken from `data/acdc_testing_samplesheet.csv` filtered by Info.cfg `Group`.
  - `cohort_ids.csv`: `patient_id, dataset, disease, vendor`, used as a cross-check in T20.
- [ ] Assert the counts: M&Ms-2 NOR 75 and HCM 60 (S 22/30, P 35/27, G 18/3); ACDC NOR 10 and HCM 10.

**Acceptance:** the files exist and the counts assert; a FEATURES_ONLY dry run with `-stub` or `-preview`, if it works, sees exactly these subjects.

### T14: Re-extract nnFormer features (normalized + raw) on SLURM
**Status:** TODO · **Pri:** P0 · **Target:** Oct 11 · **Depends on:** T12, T13

**Runs:** 2 datasets × 2 configs. The existing nnFormer masks are reused; nothing is re-segmented.

```bash
# template, one per {dataset in mms2,acdc} x {cfg in norm,raw}
nextflow run main.nf -entry FEATURES_ONLY -profile slurm \
  --slurm_account "$SORAT_SLURM_ACCOUNT" \
  --models nnformer --feature_extraction.model_tag nnformer__fold0 \
  --input results_hcm_vendor/inputs/<dataset>_nor_hcm.csv \
  --feature_extraction.enabled true --feature_extraction.mask_source predictions \
  --feature_extraction.results_dir $PWD/results/<MMS_2|ACDC> \
  --feature_extraction.output_dir $PWD/results_hcm_vendor/features/nnformer/<dataset>/<cfg> \
  --feature_extraction.virtualenv_path /scratch/st-zlaksman-1/pmoheban/venvs/sorat-features-conda \
  --feature_extraction.require_virtualenv true \
  <radiomics flags for cfg from T11> \
  --outdir $PWD/results_hcm_vendor/pipeline_runs/<dataset>_<cfg>
```

**Tasks:**
- [ ] Wrap the four runs in `hcm_vendor/scripts/run_feature_extraction.sh`.
- [ ] Check that DISCOVER_POSTPROCESS_INPUTS only emits rows for subjects in the subset samplesheet.
- [ ] Check that `sorat-features-conda` Python major.minor matches the container Python, since the module errors otherwise.
- [ ] Expected files per config: M&Ms-2 has 135×2 = 270 CSVs; ACDC has 20×2 = 40.
- [ ] Post-run check script (`python -m hcmv check-features`):
  - The file counts above.
  - No empty files.
  - No `radiomics_error` column.
  - No all-NaN columns.
  - Wall-thickness distributions match T12 expectations.
- [ ] Save `pipeline_info/` and `versions.yml`, and record the Nextflow run names in the log.

**Acceptance:** all four runs complete; the checks pass; `results/` is untouched (`git status` and mtimes unchanged).

---

## Phase 2: Dataset assembly

### T20: Cohort/metadata table
**Status:** TODO · **Pri:** P0 · **Target:** Oct 9 · **Depends on:** T02

**Tasks:**
- [ ] `hcmv/cohort.py` builds `results_hcm_vendor/tables/cohort.parquet` with these columns:
  - `subject_id`: globally unique, e.g. `mms2_071` or `acdc_patient101`.
  - `dataset`, `source_id`.
  - `disease` (NOR/HCM) and `y` (1 = HCM).
  - `vendor`, mapped to `Siemens|Philips|GE`.
  - `scanner`, `field_T`.
  - `role`: `train_pool` for Siemens and Philips M&Ms-2, `ge_check` for GE M&Ms-2, `external` for ACDC.
  - `mms2_challenge_split`: informational only.
- [ ] M&Ms-2 source: `dataset_information.csv`. Drop empty rows and zero-pad IDs.
- [ ] ACDC source: Info.cfg `Group`. `vendor=Siemens`; `field_T` is NaN (unknown per subject).
- [ ] Hard assertions on every count in "Verified facts".
- [ ] Unit test on a small fixture CSV with padding rows.

**Acceptance:** the table builds and the assertions pass.

### T21: Feature-table builder
**Status:** TODO · **Pri:** P0 · **Target:** Oct 13 · **Depends on:** T14, T20

**Tasks:**
- [ ] `hcmv/features.py`: `build_feature_table(dataset, source={nnformer,gt}, cfg={norm,raw})`.
  1. Read the per-phase CSVs and take the ID **from the filename**.
  2. Merge ED and ES into one row per subject.
  3. Drop meta columns (`*_mask_file`, `*_mask_source`, `*_voxel_volume_ml`, `*_patient_id`, `*_phase`) and the trailing duplicate `myocardial_mass_g`.
  4. Drop exact-duplicate and constant columns, and log what was dropped.
- [ ] Derived features:
  - `lv_edv, lv_esv, rv_edv, rv_esv`, as aliases.
  - `lv_sv = lv_edv − lv_esv`; `rv_sv` likewise.
  - `lvef = 100·lv_sv/lv_edv`; `rvef` likewise.
  - `mass_to_volume = ed_myocardial_mass_g / lv_edv`.
- [ ] Guard divide-by-zero by setting NaN, and flag those subjects.
- [ ] No BSA indexing, because M&Ms-2 has no height or weight.
- [ ] Family map, by regex on the column name:
  - **clinical `F_c`** = ED/ES volumes, mass, wall-thickness mean/max, plus the derived features. `wall_thickness_p95` goes to a separate `sensitivity` group.
  - **shape `F_s`** = `radiomics_original_shape_*`.
  - **texture `F_t`** = `radiomics_original_firstorder_*` + `radiomics_original_glcm_*`.
- [ ] Expected size before pruning: 2×62 + 5 derived ≈ 129.
- [ ] Output: `results_hcm_vendor/tables/features_<dataset>_<source>_<cfg>.parquet`, joined to the cohort table.
- [ ] Also write `data_dictionary.csv` (feature, family, phase, unit, description).
- [ ] Unit tests with fixture CSVs for the ID parsing, merge, derived-feature arithmetic, family assignment and duplicate removal.

**Acceptance:** one row per subject; 135 M&Ms-2 + 20 ACDC; the family counts are logged; the tests pass.

### T22: Data QC + exploratory report
**Status:** TODO · **Pri:** P1 · **Target:** Oct 15 · **Depends on:** T21

**Tasks:**
- [ ] Decide D6 (failed-case rule) with the user, apply it, and log the excluded subjects with reasons. Report nnFormer Dice by vendor from `results/MMS_2/comparison/aggregated_metrics.csv` as QC. That file is in long format with unpadded IDs, so normalize them.
- [ ] Missingness, and distributions by vendor × disease for the key clinical features.
- [ ] Univariate HCM-vs-NOR AUC per feature, within each vendor.
- [ ] NOR-only vendor effect size (η² or Kruskal–Wallis) per feature and per family.
- [ ] Feature correlation clustermap; PCA scatter coloured by vendor and by disease.
- [ ] Raw vs normalized texture comparison, showing how normalization shrinks the vendor separation.
- [ ] Output: `results_hcm_vendor/qc/` (figures + `qc_summary.md`). Some of these feed the midterm.

**Acceptance:** the report is generated by one command; key observations are copied into this ticket's log.

---

## Phase 3: Modelling framework

### T30: CV splitting + leakage-safe preprocessing
**Status:** TODO · **Pri:** P0 · **Target:** Oct 15 · **Depends on:** T21

**Tasks:**
- [ ] `splits.py`:
  - Outer: `RepeatedStratifiedKFold(5, 5)`. For pooled E1, stratify on `y × vendor` so every fold keeps the vendor mix.
  - Inner: `StratifiedKFold(5)`.
  - All seeded from config.
  - A single split = one subject per row, so splitting is patient-level by construction.
- [ ] `preprocessing.py`, all steps inside the sklearn `Pipeline`, fitted on training folds only:
  1. `FamilySelector(families)`.
  2. `SimpleImputer(median)`.
  3. `VarianceThreshold(0)`.
  4. `CorrelationFilter(threshold=0.95)`: a custom transformer that keeps the first of each correlated pair, in a deterministic order. Implement `get_feature_names_out`.
  5. `StandardScaler`.
- [ ] Tests:
  - Leakage: fitting the pipeline on train changes nothing when the test rows are altered.
  - The correlation filter behaves deterministically.
  - `get_feature_names_out` round-trips.

**Acceptance:** the tests pass; a pipeline object can be cloned by GridSearchCV.

### T31: Model zoo + hyperparameter grids
**Status:** TODO · **Pri:** P0 · **Target:** Oct 17 · **Depends on:** T30

**Models** (all class-balanced, since vendor class ratios differ: Siemens 30/22, Philips 27/35):
- **Elastic-net LR:** `LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000, class_weight='balanced')`. Grid: `C ∈ logspace(-3, 2, 8)`, `l1_ratio ∈ {0.1, 0.5, 0.9}`.
- **RBF SVM:** `SVC(kernel='rbf', probability=True, class_weight='balanced')`. Grid: `C ∈ {0.1, 1, 10, 100}`, `gamma ∈ {'scale', 1e-3, 1e-2, 1e-1}`. Here `gamma = 1/σ²` in the proposal's notation. Platt scaling gives the probabilities needed for the Brier score.
- **Random forest:** `n_estimators=500, class_weight='balanced'`. Grid: `max_depth ∈ {None, 3, 5}`, `max_features ∈ {'sqrt', 0.3}`, `min_samples_leaf ∈ {1, 3, 5}`.
- **Gradient-boosted trees (XGBoost):** `tree_method='hist', n_jobs=1, subsample=0.8, colsample_bytree=0.8`, with `scale_pos_weight` set from the training fold. Grid: `n_estimators ∈ {100, 300}`, `learning_rate ∈ {0.03, 0.1}`, `max_depth ∈ {2, 3}`, `min_child_weight ∈ {1, 3}`.
- Inner-CV scoring is `roc_auc`. Grids live in `configs/study.yaml`. Use `RandomizedSearchCV` only if the runtime budget in T34 demands it.

**Tasks:**
- [ ] `models.py`: `get_model(name) -> (estimator, param_grid)`.
- [ ] Tests: each model fits and predicts probabilities on a toy dataset inside the full pipeline.

**Acceptance:** all four models work in `GridSearchCV(Pipeline)`.

### T32: PyTorch MLP as a scikit-learn estimator
**Status:** TODO · **Pri:** P0 · **Target:** Oct 18 · **Depends on:** T30

**Spec:**
- `TorchMLPClassifier(BaseEstimator, ClassifierMixin)`.
- Architecture: 2 hidden layers with ReLU and dropout, and a single logit output (sigmoid happens in `predict_proba`).
- Loss: `BCEWithLogitsLoss(pos_weight)`. Optimizer: Adam with `lr=1e-3` and `weight_decay`.
- Early stopping on a stratified 20% validation split carved *from the training data it is given*, so it stays inside the fold. `max_epochs=1000`, `patience=50`; restore the best weights.
- Full-batch or batch size 32.
- Deterministic seeding (`torch.manual_seed`, `use_deterministic_algorithms`). `device='cpu'` by default.
- Implements `fit / predict_proba / predict / classes_`, with a picklable state.
- Grid: `hidden ∈ {(32,16), (64,32)}`, `dropout ∈ {0.2, 0.5}`, `weight_decay ∈ {1e-4, 1e-3, 1e-2}`.

**Tasks:**
- [ ] `mlp.py` plus tests:
  - The relevant subset of sklearn `check_estimator`, or at least `clone` / `get_params`.
  - It learns a separable toy set to AUC > 0.95.
  - It gives identical predictions for the same seed.
  - It is usable in `GridSearchCV`.

**Acceptance:** the tests pass; one outer fold of E1 with the MLP runs in reasonable time (log the time).

### T33: Metrics, bootstrap CIs, paired bootstrap tests
**Status:** TODO · **Pri:** P0 · **Target:** Oct 18 · **Depends on:** T02

**Tasks:**
- [ ] `stats.py`:
  - Metrics: ROC-AUC, balanced accuracy, sensitivity, specificity and Brier score. The threshold is decided in D7 (recommended 0.5).
  - Also calibration-curve data, for figures.
- [ ] Bootstrap CI:
  - 2000 resamples, stratified by class, with the percentile method. Optionally BCa, per Carpenter & Bithell.
  - For **repeated CV**: resample *subjects*, compute the metric in each outer repeat on the resampled subjects' OOF predictions, then average over repeats. This gives a single CI that accounts for the repeats.
- [ ] **Paired bootstrap** for differences (model A vs B, family set A vs B, and Δ):
  - Resample the same subjects for both prediction sets.
  - CI of the difference.
  - Two-sided p = 2·min(P(d ≤ 0), P(d ≥ 0)).
- [ ] Permutation-test helper for T42 (p = (k+1)/(n+1)).
- [ ] Multiple-comparison helper (Holm).
- [ ] Tests: AUC against sklearn; a CI covers the true value on simulated data about 95% of the time (a quick simulation); paired-test sanity checks (identical predictions give p≈1).

**Acceptance:** the tests pass; the API is documented in docstrings.

### T34: Experiment runner, result store, SLURM scripts
**Status:** TODO · **Pri:** P0 · **Target:** Oct 21 · **Depends on:** T31, T32, T33

**Tasks:**
- [ ] `runner.py` with two primitives:
  - `run_nested_cv(table, cohort_filter, families, model, cfg)`. Writes OOF predictions (`subject_id, repeat, fold, y, prob, vendor`), the chosen hyperparameters per fold, inner-CV scores and fit times.
  - `run_transfer(table, train_filter, test_filter, families, model, cfg, seeds)`.
    - Tune with inner CV on the training set, refit on all of it, then predict the test set.
    - Stochastic models (RF, XGB, MLP) are averaged over 5 seeds.
    - Save the fitted pipelines (joblib) for SHAP.
- [ ] Result store: `results_hcm_vendor/runs/<experiment>/<family_set>/<model>/<cfg>/`. It holds `predictions.parquet`, `hyperparams.json`, `metrics.json` and `manifest.json`.
- [ ] If a result already exists with the same config hash, skip it (resumable).
- [ ] Parallelize over outer folds with joblib. Add `hcm_vendor/scripts/run_experiment.sbatch` (CPU, e.g. 32 cores and 8 h), which sources `env.sh`.
- [ ] `--smoke` mode with tiny grids and 1×2 CV, for tests and quick checks.
- [ ] Estimate the runtime from a smoke run and log it here.

**Acceptance:** a smoke run of E1 for all five models completes end-to-end on a compute node; the outputs are loadable; a rerun is a no-op.

---

## Phase 4: Experiments

**Common rules for E1–E5:**
- Feature config is `norm` unless stated otherwise.
- The primary family set is **All**. E4 repeats E1–E3 for the other sets.
- There are five models across the four families in the proposal: LR-EN, SVM, RF, XGB and MLP. RF and XGB together are the "tree ensembles" family.

### T40: E1, pooled + within-vendor nested CV
**Status:** TODO · **Pri:** P1 · **Target:** Oct 25 · **Depends on:** T34

**Cohorts:**
- **Pooled:** Siemens + Philips, n=114.
- **Siemens-only:** n=52.
- **Philips-only:** n=62.

Nested CV runs on each cohort: outer 5×5, inner 5.

**Tasks:**
- [ ] Run all 5 models × 3 cohorts on `All`.
- [ ] Report two readings of "within-vendor performance":
  - (a) Train and test inside one vendor. This is what Δ needs in E2.
  - (b) Pooled-trained OOF predictions, split by vendor.
- [ ] Table: AUC, balanced accuracy, sensitivity, specificity and Brier, each with a 95% bootstrap CI. Also ROC curves.
- [ ] Paired bootstrap comparisons between models on the pooled cohort.

**Acceptance:** the tables and figures are in `results_hcm_vendor/runs/E1/` and the headline numbers are copied into this log.

**Note:** the within-vendor training folds have only about 42 subjects each, so expect wide CIs and say so.

### T41: E2, cross-vendor transfer + generalization gap Δ
**Status:** TODO · **Pri:** P1 · **Target:** Oct 28 (preliminary for midterm; final by Nov 8) · **Depends on:** T40

**Tasks:**
- [ ] Transfers: Siemens → Philips and Philips → Siemens, all 5 models, on `All`. Uses `run_transfer`.
- [ ] Δ_{A→B} = AUC_{B→B} − AUC_{A→B}:
  - AUC_{B→B} is E1's within-vendor-B nested-CV AUC, reading (a).
  - Both terms are evaluated on the same B subjects.
  - Δ CI and p-value come from a paired bootstrap over the B subjects. Bootstrap B, recompute both AUC terms (the E1 term averaged over repeats, as in T33), then take the difference.
- [ ] Also report the shifts in balanced accuracy, sensitivity, specificity and Brier at the fixed threshold. Calibration can shift even when AUC does not, so add a calibration plot.
- [ ] Save the fitted Siemens-trained and Philips-trained pipelines for T50.

**Acceptance:** a Δ table (model × direction) with CIs, plus ROC overlays of within vs cross vendor.

### T42: E3, vendor probe on NOR only + permutation test
**Status:** TODO · **Pri:** P1 · **Target:** Oct 29 (preliminary for midterm; final by Nov 8) · **Depends on:** T34

**Design (confirm D8):**
- Data: NOR subjects only, n=75 (Siemens 22, Philips 35, GE 18), so disease cannot contribute.
- Target: vendor.
- **Primary:** a 3-class balanced accuracy, where chance = 1/3.
- **Secondary:** binary Siemens-vs-Philips (n=57).
- Feature inputs, one probe per family: `F_c`, `F_s`, `F_t(norm)`, `F_t(raw)` and `All`.
- Classifiers:
  - **Primary:** multinomial L2 logistic regression with C tuned inside (nested, small grid; cheap).
  - **Secondary:** RF with fixed hyperparameters.
- Evaluation: repeated stratified 5-fold CV of balanced accuracy.
- Permutation test: 1000 vendor-label shuffles over the same CV splits, e.g. `permutation_test_score`, or the T33 helper when nesting is involved. p = (k+1)/(1000+1).
- Holm correction across families.

**Tasks:**
- [ ] Implement it, run it, and plot each family's observed balanced accuracy against its null distribution.

**Acceptance:** a table of family, balanced accuracy with CI, permutation p and Holm-adjusted p; the plot is saved.

**Note:** all NOR subjects are at 1.5 T, so field strength cannot confound the vendor signal. Scanner models are nested within vendor; see S4.

### T60: Midterm update (due Sun Nov 1)
**Status:** TODO · **Pri:** P0 · **Target:** Nov 1 · **Depends on:** T22, T40; T41 and T42 preliminary

**Tasks:**
- [ ] Ask the user for the format and length (D10), and whether the course provides a template.
- [ ] Content:
  - Progress against the proposal.
  - **Method deviations and why:** the wall-thickness fix (D1), texture normalization with raw kept as sensitivity (D2), and the failed-case rule (D6).
  - The cohort table, QC highlights (T22), E1 results, and preliminary E2 Δ and E3 results.
  - The remaining plan and risks.
- [ ] Source goes in `hcm_vendor/reports/midterm/`. Figures are regenerated by command, not pasted by hand.

**Acceptance:** the user signs off; the submitted version's commit SHA is recorded here.

### T43: E4, feature-family ablation (+ raw vs normalized texture)
**Status:** TODO · **Pri:** P1 · **Target:** Nov 6 · **Depends on:** T40, T41, T42

**Tasks:**
- [ ] Repeat E1 and E2 for the family sets `F_c`, `F_c∪F_s`, `F_c∪F_t` and `All`. E3 per family is already covered in T42.
- [ ] Sensitivity: repeat every set that contains texture with the `raw` config.
- [ ] Key figure: Δ (with CI) by family set × model, for both directions. This tests the hypothesis that clinical features transfer with a small Δ and texture has the largest.
- [ ] Paired bootstrap tests of Δ between family sets. Use the same B subjects, so the comparison is paired. Holm-adjust.

**Acceptance:** the ablation tables and figure are done, with the hypothesis-relevant contrasts spelled out in the log.

**Compute note:** this is the most expensive stage: 4 sets × 5 models × (3 E1 cohorts + 2 E2 directions) × 2 configs for texture. Check the T34 runtime estimate first. If it is too slow, restrict the `raw` sensitivity to the best two models.

### T44: E5, external test on ACDC
**Status:** TODO · **Pri:** P1 · **Target:** Nov 8 · **Depends on:** T34

**Tasks:**
- [ ] Train on **all** M&Ms-2 NOR/HCM subjects (n=135, including GE, as in the proposal). Tune with inner CV, refit, and test on the 20 ACDC subjects.
- [ ] All models; family sets `All` and `F_c` at minimum, or all four if cheap.
- [ ] Metrics with CIs. They will be very wide at n=20, so state that. Also the ROC curve and a per-subject prediction table.
- [ ] Caveats to report:
  - The ACDC masks come from an nnFormer trained on ACDC training data, so the segmentations are in-domain and likely better than on M&Ms-2.
  - ACDC includes 3 T scans.

**Acceptance:** an E5 table and figure; the caveats written into the results notes.

### T45: GE specificity check
**Status:** TODO · **Pri:** P1 · **Target:** Nov 8 · **Depends on:** T34

**Tasks:**
- [ ] Use the model trained on the pooled Siemens+Philips set (n=114; tuned by inner CV, then refit).
- [ ] Predict the 18 GE NOR subjects and report specificity with a Wilson 95% CI.
- [ ] Report the 3 GE HCM predictions descriptively only; no sensitivity claim.
- [ ] Do this for all models on `All` and `F_c`.

**Acceptance:** a small table and one sentence per model in the log.

---

## Phase 5: Interpretation

### T50: SHAP attributions + Siemens-vs-Philips stability
**Status:** TODO · **Pri:** P1 · **Target:** Nov 14 · **Depends on:** T41

**Tasks:**
- [ ] Load the E2 fitted pipelines: the Siemens-trained and Philips-trained model for each model type.
- [ ] Compute SHAP in the pipeline's post-preprocessing feature space, naming features via `get_feature_names_out`:
  - **TreeSHAP** (exact) for RF and XGB.
  - **KernelSHAP** for LR-EN, SVM and MLP. Background = `shap.kmeans(train, 20)`, with `nsamples` of about 2d + 2048.
  - Note in the write-up that `LinearExplainer` would be exact for LR; the proposal specifies KernelSHAP, and optionally compare the two.
- [ ] Explanation set, per D9: the same 114 pooled subjects for both models.
- [ ] Global importance = mean |φ_j|.
- [ ] Handle the correlation filter. It may keep different representatives of a correlated group in each model. Aggregate |φ| to **correlation clusters**:
  - Hierarchical clustering on the pooled data, with a |r| > 0.95 cutoff.
  - This is interpretation only, not performance, so it cannot leak.
  - A feature dropped by one model counts as 0 before aggregation.
- [ ] Stability:
  - Spearman ρ of the cluster-importance rankings between the Siemens-trained and Philips-trained models, with a bootstrap CI over the explained subjects.
  - Top-10 Jaccard overlap.
  - Per-family share of total |φ|.
- [ ] Figures: a beeswarm per model and vendor, a rank-vs-rank scatter, and family-share bars.

**Acceptance:** a stability table (model × ρ with CI, Jaccard) and the figures are saved; findings noted in the log.

### T51: Statistical comparison summary + hypothesis verdicts
**Status:** TODO · **Pri:** P1 · **Target:** Nov 16 · **Depends on:** T43, T44, T45, T50

**Tasks:**
- [ ] One consolidated results table with all paired tests and Holm-adjusted p-values, grouped by question.
- [ ] Verdicts, with effect sizes and CIs:
  - **H1:** clinical features transfer with a small Δ.
  - **H2:** texture is the most vendor-predictive family (E3).
  - **H3:** texture has the largest Δ.
  - **H4:** SHAP reliance is consistent across vendors.
- [ ] Write limitations notes:
  - Small n.
  - GE has only 3 HCM.
  - ACDC is small, and its segmentations are in-domain.
  - No BSA indexing.
  - M&Ms-1 was excluded because of overlap with M&Ms-2.
  - The effect of the wall-thickness definition.
  - Texture preprocessing dependence (raw vs norm).

**Acceptance:** `results_hcm_vendor/tables/summary.md` exists and the user has reviewed the verdicts.

---

## Phase 6: Deliverables

### T61: Final figures + tables
**Status:** TODO · **Pri:** P1 · **Target:** Nov 20 · **Depends on:** T51

**Tasks:**
- [ ] `python -m hcmv figures` and `python -m hcmv tables` regenerate everything from the result store.
- [ ] Figures:
  1. Updated workflow figure.
  2. Cohort/QC panel.
  3. E1 ROC curves.
  4. E2/E4 Δ forest plot by family set.
  5. E3 null-vs-observed plot.
  6. E5 ROC curve.
  7. SHAP rank-rank plot and family shares.
  8. Calibration shift.
- [ ] Tables: cohort, E1, E2 Δ, E3, E4 contrasts, E5, GE check, SHAP stability.
- [ ] Use a consistent vendor colour map and export both PDF and PNG.

**Acceptance:** everything is regenerated by one command into `results_hcm_vendor/figures/` and `results_hcm_vendor/tables/`.

### T62: Final report
**Status:** TODO · **Pri:** P0 · **Target:** Nov 26 (due Nov 27) · **Depends on:** T61

**Tasks:**
- [ ] Confirm the format and length requirements with the user.
- [ ] Sections: introduction; data (cohort table); methods (SORAT, feature fixes and deviations from the proposal, models, protocol, statistics); results for E1–E5 and SHAP; discussion; limitations; conclusion; AI-use statement (as in the proposal).
- [ ] Source goes in `hcm_vendor/reports/final/`. Every number is traceable to a result-store file.

**Acceptance:** the user has signed off; the submission commit is tagged (e.g. `eece568-final`).

### T63: Reproducibility pass
**Status:** TODO · **Pri:** P1 · **Target:** Nov 25 · **Depends on:** T51

**Tasks:**
- [ ] `hcm_vendor/README.md`: environment setup (T01), the feature re-extraction (T14), and one-command analysis with `hcm_vendor/scripts/run_all.sh`, an sbatch chain with dependencies.
- [ ] Clean rerun of at least E1 and E2 from scratch, checking that the metrics match the stored ones within seed determinism.
- [ ] All tests green, both repo `tests/` and `hcm_vendor/tests/`.
- [ ] No stray files committed: nothing from `results_hcm_vendor/`, `work/` or the NIfTI files.
- [ ] Update `CLAUDE.md` with a short `hcm_vendor/` section.

**Acceptance:** a fresh rerun reproduces the headline numbers; the README is followed end-to-end once.

---

## Stretch (P2, only after T51 is on track)

- **S1: ground-truth-mask features as a reference.** Rerun E1 and E2 (and E3) on the T12 ground-truth features. The comparison separates how much of Δ and of the vendor signal comes from *segmentation* differences and how much from *image* differences. This is high value, and most of the data already exists from T12.
- **S2: segmenter sensitivity.** Repeat E1 and E2 with CineMA ACDC-ensemble features. That needs a T14-style re-extraction. Exclude the `cinema__mnms*` models: M&Ms-1 and M&Ms-2 training data overlap the M&Ms-2 NOR/HCM subjects, so they leak.
- **S3: harmonization baseline.** Apply ComBat to features in the E2 setting and see whether Δ closes. It must be fitted without target labels, and the leakage caveats documented.
- **S4: scanner-level probe.** Within Siemens, predict the scanner model on NOR subjects (small n; descriptive).
- **S5: AHA-segment wall thickness.** Measure max wall thickness per AHA 16-segment region, which is closer to clinical reporting.
- **S6: upstream PR.** Split T10/T11 into a clean SORAT PR to `main`, with the tests and a README feature-extraction note, separate from the analysis code.
