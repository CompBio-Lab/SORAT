# Parameters

Pass any parameter on the command line (`--name value`) or set it in `.sorat/user.config`. Where an environment variable is named, it can be used instead.

## Inputs and models

| Parameter | Default | Description |
|---|---|---|
| `--input` | none | Samplesheet CSV (see [Usage](usage.md#samplesheet)); required when mixing short-axis and atrial models |
| `--outdir` | `./results` | Output directory |
| `--models` | `all` | Models to run: `cinema`, `nnformer`, `vsa3l`, `atrial_nnunet`, or `all` |
| `--default_inputs.sax` | `null` | Preferred explicit default samplesheet for SAX-only runs (or `SORAT_SAX_SAMPLESHEET`) |
| `--sax_data_root` | `null` | Preferred SAX data root used to auto-generate defaults (ACDC-style layout) (or `SORAT_SAX_DATA_ROOT`) |
| `--sax_data_split` | `testing` | SAX split folder used with `--sax_data_root` when auto-generating defaults |
| `--default_inputs.atrial` | `null` | Preferred explicit default samplesheet for atrial-only runs (or `SORAT_ATRIAL_SAMPLESHEET`) |
| `--atrial_nnunet.dataset_root` | `null` | Preferred atrial data root for auto-generated defaults. Supports nnUNet layout (`imagesTr/`,`labelsTr/`) and MBAS-style layout (`MBAS_###/MBAS_###_gt.nii.gz`,`MBAS_###_label.nii.gz`) (or `SORAT_ATRIAL_DATA_ROOT`) |
| `--frames_mode` | `'auto'` | Frame selection: `auto` (ED/ES when info_cfg present, else all frames), `ed_es` (force 2 frames), `all` (always all frames) |
| `--max_frames` | `null` | Cap to N frames in all-frames mode by subsampling (e.g. `10`) |
| `--auto_discover_models` | `true` | Find every checkpoint inside each container; `false` uses the seeds listed in `nextflow.config` |

## Evaluation

| Parameter | Default | Description |
|---|---|---|
| `--compare` | `true` | Generate comparison report |
| `--inference_only` | `false` | Run inference only (skip metrics + report generation) |
| `--evaluation.label_schema` | `architecture_default` | Evaluation contract; use `atrial_binary_union` when a binary biatrial reference is compared with atrial labels 1--3 collapsed to foreground, or `atrial_la_cavity` when a binary left-atrial cavity reference (e.g. 2018 Atrial Segmentation Challenge) is compared with atrial label 3 |
| `--evaluation.gt_label_map` | `auto` | Raw ventricular GT labels, e.g. `rv=1,myo=2,lv=3` (ACDC) or `lv=1,myo=2,rv=3` (M&Ms, M&Ms-2). `auto` infers them from anatomy and logs a warning when the inference is ambiguous |

## Optional steps

| Parameter | Default | Description |
|---|---|---|
| `--postprocess.enabled` | `false` | Enable optional LV-intensity postprocessing (LV dark regions -> MYO) |
| `--postprocess.use_for_metrics` | `false` | If postprocess enabled, also score the corrected segmentations, reported as `<model>_pp` next to the raw model output (raw output is always scored) |
| `--postprocess.visualize` | `true` | Generate before/after/delta postprocess visualizations |
| `--visualization.enabled` | `true` | Generate ED/ES previews for each model output. If GT exists, previews include prediction-vs-GT overlays, per-structure DSC, and difference maps |
| `--debug` | `false` | Generate debug analytics report (execution/runtime/GPU/scalability/success + scientific utility metrics) |
| `--feature_extraction.enabled` | `false` | Enable interpretable feature extraction (volumes, wall thickness, radiomics) in the main workflow |
| `--feature_extraction.mask_source` | `prompt` | Feature mask source: `predictions`, `postprocess`, or `prompt` |
| `--feature_extraction.virtualenv_path` | `null` | Path to a pre-built virtualenv containing `pyradiomics` for feature extraction jobs |
| `--feature_extraction.require_virtualenv` | `false` | If `true`, fail fast unless `--feature_extraction.virtualenv_path` is provided and valid |
| `--feature_extraction.samplesheet` | `null` | Optional samplesheet override for `-entry FEATURES_ONLY` |
| `--feature_extraction.results_dir` | `null` | Existing results directory to read segmentations from in `-entry FEATURES_ONLY` |
| `--feature_extraction.output_dir` | `null` | Directory for the feature CSVs. When unset, `-entry FEATURES_ONLY` writes to this run's `<outdir>/features/raw` (or `features/postprocessed`), never into `--feature_extraction.results_dir`; the main workflow writes under its results directory |
| `--feature_extraction.model_tag` | `null` | Exact model tag(s) to extract features for, comma-separated (e.g. `nnformer__fold0`) |
| `--feature_extraction.seed` | `ensemble` | Variant choice when `model_tag` is unset: `ensemble`, `all`, or a seed such as `seed0` |

Radiomics settings (`--feature_extraction.radiomics.*`) are listed in [Features](features.md#radiomics-settings).

## Execution

| Parameter | Default | Description |
|---|---|---|
| `--slurm_account` | `null` | SLURM allocation/account (required when `-profile slurm`, can use `SORAT_SLURM_ACCOUNT`) |
| `--vsa3l_container` | `containers/sorat-vsa3l.sif` | Validated VSA-3L Apptainer image (can use `SORAT_VSA3L_CONTAINER`) |
| `--singularity_cache_dir` | `$HOME/.singularity_cache` | Per-user Singularity cache location (can use `SORAT_SINGULARITY_CACHEDIR`) |
| `--preprocess_cache_enabled` | `true` | Reuse model-specific preprocessing outputs across reruns |
| `--preprocess_cache_dir` | `${projectDir}/.cache/preprocess` | Cache root for reusable preprocessing artifacts |
| `--slurm_max_forks` | `30` | Maximum concurrent task submissions in `slurm` profile |
| `--slurm_queue_size` | `64` | Max tasks queued/submitted to executor at once |
| `--slurm_submit_rate` | `50/1min` | Submission throttling rate to reduce scheduler pressure |
| `--slurm_poll_interval` | `30 sec` | Job-status polling interval |
| `--slurm_queue_stat_interval` | `60 sec` | Queue-stat refresh interval |
| `--max_cpus`, `--max_memory`, `--max_time` | `16`, `128.GB`, `24.h` | Upper limits for any single task |
