# Usage

## Samplesheet

| Column | Required | Description |
|---|---|---|
| `patient_id` | yes | Unique identifier, used in every output file name |
| `image` | yes | 4-D cine or 3-D image: NIfTI, another format SimpleITK reads (NRRD, MetaImage), or a DICOM series directory |
| `ground_truth` | no | Reference masks: an image file, or a directory of per-frame files (ACDC and M&Ms-2 layouts). Leave empty to segment only |
| `info_cfg` | no | ACDC-style `Info.cfg` with ED and ES frame numbers |
| `image_spacing` | no | Spacing `XxYxZ` in mm (e.g. `0.625x0.625x1.25`). It replaces the spacing stored in the image and reference; use it only when the files hold a placeholder |
| `dicom_series` | no | SeriesInstanceUID, when a DICOM directory holds more than one series |

```csv
patient_id,image,ground_truth,info_cfg,image_spacing,dicom_series
patient101,/data/acdc/patient101/patient101_4d.nii.gz,/data/acdc/patient101,/data/acdc/patient101/Info.cfg,,
lasc001,/data/lasc/001/lgemri.nrrd,/data/lasc/001/laendo.nrrd,,0.625x0.625x1.25,
site001,/data/site/site001/cine_sax,,/data/site/site001/Info.cfg,,
```

Use absolute paths.

### Non-NIfTI inputs

Rows with a non-NIfTI image or reference, or with `image_spacing`, first go through `PREPARE_INPUT`. It writes NIfTI with the original geometry to `<outdir>/inputs/<patient_id>/`, together with a provenance file (source paths, SHA-256 checksums, reader, DICOM series UID and output geometry).

- **DICOM.** One image per slice position gives a 3-D volume. A cine series with several images per position gives a 4-D volume, with slices ordered along the slice normal and frames ordered by trigger time. RT-STRUCT and SEG contours are not read; provide references as image files.
- **Spacing.** SORAT never infers spacing. If every spacing is exactly 1.0 and no `image_spacing` is given, it prints a warning.

## Frames

`Info.cfg` gives ED and ES as frame numbers counted from 1, as defined by ACDC:

```text
ED: 1
ES: 13
```

SORAT converts them to array indices once. When a dataset provides per-frame phase images next to the cine (ACDC `frameNN` files, M&Ms-2 `_SA_ED`/`_SA_ES` files), SORAT checks that each selected frame matches the corresponding phase image, and stops with an error if it doesn't.

| `--frames_mode` | Behaviour |
|---|---|
| `auto` (default) | ED and ES when `info_cfg` is given, otherwise every frame |
| `ed_es` | ED and ES only |
| `all` | Every frame; `--max_frames N` subsamples |

## Labels and evaluation

All short-axis models output 1 RV, 2 MYO, 3 LV, and the atrial model outputs 1 wall, 2 RA, 3 LA. Reference label values differ between datasets, so declare them:

| Dataset | `--evaluation.gt_label_map` |
|---|---|
| ACDC | `rv=1,myo=2,lv=3` |
| M&Ms-1, M&Ms-2 | `lv=1,myo=2,rv=3` |

With the default, `auto`, the labels are inferred from anatomy and a warning is logged when the inference is ambiguous.

When the reference has fewer classes than the model, choose an evaluation schema:

| `--evaluation.label_schema` | Comparison |
|---|---|
| `architecture_default` | Each label against the same label |
| `atrial_la_cavity` | Predicted LA (label 3) against a binary LA-cavity reference |
| `atrial_binary_union` | Labels 1–3 merged, against a binary biatrial reference |

Metrics are computed in 3-D in the original image space: DSC, and HD95 in millimetres. A structure counts as a failure when DSC < 0.5, HD95 > 20 mm or the structure is missing.

## Profiles

| Profile | Runs |
|---|---|
| `local` | On this machine with Apptainer |
| `slurm` | As SLURM jobs with Apptainer; needs `--slurm_account` |
| `test` | CineMA on `test/samplesheet.csv`. Edit its placeholder paths first |

For large SLURM runs, `--slurm_max_forks`, `--slurm_queue_size` and `--slurm_submit_rate` control how fast jobs are submitted. `-resume` reuses finished tasks. Model-specific preprocessing is also cached across runs in `.cache/preprocess` (`--preprocess_cache_enabled`).

## Inputs without a samplesheet

When `--input` is omitted, SORAT builds a samplesheet:

- **Short-axis models** (`cinema`, `nnformer`, `vsa3l`, or `all`): from `--default_inputs.sax`, otherwise from an ACDC-style folder given by `--sax_data_root` and `--sax_data_split`.
- **Atrial model**: from `--default_inputs.atrial`, otherwise from `--atrial_nnunet.dataset_root` (nnU-Net `imagesTr/`/`labelsTr/` layout, or MBAS layout).

`--models all` runs the three short-axis models. A run that mixes short-axis and atrial models needs an explicit `--input`.

## Optional steps

| Option | Adds |
|---|---|
| `--postprocess.enabled true` | Relabels dark voxels at the LV border as myocardium (papillary muscles and trabeculae). The raw output is always scored; add `--postprocess.use_for_metrics true` to also score the corrected masks, reported as `<model>_pp` |
| `--feature_extraction.enabled true` | Volumes, EF, mass, wall thickness and radiomics per mask (see [Features](features.md)) |
| `--visualization.enabled false` | Turns off the ED/ES preview images, which are on by default |
| `--inference_only true` | Segmentation only, without metrics or reports |
| `--debug true` | Run-time, GPU and success analytics in `debug/` |

## Running one step on existing results

| Entry | Does |
|---|---|
| `-entry POSTPROCESS_ONLY --postprocess.results_dir <dir>` | Post-processes the segmentations of an earlier run |
| `-entry FEATURES_ONLY --feature_extraction.results_dir <dir>` | Extracts features from an earlier run's masks, written to this run's `--outdir` |
| `-entry DEBUG_ONLY` | Rebuilds the debug report from an earlier run |

These entry points read the samplesheet directly, so they expect NIfTI; point them at the converted files in `<outdir>/inputs/`.

## Troubleshooting

- **A task failed.** Its work directory (shown in the error) contains `.command.sh`, `.command.log` and `.command.err`. Fix the cause and rerun with `-resume`.
- **Frame check error.** The ED/ES numbers in `Info.cfg` don't match the dataset's phase images. Check whether the file counts frames from 0.
- **Swapped cavities or a label warning.** Set `--evaluation.gt_label_map` for the dataset.
- **No GPU in a task.** Run `validate_cuda_runtime.py` (see [Installation](installation.md#checking-the-gpu-runtime)), and check that GPU jobs request a GPU and run with `--nv`.
