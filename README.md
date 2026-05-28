# SORAT - Segmentation Orchestration and Reproducible Analysis Toolkit

[![Nextflow](https://img.shields.io/badge/nextflow-%E2%89%A523.04.0-brightgreen.svg)](https://www.nextflow.io/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![Singularity](https://img.shields.io/badge/singularity-enabled-blue.svg)](https://sylabs.io/)

## Overview

SORAT (Segmentation Orchestration and Reproducible Analysis Toolkit) is a modular Nextflow pipeline for cardiac MRI segmentation that integrates multiple deep learning models. It enables reproducible orchestration, side-by-side comparison of segmentation results, and comprehensive evaluation metrics.

### Supported Models

| Model | Description | Reference |
|-------|-------------|-----------|
| **CineMA** | Convolutional Vision Transformer for cardiac MRI | [CineMA Paper](https://arxiv.org/abs/2506.00679) |
| **nnFormer** | 3D medical image segmentation transformer | [Zhou et al., 2021](https://arxiv.org/abs/2109.03201) |
| **VSA-3L** | MONAI Ventricular Short Axis 3-Label model | [MONAI Model Zoo](https://monai.io/model-zoo.html) |
| **Atrial nnUNet** | nnUNetv2 2D atrial segmentation model (Dataset001_LGE) | [nnU-Net](https://www.nature.com/articles/s41592-020-01008-z) |

### Output Labels

SAX models (`cinema`, `nnformer`, `vsa3l`) produce ACDC-compatible labels:
- **0**: Background
- **1**: Right Ventricle (RV)
- **2**: Myocardium (MYO)
- **3**: Left Ventricle (LV)

Atrial model (`atrial_nnunet`) produces atrial labels:
- **0**: Background
- **1**: Wall
- **2**: Right Atrium (RA)
- **3**: Left Atrium (LA)

## Quick Start

### 1. Install Nextflow

```bash
curl -s https://get.nextflow.io | bash
```

### 2. Configure Local User Paths (Required)

```bash
python3 bin/sorat_setup.py
```

This writes local overrides to `.sorat/user.config` (gitignored).

### 3. Run the Pipeline

You can run SORAT directly with `nextflow run main.nf`, or use the helper runner:

```bash
./bin/sorat_run.sh [pipeline options]
```

The helper runner uses `main.nf` from the repo root and reminds you to run setup if no local user config is present.

**Using Default Inputs (Model-Dependent):**
```bash
# SAX-only runs default to generated samplesheet from --sax_data_root
./bin/sorat_run.sh -profile local

# With SLURM + Apptainer on HPC
./bin/sorat_run.sh -profile slurm

# Atrial-only runs default when atrial root data is configured
export SORAT_ATRIAL_DATA_ROOT=/path/to/nnUNet_raw/Dataset001_LGE
./bin/sorat_run.sh --models atrial_nnunet -profile slurm
```

Notes:
- If `--models` includes both SAX and atrial models, `--input` is required.
- `--models all` is SAX-only.

**Using Custom Data:**
```bash
# Run locally (Apptainer/Singularity)
./bin/sorat_run.sh \
    --input samplesheet.csv \
    --outdir results \
    --models all \
    -profile local

# Run on SLURM (HPC)
./bin/sorat_run.sh \
    --input samplesheet.csv \
    --outdir results \
    --models all \
    -profile slurm

# Run specific models only
./bin/sorat_run.sh \
    --input samplesheet.csv \
    --outdir results \
    --models cinema,nnformer \
    -profile local
```

### Pulling Containers (Apptainer)

```bash
# Pull SORAT images from GitHub Container Registry
export SORAT_GHCR_NAMESPACE="your-org"
apptainer pull sorat-cinema.sif docker://ghcr.io/${SORAT_GHCR_NAMESPACE}/sorat-cinema:latest
apptainer pull sorat-nnformer.sif docker://ghcr.io/${SORAT_GHCR_NAMESPACE}/sorat-nnformer:latest
apptainer pull sorat-vsa3l.sif docker://ghcr.io/${SORAT_GHCR_NAMESPACE}/sorat-vsa3l:latest
apptainer pull sorat-atrial-nnunet.sif docker://ghcr.io/${SORAT_GHCR_NAMESPACE}/sorat-atrial-nnunet:latest
```

### 4. Alternative: Jupyter Notebook Interface

For interactive use, open the Jupyter notebook:

```bash
jupyter notebook notebooks/run_sorat_pipeline.ipynb
```

The notebook provides:
- Visual data exploration
- Interactive model configuration
- Result visualization and comparison
- Direct model execution (without Nextflow)

## Input Format

### Samplesheet (CSV)

Create a CSV file with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| `patient_id` | Yes | Unique patient identifier |
| `image` | Yes | Path to 4D cardiac MRI NIfTI file |
| `ground_truth` | No | Path to ground truth segmentation |
| `info_cfg` | No | Optional metadata config with ED/ES frame indices (for CineMA compatibility) |

#### Example samplesheet.csv

```csv
patient_id,image,ground_truth,info_cfg
patient101,/data/patient101/patient101_4d.nii.gz,/data/patient101,/data/patient101/Info.cfg
patient102,/data/patient102/patient102_4d.nii.gz,/data/patient102,/data/patient102/Info.cfg
patient103,/data/patient103/patient103_4d.nii.gz,,
```

### Custom Dataset Templates

Use these templates when your directory structure does not match ACDC or MBAS conventions.

#### 1. Minimal generic template (any dataset structure)

```csv
patient_id,image,ground_truth,info_cfg
case001,/data/custom/case001/image.nii.gz,,
case002,/data/custom/case002/image.nii.gz,,
```

Use this when you only need inference. You can point `image` to any valid NIfTI path.

#### 2. Recommended per-model contributor template

```csv
patient_id,image,ground_truth,info_cfg
case001,/data/custom/case001/image_4d.nii.gz,/data/custom/case001/label_dir,/data/custom/case001/Info.cfg
case002,/data/custom/case002/image_4d.nii.gz,/data/custom/case002/label_dir,
```

Guidance:
- Always provide stable `patient_id` values; they are used in outputs and comparisons.
- Set `ground_truth` when you want metrics/reporting; leave empty for inference-only runs.
- `info_cfg` is optional and mainly useful for CineMA ED/ES frame metadata.
- Keep column names unchanged even if your on-disk folder names are different.

### Info.cfg Format (Optional)

The Info.cfg file specifies ED and ES frame indices:

```
ED: 0
ES: 12
Group: NOR
Height: 175
Weight: 70
```

## Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `--input` | Optional override for input samplesheet CSV (required for mixed SAX+atrial runs) |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--outdir` | `./results` | Output directory |
| `--models` | `all` | Models to run: `cinema`, `nnformer`, `vsa3l`, `atrial_nnunet`, or `all` |
| `--default_inputs.sax` | `null` | Preferred explicit default samplesheet for SAX-only runs (or `SORAT_SAX_SAMPLESHEET`) |
| `--sax_data_root` | `null` | Preferred SAX data root used to auto-generate defaults (ACDC-style layout) (or `SORAT_SAX_DATA_ROOT`) |
| `--sax_data_split` | `testing` | SAX split folder used with `--sax_data_root` when auto-generating defaults |
| `--default_inputs.atrial` | `null` | Preferred explicit default samplesheet for atrial-only runs (or `SORAT_ATRIAL_SAMPLESHEET`) |
| `--atrial_nnunet.dataset_root` | `null` | Preferred atrial data root for auto-generated defaults. Supports nnUNet layout (`imagesTr/`,`labelsTr/`) and MBAS-style layout (`MBAS_###/MBAS_###_gt.nii.gz`,`MBAS_###_label.nii.gz`) (or `SORAT_ATRIAL_DATA_ROOT`) |
| `--slurm_account` | `null` | SLURM allocation/account (required when `-profile slurm`, can use `SORAT_SLURM_ACCOUNT`) |
| `--singularity_cache_dir` | `$HOME/.singularity_cache` | Per-user Singularity cache location (can use `SORAT_SINGULARITY_CACHEDIR`) |
| `--compare` | `true` | Generate comparison report |
| `--inference_only` | `false` | Run inference only (skip metrics + report generation) |
| `--debug` | `false` | Generate debug analytics report (execution/runtime/GPU/scalability/success + scientific utility metrics) |
| `--postprocess.enabled` | `false` | Enable optional LV-intensity postprocessing (LV dark regions -> MYO) |
| `--postprocess.use_for_metrics` | `true` | If postprocess enabled, compute metrics on corrected segmentations |
| `--postprocess.visualize` | `true` | Generate before/after/delta postprocess visualizations |
| `--visualization.enabled` | `true` | Generate ED/ES previews for each model output. If GT exists, previews include prediction-vs-GT overlays, per-structure DSC, and difference maps |
| `--slurm_max_forks` | `30` | Maximum concurrent task submissions in `slurm` profile |
| `--slurm_queue_size` | `64` | Max tasks queued/submitted to executor at once |
| `--slurm_submit_rate` | `50/1min` | Submission throttling rate to reduce scheduler pressure |
| `--slurm_poll_interval` | `30 sec` | Job-status polling interval |
| `--slurm_queue_stat_interval` | `60 sec` | Queue-stat refresh interval |
| `--preprocess_cache_enabled` | `true` | Reuse model-specific preprocessing outputs across reruns |
| `--preprocess_cache_dir` | `${projectDir}/.cache/preprocess` | Cache root for reusable preprocessing artifacts |

Note: `--models all` currently runs SAX models (`cinema`, `nnformer`, `vsa3l`) and does not automatically include `atrial_nnunet`.

### Default Input Selection Rules

When `--input` is omitted, SORAT resolves input automatically:
- SAX-only runs (`cinema`, `nnformer`, `vsa3l`, or `all`) use `--default_inputs.sax` if set; otherwise SORAT generates a SAX default samplesheet from `--sax_data_root` and `--sax_data_split`.
- Atrial-only runs (`atrial_nnunet`) use `--default_inputs.atrial` if set; otherwise SORAT auto-generates an atrial default samplesheet from `--atrial_nnunet.dataset_root`.
- For MBAS-style atrial roots, SORAT automatically stages nnUNet-style paths (`imagesTr/*_0000.nii.gz`, `labelsTr/*.nii.gz`) in `.cache/generated_inputs/` and uses those paths in the generated samplesheet.
- Mixed SAX+atrial runs fail fast and require explicit `--input`.

### Advanced Options (Quick Reference)

Use these options when needed:
- `--inference_only true`: skip metrics/report generation.
- `--debug true`: generate debug analytics outputs.
- `--preprocess_cache_enabled true|false`: enable or disable preprocessing cache reuse.
- `--postprocess.enabled true`: enable optional LV -> MYO postprocessing.

For SLURM tuning, the main controls are `--slurm_max_forks`, `--slurm_queue_size`, `--slurm_submit_rate`, `--slurm_poll_interval`, and `--slurm_queue_stat_interval`.

```bash
nextflow run main.nf \
    --input samplesheet.csv \
    --models all \
    --inference_only true \
    --debug false \
    -profile slurm
```

## Output Structure

```
results/
├── cinema/
│   ├── preprocessed/           # Preprocessed data
│   └── segmentations/          # Model predictions
├── nnformer/
│   ├── preprocessed/
│   └── segmentations/
├── vsa3l/
│   ├── preprocessed/
│   └── segmentations/
├── atrial_nnunet/
│   ├── preprocessed/
│   └── segmentations/
├── metrics/
│   ├── cinema/                 # Per-model metrics
│   ├── nnformer/
│   ├── vsa3l/
│   └── atrial_nnunet/
├── comparison/
│   ├── aggregated_metrics.csv  # All metrics combined
│   ├── model_comparison.csv    # Model summary statistics
│   ├── per_patient_summary.csv # Per-patient comparison
│   ├── comparison_report.html  # Interactive HTML report
│   └── figures/                # Visualization plots
├── previews/
│   ├── <model_tag>/*_ED_preview.png
│   └── <model_tag>/*_ES_preview.png
└── pipeline_info/
    ├── execution_timeline.html
    ├── execution_report.html
    └── pipeline_dag.svg
└── debug/
    ├── text/
    │   ├── debug_report.md
    │   ├── debug_metrics_summary.json
    │   ├── task_profile.csv
    │   ├── gpu_profile.csv
    │   └── metrics_snapshot.csv
    └── figures/
        ├── runtime_by_process.png
        ├── gpu_walltime_by_process.png
        └── overall_dice_by_model.png
```

## Execution Profiles

| Profile | Description |
|---------|-------------|
| `local` | Local execution with Apptainer/Singularity |
| `slurm` | SLURM execution with Apptainer |
| `test` | Quick test with minimal data |

### Profile Selection

```bash
# Local run
./bin/sorat_run.sh -profile local --input samplesheet.csv

# SLURM run
./bin/sorat_run.sh -profile slurm --input samplesheet.csv
```

Use one execution profile at a time.

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Increase memory allocation
   nextflow run main.nf --max_memory 64.GB --input samplesheet.csv
   ```

2. **GPU Not Detected**
   ```bash
   # Ensure NVIDIA runtime is configured
   docker run --gpus all nvidia/cuda:11.8-base nvidia-smi
   ```

3. **Path Issues**
   - Use absolute paths in samplesheet
   - Ensure all input files are accessible

### Getting Help

- Check execution logs: `results/pipeline_info/`
- View work directory: `.nextflow/` and `work/`
- Enable verbose logging: `nextflow run main.nf -with-trace`

## Citation

If you use SORAT in your research, please cite:

```bibtex
@software{sorat2024,
    title = {SORAT: Segmentation Orchestration and Reproducible Analysis Toolkit},
    year = {2024},
    url = {https://github.com/your-org/SORAT}
}
```

Also cite the individual models you use:
- **nnFormer**: https://arxiv.org/abs/2109.03201
- **MONAI VSA-3L (LV quantification reference)**: https://doi.org/10.1007/978-3-030-12029-0_40
- **CineMA**: https://arxiv.org/abs/2506.00679

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [nf-core](https://nf-co.re/) for Nextflow best practices
- [MONAI](https://monai.io/) for medical imaging tools
- ACDC Challenge organizers for the benchmark dataset
