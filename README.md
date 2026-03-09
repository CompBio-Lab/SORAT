# CASC - Cardiac Automated Segmentation Comparison Pipeline

[![Nextflow](https://img.shields.io/badge/nextflow-%E2%89%A523.04.0-brightgreen.svg)](https://www.nextflow.io/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![Singularity](https://img.shields.io/badge/singularity-enabled-blue.svg)](https://sylabs.io/)

## Overview

CASC (Cardiac Automated Segmentation Comparison) is a modular Nextflow pipeline for cardiac MRI segmentation that integrates multiple deep learning models. It enables easy comparison of segmentation results across different models and provides comprehensive evaluation metrics.

### Supported Models

| Model | Description | Reference |
|-------|-------------|-----------|
| **CineMA** | Convolutional Vision Transformer for cardiac MRI | [CineMA Paper](https://arxiv.org/abs/2506.00679) |
| **nnFormer** | 3D medical image segmentation transformer | [Zhou et al., 2021](https://arxiv.org/abs/2109.03201) |
| **VSA-3L** | MONAI Ventricular Short Axis 3-Label model | [MONAI Model Zoo](https://monai.io/model-zoo.html) |

### Output Labels

All models produce segmentations with ACDC-compatible labels:
- **0**: Background
- **1**: Right Ventricle (RV)
- **2**: Myocardium (MYO)
- **3**: Left Ventricle (LV)

## Quick Start

### 1. Install Nextflow

```bash
curl -s https://get.nextflow.io | bash
```

### 2. Run the Pipeline

**Using ACDC Dataset (Default):**
```bash
# The pipeline defaults to the ACDC testing dataset
nextflow run main.nf -profile docker

# With Singularity on HPC
nextflow run main.nf -profile singularity,slurm
```

**Using Custom Data:**
```bash
# Run with Docker (recommended)
nextflow run main.nf \
    --input samplesheet.csv \
    --outdir results \
    --models all \
    -profile docker

# Run with Singularity (for HPC)
nextflow run main.nf \
    --input samplesheet.csv \
    --outdir results \
    --models all \
    -profile singularity

# Run specific models only
nextflow run main.nf \
    --input samplesheet.csv \
    --outdir results \
    --models cinema,nnformer \
    -profile docker
```

### Pulling Containers (Apptainer)

```bash
# Pull CASC images from GitHub Container Registry
apptainer pull casc-cinema.sif docker://parsaban/casc-cinema:latest
apptainer pull casc-nnformer.sif docker://parsaban/casc-nnformer:latest
apptainer pull casc-vsa3l.sif docker://parsaban/casc-vsa3l:latest
```

### 3. Alternative: Jupyter Notebook Interface

For interactive use, open the Jupyter notebook:

```bash
jupyter notebook notebooks/run_casc_pipeline.ipynb
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
| `info_cfg` | No | Path to ACDC Info.cfg file (ED/ES frame indices) |

#### Example samplesheet.csv

```csv
patient_id,image,ground_truth,info_cfg
patient101,/data/patient101/patient101_4d.nii.gz,/data/patient101,/data/patient101/Info.cfg
patient102,/data/patient102/patient102_4d.nii.gz,/data/patient102,/data/patient102/Info.cfg
patient103,/data/patient103/patient103_4d.nii.gz,,
```

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
| `--input` | Path to input samplesheet CSV |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--outdir` | `./results` | Output directory |
| `--models` | `all` | Models to run: `cinema`, `nnformer`, `vsa3l`, or `all` |
| `--compare` | `true` | Generate comparison report |
| `--inference_only` | `false` | Run inference only (skip metrics + report generation) |
| `--debug` | `false` | Generate debug analytics report (execution/runtime/GPU/scalability/success + scientific utility metrics) |

### Debug Analytics Mode

Enable `--debug` to generate metrics and figures that quantify:
- execution time and process bottlenecks,
- estimated scalable GPU consumption and GPU concurrency,
- pipeline reliability/success rates and retry behavior,
- coverage and segmentation-quality utility metrics.

```bash
nextflow run main.nf \
    --input samplesheet.csv \
    --outdir results \
    --models all \
    --debug \
    -profile slurm
```

### Inference-Only Runs

Use `--inference_only` to run segmentation without computing metrics or generating comparison reports. This keeps the current behavior intact while providing a fast inference-only mode.

```bash
# Inference-only (no metrics/report)
nextflow run main.nf \
    --input samplesheet.csv \
    --outdir results \
    --models all \
    --inference_only \
    -profile slurm
```

Notes:
- If `--inference_only` is enabled, `--compare` is ignored.
- Metrics and reports require ground-truth data in the samplesheet; inference-only does not.

### Model-Specific Parameters

#### CineMA
```bash
--cinema.trained_dataset acdc    # Training dataset (acdc, mnms, mnms2)
--cinema.seeds 0,1,2              # Seeds for ensemble
--cinema.ensemble true            # Enable ensemble prediction
```

#### nnFormer
```bash
--nnformer.fold 0                 # Model fold to use
--nnformer.tta true               # Test-time augmentation
--nnformer.mixed_precision true   # Mixed precision inference
```

#### VSA-3L
```bash
--vsa3l.input_size 256,256        # Model input size
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
├── metrics/
│   ├── cinema/                 # Per-model metrics
│   ├── nnformer/
│   └── vsa3l/
├── comparison/
│   ├── aggregated_metrics.csv  # All metrics combined
│   ├── model_comparison.csv    # Model summary statistics
│   ├── per_patient_summary.csv # Per-patient comparison
│   ├── comparison_report.html  # Interactive HTML report
│   └── figures/                # Visualization plots
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
| `standard` | Local execution with Docker |
| `docker` | Docker containers |
| `singularity` | Singularity/Apptainer containers |
| `slurm` | SLURM cluster with Singularity |
| `pbs` | PBS/Torque cluster |
| `gpu` | Enable GPU acceleration |
| `test` | Quick test with minimal data |

### Combining Profiles

```bash
# SLURM cluster with GPU support
nextflow run main.nf -profile slurm,gpu --input samplesheet.csv

# Local with Docker and GPU
nextflow run main.nf -profile docker,gpu --input samplesheet.csv
```

## Adding New Models

CASC is designed to be extensible. To add a new segmentation model:

### 1. Create a Module

Create `modules/mymodel.nf`:

```groovy
process MYMODEL_PREPROCESS {
    tag "$patient_id"
    label 'process_medium'
    
    input:
    tuple val(patient_id), path(image), path(ground_truth), path(info_cfg)
    
    output:
    tuple val(patient_id), path("${patient_id}_preprocessed"), path(ground_truth), path(info_cfg), emit: preprocessed
    
    script:
    """
    mymodel_preprocess.py --input ${image} --patient_id ${patient_id} --output_dir ${patient_id}_preprocessed
    """
}

process MYMODEL_SEGMENT {
    tag "$patient_id"
    label 'process_gpu'
    
    input:
    tuple val(patient_id), path(preprocessed_dir), path(ground_truth), path(info_cfg)
    
    output:
    tuple val(patient_id), path("${patient_id}_ED_mymodel.nii.gz"), path("${patient_id}_ES_mymodel.nii.gz"), val(meta), emit: segmentation
    
    script:
    meta = [model: 'mymodel']
    """
    mymodel_segment.py --input_dir ${preprocessed_dir} --patient_id ${patient_id} --output_prefix ${patient_id}
    """
}
```

### 2. Create Python Scripts

Add preprocessing and segmentation scripts to `bin/`:
- `bin/mymodel_preprocess.py`
- `bin/mymodel_segment.py`

### 3. Create a Dockerfile

Create `containers/mymodel/Dockerfile`:

```dockerfile
FROM ghcr.io/your-org/casc-base:latest
# Add model-specific dependencies
RUN pip install mymodel-dependencies
COPY mymodel/ /app/mymodel/
```

### 4. Register the Model

Add entry to `conf/model_registry.json`:

```json
{
    "mymodel": {
        "name": "My Model",
        "version": "1.0.0",
        "description": "Description of my model",
        "container": "ghcr.io/your-org/casc-mymodel:latest",
        "module_path": "modules/mymodel.nf",
        "enabled": true
    }
}
```

### 5. Update main.nf

Add the new model to the workflow in `main.nf`.

## Building Containers

The CASC Docker images come **pre-loaded with model weights**, so users can simply pull and run without any additional setup.

### Pulling Pre-built Images

```bash
# Pull all CASC images from GitHub Container Registry
docker pull ghcr.io/pmoheban/casc-cinema:latest
docker pull ghcr.io/pmoheban/casc-nnformer:latest
docker pull ghcr.io/pmoheban/casc-vsa3l:latest
```

### Building Images Locally

If you need to build the images yourself:

```bash
# Use the build script
chmod +x scripts/build_containers.sh
./scripts/build_containers.sh all

# Or build individually
docker build -f containers/cinema/Dockerfile -t ghcr.io/pmoheban/casc-cinema:latest .
docker build -f containers/nnformer/Dockerfile -t ghcr.io/pmoheban/casc-nnformer:latest .
docker build -f containers/vsa3l/Dockerfile -t ghcr.io/pmoheban/casc-vsa3l:latest .
```

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

If you use CASC in your research, please cite:

```bibtex
@software{casc2024,
    title = {CASC: Cardiac Automated Segmentation Comparison Pipeline},
    year = {2024},
    url = {https://github.com/your-org/CASC}
}
```

Also cite the individual models you use:

- **nnFormer**: @article{zhou2022nnformerinterleavedtransformervolumetric,
    title={nnFormer: Interleaved Transformer for Volumetric Segmentation}, 
    author={Hong-Yu Zhou and Jiansen Guo and Yinghao Zhang and Lequan Yu and Liansheng Wang and Yizhou Yu},
    year={2022},
    url={https://arxiv.org/abs/2109.03201}
}
- **MONAI VSA-3L**: @article{inbook,
    author = {Kerfoot, Eric and Clough, James and Oksuz, Ilkay and Lee, Jack and King, Andrew and Schnabel, Julia},
    year = {2019},
    month = {02},
    pages = {371-380},
    title = {Left-Ventricle Quantification Using Residual U-Net: 9th International Workshop, STACOM 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Revised Selected Papers},
    isbn = {978-3-030-12028-3},
    doi = {10.1007/978-3-030-12029-0_40}
}
- **CineMA**: @article{fu2025cinema,
    title={A versatile foundation model for cine cardiac magnetic resonance image analysis tasks},
    author={Fu, Yunguan and Bai, Wenjia and Yi, Weixi and Manisty, Charlotte and Bhuva, Anish N and Treibel, Thomas A and Moon, James C and Clarkson, Matthew J and Davies, Rhodri Huw and Hu, Yipeng},
    journal={arXiv preprint arXiv:2506.00679},
    year={2025}
}

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [nf-core](https://nf-co.re/) for Nextflow best practices
- [MONAI](https://monai.io/) for medical imaging tools
- ACDC Challenge organizers for the benchmark dataset
