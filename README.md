<div align="center">

# SORAT

**Segmentation Orchestration and Reproducible Analysis Toolkit**

Run several cardiac MRI segmentation models on the same data, map every prediction back to the original image, and evaluate them all with one tested implementation.

[![Nextflow](https://img.shields.io/badge/Nextflow-%E2%89%A523.04-23aa62?logo=nextflow&logoColor=white)](https://www.nextflow.io/)
[![Apptainer](https://img.shields.io/badge/containers-Apptainer-1d355e)](https://apptainer.org/)
[![Python](https://img.shields.io/badge/Python-3.10-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/sorat-workflow-dark.svg">
  <img alt="SORAT workflow: a samplesheet passes through one frame, label and geometry convention, is segmented in parallel by model containers, and is scored in the original image space." src="docs/assets/sorat-workflow-light.svg" width="100%">
</picture>

</div>

## Why SORAT

Published segmentation models usually ship with their own preprocessing, label values and evaluation scripts. Differences between them can change reported accuracy without raising an error. SORAT puts each model behind one data contract:

- **Frames.** ED and ES are read from `Info.cfg` (counted from 1, as in ACDC) and checked against the phase images a dataset provides.
- **Labels.** Each dataset's reference label values are declared, and every model's output is mapped to one label set.
- **Geometry.** Every prediction is resampled to the spacing, origin and orientation of the original image before it is scored.
- **Evaluation.** DSC, HD95, failure checks, clinical measurements and reports come from one implementation, with unit tests for each convention.

Each model runs in its own container, and Nextflow schedules the work on a workstation or a SLURM cluster.

## Models

| Model | Input | Labels | Reference |
|---|---|---|---|
| **CineMA** (3 fine-tuned datasets × 3 seeds, plus majority-vote ensembles) | Short-axis cine | 1 RV, 2 MYO, 3 LV | [Fu et al., 2026](https://doi.org/10.1038/s43856-026-01636-0) |
| **nnFormer** (ACDC model) | Short-axis cine | 1 RV, 2 MYO, 3 LV | [Zhou et al., 2023](https://doi.org/10.1109/TIP.2023.3293771) |
| **VSA-3L** (MONAI model zoo) | Short-axis cine | 1 RV, 2 MYO, 3 LV | [Kerfoot et al., 2019](https://doi.org/10.1007/978-3-030-12029-0_40) |
| **Atrial nnU-Net** (MBAS 2024) | 3-D LGE | 1 wall, 2 RA, 3 LA | [Isensee et al., 2021](https://doi.org/10.1038/s41592-020-01008-z) |

New models can be added from a template; see [Adding a model](docs/adding-a-model.md).

## Quick start

```bash
git clone https://github.com/CompBio-Lab/SORAT.git && cd SORAT
python3 bin/sorat_setup.py          # writes local paths and cluster settings to .sorat/user.config (gitignored)
```

Pull the model containers (prebuilt images are on Docker Hub; see [Installation](docs/installation.md)), then list your subjects in a samplesheet:

```csv
patient_id,image,ground_truth,info_cfg
patient101,/data/patient101/patient101_4d.nii.gz,/data/patient101,/data/patient101/Info.cfg
patient102,/data/patient102/patient102_4d.nii.gz,,
```

Run on a workstation or on SLURM:

```bash
nextflow run main.nf -profile local --input samplesheet.csv --models cinema,nnformer --outdir results
nextflow run main.nf -profile slurm --slurm_account <account> --input samplesheet.csv --models all \
    --evaluation.gt_label_map 'rv=1,myo=2,lv=3'
```

Ground truth and `info_cfg` are optional. Without ground truth SORAT segments only. Without `info_cfg` it segments every frame. Images can be NIfTI, NRRD, MetaImage or a DICOM series. See [Usage](docs/usage.md).

## Outputs

```text
results/
├── <model>/segmentations/     predictions in the original image space
├── metrics/                   per-frame DSC and HD95
├── comparison/                aggregated metrics and an HTML comparison report
├── previews/                  ED/ES overlays with per-structure DSC
├── features/                  volumes, EF, mass, wall thickness, radiomics (optional)
└── pipeline_info/             Nextflow execution report, timeline and trace
```

Details are in [Outputs](docs/outputs.md).

## Documentation

| | |
|---|---|
| [Installation](docs/installation.md) | Requirements, model weights, building and validating containers |
| [Usage](docs/usage.md) | Samplesheets, frames, label maps, evaluation schemas, profiles, isolated entry points |
| [Parameters](docs/parameters.md) | Every pipeline option |
| [Features](docs/features.md) | Clinical measurements and radiomics |
| [Outputs](docs/outputs.md) | Output folders and reports |
| [Adding a model](docs/adding-a-model.md) | Template generator and the wiring a new model needs |

## Development

Tests are plain `unittest` files. They need NumPy, SimpleITK and MedPy, so run them inside a model container:

```bash
apptainer exec containers/sorat-cinema.sif python -m unittest discover -s tests -v
```

## Citation

A manuscript describing SORAT is in preparation. Until then, please cite this repository and the models you use (see the table above).

## License

SORAT is released under the [MIT License](LICENSE). The vendored CineMA (`cinema/`) and nnFormer (`nnformer/`) code keep their own MIT licences. Model weights and datasets are distributed by their authors under their own terms.
