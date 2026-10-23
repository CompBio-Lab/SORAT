# Installation

## Requirements

| | |
|---|---|
| Nextflow | 23.04 or later (Java 11+) |
| Container runtime | Apptainer or Singularity for running; Docker only for building images |
| GPU | NVIDIA GPU with a driver that supports CUDA 11.8 (CUDA 12.1 for the atrial model) |
| Python | 3.8+ on the host, only for `bin/sorat_setup.py` |

```bash
curl -s https://get.nextflow.io | bash      # installs ./nextflow
git clone https://github.com/CompBio-Lab/SORAT.git && cd SORAT
python3 bin/sorat_setup.py
```

`sorat_setup.py` asks for your data roots, SLURM account and cache locations and writes them to `.sorat/user.config`. That file is gitignored and loaded automatically. Rerun with `--force` to change it, or with `--non-interactive` and flags for scripted setups (`--help` lists them).

## Containers

Prebuilt images, with the model weights included, are on Docker Hub. Pull them as Apptainer images into `containers/`:

```bash
apptainer pull containers/sorat-cinema.sif        docker://parsaban/sorat-cinema:v100-sm70
apptainer pull containers/sorat-nnformer.sif      docker://parsaban/sorat-nnformer:v100-sm70
apptainer pull containers/sorat-vsa3l.sif         docker://parsaban/sorat-vsa3l:v100-sm70
apptainer pull containers/sorat-atrial-nnunet.sif docker://parsaban/sorat-atrial-nnunet:latest
```

The pipeline expects these four file names. Use `--vsa3l_container` (or `SORAT_VSA3L_CONTAINER`) to point VSA-3L at another image, for example a new build you want to test.

To build the images yourself instead, see the next two sections.

## Model weights

The published images above already contain these weights, so you only need them to build an image yourself. They are not stored in this repository; download them from their authors and place them under `models/`:

| Model | Source | Location in `models/` |
|---|---|---|
| CineMA | [mathpluscode/CineMA on Hugging Face](https://huggingface.co/mathpluscode/CineMA), `finetuned/segmentation/` | `cinema/finetuned/segmentation/{acdc,mnms,mnms2}_sax/` (three `*.safetensors` seeds and `config.yaml` each) |
| nnFormer | ACDC model from the [nnFormer repository](https://github.com/282857341/nnFormer) | `nnformer/nnFormer_trained_models/nnFormer/3d_fullres/Task001_ACDC/nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1/` (`plans.pkl`, `fold_0/model_best.model.pkl`) |
| VSA-3L | MONAI bundle `ventricular_short_axis_3label` (`python -m monai.bundle download ventricular_short_axis_3label`) | `vsa3l/model.pt` (the bundle configs are in `configs/`) |
| Atrial nnU-Net | Trained on MBAS 2024 by the authors; copy it out of the published image (see below) | `atrial_nnunet/nnUNet_results/Dataset001_LGE/` |

## Building the containers

Each model has a Dockerfile in `containers/<model>/`. The build context is the repository root, and the weights are copied into the image.

```bash
scripts/build_containers.sh all              # or: cinema | nnformer | vsa3l | atrial_nnunet
```

Images are named `<registry>/<namespace>/sorat-<model>:<version>`, by default `docker.io/parsaban/sorat-<model>:latest`; set `SORAT_CONTAINER_REGISTRY`, `SORAT_CONTAINER_NAMESPACE` and `SORAT_CONTAINER_VERSION` to change them, and add `--push` to push. The atrial weights can be copied out of the published image:

```bash
mkdir -p models/atrial_nnunet
docker create --name atrial parsaban/sorat-atrial-nnunet:latest
docker cp atrial:/models/atrial_nnunet/nnUNet_results models/atrial_nnunet/ && docker rm atrial
```

Convert the result to an Apptainer image in `containers/`:

```bash
apptainer build containers/sorat-cinema.sif docker-daemon://<registry>/<namespace>/sorat-cinema:<version>
# or, from a pushed image:
apptainer pull containers/sorat-cinema.sif docker://<registry>/<namespace>/sorat-cinema:<version>
```

### Code changes without rebuilding

The `local` and `slurm` profiles bind `bin/` to `/app/bin` and `nnformer/` to `/app/nnformer` in every container. Changes to these folders take effect on the next run. Dependencies and weights need a rebuild.

## Checking the GPU runtime

PyTorch wheels only include kernels for some GPU generations. Before a full run on a new cluster, check that an image runs on the GPUs you will be allocated (run it on a GPU node, not a login node):

```bash
apptainer exec --nv containers/sorat-vsa3l.sif python /app/bin/validate_cuda_runtime.py --json
```

The report lists the device, its compute capability and the architectures compiled into PyTorch. The short-axis images use CUDA 11.8 and the atrial image CUDA 12.1; all include `sm_70` kernels, so they run on V100 and newer cards. SORAT runs the same check during VSA-3L model discovery and stops before submitting segmentation jobs if the image and GPU don't match.

## Clusters

The `slurm` profile was written for UBC ARC Sockeye. Each job loads Apptainer through `module load`, and GPU jobs are charged to `<account>-gpu`. On another cluster, copy the `slurm` profile from `nextflow.config` into your own config file, change `beforeScript` and `clusterOptions`, and pass the file with `-c`. Set the bind paths with `sorat_setup.py --singularity-slurm-binds`.
