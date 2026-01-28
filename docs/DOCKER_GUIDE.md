# 🐳 Container Guide for CASC Pipeline (UBC ARC Sockeye)

This guide will walk you through everything you need to know about using containers with the CASC pipeline on **UBC ARC Sockeye**, written for beginners.

> **Note for Sockeye Users**: Docker is NOT available on Sockeye for security reasons. Instead, we use **Apptainer** (formerly Singularity), which is HPC-compatible and doesn't require root access. This guide focuses on Apptainer workflows.

---

## Table of Contents
1. [What are Containers?](#1-what-are-containers)
2. [Apptainer on Sockeye (Quick Start)](#2-apptainer-on-sockeye-quick-start)
3. [Understanding Container Concepts](#3-understanding-container-concepts)
4. [Using Pre-built CASC Containers](#4-using-pre-built-casc-containers)
5. [Building Containers (Advanced)](#5-building-containers-advanced)
6. [Publishing Images to GitHub Container Registry](#6-publishing-images-to-github-container-registry)
7. [Running GPU Jobs on Sockeye](#7-running-gpu-jobs-on-sockeye)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. What are Containers?

Containers are packages that bundle applications with all their dependencies into isolated, reproducible environments. Think of a container as a lightweight virtual machine that includes:
- The application code
- All required libraries and dependencies
- A specific version of Python, CUDA, etc.
- Pre-trained model weights

**Why use containers for CASC on Sockeye?**
- **Reproducibility**: Same environment everywhere
- **No dependency conflicts**: Each model has its own isolated environment
- **Easy sharing**: Pull images from a registry instead of manual setup
- **No root required**: Apptainer runs without administrator privileges

### Docker vs Apptainer

| Feature | Docker | Apptainer (Singularity) |
|---------|--------|-------------------------|
| Root access required | Yes | **No** ✓ |
| HPC compatible | No | **Yes** ✓ |
| Available on Sockeye | **No** | **Yes** ✓ |
| Image format | Layered cache | Single `.sif` file |
| File system | Isolated | Auto-mounts $HOME, /scratch |

**On Sockeye, you must use Apptainer** - Docker is not available for security reasons.

---

## 2. Apptainer on Sockeye (Quick Start)

### Loading Apptainer

Every time you log in to Sockeye, you need to load the required modules:

```bash
# Load gcc and apptainer modules (REQUIRED before any apptainer command)
module load gcc apptainer

# Verify it's loaded
apptainer --version
```

### Pulling CASC Images

Navigate to your project space and pull the CASC containers:

```bash
# Navigate to your project space (recommended for storing containers)
cd /arc/project/st-zlaksman-1/pmoheban
mkdir -p containers && cd containers

# Load modules
module load gcc apptainer

# Pull CASC images from GitHub Container Registry
apptainer pull casc-cinema.sif docker://parsaban/casc-cinema:latest
apptainer pull casc-nnformer.sif docker://parsaban/casc-nnformer:latest
apptainer pull casc-vsa3l.sif docker://parsaban/casc-vsa3l:latest

# List your downloaded containers
ls -lh *.sif
```

### Running an Interactive Shell

```bash
# Start an interactive shell inside the container
apptainer shell casc-cinema.sif

# You'll see your prompt change to:
# Apptainer>

# Run commands inside the container
python --version
pip list

# Exit the container
exit
```

### Running a Command Directly

```bash
# Execute a single command inside the container
apptainer exec casc-cinema.sif python --version

# Run a Python script
apptainer exec casc-cinema.sif python /app/bin/cinema_segment.py --help
```

---

## 3. Understanding Container Concepts

### Key Terms

| Term | Definition |
|------|------------|
| **Image** | A read-only template with instructions for creating a container (like a class) |
| **Container** | A runnable instance of an image (like an object) |
| **SIF file** | Singularity Image Format - Apptainer's single-file container format |
| **Registry** | A service that stores container images (e.g., GitHub Container Registry, Docker Hub) |
| **Tag** | A label for different versions of an image (e.g., `latest`, `v1.0`) |
| **Bind mount** | A way to make host directories accessible inside the container |

### Image Naming Convention

```
registry/namespace/image-name:tag

Example:
ghcr.io/pmoheban/casc-cinema:latest
└─────┘ └───────┘ └─────────┘ └────┘
registry  org/user  image name  version
```

### Sockeye Directory Structure

On Sockeye, you have access to several storage locations:

| Path | Purpose | Notes |
|------|---------|-------|
| `/home/<cwl>` | Personal scripts, configs | 50GB quota, read-only on compute nodes |
| `/arc/project/<alloc-code>` | Shared project data | 1TB default, read-only on compute nodes |
| `/scratch/<alloc-code>` | Computational work | 5TB, read-write on all nodes |

**Recommended**: Store containers in `/arc/project/`, run jobs from `/scratch/`

---

## 4. Using Pre-built CASC Containers

### Pulling Images

```bash
# Navigate to project space for storing containers
cd /arc/project/st-zlaksman-1/pmoheban
mkdir -p CASC/containers && cd CASC/containers

# Load required modules
module load gcc apptainer

# Pull all CASC images
apptainer pull --name casc-cinema.sif docker://ghcr.io/pmoheban/casc-cinema:latest
apptainer pull --name casc-nnformer.sif docker://ghcr.io/pmoheban/casc-nnformer:latest
apptainer pull --name casc-vsa3l.sif docker://ghcr.io/pmoheban/casc-vsa3l:latest

# List downloaded images
ls -lh *.sif
```

### Running a Container Interactively

```bash
# Load modules (do this every session)
module load gcc apptainer

# Basic interactive session
apptainer shell casc-cinema.sif

# With GPU support (must be on a GPU node - see Section 7)
apptainer shell --nv casc-cinema.sif

# Bind additional directories if needed
apptainer shell --bind /scratch/st-zlaksman-1:/scratch casc-cinema.sif
```

**Understanding bind mounts:**
- Apptainer automatically mounts your `$HOME`, `/scratch`, and current directory
- Use `--bind /host/path:/container/path` to mount additional directories
- This is how you get data in and results out

### Running Segmentation

```bash
# Load modules
module load gcc apptainer

# Set paths
CONTAINER_DIR="/arc/project/st-zlaksman-1/pmoheban/CASC/containers"
DATA_DIR="/scratch/st-zlaksman-1/pmoheban/ACDC/database"
OUTPUT_DIR="/scratch/st-zlaksman-1/pmoheban/CASC/results"

# Run CineMA segmentation on a patient
apptainer exec --nv \
    --bind ${DATA_DIR}:/data \
    --bind ${OUTPUT_DIR}:/output \
    ${CONTAINER_DIR}/casc-cinema.sif \
    python /app/bin/cinema_segment.py \
        --input_dir /data/testing/patient101 \
        --patient_id patient101 \
        --output_prefix /output/patient101_cinema \
        --model_dir /models/cinema

# Run nnFormer
apptainer exec --nv \
    --bind ${DATA_DIR}:/data \
    --bind ${OUTPUT_DIR}:/output \
    ${CONTAINER_DIR}/casc-nnformer.sif \
    python /app/bin/nnformer_segment.py \
        --input_dir /data/testing/patient101 \
        --patient_id patient101 \
        --output_prefix /output/patient101_nnformer \
        --model_dir /models/nnformer

# Run VSA-3L
apptainer exec --nv \
    --bind ${DATA_DIR}:/data \
    --bind ${OUTPUT_DIR}:/output \
    ${CONTAINER_DIR}/casc-vsa3l.sif \
    python /app/bin/vsa3l_segment.py \
        --input_dir /data/testing/patient101 \
        --patient_id patient101 \
        --output_prefix /output/patient101_vsa3l \
        --model_dir /models/vsa3l
```

### Useful Apptainer Commands

```bash
# Load modules first!
module load gcc apptainer

# Get help
apptainer help

# List available commands
apptainer help exec
apptainer help shell

# Inspect a container
apptainer inspect casc-cinema.sif

# Run a container (default command)
apptainer run casc-cinema.sif

# Execute a specific command
apptainer exec casc-cinema.sif python --version

# Interactive shell
apptainer shell casc-cinema.sif

# Check what's inside a container
apptainer exec casc-cinema.sif ls /app
apptainer exec casc-cinema.sif pip list
```

---

## 5. Building Containers (Advanced)

If you want to build the images yourself (e.g., to customize or update model weights):

### Building with Apptainer on Sockeye

Apptainer allows building containers directly on Sockeye without root access:

```bash
# Navigate to CASC directory
cd /scratch/st-zlaksman-1/pmoheban/CASC

# Load modules
module load gcc apptainer

# Build from a Docker image (creates a sandbox for modification)
apptainer build --sandbox casc-cinema.sandbox docker://pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Shell into the sandbox in write mode to install packages
apptainer shell --writable casc-cinema.sandbox

# Inside the container, install dependencies
pip install nibabel SimpleITK safetensors einops monai

# Exit and convert to SIF file
exit
apptainer build casc-cinema.sif casc-cinema.sandbox
```

### Building from a Definition File

Create a definition file (e.g., `cinema.def`):

```bash
cat > cinema.def << 'EOF'
Bootstrap: docker
From: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

%post
    apt-get update && apt-get install -y git
    pip install --no-cache-dir \
        nibabel \
        SimpleITK \
        safetensors \
        einops \
        monai \
        pandas \
        matplotlib

%environment
    export LC_ALL=C

%runscript
    exec python "$@"

%labels
    Author pmoheban
    Version 1.0
    Description CineMA cardiac segmentation model
EOF
```

Build the container:

```bash
module load gcc apptainer
apptainer build casc-cinema.sif cinema.def
```

### Directory Structure for Building

Before building, ensure you have this structure:

```
CASC/
├── containers/
│   ├── cinema/
│   │   └── Dockerfile      # For Docker builds (off-cluster)
│   ├── nnformer/
│   │   └── Dockerfile
│   └── vsa3l/
│       └── Dockerfile
├── bin/                    # Python scripts
├── models/                 # Model weights (you need to add these!)
│   ├── cinema/
│   │   └── model.safetensors
│   ├── nnformer/
│   │   └── model.pkl
│   └── vsa3l/
│       └── model.pt
└── external/               # External code repositories
    ├── cinema/             # CineMA source code
    ├── nnformer/           # nnFormer source code
    └── configs/            # MONAI bundle configs
```

### Building on a Local Machine (with Docker)

If you need to build with Docker (e.g., on your laptop), then transfer to Sockeye:

```bash
# On your local machine with Docker installed
cd /path/to/CASC

# Build Docker images
docker build -f containers/cinema/Dockerfile -t ghcr.io/pmoheban/casc-cinema:latest .

# Save as tar file for transfer
docker save ghcr.io/pmoheban/casc-cinema:latest | gzip > casc-cinema.tar.gz

# Transfer to Sockeye
scp casc-cinema.tar.gz <cwl>@dtn.sockeye.arc.ubc.ca:/scratch/st-zlaksman-1/pmoheban/

# On Sockeye, convert to SIF
module load gcc apptainer
apptainer build casc-cinema.sif docker-archive://casc-cinema.tar.gz
```

---

## 6. Publishing Images to GitHub Container Registry

To share your containers with others, you can push them to GitHub Container Registry (GHCR).

### Step 1: Create a GitHub Personal Access Token (PAT)

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name like "CASC Docker Push"
4. Select scopes:
   - `write:packages` (to push images)
   - `read:packages` (to pull images)
   - `delete:packages` (optional, to delete old versions)
5. Click "Generate token" and **save the token** (you won't see it again!)

### Step 2: Build and Push with Docker (on local machine)

You'll need Docker on a local machine to push to GHCR:

```bash
# Set your GitHub username and token
export GITHUB_USER="pmoheban"
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"  # Your PAT

# Log in to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USER --password-stdin

# Build images
docker build -f containers/cinema/Dockerfile -t ghcr.io/pmoheban/casc-cinema:latest .

# Push to GHCR
docker push ghcr.io/pmoheban/casc-cinema:latest
```

### Step 3: Make Images Public (Optional)

By default, packages on GHCR are private. To make them public:

1. Go to `github.com/pmoheban?tab=packages`
2. Click on each package (e.g., `casc-cinema`)
3. Click "Package settings" (gear icon)
4. Scroll to "Danger Zone" → Change package visibility → Make public

### Step 4: Pull on Sockeye

Once published, anyone can pull the images on Sockeye:

```bash
module load gcc apptainer
apptainer pull --name casc-cinema.sif docker://ghcr.io/pmoheban/casc-cinema:latest
```

---

## 7. Running GPU Jobs on Sockeye

**Important**: You cannot run GPU code on the login nodes. You must submit a job or get an interactive GPU session.

### Interactive GPU Session

Create a script `gpu_interactive.sh`:

```bash
#!/bin/bash
salloc --time=1:00:00 --mem=16G --nodes=1 --gpus=1 --account=st-zlaksman-1-gpu
```

Run it to get a GPU node:

```bash
chmod +x gpu_interactive.sh
./gpu_interactive.sh
```

Once on a GPU node:

```bash
# Verify GPU is available
nvidia-smi

# Load modules
module load gcc cuda apptainer

# Run container with GPU support (--nv flag is REQUIRED for GPU access)
apptainer exec --nv /arc/project/st-zlaksman-1/pmoheban/CASC/containers/casc-cinema.sif \
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Batch GPU Job Script

Create a job script `run_casc_gpu.sh`:

```bash
#!/bin/bash
#SBATCH --time=04:00:00              # Request 4 hours
#SBATCH --account=st-zlaksman-1-gpu  # Your allocation with -gpu suffix!
#SBATCH --nodes=1                    # Request 1 node
#SBATCH --gpus-per-node=1            # Request 1 GPU per node
#SBATCH --mem=32G                    # Request 32 GB of memory
#SBATCH --job-name=casc_seg          # Job name
#SBATCH --output=casc_%j.out         # Output file (%j = job ID)
#SBATCH --error=casc_%j.err          # Error file
#SBATCH --mail-user=your-email@ubc.ca
#SBATCH --mail-type=ALL

# Load required modules
module load gcc cuda apptainer

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Set paths
CONTAINER_DIR="/arc/project/st-zlaksman-1/pmoheban/CASC/containers"
DATA_DIR="/scratch/st-zlaksman-1/pmoheban/ACDC/database"
OUTPUT_DIR="/scratch/st-zlaksman-1/pmoheban/CASC/results"

# Run CineMA segmentation
apptainer exec --nv \
    --bind ${DATA_DIR}:/data \
    --bind ${OUTPUT_DIR}:/output \
    ${CONTAINER_DIR}/casc-cinema.sif \
    python /app/bin/cinema_segment.py \
        --input_dir /data/testing/patient101 \
        --patient_id patient101 \
        --output_prefix /output/patient101_cinema \
        --model_dir /models/cinema

echo "Job completed!"
```

Submit the job:

```bash
cd /scratch/st-zlaksman-1/pmoheban/CASC
sbatch run_casc_gpu.sh

# Check job status
squeue -u $USER

# View output
cat casc_*.out
```

### Running the CASC Nextflow Pipeline on Sockeye

The CASC pipeline has built-in Sockeye support:

```bash
# Navigate to CASC directory
cd /scratch/st-zlaksman-1/pmoheban/CASC

# Load Nextflow (if available, or install locally)
# curl -s https://get.nextflow.io | bash

# Run with Singularity/Apptainer and SLURM
./nextflow run main.nf \
    --input data/acdc_testing_samplesheet.csv \
    --outdir results \
    -profile singularity,slurm

# Or specify the GPU account explicitly
./nextflow run main.nf \
    --input data/acdc_testing_samplesheet.csv \
    --outdir results \
    -profile singularity,slurm,gpu \
    --slurm_account st-zlaksman-1-gpu
```

---

## 8. Troubleshooting

### Common Issues on Sockeye

**Error: "apptainer: command not found"**
```bash
# You forgot to load the modules!
module load gcc apptainer
```

**Error: "CUDA not available" or GPU not detected**
```bash
# Make sure you're on a GPU node (not login node)
# Check if you used --nv flag
apptainer exec --nv container.sif python -c "import torch; print(torch.cuda.is_available())"

# Verify GPU allocation
nvidia-smi
```

**Error: "Read-only file system" when trying to write**
```bash
# /arc/project and /home are read-only on compute nodes
# Write to /scratch instead
cd /scratch/st-zlaksman-1/pmoheban
```

**Error: "No space left on device" during pull**
```bash
# Apptainer uses cache in $HOME by default
# Set cache to scratch space
export APPTAINER_CACHEDIR=/scratch/st-zlaksman-1/pmoheban/.apptainer_cache
mkdir -p $APPTAINER_CACHEDIR

# Then retry the pull
apptainer pull --name container.sif docker://...
```

**Error: "unauthorized" when pulling from GHCR**
```bash
# If the image is private, you need to authenticate
# Create a token at github.com with read:packages scope
apptainer remote login --username pmoheban docker://ghcr.io
```

**Job pending forever in queue**
```bash
# Check your allocation code - GPU jobs need -gpu suffix
#SBATCH --account=st-zlaksman-1-gpu  # NOT st-zlaksman-1

# Check available partitions
sinfo

# Check your fairshare
squeue -u $USER
```

### Checking Your Environment

```bash
# Check loaded modules
module list

# Check Apptainer version
apptainer --version

# Check available GPU (on GPU node only)
nvidia-smi

# Check storage quota
print_quota

# Check job status
squeue -u $USER

# Check detailed job info
scontrol show job <jobid>
```

### Getting Help

```bash
# Apptainer help
apptainer help
apptainer help exec
apptainer help shell

# SLURM help
man sbatch
man squeue

# UBC ARC Support
# Email: arc.support@ubc.ca
# Documentation: https://confluence.it.ubc.ca/display/UARC
```

---

## Quick Reference Card for Sockeye

```bash
# === SETUP (every session) ===
module load gcc apptainer

# === PULLING IMAGES ===
cd /arc/project/st-zlaksman-1/pmoheban/CASC/containers
apptainer pull --name casc-cinema.sif docker://ghcr.io/pmoheban/casc-cinema:latest

# === INTERACTIVE SESSION ===
apptainer shell casc-cinema.sif
apptainer shell --nv casc-cinema.sif  # With GPU (on GPU node)

# === RUNNING COMMANDS ===
apptainer exec casc-cinema.sif python --version
apptainer exec --nv casc-cinema.sif python script.py  # With GPU

# === GPU JOB ===
salloc --time=1:00:00 --mem=16G --gpus=1 --account=st-zlaksman-1-gpu

# === BATCH JOB ===
sbatch run_casc_gpu.sh
squeue -u $USER
scancel <jobid>

# === BUILDING ===
apptainer build --sandbox mycontainer.sandbox docker://pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
apptainer shell --writable mycontainer.sandbox
apptainer build mycontainer.sif mycontainer.sandbox

# === STORAGE ===
# Containers: /arc/project/st-zlaksman-1/pmoheban/
# Work/Output: /scratch/st-zlaksman-1/pmoheban/
# Check quota: print_quota
```

---

## Additional Resources

- [UBC ARC Sockeye QuickStart Guide](https://confluence.it.ubc.ca/display/UARC/QuickStart+Guide)
- [Using Apptainer on Sockeye](https://confluence.it.ubc.ca/display/UARC/Using+Apptainer+or+Singularity+Containers)
- [Running Jobs on Sockeye](https://confluence.it.ubc.ca/display/UARC/Running+Jobs)
- [GPU Jobs on Sockeye](https://confluence.it.ubc.ca/display/UARC/GPU+Jobs+in+Detail)
- [ARC Support](mailto:arc.support@ubc.ca)
