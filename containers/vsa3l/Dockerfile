# =============================================================================
# SORAT VSA-3L (MONAI) Docker Image
# =============================================================================
# MONAI Ventricular Short Axis 3-Label segmentation model with pre-trained weights
# 
# Build (from SORAT root directory):
#   docker build -t docker.io/<dockerhub-username>/sorat-vsa3l:v100-sm70 -f containers/vsa3l/Dockerfile .
#
# Push:
#   docker push docker.io/<dockerhub-username>/sorat-vsa3l:v100-sm70
#
# Run interactively:
#   docker run --gpus all -it -v /path/to/data:/data docker.io/<dockerhub-username>/sorat-vsa3l:v100-sm70
# =============================================================================

# Sockeye's GPU partition is Tesla V100 (compute capability 7.0). Newer
# PyTorch CUDA builds may omit SM70 kernels, producing
# "no kernel image is available for execution on the device" at inference.
# Keep this base explicit and V100-compatible; do not replace it with a
# mutable latest tag.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.source="https://github.com/CompBio-Lab/SORAT"
LABEL description="MONAI VSA-3L cardiac segmentation model for SORAT pipeline"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies without allowing a dependency resolver
# to replace the CUDA-enabled torch shipped by the base image.
RUN python -m pip install --upgrade "pip<25" && \
    python -m pip install --no-cache-dir \
        "numpy<2" "scipy<1.14" "pandas<2.3" "matplotlib<3.9" \
        "SimpleITK==2.3.1" "nibabel<6" "medpy==0.5.2" "tqdm<5" \
        "scikit-image<0.24" "Pillow<11" packaging && \
    python -m pip install --no-cache-dir --no-deps "monai==1.3.2"

# Create directories
RUN mkdir -p /app /data /output /app/bin /models/vsa3l/configs

WORKDIR /app

# Copy MONAI bundle configuration files
COPY configs/ /models/vsa3l/configs/

# Copy model weights
COPY models/vsa3l/ /models/vsa3l/

# Copy pipeline scripts
COPY bin/vsa3l_preprocess.py bin/vsa3l_segment.py bin/validate_cuda_runtime.py bin/compute_metrics.py bin/frame_manifest.py bin/geometry_utils.py /app/bin/
RUN chmod +x /app/bin/*.py

# Validate the image's compiled CUDA targets during the build. This is a
# metadata check (no GPU is needed) and prevents publishing a VSA-3L image
# that cannot run on the cluster's V100 nodes.
RUN python -c "import torch; arch = set((torch._C._cuda_getArchFlags() or '').split()); assert torch.__version__.startswith('2.1.0'), torch.__version__; assert torch.version.cuda == '11.8', torch.version.cuda; assert 'sm_70' in arch, arch"

# Set environment
ENV PATH="/app/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

ENTRYPOINT ["python"]
CMD ["--help"]
