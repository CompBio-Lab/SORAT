# =============================================================================
# CASC nnFormer Docker Image
# =============================================================================
# nnFormer cardiac segmentation model with pre-trained weights
# 
# Build (from CASC root directory):
#   docker build -t ghcr.io/pmoheban/casc-nnformer:latest -f containers/nnformer/Dockerfile .
#
# Push:
#   docker push ghcr.io/pmoheban/casc-nnformer:latest
#
# Run interactively:
#   docker run --gpus all -it -v /path/to/data:/data ghcr.io/pmoheban/casc-nnformer:latest
# =============================================================================

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="pmoheban"
LABEL description="nnFormer cardiac segmentation model for CASC pipeline"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# nnFormer environment variables (paths inside container)
ENV nnFormer_raw_data_base=/models/nnformer/nnFormer_raw
ENV nnFormer_preprocessed=/models/nnformer/nnFormer_preprocessed
ENV RESULTS_FOLDER=/models/nnformer/nnFormer_trained_models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install \
    numpy scipy pandas matplotlib \
    SimpleITK nibabel medpy tqdm \
    batchgenerators einops \
    scikit-image Pillow \
    timm

# Create directories
RUN mkdir -p /app /data /output /app/bin \
    /models/nnformer/nnFormer_raw \
    /models/nnformer/nnFormer_preprocessed \
    /models/nnformer/nnFormer_trained_models

WORKDIR /app

# Copy nnFormer source code
COPY nnformer/ /app/nnformer/

# Copy model weights (trained model files)
COPY models/nnformer/ /models/nnformer/

# Copy pipeline scripts
COPY bin/nnformer_preprocess.py bin/nnformer_segment.py bin/compute_metrics.py /app/bin/
RUN chmod +x /app/bin/*.py

# Set environment
ENV PATH="/app/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

ENTRYPOINT ["python"]
CMD ["--help"]
