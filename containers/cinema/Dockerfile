# =============================================================================
# CASC CineMA Docker Image
# =============================================================================
# CineMA cardiac segmentation model with pre-trained weights
# 
# Build (from CASC root directory):
#   docker build -t ghcr.io/pmoheban/casc-cinema:latest -f containers/cinema/Dockerfile .
#
# Push:
#   docker push ghcr.io/pmoheban/casc-cinema:latest
#
# Run interactively:
#   docker run --gpus all -it -v /path/to/data:/data ghcr.io/pmoheban/casc-cinema:latest
# =============================================================================

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="pmoheban"
LABEL description="CineMA cardiac segmentation model for CASC pipeline"
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

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install \
    numpy scipy pandas matplotlib \
    SimpleITK nibabel medpy tqdm \
    monai scikit-image Pillow \
    einops timm hydra-core omegaconf safetensors

# Create directories
RUN mkdir -p /app /data /output /app/bin \
    /models/cinema/finetuned/segmentation

WORKDIR /app

# Copy CineMA source code
COPY cinema/ /app/cinema/

# Copy model weights (included in image for convenience)
COPY models/cinema/ /models/cinema/

# Copy pipeline scripts
COPY bin/cinema_preprocess.py bin/cinema_segment.py bin/compute_metrics.py /app/bin/
RUN chmod +x /app/bin/*.py

# Set environment
ENV PATH="/app/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

ENTRYPOINT ["python"]
CMD ["--help"]
