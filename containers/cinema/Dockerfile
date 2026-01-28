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
    numpy scipy pandas matplotlib seaborn \
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

# Create missing CineMA config(s) inside the container (keeps repo clean)
RUN if [ ! -f /models/cinema/finetuned/segmentation/mnms2_sax/config.yaml ]; then \
        mkdir -p /models/cinema/finetuned/segmentation/mnms2_sax && \
        printf '%s\n' \
        "grad_ckpt: True" \
        "data:" \
        "    sax:" \
        "        spacing: [1.0, 1.0, 10.0]" \
        "        patch_size: [192, 192, 16]" \
        "        in_chans: 1" \
        "model:" \
        "    name: convunetr" \
        "    views: sax" \
        "    out_chans: 4" \
        "    convunetr:" \
        "        size: base" \
        "        enc_patch_size: [4, 4, 1]" \
        "        enc_scale_factor: [2, 2, 1]" \
        "        enc_conv_chans: [64, 128]" \
        "        enc_conv_n_blocks: 2" \
        "        dec_chans: [32, 64, 128, 256, 512]" \
        "        dec_patch_size: [2, 2, 1]" \
        "        dec_scale_factor: [2, 2, 1]" \
        "        dropout: 0.1" \
        "        drop_path: 0.1" \
        > /models/cinema/finetuned/segmentation/mnms2_sax/config.yaml; \
    fi

# Validate CineMA model configs are present to avoid runtime failures
RUN test -f /models/cinema/finetuned/segmentation/acdc_sax/config.yaml && \
    test -f /models/cinema/finetuned/segmentation/mnms_sax/config.yaml && \
    test -f /models/cinema/finetuned/segmentation/mnms2_sax/config.yaml

# Copy pipeline scripts
COPY bin/cinema_preprocess.py bin/cinema_segment.py bin/compute_metrics.py /app/bin/
RUN chmod +x /app/bin/*.py

# Set environment
ENV PATH="/app/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

ENTRYPOINT ["python"]
CMD ["--help"]
