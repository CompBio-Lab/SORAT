# =============================================================================
# SORAT CineMA Docker Image
# =============================================================================
# CineMA cardiac segmentation model with pre-trained weights
# 
# Build (from SORAT root directory):
#   docker build -t ghcr.io/your-org/sorat-cinema:latest -f containers/cinema/Dockerfile .
#
# Push:
#   docker push ghcr.io/your-org/sorat-cinema:latest
#
# Run interactively:
#   docker run --gpus all -it -v /path/to/data:/data ghcr.io/your-org/sorat-cinema:latest
# =============================================================================

# Preserve V100 (SM70) support, matching the VSA-3L runtime.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.source="https://github.com/CompBio-Lab/SORAT"
LABEL description="CineMA cardiac segmentation model for SORAT pipeline"
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

# Constrain torch and torchvision throughout dependency resolution so MONAI/timm
# cannot silently replace the base runtime with a build that drops V100 support.
RUN printf '%s\n' 'torch==2.1.0' 'torchvision==0.16.0' 'torchaudio==2.1.0' 'numpy<2' > /opt/sorat-constraints.txt
ENV PIP_CONSTRAINT=/opt/sorat-constraints.txt
RUN python -m pip install --upgrade "pip<25" && python -m pip install \
    "numpy<2" "scipy<1.14" "pandas<2.3" "matplotlib<3.9" "seaborn<0.14" \
    "SimpleITK==2.3.1" "nibabel<6" "medpy==0.5.2" "tqdm<5" \
    "monai==1.3.2" "scikit-image<0.24" "Pillow<11" \
    "einops==0.8.0" "timm==0.9.16" "hydra-core==1.3.2" "omegaconf==2.3.0" "safetensors==0.4.5"

# No GPU is needed to verify the compiled targets and matching torchvision.
RUN python -c "import torch, torchvision; arch = set((torch._C._cuda_getArchFlags() or '').split()); assert torch.__version__.split('+')[0] == '2.1.0', torch.__version__; assert torchvision.__version__.split('+')[0] == '0.16.0', torchvision.__version__; assert torch.version.cuda == '11.8', torch.version.cuda; assert 'sm_70' in arch, arch; assert torchvision.extension._has_ops(), 'torchvision compiled ops unavailable'"

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
COPY bin/cinema_preprocess.py bin/cinema_segment.py bin/compute_metrics.py bin/frame_manifest.py \
     bin/geometry_utils.py \
     bin/aggregate_results.py bin/discover_postprocess_inputs.py bin/extract_features.py \
     bin/generate_debug_report.py bin/generate_postprocess_comparison.py bin/generate_report.py \
     bin/generate_segmentation_preview.py bin/postprocess_lv_myo.py \
     /app/bin/
RUN chmod +x /app/bin/*.py

# Set environment
ENV PATH="/app/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

RUN python -c "from cinema.segmentation.convunetr import get_model"

ENTRYPOINT ["python"]
CMD ["--help"]
