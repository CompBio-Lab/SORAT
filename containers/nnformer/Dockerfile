# =============================================================================
# SORAT nnFormer Docker Image
# =============================================================================
# nnFormer cardiac segmentation model with pre-trained weights
# 
# Build (from SORAT root directory):
#   docker build -t ghcr.io/your-org/sorat-nnformer:latest -f containers/nnformer/Dockerfile .
#
# Push:
#   docker push ghcr.io/your-org/sorat-nnformer:latest
#
# Run interactively:
#   docker run --gpus all -it -v /path/to/data:/data ghcr.io/your-org/sorat-nnformer:latest
# =============================================================================

# Preserve V100 (SM70) support, matching CINEMA and VSA-3L.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.source="https://github.com/CompBio-Lab/SORAT"
LABEL description="nnFormer cardiac segmentation model for SORAT pipeline"
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

# Keep timm and transitive dependencies from replacing the base CUDA runtime.
RUN printf '%s\n' 'torch==2.1.0' 'torchvision==0.16.0' 'torchaudio==2.1.0' 'numpy<2' > /opt/sorat-constraints.txt
ENV PIP_CONSTRAINT=/opt/sorat-constraints.txt
RUN python -m pip install --upgrade "pip<25" && python -m pip install \
    "numpy<2" "scipy<1.14" "pandas<2.3" "matplotlib<3.9" \
    "SimpleITK==2.3.1" "nibabel<6" "medpy==0.5.2" "tqdm<5" \
    "batchgenerators==0.25.1" "einops==0.8.0" \
    "scikit-image<0.24" "Pillow<11" "timm==0.9.16"

# No GPU is needed to verify the compiled targets and matching torchvision.
RUN python -c "import torch, torchvision; arch = set((torch._C._cuda_getArchFlags() or '').split()); assert torch.__version__.split('+')[0] == '2.1.0', torch.__version__; assert torchvision.__version__.split('+')[0] == '0.16.0', torchvision.__version__; assert torch.version.cuda == '11.8', torch.version.cuda; assert 'sm_70' in arch, arch; assert torchvision.extension._has_ops(), 'torchvision compiled ops unavailable'"

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
COPY bin/nnformer_preprocess.py bin/nnformer_segment.py bin/compute_metrics.py bin/frame_manifest.py bin/geometry_utils.py /app/bin/
RUN chmod +x /app/bin/*.py

# Set environment
ENV PATH="/app/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

RUN python -c "from nnformer.network_architecture.nnFormer_acdc import nnFormer"

ENTRYPOINT ["python"]
CMD ["--help"]
