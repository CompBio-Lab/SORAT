# =============================================================================
# SORAT Atrial nnUNet Docker Image
# =============================================================================
# Atrial nnUNetv2 segmentation model with pre-trained weights.
#
# Build (from SORAT root directory):
#   docker build -t <namespace>/sorat-atrial-nnunet:<tag> -f containers/atrial_nnunet/Dockerfile .
# =============================================================================

# PyTorch 2.1.2 (CUDA 12.1) comes with the base image; reinstalling it in a later
# layer would ship a second copy of torch in every pull.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.source="https://github.com/CompBio-Lab/SORAT"
LABEL description="Atrial nnUNetv2 segmentation model for SORAT pipeline"
LABEL version="1.0.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Versions pinned to those in the image used for the published results.
RUN pip install --upgrade pip && \
    pip install \
        "numpy==1.26.4" "scipy==1.13.1" "pandas==2.2.2" "matplotlib==3.8.4" \
        "SimpleITK==2.5.5" "nibabel==5.4.2" "scikit-image==0.25.2" "MedPy==0.5.2" tqdm \
        "acvl-utils==0.2.6" "dynamic-network-architectures==0.4.4" "batchgenerators==0.25.3" "batchgeneratorsv2==0.3.3" \
        "graphviz==0.21" "imagecodecs==2025.3.30" "seaborn==0.13.2" "yacs==0.1.8" \
        requests "scikit-learn==1.7.2" "tifffile==2025.5.10" "dicom2nifti==2.6.2" "einops==0.8.2" "json-tricks==3.17.3" && \
    pip install --no-deps "nnunetv2==2.7.0"

# Build-time sanity checks to catch dependency drift before push.
RUN python - <<'PY'
import torch, numpy, scipy
from scipy import ndimage  # noqa: F401
import nnunetv2  # noqa: F401
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: F401

import subprocess
subprocess.run(["python", "-m", "pip", "check"], check=True)

assert torch.__version__.startswith('2.1.2'), torch.__version__
assert str(torch.version.cuda).startswith('12.1'), torch.version.cuda
assert numpy.__version__.startswith('1.26.'), numpy.__version__
print('Validated:', torch.__version__, torch.version.cuda, numpy.__version__, scipy.__version__)
PY

RUN mkdir -p /app /app/bin /data /output \
    /models/atrial_nnunet/nnUNet_raw \
    /models/atrial_nnunet/nnUNet_preprocessed \
    /models/atrial_nnunet/nnUNet_results

WORKDIR /app

COPY models/atrial_nnunet/ /models/atrial_nnunet/
COPY bin/atrial_nnunet_preprocess.py bin/atrial_nnunet_segment.py bin/compute_metrics.py bin/discover_models.py bin/nnunet_predict_compat.py bin/frame_manifest.py bin/geometry_utils.py /app/bin/
RUN chmod +x /app/bin/*.py

ENV PATH="/app/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

ENTRYPOINT ["python"]
CMD ["--help"]
