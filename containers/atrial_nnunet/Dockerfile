# =============================================================================
# SORAT Atrial nnUNet Docker Image
# =============================================================================
# Atrial nnUNetv2 segmentation model with pre-trained weights.
#
# Build (from SORAT root directory):
#   docker build -t ghcr.io/your-org/sorat-atrial-nnunet:latest -f containers/atrial_nnunet/Dockerfile .
# =============================================================================

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

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

RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple \
        "torch==2.1.2" "torchvision==0.16.2" "torchaudio==2.1.2" && \
    pip install \
        "numpy==1.26.4" "scipy==1.13.1" "pandas==2.2.2" "matplotlib==3.8.4" \
        SimpleITK nibabel scikit-image medpy tqdm \
        acvl-utils dynamic-network-architectures batchgenerators batchgeneratorsv2 \
        graphviz imagecodecs seaborn yacs \
        requests scikit-learn tifffile dicom2nifti einops json_tricks && \
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
