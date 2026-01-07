# =============================================================================
# CASC Base Docker Image
# =============================================================================
# Base image with common dependencies for all cardiac segmentation models
# 
# Build:
#   docker build -t ghcr.io/pmoheban/casc-base:latest containers/base/
#
# Push:
#   docker push ghcr.io/pmoheban/casc-base:latest
# =============================================================================

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="pmoheban"
LABEL description="Base Docker image for CASC cardiac segmentation pipeline"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Create directories
RUN mkdir -p /app /data /models /output /app/bin

# Set working directory
WORKDIR /app

# Add bin to PATH
ENV PATH="/app/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Default command
CMD ["bash"]
