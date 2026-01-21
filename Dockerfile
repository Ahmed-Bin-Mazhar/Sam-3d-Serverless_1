# Optimized RunPod Serverless Dockerfile for SAM-3D-Objects
# - Uses a CUDA + PyTorch runtime base
# - Installs only needed OS deps (incl. build tools for some CUDA/PyTorch extensions)
# - Installs Python deps once (with Kaolin/NVIDIA indexes set *before* pip)
# - Clones SAM-3D-Objects (optionally pinned to a commit/tag)
# - Copies handler.py and starts RunPod serverless worker

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ----------------------------
# System dependencies
# ----------------------------
# build-essential/cmake/ninja are often needed for pip packages that compile extensions (e.g., nvdiffrast / some 3D deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Python packaging setup
# ----------------------------
RUN python -m pip install --upgrade pip setuptools wheel

# Kaolin / NVIDIA wheels (must be set BEFORE installing requirements)
# Adjust these only if you intentionally use different torch/cuda versions.
ENV PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com" \
    PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html"

# ----------------------------
# Install Python dependencies
# ----------------------------
# Copy first to maximize layer caching when your code changes
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# RunPod SDK + HuggingFace hub (explicit, since handler imports them)
RUN pip install runpod huggingface-hub

# ----------------------------
# Get SAM-3D-Objects repo
# ----------------------------
# Optionally pin to a commit for reproducible builds:
#   docker build --build-arg SAM3D_REF=<commit_or_tag> .
ARG SAM3D_REPO="https://github.com/facebookresearch/sam-3d-objects.git"
ARG SAM3D_REF="main"

RUN git clone --depth 1 --branch "${SAM3D_REF}" "${SAM3D_REPO}" /app/sam-3d-objects

# Install SAM-3D-Objects (extras used by your inference code)
WORKDIR /app/sam-3d-objects
RUN pip install -e ".[inference]" \
 && pip install -e ".[p3d]" \
 && pip install -e ".[dev]" || true
RUN pip install -r /app/sam-3d-objects/requirements.txt
# NOTE: If any of these extras fail due to optional deps, you can remove "|| true"
# and fix the missing dependency explicitly (recommended for production).
ENV CONDA_PREFIX=/opt/conda
# ----------------------------
# Copy handler
# ----------------------------
WORKDIR /app
COPY handler.py /app/handler.py

# Helpful defaults for HF caching (your handler already defaults to /runpod-volume/sam3d/hf_cache)
ENV HF_HOME=/runpod-volume/sam3d/hf_cache

CMD ["python", "-u", "/app/handler.py"]
