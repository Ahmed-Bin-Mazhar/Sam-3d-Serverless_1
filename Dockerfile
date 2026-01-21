FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CONDA_PREFIX=/opt/conda

WORKDIR /app

# ----------------------------
# OS deps (needed for building some pip packages)
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    ffmpeg libgl1 libglib2.0-0 \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Kaolin/NVIDIA wheels
ENV PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com" \
    PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html"

# ----------------------------
# Your requirements
# ----------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# RunPod + HuggingFace hub
RUN pip install runpod huggingface-hub

# ----------------------------
# Clone SAM-3D Objects
# ----------------------------
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /app/sam-3d-objects

# ----------------------------
# Install SAM-3D requirements (BUT exclude PyTorch3D lines)
# ----------------------------
RUN echo "===== SAM3D requirements.txt =====" && \
    sed -n '1,200p' /app/sam-3d-objects/requirements.txt && \
    echo "==================================" && \
    # Remove problematic CUDA/C++ deps that often fail from requirements.txt
    # We'll install them separately in a controlled way.
    sed -E '/(^pytorch3d\b|^pytorch3d\s*@|^git\+https:\/\/github\.com\/facebookresearch\/pytorch3d)/d' \
        /app/sam-3d-objects/requirements.txt \
        > /tmp/sam3d_requirements_no_p3d.txt && \
    pip install -r /tmp/sam3d_requirements_no_p3d.txt

# ----------------------------
# Install PyTorch3D separately (official wheel index pattern)
# - For: Python 3.10, CUDA 12.1, PyTorch 2.1.0
# ----------------------------
RUN pip install "pytorch3d" -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html

# ----------------------------
# Install SAM-3D as editable
# ----------------------------
WORKDIR /app/sam-3d-objects
RUN pip install -e ".[inference]" \
 && pip install -e ".[p3d]" \
 && pip install -e ".[dev]" || true
RUN pip install -r /app/sam-3d-objects/requirements.txt
# ----------------------------
# Copy handler
# ----------------------------
WORKDIR /app
COPY handler.py /app/handler.py

ENV HF_HOME=/runpod-volume/sam3d/checkpoints/hf

CMD ["python", "-u", "/app/handler.py"]


