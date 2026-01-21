FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CONDA_PREFIX=/opt/conda

WORKDIR /app

# ----------------------------
# OS deps
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    ffmpeg libgl1 libglib2.0-0 \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Kaolin/NVIDIA wheels (must be set BEFORE pip installs)
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
# Install SAM3D requirements.txt (requested)
# NOTE: This step can fail if their requirements include CUDA/C++ builds (e.g. pytorch3d via git).
# If it fails again, we’ll need to switch back to the “safe-filter + install heavy deps separately” approach.
# ----------------------------
RUN pip install -v -r /app/sam-3d-objects/requirements.txt

# ----------------------------
# Install PyTorch3D wheels (recommended even if requirements installs p3d)
# For Python 3.10 + CUDA 12.1 + PyTorch 2.1.0
# If pytorch3d is already installed, pip will keep/upgrade accordingly.
# ----------------------------
RUN pip install -v pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html

# ----------------------------
# Install SAM-3D with extras (requested)
# Using `|| true` exactly as you asked, but for production you should remove it so builds fail loudly.
# ----------------------------
WORKDIR /app/sam-3d-objects
RUN pip install -e ".[inference]" \
 && pip install -e ".[p3d]" \
 && pip install -e ".[dev]" || true

# ----------------------------
# Copy handler
# ----------------------------
WORKDIR /app
COPY handler.py /app/handler.py

ENV HF_HOME=/runpod-volume/sam3d/checkpoints/hf

CMD ["python", "-u", "/app/handler.py"]


