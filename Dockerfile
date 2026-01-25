FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace
SHELL ["/bin/bash", "-lc"]

# ----------------------------
# System deps
# ----------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      bash ca-certificates \
      git wget curl \
      build-essential cmake \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1; \
    rm -rf /var/lib/apt/lists/*

# ----------------------------
# Miniforge (mamba)
# ----------------------------
RUN set -eux; \
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
      -O /tmp/Miniforge3-Linux-x86_64.sh; \
    bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /workspace/mamba; \
    rm -f /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH=/workspace/mamba/bin:$PATH
ENV CONDA_AUTO_ACTIVATE_BASE=false

# ----------------------------
# Clone repo
# ----------------------------
RUN git clone https://github.com/bilalfawadkhan/sam-3d-objects.git /workspace/sam-3d-objects
WORKDIR /workspace/sam-3d-objects

# ----------------------------
# Create env
# ----------------------------
RUN set -eux; \
    mamba env create -f environments/default.yml

# ----------------------------
# IMPORTANT: avoid NumPy 2.x ABI issues
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects python -m pip install --upgrade pip setuptools wheel; \
    mamba run -n sam3d-objects pip install --no-cache-dir "numpy<2"

RUN mamba run -n sam3d-objects pip install --no-cache-dir \
    loguru seaborn

# ----------------------------
# Install torch + cuda + pytorch3d (prebuilt)
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects mamba install -y \
      -c pytorch -c nvidia -c conda-forge \
      pytorch torchvision pytorch-cuda=12.1

RUN set -eux; \
    mamba run -n sam3d-objects mamba install -y \
      -c pytorch3d -c pytorch -c nvidia -c conda-forge \
      pytorch3d

# Some environments bring a binutils activation script that can break in minimal images
RUN rm -f /workspace/mamba/envs/sam3d-objects/etc/conda/activate.d/activate-binutils_linux-64.sh 2>/dev/null || true

# hydra fix / patch support
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir "hydra-core>=1.3,<1.4"

# ----------------------------
# Install SAM3D repo WITHOUT letting pip pull deps again
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir -e . --no-deps

# Apply patch (use python to avoid exec permission issues)
RUN set -eux; \
    mamba run -n sam3d-objects python ./patching/hydra

# ----------------------------
# utils3d pin (your known-good)
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects pip uninstall -y utils3d || true; \
    mamba run -n sam3d-objects pip install --no-cache-dir \
      "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

# ----------------------------
# Runtime deps + RunPod SDK + HF client
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir \
      runpod pillow opencv-python-headless imageio tqdm \
      "huggingface-hub[cli]<1.0"

# Quick sanity check
RUN set -eux; \
    mamba run -n sam3d-objects python - <<'PY'
import torch, pytorch3d, numpy
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("pytorch3d:", pytorch3d.__version__)
print("numpy:", numpy.__version__)
PY

# ----------------------------
# Cleanup (optional)
# ----------------------------
RUN set -eux; \
    apt-get purge -y --auto-remove build-essential cmake; \
    rm -rf /var/lib/apt/lists/*; \
    mamba clean -a -y; \
    rm -rf /workspace/mamba/pkgs

# ----------------------------
# Copy handler
# ----------------------------
COPY handler.py /workspace/sam-3d-objects/handler.py

CMD ["/workspace/mamba/envs/sam3d-objects/bin/python", "-u", "/workspace/sam-3d-objects/handler.py"]