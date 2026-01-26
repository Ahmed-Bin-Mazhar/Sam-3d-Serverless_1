FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# CUDA paths (helps torch extensions like gsplat find CUDA)
# NOTE: on this image, /usr/local/cuda is usually a symlink to /usr/local/cuda-12.1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

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
      build-essential cmake ninja-build pkg-config \
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
# Repo (change if you want a fork)
# ----------------------------
ARG REPO_URL="https://github.com/facebookresearch/sam-3d-objects.git"
ARG REPO_DIR="/workspace/sam-3d-objects"

RUN git clone --depth=1 ${REPO_URL} ${REPO_DIR}
WORKDIR ${REPO_DIR}

# ----------------------------
# Create env
# ----------------------------
RUN set -eux; \
    mamba env create -f environments/default.yml

# Pip indexes (optional)
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# ----------------------------
# Tooling + NumPy pin (avoid NumPy 2.x ABI issues)
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects python -m pip install --upgrade pip setuptools wheel; \
    mamba run -n sam3d-objects pip install --no-cache-dir "numpy<2"

# ----------------------------
# Ensure CUDA-enabled torch (avoid pip CPU torch overriding conda)
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects pip uninstall -y torch torchvision torchaudio || true; \
    mamba run -n sam3d-objects mamba install -y -c pytorch -c nvidia \
      pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.1

# PyTorch3D
RUN set -eux; \
    mamba run -n sam3d-objects mamba install -y \
      -c pytorch3d -c pytorch -c nvidia -c conda-forge \
      pytorch3d

# Common python utilities you wanted
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir loguru seaborn

# Install project extras WITHOUT deps (prevents troublesome deps like nvidia-pyindex)
RUN set -eux; \
    mamba run -n sam3d-objects pip install -e ".[dev]" --no-deps; \
    mamba run -n sam3d-objects pip install -e ".[p3d]" --no-deps

# (Optional) Kaolin (you had this)
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir kaolin==0.17.0

# Sanity (torch.version.cuda should NOT be None)
RUN set -eux; \
    mamba run -n sam3d-objects python -c "import torch, pytorch3d; print('torch:', torch.__version__, 'torch.version.cuda:', torch.version.cuda, 'p3d:', pytorch3d.__version__)"

# ----------------------------
# Build helpers for CUDA extensions (inside env, optional but safe)
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects mamba install -y -c conda-forge \
      ninja cmake gcc_linux-64 gxx_linux-64

# ----------------------------
# gsplat (clone + local install)
# ----------------------------
RUN set -eux; \
    # prove CUDA is visible
    echo "CUDA_HOME=$CUDA_HOME"; \
    ls -la "$CUDA_HOME"; \
    ls -la "$CUDA_HOME/lib64" || true; \
    ls -la "$CUDA_HOME/include" || true; \
    which nvcc; \
    nvcc --version; \
    # ensure torch is still CUDA build
    mamba run -n sam3d-objects python - <<'PY'
import os, torch
print("CUDA_HOME env:", os.environ.get("CUDA_HOME"))
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
PY
    rm -rf /tmp/gsplat; \
    git clone --recursive https://github.com/nerfstudio-project/gsplat.git /tmp/gsplat; \
    cd /tmp/gsplat; \
    git checkout 2323de5905d5e90e035f792fe65bad0fedd413e7; \
    git submodule update --init --recursive; \
    env CUDA_HOME="$CUDA_HOME" \
        PATH="$CUDA_HOME/bin:$PATH" \
        LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH" \
        TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
      mamba run -n sam3d-objects pip install --no-cache-dir --no-build-isolation -e .; \
    rm -rf /tmp/gsplat

# ----------------------------
# utils3d pinned fork
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects pip uninstall -y utils3d || true; \
    mamba run -n sam3d-objects pip install --no-cache-dir \
      "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

# Other deps you had
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir open3d==0.18.0; \
    mamba run -n sam3d-objects pip install --no-cache-dir gradio==5.49.0; \
    mamba run -n sam3d-objects pip install --no-cache-dir timm

# Some envs bring a binutils activation script that can break in minimal images
RUN rm -f /workspace/mamba/envs/sam3d-objects/etc/conda/activate.d/activate-binutils_linux-64.sh 2>/dev/null || true

# Apply patch (use python to avoid exec permission issues)
RUN set -eux; \
    mamba run -n sam3d-objects python ./patching/hydra

# ----------------------------
# Runtime deps + RunPod SDK + HF client
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir \
      runpod pillow opencv-python-headless imageio tqdm \
      "huggingface-hub[cli]<1.0"

# Final sanity (cuda_available may be False during docker build; that's OK)
RUN set -eux; \
    mamba run -n sam3d-objects python - <<'PY'
import torch, pytorch3d, numpy
print("torch:", torch.__version__, "torch.version.cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("pytorch3d:", pytorch3d.__version__)
print("numpy:", numpy.__version__)
PY

# ----------------------------
# Cleanup (optional)
# ----------------------------
RUN set -eux; \
    apt-get purge -y --auto-remove build-essential cmake ninja-build pkg-config || true; \
    rm -rf /var/lib/apt/lists/*; \
    mamba clean -a -y; \
    rm -rf /workspace/mamba/pkgs

# ----------------------------
# Copy handler
# ----------------------------
COPY handler.py /workspace/sam-3d-objects/handler.py

CMD ["/workspace/mamba/envs/sam3d-objects/bin/python", "-u", "/workspace/sam-3d-objects/handler.py"]
