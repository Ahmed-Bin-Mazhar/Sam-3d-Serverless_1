FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# CUDA paths (torch extensions like gsplat rely on these)
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
# Repo
# ----------------------------
ARG REPO_URL="https://github.com/facebookresearch/sam-3d-objects.git"
ARG REPO_DIR="/workspace/sam-3d-objects"
RUN git clone --depth=1 ${REPO_URL} ${REPO_DIR}
WORKDIR ${REPO_DIR}

# ----------------------------
# Create env (from repo yml)
# ----------------------------
RUN set -eux; \
    mamba env create -f environments/default.yml

# ----------------------------
# IMPORTANT: lock pip to never override torch by accident
# (we install CUDA torch via conda, and then avoid pip deps that would change it)
# ----------------------------

# Pin NumPy <2 to avoid ABI issues
RUN set -eux; \
    mamba run -n sam3d-objects python -m pip install --upgrade pip setuptools wheel; \
    mamba run -n sam3d-objects pip install --no-cache-dir "numpy<2"

# Remove any torch that default.yml might have pulled in via pip/conda
RUN set -eux; \
    mamba run -n sam3d-objects pip uninstall -y torch torchvision torchaudio || true; \
    mamba run -n sam3d-objects mamba remove -y pytorch torchvision torchaudio pytorch-cuda || true

# Install CUDA-enabled torch strictly via conda
RUN set -eux; \
    mamba run -n sam3d-objects mamba install -y -c pytorch -c nvidia \
      pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.1

# Install PyTorch3D via conda (matches CUDA torch better than pip)
RUN set -eux; \
    mamba run -n sam3d-objects mamba install -y \
      -c pytorch3d -c pytorch -c nvidia -c conda-forge \
      pytorch3d

# Sanity: torch must be CUDA build (torch.version.cuda must NOT be None)
RUN set -eux; \
    mamba run -n sam3d-objects python -c "import torch; print('torch:', torch.__version__, 'torch.version.cuda:', torch.version.cuda)"; \
    test "$(mamba run -n sam3d-objects python -c "import torch; print(torch.version.cuda or '')")" != ""

# Small utilities (safe)
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir --no-deps loguru seaborn

# Project extras WITHOUT deps (prevents nvidia-pyindex and prevents pip from touching torch)
RUN set -eux; \
    mamba run -n sam3d-objects pip install -e ".[dev]" --no-deps; \
    mamba run -n sam3d-objects pip install -e ".[p3d]" --no-deps

# Kaolin: install without deps so it doesn't upgrade/replace torch
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir --no-deps kaolin==0.17.0

# Re-check torch after any pip installs
RUN set -eux; \
    mamba run -n sam3d-objects python -c "import torch; print('torch:', torch.__version__, 'torch.version.cuda:', torch.version.cuda)"; \
    test "$(mamba run -n sam3d-objects python -c "import torch; print(torch.version.cuda or '')")" != ""

# Build helpers (CUDA extensions)
RUN set -eux; \
    mamba run -n sam3d-objects mamba install -y -c conda-forge \
      ninja cmake gcc_linux-64 gxx_linux-64

# ----------------------------
# gsplat (clone + install from pinned commit)
# ----------------------------
RUN set -eux; \
    # show CUDA compiler + torch CUDA build
    which nvcc; nvcc --version; \
    mamba run -n sam3d-objects python -c "import torch; print('torch:', torch.__version__, 'torch.version.cuda:', torch.version.cuda)"; \
    rm -rf /tmp/gsplat; \
    git clone --recursive https://github.com/nerfstudio-project/gsplat.git /tmp/gsplat; \
    cd /tmp/gsplat; \
    git checkout 2323de5905d5e90e035f792fe65bad0fedd413e7; \
    git submodule update --init --recursive; \
    env CUDA_HOME="$CUDA_HOME" \
        PATH="$CUDA_HOME/bin:$PATH" \
        LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH" \
        TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
      mamba run -n sam3d-objects pip install --no-cache-dir --no-build-isolation .; \
    rm -rf /tmp/gsplat

# ----------------------------
# utils3d pinned fork
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects pip uninstall -y utils3d || true; \
    mamba run -n sam3d-objects pip install --no-cache-dir \
      "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

# Other deps (keep them, but avoid deps that can alter torch)
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir --no-deps open3d==0.18.0; \
    mamba run -n sam3d-objects pip install --no-cache-dir --no-deps gradio==5.49.0; \
    mamba run -n sam3d-objects pip install --no-cache-dir --no-deps timm==0.9.16; \
    mamba run -n sam3d-objects pip install --no-cache-dir --no-deps werkzeug==3.0.6 flask==3.0.3 loguru==0.7.2

# Some envs bring a binutils activation script that can break in minimal images
RUN rm -f /workspace/mamba/envs/sam3d-objects/etc/conda/activate.d/activate-binutils_linux-64.sh 2>/dev/null || true

# Apply patch (use python to avoid exec permission issues)
RUN set -eux; \
    mamba run -n sam3d-objects python ./patching/hydra

# ----------------------------
# Runtime deps + RunPod SDK + HF client
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir --no-deps \
      runpod pillow opencv-python-headless imageio tqdm; \
    mamba run -n sam3d-objects pip install --no-cache-dir --no-deps "huggingface-hub[cli]<1.0"

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
