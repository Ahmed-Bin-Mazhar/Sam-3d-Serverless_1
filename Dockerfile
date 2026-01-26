FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# CUDA paths (for torch extensions like gsplat)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

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
# Miniforge (mamba + conda)
# ----------------------------
RUN set -eux; \
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
      -O /tmp/Miniforge3-Linux-x86_64.sh; \
    bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda; \
    rm -f /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_AUTO_ACTIVATE_BASE=false

# Configure channels with conda (more compatible than mamba config in your setup)
RUN set -eux; \
    conda config --set channel_priority strict; \
    conda config --remove channels defaults || true; \
    conda config --add channels pytorch; \
    conda config --add channels nvidia; \
    conda config --add channels conda-forge

# ----------------------------
# Clone repo
# ----------------------------
ARG REPO_URL="https://github.com/facebookresearch/sam-3d-objects.git"
ARG REPO_DIR="/workspace/sam-3d-objects"
RUN git clone --depth=1 ${REPO_URL} ${REPO_DIR}
WORKDIR ${REPO_DIR}

# ----------------------------
# Create a CLEAN env (do NOT use default.yml)
# ----------------------------
RUN set -eux; \
    mamba create -y -n sam3d-objects python=3.11

# Upgrade pip tooling + pin numpy<2
RUN set -eux; \
    mamba run -n sam3d-objects python -m pip install --upgrade pip setuptools wheel; \
    mamba run -n sam3d-objects pip install --no-cache-dir "numpy<2"

# ----------------------------
# Install CUDA-enabled PyTorch (this avoids the libcublas conflict)
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects mamba install -y \
      pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.1 \
      -c pytorch -c nvidia

# Sanity: must be CUDA build (NOT None)
RUN set -eux; \
    mamba run -n sam3d-objects python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
assert torch.version.cuda is not None, "CPU torch installed (torch.version.cuda is None)"
print("cuda available:", torch.cuda.is_available())
PY

# ----------------------------
# PyTorch3D (match your CUDA torch)
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects mamba install -y \
      pytorch3d \
      -c pytorch3d -c pytorch -c nvidia -c conda-forge

# ----------------------------
# Install project extras without pulling deps (prevents CPU torch / cuda stack drift)
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects pip install -e ".[dev]" --no-deps; \
    mamba run -n sam3d-objects pip install -e ".[p3d]" --no-deps

# Optional: if sam3d_objects requires exact werkzeug, pin it
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir "werkzeug==3.0.6"

# Kaolin / other requested packages
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir kaolin==0.17.0; \
    mamba run -n sam3d-objects pip install --no-cache-dir open3d==0.18.0; \
    mamba run -n sam3d-objects pip install --no-cache-dir gradio==5.49.0; \
    mamba run -n sam3d-objects pip install --no-cache-dir timm

# utils3d pinned fork
RUN set -eux; \
    mamba run -n sam3d-objects pip uninstall -y utils3d || true; \
    mamba run -n sam3d-objects pip install --no-cache-dir \
      "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

# Some envs bring a binutils activation script that can break in minimal images
RUN rm -f /opt/conda/envs/sam3d-objects/etc/conda/activate.d/activate-binutils_linux-64.sh 2>/dev/null || true

# Apply patch
RUN set -eux; \
    mamba run -n sam3d-objects python ./patching/hydra

# ----------------------------
# gsplat (CUDA extension)
# IMPORTANT: install inside env with CUDA_HOME visible
# ----------------------------
ARG GSPLAT_COMMIT="2323de5905d5e90e035f792fe65bad0fedd413e7"
RUN set -eux; \
    rm -rf /tmp/gsplat; \
    git clone --recursive https://github.com/nerfstudio-project/gsplat.git /tmp/gsplat; \
    cd /tmp/gsplat; \
    git checkout ${GSPLAT_COMMIT}; \
    git submodule update --init --recursive; \
    mamba run -n sam3d-objects bash -lc '\
      export CUDA_HOME=/usr/local/cuda; \
      export PATH="$CUDA_HOME/bin:$PATH"; \
      export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"; \
      export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"; \
      python -c "import os, torch; print(\"CUDA_HOME=\", os.environ.get(\"CUDA_HOME\")); print(\"torch=\", torch.__version__, \"cuda=\", torch.version.cuda)"; \
      pip install --no-cache-dir --no-build-isolation . \
    '; \
    rm -rf /tmp/gsplat

# ----------------------------
# RunPod deps
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir \
      runpod pillow opencv-python-headless imageio tqdm \
      "huggingface-hub[cli]<1.0"

# Final sanity
RUN set -eux; \
    mamba run -n sam3d-objects python - <<'PY'
import torch, pytorch3d, numpy
print("torch:", torch.__version__, "torch.version.cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("pytorch3d:", pytorch3d.__version__)
print("numpy:", numpy.__version__)
PY

# Cleanup (optional)
RUN set -eux; \
    apt-get purge -y --auto-remove build-essential cmake ninja-build pkg-config || true; \
    rm -rf /var/lib/apt/lists/*; \
    mamba clean -a -y; \
    rm -rf /opt/conda/pkgs

# Copy handler
COPY handler.py /workspace/sam-3d-objects/handler.py

CMD ["/opt/conda/envs/sam3d-objects/bin/python", "-u", "/workspace/sam-3d-objects/handler.py"]
