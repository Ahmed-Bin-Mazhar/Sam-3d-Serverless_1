FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# CUDA paths (torch extensions like gsplat look for this)
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
    bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /workspace/mamba; \
    rm -f /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH=/workspace/mamba/bin:$PATH
ENV CONDA_AUTO_ACTIVATE_BASE=false

# Prefer channels + strict priority (use conda, not mamba)
RUN set -eux; \
    conda config --set channel_priority strict; \
    conda config --remove channels defaults || true; \
    conda config --add channels conda-forge; \
    conda config --add channels nvidia; \
    conda config --add channels pytorch

# ----------------------------
# Repo
# ----------------------------
ARG REPO_URL="https://github.com/facebookresearch/sam-3d-objects.git"
ARG REPO_DIR="/workspace/sam-3d-objects"
RUN git clone --depth=1 ${REPO_URL} ${REPO_DIR}
WORKDIR ${REPO_DIR}

# ----------------------------
# Create env (from repo)
# ----------------------------
RUN set -eux; \
    mamba env create -f environments/default.yml

# ----------------------------
# Pip indexes (optional)
# ----------------------------
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# ----------------------------
# Tooling + NumPy pin (avoid NumPy 2.x ABI issues)
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects python -m pip install --upgrade pip setuptools wheel; \
    mamba run -n sam3d-objects pip install --no-cache-dir "numpy<2"

# ----------------------------
# FORCE CUDA PyTorch (remove any CPU torch first)
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects mamba remove -y pytorch torchvision torchaudio pytorch-cuda libtorch || true; \
    mamba run -n sam3d-objects pip uninstall -y torch torchvision torchaudio || true; \
    mamba run -n sam3d-objects mamba install -y \
      pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.1 \
      -c pytorch -c nvidia --override-channels

# ----------------------------
# PyTorch3D (install AFTER CUDA torch)
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects mamba install -y \
      pytorch3d \
      -c pytorch3d -c pytorch -c nvidia -c conda-forge --override-channels

# Sanity: torch.version.cuda must NOT be None
RUN set -eux; \
    mamba run -n sam3d-objects python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
assert torch.version.cuda is not None, "CPU torch installed (torch.version.cuda is None)"
PY

# ----------------------------
# Small utilities you wanted
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir loguru seaborn

# ----------------------------
# Install project extras WITHOUT deps (avoids nvidia-pyindex etc)
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects pip install -e ".[dev]" --no-deps; \
    mamba run -n sam3d-objects pip install -e ".[p3d]" --no-deps

# (Optional) Kaolin
RUN set -eux; \
    mamba run -n sam3d-objects pip install --no-cache-dir kaolin==0.17.0

# ----------------------------
# gsplat (CUDA extension)
# - Needs CUDA_HOME visible INSIDE build backend
# - Disable build isolation so it can see torch already installed
# ----------------------------
ARG GSPLAT_COMMIT="2323de5905d5e90e035f792fe65bad0fedd413e7"
RUN set -eux; \
    mamba run -n sam3d-objects bash -lc '\
      export CUDA_HOME=/usr/local/cuda; \
      export PATH="$CUDA_HOME/bin:$PATH"; \
      export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"; \
      export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"; \
      python -c "import os, torch; print(\"CUDA_HOME=\", os.environ.get(\"CUDA_HOME\")); print(\"torch=\", torch.__version__, \"cuda=\", torch.version.cuda)"; \
      pip install --no-cache-dir --no-build-isolation \
        "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@'"${GSPLAT_COMMIT}"'" \
    '

# ----------------------------
# utils3d pinned fork
# ----------------------------
RUN set -eux; \
    mamba run -n sam3d-objects pip uninstall -y utils3d || true; \
    mamba run -n sam3d-objects pip install --no-cache-dir \
      "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

# ----------------------------
# Other deps you had
# ----------------------------
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

# Final sanity
RUN set -eux; \
    mamba run -n sam3d-objects python - <<'PY'
import torch, numpy
print("torch:", torch.__version__, "torch.version.cuda:", torch.version.cuda)
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
