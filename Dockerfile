FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_AUTO_ACTIVATE_BASE=false \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# ----------------------------
# OS deps (keep runtime libs, install build tools temporarily)
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    git wget curl \
    build-essential cmake ninja-build \
  && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Miniforge (mamba)
# ----------------------------
RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    -O /tmp/Miniforge3-Linux-x86_64.sh \
 && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
 && rm -f /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-lc"]

# ----------------------------
# Clone repo (shallow for speed)
# ----------------------------
RUN git clone --depth 1 https://github.com/bilalfawadkhan/sam-3d-objects.git /workspace/sam-3d-objects
WORKDIR /workspace/sam-3d-objects

# ----------------------------
# Create env + core conda deps (do early for caching)
# ----------------------------
RUN mamba env create -n sam3d-objects -f environments/default.yml \
 && rm -f /opt/conda/envs/sam3d-objects/etc/conda/activate.d/activate-binutils_linux-64.sh 2>/dev/null || true

# Upgrade packaging toolchain inside env
RUN mamba run -n sam3d-objects python -m pip install --upgrade pip setuptools wheel

# ----------------------------
# Requirements (IMPORTANT: install into the env, not base!)
# ----------------------------
COPY requirements1.txt /workspace/requirements1.txt
RUN mamba run -n sam3d-objects python -m pip install -r /workspace/requirements1.txt

# ----------------------------
# Torch + pytorch3d (conda)
# ----------------------------
# (Optional but helps prevent weird solver behavior)
RUN mamba config --set channel_priority strict

# Install torch into the target env (NO mamba run)
RUN mamba install -n sam3d-objects -y \
    -c pytorch -c nvidia -c conda-forge \
    pytorch torchvision pytorch-cuda=12.1 \
 && mamba install -n sam3d-objects -y \
    -c pytorch3d -c pytorch -c nvidia -c conda-forge \
    pytorch3d


# ----------------------------
# Hugging Face tooling
# ----------------------------
RUN mamba run -n sam3d-objects python -m pip install "huggingface-hub[cli]<1.0" \
 && mamba run -n sam3d-objects python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)" \
 && mamba run -n sam3d-objects huggingface-cli --help >/dev/null

# ----------------------------
# Hydra pin + patch
# ----------------------------
RUN mamba run -n sam3d-objects python -m pip install "hydra-core>=1.3,<1.4" \
 && mamba run -n sam3d-objects python ./patching/hydra

# ----------------------------
# Install project + extra libs (combine to reduce layers)
# ----------------------------
RUN mamba run -n sam3d-objects python -m pip install -e ".[inference]" || \
    mamba run -n sam3d-objects python -m pip install -e . --no-deps

RUN mamba run -n sam3d-objects python -m pip install \
      trimesh xatlas \
      timm astor easydict einops fvcore iopath yacs \
      scipy scikit-image imageio opencv-python-headless tqdm runpod numpy pillow \
      spconv-cu121 \
 && mamba run -n sam3d-objects mamba install -y -c conda-forge open3d

# ----------------------------
# utils3d pin (single RUN)
# ----------------------------
RUN mamba run -n sam3d-objects python -m pip uninstall -y utils3d || true \
 && mamba run -n sam3d-objects python -m pip install \
      "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

# Sanity check
RUN mamba run -n sam3d-objects python -c "import torch; print('torch ok:', torch.__version__); import utils3d; print('utils3d ok')"

# ----------------------------
# Copy serverless entrypoints
# ----------------------------
COPY handler.py /workspace/sam-3d-objects/handler.py
COPY generate_3d_subprocess.py /workspace/sam-3d-objects/generate_3d_subprocess.py

# ----------------------------
# Cleanup (remove build tooling + shrink conda)
# ----------------------------
RUN apt-get purge -y --auto-remove \
      git wget curl build-essential cmake ninja-build \
 && rm -rf /var/lib/apt/lists/* \
 && mamba clean -a -y \
 && rm -rf /opt/conda/pkgs

CMD ["/opt/conda/envs/sam3d-objects/bin/python", "-u", "/workspace/sam-3d-objects/handler.py"]
