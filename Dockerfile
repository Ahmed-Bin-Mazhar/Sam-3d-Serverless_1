FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

# ----------------------------
# Runtime deps
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Build deps (we purge later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Miniforge (mamba)
# ----------------------------
RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    -O /tmp/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm -f /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-lc"]
ENV CONDA_AUTO_ACTIVATE_BASE=false



# ----------------------------
# Clone SAM3D repo (your fork)
# ----------------------------
RUN git clone https://github.com/bilalfawadkhan/sam-3d-objects.git /workspace/sam-3d-objects
WORKDIR /workspace/sam-3d-objects

# ----------------------------
# Create conda env (explicit name to avoid surprises)
# ----------------------------
RUN mamba env create -n sam3d-objects -f environments/default.yml

RUN mamba run -n sam3d-objects pip install --no-cache-dir "huggingface-hub[cli]<1.0"
RUN mamba run -n sam3d-objects huggingface-cli --version



# Fix the binutils activate script issue you saw earlier
RUN rm -f /opt/conda/envs/sam3d-objects/etc/conda/activate.d/activate-binutils_linux-64.sh 2>/dev/null || true

# Upgrade packaging toolchain inside env
RUN mamba run -n sam3d-objects python -m pip install --upgrade pip setuptools wheel

# ----------------------------
# Install PyTorch CUDA 12.1 FIRST (prevents "torch not found" during builds)
# ----------------------------
RUN mamba run -n sam3d-objects mamba install -y \
  -c pytorch -c nvidia -c conda-forge \
  pytorch torchvision pytorch-cuda=12.1

# Optional: pytorch3d (keep if your pipeline needs it)
RUN mamba run -n sam3d-objects mamba install -y \
  -c pytorch3d -c pytorch -c nvidia -c conda-forge \
  pytorch3d

# Hydra pin + patch (matches your flow)
RUN mamba run -n sam3d-objects pip install --no-cache-dir "hydra-core>=1.3,<1.4"
RUN mamba run -n sam3d-objects python ./patching/hydra

# ----------------------------
# Install sam-3d-objects (prefer extras if available)
# If extras fail for your fork, fallback to `-e . --no-deps` like your old file.
# ----------------------------
RUN mamba run -n sam3d-objects pip install --no-cache-dir -e ".[inference]" || \
    mamba run -n sam3d-objects pip install --no-cache-dir -e . --no-deps

# utils3d pin (from your old Dockerfile)
RUN mamba run -n sam3d-objects pip uninstall -y utils3d || true && \
    mamba run -n sam3d-objects pip install --no-cache-dir \
      "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

# App deps needed by handler/subprocess (serverless)
RUN mamba run -n sam3d-objects pip install --no-cache-dir \
    runpod numpy pillow opencv-python-headless imageio tqdm \
    "huggingface-hub[cli]<1.0"

# Quick sanity check
RUN mamba run -n sam3d-objects python -c "import torch; print('torch ok:', torch.__version__); import utils3d; print('utils3d ok')"


# ----------------------------
# Copy serverless entrypoints
# ----------------------------
COPY handler.py /workspace/sam-3d-objects/handler.py
COPY generate_3d_subprocess.py /workspace/sam-3d-objects/generate_3d_subprocess.py

# ----------------------------
# Cleanup to shrink image
# ----------------------------
RUN apt-get purge -y --auto-remove \
      git wget curl build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && mamba clean -a -y \
    && rm -rf /opt/conda/pkgs

# ----------------------------
# RunPod Serverless entrypoint
# ----------------------------
CMD ["/opt/conda/envs/sam3d-objects/bin/python", "-u", "/workspace/sam-3d-objects/handler.py"]
