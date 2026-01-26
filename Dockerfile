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

# Installing PyTorch & Dependencies (pip indexes)
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

RUN mamba run -n sam3d-objects python -m pip install --upgrade pip setuptools wheel && \
    mamba run -n sam3d-objects pip install --no-cache-dir "numpy<2"

RUN mamba run -n sam3d-objects pip install --no-cache-dir \
    loguru seaborn

RUN mamba run -n sam3d-objects pip install --no-cache-dir --no-build-isolation --no-use-pep517 "nvidia-pyindex==1.0.9"

RUN mamba run -n sam3d-objects pip install -e ".[dev]" && \
    mamba run -n sam3d-objects pip install -e ".[p3d]"


RUN mamba run -n sam3d-objects python -c "import torch, pytorch3d; print('cuda:', torch.version.cuda, 'avail:', torch.cuda.is_available())"

RUN mamba run -n sam3d-objects pip install --no-cache-dir git+https://github.com/nerfstudio-project/gsplat.git


RUN set -eux; \
    mamba run -n sam3d-objects pip uninstall -y utils3d || true; \
    mamba run -n sam3d-objects pip install --no-cache-dir \
      "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"


RUN mamba run -n sam3d-objects pip install --no-cache-dir open3d==0.18.0
RUN mamba run -n sam3d-objects pip install --no-cache-dir gradio==5.49.0

# Some environments bring a binutils activation script that can break in minimal images
RUN rm -f /workspace/mamba/envs/sam3d-objects/etc/conda/activate.d/activate-binutils_linux-64.sh 2>/dev/null || true


RUN mamba run -n sam3d-objects pip install --no-cache-dir timm


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