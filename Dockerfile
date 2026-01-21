FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ----------------------------
# System deps
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates bash \
    build-essential cmake \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Miniconda (as per your script, but install to /opt/conda)
# ----------------------------
ENV CONDA_ROOT=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_ROOT && \
    rm -f /tmp/miniconda.sh

ENV PATH=$CONDA_ROOT/bin:$PATH
SHELL ["/bin/bash", "-lc"]

# ----------------------------
# Copy your app files first (so rebuilds are faster)
# ----------------------------
COPY requirements.txt /workspace/requirements.txt
COPY handler.py /workspace/handler.py
COPY generate_3d_subprocess.py /workspace/generate_3d_subprocess.py
# (Optional) only if you still want FastAPI local runs; serverless doesn't need it
COPY api.py /workspace/api.py

# ----------------------------
# Clone SAM3D repo
# ----------------------------
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /workspace/sam-3d-objects

# ----------------------------
# Create/Update conda env from sam-3d-objects environments/default.yml
# Then install exactly like your setup.sh
# ----------------------------
WORKDIR /workspace/sam-3d-objects
RUN conda env create -f environments/default.yml || conda env update -f environments/default.yml --prune

# Use the env for the remaining RUN lines
ENV CONDA_DEFAULT_ENV=sam3d-objects
ENV PATH=$CONDA_ROOT/envs/sam3d-objects/bin:$PATH

# --- 4. Installing PyTorch & Dependencies (same as your setup.sh) ---
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
RUN pip install -e '.[dev]' && \
    pip install -e '.[p3d]'

# --- 5. Installing Inference & Patching (same as your setup.sh) ---
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
RUN pip install -e '.[inference]' && \
    if [ -f "./patching/hydra" ]; then chmod +x ./patching/hydra && ./patching/hydra; fi

# --- 6. Downloading Model Checkpoints (same as your setup.sh) ---
RUN pip install 'huggingface-hub[cli]<1.0' && \
    TAG=hf && \
    if [ ! -d "checkpoints/${TAG}" ]; then \
      mkdir -p checkpoints && \
      huggingface-cli download \
        --repo-type model \
        --local-dir checkpoints/${TAG}-download \
        --max-workers 1 \
        facebook/sam-3d-objects && \
      mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG} && \
      rm -rf checkpoints/${TAG}-download; \
    else \
      echo "Checkpoints already present. Skipping download."; \
    fi

# --- 7. Final Requirements (Root) (same as your setup.sh) ---
WORKDIR /workspace
RUN pip install --no-cache-dir runpod && \
    if [ -f "requirements.txt" ]; then \
      pip install -r requirements.txt && \
      pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation; \
    else \
      echo "No requirements.txt found in /workspace"; \
    fi

ENV PYTHONUNBUFFERED=1

# For RunPod Serverless: handler.py is the entrypoint
CMD ["bash", "-lc", "python -u /workspace/handler.py"]
