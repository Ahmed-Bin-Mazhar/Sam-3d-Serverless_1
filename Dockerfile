# RunPod GPU base image (includes CUDA + common tooling)
FROM runpod/base:0.6.2-cuda12.1.0

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# --- OS deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# --- Miniforge / Mamba ---
# Install into /opt/conda (typical) and expose on PATH
RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    -O /tmp/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm -f /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:${PATH}"

# (Optional) speed / reliability for conda
RUN conda config --set channel_priority strict && \
    conda config --add channels conda-forge

# --- Get SAM 3D Objects code ---
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /workspace/sam-3d-objects
WORKDIR /workspace/sam-3d-objects

# --- Create env from repo YAML ---
# This is your step 8–9 (but done in build).
RUN mamba env create -f environments/default.yml

# Your pip index settings (steps 10 + 13)
# Keep them as ENV so pip inside env sees them.
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# --- Install extras (steps 11,12,14) + hydra patch (step 15) ---
# Use mamba run so we don't need "mamba activate" in Docker build layers.
RUN mamba run -n sam3d-objects pip install --no-cache-dir -e ".[dev]" && \
    mamba run -n sam3d-objects pip install --no-cache-dir -e ".[p3d]" && \
    mamba run -n sam3d-objects pip install --no-cache-dir -e ".[inference]" && \
    ./patching/hydra

# --- Hugging Face CLI lib (your step 16) ---
RUN mamba run -n sam3d-objects pip install --no-cache-dir "huggingface-hub[cli]<1.0"

# --- RunPod SDK (required for serverless handler) ---
RUN mamba run -n sam3d-objects pip install --no-cache-dir runpod

# Copy handler into image
WORKDIR /workspace
COPY handler.py /workspace/handler.py

# Start the serverless worker
CMD ["bash", "-lc", "mamba run -n sam3d-objects python -u /workspace/handler.py"]
