FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    build-essential cmake \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Python
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

# Install runpod SDK + your requirements
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir runpod && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# Bring your code
COPY handler.py /workspace/handler.py
COPY api.py /workspace/api.py
COPY generate_3d_subprocess.py /workspace/generate_3d_subprocess.py

# Clone SAM3D repo and pull checkpoints (you can embed your setup.sh logic here)
# Your setup.sh shows the expected repo layout and checkpoints path. :contentReference[oaicite:7]{index=7}
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /workspace/sam-3d-objects

# Optional but recommended: download checkpoints at build time (avoids cold-start downloads)
# If HF requires auth for this model, inject HF_TOKEN as a build arg and do `huggingface-cli login --token $HF_TOKEN`.
RUN pip install --no-cache-dir 'huggingface-hub[cli]<1.0' && \
    huggingface-cli download \
      --repo-type model \
      --local-dir /workspace/sam-3d-objects/checkpoints/hf-download \
      --max-workers 1 \
      facebook/sam-3d-objects && \
    mv /workspace/sam-3d-objects/checkpoints/hf-download/checkpoints /workspace/sam-3d-objects/checkpoints/hf && \
    rm -rf /workspace/sam-3d-objects/checkpoints/hf-download

ENV PYTHONUNBUFFERED=1
CMD ["python3", "-u", "handler.py"]
