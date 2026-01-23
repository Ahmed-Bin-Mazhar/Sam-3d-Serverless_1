# Use the devel version instead of runtime because SAM-3D (Kaolin/Pointops) 
# needs to compile CUDA kernels during the 'pip install' phase.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set environment rules
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # IMPORTANT: inference.py uses this to find CUDA. 
    # In this image, nvcc and headers are in /usr/local/cuda.
    CONDA_PREFIX=/usr/local/cuda \
    CUDA_HOME=/usr/local/cuda

WORKDIR /app

# Install OS dependencies for 3D and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    ffmpeg libgl1 libglib2.0-0 \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Configure Pip to find the high-performance 3D libraries
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121 https://pypi.nvidia.com" \
    PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html"

# Install core project requirements
# Ensure 'runpod', 'omegaconf', 'hydra-core', and 'utils3d' are in your requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Clone and install SAM-3D
RUN git clone --depth 1 https://github.com/facebookresearch/sam-3d-objects.git /app/sam-3d-objects
WORKDIR /app/sam-3d-objects
RUN pip install .

# Copy your specific handler
COPY handler.py /app/sam-3d-objects/handler.py

# Set the checkpoint directory (ensure you mount a volume here in RunPod)
ENV HF_HOME=/runpod-volume/sam-3d-objects/checkpoints/hf

# Execute the RunPod handler
CMD ["python", "-u", "/app/sam-3d-objects/handler.py"]