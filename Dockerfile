FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ----------------------------
# System dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Clone SAM-3D
# ----------------------------
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git
WORKDIR /app/sam-3d-objects

COPY requirements.txt ./requirements.txt

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
# ----------------------------
# Python dependencies
# ----------------------------
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install -e '.[dev]'
RUN pip install -e '.[p3d]'
RUN pip install -e '.[inference]'
RUN pip install runpod huggingface-hub pillow
 
# ----------------------------
# Copy handler
# ----------------------------
COPY handler.py /app/handler.py

CMD ["python3", "-u","/app/handler.py"]


