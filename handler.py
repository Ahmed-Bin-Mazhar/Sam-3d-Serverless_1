import base64
import os
import tempfile
import traceback
from typing import Dict, Any

import runpod
import torch
from huggingface_hub import snapshot_download

# -------------------------------
# SAM-3D imports
# -------------------------------
import sys
sys.path.append("/app/sam-3d-objects")

from inference import Inference, load_image, load_single_mask


# -------------------------------
# Persistent volume paths
# -------------------------------
VOLUME_ROOT = "/runpod-volume/sam3d"
CHECKPOINT_DIR = f"{VOLUME_ROOT}/checkpoints/hf"
PIPELINE_YAML = f"{CHECKPOINT_DIR}/pipeline.yaml"
HF_CACHE_DIR = f"{VOLUME_ROOT}/hf_cache"

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_HOME = os.environ.get("HF_HOME", HF_CACHE_DIR)


# -------------------------------
# Checkpoint setup (cold start)
# -------------------------------
def ensure_checkpoints():
    if os.path.exists(PIPELINE_YAML):
        print("[SAM3D] Checkpoints already present.", flush=True)
        return

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is required")

    print("[SAM3D] Downloading checkpoints from HuggingFace...", flush=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(HF_CACHE_DIR, exist_ok=True)

    snapshot_download(
        repo_id="facebook/sam-3d-objects",
        repo_type="model",
        token=HF_TOKEN,
        local_dir=CHECKPOINT_DIR,
        local_dir_use_symlinks=False,
        cache_dir=HF_HOME,
        max_workers=1,
    )

    print("[SAM3D] Checkpoints downloaded.", flush=True)


# -------------------------------
# Model warm load (once per worker)
# -------------------------------
print("[SAM3D] Initializing worker...", flush=True)
ensure_checkpoints()

print("[SAM3D] Loading model...", flush=True)
MODEL = Inference(PIPELINE_YAML, compile=False)
MODEL.eval()

if torch.cuda.is_available():
    print("[SAM3D] Using GPU", flush=True)
else:
    print("[SAM3D] WARNING: CUDA not available, running on CPU", flush=True)

print("[SAM3D] Model ready.", flush=True)


# -------------------------------
# Utility helpers
# -------------------------------
def b64_to_file(b64: str, suffix: str) -> str:
    data = base64.b64decode(b64)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# -------------------------------
# Inference core
# -------------------------------
def run_sam3d(image_b64: str, mask_b64: str, seed: int) -> str:
    image_path = mask_path = out_ply = None

    try:
        image_path = b64_to_file(image_b64, ".png")
        mask_path = b64_to_file(mask_b64, ".png")
        out_ply = tempfile.mktemp(suffix=".ply")

        image = load_image(image_path)
        mask = load_single_mask(mask_path)

        if image is None or mask is None:
            raise ValueError("Invalid image or mask data")

        with torch.no_grad():
            output = MODEL(image, mask, seed=seed)

        output["gs"].save_ply(out_ply)

        return file_to_b64(out_ply)

    finally:
        for p in (image_path, mask_path, out_ply):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# -------------------------------
# RunPod handler
# -------------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input", {})

        image_b64 = inp.get("imageBase64")
        mask_b64 = inp.get("maskBase64")
        options = inp.get("options", {})

        seed = int(options.get("seed", 42))
        outputs = options.get("output", ["ply"])

        if not image_b64 or not mask_b64:
            return {
                "success": False,
                "error": "imageBase64 and maskBase64 are required"
            }

        if "ply" not in outputs:
            return {
                "success": False,
                "error": "Only 'ply' output is supported"
            }

        ply_b64 = run_sam3d(image_b64, mask_b64, seed)

        return {
            "success": True,
            "plyBase64": ply_b64
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "success": False,
            "error": "CUDA out of memory"
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


# -------------------------------
# Start RunPod worker
# -------------------------------
runpod.serverless.start({
    "handler": handler
})
