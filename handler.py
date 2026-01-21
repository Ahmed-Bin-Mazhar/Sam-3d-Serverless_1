import base64
import os
import tempfile
from typing import Dict, Any

import runpod
import torch

# SAM-3D imports
import sys
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

from huggingface_hub import snapshot_download


# -------------------------------------------------
# Fixed volume paths (sam3d)
# -------------------------------------------------
VOLUME_ROOT = "/runpod-volume/sam3d"
CHECKPOINT_DIR = f"{VOLUME_ROOT}/checkpoints/hf"
PIPELINE_YAML = f"{CHECKPOINT_DIR}/pipeline.yaml"
HF_CACHE_DIR = f"{VOLUME_ROOT}/hf_cache"

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_HOME = os.environ.get("HF_HOME", HF_CACHE_DIR)


# -------------------------------------------------
# Ensure checkpoints exist (once per container)
# -------------------------------------------------
def ensure_checkpoints():
    if os.path.exists(PIPELINE_YAML):
        return

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is required")

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


# -------------------------------------------------
# Load model once (warm start)
# -------------------------------------------------
print("[handler] Ensuring checkpoints...", flush=True)
ensure_checkpoints()

print("[handler] Loading SAM-3D model...", flush=True)
MODEL = Inference(PIPELINE_YAML, compile=False)
print("[handler] Model loaded.", flush=True)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def b64_to_file(b64: str, suffix: str) -> str:
    data = base64.b64decode(b64)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# -------------------------------------------------
# Inference
# -------------------------------------------------
def run_sam3d(image_b64: str, mask_b64: str, seed: int) -> str:
    image_path = mask_path = out_ply = None

    try:
        image_path = b64_to_file(image_b64, ".png")
        mask_path = b64_to_file(mask_b64, ".png")
        out_ply = tempfile.mktemp(suffix=".ply")

        image = load_image(image_path)
        mask = load_single_mask(mask_path)

        with torch.no_grad():
            output = MODEL(image, mask, seed=seed)

        output["gs"].save_ply(out_ply)
        return file_to_b64(out_ply)

    finally:
        for p in [image_path, mask_path, out_ply]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# -------------------------------------------------
# RunPod handler (API entrypoint)
# -------------------------------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
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

    try:
        ply_b64 = run_sam3d(image_b64, mask_b64, seed)
        return {
            "success": True,
            "plyBase64": ply_b64
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


runpod.serverless.start({"handler": handler})
