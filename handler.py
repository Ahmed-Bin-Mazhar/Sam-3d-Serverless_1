# handler.py — RunPod Serverless SAM-3D Objects
# - Robust cold-start checkpoint download to /runpod-volume
# - Warm model load (once per worker)
# - Strict input validation
# - Safe temp-file handling
# - Keeps the process alive even if RunPod listener returns (prevents auto-stop loops)

import base64
import os
import tempfile
import time
import traceback
from typing import Dict, Any

import runpod
import torch
from huggingface_hub import snapshot_download
print("RUNPOD QUEUE WORKER BOOTING", flush=True)

# -------------------------------------------------
# Paths / env
# -------------------------------------------------
VOLUME_ROOT = "/runpod-volume/sam3d"
CHECKPOINT_DIR = f"{VOLUME_ROOT}/checkpoints/hf"
PIPELINE_YAML = f"{CHECKPOINT_DIR}/pipeline.yaml"
HF_CACHE_DIR = f"{VOLUME_ROOT}/hf_cache"

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_HOME = os.environ.get("HF_HOME", HF_CACHE_DIR)

# (Optional) reduce CPU thread oversubscription
# Uncomment if you see high CPU usage:
# os.environ.setdefault("OMP_NUM_THREADS", "4")
# os.environ.setdefault("MKL_NUM_THREADS", "4")
# torch.set_num_threads(4)

# -------------------------------------------------
# Import SAM-3D inference code
# -------------------------------------------------
# The repo is cloned to /app/sam-3d-objects in the Dockerfile.
# The inference module lives under /app/sam-3d-objects/notebook.
import sys
sys.path.insert(0, "/app/sam-3d-objects/notebook")

from inference import Inference, load_image, load_single_mask  # noqa: E402


# -------------------------------------------------
# Checkpoints (cold start)
# -------------------------------------------------
def ensure_checkpoints() -> None:
    """
    Download HF model snapshot to CHECKPOINT_DIR on the network volume.
    If PIPELINE_YAML exists, assume checkpoints are present.
    """
    if os.path.exists(PIPELINE_YAML):
        print(f"[SAM3D] Checkpoints already present: {PIPELINE_YAML}", flush=True)
        return

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is required to download checkpoints")

    print("[SAM3D] Downloading checkpoints from Hugging Face...", flush=True)

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

    if not os.path.exists(PIPELINE_YAML):
        raise RuntimeError(f"Checkpoint download finished but {PIPELINE_YAML} not found")

    print("[SAM3D] Checkpoints downloaded and verified.", flush=True)


# -------------------------------------------------
# Model (warm start)
# -------------------------------------------------
print("[SAM3D] Worker boot: starting init...", flush=True)
ensure_checkpoints()

print("[SAM3D] Loading model...", flush=True)
MODEL = Inference(PIPELINE_YAML, compile=False)
MODEL.eval()

if torch.cuda.is_available():
    print(f"[SAM3D] CUDA available. Device: {torch.cuda.get_device_name(0)}", flush=True)
else:
    print("[SAM3D] WARNING: CUDA not available, running on CPU (will be very slow).", flush=True)

print("[SAM3D] Model ready.", flush=True)


# -------------------------------------------------
# Helpers: base64 <-> temp files
# -------------------------------------------------
def b64_to_file(b64: str, suffix: str) -> str:
    try:
        data = base64.b64decode(b64, validate=True)
    except Exception:
        # Some clients send base64 without padding; try forgiving decode
        data = base64.b64decode(b64)

    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# -------------------------------------------------
# Inference core
# -------------------------------------------------
def run_sam3d(image_b64: str, mask_b64: str, seed: int) -> str:
    image_path = mask_path = out_ply = None
    try:
        image_path = b64_to_file(image_b64, ".png")
        mask_path = b64_to_file(mask_b64, ".png")
        out_ply = tempfile.mktemp(suffix=".ply")

        image = load_image(image_path)
        mask = load_single_mask(mask_path)

        if image is None or mask is None:
            raise ValueError("Invalid image or mask (failed to load PNG)")

        with torch.no_grad():
            output = MODEL(image, mask, seed=seed)

        if "gs" not in output:
            raise RuntimeError("Model output missing 'gs' object; cannot save ply")

        output["gs"].save_ply(out_ply)
        return file_to_b64(out_ply)

    finally:
        for p in (image_path, mask_path, out_ply):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# -------------------------------------------------
# RunPod handler
# -------------------------------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input", {}) or {}

        image_b64 = inp.get("imageBase64")
        mask_b64 = inp.get("maskBase64")
        options = inp.get("options", {}) or {}

        seed = int(options.get("seed", 42))
        outputs = options.get("output", ["ply"])

        if not image_b64 or not mask_b64:
            return {"success": False, "error": "imageBase64 and maskBase64 are required"}

        # Allow outputs to be a string or list
        if isinstance(outputs, str):
            outputs = [outputs]

        if "ply" not in outputs:
            return {"success": False, "error": "Only 'ply' output is supported"}

        ply_b64 = run_sam3d(image_b64, mask_b64, seed)
        return {"success": True, "plyBase64": ply_b64}

    except torch.cuda.OutOfMemoryError:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return {"success": False, "error": "CUDA out of memory"}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# -------------------------------------------------
# Start RunPod worker (and keep process alive)
# -------------------------------------------------
def main():
    print("[BOOT] Starting RunPod queue worker", flush=True)

    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": False
    })

    # HARD BLOCK — RunPod sometimes kills otherwise
    print("[BOOT] Worker started, entering hard block", flush=True)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()

