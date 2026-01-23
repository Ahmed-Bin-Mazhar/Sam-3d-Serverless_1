import base64
import os
import sys
import shutil
import tempfile
import traceback
from typing import Dict, Any

import runpod
import torch
from huggingface_hub import snapshot_download

# -------------------------------------------------
# 1. ENV & PATH SETUP
# -------------------------------------------------
# satisfy inference.py's need for CONDA_PREFIX to find CUDA
os.environ.setdefault("CONDA_PREFIX", "/usr/local/cuda")
os.environ["LIDRA_SKIP_INIT"] = "true"

# Add repo to path so imports like 'sam3d_objects' work
sys.path.insert(0, "/app/sam-3d-objects")
sys.path.insert(0, "/app/sam-3d-objects/notebook")

from inference import Inference, load_image, load_single_mask

# -------------------------------------------------
# 2. STORAGE & CHECKPOINTS
# -------------------------------------------------
VOLUME_ROOT = "/runpod-volume/sam3d"
CHECKPOINT_DIR = f"{VOLUME_ROOT}/checkpoints/hf"
PIPELINE_YAML = f"{CHECKPOINT_DIR}/pipeline.yaml"

def ensure_checkpoints():
    if not os.path.exists(PIPELINE_YAML):
        print("[SAM3D] Checkpoints missing. Downloading...", flush=True)
        snapshot_download(
            repo_id="facebook/sam-3d-objects",
            local_dir=CHECKPOINT_DIR,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN")
        )

# -------------------------------------------------
# 3. WARM-UP (LOAD MODEL INTO GPU)
# -------------------------------------------------
ensure_checkpoints()
MODEL = Inference(PIPELINE_YAML, compile=False)
if torch.cuda.is_available():
    MODEL._pipeline.to("cuda")

# -------------------------------------------------
# 4. BASE64 TO 3D LOGIC
# -------------------------------------------------
def run_inference(image_b64: str, mask_b64: str, seed: int):
    # Create unique temp workspace for this API call
    tmp_dir = tempfile.mkdtemp()
    try:
        # 1. Define internal paths
        img_path = os.path.join(tmp_dir, "input.png")
        mask_dir = os.path.join(tmp_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        mask_file = os.path.join(mask_dir, "0.png") # SAM3D looks for 0.png
        out_ply = os.path.join(tmp_dir, "result.ply")

        # 2. Clean Base64 (remove headers if present)
        def clean_b64(data):
            if isinstance(data, str) and "," in data:
                return data.split(",")[1]
            return data

        # 3. Save Base64 strings to temporary files
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(clean_b64(image_b64)))
        with open(mask_file, "wb") as f:
            f.write(base64.b64decode(clean_b64(mask_b64)))

        # 4. Execute SAM-3D logic
        image_np = load_image(img_path)
        mask_np = load_single_mask(mask_dir)

        with torch.no_grad():
            output = MODEL(image_np, mask_np, seed=seed)

        # 5. Export Result
        if "gs" in output:
            output["gs"].save_ply(out_ply)
            with open(out_ply, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise RuntimeError("Model did not return 'gs' object.")

    finally:
        # Delete all temp files to keep the container slim
        shutil.rmtree(tmp_dir)

# -------------------------------------------------
# 5. RUNPOD API HANDLER
# -------------------------------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Extract data from the API request
        payload = job.get("input", {})
        img_b64 = payload.get("imageBase64")
        msk_b64 = payload.get("maskBase64")
        seed = int(payload.get("seed", 42))

        if not img_b64 or not msk_b64:
            return {"status": "error", "message": "Missing image or mask base64"}

        # Run the 3D reconstruction
        ply_result = run_inference(img_b64, msk_b64, seed)

        return {
            "status": "success",
            "ply_base64": ply_result
        }

    except Exception as e:
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})