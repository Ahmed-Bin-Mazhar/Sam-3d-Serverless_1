import os
import io
import json
import base64
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import runpod


# ----------------------------
# Env fixes for serverless
# ----------------------------
def _ensure_env_for_inference() -> None:
    """
    RunPod serverless workers often don't set CONDA_PREFIX even if running inside a conda env.
    The notebook/inference.py expects it at import-time (it does os.environ["CONDA_PREFIX"]).
    We also ensure CUDA_HOME has a sane default for CUDA base images.
    """
    import sys

    os.environ.setdefault("CONDA_PREFIX", sys.prefix)
    os.environ.setdefault("CUDA_HOME", os.environ.get("CUDA_HOME") or "/usr/local/cuda")


# --- HF download at runtime (no interactive login) ---
def ensure_checkpoints(tag: str = "hf") -> Path:
    """
    Ensures checkpoints/<tag>/pipeline.yaml exists.
    If missing, downloads from HF model repo and places into checkpoints/<tag>.
    """
    ckpt_dir = Path("/workspace/sam-3d-objects/checkpoints")
    target = ckpt_dir / tag / "pipeline.yaml"
    if target.exists():
        return target

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Missing HF_TOKEN (or HUGGINGFACE_TOKEN). "
            "Set it as a RunPod Secret so the worker can download checkpoints."
        )

    from huggingface_hub import snapshot_download

    tmp = ckpt_dir / f"{tag}-download"
    if tmp.exists():
        shutil.rmtree(tmp)

    tmp.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="facebook/sam-3d-objects",
        repo_type="model",
        local_dir=str(tmp),
        local_dir_use_symlinks=False,
        token=hf_token,
        max_workers=1,
    )

    # Move tmp/checkpoints -> checkpoints/<tag>
    downloaded_checkpoints = tmp / "checkpoints"
    if not downloaded_checkpoints.exists():
        raise RuntimeError(
            f"HF download finished but '{downloaded_checkpoints}' not found. "
            f"Contents: {list(tmp.iterdir())}"
        )

    dest = ckpt_dir / tag
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(downloaded_checkpoints), str(dest))

    shutil.rmtree(tmp, ignore_errors=True)

    if not target.exists():
        raise RuntimeError(f"Expected {target} after download, but it does not exist.")

    return target


# --- Lazy-load inference (keep warm on a worker) ---
_INFERENCE = None


def get_inference(tag: str = "hf", compile_model: bool = False):
    global _INFERENCE
    if _INFERENCE is not None:
        return _INFERENCE

    pipeline_yaml = ensure_checkpoints(tag=tag)

    import sys
    sys.path.append("/workspace/sam-3d-objects/notebook")

    # ✅ critical env fixes before importing inference
    _ensure_env_for_inference()

    from inference import Inference  # type: ignore

    _INFERENCE = Inference(str(pipeline_yaml), compile=bool(compile_model))
    return _INFERENCE


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _b64_to_bytes(b64_str: str) -> bytes:
    # Allow "data:image/png;base64,...." as well
    if "," in b64_str and "base64" in b64_str[:80].lower():
        b64_str = b64_str.split(",", 1)[1]
    return base64.b64decode(b64_str)


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod job format:
    {
      "id": "...",
      "input": {...}
    }
    """
    job_input = job.get("input", {}) or {}

    # Support BOTH:
    #  - top-level keys: seed, output_format
    #  - client "options": {"seed": ..., "output": ["ply", ...]}
    opts = job_input.get("options") or {}

    tag = str(job_input.get("tag", "hf"))
    seed = int(job_input.get("seed", opts.get("seed", 42)))
    compile_model = bool(job_input.get("compile", False))

    # Output format handling:
    # - prefer options.output[0] if provided, else output_format
    output_format = None
    out_list = opts.get("output")
    if isinstance(out_list, list) and len(out_list) > 0:
        output_format = str(out_list[0]).lower()
    else:
        output_format = str(job_input.get("output_format", "ply")).lower()

    if output_format not in ("ply",):
        return {"error": f"Unsupported output_format: {output_format}. Use 'ply'."}

    # Prepare temp workspace
    tmpdir = Path(tempfile.mkdtemp(prefix="sam3d_"))
    img_path = tmpdir / "image.png"
    mask_path = tmpdir / "mask.png"

    # Ingest inputs
    image_b64 = job_input.get("image_base64")
    mask_b64 = job_input.get("mask_base64")
    image_url = job_input.get("image_url")
    mask_url = job_input.get("mask_url")

    try:
        if image_b64 and mask_b64:
            _write_bytes(img_path, _b64_to_bytes(str(image_b64)))
            _write_bytes(mask_path, _b64_to_bytes(str(mask_b64)))
        elif image_url and mask_url:
            import urllib.request

            _write_bytes(img_path, urllib.request.urlopen(str(image_url)).read())
            _write_bytes(mask_path, urllib.request.urlopen(str(mask_url)).read())
        else:
            return {
                "error": "Provide either (image_base64 + mask_base64) OR (image_url + mask_url)."
            }

        # Load image via notebook utilities
        import sys
        sys.path.append("/workspace/sam-3d-objects/notebook")

        # ✅ critical env fixes before importing inference helpers
        _ensure_env_for_inference()

        from inference import load_image  # type: ignore

        image = load_image(str(img_path))

        # Load mask (single mask file)
        from PIL import Image
        import numpy as np

        mask_img = Image.open(mask_path).convert("L")
        mask = (np.array(mask_img) > 0).astype(np.uint8)

        # Run inference
        inf = get_inference(tag=tag, compile_model=compile_model)
        out = inf(image, mask, seed=seed)

        # Export Gaussian splat as PLY
        ply_file = tmpdir / "splat.ply"
        out["gs"].save_ply(str(ply_file))

        ply_b64 = base64.b64encode(ply_file.read_bytes()).decode("utf-8")

        # GPU info (optional)
        import torch

        gpu_info = {
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

        # Return both key styles to be robust with different clients
        return {
            "output_format": "ply",
            "plyBase64": ply_b64,
            "ply_base64": ply_b64,
            "seed": seed,
            "tag": tag,
            "gpu": gpu_info,
        }

    finally:
        # Clean temp files
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


# Start the Serverless worker (RunPod standard)
runpod.serverless.start({"handler": handler})
