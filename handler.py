# handler.py
#
# RunPod Serverless handler for SAM-3D-Objects:
# - Ensures SAM3D checkpoints are downloaded at runtime (Option A) using HF_TOKEN
# - Accepts base64 image+mask
# - Runs generate_3d_subprocess.py (isolated process for spconv stability)
# - Returns outputs (PLY base64, optional GIF base64, optional asset URLs)

import base64
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import runpod

# ----------------------------
# Paths (adjust only if you change your container layout)
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parent  # /workspace/sam-3d-objects
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints" / "hf"
PIPELINE_YAML = CHECKPOINTS_DIR / "pipeline.yaml"
HF_DOWNLOAD_DIR = REPO_ROOT / "checkpoints" / "hf-download"
ASSETS_DIR = REPO_ROOT / "assets"
SUBPROCESS_SCRIPT = REPO_ROOT / "generate_3d_subprocess.py"

LOCK_FILE = REPO_ROOT / ".hf_checkpoints.lock"


# ----------------------------
# Small helpers
# ----------------------------
def _b64_to_temp_file(b64_str: str, suffix: str) -> str:
    data = base64.b64decode(b64_str)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _extract_between(text: str, start: str, end: str) -> Optional[str]:
    if start not in text or end not in text:
        return None
    s = text.split(start, 1)[1]
    return s.split(end, 1)[0].strip()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _acquire_lock(lock_path: Path, timeout_s: int = 1800) -> None:
    """
    Inter-process file lock to prevent multiple concurrent downloads.
    Linux-only (RunPod workers are Linux).
    """
    import fcntl

    _ensure_dir(lock_path.parent)
    f = open(lock_path, "a+")
    start = time.time()
    while True:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Keep handle open by storing on module
            globals()["_LOCK_HANDLE"] = f
            return
        except BlockingIOError:
            if time.time() - start > timeout_s:
                raise RuntimeError("Timeout waiting for checkpoint download lock.")
            time.sleep(0.25)


def _release_lock() -> None:
    import fcntl

    f = globals().get("_LOCK_HANDLE")
    if f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                f.close()
            except Exception:
                pass
        globals()["_LOCK_HANDLE"] = None


def ensure_checkpoints() -> None:
    """
    Option A:
    - If checkpoints are missing, download them using HF_TOKEN at runtime.
    - Requires that the model access has been granted to your HF account.
    """
    # Fast path: already present
    if PIPELINE_YAML.exists():
        return

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "SAM3D checkpoints are missing and HF_TOKEN (or HUGGINGFACE_TOKEN) is not set."
        )

    _acquire_lock(LOCK_FILE)
    try:
        # Re-check after acquiring lock
        if PIPELINE_YAML.exists():
            return

        _ensure_dir(ASSETS_DIR)
        _ensure_dir(HF_DOWNLOAD_DIR)

        # We rely on huggingface-cli being installed in the env.
        # Provide token via env so it authenticates.
        env = dict(os.environ)
        env["HF_TOKEN"] = hf_token
        # Optional cache dir to reduce repeated downloads (can be on a network volume if you mount one)
        env.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))

        cmd = [
            "huggingface-cli",
            "download",
            "--repo-type",
            "model",
            "--local-dir",
            str(HF_DOWNLOAD_DIR),
            "--max-workers",
            "1",
            "facebook/sam-3d-objects",
        ]

        print("[handler] Downloading SAM3D checkpoints (gated HF repo) ...")
        subprocess.run(cmd, check=True, env=env)

        # Expected layout from HF: hf-download/checkpoints/*
        src = HF_DOWNLOAD_DIR / "checkpoints"
        dst = CHECKPOINTS_DIR

        if not src.exists():
            raise RuntimeError(
                f"Download completed but expected '{src}' not found. "
                f"Contents: {list(HF_DOWNLOAD_DIR.glob('*'))}"
            )

        # Move into expected path
        _ensure_dir(dst.parent)
        if dst.exists():
            # If something partial exists, remove it to avoid mixing
            for p in dst.glob("*"):
                try:
                    if p.is_file():
                        p.unlink()
                except Exception:
                    pass
        else:
            dst.mkdir(parents=True, exist_ok=True)

        # Move all files from src -> dst
        for item in src.iterdir():
            target = dst / item.name
            if target.exists():
                if target.is_file():
                    target.unlink()
            item.rename(target)

        # Cleanup download dir
        try:
            for p in HF_DOWNLOAD_DIR.glob("*"):
                if p.is_dir():
                    # best effort cleanup
                    pass
            # Remove the whole hf-download folder
            import shutil

            shutil.rmtree(HF_DOWNLOAD_DIR, ignore_errors=True)
        except Exception:
            pass

        if not PIPELINE_YAML.exists():
            raise RuntimeError(
                f"Checkpoints moved but pipeline.yaml not found at {PIPELINE_YAML}"
            )

        print(f"[handler] ✓ Checkpoints ready at: {CHECKPOINTS_DIR}")

    finally:
        _release_lock()


def run_generation(image_b64: str, mask_b64: str, seed: int) -> Dict[str, Any]:
    _ensure_dir(ASSETS_DIR)

    image_path = mask_path = out_ply_path = None
    try:
        image_path = _b64_to_temp_file(image_b64, ".png")
        mask_path = _b64_to_temp_file(mask_b64, ".png")

        out_ply_path = os.path.join(
            tempfile.gettempdir(), f"out_{os.getpid()}_{int(time.time())}.ply"
        )

        if not SUBPROCESS_SCRIPT.exists():
            raise RuntimeError(f"Missing subprocess script at: {SUBPROCESS_SCRIPT}")

        cmd = [
            sys.executable,
            str(SUBPROCESS_SCRIPT),
            image_path,
            mask_path,
            str(seed),
            out_ply_path,
            str(ASSETS_DIR),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min
            cwd=str(REPO_ROOT),
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if result.returncode != 0:
            # Return helpful tail
            return {
                "success": False,
                "error": "3D generation subprocess failed",
                "stdout_tail": stdout[-6000:],
                "stderr_tail": stderr[-6000:],
            }

        gif_b64_out = _extract_between(stdout, "GIF_DATA_START", "GIF_DATA_END")
        mesh_url = _extract_between(stdout, "MESH_URL_START", "MESH_URL_END")
        ply_url = _extract_between(stdout, "PLY_URL_START", "PLY_URL_END")

        ply_b64_out = None
        ply_size = None
        if os.path.exists(out_ply_path):
            with open(out_ply_path, "rb") as f:
                ply_bytes = f.read()
            ply_b64_out = base64.b64encode(ply_bytes).decode("utf-8")
            ply_size = len(ply_bytes)

        return {
            "success": True,
            "ply_b64": ply_b64_out,  # may be None if only GIF exists
            "ply_size_bytes": ply_size,
            "gif_b64": gif_b64_out,  # may be None
            "mesh_url": mesh_url,    # may be None
            "ply_url": ply_url,      # may be None
            "stdout_tail": stdout[-2000:],  # helpful for debugging, trim if you want
        }

    finally:
        for p in [image_path, mask_path, out_ply_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected input payload (RunPod job['input']):

    {
      "imageBase64": "<base64 PNG/JPG>",
      "maskBase64":  "<base64 PNG mask>",
      "seed": 42,
      "return": ["ply_b64", "gif_b64", "mesh_url", "ply_url"]   # optional
    }

    Notes:
    - If you want smaller responses, request only URLs (mesh_url/ply_url) and omit base64.
    - Ensure HF_TOKEN is set in the worker environment for gated checkpoints.
    """
    inp = job.get("input") or {}

    image_b64 = inp.get("imageBase64") or inp.get("image")
    mask_b64 = inp.get("maskBase64") or inp.get("mask")
    seed = int(inp.get("seed", 42))

    if not image_b64 or not mask_b64:
        return {
            "success": False,
            "error": "Missing 'imageBase64' and/or 'maskBase64' in job.input",
        }

    # Ensure checkpoints (Option A)
    try:
        ensure_checkpoints()
    except Exception as e:
        return {
            "success": False,
            "error": f"Checkpoint bootstrap failed: {str(e)}",
        }

    # Run generation
    out = run_generation(image_b64, mask_b64, seed)

    # Optional: allow caller to request a subset of outputs
    want = inp.get("return")
    if isinstance(want, list) and out.get("success"):
        filtered = {"success": True}
        for k in want:
            if k in out:
                filtered[k] = out[k]
        # Always keep these if present
        for k in ["ply_size_bytes"]:
            if k in out:
                filtered[k] = out[k]
        return filtered

    return out


runpod.serverless.start({"handler": handler})
