# handler.py
#
# RunPod Serverless handler for SAM-3D-Objects (Option A):
# - Downloads gated checkpoints at runtime using HF_TOKEN
# - Uses RunPod Network Volume (auto-detects sam3d* mount under /runpod-volume)
# - Accepts base64 image+mask
# - Runs generate_3d_subprocess.py in a separate process
# - Fixes KeyError: CONDA_PREFIX by injecting CONDA_PREFIX/CUDA_HOME into subprocess env
# - Returns schema compatible with your client: plyBase64, gifBase64, meshUrl, plyUrl
#
# Notes on warnings:
# - open3d missing => mesh simplification disabled (OK for PLY output)
# - trimesh missing => only needed for GLB conversion/export (OK for PLY output)

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import runpod


# ----------------------------
# Paths (auto-detect RunPod volume sam3d*)
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parent  # /workspace/sam-3d-objects
RUNPOD_VOLUME_ROOT = Path("/runpod-volume")


def _detect_volume(prefix: str = "sam3d") -> Path:
    """
    Detect a mounted RunPod volume directory under /runpod-volume that starts with `prefix`.

    Best practice (optional): set SAM3D_VOLUME_DIR to the exact full path to avoid ambiguity.
      e.g. SAM3D_VOLUME_DIR=/runpod-volume/sam3d-eu-cz-1
    """
    override = os.environ.get("SAM3D_VOLUME_DIR")
    if override:
        return Path(override)

    if not RUNPOD_VOLUME_ROOT.exists():
        raise RuntimeError("'/runpod-volume' does not exist. Volume is not mounted.")

    candidates = [
        p for p in RUNPOD_VOLUME_ROOT.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    ]

    if not candidates:
        existing = [p.name for p in RUNPOD_VOLUME_ROOT.iterdir()]
        raise RuntimeError(
            f"No volume starting with '{prefix}' found under /runpod-volume. Found: {existing}"
        )

    return sorted(candidates, key=lambda p: p.name)[0]


VOLUME_ROOT = _detect_volume(prefix=os.environ.get("SAM3D_VOLUME_PREFIX", "sam3d"))

CHECKPOINTS_DIR = VOLUME_ROOT / "checkpoints" / "hf"
PIPELINE_YAML = CHECKPOINTS_DIR / "pipeline.yaml"

HF_DOWNLOAD_DIR = VOLUME_ROOT / "checkpoints" / "hf-download"
HF_CACHE_DIR = VOLUME_ROOT / "hf_cache"

ASSETS_DIR = REPO_ROOT / "assets"
SUBPROCESS_SCRIPT = REPO_ROOT / "generate_3d_subprocess.py"

LOCK_FILE = VOLUME_ROOT / ".hf_checkpoints.lock"


# ----------------------------
# Helpers
# ----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def _acquire_lock(lock_path: Path, timeout_s: int = 1800) -> None:
    import fcntl

    _ensure_dir(lock_path.parent)
    f = open(lock_path, "a+")
    start = time.time()
    while True:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
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


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "SAM3D checkpoints are missing and HF_TOKEN (or HUGGINGFACE_TOKEN) is not set."
        )
    return token


def _preflight_storage() -> None:
    _ensure_dir(HF_CACHE_DIR)
    _ensure_dir(HF_DOWNLOAD_DIR.parent)
    _ensure_dir(CHECKPOINTS_DIR.parent)

    test_file = VOLUME_ROOT / ".write_test"
    try:
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
    except Exception as e:
        raise RuntimeError(f"Volume not writable at {VOLUME_ROOT}: {e}") from e


def _download_with_cli(token: str) -> None:
    env = dict(os.environ)
    env["HF_TOKEN"] = token
    env["HF_HOME"] = str(HF_CACHE_DIR)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    _ensure_dir(HF_DOWNLOAD_DIR)

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

    print(f"[handler] Using volume: {VOLUME_ROOT}", flush=True)
    print("[handler] Downloading checkpoints via huggingface-cli ...", flush=True)
    subprocess.run(cmd, check=True, env=env, timeout=30 * 60)


def _download_with_python(token: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "huggingface-cli not found and huggingface_hub not available. "
            "Install: pip install 'huggingface-hub[cli]<1.0'"
        ) from e

    _ensure_dir(HF_CACHE_DIR)
    _ensure_dir(HF_DOWNLOAD_DIR)

    print(f"[handler] Using volume: {VOLUME_ROOT}", flush=True)
    print("[handler] Downloading checkpoints via huggingface_hub.snapshot_download ...", flush=True)

    local_repo_dir = snapshot_download(
        repo_id="facebook/sam-3d-objects",
        repo_type="model",
        token=token,
        cache_dir=str(HF_CACHE_DIR),
        local_dir=str(HF_DOWNLOAD_DIR),
        local_dir_use_symlinks=False,
        max_workers=1,
    )

    if not Path(local_repo_dir).exists():
        raise RuntimeError("snapshot_download did not produce a valid local directory.")


def _move_download_into_place() -> None:
    src = HF_DOWNLOAD_DIR / "checkpoints"
    dst = CHECKPOINTS_DIR

    if not src.exists():
        raise RuntimeError(
            f"Download completed but expected '{src}' not found. "
            f"Contents: {list(HF_DOWNLOAD_DIR.glob('*'))}"
        )

    _ensure_dir(dst)

    # clean destination
    for p in dst.glob("*"):
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            if target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target, ignore_errors=True)
        item.rename(target)

    shutil.rmtree(HF_DOWNLOAD_DIR, ignore_errors=True)

    if not PIPELINE_YAML.exists():
        raise RuntimeError(f"pipeline.yaml not found at {PIPELINE_YAML} after move")


def ensure_checkpoints() -> None:
    if PIPELINE_YAML.exists():
        return

    _preflight_storage()
    token = _hf_token()

    _acquire_lock(LOCK_FILE)
    try:
        if PIPELINE_YAML.exists():
            return

        _ensure_dir(ASSETS_DIR)

        t0 = time.time()
        try:
            _download_with_cli(token)
        except FileNotFoundError:
            print("[handler] huggingface-cli not found; falling back to python download...", flush=True)
            _download_with_python(token)

        _move_download_into_place()
        print(f"[handler] ✓ Checkpoints ready in {time.time()-t0:.1f}s at {CHECKPOINTS_DIR}", flush=True)

    finally:
        _release_lock()


def _build_subprocess_env() -> Dict[str, str]:
    """
    Serverless does not 'conda activate', so notebook/inference.py may crash with KeyError: CONDA_PREFIX.
    We inject CONDA_PREFIX and CUDA_HOME for the subprocess.
    """
    env = os.environ.copy()

    conda_prefix = env.get("CONDA_PREFIX")
    if not conda_prefix:
        exe = Path(sys.executable).resolve()
        # /opt/conda/envs/sam3d-objects/bin/python -> /opt/conda/envs/sam3d-objects
        conda_prefix = str(exe.parent.parent)
        env["CONDA_PREFIX"] = conda_prefix

    env.setdefault("CUDA_HOME", conda_prefix)

    # Keep HF cache on the volume for any hub access during runtime
    env.setdefault("HF_HOME", str(HF_CACHE_DIR))

    return env


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

        env = _build_subprocess_env()

        print("[handler] Starting subprocess generation...", flush=True)
        t0 = time.time()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(REPO_ROOT),
            env=env,  # <<< critical fix for CONDA_PREFIX
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        print(f"[handler] Subprocess finished in {time.time()-t0:.1f}s (rc={result.returncode})", flush=True)

        if result.returncode != 0:
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
        if os.path.exists(out_ply_path):
            with open(out_ply_path, "rb") as f:
                ply_b64_out = base64.b64encode(f.read()).decode("utf-8")

        return {
            "success": True,
            "ply_b64": ply_b64_out,
            "gif_b64": gif_b64_out,
            "mesh_url": mesh_url,
            "ply_url": ply_url,
            "stdout_tail": stdout[-2000:],
        }

    finally:
        for p in [image_path, mask_path, out_ply_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input") or {}

    image_b64 = inp.get("imageBase64") or inp.get("image")
    mask_b64 = inp.get("maskBase64") or inp.get("mask")

    opts = inp.get("options") or {}
    seed = int(inp.get("seed", opts.get("seed", 42)))

    if not image_b64 or not mask_b64:
        return {"success": False, "error": "Missing imageBase64/maskBase64 in job.input"}

    try:
        ensure_checkpoints()
    except Exception as e:
        return {"success": False, "error": f"Checkpoint bootstrap failed: {str(e)}"}

    out = run_generation(image_b64, mask_b64, seed)

    if out.get("success"):
        return {
            "success": True,
            "plyBase64": out.get("ply_b64"),
            "gifBase64": out.get("gif_b64"),
            "meshUrl": out.get("mesh_url"),
            "plyUrl": out.get("ply_url"),
        }

    return out


runpod.serverless.start({"handler": handler})
