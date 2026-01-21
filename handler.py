import base64
import os
import tempfile
import subprocess
import sys
from typing import Dict, Any

import runpod

# If you keep your subprocess next to handler.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBPROCESS_SCRIPT = os.path.join(SCRIPT_DIR, "generate_3d_subprocess.py")

# RunPod best practice: do heavy imports lazily / in subprocess (you already do)
# So the handler stays lightweight and avoids spconv state issues.
# generate_3d_subprocess.py already sets critical env-vars before torch import. :contentReference[oaicite:2]{index=2}


def _b64_to_file(b64_str: str, suffix: str) -> str:
    raw = base64.b64decode(b64_str)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless handler
    Input:  job["input"]["imageBase64"], job["input"]["maskBase64"], job["input"]["seed"]
    Output: base64 PLY (+ GIF/GLB URLs if produced by subprocess)
    """
    inp = job.get("input", {}) or {}

    image_b64 = inp.get("imageBase64") or inp.get("image")
    mask_b64 = inp.get("maskBase64") or inp.get("mask")
    seed = int(inp.get("seed", 42))

    if not image_b64 or not mask_b64:
        return {
            "error": "Missing imageBase64/maskBase64 (or image/mask).",
            "expected": {
                "imageBase64": "base64-encoded PNG/JPG bytes",
                "maskBase64": "base64-encoded PNG bytes (binary mask)",
                "seed": 42
            }
        }

    image_path = mask_path = ply_out_path = None

    try:
        image_path = _b64_to_file(image_b64, suffix=".png")
        mask_path = _b64_to_file(mask_b64, suffix=".png")

        # output ply path
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            ply_out_path = tmp.name

        # assets directory inside container (subprocess will write URLs relative to /assets)
        assets_dir = os.path.join(SCRIPT_DIR, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        # Run the subprocess (SAM3D inference)
        # It prints markers: GIF_DATA_START/END, MESH_URL_START/END, PLY_URL_START/END :contentReference[oaicite:3]{index=3}
        result = subprocess.run(
            [sys.executable, SUBPROCESS_SCRIPT, image_path, mask_path, str(seed), ply_out_path, assets_dir],
            capture_output=True,
            text=True,
            timeout=900,
        )

        if result.returncode != 0:
            return {
                "error": "SAM3D subprocess failed",
                "stderr": result.stderr[-4000:],
                "stdout": result.stdout[-4000:],
            }

        # Always return PLY as base64 (primary artifact)
        if not os.path.exists(ply_out_path):
            return {"error": "PLY output missing after subprocess completed."}

        with open(ply_out_path, "rb") as f:
            ply_bytes = f.read()

        ply_b64 = base64.b64encode(ply_bytes).decode("utf-8")

        # Extract optional markers from stdout
        stdout = result.stdout or ""

        def extract_between(start: str, end: str):
            if start in stdout and end in stdout:
                a = stdout.find(start) + len(start)
                b = stdout.find(end)
                return stdout[a:b].strip()
            return None

        gif_b64 = extract_between("GIF_DATA_START", "GIF_DATA_END")
        mesh_url = extract_between("MESH_URL_START", "MESH_URL_END")
        ply_url = extract_between("PLY_URL_START", "PLY_URL_END")

        return {
            "success": True,
            "seed": seed,
            "ply_b64": ply_b64,
            "ply_size_bytes": len(ply_bytes),
            "gif_b64": gif_b64,       # optional
            "mesh_url": mesh_url,     # optional (GLB saved in assets/)
            "ply_url": ply_url,       # optional (PLY saved in assets/)
            "debug_tail": stdout[-1500:],  # helpful during bring-up; remove once stable
        }

    except subprocess.TimeoutExpired:
        return {"error": "Timed out running SAM3D subprocess (over 15 minutes)."}
    finally:
        for p in [image_path, mask_path, ply_out_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


runpod.serverless.start({"handler": handler})
