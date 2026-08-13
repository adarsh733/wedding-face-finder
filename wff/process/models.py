"""Fetch and cache the ONNX weights.

LANDMINE #1 (docs/ARCHITECTURE.md): the InsightFace *code* is open, but these
pretrained *weights* are licensed for non-commercial research use, as far as we
currently understand it. Fine for building, testing and demoing. NOT fine the
day someone pays us.

That is why nothing outside this file names them. Detector and Embedder are
interfaces; swapping in a permissively-licensed model is one file.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests

BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)

# name -> (member inside the zip, what it does)
REQUIRED = {
    "det_10g.onnx": "SCRFD 10GF face detector, 5 landmarks",
    "w600k_r50.onnx": "ArcFace ResNet50, 512-d embeddings",
}


def model_path(model_dir: str, name: str) -> Path:
    return Path(model_dir).expanduser() / "buffalo_l" / name


def ensure_models(model_dir: str, quiet: bool = False) -> dict[str, Path]:
    """Download buffalo_l once and keep the two files we actually use."""
    target_dir = Path(model_dir).expanduser() / "buffalo_l"
    target_dir.mkdir(parents=True, exist_ok=True)

    paths = {name: target_dir / name for name in REQUIRED}
    missing = [name for name, path in paths.items() if not path.exists()]
    if not missing:
        return paths

    if not quiet:
        print(f"Downloading face models ({', '.join(missing)}) -- one time, ~275 MB...")

    try:
        response = requests.get(BUFFALO_L_URL, stream=True, timeout=(10, 600))
        response.raise_for_status()
        blob = io.BytesIO()
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1 << 20):
            blob.write(chunk)
            downloaded += len(chunk)
            if not quiet and downloaded % (20 << 20) < (1 << 20):
                print(f"  {downloaded / (1 << 20):.0f} MB")
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not download face models from {BUFFALO_L_URL} ({exc}).\n"
            f"Download it manually and unzip det_10g.onnx and w600k_r50.onnx into:\n"
            f"  {target_dir}"
        ) from exc

    blob.seek(0)
    with zipfile.ZipFile(blob) as archive:
        members = {Path(n).name: n for n in archive.namelist()}
        for name in REQUIRED:
            if name not in members:
                raise RuntimeError(
                    f"{name} not found in the model archive. Contents: {list(members)}"
                )
            with archive.open(members[name]) as src:
                paths[name].write_bytes(src.read())

    if not quiet:
        print(f"Models ready in {target_dir}")
    return paths


def make_session(path: Path, num_threads: int = 0):
    """One place that knows about onnxruntime, so the GPU swap is one line."""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    if num_threads > 0:
        options.intra_op_num_threads = num_threads
    # CPUExecutionProvider today. On a rented GPU this becomes
    # ['CUDAExecutionProvider', 'CPUExecutionProvider'] and nothing else changes.
    return onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
