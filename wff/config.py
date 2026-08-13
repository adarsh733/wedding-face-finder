"""Every tunable number in one place, overridable by environment variable.

Nothing in the pipeline hard-codes a threshold. Stage 2 gets re-run dozens of
times a day with different knobs; they all live here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# THE MODEL VERSION STAMP
# ---------------------------------------------------------------------------
# docs/ARCHITECTURE.md, "The model version stamp — add it today".
#
# Embeddings made by two different models are mutually unreadable. Nothing
# crashes; guests just silently stop finding their photos, forever. Every face
# we write carries this string, and every search must filter on it.
#
# Bump this whenever the detector, the alignment, the recogniser, or the
# preprocessing changes in any way that moves the embedding.
EMBEDDING_MODEL_VERSION = "buffalo_l.w600k_r50/arcface-112/v1"


# ---------------------------------------------------------------------------
# THE SETTINGS FILE
# ---------------------------------------------------------------------------
# Everything above and below is a number with a sane default. The Google API
# key is the one setting with no default and no way to guess it, and it used to
# live ONLY in an environment variable -- which meant it had to be re-typed in
# every new terminal window and was invisible to the console started from a
# desktop shortcut. A run against a Drive link then died in under a second
# saying "no key", which is true but useless.
#
# So: it is written once to a file in the home folder, outside the repo, and
# read from there. A real environment variable still wins, so a one-off
# override for testing is still one line.
SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".wff", "settings.env")

# Also read from the project folder, for a checkout that wants its own key.
# Never committed -- see .gitignore.
_PROJECT_ENV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")

_settings_loaded = False


def _read_env_file(path: str) -> None:
    """KEY=value lines into os.environ. Anything already set wins."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        # A real environment variable is a deliberate override; never clobber it.
        if name and not os.environ.get(name):
            os.environ[name] = value


def load_settings(force: bool = False) -> None:
    global _settings_loaded
    if _settings_loaded and not force:
        return
    _settings_loaded = True
    _read_env_file(_PROJECT_ENV)   # the checkout's own key, if it has one
    _read_env_file(SETTINGS_FILE)  # otherwise the one on this machine


def google_api_key() -> str:
    """The key that reads PUBLIC Drive folders. Not a login, not an OAuth token."""
    load_settings()
    return os.environ.get("WFF_GOOGLE_API_KEY", "").strip()


def save_google_api_key(key: str) -> str:
    """Write the key to the settings file, keeping any other lines. Returns the path."""
    key = key.strip()
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    kept: list[str] = []
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
            kept = [
                line.rstrip("\n")
                for line in handle
                if not line.strip().startswith("WFF_GOOGLE_API_KEY=")
            ]
    except OSError:
        pass
    kept.append(f"WFF_GOOGLE_API_KEY={key}")
    with open(SETTINGS_FILE, "w", encoding="utf-8") as handle:
        handle.write("\n".join(kept).strip() + "\n")
    try:
        os.chmod(SETTINGS_FILE, 0o600)  # no-op on Windows; correct on the server
    except OSError:
        pass
    os.environ["WFF_GOOGLE_API_KEY"] = key  # live in this process immediately
    return SETTINGS_FILE


# Read at import, deliberately: the defaults below are evaluated when this
# module is first imported, so a settings file loaded any later would silently
# have no effect on any of the numbers.
load_settings()


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw not in (None, "") else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw not in (None, "") else default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class DetectConfig:
    """Detection — the half of the accuracy story that shrinking destroys."""

    # SCRFD internally works at a fixed square input. A 6000px photo letterboxed
    # into 640 is scaled by 0.107, so a 90px back-row face arrives as 9px and is
    # invisible. This is *why* tiling exists; see plan_tiles().
    det_size: int = _env_int("WFF_DET_SIZE", 640)
    det_thresh: float = _env_float("WFF_DET_THRESH", 0.5)

    # Tiling. We choose the grid so that every tile, once letterboxed into
    # det_size, is scaled by at least target_scale. 0.25 keeps a 90px face at
    # ~22px in the network's view, which SCRFD still resolves.
    tiling_enabled: bool = _env_bool("WFF_TILING", True)
    tile_target_scale: float = _env_float("WFF_TILE_TARGET_SCALE", 0.25)
    tile_overlap: float = _env_float("WFF_TILE_OVERLAP", 0.20)
    max_tiles_per_axis: int = _env_int("WFF_MAX_TILES_PER_AXIS", 4)
    # Below this, the full-image pass alone already sees faces at a usable
    # scale; tiling would only cost CPU.
    min_side_for_tiling: int = _env_int("WFF_MIN_SIDE_FOR_TILING", 1600)

    # Merging the full-image pass with the tile passes. Two boxes overlapping
    # by more than this are the same face found twice.
    nms_iou: float = _env_float("WFF_NMS_IOU", 0.40)
    # Also suppress when one box sits almost entirely INSIDE another, even if
    # IoU is below the line. Measured on a real photo: a degenerate sliver of a
    # face scored IoU 0.398 against the real detection of the same face and
    # survived, giving one person two faces in one photo. See nms().
    nms_containment: float = _env_float("WFF_NMS_CONTAINMENT", 0.80)


@dataclass
class QualityConfig:
    """The gate. Rejecting a face costs one photo; admitting a bad one costs trust."""

    # Face height measured in the ORIGINAL photo, not a resized copy.
    min_face_px: int = _env_int("WFF_MIN_FACE_PX", 50)
    # 50-80px faces are kept but second-class: they may JOIN a person, they may
    # never START one. docs/ARCHITECTURE.md step 3/4.
    second_class_face_px: int = _env_int("WFF_SECOND_CLASS_FACE_PX", 80)

    min_det_score: float = _env_float("WFF_MIN_DET_SCORE", 0.60)
    # A face box is roughly square-ish. Measured over 1,806 real detections:
    # median width/height 0.787, p1 of accepted 0.627, and only 2 boxes fell
    # below 0.35 -- both degenerate slivers, not faces. These bounds are wide
    # enough to never touch a real face and narrow enough to drop nonsense.
    min_aspect_ratio: float = _env_float("WFF_MIN_ASPECT", 0.40)
    max_aspect_ratio: float = _env_float("WFF_MAX_ASPECT", 2.50)
    # Variance of the Laplacian on the aligned 112x112 crop. Scale-dependent,
    # which is exactly why we measure it after alignment, when every face is
    # the same size.
    min_blur_var: float = _env_float("WFF_MIN_BLUR_VAR", 12.0)
    # Degrees from frontal. Beyond this the embedding degrades badly.
    max_yaw_deg: float = _env_float("WFF_MAX_YAW_DEG", 50.0)


@dataclass
class DerivativeConfig:
    """The only pixels we keep. Cut from the full-resolution original."""

    preview_px: int = _env_int("WFF_PREVIEW_PX", 2048)
    thumb_px: int = _env_int("WFF_THUMB_PX", 400)
    preview_quality: int = _env_int("WFF_PREVIEW_QUALITY", 82)
    thumb_quality: int = _env_int("WFF_THUMB_QUALITY", 78)
    # Face crops exist so a human can SEE what the machine found. They are the
    # accuracy meter, not a product artefact.
    save_face_crops: bool = _env_bool("WFF_SAVE_FACE_CROPS", True)
    face_crop_px: int = _env_int("WFF_FACE_CROP_PX", 112)


@dataclass
class ClusterConfig:
    """Stage 2. Cheap to re-run, which is the whole point of the stage split."""

    # PASS 1 — deliberately tight. Cosine distance. The bride WILL split into
    # several fragments here and that is the intended behaviour.
    pass1_threshold: float = _env_float("WFF_PASS1_THRESHOLD", 0.42)

    # PASS 2 — merge fragments, but only with evidence.
    pass2_threshold: float = _env_float("WFF_PASS2_THRESHOLD", 0.55)
    # The same-photo rule: two faces in one photo cannot be one person.
    enforce_same_photo_rule: bool = _env_bool("WFF_SAME_PHOTO_RULE", True)

    # A group smaller than this is not a person, it is a leftover. Leftovers
    # keep person_id = NULL and are never deleted — tier-2 search is what finds
    # the cousin who is in 3 photos.
    min_faces_per_person: int = _env_int("WFF_MIN_FACES_PER_PERSON", 3)

    # Second-class faces (50-80px) are assigned to an existing person only if
    # they are this close to its centroid. Stricter than pass 1 on purpose.
    second_class_assign_threshold: float = _env_float("WFF_SECOND_CLASS_ASSIGN", 0.38)

    # Sanity alarms. docs/ARCHITECTURE.md "The safety rule".
    alarm_min_people_per_1000_photos: float = _env_float("WFF_ALARM_MIN_PEOPLE", 10.0)
    alarm_max_people_per_1000_photos: float = _env_float("WFF_ALARM_MAX_PEOPLE", 125.0)


@dataclass
class StorageConfig:
    """Object URIs only. See "The one rule that keeps the GPU migration cheap"."""

    backend: str = _env_str("WFF_STORAGE", "local")
    # Where the local backend materialises r2://bucket/key. Dev only; nothing
    # in the pipeline is allowed to know this path exists.
    local_root: str = _env_str("WFF_STORAGE_ROOT", ".wff-store")
    bucket: str = _env_str("WFF_BUCKET", "wff")


@dataclass
class Config:
    detect: DetectConfig = field(default_factory=DetectConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    derivatives: DerivativeConfig = field(default_factory=DerivativeConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    embedding_model_version: str = EMBEDDING_MODEL_VERSION
    # Where ONNX weights are cached. This IS a local path, deliberately: model
    # weights are not pipeline data, they are part of the runtime.
    model_dir: str = _env_str(
        "WFF_MODEL_DIR", os.path.join(os.path.expanduser("~"), ".wff", "models")
    )
    num_threads: int = _env_int("WFF_NUM_THREADS", 0)  # 0 = let onnxruntime decide
    # How many threads libheif gets for decoding an iPhone .HEIC. Its own
    # default is 4, which leaves half of an 8-thread machine idle on what is
    # 27% of the per-photo cost. 0 = match the machine (capped -- measured on
    # real photos: 4 threads 418ms, 8 threads 357ms, 12 threads 372ms, so more
    # is not monotonically better). Has no effect on JPEGs.
    decode_threads: int = _env_int("WFF_DECODE_THREADS", 0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config() -> Config:
    return Config()
