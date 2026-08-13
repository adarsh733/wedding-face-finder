"""Decode a photo into memory, right way up. Never writes to disk.

STEP 3.1 -- ROTATION. Photos carry a hidden "which way is up" tag. Ignore it
and half the wedding is sideways and the face-finder sees nothing. It is the
most common silent bug in image pipelines. Apply it first, always.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from datetime import datetime
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

# iPhone .HEIC. Registers itself with Pillow on import.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:  # pragma: no cover - optional at dev time
    HEIF_AVAILABLE = False

# Beyond this, more threads measured SLOWER (12 threads 372ms vs 8 threads
# 357ms on a 12 MP HEIC), so matching a 32-core machine would be a pessimism.
_MAX_DECODE_THREADS = 8


def configure_decoder(threads: int = 0) -> int:
    """Tell libheif how many threads to use. Returns what was actually set.

    Opening a 12 MP .HEIC is 27% of the per-photo cost -- the second most
    expensive thing after face detection -- and libheif's own default of 4
    leaves half of an 8-thread machine idle on it. Process-wide, so the
    processor sets it once per job rather than per photo.

    Safe to call when pillow-heif is missing; there is simply nothing to set.
    """
    if not HEIF_AVAILABLE:  # pragma: no cover - pillow-heif is a hard dep today
        return 0
    if threads <= 0:
        threads = min(_MAX_DECODE_THREADS, os.cpu_count() or 4)
    pillow_heif.options.DECODE_THREADS = threads
    return threads

# A 24 MP photo is a legitimate input, not a decompression bomb.
Image.MAX_IMAGE_PIXELS = 300_000_000

_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_DATETIME = 306


@dataclass
class LoadedImage:
    """The full-resolution original, in memory, for about half a second."""

    bgr: np.ndarray          # HxWx3 uint8, OpenCV convention
    width: int
    height: int
    taken_at: datetime | None
    exif_rotated: bool

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000


def _parse_exif_datetime(raw: str) -> datetime | None:
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def load_image(stream: BinaryIO) -> LoadedImage:
    """Read a photo stream to a right-way-up BGR array at NATIVE resolution.

    No resizing happens here. Not now, not later, not "just to be safe". The
    whole product depends on measuring faces while the detail is still there.
    """
    data = stream.read()
    with Image.open(io.BytesIO(data)) as img:
        taken_at = None
        try:
            exif = img.getexif()
            raw = exif.get(_EXIF_DATETIME_ORIGINAL) or exif.get(_EXIF_DATETIME)
            if raw:
                taken_at = _parse_exif_datetime(str(raw))
        except Exception:
            taken_at = None

        before = img.size
        upright = ImageOps.exif_transpose(img)
        rotated = upright.size != before
        if upright.mode != "RGB":
            upright = upright.convert("RGB")
        rgb = np.asarray(upright, dtype=np.uint8)

    # RGB -> BGR without importing cv2 here; a view-reverse plus copy is
    # measurably cheaper than cvtColor on large arrays.
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    height, width = bgr.shape[:2]
    return LoadedImage(
        bgr=bgr,
        width=width,
        height=height,
        taken_at=taken_at,
        exif_rotated=rotated,
    )
