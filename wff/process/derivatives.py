"""The only pixels we keep.

STEP 3.6. Both cut from the full-resolution original, in memory, immediately
before it is discarded.

    Preview   2048px  ~400 KB   what a guest sees when they tap a photo
    Thumbnail  400px   ~40 KB   what fills the results grid

The original is never written to our disk and never uploaded to our storage.
Water passing through a filter is not water sitting in a reservoir.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import DerivativeConfig


def _resize_long_edge(image: np.ndarray, target: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= target:
        return image
    scale = target / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    # INTER_AREA averages when shrinking. Anything else aliases, and a preview
    # is the thing a guest actually looks at.
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality), int(cv2.IMWRITE_JPEG_OPTIMIZE), 1],
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()


def make_derivatives(
    image_bgr: np.ndarray, config: DerivativeConfig
) -> tuple[bytes, bytes]:
    """(preview_jpeg, thumbnail_jpeg) from the full-resolution array."""
    preview = _resize_long_edge(image_bgr, config.preview_px)
    preview_bytes = _encode_jpeg(preview, config.preview_quality)
    # Derive the thumbnail from the preview, not the original: it is 9x less
    # data to resample and visually identical at 400px.
    thumb = _resize_long_edge(preview, config.thumb_px)
    thumb_bytes = _encode_jpeg(thumb, config.thumb_quality)
    return preview_bytes, thumb_bytes


def encode_face_crop(aligned_bgr: np.ndarray, quality: int = 90) -> bytes:
    """The aligned 112x112 crop, exactly as the model saw it.

    Saved so a human can look at what the machine looked at. When a cluster is
    wrong, this is the first place you find out why.
    """
    return _encode_jpeg(aligned_bgr, quality)
