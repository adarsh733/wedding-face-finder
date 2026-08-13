"""Straighten a face into the 112x112 square ArcFace expects.

STEP 3.3 -- ALIGNMENT. The model does not want a rectangle cut out of the photo.
It wants the face *straightened*: eyes at fixed positions, like a passport
photo. The detector hands us 5 landmarks; we use them to rotate, scale and
translate the face into a canonical square.

Skipping this costs real accuracy and is easy to forget, because the output
still looks like a face either way.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# The canonical ArcFace 5-point template for a 112x112 crop, in pixels.
# Order matches the detector: left eye, right eye, nose, left mouth, right mouth
# (left/right as the viewer sees them).
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama 1991). src -> dst, 3x3.

    Deterministic and exact, unlike cv2.estimateAffinePartial2D, which is
    RANSAC-based and can wobble on only 5 points. Repeatability matters: while
    tuning, any change in the output must be caused by our change and nothing
    else.
    """
    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    covariance = dst_demean.T @ src_demean / num

    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        d[dim - 1] = -1

    transform = np.eye(dim + 1, dtype=np.float64)
    U, S, Vt = np.linalg.svd(covariance)
    rank = np.linalg.matrix_rank(covariance)

    if rank == 0:
        return np.full((dim + 1, dim + 1), np.nan)
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            transform[:dim, :dim] = U @ Vt
        else:
            saved = d[dim - 1]
            d[dim - 1] = -1
            transform[:dim, :dim] = U @ np.diag(d) @ Vt
            d[dim - 1] = saved
    else:
        transform[:dim, :dim] = U @ np.diag(d) @ Vt

    variance = src_demean.var(axis=0).sum()
    scale = 1.0 if variance == 0 else (1.0 / variance) * (S @ d)
    transform[:dim, dim] = dst_mean - scale * (transform[:dim, :dim] @ src_mean.T)
    transform[:dim, :dim] *= scale
    return transform


def transform_scale(matrix: np.ndarray) -> float:
    """The uniform scale factor of a similarity matrix."""
    return float(math.sqrt(abs(np.linalg.det(matrix[:2, :2]))))


def align_face(
    image: np.ndarray, kps: np.ndarray, out_size: int = 112
) -> tuple[np.ndarray | None, float]:
    """Warp a face from the full-resolution photo into an out_size square.

    Returns (aligned BGR image, scale factor). Scale < 1 means the face was
    larger than the template and got shrunk; > 1 means it was smaller and got
    stretched, which is the case we do not trust.

    ANTI-ALIASING, and why it is not a nicety:
    A bride's close-up face can be 1500px tall. Mapping that to 112 is a 13x
    reduction. cv2.warpAffine with INTER_LINEAR samples roughly one source
    pixel per output pixel, so 12 of every 13 pixels are simply skipped -- the
    result is an aliased, high-frequency mess that is NOT what a clean 112px
    face looks like, and the embedding drifts accordingly. So for any large
    reduction we first shrink a region of interest with INTER_AREA (which
    averages, correctly) and only then warp.
    """
    template = ARCFACE_TEMPLATE_112 * (out_size / 112.0)
    src = np.asarray(kps, dtype=np.float64).reshape(5, 2)

    matrix = _umeyama(src, template)
    if not np.isfinite(matrix).all():
        return None, 0.0
    scale = transform_scale(matrix)
    if scale <= 0:
        return None, 0.0

    if scale >= 0.5:
        aligned = cv2.warpAffine(
            image, matrix[:2], (out_size, out_size), flags=cv2.INTER_LINEAR
        )
        return aligned, scale

    # Large reduction: pre-shrink a region of interest with a filter that
    # actually averages, then redo the fit in that reduced space.
    factor = max(2, int(0.75 / scale))
    inverse = np.linalg.inv(matrix)
    corners = np.array(
        [[0, 0, 1], [out_size, 0, 1], [out_size, out_size, 1], [0, out_size, 1]],
        dtype=np.float64,
    )
    source_corners = (inverse @ corners.T).T[:, :2]

    height, width = image.shape[:2]
    # np.ptp(), not ndarray.ptp() -- the method was removed in NumPy 2.0.
    pad = 0.15 * max(
        float(np.ptp(source_corners[:, 0])), float(np.ptp(source_corners[:, 1])), 1.0
    )
    x0 = int(max(0, math.floor(source_corners[:, 0].min() - pad)))
    y0 = int(max(0, math.floor(source_corners[:, 1].min() - pad)))
    x1 = int(min(width, math.ceil(source_corners[:, 0].max() + pad)))
    y1 = int(min(height, math.ceil(source_corners[:, 1].max() + pad)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        aligned = cv2.warpAffine(
            image, matrix[:2], (out_size, out_size), flags=cv2.INTER_LINEAR
        )
        return aligned, scale

    roi = image[y0:y1, x0:x1]
    new_w = max(1, (x1 - x0) // factor)
    new_h = max(1, (y1 - y0) // factor)
    reduced = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Exact, rather than composing matrices: put the landmarks in reduced-ROI
    # coordinates and fit again.
    scale_x = new_w / float(x1 - x0)
    scale_y = new_h / float(y1 - y0)
    src_reduced = np.empty_like(src)
    src_reduced[:, 0] = (src[:, 0] - x0) * scale_x
    src_reduced[:, 1] = (src[:, 1] - y0) * scale_y

    matrix_reduced = _umeyama(src_reduced, template)
    if not np.isfinite(matrix_reduced).all():
        return None, 0.0
    aligned = cv2.warpAffine(
        reduced, matrix_reduced[:2], (out_size, out_size), flags=cv2.INTER_LINEAR
    )
    return aligned, scale
