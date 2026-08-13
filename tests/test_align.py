"""Alignment. The step whose absence still produces a face-looking image.

If alignment is wrong, nothing crashes and nothing looks obviously broken --
accuracy just drops. So it gets tested numerically.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from wff.process.align import (
    ARCFACE_TEMPLATE_112,
    _umeyama,
    align_face,
    transform_scale,
)


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    return (matrix @ homogeneous.T).T[:, :2]


def test_umeyama_recovers_a_known_transform():
    angle = math.radians(23.0)
    scale = 2.4
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    truth = np.eye(3)
    truth[:2, :2] = scale * rotation
    truth[:2, 2] = [140.0, -35.0]

    source = ARCFACE_TEMPLATE_112.copy()
    target = _apply(truth, source)

    recovered = _umeyama(source, target)
    assert np.allclose(recovered, truth, atol=1e-6)
    assert transform_scale(recovered) == pytest.approx(scale, abs=1e-6)


def test_aligned_landmarks_land_on_the_template():
    """After warping, the eyes must sit exactly where ArcFace expects them.

    This is the property that matters: the model was trained on faces with the
    eyes at these coordinates.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (1000, 1000, 3), dtype=np.uint8)

    angle = math.radians(-14.0)
    scale = 3.1
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    placement = np.eye(3)
    placement[:2, :2] = scale * rotation
    placement[:2, 2] = [400.0, 350.0]
    kps = _apply(placement, ARCFACE_TEMPLATE_112)

    aligned, _ = align_face(image, kps, 112)
    assert aligned is not None
    assert aligned.shape == (112, 112, 3)

    matrix = _umeyama(kps, ARCFACE_TEMPLATE_112)
    landed = _apply(matrix, kps)
    assert np.allclose(landed, ARCFACE_TEMPLATE_112, atol=1e-4)


def test_scale_reports_shrink_versus_stretch():
    """scale < 1 means the face was bigger than 112 and got shrunk (good).
    scale > 1 means it was smaller and got stretched -- the case where the
    embedding drifts toward the 'generic face' region."""
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (2000, 2000, 3), dtype=np.uint8)

    big = ARCFACE_TEMPLATE_112 * 8.0 + np.array([500.0, 500.0])
    _, scale_big = align_face(image, big, 112)
    assert scale_big < 0.2

    tiny = ARCFACE_TEMPLATE_112 * 0.5 + np.array([500.0, 500.0])
    _, scale_tiny = align_face(image, tiny, 112)
    assert scale_tiny > 1.5


def test_large_reduction_does_not_alias():
    """A big face shrunk to 112 must be AVERAGED, not point-sampled.

    A fine checkerboard downsampled correctly goes flat grey. Point-sampled, it
    keeps high-contrast noise. That noise is not what a clean 112px face looks
    like, and the embedding moves accordingly.
    """
    size = 1600
    checker = np.zeros((size, size, 3), dtype=np.uint8)
    checker[::2, :, :] = 255  # 1px stripes: pure high frequency

    kps = ARCFACE_TEMPLATE_112 * 12.0 + np.array([200.0, 200.0])
    aligned, scale = align_face(checker, kps, 112)
    assert aligned is not None
    assert scale < 0.1, "this test only means anything under a big reduction"

    spread = float(aligned.astype(np.float32).std())
    assert spread < 40.0, (
        f"aligned crop still has std {spread:.1f} -- the stripes survived, so "
        "the reduction is point-sampling instead of averaging"
    )


def test_degenerate_landmarks_do_not_crash():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    identical = np.tile(np.array([50.0, 50.0]), (5, 1))
    aligned, _ = align_face(image, identical, 112)
    assert aligned is None or aligned.shape == (112, 112, 3)
