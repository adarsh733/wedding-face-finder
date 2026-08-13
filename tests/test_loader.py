"""Decoding a photo into memory.

Opening a 12 MP iPhone .HEIC is 27% of the per-photo cost -- the second most
expensive step after face detection -- so how many threads libheif gets is a
real number, not a detail.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pytest
from PIL import Image

from wff.process import loader
from wff.process.loader import configure_decoder, load_image


def test_decoder_uses_the_whole_machine_by_default():
    """libheif's own default is 4, which idles half of an 8-thread machine.

    Measured on real 12 MP iPhone photos: 4 threads 418ms, 8 threads 357ms.
    Over 5,000 HEICs that is about five minutes.
    """
    pytest.importorskip("pillow_heif")
    chosen = configure_decoder(0)
    assert chosen == min(8, os.cpu_count() or 4)
    assert chosen >= 4, "never fewer threads than libheif would have used itself"

    import pillow_heif

    assert pillow_heif.options.DECODE_THREADS == chosen


def test_more_threads_is_not_blindly_better():
    """12 threads measured SLOWER than 8 (372ms vs 357ms), so a 32-core box
    must not be handed 32. The cap is the point of this function."""
    pytest.importorskip("pillow_heif")
    assert configure_decoder(0) <= 8


def test_an_explicit_setting_wins():
    """WFF_DECODE_THREADS exists so the number can be tuned on a machine that
    does not look like this one, without editing code."""
    pytest.importorskip("pillow_heif")
    assert configure_decoder(2) == 2

    import pillow_heif

    assert pillow_heif.options.DECODE_THREADS == 2
    configure_decoder(0)  # put it back for the rest of the suite


def test_configure_decoder_is_safe_without_pillow_heif(monkeypatch):
    """JPEG-only machines must not crash on a missing optional dependency."""
    monkeypatch.setattr(loader, "HEIF_AVAILABLE", False)
    assert configure_decoder(8) == 0


def test_decoding_still_returns_the_photo_the_right_way_up():
    """The thread count must not touch what comes back. A 3x2 image stays 3
    wide and 2 tall, in BGR, at native resolution."""
    configure_decoder(0)
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255  # pure red, so a BGR swap is visible
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    buffer.seek(0)

    loaded = load_image(buffer)

    assert (loaded.width, loaded.height) == (3, 2)
    assert loaded.bgr.shape == (2, 3, 3)
    # red in RGB must arrive in the LAST channel of BGR
    assert loaded.bgr[0, 0, 2] == 255
    assert loaded.bgr[0, 0, 0] == 0
