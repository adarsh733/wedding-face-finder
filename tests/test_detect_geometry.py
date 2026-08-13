"""Tiling, NMS and input framing. Pure geometry, no model needed.

These are the calculations that decide whether a back-row face is ever seen at
all, and they fail silently -- a wrong tile grid produces a plausible-looking
run that just finds fewer people.
"""
from __future__ import annotations

import numpy as np
import pytest

from wff.process.detect import (
    SCRFDDetector,
    Tile,
    _is_clipped_at_interior_edge,
    _pad_to_stride,
    nms,
    plan_tiles,
)


def test_small_image_gets_no_tiles():
    """A 2 MP photo already gives the detector enough scale. Tiling would only
    burn CPU."""
    assert plan_tiles(1600, 1200, 640, 0.25, 0.2, 4) == []


def test_24mp_photo_gets_a_grid():
    tiles = plan_tiles(6000, 4000, 640, 0.25, 0.2, 4)
    assert len(tiles) > 1
    cols = {t.col for t in tiles}
    rows = {t.row for t in tiles}
    assert len(cols) >= 3, "a 6000px wide photo needs at least 3 columns at 0.25 scale"
    assert len(tiles) == len(cols) * len(rows)


def test_every_tile_meets_the_target_scale():
    """The whole point of the grid: once letterboxed into det_size, each tile
    must be scaled by at least target_scale. Otherwise a 90px face still
    arrives too small and tiling achieved nothing."""
    det_size, target = 640, 0.25
    for width, height in [(6000, 4000), (8000, 6000), (4000, 3000), (5472, 3648)]:
        for tile in plan_tiles(width, height, det_size, target, 0.2, 4):
            scale = det_size / max(tile.w, tile.h)
            assert scale >= target - 1e-6, (
                f"{width}x{height} tile {tile.w}x{tile.h} scales to {scale:.3f}, "
                f"below the {target} target"
            )


def test_tiles_cover_the_whole_photo():
    """A gap in the grid is a strip of the photo where nobody is ever found."""
    width, height = 6000, 4000
    tiles = plan_tiles(width, height, 640, 0.25, 0.2, 4)
    covered = np.zeros((height // 10, width // 10), dtype=bool)
    for tile in tiles:
        covered[tile.y // 10 : (tile.y + tile.h) // 10, tile.x // 10 : (tile.x + tile.w) // 10] = True
    assert covered.all(), "tile grid leaves an uncovered region"


def test_tiles_actually_overlap():
    """Overlap is what lets a face straddling a boundary be whole in the
    neighbouring tile."""
    tiles = plan_tiles(6000, 4000, 640, 0.25, 0.2, 4)
    row0 = sorted([t for t in tiles if t.row == 0], key=lambda t: t.col)
    assert len(row0) >= 2
    for left, right in zip(row0, row0[1:]):
        assert right.x < left.x + left.w, "adjacent tiles do not overlap"


def test_tile_count_is_capped():
    tiles = plan_tiles(20000, 20000, 640, 0.25, 0.2, 3)
    assert max(t.col for t in tiles) <= 2
    assert max(t.row for t in tiles) <= 2


def test_nms_removes_the_duplicate_of_one_face():
    boxes = np.array(
        [
            [100, 100, 200, 200],   # found by the full-image pass
            [104, 102, 203, 201],   # the same face, found again in a tile
            [500, 500, 600, 600],   # a different face
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.95, 0.8], dtype=np.float32)
    keep = sorted(nms(boxes, scores, 0.4))
    assert keep == [1, 2]


def test_nms_keeps_two_faces_that_merely_touch():
    """Two people standing shoulder to shoulder must not collapse into one."""
    boxes = np.array([[0, 0, 100, 100], [95, 0, 195, 100]], dtype=np.float32)
    scores = np.array([0.9, 0.9], dtype=np.float32)
    assert len(nms(boxes, scores, 0.4)) == 2


def test_nms_suppresses_a_sliver_hiding_inside_a_real_face():
    """Regression from a real photo (IMG_3526.HEIC, ev_manali01).

    The whole-image pass found a face at 286x389; a tile found a degenerate
    116x382 sliver of the SAME face. IoU was 0.398 -- just under the 0.40
    threshold -- so both survived, and one person ended up with two faces in one
    photo, which cannot happen. IoU under-penalises a small box inside a big
    one because the union is inflated; containment catches it.
    """
    boxes = np.array(
        [
            [1546.0, 697.0, 1546 + 286, 697 + 389],
            [1563.0, 697.0, 1563 + 116, 697 + 382],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.91, 0.62], dtype=np.float32)

    x1 = max(boxes[0][0], boxes[1][0])
    x2 = min(boxes[0][2], boxes[1][2])
    inter = (x2 - x1) * (min(boxes[0][3], boxes[1][3]) - max(boxes[0][1], boxes[1][1]))
    area_a = 286 * 389
    area_b = 116 * 382
    iou = inter / (area_a + area_b - inter)
    assert 0.35 < iou < 0.40, "the test data must reproduce the near-miss IoU"

    assert len(nms(boxes, scores, 0.40)) == 1, "the sliver survived NMS again"
    # And the surviving one is the real detection, not the sliver.
    assert nms(boxes, scores, 0.40)[0] == 0


def test_containment_does_not_suppress_a_genuinely_smaller_nearby_face():
    """A child standing in front of an adult is small AND close, but not
    contained. Suppressing that would lose a real person."""
    boxes = np.array(
        [[0.0, 0.0, 200.0, 260.0], [210.0, 120.0, 300.0, 235.0]], dtype=np.float32
    )
    scores = np.array([0.95, 0.80], dtype=np.float32)
    assert len(nms(boxes, scores, 0.40)) == 2


def test_nms_on_empty():
    assert nms(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.4) == []


def test_truncated_face_at_an_interior_edge_is_dropped():
    """A face cut in half by a tile boundary produces a truncated box that can
    OUTSCORE the complete detection from the neighbouring tile. NMS would then
    keep the wrong one, so we drop it up front."""
    tile = Tile(x=0, y=0, w=1000, h=1000, row=0, col=0)
    running_off = np.array([950.0, 400.0, 1060.0, 510.0], dtype=np.float32)
    assert _is_clipped_at_interior_edge(running_off, tile, 6000, 4000)


def test_face_at_a_real_photo_edge_is_kept():
    """The same overrun at the actual edge of the photo is genuine -- there is
    no neighbouring tile to find it better."""
    tile = Tile(x=5000, y=0, w=1000, h=1000, row=0, col=2)
    running_off = np.array([950.0, 400.0, 1060.0, 510.0], dtype=np.float32)
    assert not _is_clipped_at_interior_edge(running_off, tile, 6000, 4000)


# ---------------------------------------------------------------------------
# How the photo is framed for the network
#
# We run the detector FIVE times per photo (whole image + four tiles), so it is
# 56% of the entire per-photo cost. Padding a 4:3 photo out to a square spends a
# quarter of that on black. These tests pin both halves of the fix: the padding
# stays tight, and the coordinates that come back are still right.
# ---------------------------------------------------------------------------


def test_padding_is_the_minimum_the_strides_require():
    # Already aligned -- must not grow at all.
    assert _pad_to_stride(480, 640, 32) == (480, 640)
    # Not aligned -- grows to the next multiple, and no further.
    assert _pad_to_stride(481, 641, 32) == (512, 672)
    assert _pad_to_stride(1, 1, 32) == (32, 32)
    # The 5-level exports have a coarsest stride of 128.
    assert _pad_to_stride(480, 640, 128) == (512, 640)


def test_a_4_by_3_photo_does_not_get_padded_to_a_square():
    """The whole optimisation, stated as a property.

    A 640x640 canvas for a 4:3 photo is 25% zeros, convolved at full price on
    every one of the five passes. Anything that reintroduces the square canvas
    should fail here rather than quietly costing 40 minutes on a 7,000-photo
    folder.
    """
    detector, session = _fake_detector()
    detector._forward(_photo(4032, 3024), 640, 0.5)

    height, width = session.last_input_shape
    assert (height, width) == (480, 640), "4:3 photo should give a 640x480 canvas"

    # Not one pixel of this canvas is padding: 640x480 is already a whole
    # number of strides, so the photo fills it exactly. The old square canvas
    # was 25% black, and that black cost the same to convolve as a face.
    content_h, content_w = session.last_content_shape
    assert (content_h, content_w) == (height, width)
    assert (height * width) / (640 * 640) == 0.75, "should be 25% less work"

    # And the general rule, not just this one photo.
    assert height % detector.stride_max == 0
    assert width % detector.stride_max == 0


def test_a_portrait_photo_is_framed_tightly_too():
    detector, session = _fake_detector()
    detector._forward(_photo(3024, 4032), 640, 0.5)
    assert session.last_input_shape == (640, 480)


def test_padding_never_exceeds_one_stride_on_an_awkward_shape():
    """A panorama, and a shape whose fitted size lands mid-stride."""
    for width, height in [(4000, 3000), (5000, 2000), (1999, 3001), (6000, 4000)]:
        detector, session = _fake_detector()
        detector._forward(_photo(width, height), 640, 0.5)
        canvas_h, canvas_w = session.last_input_shape
        fitted_h, fitted_w = session.last_content_shape
        assert 0 <= canvas_h - fitted_h < detector.stride_max
        assert 0 <= canvas_w - fitted_w < detector.stride_max


def test_a_face_decodes_to_the_same_spot_as_with_a_square_canvas():
    """The load-bearing half: tighter framing must not move the answer.

    A portrait photo is the case that would break silently. Its tight canvas is
    640x480, so a feature-map row is 60 cells wide instead of 80 -- read the
    anchor grid off det_size instead of the real canvas and every detection
    lands somewhere else, with no error raised. We plant one detection at the
    centre of the feature map and check it comes back at the centre of the
    photo.
    """
    detector, session = _fake_detector()
    # stride 8 on a 640x480 canvas -> an 80-row by 60-column feature map.
    session.plant(stride=8, row=40, col=30, feature_map_width=60)

    boxes, kps, scores = detector._forward(_photo(3024, 4032), 640, 0.5)

    assert len(boxes) == 1, "the planted detection should survive the threshold"
    centre_x = (boxes[0][0] + boxes[0][2]) / 2
    centre_y = (boxes[0][1] + boxes[0][3]) / 2
    # anchor (col*8, row*8) = (240, 320) in canvas space; scale is 640/4032.
    assert centre_x == pytest.approx(3024 / 2, abs=1.0)
    assert centre_y == pytest.approx(4032 / 2, abs=1.0)
    assert kps.shape == (1, 5, 2)


def test_every_output_level_is_decoded_on_a_non_square_canvas():
    """Each FPN level divides the canvas differently; a wrong grid on any one of
    them raises a reshape error or silently drops that level's faces."""
    detector, session = _fake_detector()
    for stride, fm_width in [(8, 60), (16, 30), (32, 15)]:
        session.reset()
        session.plant(stride=stride, row=2, col=1, feature_map_width=fm_width)
        boxes, _, _ = detector._forward(_photo(3024, 4032), 640, 0.5)
        assert len(boxes) == 1, f"stride {stride} level was not decoded"


# -- the stand-in ------------------------------------------------------------
#
# Loading the real 275 MB det_10g.onnx into a unit test would make the suite
# slow and undownloadable offline. This returns correctly shaped outputs for
# whatever input it is handed, which is exactly what is needed to check that
# the anchor grid is derived from the real canvas.


class _FakeSession:
    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self) -> None:
        self.last_input_shape: tuple[int, int] = (0, 0)
        self.last_content_shape: tuple[int, int] = (0, 0)
        self._planted: list[tuple[int, int]] = []

    # the bits of an onnxruntime session the detector actually uses
    def get_inputs(self):
        return [_Named("input.1")]

    def get_outputs(self):
        return [_Named(f"out{i}") for i in range(9)]

    def reset(self) -> None:
        self._planted = []

    def plant(self, stride: int, row: int, col: int, feature_map_width: int) -> None:
        """Make one anchor score 1.0, at (row, col) of that stride's grid."""
        level = self.STRIDES.index(stride)
        index = (row * feature_map_width + col) * self.NUM_ANCHORS
        self._planted.append((level, index))

    def run(self, output_names, feed):
        blob = feed["input.1"]
        _, _, height, width = blob.shape
        self.last_input_shape = (height, width)
        # Where the real pixels stop and our padding begins.
        non_zero = np.argwhere(blob[0].sum(axis=0) != -3 * 127.5 / 128.0)
        self.last_content_shape = (
            (int(non_zero[:, 0].max()) + 1, int(non_zero[:, 1].max()) + 1)
            if len(non_zero)
            else (height, width)
        )

        outputs = []
        for stride in self.STRIDES:  # scores
            count = (height // stride) * (width // stride) * self.NUM_ANCHORS
            outputs.append(np.zeros((count, 1), dtype=np.float32))
        for stride in self.STRIDES:  # bbox distances -- zero = a point box
            count = (height // stride) * (width // stride) * self.NUM_ANCHORS
            outputs.append(np.zeros((count, 4), dtype=np.float32))
        for stride in self.STRIDES:  # landmark offsets
            count = (height // stride) * (width // stride) * self.NUM_ANCHORS
            outputs.append(np.zeros((count, 10), dtype=np.float32))
        for level, index in self._planted:
            outputs[level][index] = 1.0
        return outputs


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name
        self.shape = [1, 3, "?", "?"]


def _fake_detector():
    """A real SCRFDDetector wired to the stand-in session."""
    from wff.process import detect as detect_module

    session = _FakeSession()
    original = detect_module.make_session
    detect_module.make_session = lambda path, threads=0: session
    try:
        detector = SCRFDDetector(__file__, 0)  # path is never opened
    finally:
        detect_module.make_session = original
    return detector, session


def _photo(width: int, height: int) -> np.ndarray:
    """A mid-grey photo. Deliberately NOT black: the padding we are measuring is
    black, and a black photo would be indistinguishable from it."""
    return np.full((height, width, 3), 200, dtype=np.uint8)
