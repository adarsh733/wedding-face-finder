"""SCRFD face detection, run twice: whole image, then a grid of native-res tiles.

THE PROBLEM THIS SOLVES
-----------------------
SCRFD works at a fixed square input (640x640 by default). A 6000x4000 photo
letterboxed into that is scaled by 0.107, so a 90px back-row face arrives at the
network as 9px and is simply not there.

Processing the original at full resolution achieves NOTHING on its own, because
the detector shrinks its input anyway. Tiling is what makes "detect at native
resolution" mean something:

    One 6000px photo            Looked at as 3x3 overlapping tiles
    +------------------+        +-----+-----+-----+
    |  30 people, all  |  --->  +-----+-----+-----+   each tile gets
    |  faces ~90px     |        +-----+-----+-----+   full attention
    +------------------+        +-----+-----+-----+
      detector sees 9px            detector sees ~22px  OK

Then merge both passes and drop the duplicates (NMS).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import DetectConfig
from .models import make_session


@dataclass
class Detection:
    """One face, in ORIGINAL photo coordinates. Never in tile or resized space."""

    bbox: np.ndarray      # (4,) float32 x1,y1,x2,y2
    kps: np.ndarray       # (5,2) float32 -- L eye, R eye, nose, L mouth, R mouth
    score: float
    source: str           # 'full' or 'tile:r,c' -- kept for debugging the grid

    @property
    def height_px(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def width_px(self) -> float:
        return float(self.bbox[2] - self.bbox[0])


# ---------------------------------------------------------------------------
# Tile planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    w: int
    h: int
    row: int
    col: int


def plan_tiles(
    width: int,
    height: int,
    det_size: int,
    target_scale: float,
    overlap: float,
    max_per_axis: int,
) -> list[Tile]:
    """Choose the smallest grid that keeps faces big enough for the detector.

    A tile of size (tw, th) letterboxed into det_size is scaled by
    det_size / max(tw, th). We want that scale >= target_scale, so:

        max(tw, th) <= det_size / target_scale

    Solving for the tile count on an axis of length L, with fractional overlap
    o between neighbours:

        tw = L / (n - (n-1)*o)   =>   n >= (L/max_edge - o) / (1 - o)

    This is adaptive rather than a hard-coded 3x3: a 24 MP photo gets 3x3, a
    2 MP one gets 1x1 and costs nothing.
    """
    max_edge = det_size / max(target_scale, 1e-6)

    def axis_count(length: int) -> int:
        if length <= max_edge:
            return 1
        n = math.ceil((length / max_edge - overlap) / (1.0 - overlap))
        return max(1, min(int(n), max_per_axis))

    ncols = axis_count(width)
    nrows = axis_count(height)
    if ncols == 1 and nrows == 1:
        return []

    def axis_starts(length: int, n: int) -> list[tuple[int, int]]:
        if n == 1:
            return [(0, length)]
        edge = length / (n - (n - 1) * overlap)
        stride = edge * (1.0 - overlap)
        spans = []
        for i in range(n):
            start = int(round(i * stride))
            size = int(round(edge))
            if start + size > length:      # last tile flush against the edge
                start = max(0, length - size)
                size = length - start
            spans.append((start, size))
        return spans

    tiles: list[Tile] = []
    for row, (y, th) in enumerate(axis_starts(height, nrows)):
        for col, (x, tw) in enumerate(axis_starts(width, ncols)):
            tiles.append(Tile(x=x, y=y, w=tw, h=th, row=row, col=col))
    return tiles


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    containment_threshold: float = 0.80,
) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, best first.

    Suppresses on EITHER plain IoU or containment (intersection over the
    smaller box's area).

    Why containment is needed, from a real photo (IMG_3526.HEIC): the whole-image
    pass found a face at 286x389, and a tile found a degenerate 116x382 sliver of
    the same face. Their IoU was 0.398 -- a hair under the 0.40 threshold -- so
    both survived, and one person ended up with two faces in one photo, which is
    impossible. IoU systematically under-penalises a small box sitting inside a
    big one, because the big box's area inflates the union. Containment does not
    have that blind spot: that sliver was ~99% inside the real detection.
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort(kind="stable")[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        smaller = np.minimum(areas[i], areas[rest])
        containment = np.where(smaller > 0, inter / np.maximum(smaller, 1e-9), 0.0)
        order = rest[(iou <= iou_threshold) & (containment <= containment_threshold)]
    return keep


def _pad_to_stride(height: int, width: int, stride_max: int) -> tuple[int, int]:
    """Smallest canvas >= (height, width) that every FPN level divides exactly.

    The detector decodes each output level by walking a (canvas / stride) grid
    of anchors, so the canvas must be a whole number of strides on both axes.
    That -- and nothing more -- is the real constraint. The upstream code
    satisfies it by padding out to a full det_size square, which is up to 160
    wasted rows; this is at most stride_max - 1.
    """
    def up(value: int) -> int:
        return ((value + stride_max - 1) // stride_max) * stride_max

    return up(height), up(width)


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, 0] + distance[:, i])
        preds.append(points[:, 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


class SCRFDDetector:
    """SCRFD via ONNX Runtime. Anchor decoding implemented here, not imported.

    Keeping the decode in our own code is what lets us drive the network
    tile-by-tile, which the upstream FaceAnalysis wrapper does not expose.
    """

    def __init__(self, model_path: Path, num_threads: int = 0) -> None:
        self.session = make_session(model_path, num_threads)
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.input_mean = 127.5
        self.input_std = 128.0

        # The output count tells us the FPN shape. det_10g.onnx gives 9.
        num_outputs = len(self.output_names)
        if num_outputs == 6:
            self.fmc, self.strides, self.num_anchors, self.use_kps = 3, [8, 16, 32], 2, False
        elif num_outputs == 9:
            self.fmc, self.strides, self.num_anchors, self.use_kps = 3, [8, 16, 32], 2, True
        elif num_outputs == 10:
            self.fmc, self.strides, self.num_anchors, self.use_kps = 5, [8, 16, 32, 64, 128], 1, False
        elif num_outputs == 15:
            self.fmc, self.strides, self.num_anchors, self.use_kps = 5, [8, 16, 32, 64, 128], 1, True
        else:
            raise RuntimeError(f"Unrecognised SCRFD export with {num_outputs} outputs")
        if not self.use_kps:
            raise RuntimeError(
                "This SCRFD export has no landmarks. Alignment needs the 5 points; "
                "without them recognition accuracy drops badly. Use det_10g.onnx."
            )
        # The coarsest FPN level sets how much the input has to be padded by.
        # 32 for det_10g, 128 for the 5-level exports -- read it, never assume.
        self.stride_max = max(self.strides)
        self._anchor_cache: dict[tuple[int, int, int], np.ndarray] = {}

    # -- one forward pass on one array ------------------------------------
    def _forward(self, image: np.ndarray, det_size: int, thresh: float):
        """Fit into det_size, run the net, decode to `image` coordinates.

        WHY THE CANVAS IS NOT A SQUARE.
        The obvious thing -- and what the upstream code does -- is to paste the
        resized photo into a det_size x det_size square of zeros. But a 4:3
        photo only fills 640x480 of a 640x640 square, so a QUARTER of every
        forward pass is spent convolving black. We run the detector five times
        per photo (whole image + four tiles), so that is a quarter of the single
        most expensive thing the pipeline does, thrown away.

        The network is fully convolutional and this ONNX export has dynamic
        height and width (only the BATCH dim is frozen at 1), so it will happily
        take 640x480. We pad only up to the next multiple of the coarsest
        stride, which is what the feature-map arithmetic below actually
        requires -- at most 31 rows or columns instead of 160.

        Measured on 16 real photos (4032x3024 HEIC + JPG, both orientations):
        28.6% faster detection, and the SAME faces -- 23 of 23 matched, worst
        bounding-box corner moved 0.99px and worst landmark 0.375px, on a
        4032px-wide photo. See tests/test_detect_geometry.py.
        """
        img_h, img_w = image.shape[:2]
        im_ratio = img_h / float(img_w)
        if im_ratio > 1.0:
            new_h = det_size
            new_w = max(1, int(round(new_h / im_ratio)))
        else:
            new_w = det_size
            new_h = max(1, int(round(new_w * im_ratio)))
        scale = new_h / float(img_h)

        resized = cv2.resize(
            image,
            (new_w, new_h),
            # INTER_AREA is the correct filter when shrinking. INTER_LINEAR
            # aliases, and aliasing on a 60px face is exactly the detail we are
            # trying to protect.
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        # Round up to the coarsest stride so every FPN level divides exactly.
        # Top-left aligned padding, so undoing it is still a single divide.
        canvas_h, canvas_w = _pad_to_stride(new_h, new_w, self.stride_max)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized

        blob = cv2.dnn.blobFromImage(
            canvas,
            1.0 / self.input_std,
            (canvas_w, canvas_h),
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        boxes_all, kps_all, scores_all = [], [], []
        for idx, stride in enumerate(self.strides):
            scores = outputs[idx].reshape(-1)
            bbox_preds = outputs[idx + self.fmc].reshape(-1, 4) * stride
            kps_preds = outputs[idx + self.fmc * 2].reshape(-1, 10) * stride

            fm_h, fm_w = canvas_h // stride, canvas_w // stride
            key = (fm_h, fm_w, stride)
            anchors = self._anchor_cache.get(key)
            if anchors is None:
                grid = np.stack(np.mgrid[:fm_h, :fm_w][::-1], axis=-1).astype(np.float32)
                anchors = (grid * stride).reshape(-1, 2)
                if self.num_anchors > 1:
                    anchors = np.stack([anchors] * self.num_anchors, axis=1).reshape(-1, 2)
                self._anchor_cache[key] = anchors

            positive = np.where(scores >= thresh)[0]
            if positive.size == 0:
                continue
            boxes_all.append(_distance2bbox(anchors[positive], bbox_preds[positive]))
            kps_all.append(_distance2kps(anchors[positive], kps_preds[positive]).reshape(-1, 5, 2))
            scores_all.append(scores[positive])

        if not boxes_all:
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0, 5, 2), np.float32),
                np.zeros((0,), np.float32),
            )

        # Undo the letterbox: one divide, because padding was top-left.
        boxes = np.vstack(boxes_all).astype(np.float32) / scale
        kps = np.vstack(kps_all).astype(np.float32) / scale
        scores = np.concatenate(scores_all).astype(np.float32)
        return boxes, kps, scores

    # -- the two-pass public API -------------------------------------------
    def detect(self, image: np.ndarray, config: DetectConfig) -> list[Detection]:
        """Whole image + native-resolution tiles, merged and de-duplicated."""
        height, width = image.shape[:2]
        candidates: list[Detection] = []

        # Pass A -- the whole photo. Cheap, and catches every big face.
        boxes, kps, scores = self._forward(image, config.det_size, config.det_thresh)
        for box, points, score in zip(boxes, kps, scores):
            candidates.append(Detection(box, points, float(score), "full"))

        # Pass B -- tiles at native resolution. Catches the back row.
        should_tile = (
            config.tiling_enabled
            and max(width, height) >= config.min_side_for_tiling
        )
        if should_tile:
            tiles = plan_tiles(
                width,
                height,
                config.det_size,
                config.tile_target_scale,
                config.tile_overlap,
                config.max_tiles_per_axis,
            )
            for tile in tiles:
                crop = image[tile.y : tile.y + tile.h, tile.x : tile.x + tile.w]
                t_boxes, t_kps, t_scores = self._forward(
                    crop, config.det_size, config.det_thresh
                )
                for box, points, score in zip(t_boxes, t_kps, t_scores):
                    # A face straddling an INTERIOR tile edge is truncated here
                    # and whole in the neighbour. A truncated box can outscore
                    # the complete one, so drop it rather than trust NMS.
                    if _is_clipped_at_interior_edge(box, tile, width, height):
                        continue
                    box = box.copy()
                    box[0] += tile.x
                    box[2] += tile.x
                    box[1] += tile.y
                    box[3] += tile.y
                    points = points.copy()
                    points[:, 0] += tile.x
                    points[:, 1] += tile.y
                    candidates.append(
                        Detection(box, points, float(score), f"tile:{tile.row},{tile.col}")
                    )

        if not candidates:
            return []

        all_boxes = np.stack([d.bbox for d in candidates])
        all_scores = np.array([d.score for d in candidates], dtype=np.float32)
        keep = nms(all_boxes, all_scores, config.nms_iou, config.nms_containment)
        merged = [candidates[i] for i in keep]

        # Clamp to the photo. Regression can push a box slightly outside.
        for det in merged:
            det.bbox[0] = max(0.0, det.bbox[0])
            det.bbox[1] = max(0.0, det.bbox[1])
            det.bbox[2] = min(float(width), det.bbox[2])
            det.bbox[3] = min(float(height), det.bbox[3])
        return [d for d in merged if d.width_px > 1 and d.height_px > 1]


def _is_clipped_at_interior_edge(
    box: np.ndarray, tile: Tile, image_w: int, image_h: int, margin: float = 0.12
) -> bool:
    """True if this box runs off a tile edge that is not also a photo edge."""
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    tol_x, tol_y = bw * margin, bh * margin

    if box[0] < -tol_x and tile.x > 0:
        return True
    if box[1] < -tol_y and tile.y > 0:
        return True
    if box[2] > tile.w + tol_x and (tile.x + tile.w) < image_w:
        return True
    if box[3] > tile.h + tol_y and (tile.y + tile.h) < image_h:
        return True
    return False
