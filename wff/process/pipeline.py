"""One photo, end to end. The expensive step, run once per photo, ever.

    stream photo from source  --->  in memory, never saved to disk
           |
           +- 1. fix rotation (EXIF)
           +- 2. find faces -- whole image + native-resolution tiles
           +- 3. align each face -- eyes to fixed positions
           +- 4. score quality -- size, sharpness, angle
           +- 5. fingerprint -- 112x112 crop -> 512 numbers
           +- 6. make 2048px preview + 400px thumbnail
           +- 7. return the small stuff, DISCARD the original

The ordering is the entire correction from Draft 1. Detect and fingerprint at
native resolution FIRST; shrink only afterwards, for delivery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import BinaryIO

import numpy as np

from ..config import Config
from .align import align_face
from .derivatives import encode_face_crop, make_derivatives
from .detect import Detection, SCRFDDetector
from .embed import ArcFaceEmbedder
from .loader import configure_decoder, load_image
from .models import ensure_models
from .quality import FaceQuality, score_face


@dataclass
class ProcessedFace:
    index: int
    detection: Detection
    quality: FaceQuality
    embedding: np.ndarray | None = None   # unit-length (512,), accepted faces only
    crop_jpeg: bytes | None = None
    align_scale: float = 0.0


@dataclass
class ProcessedPhoto:
    photo_id: str
    path: str
    sub_album: str
    width: int = 0
    height: int = 0
    taken_at: datetime | None = None
    exif_rotated: bool = False
    faces: list[ProcessedFace] = field(default_factory=list)
    detected_count: int = 0
    preview_jpeg: bytes | None = None
    thumb_jpeg: bytes | None = None
    error: str | None = None

    @property
    def accepted_faces(self) -> list[ProcessedFace]:
        return [f for f in self.faces if f.quality.accepted]


class PhotoProcessor:
    """Holds the two ONNX sessions. Build once, reuse for the whole job."""

    def __init__(self, config: Config, quiet: bool = False) -> None:
        self.config = config
        # Process-wide and cheap, so it belongs here rather than per photo.
        self.decode_threads = configure_decoder(config.decode_threads)
        paths = ensure_models(config.model_dir, quiet=quiet)
        self.detector = SCRFDDetector(paths["det_10g.onnx"], config.num_threads)
        self.embedder = ArcFaceEmbedder(paths["w600k_r50.onnx"], config.num_threads)

    def process(
        self, stream: BinaryIO, photo_id: str, path: str, sub_album: str
    ) -> ProcessedPhoto:
        result = ProcessedPhoto(photo_id=photo_id, path=path, sub_album=sub_album)
        try:
            loaded = load_image(stream)
        except Exception as exc:
            result.error = f"decode failed: {type(exc).__name__}: {exc}"
            return result

        result.width = loaded.width
        result.height = loaded.height
        result.taken_at = loaded.taken_at
        result.exif_rotated = loaded.exif_rotated
        image = loaded.bgr

        try:
            detections = self.detector.detect(image, self.config.detect)
            result.detected_count = len(detections)

            # Align and gate first, then embed the survivors in ONE batch.
            # Embedding is the expensive call; batching it is most of the
            # difference between a 4-hour job and a 7-hour one.
            faces: list[ProcessedFace] = []
            to_embed: list[np.ndarray] = []
            embed_targets: list[ProcessedFace] = []

            for index, detection in enumerate(detections):
                aligned, scale = align_face(
                    image, detection.kps, self.config.derivatives.face_crop_px
                )
                if aligned is None:
                    continue
                quality = score_face(
                    detection, aligned, loaded.width, loaded.height, self.config.quality
                )
                face = ProcessedFace(
                    index=index,
                    detection=detection,
                    quality=quality,
                    align_scale=scale,
                )
                if self.config.derivatives.save_face_crops:
                    face.crop_jpeg = encode_face_crop(aligned)
                if quality.accepted:
                    to_embed.append(aligned)
                    embed_targets.append(face)
                faces.append(face)

            if to_embed:
                embeddings = self.embedder.embed_batch(to_embed)
                for face, embedding in zip(embed_targets, embeddings):
                    face.embedding = embedding

            result.faces = faces

            # ONLY NOW do we make the small copies.
            preview, thumb = make_derivatives(image, self.config.derivatives)
            result.preview_jpeg = preview
            result.thumb_jpeg = thumb
        except Exception as exc:
            result.error = f"processing failed: {type(exc).__name__}: {exc}"
        finally:
            # Release the 72 MB array before the next photo is read.
            del image
            loaded.bgr = np.empty((0, 0, 3), dtype=np.uint8)

        return result
