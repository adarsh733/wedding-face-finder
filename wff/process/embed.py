"""ArcFace: an aligned 112x112 face in, 512 numbers out.

STEP 3.5. The numbers are then scaled so every fingerprint sits on a sphere of
the same size, which turns "how similar are these two faces?" into a single
dot product.

Everything downstream assumes unit length. Cosine distance is 1 - dot, and the
clustering thresholds are calibrated against that. Return an un-normalised
vector from here and every threshold in the system silently means something
else.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import make_session

EMBEDDING_DIM = 512


class ArcFaceEmbedder:
    def __init__(self, model_path: Path, num_threads: int = 0) -> None:
        self.session = make_session(model_path, num_threads)
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        self.output_name = self.session.get_outputs()[0].name
        shape = inputs[0].shape
        # NCHW; dims may be strings when the export is dynamic.
        self.input_size = int(shape[2]) if isinstance(shape[2], int) else 112

    def embed_batch(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        """(N, 512) float32, L2-normalised. Empty input -> (0, 512)."""
        if not aligned_faces:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        blob = cv2.dnn.blobFromImages(
            aligned_faces,
            1.0 / 127.5,
            (self.input_size, self.input_size),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        raw = self.session.run([self.output_name], {self.input_name: blob})[0]
        embeddings = np.asarray(raw, dtype=np.float32).reshape(len(aligned_faces), -1)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # A zero vector would become NaN and quietly poison every distance it
        # touches. It should never happen; if it does we want a zero row, which
        # is at least detectable.
        norms = np.where(norms < 1e-10, 1.0, norms)
        return (embeddings / norms).astype(np.float32)

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        return self.embed_batch([aligned_face])[0]
