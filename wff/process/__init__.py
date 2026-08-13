"""Stage 1 per-photo processing.

Order matters more than anything else in this package. See pipeline.py.
"""
from .align import ARCFACE_TEMPLATE_112, align_face
from .derivatives import encode_face_crop, make_derivatives
from .detect import Detection, SCRFDDetector, Tile, nms, plan_tiles
from .embed import EMBEDDING_DIM, ArcFaceEmbedder
from .loader import LoadedImage, load_image
from .models import ensure_models, model_path
from .pipeline import PhotoProcessor, ProcessedFace, ProcessedPhoto
from .quality import FaceQuality, blur_variance, estimate_yaw, score_face

__all__ = [
    "ARCFACE_TEMPLATE_112",
    "align_face",
    "encode_face_crop",
    "make_derivatives",
    "Detection",
    "SCRFDDetector",
    "Tile",
    "nms",
    "plan_tiles",
    "EMBEDDING_DIM",
    "ArcFaceEmbedder",
    "LoadedImage",
    "load_image",
    "ensure_models",
    "model_path",
    "PhotoProcessor",
    "ProcessedFace",
    "ProcessedPhoto",
    "FaceQuality",
    "blur_variance",
    "estimate_yaw",
    "score_face",
]
