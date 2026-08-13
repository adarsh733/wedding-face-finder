"""The gate. Four numbers per face, and a verdict.

STEP 3.4. Rejecting a good face costs one photo. Admitting a bad one poisons a
cluster and shows a guest a stranger. The gate is deliberately unbalanced in
favour of rejecting.

The 50-80px band is kept but marked SECOND CLASS: those faces may JOIN an
existing person, they may never START one. A person invented out of blurry
back-row faces is precisely how "everyone's blurry face lands near everyone
else's" becomes a wrong match.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import QualityConfig
from .detect import Detection

# A generic 5-point 3D face, in millimetres, for the pose solve. Coordinates
# are the conventional OpenCV camera frame: +x right, +y down, +z away.
_MODEL_POINTS_3D = np.array(
    [
        [-33.0, -32.0, -28.0],   # left eye centre
        [33.0, -32.0, -28.0],    # right eye centre
        [0.0, 0.0, 0.0],         # nose tip
        [-28.0, 35.0, -24.0],    # left mouth corner
        [28.0, 35.0, -24.0],     # right mouth corner
    ],
    dtype=np.float64,
)


@dataclass
class FaceQuality:
    face_h_px: float        # height in the ORIGINAL photo -- the number that matters
    det_score: float
    blur_var: float
    yaw_deg: float
    yaw_method: str         # 'pnp' or 'landmark-ratio'
    quality_score: float    # 0..1, only for ranking (cover photo, best-face)
    accepted: bool
    second_class: bool
    reject_reason: str = ""


def estimate_yaw(kps: np.ndarray, image_w: int, image_h: int) -> tuple[float, str]:
    """Head turn in degrees. Positive and negative both mean "turned away".

    solvePnP against a generic 3D face with an assumed focal length. It is an
    estimate, not a measurement -- we only ever threshold it at ~50 degrees, and
    it is comfortably good enough for that.
    """
    image_points = np.asarray(kps, dtype=np.float64).reshape(5, 2)
    focal = float(max(image_w, image_h))
    camera = np.array(
        [[focal, 0, image_w / 2.0], [0, focal, image_h / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )
    try:
        ok, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            camera,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_EPNP,
        )
        if ok:
            rotation, _ = cv2.Rodrigues(rvec)
            # Yaw about the vertical axis.
            sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
            if sy > 1e-6:
                yaw = math.degrees(math.atan2(-rotation[2, 0], sy))
                if abs(yaw) <= 90.0:
                    return yaw, "pnp"
    except cv2.error:
        pass

    # Fallback: where the nose sits between the eyes. 0.5 is frontal.
    left_eye, right_eye, nose = image_points[0], image_points[1], image_points[2]
    span = right_eye[0] - left_eye[0]
    if abs(span) < 1e-6:
        return 90.0, "landmark-ratio"
    ratio = (nose[0] - left_eye[0]) / span
    return float((ratio - 0.5) * 180.0), "landmark-ratio"


def blur_variance(aligned_bgr: np.ndarray) -> float:
    """Variance of the Laplacian on the ALIGNED crop.

    Measured after alignment on purpose: the metric is scale-dependent, and
    after alignment every face is exactly the same size, so the numbers are
    comparable across a 1500px close-up and a 60px back-row face.
    """
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_face(
    detection: Detection,
    aligned_bgr: np.ndarray,
    image_w: int,
    image_h: int,
    config: QualityConfig,
) -> FaceQuality:
    face_h = detection.height_px
    det_score = detection.score
    blur = blur_variance(aligned_bgr)
    yaw, yaw_method = estimate_yaw(detection.kps, image_w, image_h)

    aspect = detection.width_px / max(face_h, 1e-6)

    reject = ""
    if face_h < config.min_face_px:
        reject = f"too small ({face_h:.0f}px < {config.min_face_px}px)"
    elif not (config.min_aspect_ratio <= aspect <= config.max_aspect_ratio):
        # A box far from square is not a face. Seen on real photos as a narrow
        # vertical sliver of a real face, which then embeds as a second person.
        reject = f"implausible shape (w/h {aspect:.2f})"
    elif det_score < config.min_det_score:
        reject = f"low confidence ({det_score:.2f} < {config.min_det_score})"
    elif blur < config.min_blur_var:
        reject = f"too blurry (laplacian var {blur:.1f} < {config.min_blur_var})"
    elif abs(yaw) > config.max_yaw_deg:
        reject = f"turned away ({abs(yaw):.0f} deg > {config.max_yaw_deg} deg)"

    accepted = reject == ""
    second_class = accepted and face_h < config.second_class_face_px

    # Ranking only -- never a gate. Used to pick a person's cover tile and the
    # best face per photo.
    size_term = min(1.0, face_h / 200.0)
    blur_term = min(1.0, blur / 200.0)
    yaw_term = max(0.0, 1.0 - abs(yaw) / 90.0)
    quality = 0.40 * size_term + 0.25 * blur_term + 0.20 * yaw_term + 0.15 * det_score

    return FaceQuality(
        face_h_px=face_h,
        det_score=det_score,
        blur_var=blur,
        yaw_deg=yaw,
        yaw_method=yaw_method,
        quality_score=float(quality),
        accepted=accepted,
        second_class=second_class,
        reject_reason=reject,
    )
