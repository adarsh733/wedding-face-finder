"""STAGE 2 runner -- reads the Stage 1 cache, groups faces, scores the result.

Finishes in seconds, so a hundred tuning iterations a day on a laptop is normal.
This split is what makes "no GPU" a non-issue.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .cluster.metrics import Evaluation, SanityReport, evaluate, sanity_check
from .cluster.two_pass import ClusterResult, cluster_faces
from .config import Config, load_config
from .stage1 import EventPaths
from .storage import ObjectStore, get_store


@dataclass
class FaceTable:
    """The accepted faces of one event, ready to cluster."""

    face_ids: list[str]
    photo_ids: list[str]
    photo_paths: list[str]
    sub_albums: list[str]
    face_indices: list[int]
    quality: np.ndarray
    second_class: np.ndarray
    face_h_px: np.ndarray
    embeddings: np.ndarray
    rejected_count: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    version_mismatch_count: int = 0

    def __len__(self) -> int:
        return len(self.face_ids)


def load_faces(
    event_id: str,
    store: ObjectStore | None = None,
    config: Config | None = None,
) -> FaceTable:
    import pyarrow.parquet as pq

    cfg = config or load_config()
    obj_store = store or get_store(cfg)
    paths = EventPaths(cfg.storage.bucket, event_id)

    if not obj_store.exists(paths.faces_parquet):
        raise FileNotFoundError(
            f"No faces.parquet for event {event_id!r}. Run stage 1 first."
        )

    import io

    table = pq.read_table(io.BytesIO(obj_store.get_bytes(paths.faces_parquet)))
    data = table.to_pydict()

    accepted_mask = data["accepted"]
    versions = data["embedding_model_version"]

    keep: list[int] = []
    rejected = 0
    reasons: dict[str, int] = {}
    mismatched = 0
    for index, is_accepted in enumerate(accepted_mask):
        if not is_accepted:
            rejected += 1
            reason = (data["reject_reason"][index] or "unknown").split(" (")[0]
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        # THE MODEL VERSION STAMP. Fingerprints from two model versions are
        # mutually unreadable -- like some measurements in centimetres and some
        # in inches with no label. Nothing crashes; guests just silently stop
        # finding their photos. Every read filters on it.
        if versions[index] != cfg.embedding_model_version:
            mismatched += 1
            continue
        keep.append(index)

    embeddings = np.zeros((len(keep), 512), dtype=np.float32)
    for row, index in enumerate(keep):
        vector = data["embedding"][index]
        # An accepted face always has 512 values. Anything else would be a
        # Stage 1 bug, and a zero row is at least detectable rather than a
        # crash halfway through a tuning run.
        if vector is not None and len(vector) == 512:
            embeddings[row] = np.asarray(vector, dtype=np.float32)

    return FaceTable(
        face_ids=[data["face_id"][i] for i in keep],
        photo_ids=[data["photo_id"][i] for i in keep],
        photo_paths=[data["photo_path"][i] for i in keep],
        sub_albums=[data["sub_album"][i] for i in keep],
        face_indices=[data["face_index"][i] for i in keep],
        quality=np.array([data["quality_score"][i] for i in keep], dtype=np.float32),
        second_class=np.array([data["second_class"][i] for i in keep], dtype=bool),
        face_h_px=np.array([data["face_h_px"][i] for i in keep], dtype=np.float32),
        embeddings=embeddings,
        rejected_count=rejected,
        reject_reasons=reasons,
        version_mismatch_count=mismatched,
    )


@dataclass
class Stage2Output:
    event_id: str
    faces: FaceTable
    result: ClusterResult
    sanity: SanityReport
    photo_count: int
    evaluation: Evaluation | None = None
    ground_truth: list[str] | None = None


def _ground_truth_from_paths(faces: FaceTable) -> list[str] | None:
    """The cricket test set puts one person per folder, so the folder IS the
    answer. Real weddings have no such labels and this returns None.

    A folder layout only counts as ground truth if it looks deliberate: at
    least two folders, and no folder holding almost everything.
    """
    albums = [album or "(root)" for album in faces.sub_albums]
    distinct = set(albums)
    if len(distinct) < 2 or "(root)" in distinct:
        return None
    largest = max(albums.count(name) for name in distinct)
    if largest > 0.9 * len(albums):
        return None
    return albums


def run_stage2(
    event_id: str,
    store: ObjectStore | None = None,
    config: Config | None = None,
    use_ground_truth: bool = True,
) -> Stage2Output:
    cfg = config or load_config()
    obj_store = store or get_store(cfg)
    paths = EventPaths(cfg.storage.bucket, event_id)

    faces = load_faces(event_id, obj_store, cfg)
    result = cluster_faces(
        embeddings=faces.embeddings,
        photo_ids=faces.photo_ids,
        quality_scores=faces.quality,
        second_class=faces.second_class,
        config=cfg.cluster,
    )

    photo_count = len(set(faces.photo_ids))
    if obj_store.exists(paths.event_json):
        try:
            photo_count = json.loads(obj_store.get_text(paths.event_json)).get(
                "photo_count", photo_count
            )
        except (json.JSONDecodeError, KeyError):
            pass

    output = Stage2Output(
        event_id=event_id,
        faces=faces,
        result=result,
        sanity=sanity_check(result, photo_count, cfg.cluster),
        photo_count=photo_count,
    )

    if use_ground_truth:
        truth = _ground_truth_from_paths(faces)
        if truth:
            output.ground_truth = truth
            output.evaluation = evaluate(result.labels, truth)

    return output
