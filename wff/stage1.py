"""STAGE 1 -- expensive, runs ONCE per wedding, cached to disk.

    photos -> manifest -> detect (tiled, native res) -> quality gate
           -> 512-number fingerprints -> faces.parquet

Everything here is written to be RESUMABLE. A 4,000-photo job takes 4-6 hours on
a desktop CPU. If the machine dies at photo 2,800, we restart and skip the 2,800
already done. Without that, a crash means starting over, and on a 4-hour job
that is the difference between a product and a toy.

The crash-safety mechanism is three append-only journals plus one rule: a photo
counts as done only once its line lands in state.jsonl, which happens LAST.
Anything written by an attempt that died before that point is orphaned and
ignored at consolidation time -- never deleted, never trusted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterator

import numpy as np

from .config import Config, load_config
from .ingest.base import FileRef, IngestAdapter, SourceStats
from .process.embed import EMBEDDING_DIM
from .process.pipeline import PhotoProcessor, ProcessedPhoto
from .storage import ObjectStore, get_store, join_uri

EMBEDDING_ROW_BYTES = EMBEDDING_DIM * 4  # float32 on disk, float16 in the parquet


# ---------------------------------------------------------------------------
# Where everything lives. Object URIs only.
# ---------------------------------------------------------------------------


class EventPaths:
    def __init__(self, bucket: str, event_id: str) -> None:
        self.event_id = event_id
        self.root = f"r2://{bucket}/events/{event_id}"

    @property
    def event_json(self) -> str:
        return join_uri(self.root, "event.json")

    @property
    def manifest(self) -> str:
        return join_uri(self.root, "manifest.jsonl")

    @property
    def state(self) -> str:
        return join_uri(self.root, "state.jsonl")

    @property
    def faces_jsonl(self) -> str:
        return join_uri(self.root, "faces.jsonl")

    @property
    def embeddings(self) -> str:
        return join_uri(self.root, "embeddings.f32")

    @property
    def faces_parquet(self) -> str:
        return join_uri(self.root, "faces.parquet")

    @property
    def photos_parquet(self) -> str:
        return join_uri(self.root, "photos.parquet")

    def preview(self, photo_id: str) -> str:
        return join_uri(self.root, "prev", f"{photo_id}.jpg")

    def thumb(self, photo_id: str) -> str:
        return join_uri(self.root, "thumb", f"{photo_id}.jpg")

    def crop(self, photo_id: str, face_index: int) -> str:
        return join_uri(self.root, "crops", f"{photo_id}_{face_index:02d}.jpg")


# ---------------------------------------------------------------------------
# Step 2 -- Manifest
# ---------------------------------------------------------------------------


@dataclass
class ManifestEntry:
    photo_id: str
    source_id: str
    name: str
    path: str
    sub_album: str
    size: int | None = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "ManifestEntry":
        return ManifestEntry(**json.loads(line))


def build_manifest(
    adapter: IngestAdapter,
    event_id: str,
    store: ObjectStore,
    config: Config | None = None,
) -> tuple[list[ManifestEntry], SourceStats]:
    """Walk the whole link and write down what is there, before any real work.

    Also the moment we capture source_folder_id and each source_file_id, which
    is what makes "download the true original" possible at zero storage cost.
    """
    cfg = config or load_config()
    paths = EventPaths(cfg.storage.bucket, event_id)

    refs, stats = adapter.scan()
    entries = [
        ManifestEntry(
            photo_id=f"{index:06d}",
            source_id=ref.source_id,
            name=ref.name,
            path=ref.path,
            sub_album=ref.sub_album,
            size=ref.size,
        )
        for index, ref in enumerate(refs)
    ]

    store.put_text(
        paths.manifest, "\n".join(entry.to_json() for entry in entries) + "\n"
    )
    store.put_text(
        paths.event_json,
        json.dumps(
            {
                "event_id": event_id,
                "source_type": adapter.source_type,
                "source_folder_id": adapter.source_folder_id(),
                "embedding_model_version": cfg.embedding_model_version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "photo_count": len(entries),
                "skipped": {
                    "raw": stats.raw,
                    "videos": stats.videos,
                    "other": stats.other,
                },
                "sub_albums": stats.sub_albums,
                "total_source_bytes": stats.total_bytes,
                "config": cfg.to_dict(),
            },
            indent=2,
        ),
        "application/json",
    )
    return entries, stats


def read_manifest(store: ObjectStore, paths: EventPaths) -> list[ManifestEntry]:
    text = store.get_text(paths.manifest)
    return [ManifestEntry.from_json(line) for line in text.splitlines() if line.strip()]


def read_done_photo_ids(store: ObjectStore, paths: EventPaths) -> dict[str, dict]:
    """photo_id -> the winning state record. Later lines win on retry."""
    if not store.exists(paths.state):
        return {}
    done: dict[str, dict] = {}
    for line in store.get_text(paths.state).splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn last line from a crash -- ignore it
        done[record["photo_id"]] = record
    return done


# ---------------------------------------------------------------------------
# Step 3 -- Process
# ---------------------------------------------------------------------------


@dataclass
class Stage1Summary:
    event_id: str
    photos_total: int = 0
    photos_processed: int = 0
    photos_skipped_done: int = 0
    photos_failed: int = 0
    faces_detected: int = 0
    faces_accepted: int = 0
    faces_second_class: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    stopped_early: bool = False


def _append_embeddings(
    store: ObjectStore, paths: EventPaths, embeddings: list[np.ndarray]
) -> int:
    """Append rows, return the starting row index. Aligns past torn writes."""
    with store.open_append(paths.embeddings) as fh:
        position = fh.tell()
        remainder = position % EMBEDDING_ROW_BYTES
        if remainder:
            # A previous attempt died mid-row. Pad forward rather than truncate,
            # so the offsets already recorded in state.jsonl stay valid.
            fh.write(b"\x00" * (EMBEDDING_ROW_BYTES - remainder))
            position += EMBEDDING_ROW_BYTES - remainder
        start_row = position // EMBEDDING_ROW_BYTES
        if embeddings:
            block = np.stack(embeddings).astype(np.float32)
            fh.write(block.tobytes())
        fh.flush()
    return start_row


def process_event(
    adapter: IngestAdapter,
    event_id: str,
    store: ObjectStore | None = None,
    config: Config | None = None,
    limit: int | None = None,
    resume: bool = True,
    progress: bool = True,
    should_stop: "Callable[[], bool] | None" = None,
) -> Stage1Summary:
    """Run Stage 1 over an already-built manifest.

    `should_stop` is checked between photos. A stop leaves the journals valid
    and the job resumable -- exactly like a crash, which this code is already
    built to survive. It exists so the Stop button in the testing console does
    not have to kill the process.
    """
    import time

    cfg = config or load_config()
    obj_store = store or get_store(cfg)
    paths = EventPaths(cfg.storage.bucket, event_id)

    entries = read_manifest(obj_store, paths)
    done = read_done_photo_ids(obj_store, paths) if resume else {}
    pending = [e for e in entries if e.photo_id not in done]
    if limit is not None:
        pending = pending[:limit]

    summary = Stage1Summary(
        event_id=event_id,
        photos_total=len(entries),
        photos_skipped_done=len(entries) - len([e for e in entries if e.photo_id not in done]),
    )
    if not pending:
        return summary

    processor = PhotoProcessor(cfg, quiet=not progress)

    iterator: Iterator[ManifestEntry] = iter(pending)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(pending, unit="photo", desc=f"stage1 {event_id}")
        except ImportError:
            pass

    started = time.time()
    for entry in iterator:
        if should_stop is not None and should_stop():
            summary.stopped_early = True
            break
        ref = FileRef(
            source_id=entry.source_id,
            name=entry.name,
            path=entry.path,
            size=entry.size,
        )
        try:
            with adapter.open(ref) as stream:
                result = processor.process(
                    stream, entry.photo_id, entry.path, entry.sub_album
                )
        except Exception as exc:
            result = ProcessedPhoto(
                photo_id=entry.photo_id,
                path=entry.path,
                sub_album=entry.sub_album,
                error=f"read failed: {type(exc).__name__}: {exc}",
            )

        _commit_photo(obj_store, paths, cfg, result, summary)

    summary.elapsed_seconds = time.time() - started
    return summary


def _commit_photo(
    store: ObjectStore,
    paths: EventPaths,
    cfg: Config,
    result: ProcessedPhoto,
    summary: Stage1Summary,
) -> None:
    """Write one photo's output. state.jsonl LAST -- that is what makes it done."""
    if result.error:
        summary.photos_failed += 1
        with store.open_append(paths.state) as fh:
            fh.write(
                (
                    json.dumps(
                        {
                            "photo_id": result.photo_id,
                            "status": "failed",
                            "error": result.error,
                            "path": result.path,
                        }
                    )
                    + "\n"
                ).encode("utf-8")
            )
        return

    accepted = result.accepted_faces
    embeddings = [f.embedding for f in accepted if f.embedding is not None]
    start_row = _append_embeddings(store, paths, embeddings)

    # Derivatives and crops. Idempotent: a re-run overwrites the same keys.
    if result.preview_jpeg:
        store.put_bytes(paths.preview(result.photo_id), result.preview_jpeg, "image/jpeg")
    if result.thumb_jpeg:
        store.put_bytes(paths.thumb(result.photo_id), result.thumb_jpeg, "image/jpeg")

    face_lines: list[str] = []
    embedding_cursor = 0
    for face in result.faces:
        quality = face.quality
        if face.crop_jpeg:
            store.put_bytes(
                paths.crop(result.photo_id, face.index), face.crop_jpeg, "image/jpeg"
            )
        embedding_ref = -1
        if face.embedding is not None:
            embedding_ref = start_row + embedding_cursor
            embedding_cursor += 1

        bbox = face.detection.bbox
        face_lines.append(
            json.dumps(
                {
                    # batch key -- ties this line to one attempt at this photo,
                    # so orphans from a crashed attempt are ignorable
                    "batch": start_row,
                    "face_id": f"{result.photo_id}_{face.index:02d}",
                    "photo_id": result.photo_id,
                    "photo_path": result.path,
                    "sub_album": result.sub_album,
                    "face_index": face.index,
                    "bbox_x": round(float(bbox[0]), 2),
                    "bbox_y": round(float(bbox[1]), 2),
                    "bbox_w": round(float(bbox[2] - bbox[0]), 2),
                    "bbox_h": round(float(bbox[3] - bbox[1]), 2),
                    "det_score": round(quality.det_score, 4),
                    "quality_score": round(quality.quality_score, 4),
                    "blur": round(quality.blur_var, 2),
                    "yaw": round(quality.yaw_deg, 2),
                    "yaw_method": quality.yaw_method,
                    "face_h_px": round(quality.face_h_px, 1),
                    "align_scale": round(face.align_scale, 4),
                    "detector_source": face.detection.source,
                    "accepted": quality.accepted,
                    "second_class": quality.second_class,
                    "reject_reason": quality.reject_reason,
                    "embedding_ref": embedding_ref,
                    "embedding_model_version": cfg.embedding_model_version,
                }
            )
        )
        summary.faces_detected += 1
        if quality.accepted:
            summary.faces_accepted += 1
            if quality.second_class:
                summary.faces_second_class += 1
        else:
            key = quality.reject_reason.split(" (")[0]
            summary.reject_reasons[key] = summary.reject_reasons.get(key, 0) + 1

    if face_lines:
        with store.open_append(paths.faces_jsonl) as fh:
            fh.write(("\n".join(face_lines) + "\n").encode("utf-8"))

    with store.open_append(paths.state) as fh:
        fh.write(
            (
                json.dumps(
                    {
                        "photo_id": result.photo_id,
                        "status": "done",
                        "batch": start_row,
                        "path": result.path,
                        "sub_album": result.sub_album,
                        "width": result.width,
                        "height": result.height,
                        "taken_at": result.taken_at.isoformat() if result.taken_at else None,
                        "exif_rotated": result.exif_rotated,
                        "face_count": len(result.faces),
                        "accepted_count": len(accepted),
                        "detected_count": result.detected_count,
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
    summary.photos_processed += 1


# ---------------------------------------------------------------------------
# Consolidation -- the journals become the cache Stage 2 reads
# ---------------------------------------------------------------------------


def consolidate(
    event_id: str, store: ObjectStore | None = None, config: Config | None = None
) -> tuple[int, int]:
    """Turn the append-only journals into faces.parquet + photos.parquet.

    This is where orphaned records from crashed attempts get dropped: a face
    line survives only if its `batch` matches the batch its photo committed.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = config or load_config()
    obj_store = store or get_store(cfg)
    paths = EventPaths(cfg.storage.bucket, event_id)

    state = read_done_photo_ids(obj_store, paths)
    winning_batch = {
        photo_id: record.get("batch")
        for photo_id, record in state.items()
        if record.get("status") == "done"
    }

    # face_id -> record, last write wins.
    #
    # The batch key alone is not quite enough: a photo whose faces were ALL
    # rejected writes no embedding rows, so a retry of it lands on the same
    # start_row and its orphaned lines look legitimate. Keying by face_id
    # (photo_id + face_index, stable across attempts) collapses those, and
    # dict ordering keeps the newest attempt.
    by_face_id: dict[str, dict] = {}
    if obj_store.exists(paths.faces_jsonl):
        for line in obj_store.get_text(paths.faces_jsonl).splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if winning_batch.get(record["photo_id"]) != record.get("batch"):
                continue  # orphan from an attempt that never committed
            by_face_id[record["face_id"]] = record
    faces = list(by_face_id.values())

    # Load the embeddings the surviving faces point at.
    embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    if obj_store.exists(paths.embeddings):
        raw = obj_store.get_bytes(paths.embeddings)
        usable = len(raw) - (len(raw) % EMBEDDING_ROW_BYTES)
        embeddings = np.frombuffer(raw[:usable], dtype=np.float32).reshape(
            -1, EMBEDDING_DIM
        )

    # An empty list, not a null, for a face with no embedding. pyarrow writes a
    # null inside a fixed_size_list happily and then cannot read it back
    # ("Expected all lists to be of size=512 but index N had size=0"), so the
    # column is variable-length and "no embedding" is honestly an empty list.
    # Accepted faces all carry exactly 512 values, so stacking them is still a
    # dense matrix.
    vectors: list[list[float]] = []
    for record in faces:
        ref = record.get("embedding_ref", -1)
        if ref is None or ref < 0 or ref >= len(embeddings):
            record["embedding_ref"] = -1
            vectors.append([])
        else:
            # float16 in the parquet: 33 MB per wedding instead of 66 MB, and
            # the precision loss is far below the noise in the embedding itself.
            vectors.append(embeddings[ref].astype(np.float16).tolist())

    columns = [
        "batch", "face_id", "photo_id", "photo_path", "sub_album", "face_index",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "det_score", "quality_score",
        "blur", "yaw", "yaw_method", "face_h_px", "align_scale", "detector_source",
        "accepted", "second_class", "reject_reason", "embedding_ref",
        "embedding_model_version",
    ]
    table_data = {name: [record.get(name) for record in faces] for name in columns}
    table_data["event_id"] = [event_id] * len(faces)
    table_data["embedding"] = vectors

    schema = pa.schema(
        [
            ("event_id", pa.string()), ("face_id", pa.string()), ("photo_id", pa.string()),
            ("photo_path", pa.string()), ("sub_album", pa.string()),
            ("face_index", pa.int32()), ("batch", pa.int64()),
            ("bbox_x", pa.float32()), ("bbox_y", pa.float32()),
            ("bbox_w", pa.float32()), ("bbox_h", pa.float32()),
            ("det_score", pa.float32()), ("quality_score", pa.float32()),
            ("blur", pa.float32()), ("yaw", pa.float32()), ("yaw_method", pa.string()),
            ("face_h_px", pa.float32()), ("align_scale", pa.float32()),
            ("detector_source", pa.string()), ("accepted", pa.bool_()),
            ("second_class", pa.bool_()), ("reject_reason", pa.string()),
            ("embedding_ref", pa.int64()),
            ("embedding_model_version", pa.string()),
            ("embedding", pa.list_(pa.float16())),
        ]
    )
    faces_table = pa.table(
        {name: table_data[name] for name in schema.names}, schema=schema
    )
    buffer = pa.BufferOutputStream()
    pq.write_table(faces_table, buffer, compression="zstd")
    obj_store.put_bytes(paths.faces_parquet, buffer.getvalue().to_pybytes())

    photos = [record for record in state.values() if record.get("status") == "done"]
    photo_schema = pa.schema(
        [
            ("event_id", pa.string()), ("photo_id", pa.string()), ("path", pa.string()),
            ("sub_album", pa.string()), ("width", pa.int32()), ("height", pa.int32()),
            ("taken_at", pa.string()), ("exif_rotated", pa.bool_()),
            ("face_count", pa.int32()), ("accepted_count", pa.int32()),
            ("detected_count", pa.int32()),
        ]
    )
    photos_table = pa.table(
        {
            "event_id": [event_id] * len(photos),
            "photo_id": [p["photo_id"] for p in photos],
            "path": [p.get("path") for p in photos],
            "sub_album": [p.get("sub_album") for p in photos],
            "width": [p.get("width") for p in photos],
            "height": [p.get("height") for p in photos],
            "taken_at": [p.get("taken_at") for p in photos],
            "exif_rotated": [p.get("exif_rotated") for p in photos],
            "face_count": [p.get("face_count") for p in photos],
            "accepted_count": [p.get("accepted_count") for p in photos],
            "detected_count": [p.get("detected_count") for p in photos],
        },
        schema=photo_schema,
    )
    buffer = pa.BufferOutputStream()
    pq.write_table(photos_table, buffer, compression="zstd")
    obj_store.put_bytes(paths.photos_parquet, buffer.getvalue().to_pybytes())

    return len(faces), len(photos)
