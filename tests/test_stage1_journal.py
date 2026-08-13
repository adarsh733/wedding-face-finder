"""The append-only journals, and what happens when a job dies mid-photo.

A 4,000-photo job takes 4-6 hours. Crashes are not hypothetical. The rule under
test: a photo counts as done only once its line lands in state.jsonl, and
anything written by an attempt that died before that point is ignored.

These tests write the journals by hand rather than running the real processor,
so they need no model and no photos.
"""
from __future__ import annotations

import json

import numpy as np

from wff.config import Config
from wff.stage1 import (
    EMBEDDING_ROW_BYTES,
    EventPaths,
    _append_embeddings,
    consolidate,
    read_done_photo_ids,
)
from wff.storage import LocalObjectStore


def _config(tmp_path) -> Config:
    config = Config()
    config.storage.local_root = str(tmp_path)
    config.storage.bucket = "wff"
    return config


def _write_face(store, paths, photo_id, face_index, batch, embedding_ref, config):
    line = json.dumps(
        {
            "batch": batch,
            "face_id": f"{photo_id}_{face_index:02d}",
            "photo_id": photo_id,
            "photo_path": f"{photo_id}.jpg",
            "sub_album": "album",
            "face_index": face_index,
            "bbox_x": 1.0, "bbox_y": 2.0, "bbox_w": 100.0, "bbox_h": 120.0,
            "det_score": 0.9, "quality_score": 0.8, "blur": 50.0, "yaw": 3.0,
            "yaw_method": "pnp", "face_h_px": 120.0, "align_scale": 0.9,
            "detector_source": "full",
            "accepted": embedding_ref >= 0,
            "second_class": False,
            "reject_reason": "" if embedding_ref >= 0 else "too small",
            "embedding_ref": embedding_ref,
            "embedding_model_version": config.embedding_model_version,
        }
    )
    with store.open_append(paths.faces_jsonl) as fh:
        fh.write((line + "\n").encode())


def _commit_state(store, paths, photo_id, batch, face_count):
    with store.open_append(paths.state) as fh:
        fh.write(
            (
                json.dumps(
                    {
                        "photo_id": photo_id, "status": "done", "batch": batch,
                        "path": f"{photo_id}.jpg", "sub_album": "album",
                        "width": 6000, "height": 4000, "taken_at": None,
                        "exif_rotated": False, "face_count": face_count,
                        "accepted_count": face_count, "detected_count": face_count,
                    }
                )
                + "\n"
            ).encode()
        )


def test_embedding_rows_are_addressed_by_position(tmp_path):
    store = LocalObjectStore(str(tmp_path))
    paths = EventPaths("wff", "ev1")

    first = _append_embeddings(store, paths, [np.ones(512, dtype=np.float32)] * 2)
    second = _append_embeddings(store, paths, [np.zeros(512, dtype=np.float32)] * 3)
    assert first == 0
    assert second == 2


def test_a_torn_write_is_padded_not_misread(tmp_path):
    """If a crash left half a row behind, the next append must not start at a
    fractional row -- every offset already recorded in state.jsonl would shift.
    """
    store = LocalObjectStore(str(tmp_path))
    paths = EventPaths("wff", "ev1")

    _append_embeddings(store, paths, [np.ones(512, dtype=np.float32)])
    with store.open_append(paths.embeddings) as fh:
        fh.write(b"\x01" * 900)          # a row that never finished

    start = _append_embeddings(store, paths, [np.full(512, 7.0, dtype=np.float32)])
    assert start == 2, "the partial row must be padded to a whole row, not overlapped"

    raw = store.get_bytes(paths.embeddings)
    assert len(raw) % EMBEDDING_ROW_BYTES == 0
    written = np.frombuffer(raw, dtype=np.float32).reshape(-1, 512)
    assert np.allclose(written[2], 7.0), "the new row is not where we said it was"


def test_resume_skips_committed_photos(tmp_path):
    store = LocalObjectStore(str(tmp_path))
    paths = EventPaths("wff", "ev1")
    _commit_state(store, paths, "000001", 0, 1)
    _commit_state(store, paths, "000002", 1, 1)

    done = read_done_photo_ids(store, paths)
    assert set(done) == {"000001", "000002"}


def test_a_torn_final_line_is_ignored(tmp_path):
    """Power loss mid-write leaves an unparseable last line. It must not take
    the whole resume down with it."""
    store = LocalObjectStore(str(tmp_path))
    paths = EventPaths("wff", "ev1")
    _commit_state(store, paths, "000001", 0, 1)
    with store.open_append(paths.state) as fh:
        fh.write(b'{"photo_id": "000002", "stat')

    done = read_done_photo_ids(store, paths)
    assert set(done) == {"000001"}


def test_consolidate_drops_faces_from_an_uncommitted_attempt(tmp_path):
    """The crash case: an attempt wrote its faces, then died before state.

    Its rows must never reach faces.parquet -- they would double-count that
    photo's faces and skew every threshold we tune afterwards.
    """
    config = _config(tmp_path)
    store = LocalObjectStore(str(tmp_path))
    paths = EventPaths("wff", "ev1")

    # Attempt 1: two faces written, then the machine died. No state line.
    _append_embeddings(store, paths, [np.ones(512, dtype=np.float32)] * 2)
    _write_face(store, paths, "000001", 0, 0, 0, config)
    _write_face(store, paths, "000001", 1, 0, 1, config)

    # Attempt 2 after resume: same photo, committed properly this time.
    start = _append_embeddings(store, paths, [np.full(512, 0.5, dtype=np.float32)] * 2)
    _write_face(store, paths, "000001", 0, start, start, config)
    _write_face(store, paths, "000001", 1, start, start + 1, config)
    _commit_state(store, paths, "000001", start, 2)

    face_count, photo_count = consolidate("ev1", store, config)
    assert face_count == 2, "the dead attempt's faces leaked into the cache"
    assert photo_count == 1


def test_consolidate_dedupes_a_zero_embedding_retry(tmp_path):
    """A photo whose faces were ALL rejected writes no embedding rows, so a
    retry lands on the same batch number and its orphans look legitimate.
    Dedupe by face_id is what catches that."""
    config = _config(tmp_path)
    store = LocalObjectStore(str(tmp_path))
    paths = EventPaths("wff", "ev1")

    batch = _append_embeddings(store, paths, [])          # attempt 1, no faces kept
    _write_face(store, paths, "000001", 0, batch, -1, config)

    batch2 = _append_embeddings(store, paths, [])         # attempt 2, same batch
    assert batch2 == batch
    _write_face(store, paths, "000001", 0, batch2, -1, config)
    _commit_state(store, paths, "000001", batch2, 1)

    face_count, _ = consolidate("ev1", store, config)
    assert face_count == 1, "the same face was written to the cache twice"


def test_consolidated_parquet_round_trips(tmp_path):
    """Regression: nulls inside a fixed_size_list write fine and then cannot be
    read back. Rejected faces carry an empty embedding list instead."""
    config = _config(tmp_path)
    store = LocalObjectStore(str(tmp_path))
    paths = EventPaths("wff", "ev1")

    start = _append_embeddings(store, paths, [np.ones(512, dtype=np.float32)])
    _write_face(store, paths, "000001", 0, start, start, config)   # accepted
    _write_face(store, paths, "000001", 1, start, -1, config)      # rejected
    _commit_state(store, paths, "000001", start, 2)

    consolidate("ev1", store, config)

    from wff.stage2 import load_faces

    faces = load_faces("ev1", store, config)
    assert len(faces) == 1, "only the accepted face should survive into stage 2"
    assert faces.rejected_count == 1
    assert faces.embeddings.shape == (1, 512)


def test_model_version_mismatch_is_filtered_out(tmp_path):
    """Embeddings from two model versions are mutually unreadable. Mixing them
    does not crash -- guests just silently stop finding their photos."""
    config = _config(tmp_path)
    store = LocalObjectStore(str(tmp_path))
    paths = EventPaths("wff", "ev1")

    start = _append_embeddings(store, paths, [np.ones(512, dtype=np.float32)] * 2)
    _write_face(store, paths, "000001", 0, start, start, config)

    stale = Config()
    stale.embedding_model_version = "some-older-model/v0"
    _write_face(store, paths, "000001", 1, start, start + 1, stale)
    _commit_state(store, paths, "000001", start, 2)

    consolidate("ev1", store, config)

    from wff.stage2 import load_faces

    faces = load_faces("ev1", store, config)
    assert len(faces) == 1
    assert faces.version_mismatch_count == 1
