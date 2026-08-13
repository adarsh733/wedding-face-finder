"""Tests for the testing console.

The console's job is to tell the truth about the pipeline. So these tests care
about the ways it could lie: showing a stale grouping after a re-tune, letting a
judgement land on the wrong pile, hiding faces from an "exhaustive" list, or
serving a file it was never meant to reach.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from wff.config import ClusterConfig, Config, StorageConfig
from wff.stage1 import EventPaths
from wff.storage.local import LocalObjectStore
from wff.web import feedback as fb
from wff.web.app import create_app, group_key

EVENT = "ev_webtest"


def _write_parquet(store, config, event_id, faces):
    """Build a minimal faces.parquet the same shape Stage 1 writes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = EventPaths(config.storage.bucket, event_id)
    columns = {
        "event_id": [event_id] * len(faces),
        "face_id": [f["face_id"] for f in faces],
        "photo_id": [f["photo_id"] for f in faces],
        "photo_path": [f.get("photo_path", f["photo_id"] + ".jpg") for f in faces],
        "sub_album": [f.get("sub_album", "") for f in faces],
        "face_index": [int(f["face_id"].split("_")[1]) for f in faces],
        "batch": [0] * len(faces),
        "bbox_x": [10.0] * len(faces),
        "bbox_y": [20.0] * len(faces),
        "bbox_w": [100.0] * len(faces),
        "bbox_h": [120.0] * len(faces),
        "det_score": [0.9] * len(faces),
        "quality_score": [f.get("quality", 0.8) for f in faces],
        "blur": [40.0] * len(faces),
        "yaw": [5.0] * len(faces),
        "yaw_method": ["landmarks"] * len(faces),
        "face_h_px": [float(f.get("face_h_px", 300)) for f in faces],
        "align_scale": [1.0] * len(faces),
        "detector_source": ["whole"] * len(faces),
        "accepted": [f.get("accepted", True) for f in faces],
        "second_class": [f.get("second_class", False) for f in faces],
        "reject_reason": [f.get("reject_reason", "") for f in faces],
        "embedding_ref": list(range(len(faces))),
        "embedding_model_version": [config.embedding_model_version] * len(faces),
        "embedding": [
            np.asarray(f["embedding"], dtype=np.float16).tolist() if f.get("accepted", True) else []
            for f in faces
        ],
    }
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
            ("embedding_ref", pa.int64()), ("embedding_model_version", pa.string()),
            ("embedding", pa.list_(pa.float16())),
        ]
    )
    buffer = pa.BufferOutputStream()
    pq.write_table(pa.table(columns, schema=schema), buffer)
    store.put_bytes(paths.faces_parquet, buffer.getvalue().to_pybytes())

    photo_ids = sorted({f["photo_id"] for f in faces})
    photo_schema = pa.schema(
        [
            ("event_id", pa.string()), ("photo_id", pa.string()), ("path", pa.string()),
            ("sub_album", pa.string()), ("width", pa.int32()), ("height", pa.int32()),
            ("taken_at", pa.string()), ("exif_rotated", pa.bool_()),
            ("face_count", pa.int32()), ("accepted_count", pa.int32()),
            ("detected_count", pa.int32()),
        ]
    )
    buffer = pa.BufferOutputStream()
    pq.write_table(
        pa.table(
            {
                "event_id": [event_id] * len(photo_ids),
                "photo_id": photo_ids,
                "path": [p + ".jpg" for p in photo_ids],
                "sub_album": [""] * len(photo_ids),
                "width": [4000] * len(photo_ids),
                "height": [3000] * len(photo_ids),
                "taken_at": [None] * len(photo_ids),
                "exif_rotated": [False] * len(photo_ids),
                "face_count": [1] * len(photo_ids),
                "accepted_count": [1] * len(photo_ids),
                "detected_count": [1] * len(photo_ids),
            },
            schema=photo_schema,
        ),
        buffer,
    )
    store.put_bytes(paths.photos_parquet, buffer.getvalue().to_pybytes())
    store.put_text(
        paths.event_json,
        json.dumps({"event_id": event_id, "photo_count": len(photo_ids), "source_type": "local_folder"}),
    )
    store.put_text(paths.state, "".join(
        json.dumps({"photo_id": p, "status": "done", "batch": 0}) + "\n" for p in photo_ids
    ))


def _vector(seed: int, drift: float = 0.0) -> np.ndarray:
    """A repeatable unit vector. `drift` walks it away from its base direction."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=512).astype(np.float32)
    if drift:
        noise = np.random.default_rng(seed + 9999).normal(size=512).astype(np.float32)
        base = base / np.linalg.norm(base) + drift * noise / np.linalg.norm(noise)
    return base / np.linalg.norm(base)


@pytest.fixture()
def app(tmp_path):
    config = Config(storage=StorageConfig(backend="local", local_root=str(tmp_path / "store"), bucket="wff"))
    store = LocalObjectStore(config.storage.local_root)

    # Three people, four faces each, in distinct photos so the same-photo rule
    # never fires. Two of them (A and B) are placed close together so a loose
    # threshold merges them -- that is what the tuning tests need.
    faces = []
    for person, seed in enumerate([11, 22, 33]):
        for shot in range(4):
            faces.append(
                {
                    "face_id": f"{person}{shot:05d}_00",
                    "photo_id": f"{person}{shot:05d}",
                    "embedding": _vector(seed, drift=0.15 * shot),
                    "face_h_px": 300 - 60 * shot,
                    "quality": 0.9 - 0.05 * shot,
                }
            )
    faces.append(
        {
            "face_id": "900001_00",
            "photo_id": "900001",
            "embedding": _vector(11, drift=0.9),
            "accepted": True,
            "face_h_px": 70,
            "second_class": True,
            "quality": 0.4,
        }
    )
    faces.append(
        {
            "face_id": "900002_00",
            "photo_id": "900002",
            "embedding": np.zeros(512, dtype=np.float32),
            "accepted": False,
            "reject_reason": "too small (41px)",
        }
    )
    _write_parquet(store, config, EVENT, faces)

    flask_app = create_app(config, store)
    flask_app.config.update(TESTING=True)
    flask_app.config["WFF_TEST_STORE"] = store
    flask_app.config["WFF_CONFIG"] = config
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


# -- the page renders --------------------------------------------------------


def test_home_lists_the_event(client):
    response = client.get("/")
    assert response.status_code == 200
    assert EVENT in response.get_data(as_text=True)


def test_the_run_list_is_newest_first_by_when_it_last_ran(app):
    """Adarsh reads this list top-down every day, so the top of it must be the
    most recent thing that happened.

    The trap is sorting on the ingest date: a folder ingested on Tuesday and
    re-run this morning would sit below folders nobody has touched since. The
    date shown and the date sorted on have to be the same date.
    """
    from wff.runs import RunRecord, append_run
    from wff.web.data import list_events

    store = app.config["WFF_TEST_STORE"]
    config = app.config["WFF_CONFIG"]

    # Three folders ingested in one order...
    for name, created in [
        ("ev_older", "2026-08-01T09:00:00+00:00"),
        ("ev_middle", "2026-08-02T09:00:00+00:00"),
        ("ev_newest_folder", "2026-08-03T09:00:00+00:00"),
    ]:
        store.put_text(
            EventPaths(config.storage.bucket, name).event_json,
            json.dumps({"event_id": name, "created_at": created, "photo_count": 1}),
        )
    # ...and the OLDEST of them re-run most recently.
    append_run(
        store,
        config.storage.bucket,
        RunRecord(
            event_id="ev_older",
            started_at="2026-08-09T10:00:00+00:00",
            finished_at="2026-08-09T10:04:00+00:00",
            total_seconds=240.0,
        ),
    )

    order = [e.event_id for e in list_events(store, config)]
    assert order.index("ev_older") < order.index("ev_newest_folder")

    by_id = {e.event_id: e for e in list_events(store, config)}
    assert by_id["ev_older"].when == "2026-08-09T10:04:00+00:00"   # the run
    assert by_id["ev_middle"].when == "2026-08-02T09:00:00+00:00"  # the ingest

    # And every date on the page descends, with no gaps that skip the sort.
    stamps = [e.when for e in list_events(store, config) if e.when]
    assert stamps == sorted(stamps, reverse=True)


def test_the_newest_run_wins_even_if_the_journal_is_out_of_order(app):
    """Two processes append to runs.jsonl -- the console and the CLI -- so the
    last line is not guaranteed to be the latest run."""
    from wff.runs import RunRecord, append_run
    from wff.web.data import load_event_summary

    store = app.config["WFF_TEST_STORE"]
    config = app.config["WFF_CONFIG"]
    for stamp in ["2026-08-09T10:00:00+00:00", "2026-08-05T10:00:00+00:00"]:
        append_run(
            store,
            config.storage.bucket,
            RunRecord(event_id="ev_outoforder", finished_at=stamp, total_seconds=1.0),
        )
    summary = load_event_summary("ev_outoforder", store, config)
    assert summary.last_run_at == "2026-08-09T10:00:00+00:00"


def test_the_dashboard_reads_each_folder_s_files_once(app, monkeypatch):
    """Every extra read here is paid once per folder, and the folder list only
    grows. This regressed silently once already: the dashboard re-read the run
    journal and the photo parquet that building the summaries had just read, so
    six folders cost twelve of each.

    Counting reads rather than timing them -- a stopwatch on a 6-folder store
    measures the disk cache, not the code.
    """
    from wff.web import data as data_module

    config = app.config["WFF_CONFIG"]
    store = app.config["WFF_TEST_STORE"]
    reads: list[str] = []

    real_runs = data_module.read_runs
    real_counts = data_module.read_photo_counts
    monkeypatch.setattr(
        data_module,
        "read_runs",
        lambda s, b, e: (reads.append(f"runs:{e}"), real_runs(s, b, e))[1],
    )
    monkeypatch.setattr(
        data_module,
        "read_photo_counts",
        lambda s, c, e: (reads.append(f"parquet:{e}"), real_counts(s, c, e))[1],
    )

    data_module.build_dashboard(store, config)
    assert len(reads) == len(set(reads)), f"read twice: {sorted(reads)}"


def test_review_page_renders_every_accepted_face(client):
    """"Exhaustively" is the requirement. A cap that silently hides faces would
    make the page useless as an accuracy meter -- the hidden ones are exactly
    where the mistakes hide."""
    body = client.get(f"/e/{EVENT}").get_data(as_text=True)
    assert body.count('class="face') >= 13   # 12 clustered + 1 leftover
    assert "900002_00" not in body           # the rejected face is not a result


def test_rejected_faces_are_reported_with_their_reason(client):
    body = client.get(f"/e/{EVENT}").get_data(as_text=True)
    assert "too small" in body


def test_unknown_event_is_not_a_500(client):
    assert client.get("/e/does-not-exist").status_code == 200  # renders "not ready"
    assert client.get("/e/../../etc/passwd").status_code in (404, 308)


# -- tuning ------------------------------------------------------------------


def test_thresholds_from_the_url_change_the_grouping(client):
    """The address bar is still the tuning knob for US, so a result can be
    reproduced exactly by pasting a URL -- even though the page no longer asks
    anyone to choose one."""
    tight = client.get(f"/e/{EVENT}?p1=0.20&p2=0.20").get_data(as_text=True)
    loose = client.get(f"/e/{EVENT}?p1=0.42&p2=0.95").get_data(as_text=True)
    assert "The people it found" in tight and "The people it found" in loose
    assert tight != loose


def test_a_silly_threshold_is_clamped_not_crashed(client):
    for query in ("p2=99", "p2=-4", "p2=banana", "minf=99999", "p1="):
        assert client.get(f"/e/{EVENT}?{query}").status_code == 200


def test_tuned_view_is_marked_as_not_the_default(client):
    """A hand-pinned URL must never masquerade as the automatic result.

    Without this the panel would say "set automatically" while showing whatever
    threshold somebody last pasted -- the one thing that would make the page
    lie about how it grouped.
    """
    body = client.get(f"/e/{EVENT}?p2=0.80").get_data(as_text=True)
    assert "Not the automatic grouping" in body
    assert "set automatically" not in body


def test_the_normal_page_never_shows_a_slider(client):
    """The controls are hidden because Adarsh cannot answer "how alike is alike"
    and neither can a photographer. They exist only behind ?expert=1."""
    plain = client.get(f"/e/{EVENT}").get_data(as_text=True)
    assert 'type="range"' not in plain
    assert 'class="tuner"' not in plain
    # The only trace on the normal page is one line inside the ••• menu, and it
    # says what it is for.
    assert plain.count("Manual grouping controls") == 1
    assert "Diagnostics for us" in plain

    expert = client.get(f"/e/{EVENT}?expert=1").get_data(as_text=True)
    assert 'type="range"' in expert
    assert 'class="tuner"' in expert


# -- feedback ----------------------------------------------------------------


def _person_ids(client, query=""):
    body = client.get(f"/e/{EVENT}{query}").get_data(as_text=True)
    import re

    return sorted({int(m) for m in re.findall(r'data-person="(\d+)"', body)})


def test_approving_a_pile_is_saved_and_scored(client):
    person_id = _person_ids(client)[0]
    response = client.post(
        f"/api/events/{EVENT}/feedback",
        json={"kind": "person_ok", "person_id": person_id, "reviewer": "adarsh"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] and payload["record_id"]
    assert payload["score"]["same_pairs_known"] > 0
    assert payload["score"]["diff_pairs_merged"] == 0


def test_approval_survives_a_page_reload(client):
    """A pile answered yesterday must come back ticked.

    The review window decides what to open next from `data-answered`, so if
    this were rendered wrong the window would walk somebody through twenty
    piles they had already done.
    """
    import re

    person_id = _person_ids(client)[0]
    client.post(
        f"/api/events/{EVENT}/feedback",
        json={"kind": "person_ok", "person_id": person_id, "reviewer": "devesh"},
    )
    body = client.get(f"/e/{EVENT}").get_data(as_text=True)
    card = re.search(rf'data-person="{person_id}".*?</article>', body, re.S).group(0)
    assert 'data-answered="1"' in card
    assert "1 of" in body  # the strip at the top counted it


def test_marking_a_face_wrong_retracts_a_conflicting_approval(client):
    """A reviewer who ticks a pile and then spots a bad face inside it is
    refining their answer, not contradicting themselves. If the tick stayed, the
    score would report a phantom conflict."""
    person_id = _person_ids(client)[0]
    client.post(
        f"/api/events/{EVENT}/feedback",
        json={"kind": "person_ok", "person_id": person_id, "reviewer": "adarsh"},
    )
    body = client.get(f"/e/{EVENT}").get_data(as_text=True)
    import re

    card = re.search(
        rf'data-person="{person_id}".*?</article>', body, re.S
    ).group(0)
    face_id = re.search(r'data-face="([^"]+)"', card).group(1)

    response = client.post(
        f"/api/events/{EVENT}/feedback",
        json={"kind": "face_wrong", "face_id": face_id, "person_id": person_id},
    )
    assert response.status_code == 200
    score = response.get_json()["score"]
    assert score["contradictions"] == 0
    # "All one person EXCEPT this one" -- the rest of the approval must survive,
    # or one flag silently destroys a reviewer's whole verdict on the pile.
    assert score["same_pairs_known"] > 0
    assert score["diff_pairs_known"] > 0


def test_a_judgement_cannot_be_attached_to_a_face_in_a_different_pile(client):
    people = _person_ids(client)
    if len(people) < 2:
        pytest.skip("needs two piles")
    body = client.get(f"/e/{EVENT}").get_data(as_text=True)
    import re

    second = re.search(rf'data-person="{people[1]}".*?</article>', body, re.S).group(0)
    face_in_second = re.search(r'data-face="([^"]+)"', second).group(1)
    response = client.post(
        f"/api/events/{EVENT}/feedback",
        json={"kind": "face_wrong", "face_id": face_in_second, "person_id": people[0]},
    )
    assert response.status_code == 400


def test_unknown_face_is_rejected(client):
    response = client.post(
        f"/api/events/{EVENT}/feedback",
        json={"kind": "not_a_face", "face_id": "999999_99"},
    )
    assert response.status_code == 400


def test_unknown_kind_is_rejected(client):
    response = client.post(
        f"/api/events/{EVENT}/feedback", json={"kind": "delete_everything"}
    )
    assert response.status_code == 400


def test_pair_judgement_records_a_must_not_link(client):
    body = client.get(f"/e/{EVENT}").get_data(as_text=True)
    import re

    faces = re.findall(r'data-face="([^"]+)"', body)
    response = client.post(
        f"/api/events/{EVENT}/feedback",
        json={"kind": "pair", "face_a": faces[0], "face_b": faces[-1], "same": False},
    )
    assert response.status_code == 200
    assert response.get_json()["score"]["diff_pairs_known"] == 1


def test_retract_undoes_a_judgement(client):
    person_id = _person_ids(client)[0]
    record_id = client.post(
        f"/api/events/{EVENT}/feedback",
        json={"kind": "person_ok", "person_id": person_id},
    ).get_json()["record_id"]

    response = client.post(f"/api/events/{EVENT}/feedback/{record_id}/retract", json={})
    assert response.status_code == 200
    assert response.get_json()["score"]["same_pairs_known"] == 0
    assert client.post(
        f"/api/events/{EVENT}/feedback/{record_id}/retract", json={}
    ).status_code == 404


def test_feedback_export_is_the_raw_journal(client):
    person_id = _person_ids(client)[0]
    client.post(
        f"/api/events/{EVENT}/feedback",
        json={"kind": "person_ok", "person_id": person_id, "reviewer": "adarsh"},
    )
    response = client.get(f"/e/{EVENT}/feedback.jsonl")
    assert response.status_code == 200
    lines = [json.loads(l) for l in response.get_data(as_text=True).splitlines() if l.strip()]
    assert lines and lines[0]["reviewer"] == "adarsh"
    assert lines[0]["same_group"]


def test_judgements_are_still_scored_after_the_thresholds_change(client):
    """THE test for the whole design. Approve a pile, then re-group at a
    different threshold: the judgement must still be counted, because it was
    stored against face ids and not against a person number."""
    person_id = _person_ids(client)[0]
    client.post(
        f"/api/events/{EVENT}/feedback",
        json={"kind": "person_ok", "person_id": person_id},
    )
    before = client.get(f"/api/events/{EVENT}/score").get_json()
    after = client.get(f"/api/events/{EVENT}/score?p1=0.30&p2=0.90").get_json()
    assert before["same_pairs_known"] == after["same_pairs_known"] > 0


def test_group_key_depends_only_on_membership_not_order():
    assert group_key(["b", "a"]) == group_key(["a", "b"])
    assert group_key(["a", "b"]) != group_key(["a", "b", "c"])


# -- images and paths --------------------------------------------------------


def test_crop_route_rejects_a_path_traversal(client):
    for name in ("../../../event.json", "..%2f..%2fstate.jsonl", "000001_00.jpg"):
        assert client.get(f"/i/{EVENT}/crop/{name}").status_code in (404, 308)


def test_image_route_rejects_an_unknown_kind(client):
    assert client.get(f"/i/{EVENT}/original/000001").status_code == 404


def test_missing_crop_is_a_404_not_a_crash(client):
    assert client.get(f"/i/{EVENT}/crop/000001_00").status_code == 404


def test_photo_api_returns_boxes_as_percentages(client):
    response = client.get(f"/api/events/{EVENT}/photo/000001")
    if response.status_code == 404:
        pytest.skip("no such photo in the fixture")
    data = response.get_json()
    for face in data["faces"]:
        left, top, width, height = face["box"]
        assert 0 <= left <= 100 and 0 <= top <= 100
        assert 0 < width <= 100 and 0 < height <= 100


# -- the pair chooser --------------------------------------------------------


def test_judge_page_offers_borderline_pairs_only(client):
    from wff.web.app import _candidate_pairs
    from wff.web.data import EventCache

    app_store = client.application.config["STORE"]
    config = client.application.config["CONFIG"]
    cache = EventCache(app_store, config)
    settings = ClusterConfig(pass1_threshold=0.42, pass2_threshold=0.55)
    output = cache.stage2(EVENT, settings)
    pairs = _candidate_pairs(
        output, cache.distance_fn(EVENT), fb.derive_truth([]), settings
    )
    for pair in pairs:
        assert 0.35 <= pair["distance"] <= 0.80
    assert client.get(f"/e/{EVENT}/judge").status_code == 200


def test_judge_page_survives_an_event_with_one_person(app, tmp_path):
    """A folder of one person's selfies produces no pairs at all. The page has
    to say so rather than divide by zero."""
    store = app.config["STORE"]
    config = app.config["CONFIG"]
    faces = [
        {
            "face_id": f"7{shot:05d}_00",
            "photo_id": f"7{shot:05d}",
            "embedding": _vector(55, drift=0.05 * shot),
        }
        for shot in range(4)
    ]
    _write_parquet(store, config, "ev_solo", faces)
    response = app.test_client().get("/e/ev_solo/judge")
    assert response.status_code in (200, 302)


# -- progress ----------------------------------------------------------------


def test_progress_is_read_from_stage1s_own_journal(app):
    """Progress must not need a hook inside the expensive stage: state.jsonl is
    already the authority on what is finished, and reading it keeps the bar
    honest across a resume."""
    from wff.web.data import count_processed

    store = app.config["STORE"]
    config = app.config["CONFIG"]
    paths = EventPaths(config.storage.bucket, EVENT)
    done, failed = count_processed(store, paths)
    assert done == 14 and failed == 0

    with store.open_append(paths.state) as fh:
        fh.write((json.dumps({"photo_id": "zzz", "status": "failed", "error": "x"}) + "\n").encode())
        fh.write(b'{"photo_id": "torn", "sta')  # crashed mid-write
    done_after, failed_after = count_processed(store, paths)
    assert done_after == 14 and failed_after == 1


def test_a_retried_photo_is_counted_once(app):
    """state.jsonl is append-only, so a resumed photo appears twice. Counting
    lines instead of distinct photo ids would report more than 100%."""
    from wff.web.data import count_processed

    store = app.config["STORE"]
    config = app.config["CONFIG"]
    paths = EventPaths(config.storage.bucket, EVENT)
    with store.open_append(paths.state) as fh:
        fh.write((json.dumps({"photo_id": "000001", "status": "done", "batch": 9}) + "\n").encode())
    done, _ = count_processed(store, paths)
    assert done == 14


# -- the progress tail -------------------------------------------------------
#
# The watcher polls every 1.5 seconds for the whole length of a run, so it reads
# the journal by byte offset instead of whole. That makes it a second
# implementation of the same count, and a second implementation is only safe
# while it is pinned to the first -- hence the agreement test below.


def _append(store, uri: str, *records: dict) -> None:
    with store.open_append(uri) as fh:
        for record in records:
            fh.write((json.dumps(record) + "\n").encode())


def test_the_tail_agrees_with_the_whole_file_count_at_every_step(app):
    """Pinned to `count_processed`, line by line, including the awkward cases.

    Two ways of counting the same journal will drift apart eventually; this is
    the test that notices. If it ever fails, the tail is wrong -- the whole-file
    read is the definition."""
    from wff.web.data import ProgressTail, count_processed

    store = app.config["STORE"]
    paths = EventPaths(app.config["CONFIG"].storage.bucket, EVENT)
    tail = ProgressTail()
    assert tail.read(store, paths.state) == count_processed(store, paths)

    for records in (
        [{"photo_id": "aaa", "status": "done"}],
        [{"photo_id": "bbb", "status": "failed", "error": "unreadable"}],
        [{"photo_id": "aaa", "status": "done"}],          # retried, same answer
        [{"photo_id": "bbb", "status": "done"}],          # retried, now fine
        [{"photo_id": "ccc", "status": "done"}, {"photo_id": "ddd", "status": "failed"}],
    ):
        _append(store, paths.state, *records)
        assert tail.read(store, paths.state) == count_processed(store, paths)

    assert tail.read(store, paths.state) == (17, 1)


def test_the_tail_reads_each_byte_of_the_journal_once(app):
    """The point of the exercise. Re-reading the journal every 1.5 seconds cost
    18.7 ms at 2,450 photos on a real folder and grew with it, so the progress
    bar taxed the run it was measuring. Counted rather than timed: a stopwatch
    on a file this small measures the disk cache."""
    from contextlib import contextmanager

    from wff.web.data import ProgressTail

    class CountingStore:
        """Delegates to the real store, counting the bytes actually handed out."""

        def __init__(self, inner):
            self._inner = inner
            self.bytes_read = 0

        def exists(self, uri):
            return self._inner.exists(uri)

        @contextmanager
        def open_read(self, uri):
            outer = self

            class Handle:
                def __init__(self, handle):
                    self._handle = handle

                def seek(self, *args):
                    return self._handle.seek(*args)

                def tell(self):
                    return self._handle.tell()

                def read(self, *args):
                    chunk = self._handle.read(*args)
                    outer.bytes_read += len(chunk)
                    return chunk

            with self._inner.open_read(uri) as handle:
                yield Handle(handle)

    store = app.config["STORE"]
    paths = EventPaths(app.config["CONFIG"].storage.bucket, EVENT)
    counting = CountingStore(store)
    size = len(store.get_bytes(paths.state))

    tail = ProgressTail()
    tail.read(counting, paths.state)
    assert counting.bytes_read == size  # the first read is the whole file

    for _ in range(20):  # 30 seconds of watching, with nothing being written
        tail.read(counting, paths.state)
    assert counting.bytes_read == size  # ...and it stayed the whole file, once

    _append(counting._inner, paths.state, {"photo_id": "eee", "status": "done"})
    tail.read(counting, paths.state)
    assert counting.bytes_read < size * 1.5  # only the new line, not all of it


def test_a_torn_line_is_counted_once_it_is_finished(app):
    """The journal is being appended to underneath the reader, so the last line
    is routinely half-written. A whole-file read simply skips it and picks it up
    next time; a tail that did the same would step past those bytes and lose the
    photo for the rest of the run."""
    from wff.web.data import ProgressTail

    store = app.config["STORE"]
    paths = EventPaths(app.config["CONFIG"].storage.bucket, EVENT)
    tail = ProgressTail()
    tail.read(store, paths.state)

    with store.open_append(paths.state) as fh:
        fh.write(b'{"photo_id": "half", "sta')
    assert tail.read(store, paths.state) == (14, 0)

    with store.open_append(paths.state) as fh:
        fh.write(b'tus": "done"}\n')
    assert tail.read(store, paths.state) == (15, 0)


def test_a_photo_that_failed_and_then_succeeded_is_not_counted_twice(app):
    """Stage 1 retries, so the same photo can be journalled failed and then
    done. Running totals that only ever add would leave it in both columns, and
    failed + done would exceed the folder."""
    from wff.web.data import ProgressTail

    store = app.config["STORE"]
    paths = EventPaths(app.config["CONFIG"].storage.bucket, EVENT)
    tail = ProgressTail()
    tail.read(store, paths.state)

    _append(store, paths.state, {"photo_id": "flaky", "status": "failed", "error": "x"})
    assert tail.read(store, paths.state) == (14, 1)

    _append(store, paths.state, {"photo_id": "flaky", "status": "done"})
    assert tail.read(store, paths.state) == (15, 0)


def test_a_replaced_journal_starts_the_count_over(app):
    """A shorter file at the same name is a different folder's journal, not a
    rewind. Seeking to the old offset would read from the middle of it."""
    from wff.web.data import ProgressTail

    store = app.config["STORE"]
    paths = EventPaths(app.config["CONFIG"].storage.bucket, EVENT)
    tail = ProgressTail()
    assert tail.read(store, paths.state) == (14, 0)

    store.put_text(
        paths.state,
        json.dumps({"photo_id": "new", "status": "done"}) + "\n",
    )
    assert tail.read(store, paths.state) == (1, 0)


def test_a_missing_journal_is_no_photos_not_a_crash(app):
    """A run that is still checking the folder has not written one yet."""
    from wff.web.data import ProgressTail

    store = app.config["STORE"]
    config = app.config["CONFIG"]
    tail = ProgressTail()
    missing = EventPaths(config.storage.bucket, "ev_does_not_exist")
    assert tail.read(store, missing.state) == (0, 0)


def test_stage1_stop_hook_leaves_the_run_resumable():
    """The Stop button must behave like a crash, which Stage 1 already survives:
    stop between photos, never mid-write."""
    import inspect

    from wff.stage1 import process_event

    source = inspect.getsource(process_event)
    assert "should_stop" in inspect.signature(process_event).parameters
    # The check has to precede the work, not follow the commit.
    assert source.index("should_stop()") < source.index("_commit_photo")


# -- watching a run happen ---------------------------------------------------


def _face_line(photo_id: str, index: int, height: int = 300, accepted: bool = True):
    return json.dumps(
        {
            "face_id": f"{photo_id}_{index:02d}",
            "photo_id": photo_id,
            "photo_path": f"D:/photos/{photo_id}.jpg",
            "face_h_px": height,
            "accepted": accepted,
            "second_class": False,
            "reject_reason": "" if accepted else "too small",
        }
    )


def test_live_tail_reads_only_what_is_new(app):
    """The wall of faces follows Stage 1's journal as it is written.

    Re-reading the whole file every two seconds would cost more disk than the
    pipeline itself by the end of a 4,000-photo wedding, so the tail keeps a
    byte offset. The failure that buys is double counting -- which is exactly
    what this asserts cannot happen.
    """
    from wff.web.data import LiveTail

    store = app.config["STORE"]
    config = app.config["CONFIG"]
    paths = EventPaths(config.storage.bucket, "ev_live")
    tail = LiveTail()

    with store.open_append(paths.faces_jsonl) as fh:
        fh.write((_face_line("000001", 0) + "\n" + _face_line("000001", 1) + "\n").encode())

    faces, totals = tail.read(store, paths.faces_jsonl)
    assert totals.detected == 2 and totals.accepted == 2
    assert len(faces) == 2

    # A second poll with nothing new must not count anything twice.
    faces, totals = tail.read(store, paths.faces_jsonl)
    assert totals.detected == 2
    assert len(faces) == 2

    with store.open_append(paths.faces_jsonl) as fh:
        fh.write((_face_line("000002", 0, height=60) + "\n").encode())
    faces, totals = tail.read(store, paths.faces_jsonl)
    assert totals.detected == 3
    assert totals.too_small == 1          # 60px is under what the recogniser wants
    assert totals.photos_seen == 2


def test_live_tail_survives_a_half_written_line(app):
    """Stage 1 is appending underneath us, so the last line is routinely torn.

    A torn line must be neither counted nor lost -- it has to be counted once,
    when the rest of it arrives.
    """
    from wff.web.data import LiveTail

    store = app.config["STORE"]
    config = app.config["CONFIG"]
    paths = EventPaths(config.storage.bucket, "ev_torn")
    tail = LiveTail()

    complete = _face_line("000001", 0)
    with store.open_append(paths.faces_jsonl) as fh:
        fh.write((complete + "\n" + complete[:40]).encode())

    _faces, totals = tail.read(store, paths.faces_jsonl)
    assert totals.detected == 1, "the half line must not be counted yet"

    with store.open_append(paths.faces_jsonl) as fh:
        fh.write((complete[40:] + "\n").encode())
    _faces, totals = tail.read(store, paths.faces_jsonl)
    assert totals.detected == 2, "and must be counted once it is whole"


def test_live_endpoint_names_all_six_steps(client):
    """Adarsh's complaint about the old screen: a bar and a word. Every step is
    named, including the ones that have not started."""
    response = client.get(f"/api/events/{EVENT}/live")
    assert response.status_code == 200
    steps = response.get_json()["steps"]
    assert len(steps) == 6
    assert [step["title"] for step in steps][:2] == ["Check the folder", "Count the photos"]
    assert all(step["state"] in ("done", "now", "todo", "failed") for step in steps)


def test_rejected_faces_never_reach_the_live_wall(app):
    """The wall shows what was kept. Showing a face that was then thrown away
    would make the run look better than it is."""
    from wff.web.data import LiveTail

    store = app.config["STORE"]
    config = app.config["CONFIG"]
    paths = EventPaths(config.storage.bucket, "ev_rejects")
    tail = LiveTail()
    with store.open_append(paths.faces_jsonl) as fh:
        fh.write((_face_line("000001", 0, accepted=False) + "\n").encode())

    faces, totals = tail.read(store, paths.faces_jsonl)
    assert faces == []
    assert totals.detected == 1 and totals.accepted == 0
    assert totals.reject_reasons == {"too small": 1}


# -- the home page: biggest first, in size groups ----------------------------


def test_runs_are_ordered_biggest_first_and_grouped_by_size(app):
    """Adarsh's reason for the change: a 4,000-photo wedding is where this
    product works or does not, and a 17-photo test folder run this morning is
    not more important than it.

    Size is the photo count, not the gigabytes: 400 phone photos and 400
    camera JPEGs cost the same afternoon.
    """
    from wff.web.data import group_by_size, list_events

    store = app.config["WFF_TEST_STORE"]
    config = app.config["WFF_CONFIG"]
    for name, photos in [("ev_big", 900), ("ev_mid", 300), ("ev_tiny", 12)]:
        store.put_text(
            EventPaths(config.storage.bucket, name).event_json,
            json.dumps(
                {
                    "event_id": name,
                    "photo_count": photos,
                    "created_at": "2026-08-01T09:00:00+00:00",
                }
            ),
        )

    events = list_events(store, config)
    sizes = [event.size for event in events]
    assert sizes == sorted(sizes, reverse=True), "biggest first, whatever the date"

    groups = group_by_size(events)
    assert [group.key for group in groups][:2] == ["large", "medium"]
    by_key = {group.key: [e.event_id for e in group.events] for group in groups}
    assert "ev_big" in by_key["large"]
    assert "ev_mid" in by_key["medium"]
    assert "ev_tiny" in by_key["small"]


def test_an_empty_size_never_promotes_another_group_to_the_top(app):
    """Large is always the first heading when it exists, and simply absent when
    it does not. A page whose sections move around depending on the day cannot
    be read at a glance."""
    from wff.web.data import EventSummary, group_by_size

    small_only = [
        EventSummary(event_id="a", photo_count=4),
        EventSummary(event_id="b", photo_count=90),
    ]
    assert [group.key for group in group_by_size(small_only)] == ["small"]

    mixed = small_only + [EventSummary(event_id="c", photo_count=800)]
    assert [group.key for group in group_by_size(mixed)] == ["large", "small"]


def test_each_size_heading_says_what_it_means(client):
    """The colour legend above the run list is gone -- the cards it explained
    are now plain white, so a legend of three colours explained nothing.

    But the *sentence* it carried has to survive: without it, "Small runs" is a
    heading nobody can act on, because the boundary is invisible.
    """
    body = client.get("/").get_data(as_text=True)
    assert "Small runs" in body
    assert "under 100" in body, "the heading has to state its own boundary"


def test_a_run_carries_its_full_name_even_when_the_card_cuts_it_short(client):
    """A folder id is 30 characters and the card shows one line, so long names
    are cut with an ellipsis rather than stacked one letter per line down the
    side of the card.

    Cutting is only safe because the whole name is still on the element to be
    read on hover. Drop that and long names become permanently unreadable, with
    nothing on screen to show anything is missing.
    """
    client.post(f"/api/events/{EVENT}/label", json={"name": "a name far too long to fit on one line"})
    body = client.get("/").get_data(as_text=True)
    assert 'title="a name far too long to fit on one line"' in body


def test_the_home_page_offers_to_name_who_ran_an_untracked_folder(client):
    """"before we tracked who" said nothing to anybody. The folders from before
    the console asked for a name now ask for one."""
    body = client.get("/").get_data(as_text=True)
    assert "before we tracked who" not in body
    assert "set who ran this" in body


# -- renaming, and saying who ran it -----------------------------------------


def test_renaming_a_run_changes_the_label_and_nothing_else(client):
    """The id is on thousands of crop filenames, so it cannot move. The rename
    is a label beside the run, and the id stays visible next to it."""
    response = client.post(
        f"/api/events/{EVENT}/label", json={"name": "Sharma wedding - day 1"}
    )
    assert response.status_code == 200
    assert response.get_json()["name"] == "Sharma wedding - day 1"

    body = client.get("/").get_data(as_text=True)
    assert "Sharma wedding - day 1" in body
    assert EVENT in body, "the folder id must still be findable on the page"

    review = client.get(f"/e/{EVENT}").get_data(as_text=True)
    assert "Sharma wedding - day 1" in review


def test_clearing_a_name_puts_the_id_back(client):
    client.post(f"/api/events/{EVENT}/label", json={"name": "temporary"})
    client.post(f"/api/events/{EVENT}/label", json={"name": ""})
    body = client.get("/").get_data(as_text=True)
    assert "temporary" not in body
    assert EVENT in body


def test_a_hand_set_reviewer_wins_over_the_journal(app, client):
    """Typing a name is a correction. If the journal kept winning, the folders
    this exists for could never be fixed."""
    from wff.runs import RunRecord, append_run
    from wff.web.data import load_event_summary

    store = app.config["WFF_TEST_STORE"]
    config = app.config["WFF_CONFIG"]
    append_run(store, config.storage.bucket, RunRecord(event_id=EVENT, reviewer="unknown"))

    client.post(f"/api/events/{EVENT}/label", json={"tested_by": "Devesh"})
    summary = load_event_summary(EVENT, store, config)
    assert summary.tested_by == "Devesh"
    assert summary.reviewer == "unknown", "the journal itself is never rewritten"


def test_renaming_does_not_rewrite_who_ran_it(app, client):
    """Found by using it: every POST from the browser carries a `reviewer` key
    naming whoever is at the keyboard. While the route read the field it was
    SETTING from that same key, renaming a folder silently reassigned the run
    to whoever renamed it."""
    from wff.web.data import load_event_summary

    store = app.config["WFF_TEST_STORE"]
    config = app.config["WFF_CONFIG"]
    client.post(f"/api/events/{EVENT}/label", json={"tested_by": "Devesh"})
    client.post(
        f"/api/events/{EVENT}/label",
        json={"name": "Sharma wedding", "reviewer": "Adarsh"},
    )
    assert load_event_summary(EVENT, store, config).tested_by == "Devesh"


def test_a_label_cannot_be_set_on_an_id_that_is_not_ours(client):
    assert client.post("/api/events/..%2Fetc/label", json={"name": "x"}).status_code == 404
    assert client.post(f"/api/events/{EVENT}/label", json={}).status_code == 400


def test_a_name_is_one_trimmed_line(app):
    """A pasted name carrying a newline should not be an error message."""
    from wff.web.meta import clean_name

    assert clean_name("  Sharma\n wedding  ") == "Sharma wedding"
    assert clean_name("x" * 200) == "x" * 60
    assert clean_name(None) == ""


# -- the review page: how far along ------------------------------------------


def test_progress_counts_a_pile_as_answered_only_when_it_was_answered(client):
    """Both answers to "is this one person" count: yes, and "no, that one is
    not". "Not a face at all" deliberately does not -- it is a complaint about
    the detector, and counting it would let a folder read 100% checked without
    a single person having been looked at."""
    body = client.get(f"/e/{EVENT}").get_data(as_text=True)
    assert 'id="rv-pct">0<' in body

    person_id = _person_ids(client)[0]
    response = client.post(
        f"/api/events/{EVENT}/feedback",
        json={"kind": "person_ok", "person_id": person_id},
    )
    progress = response.get_json()["score"]["progress"]
    assert progress["piles_answered"] == 1
    assert progress["percent"] == round(1 / progress["piles_total"] * 100)


def test_not_a_face_does_not_count_as_checking_a_pile(client):
    import re

    body = client.get(f"/e/{EVENT}").get_data(as_text=True)
    person_id = _person_ids(client)[0]
    card = re.search(rf'data-person="{person_id}".*?</article>', body, re.S).group(0)
    face_id = re.search(r'data-face="([^"]+)"', card).group(1)

    response = client.post(
        f"/api/events/{EVENT}/feedback",
        json={"kind": "not_a_face", "face_id": face_id},
    )
    assert response.get_json()["score"]["progress"]["piles_answered"] == 0


def test_marking_a_face_wrong_counts_the_pile_as_answered(client):
    """The window's "someone else is in here" path ends here, and the strip at
    the top has to move -- otherwise answering feels like nothing happened."""
    import re

    body = client.get(f"/e/{EVENT}").get_data(as_text=True)
    person_id = _person_ids(client)[0]
    card = re.search(rf'data-person="{person_id}".*?</article>', body, re.S).group(0)
    face_id = re.search(r'data-face="([^"]+)"', card).group(1)

    response = client.post(
        f"/api/events/{EVENT}/feedback",
        json={"kind": "face_wrong", "face_id": face_id, "person_id": person_id},
    )
    progress = response.get_json()["score"]["progress"]
    assert progress["piles_answered"] == 1
    assert progress["faces_fixed"] == 1


def test_the_review_page_asks_one_question_and_hides_the_rest(client):
    """The complaint that started this: the page was a report and a tool at
    once. The question lives in the window, the score is one line, and Judge
    pairs and Export are in the ... menu rather than the top bar."""
    body = client.get(f"/e/{EVENT}").get_data(as_text=True)

    assert "Is everyone here the same person?" in body
    assert "Your job here is one question per pile" not in body
    assert "orange edge" not in body
    assert "faces found in the wrong pile" not in body

    assert "Show the risky ones" in body
    assert "Send my answers to Devesh" in body
    assert "Quick check on tricky pairs" in body

    nav = body.split("<nav>")[1].split("</nav>")[0]
    assert "judge" not in nav.lower()
    assert "export" not in nav.lower()


# -- the one-time Google key, set from the console ---------------------------
#
# Adarsh's actual words: "we have a UI, we just wanna paste the link there and
# it should work". So the key has to be settable here, and the run he already
# pressed Start on has to survive the detour.


@pytest.fixture()
def no_key(monkeypatch, tmp_path):
    """No key on this machine, and a settings file that is not the real one."""
    from wff import config as config_module

    monkeypatch.setattr(config_module, "SETTINGS_FILE", str(tmp_path / "settings.env"))
    monkeypatch.setattr(config_module, "_PROJECT_ENV", str(tmp_path / "none.env"))
    monkeypatch.setattr(config_module, "_settings_loaded", True)
    # setenv first, then delenv: saving a key writes to os.environ so it is
    # live immediately, and monkeypatch only unwinds variables it has recorded.
    # Without the setenv, a key saved by one test stays set for every test
    # after it, and "there is no key" stops being true.
    monkeypatch.setenv("WFF_GOOGLE_API_KEY", "")
    monkeypatch.delenv("WFF_GOOGLE_API_KEY", raising=False)
    return tmp_path / "settings.env"


DRIVE_LINK = "https://drive.google.com/drive/folders/1oBr6vhYfIQoHm8ITnAJvfFCZi-rgu5sQ"


def _validation(ok, message="", detail=None):
    from wff.ingest.base import ValidationResult

    return ValidationResult(ok, message, detail or {})


def test_a_drive_link_without_a_key_asks_for_one_instead_of_failing(client, no_key):
    """It used to queue a run that could only fail, writing a failed run to the
    journal and teaching nobody anything."""
    response = client.post(
        "/runs", data={"link": DRIVE_LINK, "event_id": "ev_drivetest", "reviewer": "adarsh"}
    )
    assert response.status_code == 302
    assert "need_key=1" in response.headers["Location"]
    # The typed link comes back with it -- nothing to re-paste.
    assert "1oBr6vhYfIQoHm8ITnAJvfFCZi" in response.headers["Location"]


def test_a_local_folder_still_starts_without_any_key(client, no_key, tmp_path):
    """The key is a Drive thing. It must never stand between a local folder
    and a run."""
    photos = tmp_path / "photos"
    photos.mkdir()
    response = client.post("/runs", data={"link": str(photos), "event_id": "ev_localtest"})
    assert response.status_code == 302
    assert "need_key" not in response.headers["Location"]


def test_the_key_box_opens_with_the_link_still_in_it(client, no_key):
    body = client.get(f"/?need_key=1&link={DRIVE_LINK}&event_id=ev_drivetest").get_data(
        as_text=True
    )
    assert "that Drive link needs a key first" in body
    assert DRIVE_LINK in body
    assert "Google Drive API" in body  # the step people get wrong


def test_a_working_key_is_saved(client, no_key, monkeypatch):
    monkeypatch.setattr(
        "wff.web.app.GoogleDriveFolderAdapter.validate",
        lambda self: _validation(True, "Public folder readable: 'Photos'"),
    )
    data = client.post(
        "/api/settings/google-key", json={"key": "AIzaGOOD", "link": DRIVE_LINK}
    ).get_json()
    assert data["ok"] is True
    from wff.config import google_api_key

    assert google_api_key() == "AIzaGOOD"


def test_a_key_google_rejects_is_not_saved(client, no_key, monkeypatch):
    """Saving first and testing later is how someone ends up with a broken key
    in a file they cannot see, failing every run with the message they thought
    they had just fixed."""
    monkeypatch.setattr(
        "wff.web.app.GoogleDriveFolderAdapter.validate",
        lambda self: _validation(False, "That API key is not valid", {"status": 400}),
    )
    data = client.post(
        "/api/settings/google-key", json={"key": "AIzaBAD", "link": DRIVE_LINK}
    ).get_json()
    assert data["ok"] is False
    assert data["key_kept"] is False
    assert not no_key.exists()


def test_a_good_key_survives_a_folder_that_is_not_public(client, no_key, monkeypatch):
    """404 means Google took the key and then said the FOLDER is private. The
    key is fine; throwing it away would mean re-pasting it after fixing the
    sharing."""
    monkeypatch.setattr(
        "wff.web.app.GoogleDriveFolderAdapter.validate",
        lambda self: _validation(False, "not shared publicly", {"status": 404}),
    )
    data = client.post(
        "/api/settings/google-key", json={"key": "AIzaGOOD", "link": DRIVE_LINK}
    ).get_json()
    assert data["ok"] is False
    assert data["key_kept"] is True
    from wff.config import google_api_key

    assert google_api_key() == "AIzaGOOD"


def test_pasting_a_link_into_the_key_box_is_caught(client, no_key):
    response = client.post(
        "/api/settings/google-key", json={"key": DRIVE_LINK, "link": DRIVE_LINK}
    )
    assert response.status_code == 400
    assert "looks like a link" in response.get_json()["message"]


def test_the_key_is_never_echoed_back(client, no_key, monkeypatch):
    monkeypatch.setattr(
        "wff.web.app.GoogleDriveFolderAdapter.validate",
        lambda self: _validation(True, "Public folder readable: 'Photos'"),
    )
    body = client.post(
        "/api/settings/google-key", json={"key": "AIzaSECRET", "link": DRIVE_LINK}
    ).get_data(as_text=True)
    assert "AIzaSECRET" not in body
    assert "AIzaSECRET" not in client.get("/").get_data(as_text=True)


def test_the_line_says_drive_is_ready_once_a_key_exists(client, no_key, monkeypatch):
    """The quiet state. Someone who set this up last week should be able to see
    that it is still set up without pasting anything."""
    body = client.get("/").get_data(as_text=True)
    assert "need a one-time setup" in body

    monkeypatch.setenv("WFF_GOOGLE_API_KEY", "AIzaALREADYSET")
    body = client.get("/").get_data(as_text=True)
    assert "Google Drive links are set up" in body
    assert "change the key" in body
    assert "AIzaALREADYSET" not in body
