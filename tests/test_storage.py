"""The no-local-paths rule has to be enforced, not just documented."""
from __future__ import annotations

import pytest

from wff.storage import LocalObjectStore, join_uri, parse_uri


def test_parse_uri():
    uri = parse_uri("r2://wff/events/ev_ab12/prev/000481.jpg")
    assert uri.scheme == "r2"
    assert uri.bucket == "wff"
    assert uri.key == "events/ev_ab12/prev/000481.jpg"


@pytest.mark.parametrize(
    "bad",
    [
        r"D:\Adarsh\photos\IMG_001.jpg",
        "/home/adarsh/photos",
        "events/ev_ab12/prev/000481.jpg",
        "./relative/path.jpg",
    ],
)
def test_local_paths_are_rejected(bad):
    """A filesystem path must not be silently accepted anywhere in the pipeline.

    If this test ever gets relaxed, the GPU migration quietly becomes a rewrite.
    """
    with pytest.raises(ValueError):
        parse_uri(bad)


def test_join_uri():
    assert (
        join_uri("r2://wff/events/ev1", "crops", "000001_00.jpg")
        == "r2://wff/events/ev1/crops/000001_00.jpg"
    )


def test_roundtrip(tmp_path):
    store = LocalObjectStore(str(tmp_path))
    uri = "r2://wff/events/ev1/thing.txt"
    store.put_text(uri, "hello")
    assert store.exists(uri)
    assert store.get_text(uri) == "hello"


def test_append_is_crash_safe_in_order(tmp_path):
    store = LocalObjectStore(str(tmp_path))
    uri = "r2://wff/events/ev1/journal.jsonl"
    for i in range(3):
        with store.open_append(uri) as fh:
            fh.write(f"line{i}\n".encode())
    assert store.get_text(uri) == "line0\nline1\nline2\n"


def test_open_append_reports_end_position(tmp_path):
    """stage1 derives embedding row offsets from tell(); it must be the end."""
    store = LocalObjectStore(str(tmp_path))
    uri = "r2://wff/events/ev1/embeddings.f32"
    with store.open_append(uri) as fh:
        fh.write(b"0" * 100)
    with store.open_append(uri) as fh:
        assert fh.tell() == 100


def test_list_keys(tmp_path):
    store = LocalObjectStore(str(tmp_path))
    store.put_text("r2://wff/events/ev1/a.txt", "a")
    store.put_text("r2://wff/events/ev1/sub/b.txt", "b")
    store.put_text("r2://wff/events/ev2/c.txt", "c")
    keys = sorted(store.list_keys("r2://wff/events/ev1"))
    assert keys == ["r2://wff/events/ev1/a.txt", "r2://wff/events/ev1/sub/b.txt"]


def test_traversal_is_refused(tmp_path):
    store = LocalObjectStore(str(tmp_path))
    with pytest.raises(ValueError):
        store.put_text("r2://wff/../../escaped.txt", "nope")


def test_public_url_is_relative_for_the_report(tmp_path):
    store = LocalObjectStore(str(tmp_path))
    store.put_text("r2://wff/events/ev1/crops/000001_00.jpg", "x")
    url = store.public_url(
        "r2://wff/events/ev1/crops/000001_00.jpg",
        relative_to="r2://wff/events/ev1/report.html",
    )
    assert url == "crops/000001_00.jpg"


# -- listing folders without walking every file -----------------------------


def test_list_prefixes_returns_immediate_children_only(tmp_path):
    """Discovering the event ids used to enumerate every key, which meant
    visiting all ~10,000 crops and previews to learn six names -- a ~600 ms
    delay on every keystroke in the forecast box."""
    store = LocalObjectStore(str(tmp_path))
    store.put_text("r2://wff/events/ev_a/event.json", "{}")
    store.put_text("r2://wff/events/ev_a/crops/000001_00.jpg", "x")
    store.put_text("r2://wff/events/ev_a/crops/000002_00.jpg", "x")
    store.put_text("r2://wff/events/ev_b/event.json", "{}")

    assert sorted(store.list_prefixes("r2://wff/events")) == ["ev_a", "ev_b"]


def test_list_prefixes_is_empty_for_a_missing_prefix(tmp_path):
    store = LocalObjectStore(str(tmp_path))
    assert list(store.list_prefixes("r2://wff/events")) == []


def test_the_default_list_prefixes_agrees_with_the_local_override(tmp_path):
    """The base class derives prefixes from list_keys so a backend only
    overrides it when it can do better. If the two ever disagreed, swapping in
    R2 would silently change which events the dashboard can see.
    """
    from wff.storage.base import ObjectStore

    store = LocalObjectStore(str(tmp_path))
    store.put_text("r2://wff/events/ev_a/deep/nested/file.txt", "x")
    store.put_text("r2://wff/events/ev_b/event.json", "{}")

    derived = sorted(ObjectStore.list_prefixes(store, "r2://wff/events"))
    overridden = sorted(store.list_prefixes("r2://wff/events"))
    assert derived == overridden == ["ev_a", "ev_b"]
