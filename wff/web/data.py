"""What the web layer reads. Object URIs only -- no filesystem paths.

The console re-runs Stage 2 on every threshold change, which is the entire point
of the expensive/cheap stage split: the parquet is read once and kept, then
clustering re-runs against it in seconds. Without the cache here, every slider
nudge would re-read and re-decode the parquet.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ..cluster.autotune import AutoThreshold, choose_threshold, same_photo_distances
from ..cluster.metrics import evaluate, sanity_check
from ..cluster.two_pass import cluster_faces
from ..config import ClusterConfig, Config
from ..runs import RunRecord, RunTotals, aggregate_runs, read_runs
from ..stage1 import EventPaths
from ..stage2 import FaceTable, Stage2Output, _ground_truth_from_paths, load_faces
from ..storage import ObjectStore, parse_uri
from .meta import EventLabel, read_label

# How big a run is, in photos rather than gigabytes: 400 phone photos and 400
# RAW-adjacent JPEGs cost the same afternoon, and photo count is the number
# every other figure on the card is per-of. Boundaries chosen against what is
# actually in the store today -- Manali (851) is a real wedding-sized job, the
# roka folders (100-350) are a session, anything under 100 is test material.
LARGE_RUN = 500
MEDIUM_RUN = 100

SIZE_CLASSES = ("large", "medium", "small")
SIZE_LABELS = {
    "large": "Large runs",
    "medium": "Medium runs",
    "small": "Small runs",
}
SIZE_RANGES = {
    "large": f"{LARGE_RUN}+ photos",
    "medium": f"{MEDIUM_RUN} to {LARGE_RUN - 1}",
    "small": f"under {MEDIUM_RUN}",
}


@dataclass
class PhotoInfo:
    """Enough to draw a box on a preview without knowing the preview's size.

    Box coordinates are stored against the ORIGINAL photo, and previews are
    capped at 2048px, so the two never match. Percentages of the original are
    resolution-independent and survive any future preview size change.
    """

    photo_id: str
    path: str
    width: int = 0
    height: int = 0
    taken_at: str | None = None
    face_count: int = 0
    accepted_count: int = 0


@dataclass
class EventSummary:
    event_id: str
    photo_count: int = 0
    source_type: str = ""
    created_at: str = ""
    has_faces: bool = False
    processed: int = 0
    failed: int = 0
    total_bytes: int = 0
    model_version: str = ""
    source_folder: str = ""
    # Filled from runs.jsonl. None means this folder was processed before run
    # timing existed -- shown as "not recorded", never guessed.
    total_seconds: float | None = None
    seconds_per_photo: float | None = None
    reviewer: str = ""
    run_count: int = 0
    last_phase: str = ""
    # When this folder was last worked on, not when it was first ingested. A
    # folder created on Tuesday and re-run today belongs at the top of the list;
    # sorting on created_at would bury it under folders nobody has touched.
    last_run_at: str = ""
    # The folder's run journal, kept rather than re-read. The dashboard needs
    # the raw records to weight throughput by photos, and this object has
    # already opened the file to fill in the timings above -- reading it a
    # second time per folder is a cost that grows with the list.
    runs: list[RunRecord] = field(default_factory=list)
    # Filled from summary.json, written whenever grouping runs at the saved
    # defaults. None means nobody has opened this folder's review page yet.
    people_found: int | None = None
    faces_detected: int = 0
    faces_accepted: int = 0
    photos_with_faces: int = 0
    # What a person called this folder, and who they say ran it. Set by hand on
    # the home page; see meta.py for why they are not part of the event id.
    label: EventLabel = field(default_factory=EventLabel)

    @property
    def name(self) -> str:
        """What the page calls this run. The id is still shown beside it, small
        -- renaming must never make a folder unfindable on disk."""
        return self.label.display_name or self.event_id

    @property
    def renamed(self) -> bool:
        return bool(self.label.display_name)

    @property
    def tested_by(self) -> str:
        """A hand-set name beats the journal: setting it is a correction, and
        the only reason to type one is that the recorded name is wrong or
        missing."""
        return self.label.reviewer or self.reviewer

    @property
    def size(self) -> int:
        """The photo count this run is filed under.

        max() of the two: a folder counted but not yet processed still has a
        real size, and a resumed run's processed count can exceed the manifest
        when photos were added between sittings.
        """
        return max(self.photo_count, self.processed)

    @property
    def size_class(self) -> str:
        if self.size >= LARGE_RUN:
            return "large"
        if self.size >= MEDIUM_RUN:
            return "medium"
        return "small"

    @property
    def when(self) -> str:
        """The one date this folder is filed under, whatever we happen to know.

        The six folders processed before run timing existed have no run journal,
        so they fall back to when they were ingested -- which for those is the
        same afternoon anyway.
        """
        return self.last_run_at or self.created_at

    @property
    def progress_pct(self) -> float:
        if not self.photo_count:
            return 0.0
        return min(100.0, self.processed / self.photo_count * 100.0)

    @property
    def too_small_to_judge(self) -> bool:
        """Two photos cannot prove anything, and reporting "0 people" for them
        reads as a failure rather than as an inadequate test."""
        return self.has_faces and self.processed < 10


def _event_ids(store: ObjectStore, config: Config) -> list[str]:
    """Every event id in the store, without reading a single file.

    Uses prefix listing rather than key listing: enumerating every key meant
    walking all ~10,000 crops and previews to learn six event names.
    """
    try:
        return sorted(store.list_prefixes(f"r2://{config.storage.bucket}/events"))
    except (ValueError, OSError):
        return []


def list_events(store: ObjectStore, config: Config) -> list[EventSummary]:
    """Every event already in the store, biggest first.

    Was most-recently-worked-on first. Adarsh's reason for the change is worth
    keeping: a 4,000-photo wedding is where the product either works or does
    not, and a 17-photo test folder run this morning is not more important than
    it. Ties -- and the six test folders are full of ties -- fall back to the
    date, so the order inside a size is still newest first.
    """
    summaries = [
        load_event_summary(event_id, store, config)
        for event_id in _event_ids(store, config)
    ]
    summaries.sort(key=lambda s: (s.size, s.when or "", s.event_id), reverse=True)
    return summaries


@dataclass
class SizeGroup:
    """One heading on the home page, and the runs under it."""

    key: str
    title: str
    range_text: str
    events: list[EventSummary] = field(default_factory=list)

    @property
    def photos(self) -> int:
        return sum(event.size for event in self.events)


def group_by_size(events: list[EventSummary]) -> list[SizeGroup]:
    """Large, then medium, then small -- empty groups dropped.

    Order is fixed rather than computed: the point of the grouping is that the
    big jobs are always in the same place on the page, so an empty "large"
    section must not promote "medium" to the top spot.
    """
    buckets = {key: [] for key in SIZE_CLASSES}
    for event in events:
        buckets[event.size_class].append(event)
    return [
        SizeGroup(
            key=key,
            title=SIZE_LABELS[key],
            range_text=SIZE_RANGES[key],
            events=buckets[key],
        )
        for key in SIZE_CLASSES
        if buckets[key]
    ]


def load_event_summary(
    event_id: str, store: ObjectStore, config: Config
) -> EventSummary:
    paths = EventPaths(config.storage.bucket, event_id)
    summary = EventSummary(event_id=event_id)

    if store.exists(paths.event_json):
        try:
            meta = json.loads(store.get_text(paths.event_json))
            summary.photo_count = int(meta.get("photo_count") or 0)
            summary.source_type = meta.get("source_type") or ""
            summary.created_at = meta.get("created_at") or ""
            summary.total_bytes = int(meta.get("total_source_bytes") or 0)
            summary.model_version = meta.get("embedding_model_version") or ""
            summary.source_folder = meta.get("source_folder_id") or ""
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    summary.has_faces = store.exists(paths.faces_parquet)
    done, failed = count_processed(store, paths)
    summary.processed = done
    summary.failed = failed

    # Timing, from the run journal. Sum across runs: a folder resumed three
    # times cost the sum of those three sittings, not the last one.
    records = read_runs(store, config.storage.bucket, event_id)
    summary.runs = records
    if records:
        summary.run_count = len(records)
        timed = [r for r in records if r.timed]
        if timed:
            summary.total_seconds = sum(r.total_seconds for r in timed)
        rates = [r.seconds_per_photo for r in records if r.seconds_per_photo]
        if rates:
            summary.seconds_per_photo = sum(rates) / len(rates)
        summary.reviewer = next(
            (r.reviewer for r in reversed(records) if r.reviewer), ""
        )
        summary.last_phase = records[-1].phase
        # max(), not records[-1]: the journal is appended to by two processes
        # (the console and the CLI), so file order is not guaranteed to be
        # chronological order.
        stamps = [r.finished_at or r.started_at for r in records]
        summary.last_run_at = max((s for s in stamps if s), default="")

    # Face counts come off photos.parquet, which carries a per-photo tally and
    # is a few KB. Reading them here rather than from summary.json means the
    # dashboard shows real face numbers for folders nobody has opened yet --
    # counting faces needs no clustering, so making it wait on one was wrong.
    counts = read_photo_counts(store, config, event_id)
    summary.faces_detected = counts.faces_detected
    summary.faces_accepted = counts.faces_accepted
    summary.photos_with_faces = counts.photos_with_faces

    # People, though, genuinely requires grouping to have run.
    stored = read_event_summary(store, config, event_id)
    if stored:
        summary.people_found = stored.get("people_found")

    summary.label = read_label(store, config.storage.bucket, event_id)
    return summary


@dataclass
class PhotoCounts:
    """Per-folder tallies, read from the small parquet, no clustering needed."""

    faces_detected: int = 0
    faces_accepted: int = 0
    photos_with_faces: int = 0
    single_face_photos: int = 0
    most_faces_in_one_photo: int = 0


def read_photo_counts(
    store: ObjectStore, config: Config, event_id: str
) -> PhotoCounts:
    import io

    import pyarrow.parquet as pq

    counts = PhotoCounts()
    paths = EventPaths(config.storage.bucket, event_id)
    if not store.exists(paths.photos_parquet):
        return counts
    try:
        table = pq.read_table(
            io.BytesIO(store.get_bytes(paths.photos_parquet)),
            columns=["accepted_count", "detected_count"],
        )
    except (OSError, ValueError, KeyError):
        return counts

    accepted = table.column("accepted_count").to_pylist()
    detected = table.column("detected_count").to_pylist()
    for index, raw_accepted in enumerate(accepted):
        got = int(raw_accepted or 0)
        counts.faces_accepted += got
        counts.faces_detected += int(detected[index] or 0)
        if got <= 0:
            continue
        counts.photos_with_faces += 1
        if got == 1:
            counts.single_face_photos += 1
        counts.most_faces_in_one_photo = max(counts.most_faces_in_one_photo, got)
    return counts


# ---------------------------------------------------------------------------
# summary.json -- grouping results, cached so the dashboard need not re-cluster
# ---------------------------------------------------------------------------


def summary_uri(config: Config, event_id: str) -> str:
    from ..storage import join_uri

    return join_uri(EventPaths(config.storage.bucket, event_id).root, "summary.json")


def read_event_summary(
    store: ObjectStore, config: Config, event_id: str
) -> dict | None:
    uri = summary_uri(config, event_id)
    if not store.exists(uri):
        return None
    try:
        return json.loads(store.get_text(uri))
    except (json.JSONDecodeError, OSError):
        return None


def write_event_summary(
    store: ObjectStore,
    config: Config,
    event_id: str,
    output: Stage2Output,
    signature: str | None = None,
) -> None:
    """Cache what grouping found, so the home page never has to re-cluster.

    Called for the settings the review page actually shows: either the saved
    defaults, or the threshold the folder's own photos chose. Never for a
    hand-pinned tuning run -- the dashboard must not report whichever value
    somebody last pasted into the address bar.
    """
    store.put_text(
        summary_uri(config, event_id),
        json.dumps(
            {
                "event_id": event_id,
                "people_found": len(output.result.persons),
                "leftover_faces": output.result.leftover_face_count,
                "faces_detected": len(output.faces) + output.faces.rejected_count,
                "faces_accepted": len(output.faces),
                "photo_count": output.photo_count,
                "cluster_signature": signature or cluster_signature(config.cluster),
                "model_version": config.embedding_model_version,
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        "application/json",
    )


def count_processed(store: ObjectStore, paths: EventPaths) -> tuple[int, int]:
    """Photos finished, and photos that failed, straight off Stage 1's journal.

    Read rather than instrumented on purpose: `state.jsonl` is already the
    authority on what is done (a photo counts as done only when its line lands
    there), so progress needs no hook inside the expensive stage and stays
    correct across a resume.
    """
    if not store.exists(paths.state):
        return 0, 0
    done: dict[str, str] = {}
    try:
        text = store.get_text(paths.state)
    except (OSError, UnicodeDecodeError):
        return 0, 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn last line, mid-write
        done[record.get("photo_id", "")] = record.get("status", "")
    return (
        sum(1 for status in done.values() if status == "done"),
        sum(1 for status in done.values() if status == "failed"),
    )


def load_photos(
    event_id: str, store: ObjectStore, config: Config
) -> dict[str, PhotoInfo]:
    import io

    import pyarrow.parquet as pq

    paths = EventPaths(config.storage.bucket, event_id)
    if not store.exists(paths.photos_parquet):
        return {}
    table = pq.read_table(io.BytesIO(store.get_bytes(paths.photos_parquet)))
    data = table.to_pydict()
    photos: dict[str, PhotoInfo] = {}
    for index, photo_id in enumerate(data["photo_id"]):
        photos[photo_id] = PhotoInfo(
            photo_id=photo_id,
            path=data["path"][index] or "",
            width=int(data["width"][index] or 0),
            height=int(data["height"][index] or 0),
            taken_at=data["taken_at"][index],
            face_count=int(data["face_count"][index] or 0),
            accepted_count=int(data["accepted_count"][index] or 0),
        )
    return photos


def load_face_boxes(
    event_id: str, store: ObjectStore, config: Config
) -> dict[str, tuple[float, float, float, float]]:
    """face_id -> (left%, top%, width%, height%) within its own photo."""
    import io

    import pyarrow.parquet as pq

    paths = EventPaths(config.storage.bucket, event_id)
    if not store.exists(paths.faces_parquet):
        return {}
    table = pq.read_table(
        io.BytesIO(store.get_bytes(paths.faces_parquet)),
        columns=["face_id", "photo_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"],
    )
    data = table.to_pydict()
    photos = load_photos(event_id, store, config)

    boxes: dict[str, tuple[float, float, float, float]] = {}
    for index, face_id in enumerate(data["face_id"]):
        photo = photos.get(data["photo_id"][index])
        if not photo or not photo.width or not photo.height:
            continue
        boxes[face_id] = (
            round(float(data["bbox_x"][index]) / photo.width * 100, 3),
            round(float(data["bbox_y"][index]) / photo.height * 100, 3),
            round(float(data["bbox_w"][index]) / photo.width * 100, 3),
            round(float(data["bbox_h"][index]) / photo.height * 100, 3),
        )
    return boxes


# ---------------------------------------------------------------------------
# Watching a run happen -- reading Stage 1's journal as it is written
# ---------------------------------------------------------------------------


@dataclass
class LiveFace:
    """One face, as it lands, before any grouping exists."""

    face_id: str
    photo_id: str
    height_px: int
    accepted: bool
    second_class: bool
    reject_reason: str = ""
    photo_name: str = ""


@dataclass
class LiveTotals:
    detected: int = 0
    accepted: int = 0
    too_small: int = 0
    photos_seen: int = 0
    photos_without_faces: int = 0
    biggest_photo: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def usable_pct(self) -> float:
        if not self.detected:
            return 0.0
        return round(self.accepted / self.detected * 100)


class LiveTail:
    """Follows faces.jsonl while Stage 1 writes it.

    Stage 1 already journals every face the moment it is found -- the crop is on
    disk before the photo is even marked done -- so a live view needs no hook
    inside the expensive stage, exactly like the progress bar. What it does need
    is to not re-read the whole file every two seconds: at the end of a
    4,000-photo wedding that journal is tens of megabytes, and polling it whole
    would cost more disk than the pipeline itself.

    So this keeps a byte offset per event and reads only what is new, carrying
    a partial final line forward -- the file is being appended to underneath us,
    so the last line is routinely half-written.

    (On R2 this becomes a ranged GET, which is the same idea and one line of
    boto3. It is a tail, not a filesystem assumption.)
    """

    def __init__(self, keep: int = 60) -> None:
        self._keep = keep
        self._lock = threading.RLock()
        self._offset: dict[str, int] = {}
        self._totals: dict[str, LiveTotals] = {}
        self._recent: dict[str, list[LiveFace]] = {}
        self._partial: dict[str, str] = {}
        self._last_photo: dict[str, str] = {}

    def reset(self, uri: str) -> None:
        with self._lock:
            self._offset.pop(uri, None)
            self._totals.pop(uri, None)
            self._recent.pop(uri, None)
            self._partial.pop(uri, None)
            self._last_photo.pop(uri, None)

    def read(self, store: ObjectStore, uri: str) -> tuple[list[LiveFace], LiveTotals]:
        if not store.exists(uri):
            return [], LiveTotals()
        with self._lock:
            offset = self._offset.get(uri, 0)
            totals = self._totals.get(uri) or LiveTotals()
            recent = list(self._recent.get(uri, []))
            partial = self._partial.get(uri, "")
            last_photo = self._last_photo.get(uri, "")

        chunk = b""
        try:
            with store.open_read(uri) as handle:
                handle.seek(0, 2)
                size = handle.tell()
                if size < offset:  # the journal was replaced -- start over
                    offset, totals, recent, partial, last_photo = (
                        0,
                        LiveTotals(),
                        [],
                        "",
                        "",
                    )
                handle.seek(offset)
                chunk = handle.read()
                offset = offset + len(chunk)
        except (OSError, ValueError):
            return recent, totals

        text = partial + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        partial = lines.pop() if lines else ""

        photos_in_chunk: dict[str, int] = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            face = LiveFace(
                face_id=record.get("face_id", ""),
                photo_id=record.get("photo_id", ""),
                height_px=int(float(record.get("face_h_px") or 0)),
                accepted=bool(record.get("accepted")),
                second_class=bool(record.get("second_class")),
                reject_reason=(record.get("reject_reason") or "").split(" (")[0],
                photo_name=str(record.get("photo_path") or "").replace("\\", "/").split("/")[-1],
            )
            # Faces are journalled photo by photo, in order, so a change of
            # photo_id is a new photo -- no set of every id seen is needed.
            if face.photo_id != last_photo:
                totals.photos_seen += 1
                last_photo = face.photo_id
            totals.detected += 1
            if face.accepted:
                totals.accepted += 1
                if face.height_px < 112:
                    totals.too_small += 1
                recent.append(face)
            elif face.reject_reason:
                totals.reject_reasons[face.reject_reason] = (
                    totals.reject_reasons.get(face.reject_reason, 0) + 1
                )
            photos_in_chunk[face.photo_id] = photos_in_chunk.get(face.photo_id, 0) + 1

        if photos_in_chunk:
            totals.biggest_photo = max(
                totals.biggest_photo, max(photos_in_chunk.values())
            )
        del recent[: max(0, len(recent) - self._keep)]

        with self._lock:
            self._offset[uri] = offset
            self._totals[uri] = totals
            self._recent[uri] = recent
            self._partial[uri] = partial
            self._last_photo[uri] = last_photo
        return recent, totals


class ProgressTail:
    """Counts finished photos off `state.jsonl` without re-reading it.

    The same byte-offset trick as `LiveTail`, on the other journal, for the same
    reason. `count_processed` reads the file whole, which is right for the
    one-shot callers -- the dashboard, and the start and end of a run -- and
    wrong for the progress watcher, which asks every 1.5 seconds for the entire
    length of a run. Measured on a real 6,933-photo folder: 18.7 ms per call at
    2,450 photos, and it grows with the folder, so the cost is worst exactly
    when the run is longest.

    This is a tail, not a cache. Every call still reads whatever bytes have
    landed since the last one, so it cannot answer with a stale number -- which
    is the property that matters here, since a progress bar that lags during the
    one operation it exists to measure is worse than no progress bar.

    Status is kept per photo rather than as a running total because Stage 1's
    journal is append-only: a retried photo writes a second line, and counting
    lines would count it twice and sail the bar past 100%. Last line wins, which
    is what `count_processed` does with its dictionary.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._offset = 0
        self._partial = ""
        self._status: dict[str, str] = {}
        self._done = 0
        self._failed = 0

    def read(self, store: ObjectStore, uri: str) -> tuple[int, int]:
        with self._lock:
            if not store.exists(uri):
                return self._done, self._failed

            chunk = b""
            try:
                with store.open_read(uri) as handle:
                    handle.seek(0, 2)
                    size = handle.tell()
                    if size < self._offset:  # journal replaced -- start over
                        self._offset, self._partial = 0, ""
                        self._status, self._done, self._failed = {}, 0, 0
                    handle.seek(self._offset)
                    chunk = handle.read()
                    self._offset += len(chunk)
            except (OSError, ValueError):
                # Last known good, never zero: a transient read failure must not
                # snap the progress bar back to the start of the folder.
                return self._done, self._failed

            text = self._partial + chunk.decode("utf-8", errors="replace")
            lines = text.split("\n")
            self._partial = lines.pop() if lines else ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn line, or a write we cannot read
                photo_id = record.get("photo_id", "")
                status = record.get("status", "")
                previous = self._status.get(photo_id)
                if previous == status:
                    continue
                if previous == "done":
                    self._done -= 1
                elif previous == "failed":
                    self._failed -= 1
                self._status[photo_id] = status
                if status == "done":
                    self._done += 1
                elif status == "failed":
                    self._failed += 1

            return self._done, self._failed


# ---------------------------------------------------------------------------
# The dashboard
# ---------------------------------------------------------------------------


def _normalise_folder(path: str) -> str:
    """So the same folder ingested twice is recognised as the same photos."""
    return path.replace("\\", "/").rstrip("/").strip().lower()


@dataclass
class Dashboard:
    """Everything the home page states, computed once per request."""

    events: list[EventSummary] = field(default_factory=list)
    totals: RunTotals = field(default_factory=RunTotals)

    photos_processed: int = 0
    distinct_photos: int = 0
    duplicate_folders: int = 0
    faces_detected: int = 0
    faces_accepted: int = 0
    people_found: int = 0
    events_unmeasured: int = 0
    source_gb: float = 0.0
    machine_seconds: float = 0.0

    @property
    def has_timing(self) -> bool:
        return self.totals.has_timing

    @property
    def seconds_per_photo(self) -> float:
        return self.totals.seconds_per_photo

    @property
    def reprocessed_photos(self) -> int:
        return max(0, self.photos_processed - self.distinct_photos)


def measured_rate(store: ObjectStore, config: Config) -> RunTotals:
    """Just the throughput, from the run journals and nothing else.

    The forecast endpoint fires on every keystroke, and building the whole
    dashboard for it meant re-reading every state.jsonl, parquet and feedback
    file to multiply one number by a constant -- each reply took ~0.7s, so the
    figure on screen lagged a character behind what was typed. These journals
    are a few hundred bytes each.
    """
    records = []
    for event_id in _event_ids(store, config):
        records.extend(read_runs(store, config.storage.bucket, event_id))
    return aggregate_runs(records)


def build_dashboard(store: ObjectStore, config: Config) -> Dashboard:
    events = list_events(store, config)
    board = Dashboard(events=events)

    # Every file this page needs was already opened while the summaries were
    # built. Re-reading the journals and parquets here doubled the disk work
    # for every folder on the list -- and the list only grows.
    records: list[RunRecord] = []
    by_folder: dict[str, list[EventSummary]] = {}
    for event in events:
        records.extend(event.runs)
        board.photos_processed += event.processed
        board.faces_detected += event.faces_detected
        board.faces_accepted += event.faces_accepted
        board.source_gb += event.total_bytes / 1e9
        if event.total_seconds:
            board.machine_seconds += event.total_seconds
        if event.people_found is None:
            if event.has_faces:
                board.events_unmeasured += 1
        else:
            board.people_found += event.people_found
        key = _normalise_folder(event.source_folder) or f"?{event.event_id}"
        by_folder.setdefault(key, []).append(event)

    board.totals = aggregate_runs(records)

    # Distinct photos: the same folder processed twice (a tiling A/B, say) is
    # 851 photos of evidence, not 1,702. Reporting the raw sum would flatter us.
    for group in by_folder.values():
        board.distinct_photos += max(event.processed for event in group)
        if len(group) > 1:
            board.duplicate_folders += 1

    return board


# ---------------------------------------------------------------------------
# The Stage 2 cache -- what makes the threshold sliders feel instant
# ---------------------------------------------------------------------------


def cluster_signature(cluster_config: ClusterConfig) -> str:
    return (
        f"p1={cluster_config.pass1_threshold:.4f}"
        f":p2={cluster_config.pass2_threshold:.4f}"
        f":min={cluster_config.min_faces_per_person}"
        f":sc={cluster_config.second_class_assign_threshold:.4f}"
        f":sp={int(cluster_config.enforce_same_photo_rule)}"
    )


class EventCache:
    """One decoded parquet per event, many clusterings over it.

    Thread-safe because Flask's dev server is threaded and a background Stage 1
    job may be writing while a request reads.
    """

    def __init__(self, store: ObjectStore, config: Config) -> None:
        self._store = store
        self._config = config
        self._lock = threading.RLock()
        self._faces: dict[str, tuple[int, FaceTable]] = {}
        self._outputs: dict[tuple[str, str], Stage2Output] = {}
        self._boxes: dict[str, dict[str, tuple[float, float, float, float]]] = {}
        self._photos: dict[str, dict[str, PhotoInfo]] = {}
        self._negatives: dict[str, np.ndarray] = {}

    def invalidate(self, event_id: str) -> None:
        """Called after Stage 1 writes a new parquet."""
        with self._lock:
            self._faces.pop(event_id, None)
            self._boxes.pop(event_id, None)
            self._photos.pop(event_id, None)
            self._negatives.pop(event_id, None)
            for key in [k for k in self._outputs if k[0] == event_id]:
                self._outputs.pop(key, None)

    def faces(self, event_id: str) -> FaceTable:
        paths = EventPaths(self._config.storage.bucket, event_id)
        # Cheap staleness check: Stage 1 rewrites the whole parquet on
        # consolidation, so its size moving means the cache is stale.
        stamp = len(self._store.get_bytes(paths.faces_parquet))
        with self._lock:
            cached = self._faces.get(event_id)
            if cached and cached[0] == stamp:
                return cached[1]
        table = load_faces(event_id, self._store, self._config)
        with self._lock:
            self._faces[event_id] = (stamp, table)
        return table

    def photos(self, event_id: str) -> dict[str, PhotoInfo]:
        with self._lock:
            if event_id in self._photos:
                return self._photos[event_id]
        loaded = load_photos(event_id, self._store, self._config)
        with self._lock:
            self._photos[event_id] = loaded
        return loaded

    def boxes(self, event_id: str) -> dict[str, tuple[float, float, float, float]]:
        with self._lock:
            if event_id in self._boxes:
                return self._boxes[event_id]
        loaded = load_face_boxes(event_id, self._store, self._config)
        with self._lock:
            self._boxes[event_id] = loaded
        return loaded

    def stage2(
        self, event_id: str, cluster_config: ClusterConfig | None = None
    ) -> Stage2Output:
        settings = cluster_config or self._config.cluster
        signature = cluster_signature(settings)
        key = (event_id, signature)
        with self._lock:
            if key in self._outputs:
                return self._outputs[key]

        faces = self.faces(event_id)
        result = cluster_faces(
            embeddings=faces.embeddings,
            photo_ids=faces.photo_ids,
            quality_scores=faces.quality,
            second_class=faces.second_class,
            config=settings,
        )
        photo_count = len(set(faces.photo_ids))
        paths = EventPaths(self._config.storage.bucket, event_id)
        if self._store.exists(paths.event_json):
            try:
                photo_count = json.loads(
                    self._store.get_text(paths.event_json)
                ).get("photo_count", photo_count)
            except (json.JSONDecodeError, KeyError):
                pass

        output = Stage2Output(
            event_id=event_id,
            faces=faces,
            result=result,
            sanity=sanity_check(result, photo_count, settings),
            photo_count=photo_count,
        )
        truth = _ground_truth_from_paths(faces)
        if truth:
            output.ground_truth = truth
            output.evaluation = evaluate(result.labels, truth)

        with self._lock:
            # Keep the cache from growing without bound while somebody drags a
            # slider. Two events open at a handful of settings each is plenty.
            if len(self._outputs) > 24:
                self._outputs.clear()
            self._outputs[key] = output

        # Cache the result for the dashboard, but ONLY at the saved defaults --
        # otherwise the home page would report whichever threshold somebody last
        # dragged a slider to, which is not the state of the project.
        if signature == cluster_signature(self._config.cluster):
            try:
                write_event_summary(self._store, self._config, event_id, output)
            except OSError:
                pass  # a cache write failing must not break the page
        return output

    def auto_threshold(
        self, event_id: str, different_pairs: list[tuple[int, int]] | None = None
    ) -> AutoThreshold:
        """Where this folder's own photos say the merge line belongs.

        The expensive half -- enumerating every pair of faces that share a photo
        and measuring them -- is cached per event, because it depends only on
        Stage 1's output and never on a setting. Folding in new hand judgements
        afterwards is then arithmetic on a handful of pairs.
        """
        faces = self.faces(event_id)
        with self._lock:
            cached = self._negatives.get(event_id)
        if cached is None:
            cached = same_photo_distances(faces.embeddings, faces.photo_ids)
            with self._lock:
                self._negatives[event_id] = cached
        return choose_threshold(
            faces.embeddings,
            faces.photo_ids,
            default=self._config.cluster.pass2_threshold,
            extra_different_pairs=different_pairs,
            precomputed=cached,
        )

    def distance_fn(self, event_id: str):
        """(face_a, face_b) -> cosine distance, for threshold evidence."""
        faces = self.faces(event_id)
        index = {face_id: i for i, face_id in enumerate(faces.face_ids)}
        vectors = faces.embeddings

        def distance(face_a: str, face_b: str) -> float | None:
            i, j = index.get(face_a), index.get(face_b)
            if i is None or j is None:
                return None
            a, b = vectors[i], vectors[j]
            norm = float(np.linalg.norm(a) * np.linalg.norm(b))
            if norm == 0.0:
                return None
            return float(1.0 - np.dot(a, b) / norm)

        return distance
