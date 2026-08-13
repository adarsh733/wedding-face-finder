"""How long every run took, written down and kept.

WHY THIS FILE EXISTS. Until now the elapsed time of a run was computed while it
ran, printed once, and thrown away when the process exited. So "how long did the
Manali folder take?" had no answer the next morning, and the dashboard could not
show a single honest timing figure. Adarsh asked for total time per run, for
every run, from now on -- that means a journal, not a variable.

The six runs that already exist predate this and have no timing. They are
reported as "not recorded" rather than back-filled with a guess: a fabricated
duration would quietly poison the average that the projection calculator, and
therefore every "how long will 5,000 photos take" answer, is built on.

Both the console and the command line write here. If only the console did, a
`python -m wff run` would be invisible to the dashboard, and the totals would
silently understate the work done.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .storage import ObjectStore, join_uri

# Measured on the 851-photo Manali folder, 12 Aug 2026, tiling on, CPU only.
# Used ONLY to estimate before any run has been timed; the moment a real timed
# run exists, the measured average replaces it and the UI says which is which.
FALLBACK_SECONDS_PER_PHOTO = 2.27

DONE = "done"
FAILED = "failed"
STOPPED = "stopped"


@dataclass
class RunRecord:
    """One attempt at one folder, from the moment Start was pressed."""

    event_id: str
    run_id: str = ""
    reviewer: str = "unknown"
    trigger: str = "console"          # console | cli
    phase: str = DONE                 # done | failed | stopped
    started_at: str = ""
    finished_at: str = ""

    # THE number Adarsh asked for: wall clock, Start to finish. Includes
    # listing the folder, loading the models, processing, and grouping --
    # because that is what "how long did it take" means to a person waiting.
    total_seconds: float = 0.0
    # Stage 1 alone. Kept separately because it is the only part that scales
    # with the number of photos, so it is the only honest basis for a forecast.
    processing_seconds: float = 0.0

    photos_total: int = 0
    photos_processed: int = 0         # actually worked on during THIS run
    photos_skipped_done: int = 0      # already finished by an earlier run
    photos_failed: int = 0
    faces_detected: int = 0
    faces_accepted: int = 0
    people_found: int | None = None
    source_type: str = ""
    error: str = ""
    model_version: str = ""

    @property
    def seconds_per_photo(self) -> float | None:
        """None, not zero, when this run cannot tell us anything about speed.

        A resumed run that skipped 851 photos and processed 3 says nothing
        useful about throughput, and averaging it in would drag the forecast
        toward nonsense.
        """
        if self.photos_processed <= 0 or self.processing_seconds <= 0:
            return None
        return self.processing_seconds / self.photos_processed

    @property
    def timed(self) -> bool:
        return self.total_seconds > 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_dict(raw: dict) -> "RunRecord":
        allowed = RunRecord.__dataclass_fields__
        clean = {k: v for k, v in raw.items() if k in allowed and v is not None}
        clean.setdefault("event_id", "?")
        return RunRecord(**clean)


def runs_uri(bucket: str, event_id: str) -> str:
    return join_uri(f"r2://{bucket}/events/{event_id}", "runs.jsonl")


def append_run(store: ObjectStore, bucket: str, record: RunRecord) -> RunRecord:
    """Append-only, same discipline as Stage 1's journals.

    Append rather than overwrite so a crash can never destroy the history of
    earlier runs, and two processes writing at once cannot clobber each other.
    """
    record.run_id = record.run_id or uuid.uuid4().hex[:12]
    record.finished_at = record.finished_at or datetime.now(timezone.utc).isoformat()
    with store.open_append(runs_uri(bucket, record.event_id)) as fh:
        fh.write((record.to_json() + "\n").encode("utf-8"))
    return record


def read_runs(store: ObjectStore, bucket: str, event_id: str) -> list[RunRecord]:
    uri = runs_uri(bucket, event_id)
    if not store.exists(uri):
        return []
    records: list[RunRecord] = []
    for line in store.get_text(uri).splitlines():
        if not line.strip():
            continue
        try:
            records.append(RunRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            continue  # torn last line from a crash mid-write
    return records


@dataclass
class RunTotals:
    """What the dashboard shows about time and throughput."""

    runs: int = 0
    timed_runs: int = 0
    total_seconds: float = 0.0
    photos_processed: int = 0
    photos_failed: int = 0
    faces_detected: int = 0
    faces_accepted: int = 0
    fastest_seconds_per_photo: float | None = None
    slowest_seconds_per_photo: float | None = None
    measured_seconds_per_photo: float | None = None
    per_photo_samples: int = 0

    @property
    def has_timing(self) -> bool:
        return self.measured_seconds_per_photo is not None

    @property
    def seconds_per_photo(self) -> float:
        """What the forecast uses. Measured if we have it, otherwise the
        recorded fallback -- and `has_timing` tells the UI which to say."""
        return self.measured_seconds_per_photo or FALLBACK_SECONDS_PER_PHOTO


def aggregate_runs(records: list[RunRecord]) -> RunTotals:
    """Roll runs up into the dashboard's numbers.

    Throughput is a photo-weighted mean, not a mean of the per-run rates: a
    3-photo run and an 851-photo run are not equally informative about how long
    the next folder will take, and averaging their rates flat would let a tiny
    run swing the forecast.
    """
    totals = RunTotals(runs=len(records))
    weighted_seconds = 0.0
    weighted_photos = 0

    for record in records:
        totals.photos_processed += record.photos_processed
        totals.photos_failed += record.photos_failed
        totals.faces_detected += record.faces_detected
        totals.faces_accepted += record.faces_accepted
        if record.timed:
            totals.timed_runs += 1
            totals.total_seconds += record.total_seconds

        rate = record.seconds_per_photo
        if rate is None:
            continue
        totals.per_photo_samples += 1
        weighted_seconds += record.processing_seconds
        weighted_photos += record.photos_processed
        if totals.fastest_seconds_per_photo is None or rate < totals.fastest_seconds_per_photo:
            totals.fastest_seconds_per_photo = rate
        if totals.slowest_seconds_per_photo is None or rate > totals.slowest_seconds_per_photo:
            totals.slowest_seconds_per_photo = rate

    if weighted_photos:
        totals.measured_seconds_per_photo = weighted_seconds / weighted_photos
    return totals


def human_duration(seconds: float | None) -> str:
    """'32m 12s', '1h 07m', '5s'. Never '0s' for a missing value.

    Zero and None both mean "we never measured this" -- the six folders
    processed before timing existed. A run that genuinely took a few
    milliseconds is floored to 0.01s when recorded, so it lands in the
    "under 1s" branch and stays distinguishable from an untimed one.
    """
    if seconds is None or seconds <= 0:
        return "not recorded"
    if seconds < 1:
        return "under 1s"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def local_time(stamp: str | None) -> datetime | None:
    """An ISO stamp from a journal, moved onto this machine's wall clock.

    Everything is written in UTC, because a journal may one day be read from a
    different machine in a different place. Nobody reads a page in UTC, though,
    so every date that reaches a screen goes through here first.
    """
    if not stamp:
        return None
    try:
        moment = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone()


def human_when(stamp: str | None) -> str:
    """'Today, 1:07 am' · 'Yesterday, 9:47 pm' · '11 Aug 2026, 6:58 pm'.

    Relative for the two days a person actually remembers, absolute after that.
    Empty string for a missing stamp, so the page can say nothing rather than
    print a fake date.
    """
    moment = local_time(stamp)
    if moment is None:
        return ""
    clock = moment.strftime("%I:%M %p").lstrip("0").lower()
    days = (datetime.now().astimezone().date() - moment.date()).days
    if days == 0:
        return f"Today, {clock}"
    if days == 1:
        return f"Yesterday, {clock}"
    return f"{moment.day} {moment.strftime('%b %Y')}, {clock}"


def human_day(stamp: str | None) -> str:
    """Just the day -- 'Today', '11 Aug 2026'. For grouping headings."""
    moment = local_time(stamp)
    if moment is None:
        return ""
    days = (datetime.now().astimezone().date() - moment.date()).days
    if days == 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    return f"{moment.day} {moment.strftime('%b %Y')}"


def human_forecast(seconds: float) -> str:
    """Same idea, but rounded for a projection -- '3h 09m', not '3h 09m 14s'."""
    if seconds <= 0:
        return "no time at all"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} seconds"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"
