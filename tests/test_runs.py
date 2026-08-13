"""Tests for run timing.

The point of this feature is that a duration survives the process that measured
it. So these tests care about the ways a duration could be lost or, worse,
quietly wrong -- a fabricated or mis-attributed number is more damaging than a
missing one, because every "how long will 5,000 photos take" answer is built on
the average.
"""
from __future__ import annotations

import json
import re

import pytest

from wff.config import Config, StorageConfig
from wff.runs import (
    FALLBACK_SECONDS_PER_PHOTO,
    RunRecord,
    aggregate_runs,
    append_run,
    human_duration,
    human_forecast,
    human_when,
    read_runs,
    runs_uri,
)
from wff.storage.local import LocalObjectStore

BUCKET = "wff"


@pytest.fixture()
def store(tmp_path):
    return LocalObjectStore(str(tmp_path / "store"))


def _record(**kwargs) -> RunRecord:
    base = dict(
        event_id="ev_test",
        reviewer="adarsh",
        total_seconds=100.0,
        processing_seconds=90.0,
        photos_processed=45,
    )
    base.update(kwargs)
    return RunRecord(**base)


# -- the journal -------------------------------------------------------------


def test_a_run_survives_the_process_that_measured_it(store):
    append_run(store, BUCKET, _record(total_seconds=1932.4))
    [read_back] = read_runs(store, BUCKET, "ev_test")
    assert read_back.total_seconds == 1932.4
    assert read_back.reviewer == "adarsh"
    assert read_back.run_id and read_back.finished_at


def test_runs_accumulate_instead_of_overwriting(store):
    append_run(store, BUCKET, _record(reviewer="adarsh"))
    append_run(store, BUCKET, _record(reviewer="devesh"))
    records = read_runs(store, BUCKET, "ev_test")
    assert [r.reviewer for r in records] == ["adarsh", "devesh"]


def test_no_journal_yet_is_empty_not_an_error(store):
    assert read_runs(store, BUCKET, "never_run") == []


def test_a_torn_last_line_does_not_lose_earlier_runs(store):
    append_run(store, BUCKET, _record())
    with store.open_append(runs_uri(BUCKET, "ev_test")) as fh:
        fh.write(b'{"event_id": "ev_test", "total_sec')  # killed mid-write
    assert len(read_runs(store, BUCKET, "ev_test")) == 1


def test_an_unknown_field_from_a_future_version_is_ignored(store):
    with store.open_append(runs_uri(BUCKET, "ev_test")) as fh:
        fh.write(
            (
                json.dumps(
                    {"event_id": "ev_test", "total_seconds": 12.0, "gpu_model": "H100"}
                )
                + "\n"
            ).encode()
        )
    [record] = read_runs(store, BUCKET, "ev_test")
    assert record.total_seconds == 12.0


# -- what a run says about speed --------------------------------------------


def test_a_resumed_run_reports_no_speed(store):
    """THE case that would poison the average. A run that skipped 851 photos and
    processed none took a few seconds and says nothing about throughput;
    counting it as "0 photos in 3 seconds" would drag the forecast to nonsense.
    """
    resumed = _record(photos_processed=0, photos_skipped_done=851, processing_seconds=2.0)
    assert resumed.seconds_per_photo is None
    assert resumed.timed is True  # it still cost real waiting time


def test_a_run_with_no_timing_reports_no_speed():
    assert _record(processing_seconds=0.0).seconds_per_photo is None


def test_speed_is_processing_time_over_photos_actually_done():
    record = _record(photos_processed=100, processing_seconds=227.0)
    assert record.seconds_per_photo == pytest.approx(2.27)


# -- aggregation -------------------------------------------------------------


def test_a_resumed_run_costs_machine_time_but_not_a_speed_sample():
    totals = aggregate_runs(
        [
            _record(photos_processed=10, processing_seconds=8.0, total_seconds=14.0),
            _record(photos_processed=0, processing_seconds=0.0, total_seconds=2.0),
        ]
    )
    assert totals.timed_runs == 2
    assert totals.total_seconds == 16.0        # the 2s of waiting is real
    assert totals.per_photo_samples == 1       # but it is not evidence of speed
    assert totals.measured_seconds_per_photo == pytest.approx(0.8)


def test_throughput_is_weighted_by_photos_not_by_run():
    """A 3-photo run and an 851-photo run are not equally informative. A flat
    mean of the two rates would let a trivial run swing the forecast."""
    totals = aggregate_runs(
        [
            _record(photos_processed=3, processing_seconds=0.3),      # 0.1 s/photo
            _record(photos_processed=900, processing_seconds=1800.0),  # 2.0 s/photo
        ]
    )
    flat_mean = (0.1 + 2.0) / 2
    assert totals.measured_seconds_per_photo == pytest.approx(1800.3 / 903)
    assert totals.measured_seconds_per_photo > flat_mean


def test_fastest_and_slowest_come_from_real_runs_only():
    totals = aggregate_runs(
        [
            _record(photos_processed=10, processing_seconds=5.0),   # 0.5
            _record(photos_processed=10, processing_seconds=30.0),  # 3.0
            _record(photos_processed=0, processing_seconds=0.0),    # no opinion
        ]
    )
    assert totals.fastest_seconds_per_photo == pytest.approx(0.5)
    assert totals.slowest_seconds_per_photo == pytest.approx(3.0)


def test_with_nothing_timed_the_forecast_says_so():
    totals = aggregate_runs([])
    assert totals.has_timing is False
    assert totals.seconds_per_photo == FALLBACK_SECONDS_PER_PHOTO


def test_a_failed_run_still_counts_its_machine_time():
    """A run that died after 40 minutes cost 40 minutes. Dropping it would make
    the machine look faster than it is."""
    totals = aggregate_runs(
        [_record(phase="failed", total_seconds=2400.0, photos_processed=0, processing_seconds=0.0)]
    )
    assert totals.total_seconds == 2400.0
    assert totals.per_photo_samples == 0


# -- how durations read ------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (5, "5s"),
        (65, "1m 05s"),
        (1932, "32m 12s"),
        (4027, "1h 07m"),
    ],
)
def test_durations_read_like_a_person_wrote_them(seconds, expected):
    assert human_duration(seconds) == expected


def test_a_missing_duration_never_reads_as_zero():
    """"0s" would claim a run was instant. The six folders processed before this
    feature existed have no duration, and must say so."""
    assert human_duration(None) == "not recorded"
    assert human_duration(0) == "not recorded"


def test_a_genuinely_instant_run_is_not_confused_with_an_untimed_one():
    """Found by running it: a run that failed in 3 ms rounded to 0.0 seconds,
    which then rendered as "not recorded" -- the same as the legacy folders that
    were never measured. Sub-second runs are floored when recorded and read as
    "under 1s" so the two stay distinguishable.
    """
    assert human_duration(0.01) == "under 1s"
    assert human_duration(0.4) == "under 1s"
    assert RunRecord(event_id="e", total_seconds=0.01).timed is True


def test_a_forecast_is_rounded_not_precise():
    assert human_forecast(5000 * 2.27) == "3h 09m"
    assert human_forecast(30) == "30 seconds"


# -- the console records what it ran ----------------------------------------


def test_console_run_is_recorded_with_who_ran_it(tmp_path):
    """End to end through the job runner: a finished run must be on disk, with
    the right person's name, before it reads as finished.
    """
    import time

    from wff.web.data import EventCache
    from wff.web.jobs import JobRunner

    config = Config(
        storage=StorageConfig(local_root=str(tmp_path / "store"), bucket=BUCKET)
    )
    obj_store = LocalObjectStore(config.storage.local_root)
    runner = JobRunner(obj_store, config, EventCache(obj_store, config))

    job = runner.submit("ev_nosuch", "no://such/place", reviewer="Devesh")
    for _ in range(100):
        if job.finished:
            break
        time.sleep(0.05)

    assert job.finished, "job never reported finished"
    records = read_runs(obj_store, BUCKET, "ev_nosuch")
    assert len(records) == 1, "a failed run must still be recorded"
    assert records[0].reviewer == "Devesh"
    assert records[0].phase == "failed"
    assert records[0].total_seconds > 0


def test_a_job_is_not_finished_until_its_timing_is_written():
    """Found by running it: the phase used to flip to "done" before the journal
    was appended, so a page reloading on that boundary showed "not recorded" for
    a run that had been timed -- and stopping the server there lost it for good.
    """
    from wff.web.jobs import DONE, Job

    job = Job(job_id="j1", event_id="ev", link="x")
    job.phase = DONE
    assert job.finished is False, "done but unwritten must not read as finished"
    job.recorded = True
    assert job.finished is True


def test_the_run_record_matches_the_job_field_names():
    """Found by running it: `_record_run` read `job.photos_processed`, which does
    not exist (the field is `photos_done`), so every run silently failed to
    record while the run itself still reported success.
    """
    import inspect

    from wff.web import jobs

    source = inspect.getsource(jobs.JobRunner._record_run)
    referenced = set(re.findall(r"job\.([a-z_]+)", source))
    known = set(jobs.Job.__dataclass_fields__) | set(dir(jobs.Job))
    assert referenced <= known, f"not on Job: {sorted(referenced - known)}"


# -- dates on the page -------------------------------------------------------


def test_a_missing_stamp_never_becomes_a_date():
    """The six folders processed before timing existed have no run journal. A
    blank must stay blank -- printing today's date for them would be a lie the
    page states confidently."""
    assert human_when("") == ""
    assert human_when(None) == ""
    assert human_when("not a date") == ""


def test_dates_are_shown_on_this_machine_s_clock():
    """Journals are written in UTC so they survive being read elsewhere. A run
    that finished at 9:51 pm must not read as 4:21 pm on the page."""
    from datetime import datetime, timedelta, timezone

    local_now = datetime.now().astimezone()
    stamp = local_now.astimezone(timezone.utc).isoformat()
    assert human_when(stamp).startswith("Today, ")
    assert local_now.strftime("%I:%M %p").lstrip("0").lower() in human_when(stamp)

    yesterday = (local_now - timedelta(days=1)).astimezone(timezone.utc).isoformat()
    assert human_when(yesterday).startswith("Yesterday, ")

    old = local_now.replace(year=2026, month=8, day=1, hour=18, minute=58)
    assert human_when(old.astimezone(timezone.utc).isoformat()) == "1 Aug 2026, 6:58 pm"
