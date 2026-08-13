"""One run of the pipeline, in the background, with a progress bar and a stop
button.

Runs the SAME functions the CLI runs. There is no second pipeline here: if the
console and `python -m wff run` ever disagreed, every number the console showed
would be worthless.

Jobs are serialised through one worker thread on purpose. Stage 1 is CPU-bound
and already uses every core; two events at once would just make both slower and
make the seconds-per-photo figure meaningless for planning.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from ..config import Config
from ..ingest import adapter_for
from ..runs import RunRecord, append_run, human_duration
from ..stage1 import EventPaths, build_manifest, consolidate, process_event
from ..storage import ObjectStore
from .data import EventCache, ProgressTail

# Phases, in the order they happen.
QUEUED = "queued"
CHECKING = "checking"
MANIFEST = "manifest"
LOADING = "loading models"
PROCESSING = "processing"
CONSOLIDATING = "consolidating"
CLUSTERING = "clustering"
DONE = "done"
FAILED = "failed"
STOPPED = "stopped"

TERMINAL = {DONE, FAILED, STOPPED}


@dataclass
class Job:
    job_id: str
    event_id: str
    link: str
    reviewer: str = "unknown"
    phase: str = QUEUED
    message: str = "waiting for the worker"
    photos_total: int = 0
    photos_done: int = 0
    photos_failed: int = 0
    faces_detected: int = 0
    faces_accepted: int = 0
    people_found: int = 0
    seconds_per_photo: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    resumed_from: int = 0
    source_type: str = ""
    log: list[str] = field(default_factory=list)
    # Wall clock, Start pressed to finished. Recorded to runs.jsonl so it
    # outlives the process -- see wff/runs.py for why that mattered.
    total_seconds: float = 0.0
    processing_seconds: float = 0.0

    # Set once the run has been written to runs.jsonl. `finished` waits for it
    # so nothing can observe a completed run whose timing is not yet saved --
    # see the note on `finished` below.
    recorded: bool = False

    @property
    def finished(self) -> bool:
        """Done AND written down.

        Requiring `recorded` closes a real gap: the phase used to flip to
        "done" before the run journal was appended, so a page reloading at that
        instant showed "not recorded" for a run that had in fact been timed, and
        stopping the server on that boundary lost the timing for good.
        """
        return self.phase in TERMINAL and self.recorded

    @property
    def percent(self) -> float:
        if not self.photos_total:
            return 0.0
        return min(100.0, round(self.photos_done / self.photos_total * 100, 1))

    @property
    def eta_seconds(self) -> float:
        remaining = max(0, self.photos_total - self.photos_done)
        if not self.seconds_per_photo or not remaining:
            return 0.0
        return remaining * self.seconds_per_photo

    def note(self, line: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"{stamp}  {line}")
        del self.log[:-200]  # a run is not a logging system

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["percent"] = self.percent
        payload["finished"] = self.finished
        payload["eta_seconds"] = round(self.eta_seconds)
        return payload


class JobRunner:
    """A single worker thread and a dictionary of jobs. Deliberately small."""

    def __init__(self, store: ObjectStore, config: Config, cache: EventCache) -> None:
        self._store = store
        self._config = config
        self._cache = cache
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop_flags: dict[str, threading.Event] = {}
        self._counter = 0
        self._worker = threading.Thread(target=self._loop, daemon=True, name="wff-jobs")
        self._worker.start()

    # -- public ------------------------------------------------------------
    def submit(self, event_id: str, link: str, reviewer: str = "unknown") -> Job:
        with self._lock:
            self._counter += 1
            job_id = f"j{self._counter:04d}"
            job = Job(job_id=job_id, event_id=event_id, link=link, reviewer=reviewer)
            job.note(f"queued: {event_id} (started by {reviewer})")
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._stop_flags[job_id] = threading.Event()
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return [self._jobs[i] for i in reversed(self._order)]

    def active_for_event(self, event_id: str) -> Job | None:
        with self._lock:
            for job_id in reversed(self._order):
                job = self._jobs[job_id]
                if job.event_id == event_id and not job.finished:
                    return job
        return None

    def stop(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            flag = self._stop_flags.get(job_id)
        if not job or not flag or job.finished:
            return False
        flag.set()
        job.note("stop requested -- finishing the current photo")
        job.message = "stopping after the current photo..."
        return True

    # -- the worker --------------------------------------------------------
    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            wall_start = time.time()
            try:
                self._run(job)
            except Exception as exc:  # a job must never kill the worker
                job.phase = FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = job.error
                job.note("FAILED " + job.error)
                job.note(traceback.format_exc(limit=3))
            finally:
                job.finished_at = datetime.now(timezone.utc).isoformat()
                job.total_seconds = time.time() - wall_start
                # Recorded on EVERY ending, including failures and stops. A run
                # that died after 40 minutes cost 40 minutes; leaving it out
                # would make the machine look faster than it is.
                self._record_run(job)
                job.note(f"total time {human_duration(job.total_seconds)}")
                self._queue.task_done()

    def _record_run(self, job: Job) -> None:
        try:
            append_run(
                self._store,
                self._config.storage.bucket,
                RunRecord(
                    event_id=job.event_id,
                    reviewer=job.reviewer,
                    trigger="console",
                    phase=job.phase if job.phase in (DONE, FAILED, STOPPED) else FAILED,
                    started_at=job.started_at,
                    finished_at=job.finished_at,
                    # Floored so an instant failure is still "timed". Rounding a
                    # 3 ms failure to 0.0 would make it indistinguishable from
                    # the legacy runs that were never measured at all.
                    total_seconds=max(0.01, round(job.total_seconds, 2)),
                    processing_seconds=round(job.processing_seconds, 2),
                    photos_total=job.photos_total,
                    # photos_done counts everything finished for this folder,
                    # including photos an earlier run did, so the work THIS run
                    # performed is the difference. Getting this wrong would
                    # credit a resumed run with the whole folder and wreck the
                    # seconds-per-photo average.
                    photos_processed=max(0, job.photos_done - job.resumed_from),
                    photos_skipped_done=job.resumed_from,
                    photos_failed=job.photos_failed,
                    faces_detected=job.faces_detected,
                    faces_accepted=job.faces_accepted,
                    people_found=job.people_found or None,
                    source_type=job.source_type,
                    error=job.error,
                    model_version=self._config.embedding_model_version,
                ),
            )
        except Exception as exc:
            # Losing the timing must never turn a finished run into a failure.
            job.note(f"could not save run timing: {type(exc).__name__}: {exc}")
        finally:
            # Marked even if the write failed, or the job would never read as
            # finished and the page would poll it forever.
            job.recorded = True

    def _run(self, job: Job) -> None:
        cfg = self._config
        paths = EventPaths(cfg.storage.bucket, job.event_id)
        stop_flag = self._stop_flags[job.job_id]
        job.started_at = datetime.now(timezone.utc).isoformat()

        # -- is the source real? ------------------------------------------
        job.phase = CHECKING
        job.message = "checking the folder"
        adapter = adapter_for(job.link)
        job.source_type = adapter.source_type
        check = adapter.validate()
        job.note(f"[{adapter.source_type}] {check.message}")
        if not check.ok:
            job.phase = FAILED
            job.error = check.message
            job.message = check.message
            return

        # -- what is in there? --------------------------------------------
        if not self._store.exists(paths.manifest):
            job.phase = MANIFEST
            job.message = "listing the photos"
            entries, stats = build_manifest(adapter, job.event_id, self._store, cfg)
            job.photos_total = len(entries)
            job.note(
                f"found {len(entries):,} images, {stats.total_bytes / 1e9:.1f} GB"
                + (
                    f" (skipped {stats.videos} videos, {stats.raw} RAW, "
                    f"{stats.other} other)"
                    if (stats.videos or stats.raw or stats.other)
                    else ""
                )
            )
            if not entries:
                job.phase = FAILED
                job.error = "No images found in that folder."
                job.message = job.error
                return
        else:
            from ..stage1 import read_manifest

            job.photos_total = len(read_manifest(self._store, paths))
            job.note(f"manifest already exists: {job.photos_total:,} images")

        # One tail for the whole job: the first read is the full journal (which
        # is what "how much did an earlier run already do?" needs anyway), and
        # every read after it -- including the watcher's, every 1.5 seconds --
        # costs only the lines written since. Shared with the watcher thread on
        # purpose, so both threads count from one place and cannot disagree.
        progress = ProgressTail()
        already, failed = progress.read(self._store, paths.state)
        job.photos_done = already
        job.photos_failed = failed
        job.resumed_from = already
        if already:
            job.note(f"resuming -- {already:,} photos already done, skipping them")

        # -- the expensive stage -------------------------------------------
        job.phase = LOADING
        job.message = "loading the face models (first run downloads ~275 MB)"

        watcher = threading.Thread(
            target=self._watch_progress,
            args=(job, paths, stop_flag, progress),
            daemon=True,
            name=f"wff-progress-{job.job_id}",
        )
        watcher.start()

        summary = process_event(
            adapter,
            job.event_id,
            store=self._store,
            config=cfg,
            resume=True,
            progress=False,
            should_stop=stop_flag.is_set,
        )
        job.faces_detected = summary.faces_detected
        job.faces_accepted = summary.faces_accepted
        job.photos_failed = summary.photos_failed + failed
        job.processing_seconds = summary.elapsed_seconds
        if summary.elapsed_seconds and summary.photos_processed:
            job.seconds_per_photo = round(
                summary.elapsed_seconds / summary.photos_processed, 2
            )
        job.note(
            f"processed {summary.photos_processed:,} photos, "
            f"{summary.faces_detected:,} faces detected, "
            f"{summary.faces_accepted:,} accepted"
            + (f", {job.seconds_per_photo}s/photo" if job.seconds_per_photo else "")
        )
        for reason, count in sorted(summary.reject_reasons.items(), key=lambda kv: -kv[1]):
            job.note(f"  rejected {count:,}: {reason}")

        # -- the cache Stage 2 reads ---------------------------------------
        job.phase = CONSOLIDATING
        job.message = "writing the face cache"
        face_rows, photo_rows = consolidate(job.event_id, self._store, cfg)
        job.note(f"cached {face_rows:,} face rows across {photo_rows:,} photos")
        self._cache.invalidate(job.event_id)

        done, failed_now = progress.read(self._store, paths.state)
        job.photos_done, job.photos_failed = done, failed_now

        if stop_flag.is_set() and done < job.photos_total:
            job.phase = STOPPED
            job.message = (
                f"stopped at {done:,} of {job.photos_total:,}. "
                "Everything done so far is saved -- run it again to carry on."
            )
            job.note(job.message)
            return

        # -- the cheap stage ------------------------------------------------
        job.phase = CLUSTERING
        job.message = "grouping the faces into people"
        output = self._cache.stage2(job.event_id)
        job.people_found = len(output.result.persons)
        job.note(
            f"{len(output.faces):,} faces -> {output.result.pass1_group_count} tight "
            f"groups -> {job.people_found} people "
            f"({output.result.leftover_face_count:,} leftovers kept)"
        )
        if output.result.blocked_merges_same_photo:
            job.note(
                "the same-photo rule blocked "
                f"{output.result.blocked_merges_same_photo} merges"
            )
        for alarm in output.sanity.alarms:
            job.note("ALARM " + alarm)

        job.phase = DONE
        job.message = f"{job.people_found} people found across {done:,} photos"

    def _watch_progress(
        self,
        job: Job,
        paths: EventPaths,
        stop_flag: threading.Event,
        progress: ProgressTail,
    ) -> None:
        """Poll Stage 1's own journal instead of instrumenting it.

        `state.jsonl` is already the authority on what is finished -- a photo
        counts as done only when its line lands there. Reading it keeps the
        progress bar honest across a resume and keeps the expensive stage free
        of any knowledge that a web page exists.

        Polled through a `ProgressTail` rather than a whole-file read, because
        this loop runs every 1.5 seconds for hours: re-parsing the journal each
        time cost 18.7 ms at 2,450 photos and grew with the folder, so the
        progress bar was quietly taxing the run it was measuring.
        """
        import time

        last_done = job.photos_done
        last_change = time.time()
        while not job.finished and job.phase not in (DONE, FAILED, STOPPED):
            time.sleep(1.5)
            if job.phase in (CONSOLIDATING, CLUSTERING) or job.finished:
                return
            done, failed = progress.read(self._store, paths.state)
            if done != last_done:
                elapsed = time.time() - last_change
                if done > last_done and elapsed > 0:
                    rate = elapsed / (done - last_done)
                    # Smooth it, or the estimate jumps around on every photo.
                    job.seconds_per_photo = round(
                        rate if not job.seconds_per_photo
                        else job.seconds_per_photo * 0.7 + rate * 0.3,
                        2,
                    )
                last_done, last_change = done, time.time()
                job.phase = PROCESSING
            job.photos_done, job.photos_failed = done, failed
            if job.phase == PROCESSING:
                remaining = max(0, job.photos_total - done)
                job.message = (
                    f"{done:,} of {job.photos_total:,} photos"
                    + (
                        f" -- about {_human_time(job.eta_seconds)} left"
                        if remaining and job.seconds_per_photo
                        else ""
                    )
                )
                if stop_flag.is_set():
                    job.message = "stopping after the current photo..."


def _human_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
