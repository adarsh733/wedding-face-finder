"""The testing console: routes and nothing else.

Everything expensive lives in wff/, unchanged. This file only translates HTTP
into calls the CLI already makes, so the console and the command line can never
drift apart and disagree about what the pipeline found.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from ..config import ClusterConfig, Config, google_api_key, load_config, save_google_api_key
from ..ingest.gdrive import GoogleDriveFolderAdapter
from ..stage1 import EventPaths
from ..storage import ObjectStore, get_store
from . import feedback as fb
from . import meta
from ..runs import human_duration, human_forecast, human_when
from .data import (
    EventCache,
    LiveTail,
    build_dashboard,
    cluster_signature,
    group_by_size,
    list_events,
    load_event_summary,
    measured_rate,
    write_event_summary,
)
from .jobs import (
    CHECKING,
    CLUSTERING,
    CONSOLIDATING,
    LOADING,
    MANIFEST,
    PROCESSING,
    JobRunner,
)

# photo ids are "000481"; face ids are "000481_00". Anything else is not ours.
PHOTO_ID = re.compile(r"^\d{4,10}$")
FACE_ID = re.compile(r"^\d{4,10}_\d{1,3}$")
EVENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

DEFAULT_PEOPLE_PER_PAGE = 50


def group_key(face_ids: list[str]) -> str:
    """A stable name for one exact pile of faces.

    Ticking "this pile is clean" is a statement about the faces in it, not about
    a person_id -- person_ids are reshuffled by every threshold change. Keying
    the tick to the pile's contents means it survives a re-run that produces the
    same pile, and correctly disappears when the pile changes.
    """
    digest = hashlib.sha1("|".join(sorted(face_ids)).encode("utf-8"))
    return "grp:" + digest.hexdigest()[:16]


@dataclass
class ReviewFace:
    """One face crop as the page needs it."""

    face_id: str
    photo_id: str
    face_index: int
    photo_path: str
    height_px: int
    quality: float
    second_class: bool
    flagged: bool = False
    not_a_face: bool = False
    flag_record: str = ""
    truth_label: str | None = None
    is_alien: bool = False


@dataclass
class ReviewProgress:
    """How far through checking this folder somebody is.

    This replaced a panel of six numbers -- precision, recall, pairs merged,
    pairs split -- that answered questions nobody asked. Two facts survive:
    how much is checked, and how many faces turned out to be in the wrong
    pile. Everything else was arithmetic about the arithmetic.
    """

    piles_total: int = 0
    piles_answered: int = 0
    faces_fixed: int = 0
    answered_ids: list[int] = field(default_factory=list)

    @property
    def percent(self) -> int:
        if not self.piles_total:
            return 0
        return int(round(self.piles_answered / self.piles_total * 100))

    @property
    def done(self) -> bool:
        return self.piles_total > 0 and self.piles_answered >= self.piles_total

    @property
    def remaining(self) -> int:
        return max(0, self.piles_total - self.piles_answered)

    def as_dict(self) -> dict:
        return {
            "piles_total": self.piles_total,
            "piles_answered": self.piles_answered,
            "faces_fixed": self.faces_fixed,
            "percent": self.percent,
            "done": self.done,
            "remaining": self.remaining,
        }


@dataclass
class ReviewPerson:
    person_id: int
    label: str
    face_count: int
    photo_count: int
    faces: list[ReviewFace]
    group_key: str
    approved_by: str = ""
    approved_record: str = ""
    contaminated: bool = False
    contamination_note: str = ""
    spread_px: str = ""


KNOBS = ("p1", "p2", "minf", "sc", "sp")


def knobs_in_request() -> bool:
    """Has somebody pinned the settings in the URL?

    Matters because of what the review page does NOT do any more: ask Adarsh to
    choose a threshold. With no knobs in the address the page groups at the
    value the folder's own photos suggest (see cluster/autotune.py). With knobs
    present -- the expert view, or a link somebody pasted -- those win, and the
    page says plainly that it is not showing the automatic result.
    """
    return any(request.args.get(name) not in (None, "") for name in KNOBS)


def cluster_config_from_request(base: ClusterConfig) -> ClusterConfig:
    """Threshold knobs come off the query string so a tuning run is a URL.

    Re-running Stage 2 costs seconds, which is the entire justification for the
    expensive/cheap split in docs/ARCHITECTURE.md. Making the knobs part of the
    address means a bad result can be pasted into chat and reproduced exactly.
    """

    def number(name: str, current: float, low: float, high: float) -> float:
        raw = request.args.get(name)
        if raw in (None, ""):
            return current
        try:
            return max(low, min(high, float(raw)))
        except ValueError:
            return current

    def flag(name: str, current: bool) -> bool:
        raw = request.args.get(name)
        if raw in (None, ""):
            return current
        return raw.strip().lower() in ("1", "true", "yes", "on")

    return ClusterConfig(
        pass1_threshold=number("p1", base.pass1_threshold, 0.05, 0.95),
        pass2_threshold=number("p2", base.pass2_threshold, 0.05, 0.95),
        min_faces_per_person=int(
            number("minf", base.min_faces_per_person, 1, 50)
        ),
        second_class_assign_threshold=number(
            "sc", base.second_class_assign_threshold, 0.05, 0.95
        ),
        enforce_same_photo_rule=flag("sp", base.enforce_same_photo_rule),
        alarm_min_people_per_1000_photos=base.alarm_min_people_per_1000_photos,
        alarm_max_people_per_1000_photos=base.alarm_max_people_per_1000_photos,
    )


def create_app(
    config: Config | None = None, store: ObjectStore | None = None
) -> Flask:
    cfg = config or load_config()
    obj_store = store or get_store(cfg)
    cache = EventCache(obj_store, cfg)
    runner = JobRunner(obj_store, cfg, cache)

    app = Flask(__name__)
    app.config.update(CONFIG=cfg, STORE=obj_store, CACHE=cache, RUNNER=runner)
    app.jinja_env.trim_blocks = True
    app.jinja_env.lstrip_blocks = True

    # -- helpers ----------------------------------------------------------
    def event_paths(event_id: str) -> EventPaths:
        if not EVENT_ID.match(event_id):
            abort(404)
        return EventPaths(cfg.storage.bucket, event_id)

    def feedback_store(event_id: str) -> fb.FeedbackStore:
        return fb.FeedbackStore(obj_store, event_paths(event_id).root)

    tail = LiveTail()

    def _live_payload(store: ObjectStore, paths: EventPaths, job) -> dict:
        """What the screen shows while a folder is being read.

        Faces come off Stage 1's own journal as it is written, so the wall of
        crops fills up within seconds of pressing Start instead of after three
        hours of a progress bar. Nothing here is computed twice: the same
        journal drives the counters, the step list and the thumbnails.
        """
        faces, totals = tail.read(store, paths.faces_jsonl)
        no_faces = 0
        if job is not None and job.photos_done:
            no_faces = max(0, job.photos_done - totals.photos_seen)
        return {
            "steps": _progress_steps(job, totals),
            "faces": [
                {
                    "face_id": face.face_id,
                    "photo_id": face.photo_id,
                    "height_px": face.height_px,
                    "small": face.height_px < 112,
                    "url": url_for(
                        "image", event_id=paths.event_id, kind="crop", name=face.face_id
                    ),
                }
                for face in reversed(faces)
            ],
            "totals": {
                "detected": totals.detected,
                "accepted": totals.accepted,
                "too_small": totals.too_small,
                "usable_pct": totals.usable_pct,
                "photos_with_faces": totals.photos_seen,
                "photos_without_faces": no_faces,
                "biggest_photo": totals.biggest_photo,
                "reject_reasons": totals.reject_reasons,
            },
        }

    def require_processed(event_id: str) -> EventPaths:
        """Guard every JSON route that needs the face cache.

        Without this, an unknown event id reaches the parquet read and comes
        back as Flask's HTML error page -- which the browser then tries to parse
        as JSON, so the reviewer sees "Unexpected token '<'" instead of a
        sentence telling them what is wrong.
        """
        paths = event_paths(event_id)
        if not obj_store.exists(paths.faces_parquet):
            abort(404, description=f"{event_id} has no processed faces yet.")
        return paths

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": getattr(error, "description", "not found")}), 404
        return error, 404

    @app.errorhandler(500)
    def server_error(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "something broke -- check the terminal"}), 500
        return error, 500

    def reviewer_name() -> str:
        payload = request.get_json(silent=True) or {}
        name = (payload.get("reviewer") or request.args.get("reviewer") or "").strip()
        return name[:40] or "anonymous"

    # -- home -------------------------------------------------------------
    @app.get("/")
    def home():
        board = build_dashboard(obj_store, cfg)
        active = [job for job in runner.all() if not job.finished]
        return render_template(
            "index.html",
            board=board,
            events=board.events,
            groups=group_by_size(board.events),
            jobs=runner.all()[:8],
            active=active,
            config=cfg,
            # Drive needs a one-time key. The console is where it gets set --
            # asking someone to open a terminal for it defeats the console.
            key_set=bool(google_api_key()),
            need_key=request.args.get("need_key") == "1",
            pending_link=request.args.get("link", ""),
            pending_event_id=request.args.get("event_id", ""),
            human_duration=human_duration,
            human_forecast=human_forecast,
            human_when=human_when,
        )

    # -- the Google key ----------------------------------------------------
    @app.post("/api/settings/google-key")
    def set_google_key():
        """Save the key, but only after proving it works.

        Saving first and finding out later is how someone ends up with a
        broken key in a file they cannot see, failing every run with the same
        message they thought they had just fixed.
        """
        payload = request.get_json(silent=True) or {}
        key = (payload.get("key") or "").strip()
        link = (payload.get("link") or "").strip().strip('"')

        if not key:
            return jsonify({"ok": False, "message": "Paste the key first."}), 400
        if key.startswith("http") or "/" in key:
            return jsonify(
                {
                    "ok": False,
                    "message": "That looks like a link, not a key. The key is a long "
                    "line of letters and numbers, usually starting with AIza.",
                }
            ), 400

        if not link:
            # Nothing to test against -- take it on trust and say so.
            save_google_api_key(key)
            return jsonify(
                {
                    "ok": True,
                    "tested": False,
                    "message": "Saved. It gets tested the first time you paste a Drive link.",
                }
            )

        check = GoogleDriveFolderAdapter(link, api_key=key).validate()
        if check.ok:
            save_google_api_key(key)
            return jsonify({"ok": True, "tested": True, "message": check.message})

        # A 404 means Google accepted the key and then said the FOLDER is not
        # public. The key is good; keep it, and report the real problem.
        if check.detail.get("status") == 404:
            save_google_api_key(key)
            return jsonify({"ok": False, "key_kept": True, "message": check.message})
        return jsonify({"ok": False, "key_kept": False, "message": check.message})

    # -- the two things a person sets by hand ------------------------------
    @app.post("/api/events/<event_id>/label")
    def set_label(event_id: str):
        """Rename a run, or say who ran it.

        Deliberately one route for both: they are the same kind of claim (a
        human correcting what the machine recorded), they are stored in the
        same document, and splitting them would mean two round trips to fix a
        card that is wrong about both.
        """
        event_paths(event_id)  # 404s on an id that is not ours
        payload = request.get_json(silent=True) or {}
        # NOT "reviewer": every POST from the browser carries a `reviewer` key
        # naming whoever is sitting at the keyboard, so reading the field being
        # SET from that name meant renaming a run silently rewrote who ran it.
        if "name" not in payload and "tested_by" not in payload:
            return jsonify({"error": "nothing to set"}), 400
        label = meta.write_label(
            obj_store,
            cfg.storage.bucket,
            event_id,
            display_name=payload["name"] if "name" in payload else None,
            reviewer=payload["tested_by"] if "tested_by" in payload else None,
            updated_by=reviewer_name(),
        )
        return jsonify(
            {
                "ok": True,
                "name": label.display_name or event_id,
                "renamed": bool(label.display_name),
                "reviewer": label.reviewer,
            }
        )

    @app.get("/api/forecast")
    def forecast():
        """The projection, computed server-side from the measured average.

        Done here rather than in the browser so the page can never quote a
        speed the recorded runs do not support.
        """
        totals = measured_rate(obj_store, cfg)
        try:
            photos = max(0, min(2_000_000, int(request.args.get("photos", 0))))
        except ValueError:
            photos = 0
        return jsonify(
            {
                "photos": photos,
                "seconds": round(photos * totals.seconds_per_photo),
                "text": human_forecast(photos * totals.seconds_per_photo),
                "seconds_per_photo": round(totals.seconds_per_photo, 2),
                "measured": totals.has_timing,
            }
        )

    @app.post("/runs")
    def start_run():
        link = (request.form.get("link") or "").strip().strip('"')
        event_id = (request.form.get("event_id") or "").strip()
        who = (request.form.get("reviewer") or "").strip()[:40] or "unknown"
        if not link:
            return redirect(url_for("home", error="Paste a folder path or a link."))
        if not event_id:
            event_id = _suggest_event_id(link, obj_store, cfg)
        if not EVENT_ID.match(event_id):
            return redirect(
                url_for("home", error="Use letters, numbers, dashes for the name.")
            )
        # A Drive link with no key can only fail. Queueing it would write a
        # failed run to the journal and teach nothing -- ask for the key
        # instead, holding on to what was typed so nothing has to be re-pasted.
        if "drive.google.com" in link.lower() and not google_api_key():
            return redirect(
                url_for("home", need_key="1", link=link, event_id=event_id)
            )

        existing = runner.active_for_event(event_id)
        if existing:
            return redirect(url_for("event_page", event_id=event_id))
        job = runner.submit(event_id, link, reviewer=who)
        return redirect(url_for("event_page", event_id=event_id, job=job.job_id))

    @app.get("/api/events/<event_id>/live")
    def live_progress(event_id: str):
        """Everything the "while it runs" screen shows, in one reply.

        Deliberately one endpoint rather than three: the faces found, the step
        it is on and the counters all have to agree with each other, and three
        separate polls landing between two photos would show a face count that
        does not match the photo count beside it.
        """
        paths = event_paths(event_id)
        job = runner.get(request.args.get("job", "")) or runner.active_for_event(event_id)
        payload = _live_payload(obj_store, paths, job)
        payload["job"] = job.to_dict() if job else None
        return jsonify(payload)

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = runner.get(job_id)
        if job is None:
            abort(404)
        return jsonify(job.to_dict())

    @app.post("/api/jobs/<job_id>/stop")
    def job_stop(job_id: str):
        return jsonify({"stopped": runner.stop(job_id)})

    def resolve_settings(event_id: str, truth=None) -> tuple[ClusterConfig, object]:
        """The settings this page will group at, and why.

        Default path: nobody chooses anything. The threshold is measured off
        this folder's own photos -- every pair of faces sharing a photograph is
        a pair we know to be different people, so the boundary can be found
        without an answer key and without a slider.

        Hand judgements, where they exist, are folded into the same
        measurement: "these two are different" is the same kind of evidence as
        "these two are in one photo", just more expensive to obtain.
        """
        auto = None
        base = cfg.cluster
        if not knobs_in_request():
            try:
                pairs: list[tuple[int, int]] = []
                if truth is not None:
                    faces = cache.faces(event_id)
                    index = {face_id: i for i, face_id in enumerate(faces.face_ids)}
                    for pair in truth.must_not_link:
                        left, right = tuple(pair)[:2]
                        if left in index and right in index:
                            pairs.append((index[left], index[right]))
                auto = cache.auto_threshold(event_id, pairs or None)
                base = replace(base, pass2_threshold=auto.value)
            except (OSError, ValueError, KeyError):
                auto = None  # a measurement failing must not take the page down
        return cluster_config_from_request(base), auto

    # -- the review page ---------------------------------------------------
    @app.get("/e/<event_id>")
    def event_page(event_id: str):
        paths = event_paths(event_id)
        summary = load_event_summary(event_id, obj_store, cfg)
        job = runner.get(request.args.get("job", "")) or runner.active_for_event(event_id)

        if not obj_store.exists(paths.faces_parquet):
            return render_template(
                "event.html",
                event=summary,
                job=job,
                ready=False,
                config=cfg,
                settings=cfg.cluster,
                live=_live_payload(obj_store, paths, job),
                expert=request.args.get("expert") == "1",
            )

        judgements = feedback_store(event_id).live()
        truth = fb.derive_truth(judgements)
        settings, auto = resolve_settings(event_id, truth)
        output = cache.stage2(event_id, settings)

        # Keep the home page's people count in step with what this page shows.
        # It used to be written only when the settings matched the saved
        # defaults, and the automatic threshold is deliberately NOT the saved
        # default -- so without this the dashboard would report "not counted"
        # for every folder forever.
        if auto is not None:
            try:
                write_event_summary(
                    obj_store, cfg, event_id, output, signature=cluster_signature(settings)
                )
            except OSError:
                pass

        people, leftovers = _build_review(output, judgements, truth)

        page = max(1, int(request.args.get("page", 1) or 1))
        per_page = int(request.args.get("per", DEFAULT_PEOPLE_PER_PAGE) or 0) or len(people)
        pages = max(1, -(-len(people) // per_page)) if per_page else 1
        page = min(page, pages)
        visible = people[(page - 1) * per_page : page * per_page] if per_page else people

        score = fb.score_clustering(
            truth, output.faces.face_ids, list(output.result.labels)
        )
        evidence = fb.threshold_evidence(
            truth, output.faces.face_ids, cache.distance_fn(event_id)
        )

        return render_template(
            "event.html",
            event=summary,
            job=job,
            ready=True,
            output=output,
            settings=settings,
            defaults=cfg.cluster,
            config=cfg,
            people=visible,
            people_total=len(people),
            leftovers=leftovers,
            page=page,
            pages=pages,
            per_page=per_page,
            score=score,
            evidence=evidence,
            progress=_review_progress(output, judgements),
            judgement_count=len([j for j in judgements if j.kind != fb.NOTE]),
            notes=[j for j in judgements if j.kind == fb.NOTE],
            size_bands=_size_bands(output.faces.face_h_px),
            dirty=cluster_signature(settings) != cluster_signature(cfg.cluster),
            auto=auto,
            expert=request.args.get("expert") == "1" or knobs_in_request(),
            live=_live_payload(obj_store, paths, job),
        )

    @app.get("/e/<event_id>/judge")
    def judge_page(event_id: str):
        paths = event_paths(event_id)
        if not obj_store.exists(paths.faces_parquet):
            return redirect(url_for("event_page", event_id=event_id))
        judgements = feedback_store(event_id).live()
        truth = fb.derive_truth(judgements)
        # Same settings the review page used, so the pairs offered here are the
        # pairs that would actually move THIS grouping.
        settings, _auto = resolve_settings(event_id, truth)
        output = cache.stage2(event_id, settings)
        evidence = fb.threshold_evidence(
            truth, output.faces.face_ids, cache.distance_fn(event_id)
        )
        pairs = _candidate_pairs(output, cache.distance_fn(event_id), truth, settings)
        return render_template(
            "judge.html",
            event=load_event_summary(event_id, obj_store, cfg),
            pairs=pairs,
            settings=settings,
            evidence=evidence,
            config=cfg,
        )

    @app.get("/api/events/<event_id>/photo/<photo_id>")
    def photo_detail(event_id: str, photo_id: str):
        require_processed(event_id)
        if not PHOTO_ID.match(photo_id):
            abort(404)
        photos = cache.photos(event_id)
        info = photos.get(photo_id)
        if info is None:
            abort(404)
        boxes = cache.boxes(event_id)
        faces = [
            {"face_id": face_id, "box": box}
            for face_id, box in boxes.items()
            if face_id.split("_")[0] == photo_id
        ]
        return jsonify(
            {
                "photo_id": photo_id,
                "path": info.path,
                "width": info.width,
                "height": info.height,
                "taken_at": info.taken_at,
                "faces": faces,
                "url": url_for("image", event_id=event_id, kind="prev", name=photo_id),
            }
        )

    # -- feedback ----------------------------------------------------------
    @app.post("/api/events/<event_id>/feedback")
    def post_feedback(event_id: str):
        require_processed(event_id)
        payload = request.get_json(silent=True) or {}
        kind = payload.get("kind")
        if kind not in fb.KINDS:
            return jsonify({"error": f"unknown kind {kind!r}"}), 400

        store_fb = feedback_store(event_id)
        settings = cluster_config_from_request(cfg.cluster)
        output = cache.stage2(event_id, settings)
        by_person = _faces_by_person(output)
        face_ids = set(output.faces.face_ids)

        judgement = fb.Judgement(
            kind=kind,
            reviewer=reviewer_name(),
            model_version=cfg.embedding_model_version,
            text=(payload.get("text") or "")[:2000],
        )

        if kind == fb.PERSON_OK:
            person_id = int(payload.get("person_id", -1))
            members = by_person.get(person_id)
            if not members:
                return jsonify({"error": "no such pile"}), 400
            flagged = _flagged_faces(store_fb)
            keep = [f for f in members if f not in flagged]
            if len(keep) < 2:
                return jsonify({"error": "not enough faces left to compare"}), 400
            judgement.same_group = keep
            judgement.subject = group_key(members)
            judgement.person_label = f"Person {person_id + 1}"

        elif kind == fb.FACE_WRONG:
            face_id = payload.get("face_id", "")
            person_id = int(payload.get("person_id", -1))
            members = by_person.get(person_id, [])
            if face_id not in face_ids or face_id not in members:
                return jsonify({"error": "face is not in that pile"}), 400
            others = [f for f in members if f != face_id]
            if not others:
                return jsonify({"error": "nothing to compare against"}), 400
            # If this pile was already approved, that approval and this flag
            # contradict each other, so the old one has to go. But dropping it
            # outright would throw away the reviewer's real finding -- they said
            # "all one person", and are now saying "except this one". So the
            # replacement record carries BOTH claims: the rest of the pile stays
            # asserted as one person, and this face is separated from all of it.
            replaced = None
            for record in store_fb.live():
                if record.kind == fb.PERSON_OK and record.subject == group_key(members):
                    store_fb.retract(record.record_id, judgement.reviewer)
                    replaced = record
            judgement.different_pairs = [[face_id, other] for other in others]
            if replaced is not None:
                judgement.same_group = [
                    f for f in replaced.same_group if f != face_id
                ]
            judgement.subject = face_id
            judgement.person_label = f"Person {person_id + 1}"

        elif kind == fb.SAME_PERSON:
            a = int(payload.get("person_a", -1))
            b = int(payload.get("person_b", -1))
            left, right = by_person.get(a, []), by_person.get(b, [])
            if not left or not right or a == b:
                return jsonify({"error": "pick two different piles"}), 400
            flagged = _flagged_faces(store_fb)
            merged = [f for f in left + right if f not in flagged]
            if len(merged) < 2:
                return jsonify({"error": "not enough faces"}), 400
            judgement.same_group = merged
            judgement.subject = f"merge:{min(a, b)}-{max(a, b)}"
            judgement.person_label = f"Person {a + 1} + Person {b + 1}"

        elif kind == fb.FACE_BELONGS:
            face_id = payload.get("face_id", "")
            person_id = int(payload.get("person_id", -1))
            members = by_person.get(person_id, [])
            if face_id not in face_ids or not members:
                return jsonify({"error": "unknown face or pile"}), 400
            judgement.same_group = [face_id] + members
            judgement.subject = f"assign:{face_id}"
            judgement.person_label = f"Person {person_id + 1}"

        elif kind == fb.PAIR:
            a = payload.get("face_a", "")
            b = payload.get("face_b", "")
            if a not in face_ids or b not in face_ids or a == b:
                return jsonify({"error": "need two known, different faces"}), 400
            if payload.get("same"):
                judgement.same_group = [a, b]
            else:
                judgement.different_pairs = [[a, b]]
            judgement.subject = f"pair:{min(a, b)}-{max(a, b)}"

        elif kind == fb.NOT_A_FACE:
            face_id = payload.get("face_id", "")
            if face_id not in face_ids:
                return jsonify({"error": "unknown face"}), 400
            judgement.excluded = [face_id]
            judgement.subject = face_id

        elif kind == fb.NOTE:
            if not judgement.text:
                return jsonify({"error": "empty note"}), 400
            judgement.subject = f"note:{payload.get('person_id', 'event')}"
            if payload.get("person_id") is not None:
                judgement.person_label = f"Person {int(payload['person_id']) + 1}"

        saved = store_fb.append(judgement)
        return jsonify(
            {
                "ok": True,
                "record_id": saved.record_id,
                "kind": saved.kind,
                "reviewer": saved.reviewer,
                "score": _score_payload(event_id, settings, store_fb),
            }
        )

    @app.post("/api/events/<event_id>/feedback/<record_id>/retract")
    def retract_feedback(event_id: str, record_id: str):
        require_processed(event_id)
        store_fb = feedback_store(event_id)
        known = {r.record_id for r in store_fb.live()}
        if record_id not in known:
            return jsonify({"error": "nothing to undo"}), 404
        store_fb.retract(record_id, reviewer_name())
        settings = cluster_config_from_request(cfg.cluster)
        return jsonify({"ok": True, "score": _score_payload(event_id, settings, store_fb)})

    @app.get("/api/events/<event_id>/score")
    def score_only(event_id: str):
        require_processed(event_id)
        settings = cluster_config_from_request(cfg.cluster)
        return jsonify(_score_payload(event_id, settings, feedback_store(event_id)))

    @app.get("/e/<event_id>/feedback.jsonl")
    def export_feedback(event_id: str):
        event_paths(event_id)
        text = feedback_store(event_id).raw_text()
        return Response(
            text,
            mimetype="application/x-ndjson",
            headers={
                "Content-Disposition": f'attachment; filename="{event_id}-feedback.jsonl"'
            },
        )

    # -- images ------------------------------------------------------------
    @app.get("/i/<event_id>/<kind>/<name>")
    def image(event_id: str, kind: str, name: str):
        paths = event_paths(event_id)
        if kind == "crop":
            if not FACE_ID.match(name):
                abort(404)
            photo_id, face_index = name.rsplit("_", 1)
            uri = paths.crop(photo_id, int(face_index))
        elif kind in ("prev", "thumb"):
            if not PHOTO_ID.match(name):
                abort(404)
            uri = paths.preview(name) if kind == "prev" else paths.thumb(name)
        else:
            abort(404)
        if not obj_store.exists(uri):
            abort(404)
        return Response(
            obj_store.get_bytes(uri),
            mimetype="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # -- internals used by the routes above -------------------------------
    def _score_payload(event_id, settings, store_fb) -> dict:
        output = cache.stage2(event_id, settings)
        judgements = store_fb.live()
        truth = fb.derive_truth(judgements)
        score = fb.score_clustering(
            truth, output.faces.face_ids, list(output.result.labels)
        )
        evidence = fb.threshold_evidence(
            truth, output.faces.face_ids, cache.distance_fn(event_id)
        )
        return {
            # What the strip at the top of the review page shows. Sent back
            # with every answer so the page never has to reload to stay honest
            # about how much is left.
            "progress": _review_progress(output, judgements).as_dict(),
            "judged_faces": score.judged_faces,
            "judged_groups": score.judged_groups,
            "same_pairs_known": score.same_pairs_known,
            "same_pairs_grouped": score.same_pairs_grouped,
            "same_pairs_split": score.same_pairs_split,
            "diff_pairs_known": score.diff_pairs_known,
            "diff_pairs_merged": score.diff_pairs_merged,
            "faces_in_wrong_pile": len(score.faces_in_wrong_pile),
            "faces_split_off": len(score.faces_split_off),
            "recall": round(score.recall, 3),
            "merge_safety": round(score.merge_safety, 3),
            "headline": score.headline(),
            "contradictions": len(score.contradictions),
            "judgements": len([j for j in judgements if j.kind != fb.NOTE]),
            "threshold_summary": evidence.summary(),
            "suggested_threshold": evidence.suggested,
        }

    def _flagged_faces(store_fb: fb.FeedbackStore) -> set[str]:
        return {
            record.subject
            for record in store_fb.live()
            if record.kind in (fb.FACE_WRONG, fb.NOT_A_FACE)
        }

    app.jinja_env.globals["cluster_query"] = _cluster_query
    return app


# ---------------------------------------------------------------------------
# Turning a Stage2Output into what the template renders
# ---------------------------------------------------------------------------


def _review_progress(output, judgements) -> ReviewProgress:
    """Which piles have been answered, from the judgement journal alone.

    A pile counts as answered when somebody said "all one person" OR marked a
    face in it as somebody else -- both are answers to the one question the
    review window asks. Marking a crop as "not a face at all" is not: it is a
    complaint about the detector, and treating it as an answer would let a
    folder read 100% checked without a single person having been looked at.

    Keyed to the pile's contents (group_key), not to person_id, for the same
    reason approvals are: person ids are reshuffled by every re-group.
    """
    approved = {
        record.subject for record in judgements if record.kind == fb.PERSON_OK
    }
    wrong = {
        record.subject for record in judgements if record.kind == fb.FACE_WRONG
    }

    progress = ReviewProgress(piles_total=len(output.result.persons))
    for person_id, members in _faces_by_person(output).items():
        fixed = [face_id for face_id in members if face_id in wrong]
        progress.faces_fixed += len(fixed)
        if fixed or group_key(members) in approved:
            progress.piles_answered += 1
            progress.answered_ids.append(person_id)
    return progress


def _faces_by_person(output) -> dict[int, list[str]]:
    faces = output.faces
    return {
        person.person_id: [faces.face_ids[i] for i in person.face_indices]
        for person in output.result.persons
    }


def _build_review(output, judgements, truth) -> tuple[list[ReviewPerson], list[ReviewFace]]:
    from collections import Counter

    import numpy as np

    faces = output.faces
    result = output.result
    ground_truth = output.ground_truth

    flagged: dict[str, str] = {}
    not_faces: dict[str, str] = {}
    approvals: dict[str, tuple[str, str]] = {}
    for record in judgements:
        if record.kind == fb.FACE_WRONG:
            flagged[record.subject] = record.record_id
        elif record.kind == fb.NOT_A_FACE:
            not_faces[record.subject] = record.record_id
        elif record.kind == fb.PERSON_OK:
            approvals[record.subject] = (record.reviewer, record.record_id)

    contaminated = {
        c.person_id: c
        for c in (output.evaluation.merged_people if output.evaluation else [])
    }

    def make_face(index: int, majority: str | None = None) -> ReviewFace:
        face_id = faces.face_ids[index]
        return ReviewFace(
            face_id=face_id,
            photo_id=faces.photo_ids[index],
            face_index=faces.face_indices[index],
            photo_path=faces.photo_paths[index],
            height_px=int(faces.face_h_px[index]),
            quality=float(faces.quality[index]),
            second_class=bool(faces.second_class[index]),
            flagged=face_id in flagged,
            not_a_face=face_id in not_faces,
            flag_record=flagged.get(face_id, "") or not_faces.get(face_id, ""),
            truth_label=truth.label_of(face_id),
            is_alien=bool(
                ground_truth and majority and ground_truth[index] != majority
            ),
        )

    people: list[ReviewPerson] = []
    for person in result.persons:
        majority = None
        if ground_truth:
            majority = Counter(
                ground_truth[i] for i in person.face_indices
            ).most_common(1)[0][0]

        # Best crops first: a reviewer should judge a pile on its clearest
        # faces, then scroll to the marginal ones.
        ordered = sorted(person.face_indices, key=lambda i: -faces.quality[i])
        member_ids = [faces.face_ids[i] for i in person.face_indices]
        key = group_key(member_ids)
        reviewer, record_id = approvals.get(key, ("", ""))

        distances = []
        if len(person.face_indices) > 1:
            vectors = faces.embeddings[person.face_indices]
            centroid = person.centroid
            norms = np.linalg.norm(vectors, axis=1) * float(np.linalg.norm(centroid))
            with np.errstate(divide="ignore", invalid="ignore"):
                sims = np.where(norms > 0, vectors @ centroid / np.maximum(norms, 1e-9), 1.0)
            distances = (1.0 - sims).tolist()

        note = ""
        if person.person_id in contaminated:
            match = contaminated[person.person_id]
            note = ", ".join(
                f"{name}x{count}"
                for name, count in sorted(match.true_labels.items(), key=lambda kv: -kv[1])
            )

        heights = [int(faces.face_h_px[i]) for i in person.face_indices]
        people.append(
            ReviewPerson(
                person_id=person.person_id,
                label=person.label or f"Person {person.person_id + 1}",
                face_count=person.face_count,
                photo_count=person.photo_count,
                faces=[make_face(i, majority) for i in ordered],
                group_key=key,
                approved_by=reviewer,
                approved_record=record_id,
                contaminated=person.person_id in contaminated,
                contamination_note=note,
                spread_px=(
                    f"{min(heights)}-{max(heights)}px"
                    + (
                        # The furthest face from the pile's average. Named for
                        # what it is: "tightness 0.61" reads as if bigger were
                        # better, and it is the opposite.
                        f" · furthest face {max(distances):.2f}"
                        if distances
                        else ""
                    )
                ),
            )
        )

    people.sort(key=lambda p: (not p.contaminated, not p.approved_by == "", -p.face_count))
    leftover_indices = [i for i, label in enumerate(result.labels) if label < 0]
    leftovers = [
        make_face(i)
        for i in sorted(leftover_indices, key=lambda i: -faces.quality[i])
    ]
    return people, leftovers


# The six things that actually happen to a folder, named the way a person would
# name them. The pipeline's own phase words ("consolidating", "manifest") are
# accurate and meaningless to anyone who did not write them.
STEP_PHASES = [CHECKING, MANIFEST, LOADING, PROCESSING, CONSOLIDATING, CLUSTERING]
STEP_TITLES = [
    "Check the folder",
    "Count the photos",
    "Wake up the recogniser",
    "Look at each photo",
    "File the results",
    "Group them into people",
]


def _progress_steps(job, totals) -> list[dict]:
    """Every step, with the one in progress marked and the rest still listed.

    Showing all six even before they start is the point: a run that sits on
    "looking at each photo" for three hours is only alarming if you cannot see
    that it is step 4 of 6 and what the remaining two are.
    """
    phase = getattr(job, "phase", "") if job else ""
    current = STEP_PHASES.index(phase) if phase in STEP_PHASES else -1
    if phase in ("done", "stopped"):
        current = len(STEP_PHASES)
    if phase == "failed":
        # Whatever it was doing when it died is the step that failed.
        current = max(0, current)

    total = getattr(job, "photos_total", 0) if job else 0
    done = getattr(job, "photos_done", 0) if job else 0
    eta = getattr(job, "eta_seconds", 0) if job else 0
    source = getattr(job, "source_type", "") if job else ""
    people = getattr(job, "people_found", 0) if job else 0

    details = [
        f"{source or 'the folder'} — is it there, can we read it",
        (f"{total:,} photos found" if total else "walks every sub-folder; videos and RAW files are skipped"),
        "loads the face models into memory. About 10 seconds, once per run.",
        (
            f"{done:,} of {total:,}"
            + (f" · about {_human_left(eta)} left" if eta else "")
            if total
            else "full size, whole frame plus overlapping tiles so a small face in the back row isn't missed"
        ),
        "writes one cache file, so none of the slow work ever has to happen again",
        (
            f"{people} people found"
            if people
            else "seconds — and you can redo this as often as you like afterwards"
        ),
    ]

    steps = []
    for index, title in enumerate(STEP_TITLES):
        if phase == "failed" and index == current:
            state = "failed"
        elif index < current:
            state = "done"
        elif index == current:
            state = "now"
        else:
            state = "todo"
        step = {"title": title, "detail": details[index], "state": state}
        if index == 3 and total:
            step["percent"] = min(100.0, round(done / total * 100, 1))
            if totals.detected:
                step["detail"] += f" · {totals.detected:,} faces found so far"
        steps.append(step)
    return steps


def _human_left(seconds: float) -> str:
    """Rounded the same way the browser rounds it.

    The step list is rendered here and the headline counter is rendered in
    JavaScript from the same number; flooring in one and rounding in the other
    put "about 2 min left" directly above "about 3 min left" on the same screen.
    """
    seconds = int(seconds or 0)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{round(seconds / 60)} min"
    return f"{seconds // 3600}h {round((seconds % 3600) / 60)}m"


def _size_bands(heights) -> list[tuple[str, int, float]]:
    """The distribution that decides whether any of this works.

    28.9% of faces on the first real folder came in under 112px, the size
    ArcFace actually needs. Printing it on every run keeps that number in view
    instead of buried in a worklog.
    """
    total = len(heights)
    if not total:
        return []
    bands = [
        ("over 400px", lambda h: h >= 400),
        ("200-400px", lambda h: 200 <= h < 400),
        ("112-200px", lambda h: 112 <= h < 200),
        ("80-112px", lambda h: 80 <= h < 112),
        ("50-80px", lambda h: 50 <= h < 80),
        ("under 50px", lambda h: h < 50),
    ]
    out = []
    for name, test in bands:
        count = int(sum(1 for h in heights if test(h)))
        out.append((name, count, round(count / total * 100, 1)))
    return out


def _candidate_pairs(output, distance_of, truth, settings, limit: int = 40):
    """The pairs worth a human's two seconds.

    Not random pairs -- pairs sitting either side of the merge threshold, where
    the machine is closest to a coin flip. Judging those moves the threshold;
    judging obvious ones moves nothing.
    """
    import numpy as np

    faces = output.faces
    persons = output.result.persons
    if len(persons) < 2:
        return []

    already = {frozenset(pair) for pair in truth.must_not_link}
    for face_id, label in truth.labels.items():
        already.add(frozenset({face_id, label}))  # cheap membership guard

    centroids = np.stack([p.centroid for p in persons])
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    unit = centroids / np.maximum(norms, 1e-9)
    similarity = unit @ unit.T
    distance = 1.0 - similarity

    window_low = max(0.05, settings.pass2_threshold - 0.20)
    window_high = min(0.95, settings.pass2_threshold + 0.25)

    candidates = []
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            gap = float(distance[i, j])
            if not (window_low <= gap <= window_high):
                continue
            # The clearest face from each pile: a hard judgement should not be
            # made harder by showing two blurry crops.
            a = max(persons[i].face_indices, key=lambda k: faces.quality[k])
            b = max(persons[j].face_indices, key=lambda k: faces.quality[k])
            face_a, face_b = faces.face_ids[a], faces.face_ids[b]
            if frozenset({face_a, face_b}) in already:
                continue
            candidates.append(
                {
                    "distance": round(gap, 3),
                    "verdict": "same" if gap <= settings.pass2_threshold else "different",
                    "a": {
                        "face_id": face_a,
                        "photo_id": faces.photo_ids[a],
                        "person": persons[i].label or f"Person {persons[i].person_id + 1}",
                        "height_px": int(faces.face_h_px[a]),
                        "faces_in_pile": persons[i].face_count,
                    },
                    "b": {
                        "face_id": face_b,
                        "photo_id": faces.photo_ids[b],
                        "person": persons[j].label or f"Person {persons[j].person_id + 1}",
                        "height_px": int(faces.face_h_px[b]),
                        "faces_in_pile": persons[j].face_count,
                    },
                }
            )

    # Closest first: those are the ones a small threshold change would flip.
    candidates.sort(key=lambda c: c["distance"])
    return candidates[:limit]


def _cluster_query(settings, **overrides) -> str:
    values = {
        "p1": round(settings.pass1_threshold, 4),
        "p2": round(settings.pass2_threshold, 4),
        "minf": settings.min_faces_per_person,
        "sc": round(settings.second_class_assign_threshold, 4),
        "sp": 1 if settings.enforce_same_photo_rule else 0,
    }
    values.update(overrides)
    return "&".join(f"{k}={v}" for k, v in values.items())


def _suggest_event_id(link: str, store: ObjectStore, config: Config) -> str:
    """A readable name from the folder, uniquified. Nobody wants to invent ids."""
    import re as _re
    from datetime import date

    tail = _re.split(r"[\\/]", link.rstrip("\\/"))[-1] or "event"
    slug = _re.sub(r"[^A-Za-z0-9]+", "-", tail).strip("-").lower()[:28] or "event"
    base = f"ev-{slug}"
    existing = {summary.event_id for summary in list_events(store, config)}
    if base not in existing:
        return base
    stamp = date.today().strftime("%m%d")
    candidate = f"{base}-{stamp}"
    counter = 2
    while candidate in existing:
        candidate = f"{base}-{stamp}-{counter}"
        counter += 1
    return candidate
