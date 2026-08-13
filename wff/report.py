"""The accuracy meter.

An HTML page: "Person 1 -- 84 photos" with a grid of face crops under each. We
scroll it and count the mistakes by eye. This page is the whole feedback loop
for the next two weeks, so it is built to make errors OBVIOUS rather than to
look nice:

  - crops are the aligned 112x112 the model actually saw, not a pretty crop
  - a cluster containing two real people is flagged red, at the top
  - a face's height in the original photo is printed on it, because that is
    the number that decides whether any of this works
"""
from __future__ import annotations

import html
from collections import Counter

from .config import Config, load_config
from .stage1 import EventPaths
from .stage2 import Stage2Output
from .storage import ObjectStore, get_store, join_uri

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin:0; padding:24px; font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       background:Canvas; color:CanvasText; }
h1 { font-size:22px; margin:0 0 4px; }
h2 { font-size:16px; margin:32px 0 12px; padding-top:16px; border-top:1px solid color-mix(in srgb, CanvasText 15%, transparent); }
.sub { opacity:.65; margin:0 0 20px; }
.cards { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:18px; }
.card { border:1px solid color-mix(in srgb, CanvasText 18%, transparent); border-radius:8px;
        padding:10px 14px; min-width:120px; }
.card .n { font-size:22px; font-weight:600; display:block; }
.card .k { opacity:.65; font-size:12px; }
.alarm { border-left:4px solid #d33; background:color-mix(in srgb,#d33 10%, transparent);
         padding:10px 14px; border-radius:6px; margin:8px 0; }
.note  { border-left:4px solid #888; background:color-mix(in srgb, CanvasText 6%, transparent);
         padding:10px 14px; border-radius:6px; margin:8px 0; }
.good  { border-left:4px solid #2a2; background:color-mix(in srgb,#2a2 10%, transparent);
         padding:10px 14px; border-radius:6px; margin:8px 0; }
.person { border:1px solid color-mix(in srgb, CanvasText 15%, transparent); border-radius:10px;
          padding:12px; margin-bottom:14px; }
.person.bad { border-color:#d33; border-width:2px; }
.person h3 { margin:0 0 2px; font-size:15px; }
.person .meta { opacity:.65; font-size:12px; margin-bottom:10px; }
.grid { display:flex; flex-wrap:wrap; gap:4px; }
.face { position:relative; width:76px; }
.face img { width:76px; height:76px; object-fit:cover; border-radius:5px; display:block;
            background:color-mix(in srgb, CanvasText 10%, transparent); }
.face .px { position:absolute; left:2px; bottom:2px; font-size:9px; padding:0 3px; border-radius:3px;
            background:rgba(0,0,0,.72); color:#fff; font-variant-numeric:tabular-nums; }
.face.small img { outline:2px solid #e9a13b; outline-offset:-2px; }
.face.alien img { outline:2px solid #d33; outline-offset:-2px; }
table { border-collapse:collapse; font-size:13px; margin:8px 0 16px; }
td,th { border:1px solid color-mix(in srgb, CanvasText 15%, transparent); padding:5px 10px; text-align:left; }
th { background:color-mix(in srgb, CanvasText 7%, transparent); }
.legend { font-size:12px; opacity:.7; margin:6px 0 14px; }
code { background:color-mix(in srgb, CanvasText 10%, transparent); padding:1px 5px; border-radius:4px; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def build_report(
    output: Stage2Output,
    store: ObjectStore | None = None,
    config: Config | None = None,
    max_faces_per_person: int = 60,
) -> str:
    """Write report.html into the event's storage; return its object URI."""
    cfg = config or load_config()
    obj_store = store or get_store(cfg)
    paths = EventPaths(cfg.storage.bucket, output.event_id)
    report_uri = join_uri(paths.root, "report.html")

    faces = output.faces
    result = output.result
    truth = output.ground_truth

    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{_esc(output.event_id)} - clustering report</title>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{_esc(output.event_id)}</h1>",
        f"<p class='sub'>{output.photo_count:,} photos &middot; "
        f"{len(faces):,} accepted faces &middot; "
        f"{len(result.persons)} people found &middot; "
        f"model <code>{_esc(cfg.embedding_model_version)}</code></p>",
    ]

    # -- headline numbers -------------------------------------------------
    cards = [
        ("photos", f"{output.photo_count:,}"),
        ("faces kept", f"{len(faces):,}"),
        ("faces rejected", f"{faces.rejected_count:,}"),
        ("people", f"{len(result.persons)}"),
        ("pass-1 groups", f"{result.pass1_group_count}"),
        ("pass-2 merges", f"{result.pass2_merge_count}"),
        ("leftover faces", f"{result.leftover_face_count:,}"),
    ]
    parts.append("<div class='cards'>")
    for key, value in cards:
        parts.append(f"<div class='card'><span class='n'>{value}</span><span class='k'>{key}</span></div>")
    parts.append("</div>")

    # -- the scored verdict, when we know the answer ----------------------
    if output.evaluation:
        ev = output.evaluation
        css = "good" if not ev.merged_people else "alarm"
        parts.append(f"<div class='{css}'><strong>Scored against ground truth</strong><br>")
        parts.append("<br>".join(_esc(line) for line in ev.summary_lines()))
        parts.append("</div>")
        if ev.merged_people:
            parts.append(
                "<div class='note'>A contaminated cluster is the catastrophic "
                "failure: a guest matching into it is shown a stranger's photos. "
                "Fix by LOWERING <code>WFF_PASS2_THRESHOLD</code>.</div>"
            )

    for alarm in output.sanity.alarms:
        parts.append(f"<div class='alarm'>{_esc(alarm)}</div>")
    for note in output.sanity.notes:
        parts.append(f"<div class='note'>{_esc(note)}</div>")

    # -- what the gate threw away -----------------------------------------
    if faces.reject_reasons:
        parts.append("<h2>What the quality gate rejected</h2>")
        parts.append("<table><tr><th>reason</th><th>faces</th></tr>")
        for reason, count in sorted(faces.reject_reasons.items(), key=lambda kv: -kv[1]):
            parts.append(f"<tr><td>{_esc(reason)}</td><td>{count:,}</td></tr>")
        parts.append("</table>")

    # -- the people -------------------------------------------------------
    parts.append("<h2>People</h2>")
    parts.append(
        "<p class='legend'>Number on each crop = the face's height in the "
        "ORIGINAL photo, in pixels. "
        "<span style='outline:2px solid #e9a13b;padding:0 4px'>orange</span> = "
        "second-class (50-80px: may join a person, may never start one). "
        + (
            "<span style='outline:2px solid #d33;padding:0 4px'>red</span> = "
            "a face whose true label differs from this cluster's majority -- "
            "these are the errors that matter."
            if truth
            else ""
        )
        + "</p>"
    )

    contaminated_ids = {c.person_id for c in (output.evaluation.merged_people if output.evaluation else [])}
    ordered = sorted(
        result.persons,
        key=lambda p: (p.person_id not in contaminated_ids, -p.face_count),
    )

    for person in ordered:
        is_bad = person.person_id in contaminated_ids
        majority = None
        if truth:
            composition = Counter(truth[i] for i in person.face_indices)
            majority = composition.most_common(1)[0][0]

        parts.append(f"<div class='person{' bad' if is_bad else ''}'>")
        title = _esc(person.label)
        if truth and majority:
            title += f" &mdash; {_esc(majority)}"
        parts.append(f"<h3>{title}</h3>")

        meta = f"{person.face_count} faces in {person.photo_count} photos"
        if is_bad and output.evaluation:
            match = next(c for c in output.evaluation.merged_people if c.person_id == person.person_id)
            breakdown = ", ".join(f"{_esc(k)}&times;{v}" for k, v in sorted(match.true_labels.items(), key=lambda kv: -kv[1]))
            meta += f" &middot; <strong>CONTAMINATED: {breakdown}</strong>"
        parts.append(f"<div class='meta'>{meta}</div>")

        parts.append("<div class='grid'>")
        shown = sorted(person.face_indices, key=lambda i: -faces.quality[i])[:max_faces_per_person]
        for face_index in shown:
            crop_uri = paths.crop(faces.photo_ids[face_index], faces.face_indices[face_index])
            src = obj_store.public_url(crop_uri, relative_to=report_uri)
            classes = ["face"]
            if faces.second_class[face_index]:
                classes.append("small")
            if truth and majority and truth[face_index] != majority:
                classes.append("alien")
            tooltip = _esc(faces.photo_paths[face_index])
            parts.append(
                f"<div class='{' '.join(classes)}' title='{tooltip}'>"
                f"<img loading='lazy' src='{_esc(src)}' alt=''>"
                f"<span class='px'>{faces.face_h_px[face_index]:.0f}</span></div>"
            )
        parts.append("</div>")
        if len(person.face_indices) > max_faces_per_person:
            parts.append(
                f"<div class='meta'>showing best {max_faces_per_person} of "
                f"{person.face_count}</div>"
            )
        parts.append("</div>")

    # -- leftovers, which we deliberately keep ----------------------------
    leftovers = [i for i, label in enumerate(result.labels) if label < 0]
    if leftovers:
        parts.append(f"<h2>Leftover faces ({len(leftovers):,})</h2>")
        parts.append(
            "<p class='legend'>No person assigned (<code>person_id NULL</code>). "
            "Deliberately kept: a cousin in 3 photos is exactly the guest most "
            "delighted to find them, and tier-2 selfie search reaches these "
            "individually. Deleting them would silently lose that guest.</p>"
        )
        parts.append("<div class='grid'>")
        for face_index in sorted(leftovers, key=lambda i: -faces.quality[i])[:200]:
            crop_uri = paths.crop(faces.photo_ids[face_index], faces.face_indices[face_index])
            src = obj_store.public_url(crop_uri, relative_to=report_uri)
            classes = "face small" if faces.second_class[face_index] else "face"
            parts.append(
                f"<div class='{classes}' title='{_esc(faces.photo_paths[face_index])}'>"
                f"<img loading='lazy' src='{_esc(src)}' alt=''>"
                f"<span class='px'>{faces.face_h_px[face_index]:.0f}</span></div>"
            )
        parts.append("</div>")

    # -- the knobs, printed so a run is reproducible ----------------------
    parts.append("<h2>Settings this run used</h2><table>")
    parts.append("<tr><th>knob</th><th>value</th><th>env var</th></tr>")
    for name, value, env in [
        ("pass 1 threshold (tight)", cfg.cluster.pass1_threshold, "WFF_PASS1_THRESHOLD"),
        ("pass 2 threshold (merge)", cfg.cluster.pass2_threshold, "WFF_PASS2_THRESHOLD"),
        ("same-photo rule", cfg.cluster.enforce_same_photo_rule, "WFF_SAME_PHOTO_RULE"),
        ("min faces per person", cfg.cluster.min_faces_per_person, "WFF_MIN_FACES_PER_PERSON"),
        ("min face px", cfg.quality.min_face_px, "WFF_MIN_FACE_PX"),
        ("second-class below px", cfg.quality.second_class_face_px, "WFF_SECOND_CLASS_FACE_PX"),
        ("detector input size", cfg.detect.det_size, "WFF_DET_SIZE"),
        ("tiling", cfg.detect.tiling_enabled, "WFF_TILING"),
        ("tile target scale", cfg.detect.tile_target_scale, "WFF_TILE_TARGET_SCALE"),
    ]:
        parts.append(f"<tr><td>{_esc(name)}</td><td><code>{_esc(value)}</code></td><td><code>{env}</code></td></tr>")
    parts.append("</table>")

    parts.append("</body></html>")

    document = "".join(parts)
    obj_store.put_text(report_uri, document, "text/html")
    return report_uri
