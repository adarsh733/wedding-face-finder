"""The two things about a folder that a person, not the pipeline, decides.

WHY THIS FILE EXISTS. `ev_manali01` is an id: it has to stay stable, because
every crop, preview and journal line on disk is filed under it. But nobody
wants to read it. Adarsh asked to rename a run, and separately to fix the six
folders that were processed before the console started recording who was
driving -- both are labels a human sets afterwards, neither is something the
pipeline can work out, and neither may change a single byte of what has already
been processed.

So they live in their own small document beside the event, never in `event.json`
(written by Stage 1, and a rename must not risk that file) and never in the id
itself (renaming would orphan ~10,000 crops).

Last write wins. Two chats renaming the same folder in the same second is not a
problem worth a journal -- unlike judgements, a label carries no evidence.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from ..storage import ObjectStore, join_uri

MAX_NAME = 60


@dataclass
class EventLabel:
    """What a human called this folder, and who they say ran it."""

    display_name: str = ""
    reviewer: str = ""
    updated_at: str = ""
    updated_by: str = ""

    @property
    def empty(self) -> bool:
        return not self.display_name and not self.reviewer


def label_uri(bucket: str, event_id: str) -> str:
    return join_uri(f"r2://{bucket}/events/{event_id}", "label.json")


def read_label(store: ObjectStore, bucket: str, event_id: str) -> EventLabel:
    """The label, or an empty one. Never raises: a page must not 500 because
    somebody hand-edited a JSON file."""
    uri = label_uri(bucket, event_id)
    try:
        if not store.exists(uri):
            return EventLabel()
        raw = json.loads(store.get_text(uri))
    except (OSError, ValueError, json.JSONDecodeError):
        return EventLabel()
    if not isinstance(raw, dict):
        return EventLabel()
    return EventLabel(
        display_name=clean_name(raw.get("display_name")),
        reviewer=clean_name(raw.get("reviewer"), limit=40),
        updated_at=str(raw.get("updated_at") or ""),
        updated_by=str(raw.get("updated_by") or "")[:40],
    )


def write_label(
    store: ObjectStore,
    bucket: str,
    event_id: str,
    *,
    display_name: str | None = None,
    reviewer: str | None = None,
    updated_by: str = "",
) -> EventLabel:
    """Set either field, leaving the other alone.

    `None` means "don't touch"; an empty string means "clear it", which is how
    a rename is undone -- clearing the display name puts the id back on screen
    rather than leaving a blank card.
    """
    current = read_label(store, bucket, event_id)
    if display_name is not None:
        current.display_name = clean_name(display_name)
    if reviewer is not None:
        current.reviewer = clean_name(reviewer, limit=40)
    current.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    current.updated_by = clean_name(updated_by, limit=40)
    store.put_text(label_uri(bucket, event_id), json.dumps(asdict(current), indent=2))
    return current


def clean_name(value: object, limit: int = MAX_NAME) -> str:
    """One line, trimmed, length-capped.

    Newlines and control characters are stripped rather than rejected: this is
    a name typed into a box, and a paste carrying a stray newline should not be
    an error message.
    """
    if value is None:
        return ""
    text = str(value)
    text = "".join(" " if ch in "\r\n\t" else ch for ch in text if ch.isprintable() or ch in "\r\n\t")
    return " ".join(text.split())[:limit].strip()
