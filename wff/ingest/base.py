"""One interface, many sources.

The rest of the pipeline never knows or cares where a photo came from. Adding
Google Drive, Dropbox or a 40 GB zip later is one new file implementing this
class -- nothing downstream changes.

docs/ARCHITECTURE.md, Step 1 -- Ingest.
"""
from __future__ import annotations

import posixpath
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import BinaryIO, Iterator

# What we process. .heic is here because iPhones are common enough to matter.
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif",
}
# Skipped in v1: 40 MB each, rarely delivered to clients.
RAW_EXTENSIONS = {
    ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".srw", ".pef",
}
# Skipped, but counted, so the totals we show the photographer are not confusing.
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".m2ts", ".wmv", ".webm", ".3gp",
}


class FileKind:
    IMAGE = "image"
    RAW = "raw"
    VIDEO = "video"
    OTHER = "other"


def classify(name: str) -> str:
    ext = posixpath.splitext(name.lower())[1]
    if ext in IMAGE_EXTENSIONS:
        return FileKind.IMAGE
    if ext in RAW_EXTENSIONS:
        return FileKind.RAW
    if ext in VIDEO_EXTENSIONS:
        return FileKind.VIDEO
    return FileKind.OTHER


@dataclass(frozen=True)
class FileRef:
    """One file at the source. Never a local path -- an opaque source handle."""

    source_id: str          # stable id AT THE SOURCE: Drive file id, zip entry, rel path
    name: str               # 'IMG_4821.jpg'
    path: str               # 'Sangeet/IMG_4821.jpg' -- relative, folders preserved
    kind: str = FileKind.IMAGE
    size: int | None = None
    mime: str | None = None

    @property
    def sub_album(self) -> str:
        """'Sangeet' from 'Sangeet/IMG_4821.jpg'.

        A free feature that falls out of walking subfolders: "Priya's photos ->
        Sangeet (12)". Costs one line here and nothing later.
        """
        parent = posixpath.dirname(self.path)
        return posixpath.basename(parent) if parent else ""


@dataclass
class ValidationResult:
    ok: bool
    message: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class SourceStats:
    """What Step 2 shows the photographer before any real work starts."""

    images: int = 0
    raw: int = 0
    videos: int = 0
    other: int = 0
    unreadable: int = 0
    total_bytes: int = 0
    sub_albums: dict[str, int] = field(default_factory=dict)


class IngestAdapter(ABC):
    """Walk a source, stream one file at a time. Never stage to disk."""

    source_type: str = "unknown"

    def __init__(self, link: str) -> None:
        self.link = link

    @abstractmethod
    def validate(self) -> ValidationResult:
        """Is this link readable right now? Cheap -- called before queueing."""
        ...

    @abstractmethod
    def list_files(self) -> Iterator[FileRef]:
        """Walk everything, recursing into subfolders, keeping folder names."""
        ...

    @abstractmethod
    @contextmanager
    def open(self, ref: FileRef) -> Iterator[BinaryIO]:
        """Stream one file. The caller reads it into memory and discards it.

        Zero staging storage: download one photo, process it, drop it, move on.
        """
        ...

    def source_folder_id(self) -> str:
        """Saved on the event so we can proxy original downloads later.

        This is what makes download policy B (true originals at zero storage
        cost) possible. It only exists because we ingest from a link.
        """
        return self.link

    def scan(self) -> tuple[list[FileRef], SourceStats]:
        """Build the manifest. Images returned; everything else counted."""
        stats = SourceStats()
        images: list[FileRef] = []
        for ref in self.list_files():
            if ref.size:
                stats.total_bytes += ref.size
            if ref.kind == FileKind.IMAGE:
                images.append(ref)
                stats.images += 1
                album = ref.sub_album or "(root)"
                stats.sub_albums[album] = stats.sub_albums.get(album, 0) + 1
            elif ref.kind == FileKind.RAW:
                stats.raw += 1
            elif ref.kind == FileKind.VIDEO:
                stats.videos += 1
            else:
                stats.other += 1
        return images, stats
