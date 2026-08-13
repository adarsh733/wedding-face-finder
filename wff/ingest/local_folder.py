"""Read a folder on this machine. Development only.

This is the one adapter allowed to touch a filesystem path, because its whole
job is to pretend a folder is a remote source. Everything downstream still only
ever sees FileRef handles.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from .base import FileRef, IngestAdapter, ValidationResult, classify


class LocalFolderAdapter(IngestAdapter):
    source_type = "local"

    def __init__(self, link: str) -> None:
        # Accept both 'file:///D:/photos' and a bare path, because a human types
        # this on the command line.
        if link.startswith("file://"):
            from urllib.parse import unquote, urlparse

            parsed = urlparse(link)
            raw = unquote(parsed.path)
            if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
                raw = raw[1:]  # /D:/photos -> D:/photos
            link = raw
        super().__init__(link)
        self.root = Path(link).expanduser()

    def validate(self) -> ValidationResult:
        if not self.root.exists():
            return ValidationResult(False, f"Folder does not exist: {self.root}")
        if not self.root.is_dir():
            return ValidationResult(False, f"Not a folder: {self.root}")
        try:
            next(self.root.iterdir(), None)
        except PermissionError:
            return ValidationResult(False, f"Permission denied reading {self.root}")
        return ValidationResult(True, f"Readable folder: {self.root}")

    def list_files(self) -> Iterator[FileRef]:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            yield FileRef(
                source_id=rel,
                name=path.name,
                path=rel,
                kind=classify(path.name),
                size=size,
            )

    @contextmanager
    def open(self, ref: FileRef) -> Iterator[BinaryIO]:
        with open(self.root / ref.source_id, "rb") as fh:
            yield fh
