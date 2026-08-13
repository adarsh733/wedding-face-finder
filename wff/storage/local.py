"""Local filesystem backend. Dev only.

Materialises r2://bucket/key as <root>/bucket/key. The pipeline never learns
that a filesystem is involved -- it only ever holds URIs.
"""
from __future__ import annotations

import os
import posixpath
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from .base import ObjectStore, parse_uri


class LocalObjectStore(ObjectStore):
    def __init__(self, root: str) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    # -- internal ----------------------------------------------------------
    def _path(self, uri: str) -> Path:
        parsed = parse_uri(uri)
        # posixpath keys can contain '/', never '..' -- reject traversal.
        key = posixpath.normpath(parsed.key)
        if key.startswith("..") or os.path.isabs(key):
            raise ValueError(f"Refusing suspicious object key: {parsed.key!r}")
        return self._root / parsed.bucket / Path(key)

    # -- interface ---------------------------------------------------------
    def put_bytes(self, uri: str, data: bytes, content_type: str | None = None) -> None:
        path = self._path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a crash mid-write leaves no half-object behind.
        tmp = path.with_suffix(path.suffix + ".partial")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def get_bytes(self, uri: str) -> bytes:
        return self._path(uri).read_bytes()

    def exists(self, uri: str) -> bool:
        return self._path(uri).exists()

    def delete(self, uri: str) -> None:
        path = self._path(uri)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def list_keys(self, prefix_uri: str) -> Iterator[str]:
        parsed = parse_uri(prefix_uri)
        base = self._path(prefix_uri)
        if not base.exists():
            return
        root_for_bucket = self._root / parsed.bucket
        if base.is_file():
            yield prefix_uri
            return
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.endswith(".partial"):
                key = path.relative_to(root_for_bucket).as_posix()
                yield f"{parsed.scheme}://{parsed.bucket}/{key}"

    def list_prefixes(self, prefix_uri: str) -> Iterator[str]:
        """One directory read instead of walking the whole tree.

        The base class can derive this from list_keys, but that means visiting
        every file under the prefix. Here it is a single iterdir.
        """
        base = self._path(prefix_uri)
        if not base.is_dir():
            return
        for child in sorted(base.iterdir()):
            if child.is_dir():
                yield child.name

    @contextmanager
    def open_read(self, uri: str) -> Iterator[BinaryIO]:
        path = self._path(uri)
        with open(path, "rb") as fh:
            yield fh

    @contextmanager
    def open_append(self, uri: str) -> Iterator[BinaryIO]:
        path = self._path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as fh:
            yield fh

    def public_url(self, uri: str, relative_to: str | None = None) -> str:
        target = self._path(uri)
        if relative_to is not None:
            base = self._path(relative_to).parent
            try:
                return os.path.relpath(target, base).replace(os.sep, "/")
            except ValueError:
                pass  # different drive on Windows -- fall through to absolute
        return target.as_uri()

    # -- dev convenience ---------------------------------------------------
    @property
    def root(self) -> Path:
        """Only for CLI messages and tests. The pipeline must not call this."""
        return self._root
