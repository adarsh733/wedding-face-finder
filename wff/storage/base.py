"""The ObjectStore interface. One shape, many backends."""
from __future__ import annotations

import posixpath
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import BinaryIO, Iterator

DEFAULT_SCHEME = "r2"


@dataclass(frozen=True)
class ObjectURI:
    scheme: str
    bucket: str
    key: str

    def __str__(self) -> str:
        return f"{self.scheme}://{self.bucket}/{self.key}"


def parse_uri(uri: str) -> ObjectURI:
    """'r2://wff/events/ev_ab12/prev/000481.jpg' -> ObjectURI.

    Raises on anything that looks like a local filesystem path. That rejection
    is deliberate: it is how the no-local-paths rule stays enforced instead of
    being a comment nobody reads.
    """
    if "://" not in uri:
        raise ValueError(
            f"Not an object URI: {uri!r}. The pipeline only accepts "
            f"{DEFAULT_SCHEME}://bucket/key -- never a filesystem path. "
            "See 'The one rule that keeps the GPU migration cheap' in "
            "docs/ARCHITECTURE.md."
        )
    scheme, rest = uri.split("://", 1)
    if "/" not in rest:
        raise ValueError(f"Object URI has no key: {uri!r}")
    bucket, key = rest.split("/", 1)
    if not bucket:
        raise ValueError(f"Object URI has no bucket: {uri!r}")
    return ObjectURI(scheme=scheme, bucket=bucket, key=key.lstrip("/"))


def join_uri(uri: str, *parts: str) -> str:
    """Append path segments to an object URI."""
    parsed = parse_uri(uri)
    key = posixpath.join(parsed.key, *[p.strip("/") for p in parts if p])
    return str(ObjectURI(parsed.scheme, parsed.bucket, key))


class ObjectStore(ABC):
    """Read and write objects. Backends: local folder today, R2 tomorrow."""

    @abstractmethod
    def put_bytes(self, uri: str, data: bytes, content_type: str | None = None) -> None:
        ...

    @abstractmethod
    def get_bytes(self, uri: str) -> bytes:
        ...

    @abstractmethod
    def exists(self, uri: str) -> bool:
        ...

    @abstractmethod
    def delete(self, uri: str) -> None:
        ...

    @abstractmethod
    def list_keys(self, prefix_uri: str) -> Iterator[str]:
        """Yield full object URIs under a prefix."""
        ...

    def list_prefixes(self, prefix_uri: str) -> Iterator[str]:
        """Yield the immediate child "folder" names under a prefix.

        Exists because listing every key to discover the event ids meant walking
        every crop, preview and thumbnail in the store -- roughly ten thousand
        files to learn six names, which showed up as a ~600 ms delay on each
        keystroke in the forecast box.

        This is not a filesystem concession: S3 and R2 do exactly this natively
        with `Delimiter='/'`, returning CommonPrefixes. The default
        implementation below derives them from `list_keys` so a backend only
        overrides it if it can do better -- which both of ours can.
        """
        seen: set[str] = set()
        base = parse_uri(prefix_uri)
        for uri in self.list_keys(prefix_uri):
            remainder = parse_uri(uri).key[len(base.key):].lstrip("/")
            head = remainder.split("/", 1)[0]
            if head and head not in seen:
                seen.add(head)
                yield head

    @abstractmethod
    @contextmanager
    def open_read(self, uri: str) -> Iterator[BinaryIO]:
        """Stream an object for reading. Never loads it whole unless asked."""
        ...

    @abstractmethod
    @contextmanager
    def open_append(self, uri: str) -> Iterator[BinaryIO]:
        """Append bytes to an object.

        Real object stores cannot append; R2's implementation will buffer and
        multipart-upload. Stage 1 uses this for its crash-safe append-only
        journals, which is exactly the access pattern that makes a 4-hour job
        resumable instead of restartable.
        """
        ...

    @abstractmethod
    def public_url(self, uri: str, relative_to: str | None = None) -> str:
        """A URL an HTML report can link to.

        Local backend returns a relative filesystem link; R2 will return a
        signed or public CDN URL.
        """
        ...

    def put_text(self, uri: str, text: str, content_type: str = "text/plain") -> None:
        self.put_bytes(uri, text.encode("utf-8"), content_type)

    def get_text(self, uri: str) -> str:
        return self.get_bytes(uri).decode("utf-8")
