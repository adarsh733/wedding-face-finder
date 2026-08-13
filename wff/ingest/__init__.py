"""Ingest adapters. One interface, many sources.

Implemented now:
    local   -- a folder on this machine (dev, and the cricket test set)
    gdrive  -- a PUBLIC Drive folder, read anonymously (UNVERIFIED, open item #1)

Deliberately not written yet -- decided 12 Aug to keep week 1 focused on proving
clustering works. Each is one new file implementing IngestAdapter; nothing
downstream changes:
    dropbox -- shared-link API, or ?dl=1 for a zip
    zip     -- direct zip link, stream-extracted entry by entry over HTTP range
    onedrive, upload
"""
from __future__ import annotations

from .base import (
    FileKind,
    FileRef,
    IngestAdapter,
    SourceStats,
    ValidationResult,
    classify,
)
from .gdrive import GoogleDriveFolderAdapter, extract_folder_id
from .local_folder import LocalFolderAdapter

__all__ = [
    "FileKind",
    "FileRef",
    "IngestAdapter",
    "SourceStats",
    "ValidationResult",
    "classify",
    "GoogleDriveFolderAdapter",
    "LocalFolderAdapter",
    "extract_folder_id",
    "adapter_for",
]

_NOT_YET = {
    "dropbox.com": "Dropbox",
    "1drv.ms": "OneDrive",
    "onedrive.live.com": "OneDrive",
    "wetransfer.com": "WeTransfer",
    "pixieset.com": "Pixieset",
    "pic-time.com": "Pic-Time",
    "shootproof.com": "ShootProof",
}


def adapter_for(link: str) -> IngestAdapter:
    """Detect what kind of link this is. The photographer pastes one box."""
    lowered = link.lower()
    if "drive.google.com" in lowered:
        return GoogleDriveFolderAdapter(link)
    if lowered.startswith("file://"):
        return LocalFolderAdapter(link)
    if "://" not in lowered:
        # Either a filesystem path or a bare Drive folder id pasted alone.
        # A path always contains a separator or a dot; an id never does.
        if extract_folder_id(link) and not any(c in link for c in "/\\."):
            return GoogleDriveFolderAdapter(link)
        return LocalFolderAdapter(link)
    for host, name in _NOT_YET.items():
        if host in lowered:
            raise NotImplementedError(
                f"{name} links are designed for but not implemented yet "
                f"(see wff/ingest/__init__.py). Use a local folder for now."
            )
    raise ValueError(f"Do not know how to ingest this link: {link!r}")
