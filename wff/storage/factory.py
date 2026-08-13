from __future__ import annotations

from ..config import Config, load_config
from .base import ObjectStore
from .local import LocalObjectStore


def get_store(config: Config | None = None) -> ObjectStore:
    cfg = config or load_config()
    backend = cfg.storage.backend.lower()
    if backend == "local":
        return LocalObjectStore(cfg.storage.local_root)
    if backend == "r2":
        # Deliberately not written yet -- decided 12 Aug to stay on the local
        # backend for week 1. The interface is what protects the migration;
        # this is a ~60-line boto3 subclass against the R2 S3-compatible
        # endpoint whenever credentials exist.
        raise NotImplementedError(
            "R2 backend not implemented yet. Set WFF_STORAGE=local. "
            "Adding it means implementing ObjectStore against boto3 -- no "
            "pipeline code changes."
        )
    raise ValueError(f"Unknown storage backend: {cfg.storage.backend!r}")
