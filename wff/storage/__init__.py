"""Object storage behind a URI.

THE RULE (docs/ARCHITECTURE.md): the worker must never reference a local
filesystem path. Every input and output is an object URI -- r2://bucket/key --
even while everything runs on one desktop. Break this in week 1 and moving to a
rented GPU becomes a rewrite instead of a settings change.

Today the only backend is LocalObjectStore, which materialises those URIs under
a folder. The pipeline cannot tell, and that is the entire point.
"""
from .base import ObjectStore, ObjectURI, parse_uri, join_uri
from .local import LocalObjectStore
from .factory import get_store

__all__ = [
    "ObjectStore",
    "ObjectURI",
    "parse_uri",
    "join_uri",
    "LocalObjectStore",
    "get_store",
]
