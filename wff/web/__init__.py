"""The local testing console.

A small web page Adarsh and Devesh point at a folder. It runs the same Stage 1
and Stage 2 the CLI runs -- no second pipeline, no duplicated logic -- and then
shows every cluster in full so mistakes can be counted by hand, and captures
those hand judgements as reusable ground truth.

    python -m wff ui

Local only, no login, no upload. Week 1-2 tooling, not the product.
"""
from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
