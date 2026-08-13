"""Wedding Face Finder.

Stage 1 (expensive, once per wedding):
    photos -> detect at native resolution -> quality gate -> 512-d embeddings
           -> faces.parquet + face crops + previews/thumbnails

Stage 2 (cheap, re-runs in seconds, unlimited times):
    faces.parquet -> two-pass clustering -> people -> HTML accuracy report

See docs/ARCHITECTURE.md. The stage split is what makes CPU-only tuning viable.
"""

__version__ = "0.1.0"
