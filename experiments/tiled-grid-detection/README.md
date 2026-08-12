# Idea A — Calibrated fixed-grid tiling

See [`docs/FACE-DETECTION-EXPERIMENTS.md`](../../docs/FACE-DETECTION-EXPERIMENTS.md) for the
context and the comparison protocol against Idea B
(`experiments/multiscale-pyramid-detection/`).

## Hypothesis

The current `docs/ARCHITECTURE.md` Step 3 tiling (whole image + fixed 3×3 grid) under-tiles —
SCRFD's internal ~640px resize can shrink a back-row face below the detection floor even inside a
tile. Sizing tiles from the detector's actual resize factor, instead of guessing 3×3, should catch
those faces without changing the rest of the pipeline.

## What "done" looks like

- A tile-size formula derived from the detector's internal input size and a target minimum
  detectable face height, not a hardcoded grid count.
- Overlap + NMS at tile borders so a face split across a seam doesn't get lost or double-counted.
- Detection recall by face-height bucket, measured against a real photo once available
  (`docs/ARCHITECTURE.md` Open Item #2) — not just the cricket-player set.

## Status

Not started. Claim this folder in `.claude/ACTIVE-WORK.md` before working in it.
