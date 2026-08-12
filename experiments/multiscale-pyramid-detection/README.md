# Idea B — Multi-scale / coarse-to-fine detection

See [`docs/FACE-DETECTION-EXPERIMENTS.md`](../../docs/FACE-DETECTION-EXPERIMENTS.md) for the
context and the comparison protocol against Idea A (`experiments/tiled-grid-detection/`).

## Hypothesis

A fixed tile grid is the wrong shape for a wedding photo — a bride close-up and a 30-person group
shot don't have the same face density. A coarse first pass to find where people roughly are,
followed by native-resolution re-detection only in those regions, should adapt to actual density
and avoid the seam-splitting problem tiling has.

## What "done" looks like

- A coarse pass (cheap, whole-image) that produces candidate regions to re-examine.
- A fine pass at native resolution / whatever scale keeps faces above the detection floor, run
  only on those candidate regions.
- Detection recall by face-height bucket, measured against a real photo once available
  (`docs/ARCHITECTURE.md` Open Item #2) — not just the cricket-player set.
- A runtime comparison against Idea A — more moving parts here, so it needs to actually win to be
  worth the complexity.

## Status

Not started. Claim this folder in `.claude/ACTIVE-WORK.md` before working in it.
