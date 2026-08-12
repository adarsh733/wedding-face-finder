# Two small-face detection experiments — for Adarsh

**11 Aug review flagged a risk in `docs/ARCHITECTURE.md` Step 3.** The plan detects faces at
native resolution to fix Draft 1's shrink-before-detect bug — correct — but does it via a fixed
**3×3 tile grid**. The arithmetic doesn't obviously work out: a 6000px photo cut into 3 columns
gives ~2000-2200px tiles; SCRFD resizes its input to ~640px internally; that's another ~3.4×
downscale *inside* each tile. A 90px back-row face becomes ~26px to the detector — below even our
own 50px "unreliable" floor. The full-resolution crop-for-embedding part of the fix is still sound;
what's unproven is whether the detector even *finds* the box for a small face in the first place.

We don't know which fix is right without real data, and we're both going to want to poke at this.
Splitting it into two independent approaches, each in its own folder, means neither of us blocks
or overwrites the other's in-progress work — pick either one (or both, if you want a horse race),
claim it in `.claude/ACTIVE-WORK.md` the usual way, and go.

---

## Idea A — [`experiments/tiled-grid-detection/`](../experiments/tiled-grid-detection/)

**Fix the grid, keep the architecture.** Same tiling strategy as the current doc, but the tile
size is *calculated* from the detector's internal resize instead of guessed at 3×3. If SCRFD
resizes to 640px and we want a 90px face to survive as ≥50px, tiles need to stay under
`640 × 50 / 90 ≈ 355px`... which is a lot of tiles (something like 6×4 to 8×6 depending on the
real face-size distribution). Add overlap + NMS dedup at tile borders to stop faces getting cut in
half at a seam.

- Simple to reason about, deterministic, easy to cost (compute scales roughly with tile count).
- Risk: still a fixed grid — may over-tile empty sky/floor and under-tile a dense group shot.

## Idea B — [`experiments/multiscale-pyramid-detection/`](../experiments/multiscale-pyramid-detection/)

**Skip hard tiling, go multi-scale instead.** Run a coarse full-image pass to find where the
*people* are (roughly), then re-detect at native resolution only in those regions, at whatever
scale keeps faces above the size floor — no fixed grid, no seam-splitting problem. Closer to how
production face-search systems typically handle wide scenes with variable subject density (a bride
close-up and a 30-person group shot don't need the same treatment).

- Adapts to actual face density instead of a one-size grid.
- Risk: more moving parts, a second detector pass to tune, different compute profile — needs its
  own benchmark to know if it's actually faster or slower than Idea A in practice.

---

## Shared success metric (use this for both, so results are comparable)

Recall of face **detection** (not embedding) bucketed by native face height — same buckets as
`docs/ARCHITECTURE.md`'s pixel table (15px / 50px / 112px / 400px / 1500px). Test against the
cricket-player set first for plumbing, but **the real comparison only means something once we have
a real 24MP wedding photo** (`docs/ARCHITECTURE.md` Open Item #2) — cropped press photos don't
have a back row to lose. Don't declare a winner off synthetic data alone.

## How to not collide

- Idea A only ever touches `experiments/tiled-grid-detection/`. Idea B only ever touches
  `experiments/multiscale-pyramid-detection/`. Neither touches the other's folder or
  `scripts/build_test_dataset.py`.
- Claim your folder in `.claude/ACTIVE-WORK.md` before starting, same protocol as everything else
  in this repo.
- Whichever wins (or if both are good enough) becomes Step 3's detection strategy in
  `docs/ARCHITECTURE.md` — update that doc once there's a real result, not before.
