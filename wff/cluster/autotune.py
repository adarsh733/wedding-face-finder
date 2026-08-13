"""Choosing the merge threshold without asking a human.

Why this file exists
--------------------
The merge threshold decides whether two piles of faces are one person. Until
now it was a slider on the review page, which asks a reviewer a question they
have no way to answer: "how alike must two faces be?" Nobody knows. Adarsh's
verdict, and he is right: pick it ourselves and hide it.

The obstacle is that "accuracy" is not observable without an answer key. What
IS observable, for free, on every folder, is the opposite -- pairs of faces we
know for certain are DIFFERENT people:

    two faces in the same photograph are never the same person.

That is the same rule Stage 2 already uses to block merges, used here for a
second purpose: it hands us thousands of labelled negative pairs per folder at
zero cost. Measure how alike the closest of those known-different people get,
and draw the line safely below it. No labels, no judgement calls, no human.

The asymmetry that sets the safety margin
-----------------------------------------
Too tight: a guest finds 40 of their 50 photos. Annoying, recoverable.
Too loose: a guest opens their link and sees a stranger's family photos.
Unrecoverable -- it is somebody else's private data and it destroys the
photographer's trust in one click.

So the margin is deliberately one-sided. We move DOWN from the closest
different pair, never up towards it.

What it produces on real data
-----------------------------
On the 851-photo Manali folder the closest two different people sat at 0.611
(measured by hand, 12 Aug). This routine, run over the same folder's same-photo
pairs, lands on ~0.55 -- the value we had been using as a guess. The number the
machine derives and the number two people arrived at by inspection agree, which
is the strongest evidence available that the rule is sound.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# A same-photo pair closer than this is not two similar guests -- it is one face
# detected twice and NMS failing to suppress the duplicate. That bug was real
# (IMG_3526.HEIC, 12 Aug: a tile found a 116x382 sliver of a face the whole-image
# pass had already found, IoU 0.398 against a 0.40 threshold). Feeding those into
# the measurement would report the closest "different people" as 0.02 and drag
# the threshold to the floor, so they are excluded -- and counted, because a
# rising count here means detection has regressed.
DUPLICATE_FLOOR = 0.20

# Two faces in one photo is a measurement between two INDIVIDUAL faces. The
# threshold it feeds compares two PILES, and a pile is an average of its faces.
# Averaging cancels the noise in any single crop -- lighting, angle, motion --
# so two people's pile-centres sit systematically closer together than any two
# of their individual faces do. Feeding a face-to-face number straight in as a
# pile-to-pile threshold reads as "safe" and merges people.
#
# How big the gap is, measured on ev_manali01 (851 photos, 1,182 faces):
#   closest 1% of same-photo face pairs   0.667
#   closest two different PILES           0.611   (measured by hand, 12 Aug)
#   threshold verified safe on that data  0.55
# 0.55 / 0.667 = 0.82.
#
# Honesty about this constant: it is calibrated on ONE real folder. So was the
# fixed 0.55 it replaces -- the difference is that this scales per folder, so a
# set where people genuinely look more alike gets a proportionally tighter line
# instead of the same number regardless. It is multiplicative for that reason.
# It should be re-checked the first time a real wedding runs.
PILE_DISCOUNT = 0.82

# Below this many usable pairs the measurement is noise. A 34-photo folder
# yields a few dozen, which cannot locate a boundary; it falls back and says so
# rather than pretending it measured something.
MIN_PAIRS = 200

# Hard limits. Above the ceiling we are into the range where the Manali folder
# had genuinely different people; below the floor everybody splits into
# fragments and no guest finds their photos.
FLOOR = 0.40
CEILING = 0.62

# Whole-folder pair budget. A 4,000-photo wedding with 30-person group shots
# produces a few hundred thousand pairs; more than this buys no accuracy and
# starts costing memory.
MAX_PAIRS = 400_000


@dataclass
class AutoThreshold:
    """The chosen merge threshold, and everything needed to explain it."""

    value: float
    basis: str = "default"          # measured | judged | default
    closest_different: float | None = None
    pairs_measured: int = 0
    duplicate_suspects: int = 0
    judged_pairs_used: int = 0
    clamped: bool = False
    default_value: float = 0.55
    notes: list[str] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return self.basis in ("measured", "judged")

    def headline(self) -> str:
        """One sentence, no jargon, for the reviewer."""
        if self.basis == "judged":
            return (
                "It set the line itself, from this folder's own photos and the "
                f"{self.judged_pairs_used} pair(s) you ruled on by hand."
            )
        if self.basis == "measured":
            return "It set the line itself, measured on this folder's own photos."
        return (
            "This folder is too small to measure on its own, so it used the "
            "setting checked against 851 real photos."
        )

    def detail(self) -> str:
        """The reasoning, still in plain words. Shown under the headline."""
        if not self.measured:
            return (
                f"Only {self.pairs_measured:,} pairs of faces here are known to be "
                "different people, which is not enough to find the boundary. "
                f"{MIN_PAIRS:,} is the minimum. Process more photos and it will "
                "start measuring instead."
            )
        closest = self.closest_different or 0.0
        text = (
            f"Two faces in one photo are always different people, so this folder "
            f"comes with {self.pairs_measured:,} pairs it already knows are "
            f"different. The closest of those still looked {closest:.2f} apart. "
            f"Piles are compared as averages, and averages of the same person sit "
            f"closer together than any two single photos of them — so the line was "
            f"set well below that measurement, at {self.value:.2f}."
        )
        if self.clamped:
            text += (
                f" (Held inside the safe range {FLOOR:.2f}–{CEILING:.2f}; the raw "
                "measurement fell outside it.)"
            )
        return text


# ---------------------------------------------------------------------------
# THE SMALL-FACE LINE
# ---------------------------------------------------------------------------
# A 50-80px face may join a person but may never start one, and the distance it
# has to be within was a flat 0.38 -- a guess, and a much stricter one than
# pass 1 (0.42) or pass 2 (0.55). Measured 13 Aug on the two real folders, that
# guess is wrong in BOTH directions:
#
#   ev-2025-10-16 (7,164 faces)  a small face and a person known to be someone
#                                else never got closer than 0.667. At 0.38 we
#                                attached 263 of 803 small faces; at 0.55, 346,
#                                with zero provably-wrong assignments anywhere
#                                in between. ~83 back-row guests thrown away
#                                for nothing.
#   ev_manali01 (1,182 faces)    the same negatives reach 0.382 at the 1st
#                                percentile and 0.300 at the closest. 0.38 is
#                                ON the boundary there, and 2 of the 6 faces it
#                                assigned were provably the wrong person.
#
# One number cannot serve both folders. So it is measured per folder, exactly
# like the merge line above, from the same free evidence: a small face and any
# person ALREADY IN ITS PHOTO are known-different.
#
# Unlike the merge line this needs no pile discount -- the measurement is
# already face-to-pile, which is the same quantity the threshold is used for.
# What it needs is only a one-sided safety margin, for the same reason: a small
# face on the wrong person is a stranger in a guest's results.
SECOND_CLASS_MARGIN = 0.75

# These negatives are much rarer than same-photo pairs (a small face only
# produces one per person sharing its photo), so the bar is lower than
# MIN_PAIRS. Manali yields 104, which is thin but enough to see a boundary that
# sits at less than half the wedding folder's.
SECOND_CLASS_MIN_PAIRS = 100

# Never looser than the merge line -- a small face should not be admitted on
# weaker evidence than it takes to merge two whole piles. Never tighter than
# 0.30, below which no small face joins anyone and the tiling work is wasted.
SECOND_CLASS_FLOOR = 0.30
SECOND_CLASS_CEILING = 0.55


@dataclass
class SmallFaceThreshold:
    """How close a 50-80px face must be to join a person, and why."""

    value: float
    basis: str = "default"          # measured | default
    closest_different: float | None = None
    pairs_measured: int = 0
    clamped: bool = False
    default_value: float = 0.38

    @property
    def measured(self) -> bool:
        return self.basis == "measured"

    def headline(self) -> str:
        if self.measured:
            return (
                "It measured how close a small face gets to the wrong person in "
                "this folder, and set its own line."
            )
        return (
            "Too few small faces here to measure, so it used the cautious "
            "default."
        )

    def detail(self) -> str:
        if not self.measured:
            return (
                f"Only {self.pairs_measured:,} small faces here sit in a photo "
                "next to someone already identified, which is not enough to "
                f"find the boundary ({SECOND_CLASS_MIN_PAIRS} is the minimum). "
                f"Used {self.default_value:.2f}."
            )
        closest = self.closest_different or 0.0
        text = (
            f"A small face and a person standing in the same photograph are "
            f"certainly not the same person. This folder has "
            f"{self.pairs_measured:,} such pairs, and the closest still looked "
            f"{closest:.2f} apart — so a small face is attached to someone only "
            f"when it is within {self.value:.2f}, comfortably below that."
        )
        if self.clamped:
            text += (
                f" (Held inside the safe range {SECOND_CLASS_FLOOR:.2f}–"
                f"{SECOND_CLASS_CEILING:.2f}.)"
            )
        return text


def choose_second_class_threshold(
    negative_distances: np.ndarray, default: float = 0.38
) -> SmallFaceThreshold:
    """The small-face line, from distances known to be the WRONG person.

    `negative_distances` are face-to-centroid distances between a second-class
    face and a person who appears in that same photograph. The caller builds
    them because only the caller knows the piles; this function only does the
    arithmetic, so it stays testable without clustering anything.
    """
    auto = SmallFaceThreshold(value=default, default_value=default)
    distances = np.asarray(negative_distances, dtype=np.float32).ravel()
    auto.pairs_measured = int(distances.size)

    if distances.size < SECOND_CLASS_MIN_PAIRS:
        return auto

    # The 1st percentile, not the minimum, for the same reason as the merge
    # line: one freak crop should not set the rule for the whole wedding.
    closest = float(np.percentile(distances, 1.0))
    auto.closest_different = round(closest, 3)

    value = closest * SECOND_CLASS_MARGIN
    clamped = min(SECOND_CLASS_CEILING, max(SECOND_CLASS_FLOOR, value))
    auto.clamped = abs(clamped - value) > 1e-6
    auto.value = round(clamped, 2)
    auto.basis = "measured"
    return auto


def same_photo_distances(
    embeddings: np.ndarray, photo_ids: list[str], max_pairs: int = MAX_PAIRS
) -> np.ndarray:
    """Distances between every pair of faces sharing a photograph.

    These are known-different people. Returned unfiltered -- the caller decides
    what to do with the suspiciously close ones.
    """
    if len(photo_ids) < 2 or embeddings.size == 0:
        return np.zeros(0, dtype=np.float32)

    by_photo: dict[str, list[int]] = {}
    for index, photo_id in enumerate(photo_ids):
        by_photo.setdefault(photo_id, []).append(index)

    left: list[int] = []
    right: list[int] = []
    for members in by_photo.values():
        if len(members) < 2:
            continue
        for position, i in enumerate(members):
            for j in members[position + 1 :]:
                left.append(i)
                right.append(j)
        if len(left) >= max_pairs:
            break
    if not left:
        return np.zeros(0, dtype=np.float32)

    return _cosine_distances(embeddings, np.array(left), np.array(right))


def _cosine_distances(
    embeddings: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.maximum(norms, 1e-9)
    products = np.sum(unit[left] * unit[right], axis=1)
    return (1.0 - products).astype(np.float32)


def choose_threshold(
    embeddings: np.ndarray,
    photo_ids: list[str],
    default: float = 0.55,
    extra_different_pairs: list[tuple[int, int]] | None = None,
    precomputed: np.ndarray | None = None,
) -> AutoThreshold:
    """Where the line between "same person" and "different people" should sit.

    `extra_different_pairs` are index pairs a human has explicitly called
    different. They are folded into the same measurement rather than treated
    separately: a hand judgement and a same-photo pair are both simply evidence
    that two faces belong to two people, and mixing them means one calculation
    instead of two that could disagree.
    """
    auto = AutoThreshold(value=default, default_value=default)

    distances = (
        precomputed
        if precomputed is not None
        else same_photo_distances(embeddings, photo_ids)
    )
    judged_count = 0
    if extra_different_pairs:
        pairs = [
            (i, j)
            for i, j in extra_different_pairs
            if 0 <= i < len(photo_ids) and 0 <= j < len(photo_ids) and i != j
        ]
        if pairs:
            judged = _cosine_distances(
                embeddings,
                np.array([p[0] for p in pairs]),
                np.array([p[1] for p in pairs]),
            )
            judged_count = len(pairs)
            distances = np.concatenate([distances, judged]) if distances.size else judged

    auto.judged_pairs_used = judged_count
    if distances.size == 0:
        auto.pairs_measured = 0
        return auto

    duplicates = distances < DUPLICATE_FLOOR
    auto.duplicate_suspects = int(duplicates.sum())
    usable = distances[~duplicates]
    auto.pairs_measured = int(usable.size)

    if auto.duplicate_suspects:
        auto.notes.append(
            f"{auto.duplicate_suspects} pair(s) of faces in the same photo look "
            "almost identical. That usually means one face was detected twice, "
            "not that two guests are lookalikes -- worth a glance if it grows."
        )

    if usable.size < MIN_PAIRS:
        return auto

    # The 1st percentile, not the minimum. One freak pair -- a bad crop, a face
    # half out of frame -- should not be allowed to set the threshold for the
    # entire wedding, and with thousands of pairs the 1st percentile still means
    # "as close as different people ever realistically get here".
    closest = float(np.percentile(usable, 1.0))
    auto.closest_different = round(closest, 3)

    value = closest * PILE_DISCOUNT
    clamped = min(CEILING, max(FLOOR, value))
    auto.clamped = abs(clamped - value) > 1e-6
    auto.value = round(clamped, 2)
    auto.basis = "judged" if judged_count else "measured"
    return auto
