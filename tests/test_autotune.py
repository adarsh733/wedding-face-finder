"""Tests for choosing the merge threshold without a human.

This is the file that decides how the product groups faces, with nobody
watching. Every test here is aimed at a way it could be wrong *quietly* -- a
threshold that is too loose does not raise anything, it just shows a guest a
stranger's photos.
"""
from __future__ import annotations

import numpy as np
import pytest

from wff.cluster.autotune import (
    CEILING,
    FLOOR,
    MIN_PAIRS,
    PILE_DISCOUNT,
    SECOND_CLASS_CEILING,
    SECOND_CLASS_FLOOR,
    SECOND_CLASS_MIN_PAIRS,
    choose_second_class_threshold,
    choose_threshold,
    same_photo_distances,
)

DIM = 64


def _unit(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=DIM)
    return vector / np.linalg.norm(vector)


def _at_distance(rng: np.random.Generator, anchor: np.ndarray, distance: float):
    """A second vector exactly `distance` (cosine) away from `anchor`."""
    other = _unit(rng)
    perpendicular = other - anchor * float(np.dot(other, anchor))
    perpendicular /= np.linalg.norm(perpendicular)
    cosine = 1.0 - distance
    return anchor * cosine + perpendicular * float(np.sqrt(max(0.0, 1 - cosine**2)))


def _folder(distances: list[float], seed: int = 7):
    """One folder: each entry becomes a photo holding two different people."""
    rng = np.random.default_rng(seed)
    vectors: list[np.ndarray] = []
    photo_ids: list[str] = []
    for index, distance in enumerate(distances):
        anchor = _unit(rng)
        vectors.append(anchor)
        vectors.append(_at_distance(rng, anchor, distance))
        photo_ids += [f"{index:06d}", f"{index:06d}"]
    return np.stack(vectors).astype(np.float32), photo_ids


def test_two_faces_in_one_photo_are_used_as_known_different_people():
    """The whole idea: a folder labels its own negatives, for free."""
    embeddings, photo_ids = _folder([0.75] * 400)
    distances = same_photo_distances(embeddings, photo_ids)
    assert len(distances) == 400
    assert distances.min() == pytest.approx(0.75, abs=0.01)


def test_the_line_is_drawn_below_the_closest_different_people():
    embeddings, photo_ids = _folder(list(np.linspace(0.61, 0.95, 500)))
    auto = choose_threshold(embeddings, photo_ids, default=0.55)

    assert auto.basis == "measured"
    assert auto.closest_different == pytest.approx(0.61, abs=0.02)
    # Discounted from a face-to-face measurement to a pile-to-pile threshold,
    # and so always well below the closest pair. Never above it.
    assert auto.value == pytest.approx(0.61 * PILE_DISCOUNT, abs=0.02)
    assert auto.value < auto.closest_different


def test_a_duplicate_detection_cannot_drag_the_threshold_to_the_floor():
    """The failure this file exists to prevent.

    A face detected twice in one photo shows up as a "different people" pair at
    distance ~0.02. It is a detection bug, not two lookalike guests -- and one
    real one has already happened (IMG_3526.HEIC, 12 Aug). Left in the
    measurement it would drive the threshold to the floor and split every guest
    at the wedding into fragments.
    """
    clean = list(np.linspace(0.61, 0.95, 500))
    honest = choose_threshold(*_folder(clean), default=0.55)
    polluted = choose_threshold(*_folder(clean + [0.02] * 25), default=0.55)

    assert polluted.duplicate_suspects == 25
    assert polluted.value == pytest.approx(honest.value, abs=0.02)
    assert polluted.notes, "a duplicate that close deserves saying out loud"


def test_a_folder_too_small_to_measure_says_so_instead_of_guessing():
    embeddings, photo_ids = _folder([0.7] * 12)
    auto = choose_threshold(embeddings, photo_ids, default=0.55)

    assert auto.basis == "default"
    assert auto.value == 0.55
    assert not auto.measured
    assert str(MIN_PAIRS) in auto.detail() or "not enough" in auto.detail()


def test_no_faces_at_all_is_not_a_crash():
    auto = choose_threshold(np.zeros((0, DIM), dtype=np.float32), [], default=0.55)
    assert auto.value == 0.55
    assert auto.basis == "default"


def test_the_answer_is_held_inside_the_safe_range():
    """Whatever a folder measures, the result stays somewhere sane.

    A folder of identical twins would measure a tiny gap and ask for a
    threshold that splits everybody; a folder of ten very different people
    would ask for one loose enough to merge strangers.
    """
    far_embeddings, far_photos = _folder([0.95] * 500)
    far = choose_threshold(far_embeddings, far_photos, default=0.55)
    assert far.value == CEILING
    assert far.clamped

    close_embeddings, close_photos = _folder([0.30] * 500)
    close = choose_threshold(close_embeddings, close_photos, default=0.55)
    assert close.value == FLOOR
    assert close.clamped


def test_hand_judgements_tighten_the_line_and_are_reported():
    """A human saying "different" is the same evidence as a shared photo.

    It has to actually move the answer, or judging pairs is theatre.
    """
    embeddings, photo_ids = _folder(list(np.linspace(0.70, 0.95, 400)))
    plain = choose_threshold(embeddings, photo_ids, default=0.55)

    # Two faces from different photos that a reviewer has called different, and
    # which happen to look much more alike than anything measured so far.
    rng = np.random.default_rng(3)
    anchor = _unit(rng)
    close_pair = np.stack([anchor, _at_distance(rng, anchor, 0.50)]).astype(np.float32)
    embeddings = np.concatenate([embeddings, close_pair])
    photo_ids = photo_ids + ["900001", "900002"]
    judged_index = (len(photo_ids) - 2, len(photo_ids) - 1)

    judged = choose_threshold(
        embeddings,
        photo_ids,
        default=0.55,
        extra_different_pairs=[judged_index] * 20,
    )
    assert judged.basis == "judged"
    assert judged.judged_pairs_used == 20
    assert judged.value < plain.value


def test_the_same_folder_always_gets_the_same_answer():
    """Repeatability. A threshold that drifts between page loads would make
    every measurement taken against it meaningless."""
    embeddings, photo_ids = _folder(list(np.linspace(0.61, 0.95, 500)))
    first = choose_threshold(embeddings, photo_ids, default=0.55)
    second = choose_threshold(embeddings, photo_ids, default=0.55)
    assert first.value == second.value


def test_photos_with_one_face_contribute_nothing():
    """A folder of selfies cannot label anything, and must not pretend to."""
    rng = np.random.default_rng(11)
    embeddings = np.stack([_unit(rng) for _ in range(300)]).astype(np.float32)
    photo_ids = [f"{i:06d}" for i in range(300)]

    assert len(same_photo_distances(embeddings, photo_ids)) == 0
    auto = choose_threshold(embeddings, photo_ids, default=0.55)
    assert auto.basis == "default"


def test_a_manali_shaped_folder_lands_on_the_verified_threshold():
    """The one real calibration point, locked down.

    ev_manali01 (851 photos, 1,182 faces) measures 0.667 at the 1st percentile
    of its same-photo pairs. Two people were verified by hand to sit 0.611
    apart there, and 0.55 was verified as a safe merge threshold on that data.
    A folder shaped like it must therefore come out at 0.55 -- if a future
    change to the discount moves this, it is moving the only number we have
    ever checked against reality.
    """
    # A tail that puts the 1st percentile at 0.667, like the real folder.
    distances = [0.36, 0.39, 0.41, 0.45, 0.56] + list(np.linspace(0.667, 0.95, 495))
    auto = choose_threshold(*_folder(distances), default=0.55)

    assert auto.closest_different == pytest.approx(0.667, abs=0.02)
    assert auto.value == pytest.approx(0.55, abs=0.015)


# -- the small-face line -----------------------------------------------------
#
# 50-80px faces are the back row of the group shot -- the exact photos the
# tiled-detection work exists to rescue. The line deciding whether one may join
# a person was a flat 0.38, and it is the setting with the least evidence
# behind it in the whole system.


def test_small_face_line_is_measured_from_known_wrong_people():
    """Distances to people the small face is provably NOT. The line must land
    comfortably below the closest of them."""
    negatives = np.array([0.70] * 50 + [0.75] * 100 + [0.80] * 100, dtype=np.float32)
    auto = choose_second_class_threshold(negatives, default=0.38)

    assert auto.basis == "measured"
    assert auto.value < float(np.percentile(negatives, 1.0))
    assert auto.pairs_measured == 250


def test_small_face_line_tightens_when_the_wrong_people_are_close():
    """ev_manali01 shaped: the negatives reach 0.382 at the 1st percentile, and
    0.38 was measured to assign 2 wrong faces out of 6. The measurement must
    move DOWN from the default, not up."""
    negatives = np.concatenate([
        np.linspace(0.30, 0.40, 20),
        np.linspace(0.40, 0.95, 130),
    ]).astype(np.float32)
    auto = choose_second_class_threshold(negatives, default=0.38)

    assert auto.value < 0.38, "a folder whose wrong people sit close must tighten"
    assert auto.value >= SECOND_CLASS_FLOOR


def test_small_face_line_loosens_when_the_wrong_people_are_far():
    """ev-2025-10-16 shaped: negatives never closer than 0.667, so 0.38 was
    throwing away back-row guests for nothing."""
    negatives = np.linspace(0.70, 0.95, 400).astype(np.float32)
    auto = choose_second_class_threshold(negatives, default=0.38)

    assert auto.value > 0.38
    assert auto.value <= SECOND_CLASS_CEILING


def test_small_face_line_never_exceeds_the_merge_line():
    """A single small face must never be admitted on weaker evidence than it
    takes to merge two entire piles."""
    negatives = np.full(500, 0.99, dtype=np.float32)
    auto = choose_second_class_threshold(negatives, default=0.38)
    assert auto.value <= SECOND_CLASS_CEILING
    assert auto.clamped


def test_small_face_line_falls_back_when_there_is_nothing_to_measure():
    thin = np.full(SECOND_CLASS_MIN_PAIRS - 1, 0.8, dtype=np.float32)
    auto = choose_second_class_threshold(thin, default=0.38)
    assert auto.basis == "default"
    assert auto.value == 0.38
    assert not auto.measured

    empty = choose_second_class_threshold(np.zeros(0, dtype=np.float32), default=0.38)
    assert empty.basis == "default"
    assert empty.value == 0.38


def test_small_face_line_explains_itself_without_jargon():
    """The reviewer is not an engineer. Both sentences must read as English."""
    auto = choose_second_class_threshold(
        np.linspace(0.70, 0.95, 400).astype(np.float32), default=0.38
    )
    for text in (auto.headline(), auto.detail()):
        assert text and text[0].isupper() and text.rstrip().endswith((".", ")"))
        for jargon in ("centroid", "cosine", "embedding", "percentile", "threshold"):
            assert jargon not in text.lower(), f"{jargon!r} leaked into reviewer copy"
