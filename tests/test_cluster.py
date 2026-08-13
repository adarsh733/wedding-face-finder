"""Two-pass clustering.

The property under test throughout:

    Splitting a person is a mild failure. Merging two people is a catastrophe.

So the merge-prevention tests are the important ones. A test that only checks
"the right number of clusters came out" would pass while the system showed a
guest a stranger.
"""
from __future__ import annotations

import numpy as np
import pytest

from wff.cluster.metrics import evaluate, sanity_check
from wff.cluster.two_pass import cluster_faces
from wff.config import ClusterConfig


DIM = 512


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _faces_around(base: np.ndarray, count: int, spread: float, seed: int) -> np.ndarray:
    """`count` faces of one person, scattered around `base`.

    Note on 512 dimensions, because it is a genuine trap: adding Gaussian noise
    of scale s to a unit vector adds a perturbation of norm s*sqrt(512), so
    "scale 0.05" is actually a perturbation of norm 1.13 -- larger than the
    signal, and the faces come out near-orthogonal. Here `spread` is instead
    the magnitude of a perpendicular offset, so the angle is atan(spread) and
    the cosine distance is a predictable ~spread^2/2.
    """
    rng = np.random.default_rng(seed)
    faces = []
    for _ in range(count):
        offset = rng.normal(size=DIM)
        offset -= (offset @ base) * base          # keep it perpendicular
        offset = _unit(offset) * spread
        faces.append(_unit(base + offset))
    return np.array(faces, dtype=np.float32)


def _person_vectors(seed: int, count: int, spread: float = 0.30) -> np.ndarray:
    """`count` faces of one distinct person (a random direction of their own)."""
    rng = np.random.default_rng(seed * 1000 + 17)
    base = _unit(rng.normal(size=DIM))
    return _faces_around(base, count, spread, seed)


def _nearby_bases(count: int, separation: float, seed: int = 99) -> list[np.ndarray]:
    """`count` base directions strung along one line -- lookalike relatives.

    Consecutive bases are closer to each other than to the ones beyond, which
    is what lets a merge CHAIN be tested.
    """
    rng = np.random.default_rng(seed)
    axis = _unit(rng.normal(size=DIM))
    other = rng.normal(size=DIM)
    other = _unit(other - (other @ axis) * axis)
    return [_unit(axis + i * separation * other) for i in range(count)]


def test_distinct_people_stay_distinct():
    embeddings = np.vstack([_person_vectors(1, 8), _person_vectors(2, 8), _person_vectors(3, 8)])
    photo_ids = [f"p{i:03d}" for i in range(24)]
    result = cluster_faces(
        embeddings,
        photo_ids,
        np.ones(24, dtype=np.float32),
        np.zeros(24, dtype=bool),
        ClusterConfig(),
    )
    assert len(result.persons) == 3
    for person in result.persons:
        assert person.face_count == 8


def test_same_photo_rule_prevents_a_merge():
    """THE test. Two lookalike cousins whose embeddings are close, but who
    appear together in a photo, must never be merged -- co-occurrence is proof
    they are different people, and it beats any distance measurement.
    """
    # Deliberately near-identical: distance alone would merge these.
    base_a, base_b = _nearby_bases(2, separation=0.45, seed=7)
    cousin_a = _faces_around(base_a, 6, spread=0.1, seed=71)
    cousin_b = _faces_around(base_b, 6, spread=0.1, seed=72)
    embeddings = np.vstack([cousin_a, cousin_b])

    # Each pair shares a photo: they were photographed standing together.
    photo_ids = [f"shared{i}" for i in range(6)] * 2

    config = ClusterConfig(pass1_threshold=0.05, pass2_threshold=0.9)
    with_rule = cluster_faces(
        embeddings, photo_ids, np.ones(12, dtype=np.float32),
        np.zeros(12, dtype=bool), config,
    )
    assert with_rule.blocked_merges_same_photo > 0

    for person in with_rule.persons:
        photos = [photo_ids[i] for i in person.face_indices]
        assert len(photos) == len(set(photos)), (
            "a person contains two faces from the same photo -- that is "
            "impossible and means the same-photo rule leaked"
        )

    without_rule = cluster_faces(
        embeddings, photo_ids, np.ones(12, dtype=np.float32), np.zeros(12, dtype=bool),
        ClusterConfig(pass1_threshold=0.05, pass2_threshold=0.9,
                      enforce_same_photo_rule=False),
    )
    assert len(without_rule.persons) < len(with_rule.persons), (
        "without the rule these should collapse -- if not, the test data is "
        "not actually testing the rule"
    )


def test_same_photo_rule_holds_through_a_merge_chain():
    """A merges with B, B merges with C -- but A and C share a photo.

    Checking the rule only on the two groups being merged, rather than on their
    merged photo sets, would let this slip through.
    """
    bases = _nearby_bases(3, separation=0.45, seed=11)
    groups = [_faces_around(base, 4, spread=0.1, seed=110 + i) for i, base in enumerate(bases)]
    embeddings = np.vstack(groups)

    photo_ids = (
        [f"a{i}" for i in range(4)]
        + [f"b{i}" for i in range(4)]
        + [f"a{i}" for i in range(4)]   # group C shares every photo with group A
    )
    result = cluster_faces(
        embeddings, photo_ids, np.ones(12, dtype=np.float32), np.zeros(12, dtype=bool),
        ClusterConfig(pass1_threshold=0.05, pass2_threshold=0.9),
    )
    for person in result.persons:
        photos = [photo_ids[i] for i in person.face_indices]
        assert len(photos) == len(set(photos)), "merge chain defeated the same-photo rule"


def test_second_class_faces_cannot_start_a_person():
    """50-80px faces may JOIN a person, never START one.

    A person invented out of blurry back-row faces is how "everyone's blurry
    face lands near everyone else's" turns into a wrong match.
    """
    embeddings = _person_vectors(5, 10)
    photo_ids = [f"p{i}" for i in range(10)]
    result = cluster_faces(
        embeddings, photo_ids, np.ones(10, dtype=np.float32),
        np.ones(10, dtype=bool),      # every face second-class
        ClusterConfig(),
    )
    assert result.persons == []
    assert result.leftover_face_count == 10
    assert result.warnings


def test_second_class_faces_join_an_existing_person():
    rng = np.random.default_rng(600)
    base = _unit(rng.normal(size=DIM))
    big = _faces_around(base, 8, spread=0.30, seed=601)
    small = _faces_around(base, 4, spread=0.35, seed=602)   # same person, small faces
    embeddings = np.vstack([big, small])
    photo_ids = [f"p{i}" for i in range(12)]
    second_class = np.array([False] * 8 + [True] * 4)

    result = cluster_faces(
        embeddings, photo_ids, np.ones(12, dtype=np.float32), second_class,
        ClusterConfig(second_class_assign_threshold=0.5),
    )
    assert len(result.persons) == 1
    assert result.second_class_assigned == 4
    assert result.persons[0].face_count == 12


def test_pass1_never_births_a_group_holding_one_photo_twice():
    """Pass 2 refuses to MERGE two groups sharing a photo, but pass 1 used to
    have no such rule -- a group could be born already impossible, and pass 2
    would then build on the contradiction.

    Two people close enough that a tight pass 1 puts them together, standing in
    the same photographs. The invariant is absolute: no person, ever, holds two
    faces from one photograph.
    """
    base_a, base_b = _nearby_bases(2, separation=0.25, seed=31)
    a = _faces_around(base_a, 5, spread=0.05, seed=311)
    b = _faces_around(base_b, 5, spread=0.05, seed=312)
    embeddings = np.vstack([a, b])
    photo_ids = [f"together{i}" for i in range(5)] * 2

    loose_pass1 = ClusterConfig(pass1_threshold=0.9, pass2_threshold=0.0,
                                min_faces_per_person=2)
    result = cluster_faces(
        embeddings, photo_ids, np.ones(10, dtype=np.float32),
        np.zeros(10, dtype=bool), loose_pass1,
    )
    assert result.pass1_splits_same_photo > 0, (
        "pass 1 at threshold 0.9 must have put these in one group -- if it did "
        "not, the test is not exercising the repair"
    )
    for person in result.persons:
        photos = [photo_ids[i] for i in person.face_indices]
        assert len(photos) == len(set(photos)), (
            "a person holds two faces from one photograph -- impossible"
        )


def test_pass1_repair_only_splits_never_drops():
    """The repair is allowed to fragment a group. It is not allowed to lose a
    face -- a dropped face is a photo a guest never finds."""
    base_a, base_b = _nearby_bases(2, separation=0.25, seed=41)
    embeddings = np.vstack([
        _faces_around(base_a, 6, spread=0.05, seed=411),
        _faces_around(base_b, 6, spread=0.05, seed=412),
    ])
    photo_ids = [f"shot{i}" for i in range(6)] * 2
    result = cluster_faces(
        embeddings, photo_ids, np.ones(12, dtype=np.float32), np.zeros(12, dtype=bool),
        ClusterConfig(pass1_threshold=0.9, pass2_threshold=0.0, min_faces_per_person=1),
    )
    assigned = sorted(i for p in result.persons for i in p.face_indices)
    assert assigned == list(range(12)), "the pass-1 repair lost faces"


def test_quality_weighting_pulls_the_average_towards_the_good_faces():
    """A pile's average is what every merge and every guest selfie is compared
    against. A blurry back-row crop must not steer it as hard as a sharp
    portrait."""
    rng = np.random.default_rng(77)
    base = _unit(rng.normal(size=DIM))
    sharp = _faces_around(base, 3, spread=0.05, seed=771)     # tight on the person
    blurry = _faces_around(base, 3, spread=0.60, seed=772)    # scattered
    embeddings = np.vstack([sharp, blurry])
    photo_ids = [f"p{i}" for i in range(6)]
    quality = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1], dtype=np.float32)

    weighted = cluster_faces(
        embeddings, photo_ids, quality, np.zeros(6, dtype=bool),
        ClusterConfig(pass1_threshold=0.9, pass2_threshold=0.0, min_faces_per_person=1),
    ).persons[0].centroid
    flat = cluster_faces(
        embeddings, photo_ids, np.ones(6, dtype=np.float32), np.zeros(6, dtype=bool),
        ClusterConfig(pass1_threshold=0.9, pass2_threshold=0.0, min_faces_per_person=1),
    ).persons[0].centroid

    sharp_centre = _unit(sharp.mean(axis=0))
    assert (1 - weighted @ sharp_centre) < (1 - flat @ sharp_centre), (
        "weighting by quality did not move the average towards the sharp faces"
    )
    assert np.linalg.norm(weighted) == pytest.approx(1.0, abs=1e-5)


def test_small_face_threshold_is_measured_not_assumed():
    """0.38 was a guess. Measured on real folders it is far too strict on one
    and too loose on another, so it is derived per folder from faces known to
    be the wrong person."""
    rng = np.random.default_rng(88)
    base = _unit(rng.normal(size=DIM))
    big = _faces_around(base, 40, spread=0.30, seed=881)
    small = _faces_around(base, 20, spread=0.45, seed=882)

    # Every small face shares a photo with a DIFFERENT person, which is what
    # gives the measurement its negatives.
    other = _person_vectors(91, 40)
    embeddings = np.vstack([big, other, small])
    photo_ids = (
        [f"big{i}" for i in range(40)]
        + [f"shared{i}" for i in range(40)]
        + [f"shared{i}" for i in range(20)]
    )
    second_class = np.array([False] * 80 + [True] * 20)
    quality = np.ones(100, dtype=np.float32)

    result = cluster_faces(embeddings, photo_ids, quality, second_class, ClusterConfig())
    assert result.second_class_negatives_measured > 0
    assert result.second_class_threshold_basis in ("measured", "default")
    assert 0.0 < result.second_class_threshold_used <= 0.55


def test_small_face_auto_threshold_can_be_pinned(monkeypatch):
    """The measurement must be defeatable, or a bad folder cannot be debugged."""
    import importlib

    monkeypatch.setenv("WFF_SECOND_CLASS_AUTO", "0")
    import wff.cluster.two_pass as tp

    importlib.reload(tp)
    try:
        assert tp._SECOND_CLASS_AUTO is False
    finally:
        monkeypatch.delenv("WFF_SECOND_CLASS_AUTO")
        importlib.reload(tp)
    assert tp._SECOND_CLASS_AUTO is True


def test_small_groups_become_leftovers_not_people():
    embeddings = np.vstack([_person_vectors(8, 10), _person_vectors(9, 2)])
    photo_ids = [f"p{i}" for i in range(12)]
    result = cluster_faces(
        embeddings, photo_ids, np.ones(12, dtype=np.float32), np.zeros(12, dtype=bool),
        ClusterConfig(min_faces_per_person=3),
    )
    assert len(result.persons) == 1
    assert result.leftover_face_count == 2


def test_leftovers_are_kept_never_dropped():
    """A cousin in 2 photos gets person_id NULL, but must still be in the
    labels array -- tier-2 search reaches them individually. Dropping them
    silently loses that guest."""
    embeddings = np.vstack([_person_vectors(8, 10), _person_vectors(9, 2)])
    result = cluster_faces(
        embeddings, [f"p{i}" for i in range(12)], np.ones(12, dtype=np.float32),
        np.zeros(12, dtype=bool), ClusterConfig(min_faces_per_person=3),
    )
    assert len(result.labels) == 12
    assert (result.labels < 0).sum() == 2


def test_clustering_is_repeatable():
    """Change one number, re-run, and any difference must be caused by YOUR
    change and nothing else. Without this, tuning is guesswork."""
    embeddings = np.vstack([_person_vectors(i, 6) for i in range(4)])
    photo_ids = [f"p{i}" for i in range(24)]
    runs = [
        cluster_faces(embeddings, photo_ids, np.ones(24, dtype=np.float32),
                      np.zeros(24, dtype=bool), ClusterConfig()).labels
        for _ in range(3)
    ]
    assert np.array_equal(runs[0], runs[1])
    assert np.array_equal(runs[1], runs[2])


def test_centroids_are_unit_length():
    """Tier-1 search compares a selfie against these. If they are not unit
    length, cosine distance is wrong and every threshold means something else.
    """
    embeddings = np.vstack([_person_vectors(i, 6) for i in range(3)])
    result = cluster_faces(
        embeddings, [f"p{i}" for i in range(18)], np.ones(18, dtype=np.float32),
        np.zeros(18, dtype=bool), ClusterConfig(),
    )
    for person in result.persons:
        assert np.linalg.norm(person.centroid) == pytest.approx(1.0, abs=1e-5)


def test_cover_face_is_the_highest_quality_one():
    embeddings = _person_vectors(3, 6)
    quality = np.array([0.1, 0.2, 0.9, 0.3, 0.4, 0.5], dtype=np.float32)
    result = cluster_faces(
        embeddings, [f"p{i}" for i in range(6)], quality, np.zeros(6, dtype=bool),
        ClusterConfig(),
    )
    assert result.persons[0].cover_face_index == 2


def test_empty_input():
    result = cluster_faces(
        np.zeros((0, 512), dtype=np.float32), [], np.zeros(0), np.zeros(0, dtype=bool),
        ClusterConfig(),
    )
    assert result.persons == []


# -- the accuracy meter itself ------------------------------------------


def test_evaluate_perfect_clustering():
    labels = np.array([0, 0, 0, 1, 1, 1])
    truth = ["a", "a", "a", "b", "b", "b"]
    ev = evaluate(labels, truth)
    assert ev.pair_precision == 1.0
    assert ev.pair_recall == 1.0
    assert ev.merged_people == []


def test_evaluate_flags_a_merge_as_contamination():
    labels = np.array([0, 0, 0, 0])       # one cluster
    truth = ["a", "a", "b", "b"]          # but two real people
    ev = evaluate(labels, truth)
    assert len(ev.merged_people) == 1
    assert ev.pair_precision < 1.0
    assert ev.pair_recall == 1.0          # nothing was missed, just over-joined


def test_evaluate_reports_a_split_without_calling_it_contamination():
    labels = np.array([0, 0, 1, 1])       # one person split in two
    truth = ["a", "a", "a", "a"]
    ev = evaluate(labels, truth)
    assert ev.merged_people == []         # a split is not a contamination
    assert ev.pair_precision == 1.0       # every pair we claimed is correct
    assert ev.pair_recall < 1.0           # but we missed some
    assert ev.largest_split == 2


def test_evaluate_counts_leftovers_against_recall():
    labels = np.array([0, 0, -1, -1])
    truth = ["a", "a", "a", "a"]
    ev = evaluate(labels, truth)
    assert ev.faces_unassigned == 2
    assert ev.pair_precision == 1.0
    assert ev.pair_recall < 1.0


def test_sanity_alarm_when_too_few_people():
    from wff.cluster.two_pass import ClusterResult

    result = ClusterResult(labels=np.zeros(10, dtype=np.int64), persons=[])
    report = sanity_check(result, photo_count=4000, config=ClusterConfig())
    assert report.alarms, "0 people for 4000 photos must raise an alarm"
    assert "LOOSE" in report.alarms[0]
