"""Tests for the feedback engine.

Aimed at what fails SILENTLY. A feedback system that quietly loses a judgement,
or quietly scores against the wrong pile after a re-cluster, is worse than no
feedback system: it produces confident numbers from nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from wff.storage.local import LocalObjectStore
from wff.web import feedback as fb


@pytest.fixture()
def store(tmp_path):
    return LocalObjectStore(str(tmp_path / "store"))


@pytest.fixture()
def feedback(store):
    return fb.FeedbackStore(store, "r2://wff/events/ev_test")


def test_append_and_read_roundtrip(feedback):
    feedback.append(fb.Judgement(kind=fb.PERSON_OK, reviewer="adarsh", same_group=["a_00", "b_00"]))
    feedback.append(fb.Judgement(kind=fb.PAIR, reviewer="devesh", different_pairs=[["a_00", "c_00"]]))
    records = feedback.read_all()
    assert [r.reviewer for r in records] == ["adarsh", "devesh"]
    assert records[0].record_id and records[0].ts


def test_retraction_removes_a_judgement_without_deleting_history(feedback):
    saved = feedback.append(
        fb.Judgement(kind=fb.PERSON_OK, reviewer="adarsh", same_group=["a_00", "b_00"])
    )
    feedback.retract(saved.record_id, "adarsh")
    assert feedback.live() == []
    # The original line is still on disk -- an audit trail, not an edit.
    assert len(feedback.read_all()) == 2


def test_a_torn_last_line_does_not_lose_earlier_judgements(feedback, store):
    feedback.append(fb.Judgement(kind=fb.PERSON_OK, reviewer="a", same_group=["a_00", "b_00"]))
    with store.open_append(feedback.uri) as fh:
        fh.write(b'{"kind": "pair", "reviewer": "b", "same_gr')  # killed mid-write
    records = feedback.read_all()
    assert len(records) == 1
    assert records[0].reviewer == "a"


# -- judgements -> ground truth ---------------------------------------------


def test_same_group_becomes_one_label():
    truth = fb.derive_truth(
        [fb.Judgement(kind=fb.PERSON_OK, reviewer="a", same_group=["x_00", "y_00", "z_00"])]
    )
    labels = {truth.label_of(f) for f in ("x_00", "y_00", "z_00")}
    assert len(labels) == 1 and None not in labels
    assert truth.groups == 1


def test_transitive_merge_across_two_judgements():
    """A said pile 1 is one person, B said pile 1 and pile 2 are the same.

    The two claims have to fuse into one person, or a later split between them
    would not be counted as a mistake.
    """
    truth = fb.derive_truth(
        [
            fb.Judgement(kind=fb.PERSON_OK, reviewer="a", same_group=["x_00", "y_00"]),
            fb.Judgement(kind=fb.SAME_PERSON, reviewer="b", same_group=["y_00", "z_00"]),
        ]
    )
    assert truth.label_of("x_00") == truth.label_of("z_00")


def test_a_lone_face_carries_no_label():
    """One face on its own says nothing about any pair, so it must not be
    scored as if it did."""
    truth = fb.derive_truth(
        [fb.Judgement(kind=fb.PAIR, reviewer="a", same_group=["only_00"])]
    )
    assert truth.labels == {}
    assert truth.groups == 0


def test_contradiction_is_reported_not_silently_resolved():
    truth = fb.derive_truth(
        [
            fb.Judgement(kind=fb.PERSON_OK, reviewer="a", same_group=["x_00", "y_00"]),
            fb.Judgement(kind=fb.PAIR, reviewer="b", different_pairs=[["x_00", "y_00"]]),
        ]
    )
    assert truth.contradictions == [("x_00", "y_00")]


def test_not_a_face_is_excluded_from_labels():
    truth = fb.derive_truth(
        [
            fb.Judgement(kind=fb.PERSON_OK, reviewer="a", same_group=["x_00", "y_00"]),
            fb.Judgement(kind=fb.NOT_A_FACE, reviewer="a", excluded=["y_00"], subject="y_00"),
        ]
    )
    assert "y_00" in truth.excluded
    assert truth.label_of("y_00") is None


def test_labels_are_deterministic():
    """Two identical judgement sets in a different order must score the same,
    or the number moves for no reason between page loads."""
    records = [
        fb.Judgement(kind=fb.PERSON_OK, reviewer="a", same_group=["b_00", "c_00", "d_00"]),
        fb.Judgement(kind=fb.PERSON_OK, reviewer="a", same_group=["x_00", "y_00"]),
    ]
    first = fb.derive_truth(records)
    second = fb.derive_truth(list(reversed(records)))
    assert first.labels == second.labels


# -- scoring ----------------------------------------------------------------


def test_a_wrong_merge_is_caught():
    truth = fb.derive_truth(
        [fb.Judgement(kind=fb.PAIR, reviewer="a", different_pairs=[["x_00", "y_00"]])]
    )
    score = fb.score_clustering(truth, ["x_00", "y_00"], [0, 0])  # both in pile 0
    assert score.diff_pairs_merged == 1
    assert score.merge_safety == 0.0
    assert "loses trust" in score.headline()


def test_keeping_two_people_apart_scores_clean():
    truth = fb.derive_truth(
        [fb.Judgement(kind=fb.PAIR, reviewer="a", different_pairs=[["x_00", "y_00"]])]
    )
    score = fb.score_clustering(truth, ["x_00", "y_00"], [0, 1])
    assert score.diff_pairs_merged == 0
    assert score.merge_safety == 1.0


def test_a_split_costs_recall_but_is_not_a_merge():
    truth = fb.derive_truth(
        [fb.Judgement(kind=fb.PERSON_OK, reviewer="a", same_group=["x_00", "y_00"])]
    )
    score = fb.score_clustering(truth, ["x_00", "y_00"], [0, 1])
    assert (score.same_pairs_split, score.same_pairs_grouped) == (1, 0)
    assert score.diff_pairs_merged == 0
    assert score.recall == 0.0


def test_two_leftovers_are_not_counted_as_grouped_together():
    """Both faces unassigned means the guest finds neither. Treating -1 == -1
    as agreement would report a perfect score for having grouped nothing."""
    truth = fb.derive_truth(
        [fb.Judgement(kind=fb.PERSON_OK, reviewer="a", same_group=["x_00", "y_00"])]
    )
    score = fb.score_clustering(truth, ["x_00", "y_00"], [-1, -1])
    assert score.same_pairs_grouped == 0
    assert score.same_pairs_split == 1


def test_one_intruder_in_a_big_pile_is_reported_as_one_face_not_240_errors():
    """Found on real data. Flagging a single face inside an approved 241-face
    pile produced "240 pairings ended up in the same pile", because pairs grow
    with the square of the pile. True, and useless to a human -- the reviewer
    needs to know ONE face is misplaced.
    """
    pile = [f"m{i:03d}_00" for i in range(20)]
    intruder = "x_00"
    truth = fb.derive_truth(
        [
            fb.Judgement(
                kind=fb.FACE_WRONG,
                reviewer="a",
                same_group=pile,
                different_pairs=[[intruder, member] for member in pile],
            )
        ]
    )
    # The clustering still has the intruder inside the pile.
    face_ids = pile + [intruder]
    score = fb.score_clustering(truth, face_ids, [0] * len(face_ids))
    assert score.diff_pairs_merged == 20            # the underlying pair maths
    assert score.faces_in_wrong_pile == {intruder}  # what a human is told
    assert "1 face is sitting in a pile it does not belong to" in score.headline()


def test_blame_lands_on_the_intruder_not_the_pile_it_invaded():
    """Whichever face has less company in that pile is the one out of place."""
    left = ["a1_00", "a2_00", "a3_00"]
    truth = fb.derive_truth(
        [
            fb.Judgement(kind=fb.PERSON_OK, reviewer="r", same_group=left),
            fb.Judgement(kind=fb.PAIR, reviewer="r", different_pairs=[["a1_00", "b_00"]]),
        ]
    )
    score = fb.score_clustering(truth, left + ["b_00"], [0, 0, 0, 0])
    assert score.faces_in_wrong_pile == {"b_00"}


def test_a_split_is_reported_in_faces_and_piles():
    group = ["g1_00", "g2_00", "g3_00", "g4_00"]
    truth = fb.derive_truth([fb.Judgement(kind=fb.PERSON_OK, reviewer="r", same_group=group)])
    score = fb.score_clustering(truth, group, [0, 0, 0, 1])
    assert score.faces_split_off == {"g4_00"}
    assert score.people_split == 1
    assert not score.faces_in_wrong_pile
    assert "split across" in score.headline()


def test_one_big_approved_pile_cannot_dictate_the_threshold_on_its_own():
    """Found on real data. With 28,680 same-person pairs from one approved pile
    against 240 different-person pairs, raw counts pushed the suggested
    threshold to 1.00 -- a setting that merges every person in the wedding into
    one. The sweep has to compare RATES so lopsided judging cannot do that.
    """
    big_pile = [f"p{i:03d}_00" for i in range(40)]
    truth = fb.derive_truth(
        [
            fb.Judgement(kind=fb.PERSON_OK, reviewer="r", same_group=big_pile),
            fb.Judgement(kind=fb.PAIR, reviewer="r", different_pairs=[["q1_00", "q2_00"]]),
        ]
    )

    def distance(a: str, b: str):
        if {a, b} == {"q1_00", "q2_00"}:
            return 0.30           # two different people, unusually close
        return 0.65               # same person, unusually far apart

    evidence = fb.threshold_evidence(
        truth, big_pile + ["q1_00", "q2_00"], distance
    )
    assert evidence.overlap
    assert evidence.suggested < 0.30, "must not merge the known-different pair"
    assert evidence.suggested is not None


def test_judgements_survive_a_reclustering_that_renumbers_people():
    """The whole point of storing face ids instead of person ids.

    Same judgement, two clusterings that number the piles differently: the
    verdict must not change.
    """
    truth = fb.derive_truth(
        [fb.Judgement(kind=fb.PERSON_OK, reviewer="a", same_group=["x_00", "y_00"])]
    )
    first = fb.score_clustering(truth, ["x_00", "y_00"], [0, 0])
    second = fb.score_clustering(truth, ["x_00", "y_00"], [7, 7])
    assert first.same_pairs_grouped == second.same_pairs_grouped == 1


def test_faces_missing_from_the_run_are_skipped_not_crashed():
    """A judgement about a face that a tighter quality gate later rejected must
    be ignored, not counted, and must never raise."""
    truth = fb.derive_truth(
        [fb.Judgement(kind=fb.PERSON_OK, reviewer="a", same_group=["x_00", "gone_00"])]
    )
    score = fb.score_clustering(truth, ["x_00"], [0])
    assert score.same_pairs_known == 0


# -- threshold evidence ------------------------------------------------------


def _distance_table(pairs: dict):
    def distance(a: str, b: str):
        return pairs.get(frozenset({a, b}))

    return distance


def test_clean_gap_suggests_a_threshold_between_the_two_groups():
    truth = fb.derive_truth(
        [
            fb.Judgement(kind=fb.PERSON_OK, reviewer="a", same_group=["x_00", "y_00"]),
            fb.Judgement(kind=fb.PAIR, reviewer="a", different_pairs=[["x_00", "z_00"]]),
        ]
    )
    distance = _distance_table(
        {frozenset({"x_00", "y_00"}): 0.30, frozenset({"x_00", "z_00"}): 0.70}
    )
    evidence = fb.threshold_evidence(truth, ["x_00", "y_00", "z_00"], distance)
    assert not evidence.overlap
    assert 0.30 < evidence.suggested < 0.70


def test_overlap_breaks_the_tie_toward_the_safer_tighter_value():
    """When same-person and different-person distances overlap there is no
    clean answer. A wrong merge costs more than a wrong split, so the chosen
    cut must sit below the different-person distance, not above it."""
    truth = fb.derive_truth(
        [
            fb.Judgement(kind=fb.PERSON_OK, reviewer="a", same_group=["x_00", "y_00"]),
            fb.Judgement(kind=fb.PAIR, reviewer="a", different_pairs=[["p_00", "q_00"]]),
        ]
    )
    distance = _distance_table(
        {frozenset({"x_00", "y_00"}): 0.60, frozenset({"p_00", "q_00"}): 0.50}
    )
    evidence = fb.threshold_evidence(
        truth, ["x_00", "y_00", "p_00", "q_00"], distance
    )
    assert evidence.overlap
    assert evidence.suggested < 0.50
    assert "No single line separates them" in evidence.summary()


def test_no_judgements_means_no_pretend_measurement():
    evidence = fb.threshold_evidence(fb.derive_truth([]), [], lambda a, b: 0.5)
    assert not evidence.has_data
    assert evidence.suggested is None
    assert "guess" in evidence.summary()


def test_cosine_distance_is_what_the_evidence_measures():
    """Guards the wiring, not the maths: if the injected distance function ever
    stopped matching the clustering's own metric, every suggested threshold
    would be quietly wrong."""
    from wff.web.data import EventCache

    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    assert pytest.approx(1.0 - float(np.dot(a, b)) / norm, abs=1e-6) == 1.0
    assert hasattr(EventCache, "distance_fn")
