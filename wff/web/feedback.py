"""Human judgements, stored so they outlive the run that produced them.

THE PROBLEM THIS SOLVES. Until now precision and recall could only be measured
on the cricket folder, because that folder happens to put one person per
directory. On a real wedding nobody knows the right answer, so `docs/
ARCHITECTURE.md`'s two thresholds (0.42 / 0.55) stayed guesses -- see the
12 Aug worklog entry. Every judgement a reviewer makes here is a piece of the
right answer, on real photos.

THE ONE DESIGN RULE. A person_id is not stable: re-run clustering with a
different threshold and "Person 3" is a different pile. So no judgement is ever
stored against a person_id. Everything is stored as a claim about FACE IDS,
which are permanent:

    "these faces are all one person"      -> same_group
    "these two faces are different people" -> different_pairs

"Person 3 is clean" becomes a same_group of its 84 face ids. "This face doesn't
belong" becomes different_pairs against the rest of the pile. Both survive every
future re-clustering, every threshold change, and a model swap (the model
version is stamped on each record so a swap can invalidate them deliberately
rather than silently).

Union-find over the same_group claims turns a pile of judgements into partial
ground-truth labels, which `cluster.metrics.evaluate` already knows how to
score.
"""
from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..storage import ObjectStore, join_uri

# -- the kinds of judgement a reviewer can make ------------------------------
PERSON_OK = "person_ok"        # every face in this pile is the same person
FACE_WRONG = "face_wrong"      # this one face does not belong in this pile
SAME_PERSON = "same_person"    # two piles are actually one person (a split)
FACE_BELONGS = "face_belongs"  # this leftover face belongs to that pile
PAIR = "pair"                  # two faces, same or different (judge mode)
NOT_A_FACE = "not_a_face"      # not a face at all -- a detector error
NOTE = "note"                  # free text, no constraint

KINDS = {PERSON_OK, FACE_WRONG, SAME_PERSON, FACE_BELONGS, PAIR, NOT_A_FACE, NOTE}

# Judgements a reviewer can undo by clicking again.
TOGGLEABLE = {PERSON_OK, FACE_WRONG, NOT_A_FACE}


@dataclass
class Judgement:
    """One line of feedback.jsonl."""

    kind: str
    reviewer: str
    record_id: str = ""
    ts: str = ""
    # Claims about face ids -- the durable part.
    same_group: list[str] = field(default_factory=list)
    different_pairs: list[list[str]] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    # Context, for display and for undo. NOT used in scoring.
    subject: str = ""          # the face or pile the reviewer clicked
    person_label: str = ""     # "Person 3" as it read at the time
    text: str = ""
    retracted: bool = False
    model_version: str = ""

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in self.__dict__.items()}, ensure_ascii=False)

    @staticmethod
    def from_dict(raw: dict) -> "Judgement":
        known = {k: raw.get(k) for k in Judgement.__dataclass_fields__ if k in raw}
        base = Judgement(kind=raw.get("kind", NOTE), reviewer=raw.get("reviewer", ""))
        for key, value in known.items():
            if value is not None:
                setattr(base, key, value)
        return base


class FeedbackStore:
    """Append-only, one file per event. Same journal discipline as Stage 1.

    Append-only matters for two reasons: two browser tabs (Adarsh and Devesh)
    can write at the same time without clobbering each other, and a retraction
    is a new line rather than an edit, so the history of who thought what is
    never lost.
    """

    def __init__(self, store: ObjectStore, event_root: str) -> None:
        self._store = store
        self._uri = join_uri(event_root, "feedback.jsonl")

    @property
    def uri(self) -> str:
        return self._uri

    def append(self, judgement: Judgement) -> Judgement:
        judgement.record_id = judgement.record_id or uuid.uuid4().hex[:12]
        judgement.ts = judgement.ts or datetime.now(timezone.utc).isoformat()
        with self._store.open_append(self._uri) as fh:
            fh.write((judgement.to_json() + "\n").encode("utf-8"))
        return judgement

    def read_all(self) -> list[Judgement]:
        if not self._store.exists(self._uri):
            return []
        records: list[Judgement] = []
        for line in self._store.get_text(self._uri).splitlines():
            if not line.strip():
                continue
            try:
                records.append(Judgement.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue  # torn last line from a crash, or a hand-edit
        return records

    def live(self) -> list[Judgement]:
        """Everything not cancelled by a later retraction line.

        A retraction is itself a line, carrying `retracted=True` and the
        `record_id` it cancels in `subject`. Nothing is ever rewritten, so the
        history of who thought what, and when they changed their mind, survives.
        """
        records = self.read_all()
        cancelled = {r.subject for r in records if r.retracted}
        return [r for r in records if not r.retracted and r.record_id not in cancelled]

    def retract(self, record_id: str, reviewer: str) -> Judgement:
        return self.append(
            Judgement(
                kind="retraction",
                reviewer=reviewer,
                subject=record_id,
                retracted=True,
            )
        )

    def raw_text(self) -> str:
        return self._store.get_text(self._uri) if self._store.exists(self._uri) else ""


# ---------------------------------------------------------------------------
# Judgements -> partial ground truth
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class TruthSet:
    """What the reviewers have collectively asserted, reduced to labels."""

    labels: dict[str, str] = field(default_factory=dict)          # face_id -> "T1"
    must_not_link: set[frozenset[str]] = field(default_factory=set)  # face pairs
    excluded: set[str] = field(default_factory=set)               # not-a-face
    contradictions: list[tuple[str, str]] = field(default_factory=list)
    judged_faces: int = 0
    groups: int = 0

    def label_of(self, face_id: str) -> str | None:
        return self.labels.get(face_id)


def derive_truth(judgements: list[Judgement]) -> TruthSet:
    """Fold every claim into labels plus a set of must-not-link pairs.

    A contradiction (two faces asserted both same and different) is reported,
    never silently resolved -- it means two reviewers disagree, or one of them
    mis-clicked, and either way a human needs to look.
    """
    union = _UnionFind()
    not_link: set[frozenset[str]] = set()
    excluded: set[str] = set()

    for record in judgements:
        if record.kind == NOT_A_FACE:
            excluded.update(record.excluded or [record.subject])
        for face_id in record.same_group:
            union.add(face_id)
        for a, b in zip(record.same_group, record.same_group[1:]):
            union.union(a, b)
        for pair in record.different_pairs:
            if len(pair) == 2 and pair[0] != pair[1]:
                union.add(pair[0])
                union.add(pair[1])
                not_link.add(frozenset(pair))

    # Name the components deterministically: by size, then by smallest member,
    # so the same judgements always produce the same labels.
    members: dict[str, list[str]] = defaultdict(list)
    for face_id in union.parent:
        if face_id in excluded:
            continue
        members[union.find(face_id)].append(face_id)

    ordered = sorted(members.values(), key=lambda ms: (-len(ms), min(ms)))
    labels: dict[str, str] = {}
    groups = 0
    for group in ordered:
        if len(group) < 2:
            continue  # a lone face carries no pairwise information
        groups += 1
        for face_id in group:
            labels[face_id] = f"T{groups}"

    contradictions = [
        tuple(sorted(pair))  # type: ignore[misc]
        for pair in not_link
        if len({union.find(f) for f in pair}) == 1
    ]

    return TruthSet(
        labels=labels,
        must_not_link=not_link,
        excluded=excluded,
        contradictions=sorted(contradictions),
        judged_faces=len(labels),
        groups=groups,
    )


# ---------------------------------------------------------------------------
# Scoring a clustering against what the humans said
# ---------------------------------------------------------------------------


@dataclass
class FeedbackScore:
    """Plain-language accuracy, on real photos, from real judgements.

    Deliberately reported as four counts rather than one number. A single F1
    hides the asymmetry the whole system is built around: a wrong merge loses a
    guest's trust, a wrong split mildly disappoints them.
    """

    judged_faces: int = 0
    judged_groups: int = 0
    same_pairs_known: int = 0
    same_pairs_grouped: int = 0        # we agreed: correctly together
    same_pairs_split: int = 0          # we split what a human called one person
    diff_pairs_known: int = 0
    diff_pairs_kept_apart: int = 0
    diff_pairs_merged: int = 0         # THE CATASTROPHE: strangers in one pile
    merged_examples: list[tuple[str, str]] = field(default_factory=list)
    contradictions: list[tuple[str, str]] = field(default_factory=list)

    # Counted in FACES, not pairs. Pairs grow with the square of a pile, so one
    # misplaced face in a 241-face pile reads as "240 errors" -- true, and
    # useless. A reviewer needs to know how many faces are in the wrong place.
    faces_in_wrong_pile: set[str] = field(default_factory=set)
    faces_split_off: set[str] = field(default_factory=set)
    people_split: int = 0

    @property
    def recall(self) -> float:
        return (
            self.same_pairs_grouped / self.same_pairs_known
            if self.same_pairs_known
            else 0.0
        )

    @property
    def merge_safety(self) -> float:
        """Share of known different-person pairs we correctly kept apart."""
        return (
            self.diff_pairs_kept_apart / self.diff_pairs_known
            if self.diff_pairs_known
            else 1.0
        )

    @property
    def has_data(self) -> bool:
        return bool(self.same_pairs_known or self.diff_pairs_known)

    def headline(self) -> str:
        """One sentence, in words, about what the judgements say.

        Ordered by what actually matters: a wrong merge first (a guest sees a
        stranger), then a split (a guest misses photos), then the clean case.
        """
        if not self.has_data:
            return "No judgements yet."
        if self.faces_in_wrong_pile:
            count = len(self.faces_in_wrong_pile)
            return (
                f"{count} face{'s' if count > 1 else ''} "
                f"{'are' if count > 1 else 'is'} sitting in a pile "
                f"{'they do' if count > 1 else 'it does'} not belong to. A guest "
                "matching into that pile would be shown a stranger's photos -- "
                "this is the failure that loses trust."
            )
        if self.faces_split_off:
            return (
                "Nobody has been wrongly merged, which is the important half. "
                f"But {len(self.faces_split_off)} face(s) you said belong "
                f"together have been split across {self.people_split or 2} piles, "
                "so those photos would be divided between them."
            )
        return (
            "Everything checked so far is right: nobody merged who shouldn't be, "
            "and nothing split that you said belongs together."
        )


def score_clustering(
    truth: TruthSet,
    face_ids: list[str],
    labels: list[int],
) -> FeedbackScore:
    """Compare a clustering to the judgements, over judged pairs only.

    Only pairs a human actually ruled on are counted. That keeps the number
    honest: it never credits us for the 99% of pairs nobody looked at.
    """
    person_of = {
        face_id: int(label) for face_id, label in zip(face_ids, labels)
    }
    score = FeedbackScore(
        judged_faces=truth.judged_faces,
        judged_groups=truth.groups,
        contradictions=truth.contradictions,
    )

    by_truth: dict[str, list[str]] = defaultdict(list)
    for face_id, truth_label in truth.labels.items():
        if face_id in person_of:
            by_truth[truth_label].append(face_id)

    for group in by_truth.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                score.same_pairs_known += 1
                a, b = person_of[group[i]], person_of[group[j]]
                # A leftover (-1) is not "grouped with" anything, not even
                # another leftover: two unassigned faces are not one person.
                if a >= 0 and a == b:
                    score.same_pairs_grouped += 1
                else:
                    score.same_pairs_split += 1

        # Face-level view of the same finding: which faces left the majority.
        piles = [person_of[face_id] for face_id in group]
        distinct = {p for p in piles if p >= 0}
        if len(distinct) > 1 or -1 in piles:
            score.people_split += 1
            majority = max(distinct, key=piles.count) if distinct else None
            score.faces_split_off.update(
                face_id
                for face_id, pile in zip(group, piles)
                if pile != majority
            )

    # (asserted identity, pile) -> how many faces back that identity up in that
    # pile. A face with no asserted identity has no company by definition, which
    # makes it the intruder -- exactly right for a face just flagged as wrong.
    support: dict[tuple[str | None, int], int] = defaultdict(int)
    for face_id, truth_label in truth.labels.items():
        if face_id in person_of:
            support[(truth_label, person_of[face_id])] += 1

    for pair in truth.must_not_link:
        a_id, b_id = tuple(pair)
        if a_id not in person_of or b_id not in person_of:
            continue
        score.diff_pairs_known += 1
        a, b = person_of[a_id], person_of[b_id]
        if a >= 0 and a == b:
            score.diff_pairs_merged += 1
            # Blame the minority side. In a 241-face pile with one intruder, the
            # intruder is the mistake -- not the 240 faces it was compared
            # against. Whichever of the two has less company in that pile is the
            # one sitting in the wrong place.
            score.faces_in_wrong_pile.add(
                a_id if support[(truth.labels.get(a_id), a)] <=
                        support[(truth.labels.get(b_id), b)]
                else b_id
            )
            if len(score.merged_examples) < 20:
                score.merged_examples.append((a_id, b_id))
        else:
            score.diff_pairs_kept_apart += 1

    return score


# ---------------------------------------------------------------------------
# What the judgements say the thresholds should be
# ---------------------------------------------------------------------------


@dataclass
class ThresholdEvidence:
    """The two numbers the whole product hangs on, measured instead of guessed."""

    same_distances: list[float] = field(default_factory=list)
    diff_distances: list[float] = field(default_factory=list)
    suggested: float | None = None
    ceiling: float | None = None       # closest different-person pair seen
    floor: float | None = None         # furthest same-person pair seen
    overlap: bool = False

    @property
    def has_data(self) -> bool:
        return bool(self.same_distances and self.diff_distances)

    def summary(self) -> str:
        """Distances here are measured FACE to FACE.

        Worth stating, because the merge setting compares pile averages instead,
        and two faces of one person are always further apart than the averages
        of two piles of them. So this is guidance on where the boundary between
        people sits, not a number to paste straight into the merge knob.
        """
        if not self.has_data:
            return (
                "Judge some face pairs and this becomes a measurement instead "
                "of a guess."
            )
        if self.overlap:
            return (
                f"The hardest same-person pair of faces sits {self.floor:.3f} "
                f"apart, FURTHER than the closest different-person pair "
                f"({self.ceiling:.3f}). No single line separates them -- "
                f"{self.suggested:.2f} is the least-bad one, and it errs toward "
                "splitting rather than merging."
            )
        return (
            f"Clean gap: same-person faces reach {self.floor:.3f} apart, "
            f"different people start at {self.ceiling:.3f}. Anywhere in between "
            f"works; {self.suggested:.2f} sits in the middle."
        )


def threshold_evidence(
    truth: TruthSet,
    face_ids: list[str],
    distance_of,
) -> ThresholdEvidence:
    """Distances for every judged pair, and the best line between them.

    `distance_of(face_a, face_b) -> float | None` is injected so this module
    never touches embeddings or numpy.
    """
    index = {face_id: position for position, face_id in enumerate(face_ids)}
    evidence = ThresholdEvidence()

    by_truth: dict[str, list[str]] = defaultdict(list)
    for face_id, truth_label in truth.labels.items():
        if face_id in index:
            by_truth[truth_label].append(face_id)

    for group in by_truth.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                distance = distance_of(group[i], group[j])
                if distance is not None:
                    evidence.same_distances.append(float(distance))

    for pair in truth.must_not_link:
        a_id, b_id = tuple(pair)
        if a_id in index and b_id in index:
            distance = distance_of(a_id, b_id)
            if distance is not None:
                evidence.diff_distances.append(float(distance))

    if not evidence.has_data:
        return evidence

    evidence.floor = max(evidence.same_distances)
    evidence.ceiling = min(evidence.diff_distances)
    evidence.overlap = evidence.floor >= evidence.ceiling

    if not evidence.overlap:
        evidence.suggested = round((evidence.floor + evidence.ceiling) / 2, 3)
        return evidence

    # Overlapping. Sweep candidate cut points and pick the one with the fewest
    # mistakes, breaking ties toward the TIGHTER value -- splitting is the mild
    # failure, merging is the catastrophe.
    #
    # The candidates are the MIDPOINTS between observed distances, plus one
    # below the smallest. Using the observed distances themselves would leave
    # "just tight enough to exclude the closest wrong pair" unreachable, since
    # a cut equal to that pair's distance still merges it -- so the sweep would
    # be structurally unable to pick a safe threshold.
    observed = sorted(set(evidence.same_distances + evidence.diff_distances))
    candidates = [max(0.0, observed[0] - 0.05)]
    candidates += [
        round((low + high) / 2, 4) for low, high in zip(observed, observed[1:])
    ]
    candidates.append(min(1.0, observed[-1] + 0.05))

    best_cut, best_cost = candidates[0], None
    for cut in candidates:
        # RATES, not counts. One approved 241-face pile contributes 28,680
        # same-person pairs while a handful of judged different-person pairs
        # contributes a few dozen, so raw counts let a single big pile decide
        # the threshold on its own -- it pushed the suggestion to 1.00 on real
        # data, which would merge everybody. Rates make the two sides comparable
        # however lopsided the judging happened to be.
        merge_rate = sum(1 for d in evidence.diff_distances if d <= cut) / len(
            evidence.diff_distances
        )
        split_rate = sum(1 for d in evidence.same_distances if d > cut) / len(
            evidence.same_distances
        )
        # A merge counts triple. That ratio is a judgement call, not a
        # measurement, and it encodes the safety rule.
        cost = merge_rate * 3 + split_rate
        if best_cost is None or cost < best_cost - 1e-9:
            best_cut, best_cost = cut, cost
    evidence.suggested = round(best_cut, 3)
    return evidence
