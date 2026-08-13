"""Is the clustering any good? Two answers: alarms, and -- when we have ground
truth -- numbers.

The alarms work on any event, including a real wedding where nobody knows the
right answer. The scored evaluation only works when each face carries a true
label, which the cricket test set gives us for free (the folder name IS the
person).

Note the asymmetry throughout: a MERGE is reported as an error, a SPLIT as a
nuisance. That is not sloppiness, it is the safety rule.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..config import ClusterConfig
from .two_pass import ClusterResult


@dataclass
class SanityReport:
    people_found: int
    photos: int
    people_per_1000_photos: float
    alarms: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def sanity_check(
    result: ClusterResult, photo_count: int, config: ClusterConfig
) -> SanityReport:
    """The checks that work without knowing the right answer."""
    people = len(result.persons)
    rate = (people / photo_count * 1000.0) if photo_count else 0.0
    report = SanityReport(
        people_found=people, photos=photo_count, people_per_1000_photos=rate
    )

    if photo_count >= 200:
        if rate < config.alarm_min_people_per_1000_photos:
            report.alarms.append(
                f"ALARM: only {people} people for {photo_count} photos "
                f"({rate:.0f} per 1000). The threshold is probably too LOOSE and "
                "people are being merged into each other. This is the dangerous "
                "direction -- guests see strangers. Lower WFF_PASS2_THRESHOLD."
            )
        if rate > config.alarm_max_people_per_1000_photos:
            report.notes.append(
                f"{people} people for {photo_count} photos ({rate:.0f} per 1000) "
                "is high -- everyone is probably fragmented. Annoying, not "
                "dangerous. Raise WFF_PASS2_THRESHOLD, or use the photographer "
                "review screen to merge."
            )

    if result.blocked_merges_same_photo:
        report.notes.append(
            f"The same-photo rule blocked {result.blocked_merges_same_photo} "
            "merges that distance alone would have made. Each one was a "
            "prevented catastrophe."
        )
    if result.leftover_face_count:
        report.notes.append(
            f"{result.leftover_face_count} leftover faces kept with no person "
            "(person_id NULL). These are NOT lost -- tier-2 selfie search "
            "reaches them individually."
        )
    report.notes.extend(result.warnings)
    return report


@dataclass
class Contamination:
    person_id: int
    label: str
    face_count: int
    true_labels: dict[str, int]

    @property
    def dominant(self) -> str:
        return max(self.true_labels.items(), key=lambda kv: kv[1])[0]

    @property
    def purity(self) -> float:
        total = sum(self.true_labels.values())
        return max(self.true_labels.values()) / total if total else 0.0


@dataclass
class Evaluation:
    """Pairwise scores. 'Same person?' asked of every pair of faces."""

    pair_precision: float
    pair_recall: float
    pair_f1: float
    true_people: int
    predicted_people: int
    faces_scored: int
    faces_unassigned: int

    merged_people: list[Contamination] = field(default_factory=list)
    split_people: dict[str, int] = field(default_factory=dict)
    largest_split: int = 0

    def summary_lines(self) -> list[str]:
        lines = [
            f"pairwise precision {self.pair_precision:.3f}  "
            f"recall {self.pair_recall:.3f}  F1 {self.pair_f1:.3f}",
            f"{self.true_people} real people -> {self.predicted_people} clusters "
            f"({self.faces_unassigned} faces left over)",
        ]
        if self.merged_people:
            lines.append(
                f"!! {len(self.merged_people)} CONTAMINATED clusters "
                "(two real people in one cluster -- this is the failure that "
                "loses trust)"
            )
        else:
            lines.append("no contaminated clusters -- no two people merged")
        if self.largest_split > 1:
            lines.append(
                f"worst split: one person spread across {self.largest_split} "
                "clusters (recoverable in the review screen)"
            )
        return lines


def evaluate(labels_pred: np.ndarray, labels_true: list[str]) -> Evaluation:
    """Score a clustering against known answers.

    Leftovers (person_id -1) are treated as singleton clusters: they cannot
    create a false pair, but they do cost recall. That is the correct
    accounting -- a face we failed to group is a photo the guest does not find.
    """
    labels_pred = np.asarray(labels_pred)
    assert len(labels_pred) == len(labels_true)

    def pair_count(n: int) -> int:
        return n * (n - 1) // 2

    true_groups = Counter(labels_true)
    total_true_pairs = sum(pair_count(n) for n in true_groups.values())

    pred_groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels_pred):
        pred_groups[int(label)].append(index)

    total_pred_pairs = 0
    true_positive_pairs = 0
    contaminated: list[Contamination] = []

    for person_id, members in pred_groups.items():
        if person_id < 0:
            continue  # leftovers are singletons: no pairs claimed
        total_pred_pairs += pair_count(len(members))
        composition = Counter(labels_true[i] for i in members)
        true_positive_pairs += sum(pair_count(n) for n in composition.values())
        if len(composition) > 1:
            contaminated.append(
                Contamination(
                    person_id=person_id,
                    label=f"Person {person_id + 1}",
                    face_count=len(members),
                    true_labels=dict(composition),
                )
            )

    precision = true_positive_pairs / total_pred_pairs if total_pred_pairs else 1.0
    recall = true_positive_pairs / total_true_pairs if total_true_pairs else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # How badly is each real person fragmented?
    splits: dict[str, set[int]] = defaultdict(set)
    for index, label in enumerate(labels_pred):
        if int(label) >= 0:
            splits[labels_true[index]].add(int(label))
    split_counts = {name: len(ids) for name, ids in splits.items()}

    return Evaluation(
        pair_precision=precision,
        pair_recall=recall,
        pair_f1=f1,
        true_people=len(true_groups),
        predicted_people=len([k for k in pred_groups if k >= 0]),
        faces_scored=len(labels_pred),
        faces_unassigned=int((labels_pred < 0).sum()),
        merged_people=sorted(contaminated, key=lambda c: -c.face_count),
        split_people=split_counts,
        largest_split=max(split_counts.values()) if split_counts else 0,
    )
