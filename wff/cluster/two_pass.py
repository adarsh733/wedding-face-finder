"""STAGE 2 -- cheap, re-runs in seconds, unlimited times.

    faces.parquet -> cluster -> people

THE SAFETY RULE, which decides every trade-off in this file:

    Splitting a person is a mild failure. Merging two people is a catastrophe.

    Splitting -> Priya finds 60% of her photos and is mildly disappointed.
    Merging   -> Priya sees photos of a stranger, and stops trusting us.

So: be tight first, then merge deliberately, with evidence.

    PASS 1 -- deliberately tight
      32,136 faces -> ~380 small, very pure groups
      (the bride is split across 6 of them -- fine and expected)

    PASS 2 -- merge, with evidence
      compare the 380 group-averages to each other, and merge two groups when
      they are close AND the evidence agrees:
        - are they ever in the SAME PHOTO? if yes they are DIFFERENT PEOPLE
          -> never merge (two faces in one photo cannot be one person)
      380 -> 147 people
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ..config import ClusterConfig
from .autotune import choose_second_class_threshold

# NOTE: this belongs in ClusterConfig with every other knob, and moves there as
# soon as wff/config.py is free -- it is held by another chat window right now
# (claim C-20260813-1810). Read through getattr(config, ...) at the call site so
# the move is a pure deletion here and nothing else changes.
_SECOND_CLASS_AUTO = os.environ.get("WFF_SECOND_CLASS_AUTO", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# Exact agglomerative clustering materialises an n x n float64 distance matrix.
# At 12,000 faces that is ~1.1 GB; at 32,000 it is ~8 GB and the job dies.
# A real 4,000-photo wedding has ~32,000 faces, so this WILL be hit in
# production -- see _pass1_blocked for how we stay exact-ish above the limit.
EXACT_AGGLOMERATIVE_MAX_FACES = 12_000


@dataclass
class Person:
    person_id: int
    face_indices: list[int]
    photo_ids: set[str]
    centroid: np.ndarray
    cover_face_index: int
    label: str = ""

    @property
    def face_count(self) -> int:
        return len(self.face_indices)

    @property
    def photo_count(self) -> int:
        return len(self.photo_ids)


@dataclass
class ClusterResult:
    """labels[i] is the person_id of face i, or -1 for a leftover."""

    labels: np.ndarray
    persons: list[Person]
    pass1_group_count: int = 0
    pass2_merge_count: int = 0
    blocked_merges_same_photo: int = 0
    pass1_splits_same_photo: int = 0
    leftover_face_count: int = 0
    second_class_assigned: int = 0
    second_class_left_over: int = 0
    second_class_threshold_used: float = 0.0
    second_class_threshold_basis: str = "default"
    second_class_negatives_measured: int = 0
    used_blocking: bool = False
    warnings: list[str] = field(default_factory=list)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> bool:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        return True


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms < 1e-10, 1.0, norms)


# A pile's average is only as good as the faces in it. A sharp 400px portrait
# and a soft 55px face in the back row both used to count exactly the same,
# which drags the average towards the blurry end -- and the average is what
# every merge decision and (later) every guest's selfie is compared against.
#
# Weighting is deliberately gentle: quality_score runs 0-1, so 0.5 + q spans
# 0.5 to 1.5 and the best face in a pile can count at most three times the
# worst. Enough to stop the blurry faces steering, not enough to let one
# flattering portrait define a person.
QUALITY_WEIGHT_BASE = 0.5


def _centroid(vectors: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """The AVERAGE of a person's faces, back on the unit sphere.

    Averaging 80 photos of Priya cancels the noise from any one bad angle,
    which is why matching a selfie against a group average beats matching it
    against any single face. Weighted by face quality when it is known, so a
    blurry back-row crop does not pull the average as hard as a sharp portrait.
    """
    if weights is None:
        mean = vectors.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        total = w.sum()
        # All-zero weights would divide by zero and silently return NaN, which
        # then poisons every distance it touches. Fall back to a plain mean.
        mean = (vectors * w).sum(axis=0) / total if total > 1e-10 else vectors.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-10 else mean


def _quality_weights(quality_scores: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return QUALITY_WEIGHT_BASE + np.asarray(quality_scores, dtype=np.float32)[indices]


def _agglomerative(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    if len(embeddings) == 1:
        return np.zeros(1, dtype=np.int64)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(embeddings)


def _pass1_blocked(
    embeddings: np.ndarray, threshold: float, block_target: int, seed: int = 0
) -> np.ndarray:
    """Pass 1 for face counts that exact agglomerative cannot hold in memory.

    Pre-partition with k-means, then run EXACT agglomerative inside each block.
    k-means is only ever allowed to SPLIT, never to merge: two faces landing in
    different blocks simply become different groups, and pass 2 -- which
    compares every group to every other group and applies the same-photo rule --
    is what puts them back together.

    So the failure mode of blocking is extra fragments, which is the safe
    direction. Fixed seed, so re-runs are still repeatable.
    """
    from sklearn.cluster import MiniBatchKMeans

    n_blocks = max(2, int(np.ceil(len(embeddings) / block_target)))
    kmeans = MiniBatchKMeans(
        n_clusters=n_blocks, random_state=seed, n_init=3, batch_size=4096
    )
    blocks = kmeans.fit_predict(embeddings)

    labels = np.full(len(embeddings), -1, dtype=np.int64)
    next_label = 0
    for block_id in range(n_blocks):
        members = np.where(blocks == block_id)[0]
        if members.size == 0:
            continue
        if members.size == 1:
            labels[members[0]] = next_label
            next_label += 1
            continue
        sub_labels = _agglomerative(embeddings[members], threshold)
        for offset, member in zip(sub_labels, members):
            labels[member] = next_label + int(offset)
        next_label += int(sub_labels.max()) + 1
    return labels


def _split_same_photo_groups(
    groups: list[list[int]],
    embeddings: np.ndarray,
    photo_ids: list[str],
    quality_scores: np.ndarray,
    threshold: float,
) -> tuple[list[list[int]], int]:
    """Enforce the same-photo rule INSIDE pass 1, where it was never applied.

    Pass 2 refuses to merge two groups that share a photo, but pass 1 was a
    plain distance clustering with no such constraint -- so a group could be
    born already holding two faces from one photograph, which is impossible,
    and pass 2 would then carry that contradiction into every merge it made.

    Measured 13 Aug: rare but real -- 3 contaminated groups on ev-2025-10-16
    and 2 on ev_manali01, and one survived into a final person on each.

    The repair only ever SPLITS, which is the safe direction. Faces are taken
    best-quality first and each joins the nearest sub-group that does not
    already hold its photograph; if none qualifies it starts its own. Pass 2
    then reassembles whatever was split apart too eagerly.
    """
    repaired: list[list[int]] = []
    splits = 0

    for members in groups:
        photos = [photo_ids[i] for i in members]
        if len(photos) == len(set(photos)):
            repaired.append(members)
            continue

        splits += 1
        # Best face first, so the strongest evidence founds each sub-group.
        # Stable order on ties keeps the whole run repeatable.
        order = sorted(members, key=lambda i: (-float(quality_scores[i]), i))
        buckets: list[list[int]] = []
        bucket_photos: list[set[str]] = []

        for face in order:
            photo = photo_ids[face]
            best_bucket = -1
            best_distance = threshold
            for index, taken in enumerate(bucket_photos):
                if photo in taken:
                    continue
                centre = _centroid(
                    embeddings[np.array(buckets[index])],
                    _quality_weights(quality_scores, np.array(buckets[index])),
                )
                distance = 1.0 - float(embeddings[face] @ centre)
                if distance < best_distance:
                    best_distance = distance
                    best_bucket = index
            if best_bucket < 0:
                buckets.append([face])
                bucket_photos.append({photo})
            else:
                buckets[best_bucket].append(face)
                bucket_photos[best_bucket].add(photo)

        repaired.extend(sorted(bucket) for bucket in buckets)

    return repaired, splits


def cluster_faces(
    embeddings: np.ndarray,
    photo_ids: list[str],
    quality_scores: np.ndarray,
    second_class: np.ndarray,
    config: ClusterConfig,
) -> ClusterResult:
    """Two-pass clustering over one event's accepted faces.

    embeddings      (N, 512) unit-length
    photo_ids       length N, which photo each face came from
    quality_scores  length N, for picking cover faces
    second_class    length N bool -- 50-80px faces: may JOIN a person,
                    may never START one
    """
    total = len(embeddings)
    result = ClusterResult(labels=np.full(total, -1, dtype=np.int64), persons=[])
    if total == 0:
        return result

    embeddings = _normalise(np.asarray(embeddings, dtype=np.float32))
    photo_ids = list(photo_ids)

    # Only first-class faces get to define people.
    first_class_idx = np.where(~second_class)[0]
    if first_class_idx.size == 0:
        result.warnings.append(
            "Every accepted face is second-class (50-80px). No person can be "
            "formed from those alone -- they may only join an existing person."
        )
        result.second_class_left_over = total
        result.leftover_face_count = total
        return result

    # ---- PASS 1 -- deliberately tight ----------------------------------
    fc_embeddings = embeddings[first_class_idx]
    if len(fc_embeddings) > EXACT_AGGLOMERATIVE_MAX_FACES:
        result.used_blocking = True
        result.warnings.append(
            f"{len(fc_embeddings):,} first-class faces exceeds the "
            f"{EXACT_AGGLOMERATIVE_MAX_FACES:,} exact-clustering limit; used "
            "k-means blocking before agglomerative. Blocking can only split, "
            "never merge -- pass 2 recovers the splits."
        )
        pass1 = _pass1_blocked(
            fc_embeddings, config.pass1_threshold, EXACT_AGGLOMERATIVE_MAX_FACES
        )
    else:
        pass1 = _agglomerative(fc_embeddings, config.pass1_threshold)

    raw_group_count = int(pass1.max()) + 1
    groups: list[list[int]] = [[] for _ in range(raw_group_count)]
    for local_index, group_id in enumerate(pass1):
        groups[int(group_id)].append(int(first_class_idx[local_index]))

    # A group born holding two faces from one photograph is impossible. Repair
    # it here, before pass 2 can build on the contradiction.
    if config.enforce_same_photo_rule:
        groups, result.pass1_splits_same_photo = _split_same_photo_groups(
            groups, embeddings, photo_ids, quality_scores, config.pass1_threshold
        )

    group_count = len(groups)
    result.pass1_group_count = group_count

    # ---- PASS 2 -- merge, with evidence ---------------------------------
    centroids = np.stack(
        [
            _centroid(
                embeddings[np.array(members)],
                _quality_weights(quality_scores, np.array(members)),
            )
            for members in groups
        ]
    )
    group_photos = [set(photo_ids[i] for i in members) for members in groups]

    union = _UnionFind(group_count)
    root_photos = {i: set(group_photos[i]) for i in range(group_count)}

    if group_count > 1:
        # Unit vectors, so cosine distance is 1 - dot.
        distances = 1.0 - (centroids @ centroids.T)
        upper_i, upper_j = np.triu_indices(group_count, k=1)
        candidate_mask = distances[upper_i, upper_j] < config.pass2_threshold
        candidate_i = upper_i[candidate_mask]
        candidate_j = upper_j[candidate_mask]
        candidate_d = distances[candidate_i, candidate_j]

        # Closest first: a greedy merge order that is stable across runs.
        for order in np.argsort(candidate_d, kind="stable"):
            a, b = int(candidate_i[order]), int(candidate_j[order])
            root_a, root_b = union.find(a), union.find(b)
            if root_a == root_b:
                continue

            # THE SAME-PHOTO RULE. Two faces in one photo cannot be one person.
            # Free, obvious once stated, and it eliminates a whole class of the
            # worst errors. Checked against the MERGED sets, so a chain of
            # merges cannot sneak past it.
            if config.enforce_same_photo_rule and (
                root_photos[root_a] & root_photos[root_b]
            ):
                result.blocked_merges_same_photo += 1
                continue

            if union.union(root_a, root_b):
                new_root = union.find(root_a)
                merged = root_photos[root_a] | root_photos[root_b]
                root_photos[new_root] = merged
                for stale in (root_a, root_b):
                    if stale != new_root:
                        root_photos.pop(stale, None)
                result.pass2_merge_count += 1

    merged_groups: dict[int, list[int]] = {}
    for group_id in range(group_count):
        merged_groups.setdefault(union.find(group_id), []).extend(groups[group_id])

    # ---- Persons, and the leftovers we deliberately keep ----------------
    persons: list[Person] = []
    labels = np.full(total, -1, dtype=np.int64)
    for members in sorted(merged_groups.values(), key=len, reverse=True):
        if len(members) < config.min_faces_per_person:
            # A leftover: person_id stays NULL. NOT deleted. A distant cousin
            # in 3 photos is exactly the guest most delighted to find them, and
            # tier-2 search reaches these individually.
            continue
        person_id = len(persons)
        member_array = np.array(members, dtype=np.int64)
        best = int(member_array[int(np.argmax(quality_scores[member_array]))])
        persons.append(
            Person(
                person_id=person_id,
                face_indices=sorted(members),
                photo_ids={photo_ids[i] for i in members},
                centroid=_centroid(
                    embeddings[member_array],
                    _quality_weights(quality_scores, member_array),
                ),
                cover_face_index=best,
                label=f"Person {person_id + 1}",
            )
        )
        labels[member_array] = person_id

    # ---- Second-class faces may now JOIN, on stricter terms -------------
    second_class_idx = np.where(second_class)[0]
    # Reported even when nothing is measured, so a folder with no small faces
    # shows the setting that WOULD apply rather than a bare 0.00.
    result.second_class_threshold_used = float(config.second_class_assign_threshold)
    if persons.__len__() and second_class_idx.size:
        person_centroids = np.stack([p.centroid for p in persons])
        distances = 1.0 - (embeddings[second_class_idx] @ person_centroids.T)

        # How close a small face must be is MEASURED on this folder, not fixed.
        # A small face sitting in the same photograph as an identified person is
        # certainly not that person, which hands us a labelled negative set for
        # exactly this decision -- see autotune.choose_second_class_threshold.
        negatives = [
            distances[row][col]
            for row, face_index in enumerate(second_class_idx)
            for col, person in enumerate(persons)
            if photo_ids[face_index] in person.photo_ids
        ]
        auto_small = choose_second_class_threshold(
            np.array(negatives, dtype=np.float32),
            default=config.second_class_assign_threshold,
        )
        # Opt out with WFF_SECOND_CLASS_AUTO=0 to pin the configured value.
        small_threshold = (
            auto_small.value
            if getattr(config, "second_class_auto", _SECOND_CLASS_AUTO)
            else config.second_class_assign_threshold
        )
        result.second_class_threshold_used = float(small_threshold)
        result.second_class_threshold_basis = auto_small.basis
        result.second_class_negatives_measured = auto_small.pairs_measured

        for row, face_index in enumerate(second_class_idx):
            best_person = int(np.argmin(distances[row]))
            if distances[row][best_person] >= small_threshold:
                continue
            person = persons[best_person]
            # The same-photo rule again: if this person is already in this
            # photo, this small face is somebody else.
            if config.enforce_same_photo_rule and photo_ids[face_index] in person.photo_ids:
                continue
            person.face_indices.append(int(face_index))
            person.photo_ids.add(photo_ids[face_index])
            labels[face_index] = person.person_id
            result.second_class_assigned += 1

        # Centroids move once the small faces join, so recompute them.
        for person in persons:
            member_array = np.array(sorted(person.face_indices), dtype=np.int64)
            person.face_indices = member_array.tolist()
            person.centroid = _centroid(
                embeddings[member_array],
                _quality_weights(quality_scores, member_array),
            )

    result.labels = labels
    result.persons = persons
    result.leftover_face_count = int((labels < 0).sum())
    result.second_class_left_over = int((labels[second_class_idx] < 0).sum()) if second_class_idx.size else 0
    return result
