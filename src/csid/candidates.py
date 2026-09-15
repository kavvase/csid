from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from .keys import triple_key


def pair_code(i: int, j: int, n_categories: int) -> int:
    a, b = (i, j) if i < j else (j, i)
    return int(a) * int(n_categories) + int(b)


def decode_pair(code: int, n_categories: int) -> tuple[int, int]:
    return int(code // n_categories), int(code % n_categories)


@dataclass(frozen=True)
class CandidateReport:
    mode: str
    n_categories: int
    observed_pairs: int
    frequent_pair_edges: int
    support_triangles: int
    neighborhood_candidates: int
    explicit_candidates: int
    union_before_cap: int
    candidates_after_cap: int
    pair_candidate_min_count: int
    top_l_neighbors: int
    max_candidates: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _load_explicit_candidates(path: str | Path | None) -> set[tuple[int, int, int]]:
    if path is None:
        return set()
    p = Path(path)
    result: set[tuple[int, int, int]] = set()
    with p.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) == 1:
                row = [part.strip() for part in row[0].replace("|", ",").split(",")]
            if len(row) != 3:
                raise ValueError(f"Expected three item indexes in {p}: {row}")
            triple = triple_key(*[int(value) for value in row])
            if len(set(triple)) != 3:
                raise ValueError(f"Invalid triple in {p}: {row}")
            result.add(triple)
    return result


def _frequent_triangle_candidates(
    pair_totals: Mapping[int, int],
    *,
    n_categories: int,
    min_count: int,
) -> tuple[set[tuple[int, int, int]], int, list[list[tuple[int, int]]]]:
    adjacency: list[set[int]] = [set() for _ in range(n_categories)]
    weighted_neighbors: list[list[tuple[int, int]]] = [list() for _ in range(n_categories)]
    edge_count = 0
    for code, count in pair_totals.items():
        i, j = decode_pair(int(code), n_categories)
        weighted_neighbors[i].append((int(count), j))
        weighted_neighbors[j].append((int(count), i))
        if int(count) >= int(min_count):
            adjacency[i].add(j)
            adjacency[j].add(i)
            edge_count += 1

    triangles: set[tuple[int, int, int]] = set()
    for i in range(n_categories):
        higher = {j for j in adjacency[i] if j > i}
        for j in higher:
            common = higher.intersection(adjacency[j])
            for k in common:
                if k > j:
                    triangles.add((i, j, k))
    return triangles, edge_count, weighted_neighbors


def _neighborhood_candidates(
    weighted_neighbors: Sequence[Sequence[tuple[int, int]]],
    *,
    top_l: int,
) -> set[tuple[int, int, int]]:
    if top_l <= 1:
        return set()
    result: set[tuple[int, int, int]] = set()
    for center, values in enumerate(weighted_neighbors):
        neighbors = [item for _, item in sorted(values, key=lambda x: (-x[0], x[1]))[: int(top_l)]]
        for a, b in combinations(neighbors, 2):
            triple = triple_key(center, int(a), int(b))
            if len(set(triple)) == 3:
                result.add(triple)
    return result


def _candidate_score(
    triple: tuple[int, int, int],
    pair_totals: Mapping[int, int],
    n_categories: int,
) -> tuple[int, int]:
    i, j, k = triple
    values = (
        int(pair_totals.get(pair_code(i, j, n_categories), 0)),
        int(pair_totals.get(pair_code(i, k, n_categories), 0)),
        int(pair_totals.get(pair_code(j, k, n_categories), 0)),
    )
    return min(values), sum(values)


def generate_triple_candidates(
    pair_totals: Mapping[int, int],
    *,
    n_categories: int,
    mode: str = "auto",
    exhaustive_max_categories: int = 40,
    pair_candidate_min_count: int = 10,
    top_l_neighbors: int = 20,
    max_candidates: int | None = 250_000,
    explicit_candidates: Iterable[Sequence[int]] | None = None,
    explicit_candidate_file: str | Path | None = None,
) -> tuple[np.ndarray, CandidateReport]:
    """Generate candidate triples without enumerating all C choose 3 by default.

    Sparse mode uses two unions:

    * triangles of the pair graph with count >= ``pair_candidate_min_count``;
    * wedges among each item's top-L observed pair neighbours.

    If the pair threshold does not exceed a later all-three count threshold, every
    triple meeting that all-three threshold is guaranteed to appear in the triangle
    set, because all three of its pair supports are at least as large.
    """

    requested_mode = str(mode).lower()
    if requested_mode not in {"auto", "exhaustive", "sparse"}:
        raise ValueError("mode must be auto, exhaustive, or sparse")
    effective = (
        "exhaustive"
        if requested_mode == "exhaustive" or (requested_mode == "auto" and n_categories <= exhaustive_max_categories)
        else "sparse"
    )

    explicit = _load_explicit_candidates(explicit_candidate_file)
    if explicit_candidates is not None:
        for values in explicit_candidates:
            triple = tuple(sorted(int(v) for v in values))
            if len(triple) != 3 or len(set(triple)) != 3:
                raise ValueError(f"Invalid explicit candidate: {values}")
            explicit.add(triple)

    if effective == "exhaustive":
        candidates = set(combinations(range(n_categories), 3))
        support_triangles = len(candidates)
        neighbor_count = 0
        edge_count = len(pair_totals)
    else:
        support, edge_count, weighted = _frequent_triangle_candidates(
            pair_totals,
            n_categories=n_categories,
            min_count=pair_candidate_min_count,
        )
        neighborhood = _neighborhood_candidates(weighted, top_l=top_l_neighbors)
        candidates = support.union(neighborhood)
        support_triangles = len(support)
        neighbor_count = len(neighborhood)
    candidates.update(explicit)
    union_before_cap = len(candidates)

    if max_candidates is not None and max_candidates > 0 and len(candidates) > max_candidates:
        ranked = sorted(
            candidates,
            key=lambda t: (_candidate_score(t, pair_totals, n_categories), t),
            reverse=True,
        )[: int(max_candidates)]
        candidates = set(ranked)

    ordered = np.asarray(sorted(candidates), dtype=np.int32)
    if ordered.size == 0:
        ordered = np.empty((0, 3), dtype=np.int32)
    report = CandidateReport(
        mode=effective,
        n_categories=int(n_categories),
        observed_pairs=int(len(pair_totals)),
        frequent_pair_edges=int(edge_count),
        support_triangles=int(support_triangles),
        neighborhood_candidates=int(neighbor_count),
        explicit_candidates=int(len(explicit)),
        union_before_cap=int(union_before_cap),
        candidates_after_cap=int(len(ordered)),
        pair_candidate_min_count=int(pair_candidate_min_count),
        top_l_neighbors=int(top_l_neighbors),
        max_candidates=None if max_candidates is None else int(max_candidates),
    )
    return ordered, report


def pair_candidates_for_model(
    pair_totals: Mapping[int, int],
    triple_candidates: np.ndarray,
    *,
    n_categories: int,
    min_count: int = 1,
) -> np.ndarray:
    pairs: set[tuple[int, int]] = {
        decode_pair(int(code), n_categories) for code, count in pair_totals.items() if int(count) >= int(min_count)
    }
    for i, j, k in np.asarray(triple_candidates, dtype=np.int32):
        pairs.update(((int(i), int(j)), (int(i), int(k)), (int(j), int(k))))
    ordered = np.asarray(sorted(pairs), dtype=np.int32)
    return ordered if ordered.size else np.empty((0, 2), dtype=np.int32)
