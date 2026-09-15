from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .candidates import pair_code
from .data import SparseBasketDataset


@dataclass(frozen=True)
class PairScan:
    n_categories: int
    n_baskets: int
    max_basket_size: int
    N_by_size: np.ndarray
    singleton_by_size: np.ndarray
    pair_totals: dict[int, int]


@dataclass(frozen=True)
class SizeStratifiedCounts:
    n_categories: int
    n_baskets: int
    max_basket_size: int
    N_by_size: np.ndarray
    singleton_by_size: np.ndarray
    pair_candidates: np.ndarray
    pair_by_size: np.ndarray
    triple_candidates: np.ndarray
    triple_by_size: np.ndarray

    @property
    def pair_totals(self) -> np.ndarray:
        return self.pair_by_size.sum(axis=1)

    @property
    def triple_totals(self) -> np.ndarray:
        return self.triple_by_size.sum(axis=1)

    def subset_triples(self, keep: np.ndarray) -> SizeStratifiedCounts:
        mask = np.asarray(keep, dtype=bool)
        if mask.shape != (len(self.triple_candidates),):
            raise ValueError("triple mask shape mismatch")
        return SizeStratifiedCounts(
            n_categories=self.n_categories,
            n_baskets=self.n_baskets,
            max_basket_size=self.max_basket_size,
            N_by_size=self.N_by_size,
            singleton_by_size=self.singleton_by_size,
            pair_candidates=self.pair_candidates,
            pair_by_size=self.pair_by_size,
            triple_candidates=self.triple_candidates[mask],
            triple_by_size=self.triple_by_size[mask],
        )


def scan_pairs(
    dataset: SparseBasketDataset, *, count_pair_totals: bool = True
) -> PairScan:
    """First pass: basket sizes, singleton counts, and optional pair totals."""

    sizes = dataset.basket_sizes
    max_size = int(sizes.max(initial=0))
    N_by_size = np.bincount(sizes, minlength=max_size + 1).astype(np.int64)
    singleton = np.zeros((dataset.n_categories, max_size + 1), dtype=np.int64)
    pair_totals: dict[int, int] = defaultdict(int)
    C = dataset.n_categories

    for items in dataset.iter_baskets():
        size = len(items)
        singleton[items, size] += 1
        if count_pair_totals:
            for i, j in combinations(items.tolist(), 2):
                pair_totals[pair_code(int(i), int(j), C)] += 1

    return PairScan(
        n_categories=C,
        n_baskets=dataset.n_baskets,
        max_basket_size=max_size,
        N_by_size=N_by_size,
        singleton_by_size=singleton,
        pair_totals=dict(pair_totals),
    )


def _triple_key(i: int, j: int, k: int, n_categories: int) -> int:
    a, b, c = sorted((int(i), int(j), int(k)))
    return (a * n_categories + b) * n_categories + c


def count_for_candidates(
    dataset: SparseBasketDataset,
    *,
    pair_candidates: np.ndarray,
    triple_candidates: np.ndarray,
    base_scan: PairScan | None = None,
) -> SizeStratifiedCounts:
    """Second pass: exact size-stratified counts for selected pairs and triples."""

    C = dataset.n_categories
    if base_scan is None:
        base_scan = scan_pairs(dataset)
    if base_scan.n_categories != C or base_scan.n_baskets != dataset.n_baskets:
        raise ValueError("base_scan does not match dataset")

    pairs = np.asarray(pair_candidates, dtype=np.int32).reshape(-1, 2)
    triples = np.asarray(triple_candidates, dtype=np.int32).reshape(-1, 3)
    R = base_scan.max_basket_size + 1
    pair_counts = np.zeros((len(pairs), R), dtype=np.int64)
    triple_counts = np.zeros((len(triples), R), dtype=np.int64)
    pair_index = {pair_code(int(i), int(j), C): idx for idx, (i, j) in enumerate(pairs)}
    triple_index = {_triple_key(int(i), int(j), int(k), C): idx for idx, (i, j, k) in enumerate(triples)}
    candidate_items = set(
        np.unique(
            np.concatenate(
                [
                    pairs.ravel(),
                    triples.ravel(),
                ]
            )
        ).tolist()
    )

    for items in dataset.iter_baskets():
        size = len(items)
        values = [int(item) for item in items if int(item) in candidate_items]
        if pair_index:
            for i, j in combinations(values, 2):
                idx = pair_index.get(pair_code(int(i), int(j), C))
                if idx is not None:
                    pair_counts[idx, size] += 1
        if triple_index and size >= 3:
            for i, j, k in combinations(values, 3):
                idx = triple_index.get(_triple_key(int(i), int(j), int(k), C))
                if idx is not None:
                    triple_counts[idx, size] += 1

    return SizeStratifiedCounts(
        n_categories=C,
        n_baskets=dataset.n_baskets,
        max_basket_size=base_scan.max_basket_size,
        N_by_size=base_scan.N_by_size,
        singleton_by_size=base_scan.singleton_by_size,
        pair_candidates=pairs,
        pair_by_size=pair_counts,
        triple_candidates=triples,
        triple_by_size=triple_counts,
    )


def filter_triples_by_count(
    counts: SizeStratifiedCounts,
    *,
    min_count: int,
) -> SizeStratifiedCounts:
    if min_count <= 0:
        return counts
    return counts.subset_triples(counts.triple_totals >= int(min_count))
