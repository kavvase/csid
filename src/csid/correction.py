from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import cast

import numpy as np
import pandas as pd

from .contrasts import ContrastMatrix
from .counts import SizeStratifiedCounts
from .data import SparseBasketDataset
from .estimator import CSIDFit, fit_csid
from .keys import as_pair, as_triple, triple_key
from .stats import pearson_spearman


@dataclass(frozen=True)
class CorrectionResult:
    corrected_fit: CSIDFit
    corrected_n11: np.ndarray


def _pair_cells(counts: SizeStratifiedCounts, pair_index: int) -> np.ndarray:
    candidate = counts.pair_candidates[int(pair_index)]
    i, j = map(int, candidate)
    R = counts.max_basket_size + 1
    N = counts.N_by_size
    ci = counts.singleton_by_size[i]
    cj = counts.singleton_by_size[j]
    pij = counts.pair_by_size[int(pair_index)]

    def shift(values: np.ndarray, offset: int) -> np.ndarray:
        out = np.zeros(R, dtype=float)
        available = min(R - offset, len(values) - offset)
        if available > 0:
            out[:available] = values[offset : offset + available]
        return out

    n00 = shift(N, 0) - shift(ci, 0) - shift(cj, 0) + shift(pij, 0)
    n10 = shift(ci, 1) - shift(pij, 1)
    n01 = shift(cj, 1) - shift(pij, 1)
    n11 = shift(pij, 2)
    return cast(np.ndarray, np.maximum(np.vstack([n00, n10, n01, n11]).T, 0.0))


def _theta_map(
    triple_fit: CSIDFit,
    theta_override: Mapping[tuple[int, int, int], float] | None,
) -> dict[tuple[int, int, int], float]:
    if theta_override is not None:
        return {triple_key(*map(int, key)): float(value) for key, value in theta_override.items()}
    return {
        as_triple(candidate): float(value)
        for candidate, value in zip(triple_fit.candidates, triple_fit.theta, strict=False)
    }


def corrected_pair_fit(
    dataset: SparseBasketDataset,
    counts: SizeStratifiedCounts,
    raw_pair_fit: CSIDFit,
    triple_fit: CSIDFit,
    *,
    theta_override: Mapping[tuple[int, int, int], float] | None = None,
    epsilon: float = 0.5,
    min_rest_size: int = 2,
    ridge: float = 1.0,
    exponent_clip: float = 20.0,
    block_size: int = 50_000,
    gauge_weights: np.ndarray | None = None,
) -> CorrectionResult:
    """Remove estimated triple terms from the pair 11 cells and refit CSID-2."""

    pair_index = {as_pair(pair): q for q, pair in enumerate(counts.pair_candidates)}
    theta = _theta_map(triple_fit, theta_override)
    incidence: list[list[tuple[int, float]]] = [[] for _ in range(len(counts.pair_candidates))]
    for triple, value in theta.items():
        i, j, k = triple
        for pair, moderator in (
            ((i, j), k),
            ((i, k), j),
            ((j, k), i),
        ):
            q = pair_index.get(as_pair(pair))
            if q is not None and value != 0:
                incidence[q].append((int(moderator), float(value)))

    R = counts.max_basket_size + 1
    corrected_n11 = np.zeros((len(counts.pair_candidates), R), dtype=float)
    candidate_items = set(np.asarray(counts.pair_candidates, dtype=np.int32).ravel().tolist())
    for items in dataset.iter_baskets():
        m = len(items)
        if m < 2:
            continue
        r = m - 2
        if r >= R:
            continue
        item_set = set(map(int, items))
        present_candidates = [item for item in item_set if item in candidate_items]
        for a, b in combinations(present_candidates, 2):
            pair = as_pair((int(a), int(b)))
            q = pair_index.get(pair)
            if q is None:
                continue
            g = 0.0
            for moderator, value in incidence[q]:
                if moderator in item_set:
                    g += value
            corrected_n11[q, r] += float(np.exp(-np.clip(g, -float(exponent_clip), float(exponent_clip))))

    Q = len(counts.pair_candidates)
    lambda_hat = np.zeros((Q, R), dtype=float)
    weights = np.zeros((Q, R), dtype=float)
    r_values = np.arange(R)
    for q in range(Q):
        cells = _pair_cells(counts, q)
        cells[:, 3] = corrected_n11[q]
        smooth = cells + float(epsilon)
        lambda_hat[q] = np.log(smooth[:, 0]) - np.log(smooth[:, 1]) - np.log(smooth[:, 2]) + np.log(smooth[:, 3])
        weights[q] = 1.0 / np.sum(1.0 / smooth, axis=1)
        valid = (r_values >= int(min_rest_size)) & (cells.sum(axis=1) > 0)
        lambda_hat[q, ~valid] = 0.0
        weights[q, ~valid] = 0.0

    table = ContrastMatrix(
        k=2,
        candidates=np.asarray(counts.pair_candidates, dtype=np.int32),
        r_values=np.asarray(r_values, dtype=np.int32),
        lambda_hat=lambda_hat,
        weights=weights,
        n_all_total=np.asarray(counts.pair_totals, dtype=float),
        epsilon=float(epsilon),
        min_rest_size=int(min_rest_size),
        n_baskets=int(counts.n_baskets),
    )
    return CorrectionResult(
        corrected_fit=fit_csid(
            table,
            ridge=ridge,
            block_size=block_size,
            gauge_weights=(
                np.asarray(raw_pair_fit.mu, dtype=float)
                if gauge_weights is None
                else np.asarray(gauge_weights, dtype=float)
            ),
        ),
        corrected_n11=corrected_n11,
    )


def pair_correction_table(
    raw_fit: CSIDFit,
    corrected_fit: CSIDFit,
    *,
    category_names: Sequence[str],
) -> pd.DataFrame:
    corrected_index = {as_pair(pair): q for q, pair in enumerate(corrected_fit.candidates)}
    rows: list[dict[str, object]] = []
    for q, pair in enumerate(raw_fit.candidates):
        key = as_pair(pair)
        qc = corrected_index.get(key)
        if qc is None:
            continue
        delta = float(corrected_fit.theta[qc] - raw_fit.theta[q])
        rows.append(
            {
                "item_1_idx": key[0],
                "item_2_idx": key[1],
                "item_1": category_names[key[0]],
                "item_2": category_names[key[1]],
                "J_raw": float(raw_fit.theta[q]),
                "J_corrected": float(corrected_fit.theta[qc]),
                "delta_J": delta,
                "abs_delta_J": abs(delta),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_delta_J", ascending=False).reset_index(drop=True)


def correction_delta(raw_fit: CSIDFit, corrected_fit: CSIDFit) -> tuple[list[tuple[int, int]], np.ndarray]:
    corrected_index = {as_pair(pair): q for q, pair in enumerate(corrected_fit.candidates)}
    keys: list[tuple[int, int]] = []
    values: list[float] = []
    for q, pair in enumerate(raw_fit.candidates):
        key = as_pair(pair)
        qc = corrected_index.get(key)
        if qc is not None:
            keys.append(key)
            values.append(float(corrected_fit.theta[qc] - raw_fit.theta[q]))
    return keys, np.asarray(values, dtype=float)


def correction_transfer_summary(
    *,
    full_dataset: SparseBasketDataset,
    h2_dataset: SparseBasketDataset,
    full_counts: SizeStratifiedCounts,
    h2_counts: SizeStratifiedCounts,
    full_pair_fit: CSIDFit,
    h2_pair_fit: CSIDFit,
    full_triple_fit: CSIDFit,
    h1_triple_fit: CSIDFit,
    h2_triple_fit: CSIDFit,
    epsilon: float,
    min_rest_size: int,
    ridge2: float,
) -> pd.DataFrame:
    """Correlate H1->H2 pair corrections with H2 and full-data references."""

    full_corrected = corrected_pair_fit(
        full_dataset,
        full_counts,
        full_pair_fit,
        full_triple_fit,
        epsilon=epsilon,
        min_rest_size=min_rest_size,
        ridge=ridge2,
    ).corrected_fit
    h2_same = corrected_pair_fit(
        h2_dataset,
        h2_counts,
        h2_pair_fit,
        h2_triple_fit,
        epsilon=epsilon,
        min_rest_size=min_rest_size,
        ridge=ridge2,
    ).corrected_fit
    h1_to_h2 = corrected_pair_fit(
        h2_dataset,
        h2_counts,
        h2_pair_fit,
        h1_triple_fit,
        epsilon=epsilon,
        min_rest_size=min_rest_size,
        ridge=ridge2,
    ).corrected_fit

    def delta_map(raw: CSIDFit, corrected: CSIDFit) -> dict[tuple[int, int], float]:
        keys, values = correction_delta(raw, corrected)
        return dict(zip(keys, values.astype(float), strict=False))

    cross = delta_map(h2_pair_fit, h1_to_h2)
    same = delta_map(h2_pair_fit, h2_same)
    full = delta_map(full_pair_fit, full_corrected)
    common_h2 = sorted(set(cross).intersection(same))
    common_full = sorted(set(cross).intersection(full))
    cross_h2 = np.asarray([cross[key] for key in common_h2], dtype=float)
    same_h2 = np.asarray([same[key] for key in common_h2], dtype=float)
    cross_full = np.asarray([cross[key] for key in common_full], dtype=float)
    full_ref = np.asarray([full[key] for key in common_full], dtype=float)
    pearson_h2, spearman_h2 = pearson_spearman(cross_h2, same_h2)
    pearson_full, spearman_full = pearson_spearman(cross_full, full_ref)
    return pd.DataFrame(
        [
            {
                "comparison": "H1 theta on H2 vs H2 theta on H2",
                "n_pairs": len(common_h2),
                "pearson": pearson_h2,
                "spearman": spearman_h2,
            },
            {
                "comparison": "H1 theta on H2 vs full-data correction",
                "n_pairs": len(common_full),
                "pearson": pearson_full,
                "spearman": spearman_full,
            },
        ]
    )
