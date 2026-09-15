"""Cardinality-aware higher-order pseudolikelihood benchmark.

The benchmark fits one conditional logistic model per item.  Each conditional
contains flexible rest-cardinality effects plus the pair and triple terms from
the fixed CSID candidate family. Shared pair/triple coefficients are obtained
by averaging their incident nodewise estimates and applying one global
reference-weighted centering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit

from .data import SparseBasketDataset


@dataclass(frozen=True)
class PseudolikelihoodFit:
    pair_candidates: np.ndarray
    triple_candidates: np.ndarray
    pair_theta: np.ndarray
    triple_theta: np.ndarray
    gauge_weights: np.ndarray
    selected_ridge: float
    validation_log_loss: float
    converged: bool
    diagnostics: pd.DataFrame

    @property
    def weighted_gauge_mean(self) -> float:
        total = float(self.gauge_weights.sum())
        if total <= 0:
            return float("nan")
        return float(np.dot(self.gauge_weights, self.triple_theta) / total)


def _validation_mask(
    dataset: SparseBasketDataset,
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    n = dataset.n_baskets
    if fraction <= 0 or n < 10:
        return np.zeros(n, dtype=bool)
    rng = np.random.default_rng(seed)
    if dataset.cluster_ids is None:
        count = min(max(int(round(fraction * n)), 1), n - 1)
        chosen = rng.choice(n, size=count, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[chosen] = True
        return mask
    groups = np.asarray(dataset.cluster_ids, dtype=object)
    unique = np.unique(groups)
    if len(unique) < 2:
        return np.zeros(n, dtype=bool)
    count = min(max(int(round(fraction * len(unique))), 1), len(unique) - 1)
    chosen = rng.choice(unique, size=count, replace=False)
    return np.isin(groups, chosen)


def _node_design(
    dataset: SparseBasketDataset,
    node: int,
    pair_candidates: np.ndarray,
    triple_candidates: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    X = dataset.X.astype(float, copy=False).tocsr()
    y = np.asarray(X[:, node].toarray()).ravel()
    rest_size = dataset.basket_sizes.astype(int) - y.astype(int)
    rows = np.arange(dataset.n_baskets, dtype=np.int64)
    cardinality = sparse.csr_matrix(
        (np.ones(dataset.n_baskets), (rows, rest_size)),
        shape=(dataset.n_baskets, dataset.n_categories),
    )

    pair_rows = np.flatnonzero(np.any(pair_candidates == node, axis=1))
    pair_others = np.asarray(
        [int(pair_candidates[q, 1] if pair_candidates[q, 0] == node else pair_candidates[q, 0]) for q in pair_rows],
        dtype=int,
    )
    pair_block = X[:, pair_others] if len(pair_others) else sparse.csr_matrix((dataset.n_baskets, 0))

    triple_rows = np.flatnonzero(np.any(triple_candidates == node, axis=1))
    triple_columns: list[sparse.csr_matrix] = []
    for q in triple_rows:
        others = [int(value) for value in triple_candidates[q] if int(value) != node]
        triple_columns.append(X[:, others[0]].multiply(X[:, others[1]]).tocsr())
    triple_block = (
        sparse.hstack(triple_columns, format="csr")
        if triple_columns
        else sparse.csr_matrix((dataset.n_baskets, 0))
    )
    design = sparse.hstack((cardinality, pair_block, triple_block), format="csr")
    return design, y, pair_rows, triple_rows


def _initial_coefficients(
    design: sparse.csr_matrix,
    y: np.ndarray,
    *,
    n_cardinality: int,
) -> np.ndarray:
    result = np.zeros(design.shape[1], dtype=float)
    cardinality = design[:, :n_cardinality]
    totals = np.asarray(cardinality.sum(axis=0)).ravel()
    positives = np.asarray(cardinality.T @ y).ravel()
    probabilities = (positives + 0.5) / (totals + 1.0)
    informative = totals > 0
    result[:n_cardinality][informative] = np.log(
        probabilities[informative] / (1.0 - probabilities[informative])
    )
    return result


def _fit_node(
    design: sparse.csr_matrix,
    y: np.ndarray,
    *,
    ridge: float,
    n_cardinality: int,
    max_iter: int,
    initial: np.ndarray | None = None,
) -> OptimizeResult:
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    start = (
        _initial_coefficients(design, y, n_cardinality=n_cardinality)
        if initial is None
        else np.asarray(initial, dtype=float)
    )

    return minimize(
        _node_objective,
        start,
        args=(design, y, float(ridge), int(n_cardinality)),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": 1.0e-8, "gtol": 1.0e-4},
    )


def _node_objective(
    coefficients: np.ndarray,
    design: sparse.csr_matrix,
    y: np.ndarray,
    ridge: float,
    n_cardinality: int,
) -> tuple[float, np.ndarray]:
    eta = np.asarray(design @ coefficients).ravel()
    interaction = coefficients[n_cardinality:]
    loss = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
    loss += 0.5 * float(ridge) * float(np.dot(interaction, interaction))
    residual = expit(eta) - y
    gradient = np.asarray(design.T @ residual).ravel()
    gradient[n_cardinality:] += float(ridge) * interaction
    return loss, gradient


def _log_loss(design: sparse.csr_matrix, y: np.ndarray, coefficients: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    eta = np.asarray(design @ coefficients).ravel()
    return float(np.mean(np.logaddexp(0.0, eta) - y * eta))


def _symmetrize_nodewise_triples(
    nodewise_theta: np.ndarray,
    gauge_weights: np.ndarray,
) -> np.ndarray:
    """Average incident estimates, then apply one global centering constant."""

    values = np.asarray(nodewise_theta, dtype=float)
    weights = np.asarray(gauge_weights, dtype=float)
    if values.ndim != 2 or weights.shape != (values.shape[1],):
        raise ValueError("Nodewise estimates and gauge weights have incompatible shapes")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("gauge_weights must be non-negative with positive total weight")
    symmetric = np.nanmean(values, axis=0)
    if np.any(~np.isfinite(symmetric)):
        raise ValueError("Every triple must have at least one incident nodewise estimate")
    symmetric -= float(np.dot(weights, symmetric) / weights.sum())
    return symmetric


def fit_cardinality_pseudolikelihood(
    dataset: SparseBasketDataset,
    *,
    pair_candidates: np.ndarray,
    triple_candidates: np.ndarray,
    gauge_weights: np.ndarray,
    ridge_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    validation_fraction: float = 0.2,
    seed: int = 42,
    max_iter: int = 500,
) -> PseudolikelihoodFit:
    """Fit the symmetrized cardinality-aware order-three pseudolikelihood."""

    pairs = np.asarray(pair_candidates, dtype=np.int32)
    triples = np.asarray(triple_candidates, dtype=np.int32)
    gauge = np.asarray(gauge_weights, dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_candidates must have shape (Q, 2)")
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("triple_candidates must have shape (Q, 3)")
    if gauge.shape != (len(triples),) or np.any(gauge < 0) or not np.any(gauge > 0):
        raise ValueError("gauge_weights must be non-negative with one value per triple")
    ridges = tuple(float(value) for value in ridge_grid)
    if not ridges or any(value < 0 for value in ridges):
        raise ValueError("ridge_grid must contain non-negative values")

    validation = _validation_mask(dataset, fraction=validation_fraction, seed=seed)
    has_validation = bool(np.any(validation))
    training = ~validation
    if not has_validation:
        ridges = (ridges[0],)

    validation_scores = np.zeros(len(ridges), dtype=float)
    node_cache: list[tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]] = []
    tuning_coefficients: list[list[np.ndarray]] = []
    tuning_rows: list[dict[str, object]] = []
    for node in range(dataset.n_categories):
        design, y, pair_rows, triple_rows = _node_design(dataset, node, pairs, triples)
        node_cache.append((design, y, pair_rows, triple_rows))
        train_design, train_y = design[training], y[training]
        valid_design = design[validation] if has_validation else train_design
        valid_y = y[validation] if has_validation else train_y
        initial = _initial_coefficients(train_design, train_y, n_cardinality=dataset.n_categories)
        node_coefficients: list[np.ndarray] = []
        for ridge_index, ridge in enumerate(ridges):
            result = _fit_node(
                train_design,
                train_y,
                ridge=ridge,
                n_cardinality=dataset.n_categories,
                max_iter=max_iter,
                initial=initial,
            )
            score = _log_loss(valid_design, valid_y, np.asarray(result.x, dtype=float))
            validation_scores[ridge_index] += score
            node_coefficients.append(np.asarray(result.x, dtype=float))
            tuning_rows.append(
                {
                    "stage": "tuning",
                    "node": node,
                    "ridge": ridge,
                    "converged": bool(result.success),
                    "n_iter": int(result.nit),
                    "log_loss": score,
                }
            )
        tuning_coefficients.append(node_coefficients)

    selected_index = int(np.nanargmin(validation_scores))
    selected_ridge = ridges[selected_index]
    pair_sum = np.zeros(len(pairs), dtype=float)
    pair_count = np.zeros(len(pairs), dtype=int)
    nodewise_triple_theta = np.full(
        (dataset.n_categories, len(triples)),
        np.nan,
        dtype=float,
    )
    final_rows: list[dict[str, object]] = []
    for node, (design, y, pair_rows, triple_rows) in enumerate(node_cache):
        result = _fit_node(
            design,
            y,
            ridge=selected_ridge,
            n_cardinality=dataset.n_categories,
            max_iter=max_iter,
            initial=tuning_coefficients[node][selected_index],
        )
        coefficients = np.asarray(result.x, dtype=float)
        pair_start = dataset.n_categories
        triple_start = pair_start + len(pair_rows)
        pair_sum[pair_rows] += coefficients[pair_start:triple_start]
        pair_count[pair_rows] += 1
        nodewise_triple_theta[node, triple_rows] = coefficients[triple_start:]
        final_rows.append(
            {
                "stage": "final",
                "node": node,
                "ridge": selected_ridge,
                "converged": bool(result.success),
                "n_iter": int(result.nit),
                "log_loss": _log_loss(design, y, coefficients),
            }
        )

    pair_theta = np.divide(pair_sum, pair_count, out=np.zeros_like(pair_sum), where=pair_count > 0)
    triple_theta = _symmetrize_nodewise_triples(nodewise_triple_theta, gauge)
    diagnostics = pd.DataFrame(tuning_rows + final_rows)
    validation_loss = float(validation_scores[selected_index] / dataset.n_categories)
    final = diagnostics[diagnostics["stage"].eq("final")]
    return PseudolikelihoodFit(
        pair_candidates=pairs,
        triple_candidates=triples,
        pair_theta=pair_theta,
        triple_theta=triple_theta,
        gauge_weights=gauge,
        selected_ridge=selected_ridge,
        validation_log_loss=validation_loss,
        converged=bool(final["converged"].all()),
        diagnostics=diagnostics,
    )


def pseudolikelihood_triples_frame(
    fit: PseudolikelihoodFit,
    *,
    category_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate, theta in zip(
        fit.triple_candidates,
        fit.triple_theta,
        strict=False,
    ):
        items = tuple(map(int, candidate))
        row: dict[str, object] = {
            "candidate": "|".join(map(str, items)),
            "item_1_idx": items[0],
            "item_2_idx": items[1],
            "item_3_idx": items[2],
            "theta_pl_h1": float(theta),
        }
        if category_names is not None:
            row.update(
                {
                    "item_1": category_names[items[0]],
                    "item_2": category_names[items[1]],
                    "item_3": category_names[items[2]],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)
