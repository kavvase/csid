from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contrasts import ContrastMatrix


@dataclass(frozen=True)
class CSIDFit:
    k: int
    candidates: np.ndarray
    theta: np.ndarray
    beta: np.ndarray
    mu: np.ndarray
    se: np.ndarray
    z: np.ndarray
    raw_contrast: np.ndarray
    gauge_weights: np.ndarray
    converged: bool
    n_iter: int
    objective: float
    ridge: float
    table: ContrastMatrix

    @property
    def weighted_gauge_mean(self) -> float:
        denom = float(self.gauge_weights.sum())
        return float(np.dot(self.gauge_weights, self.theta) / denom) if denom > 0 else float("nan")


def _blocks(length: int, block_size: int):
    for start in range(0, length, block_size):
        yield slice(start, min(start + block_size, length))


def fit_csid(
    table: ContrastMatrix,
    *,
    ridge: float = 1.0,
    max_iter: int = 200,
    tol: float = 1.0e-8,
    block_size: int = 50_000,
    gauge_weights: np.ndarray | None = None,
) -> CSIDFit:
    """Gauge-constrained weighted ridge fit with blockwise passes.

    The arrays may be numpy memmaps.  Memory use is O(number of candidates +
    number of rest-size layers), not O(number of candidates * layers) beyond the
    contrast store itself.
    """

    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    W, L = table.weights, table.lambda_hat
    Q, R = W.shape
    if Q == 0:
        raise ValueError("No candidates to fit")

    mu = np.zeros(Q, dtype=float)
    raw_num = np.zeros(Q, dtype=float)
    for block in _blocks(Q, block_size):
        wb = np.asarray(W[block], dtype=float)
        lb = np.asarray(L[block], dtype=float)
        mu[block] = wb.sum(axis=1)
        raw_num[block] = np.sum(wb * lb, axis=1)
    if not np.any(mu > 0):
        raise ValueError("No informative candidate/rest-size cells")

    gauge = mu.copy() if gauge_weights is None else np.asarray(gauge_weights, dtype=float).copy()
    if gauge.shape != (Q,) or np.any(gauge < 0) or not np.any(gauge > 0):
        raise ValueError("gauge_weights must be non-negative with one value per candidate")

    theta = np.zeros(Q, dtype=float)
    beta = np.zeros(R, dtype=float)
    b = np.zeros(Q, dtype=float)
    converged = False
    n_iter = 0

    for iteration in range(1, max_iter + 1):
        for block in _blocks(Q, block_size):
            wb = np.asarray(W[block], dtype=float)
            lb = np.asarray(L[block], dtype=float)
            b[block] = np.sum(wb * (lb - beta[None, :]), axis=1)

        denom = mu + float(ridge)
        safe = denom > 0
        gamma_denom = float(np.sum((gauge[safe] ** 2) / denom[safe]))
        gamma_num = float(np.sum(gauge[safe] * b[safe] / denom[safe]))
        gamma = gamma_num / gamma_denom if gamma_denom > 0 else 0.0
        theta_new = np.zeros_like(theta)
        theta_new[safe] = (b[safe] - gamma * gauge[safe]) / denom[safe]

        beta_num = np.zeros(R, dtype=float)
        beta_den = np.zeros(R, dtype=float)
        for block in _blocks(Q, block_size):
            wb = np.asarray(W[block], dtype=float)
            lb = np.asarray(L[block], dtype=float)
            beta_num += np.sum(wb * (lb - theta_new[block, None]), axis=0)
            beta_den += wb.sum(axis=0)
        beta_new = beta.copy()
        informative = beta_den > 0
        beta_new[informative] = beta_num[informative] / beta_den[informative]

        delta = max(
            float(np.max(np.abs(theta_new - theta))),
            float(np.max(np.abs(beta_new - beta))),
        )
        theta, beta = theta_new, beta_new
        n_iter = iteration
        if delta < tol:
            converged = True
            break

    objective = float(ridge * np.dot(theta, theta))
    for block in _blocks(Q, block_size):
        wb = np.asarray(W[block], dtype=float)
        lb = np.asarray(L[block], dtype=float)
        residual = lb - theta[block, None] - beta[None, :]
        objective += float(np.sum(wb * residual * residual))

    se = np.full(Q, np.inf, dtype=float)
    positive = mu + ridge > 0
    se[positive] = 1.0 / np.sqrt(mu[positive] + ridge)
    z = np.divide(theta, se, out=np.zeros_like(theta), where=np.isfinite(se) & (se > 0))
    raw = np.divide(raw_num, mu, out=np.zeros_like(mu), where=mu > 0)
    return CSIDFit(
        k=table.k,
        candidates=np.asarray(table.candidates, dtype=np.int32),
        theta=theta,
        beta=beta,
        mu=mu,
        se=se,
        z=z,
        raw_contrast=raw,
        gauge_weights=gauge,
        converged=converged,
        n_iter=n_iter,
        objective=objective,
        ridge=float(ridge),
        table=table,
    )


def fit_to_dataframe(
    fit: CSIDFit,
    *,
    category_names: Sequence[str] | None = None,
    singleton_support: np.ndarray | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_baskets = max(fit.table.n_baskets, 1)
    for q, candidate in enumerate(fit.candidates):
        items = tuple(map(int, candidate))
        support = float(fit.table.n_all_total[q] / n_baskets)
        row: dict[str, object] = {
            **{f"item_{position}_idx": item for position, item in enumerate(items, start=1)},
            "theta_hat" if fit.k == 3 else "J_hat": float(fit.theta[q]),
            "mu": float(fit.mu[q]),
            "se": float(fit.se[q]),
            "z": float(fit.z[q]),
            "raw_contrast": float(fit.raw_contrast[q]),
            "support": support,
        }
        if category_names is not None:
            row.update(
                {f"item_{position}": category_names[item] for position, item in enumerate(items, start=1)}
            )
        if fit.k == 3 and singleton_support is not None:
            product = float(np.prod(singleton_support[list(items)]))
            row["lift3"] = support / product if product > 0 else float("nan")
            row["residual_odds_multiplier"] = float(np.exp(fit.theta[q]))
        rows.append(row)
    return pd.DataFrame(rows)


def write_fit_csv(
    fit: CSIDFit,
    path: str | Path,
    *,
    category_names: Sequence[str] | None = None,
    singleton_support: np.ndarray | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fit_to_dataframe(
        fit,
        category_names=category_names,
        singleton_support=singleton_support,
    ).to_csv(p, index=False)
