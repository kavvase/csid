"""Small exact pairwise-Ising fitter used only by synthetic audits.

The production CSID estimator never enumerates the full state space.  This
module is intentionally restricted to small synthetic systems and provides the
ordinary pairwise Ising benchmark without a cardinality potential.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass(frozen=True)
class PairwiseIsingFit:
    n_categories: int
    min_size: int
    h: np.ndarray
    pairs: np.ndarray
    J: np.ndarray
    success: bool
    message: str
    iterations: int
    gradient_inf_norm: float


def _state_features(n_categories: int, min_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.arange(1 << n_categories, dtype=np.uint64)
    bits = (
        (states[:, None] >> np.arange(n_categories, dtype=np.uint64)[None, :]) & 1
    ).astype(float)
    allowed = bits.sum(axis=1) >= int(min_size)
    bits = bits[allowed]
    pairs = np.asarray(tuple(combinations(range(n_categories), 2)), dtype=np.int32)
    pair_features = np.column_stack(
        [bits[:, int(i)] * bits[:, int(j)] for i, j in pairs]
    )
    return bits, pairs, pair_features


def fit_pairwise_ising_exact(
    X: np.ndarray,
    *,
    min_size: int = 1,
    l2: float = 1.0e-5,
    max_iter: int = 1_000,
    tol: float = 1.0e-8,
) -> PairwiseIsingFit:
    """Fit an ordinary pairwise Ising model on the observed nonempty support.

    The model is

        P(x) propto exp(h^T x + sum_{i<j} J_ij x_i x_j),  |x| >= min_size.

    It deliberately omits a cardinality potential.  The routine is used only
    to demonstrate how an omitted basket-size structure contaminates pairwise
    couplings in small synthetic systems.
    """

    X = np.asarray(X, dtype=np.uint8)
    if X.ndim != 2 or len(X) == 0 or not np.all((X == 0) | (X == 1)):
        raise ValueError("X must be a nonempty binary matrix")
    C = int(X.shape[1])
    if C > 14:
        raise ValueError("Exact pairwise-Ising fitting is limited to C <= 14")
    if np.any(X.sum(axis=1) < int(min_size)):
        raise ValueError("X contains baskets outside the requested state space")

    single_states, pairs, pair_states = _state_features(C, int(min_size))
    state_features = np.concatenate([single_states, pair_states], axis=1)
    empirical_pairs = np.asarray(
        [np.mean(X[:, int(i)] * X[:, int(j)]) for i, j in pairs], dtype=float
    )
    empirical = np.concatenate([X.mean(axis=0, dtype=float), empirical_pairs])

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        energy = state_features @ parameters
        log_z = float(logsumexp(energy))
        probabilities = np.exp(energy - log_z)
        model_mean = probabilities @ state_features
        value = log_z - float(np.dot(parameters, empirical))
        gradient = model_mean - empirical
        if l2 > 0:
            value += 0.5 * float(l2) * float(np.dot(parameters, parameters))
            gradient = gradient + float(l2) * parameters
        return float(value), np.asarray(gradient, dtype=float)

    result = minimize(
        objective,
        np.zeros(C + len(pairs), dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": int(max_iter),
            "maxfun": int(max(2_000, max_iter * 30)),
            "ftol": float(tol),
            "gtol": float(tol),
            "maxls": 50,
        },
    )
    gradient_inf_norm = float(np.linalg.norm(np.asarray(result.jac), ord=np.inf))
    success = bool(
        np.isfinite(result.fun)
        and gradient_inf_norm <= max(1.0e-4, 10.0 * float(tol))
    )
    parameters = np.asarray(result.x, dtype=float)
    return PairwiseIsingFit(
        n_categories=C,
        min_size=int(min_size),
        h=parameters[:C].copy(),
        pairs=pairs,
        J=parameters[C:].copy(),
        success=success,
        message=str(result.message),
        iterations=int(result.nit),
        gradient_inf_norm=gradient_inf_norm,
    )
