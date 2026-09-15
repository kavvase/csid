from itertools import combinations

import numpy as np
from scipy import sparse

from csid.data import SparseBasketDataset
from csid.pseudolikelihood import (
    _node_objective,
    _symmetrize_nodewise_triples,
    fit_cardinality_pseudolikelihood,
)
from csid.synthetic import SyntheticModel, default_alpha, sample_loglinear


def test_pseudolikelihood_objective_gradient():
    design = sparse.csr_matrix(
        np.asarray(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    y = np.asarray([0.0, 1.0, 1.0])
    coefficients = np.asarray([0.2, -0.1, 0.3, -0.4])
    _, gradient = _node_objective(coefficients, design, y, 0.7, 2)
    step = 1.0e-6
    numerical = np.zeros_like(coefficients)
    for index in range(len(coefficients)):
        upper, lower = coefficients.copy(), coefficients.copy()
        upper[index] += step
        lower[index] -= step
        upper_value, _ = _node_objective(upper, design, y, 0.7, 2)
        lower_value, _ = _node_objective(lower, design, y, 0.7, 2)
        numerical[index] = (upper_value - lower_value) / (2.0 * step)
    assert np.allclose(gradient, numerical, atol=1.0e-6)


def test_restricted_triple_symmetrization_only_applies_global_centering():
    nodewise = np.asarray(
        [
            [1.0, np.nan, 2.0],
            [3.0, 4.0, np.nan],
            [5.0, 6.0, 8.0],
            [np.nan, 10.0, 11.0],
        ]
    )
    weights = np.asarray([1.0, 2.0, 3.0])
    incident_means = np.nanmean(nodewise, axis=0)
    result = _symmetrize_nodewise_triples(nodewise, weights)

    assert abs(float(np.dot(weights, result))) < 1.0e-12
    assert np.allclose(np.diff(result), np.diff(incident_means))


def test_pseudolikelihood_recovers_planted_triple_and_gauge():
    categories = 5
    planted = (0, 1, 2)
    model = SyntheticModel(
        n_categories=categories,
        alpha=default_alpha(categories, strength=0.35),
        h=np.full(categories, -0.25),
        pair={(0, 1): 0.2, (2, 3): -0.15},
        triple={planted: 1.0},
        min_size=0,
    )
    X = sample_loglinear(model, 6000, seed=14)
    dataset = SparseBasketDataset.from_dense(X, name="pl-toy")
    pairs = np.asarray(tuple(combinations(range(categories), 2)), dtype=np.int32)
    triples = np.asarray(tuple(combinations(range(categories), 3)), dtype=np.int32)
    gauge = np.ones(len(triples), dtype=float)

    fit = fit_cardinality_pseudolikelihood(
        dataset,
        pair_candidates=pairs,
        triple_candidates=triples,
        gauge_weights=gauge,
        ridge_grid=(0.1,),
        validation_fraction=0.0,
        max_iter=120,
    )

    planted_index = list(map(tuple, triples)).index(planted)
    assert fit.converged
    assert abs(fit.weighted_gauge_mean) < 1e-10
    assert fit.triple_theta[planted_index] == np.max(fit.triple_theta)
    assert fit.triple_theta[planted_index] > 0.35

    repeated = fit_cardinality_pseudolikelihood(
        dataset,
        pair_candidates=pairs,
        triple_candidates=triples,
        gauge_weights=gauge,
        ridge_grid=(0.1,),
        validation_fraction=0.0,
        max_iter=120,
    )
    assert np.array_equal(fit.triple_theta, repeated.triple_theta)
