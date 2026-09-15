import numpy as np

from csid.ising import fit_pairwise_ising_exact
from csid.synthetic import SyntheticModel, default_alpha, sample_loglinear


def test_exact_pairwise_ising_small_system():
    model = SyntheticModel(
        n_categories=6,
        alpha=default_alpha(6, strength=0.0),
        h=np.asarray([-0.3, -0.2, -0.1, -0.4, -0.25, -0.35]),
        pair={},
        triple={},
        min_size=1,
    )
    X = sample_loglinear(model, 6000, seed=123)
    fit = fit_pairwise_ising_exact(X, min_size=1)
    assert fit.success
    assert len(fit.J) == 15
    assert np.sqrt(np.mean(fit.J**2)) < 0.10
