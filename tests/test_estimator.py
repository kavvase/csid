import numpy as np

from csid.analysis import AnalysisConfig, fit_generated_candidates
from csid.data import SparseBasketDataset


def random_dataset(seed=7, n=700, c=6):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, c), dtype=np.uint8)
    for row in range(n):
        size = int(rng.integers(2, min(5, c) + 1))
        X[row, rng.choice(c, size=size, replace=False)] = 1
    periods = np.where(np.arange(n) < n // 2, "H1", "H2")
    return SparseBasketDataset.from_dense(X, periods=periods)


def test_csid_fit_converges_and_satisfies_gauge(tmp_path):
    dataset = random_dataset()
    config = AnalysisConfig(
        candidate_mode="exhaustive",
        triple_min_count=0,
        min_rest_size=0,
        memory_limit_mb=64,
    )
    bundle = fit_generated_candidates(dataset, config=config, cache_dir=tmp_path)
    assert bundle.fit2.converged
    assert bundle.fit3.converged
    assert abs(bundle.fit2.weighted_gauge_mean) < 1e-10
    assert abs(bundle.fit3.weighted_gauge_mean) < 1e-10
    assert len(bundle.fit2.theta) == 15
    assert len(bundle.fit3.theta) == 20
