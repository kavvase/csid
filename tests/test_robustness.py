import numpy as np

from csid.analysis import AnalysisConfig
from csid.data import SparseBasketDataset
from csid.robustness import benchmark_candidate_scaling


def test_candidate_scaling_writes_stage_metrics(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.binomial(1, 0.45, size=(120, 6))
    X[X.sum(axis=1) < 2, :2] = 1
    dataset = SparseBasketDataset.from_dense(X, name="scale-toy")
    result = benchmark_candidate_scaling(
        {6: dataset},
        config=AnalysisConfig(
            candidate_mode="auto",
            triple_min_count=0,
            min_rest_size=1,
        ),
        output_dir=tmp_path,
    )

    assert set(result["mode"]) == {"exhaustive", "sparse"}
    assert (result["total_seconds"] >= 0).all()
    assert (result["storage_megabytes"] > 0).all()
    assert (tmp_path / "candidate_scaling.csv").exists()
