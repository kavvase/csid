from pathlib import Path

import numpy as np

from csid.analysis import AnalysisConfig
from csid.data import SparseBasketDataset
from csid.figures import build_paper_outputs
from csid.paper import PaperConfig, run_dataset
from csid.synthetic import (
    cardinality_confounding_experiment,
    randomized_triple_power_experiment,
    saturation_deprojection_experiment,
)


def make_dataset(seed=13, n=900, c=6):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, c), dtype=np.uint8)
    for row in range(n):
        size = int(rng.integers(2, 5))
        X[row, rng.choice(c, size=size, replace=False)] = 1
    periods = np.where(np.arange(n) < n // 2, "H1", "H2")
    clusters = np.asarray([f"u{row//3}" for row in range(n)])
    return SparseBasketDataset.from_dense(
        X,
        name="toy",
        category_names=[f"c{i}" for i in range(c)],
        periods=periods,
        cluster_ids=clusters,
    )


def test_end_to_end_outputs(tmp_path):
    root = tmp_path / "results"
    config = PaperConfig(
        analysis=AnalysisConfig(
            candidate_mode="exhaustive",
            triple_min_count=0,
            min_rest_size=0,
            memory_limit_mb=64,
        ),
        ccc_thresholds=(0.0, 1.0, 2.0),
        primary_z=1.0,
        ccc_permutations=3,
        conditional_examples=4,
        seed=3,
        pl_ridge_grid=(0.1,),
        pl_validation_fraction=0.1,
        pl_max_iter=80,
    )
    result = run_dataset(make_dataset(), root / "complete-journey", config=config)
    assert (root / "complete-journey" / "csid3_triples.csv").exists()
    assert (root / "complete-journey" / "concordance_threshold_sweep.csv").exists()
    assert (root / "complete-journey" / "pair_deprojection.csv").exists()
    assert (root / "complete-journey" / "external_benchmark_primary.csv").exists()
    assert (root / "complete-journey" / "full_transfer_summary.csv").exists()
    assert not result.concordance.sweep.empty
    assert not result.concordance.benchmark_primary.empty
    assert not result.concordance.full_transfer.empty
    assert np.array_equal(
        result.analysis.h1_fit.fit3.gauge_weights,
        result.analysis.h2_fit.fit3.gauge_weights,
    )
    assert np.array_equal(
        result.analysis.h1_fit.fit3.gauge_weights,
        result.analysis.full_evaluation_fit.fit3.gauge_weights,
    )
    assert abs(result.analysis.h2_fit.fit3.weighted_gauge_mean) < 1e-10

    # Reuse the toy audit for the other paper panels.
    for name in ("ta-feng", "instacart"):
        target = root / name
        target.mkdir(parents=True)
        for source in (root / "complete-journey").iterdir():
            if source.is_file():
                (target / source.name).write_bytes(source.read_bytes())

    synthetic = root / "synthetic"
    synthetic.mkdir()
    _, cardinality = cardinality_confounding_experiment(
        cardinality_strengths=(0.0, 0.8),
        replicates=2,
        n_baskets=1800,
        n_categories=7,
        seed=9,
    )
    cardinality.to_csv(synthetic / "cardinality_confounding_summary.csv", index=False)
    _, power = randomized_triple_power_experiment(
        effect_sizes=(0.2,), replicates=2, n_baskets=1000, n_categories=7, n_planted=4, seed=5
    )
    power.to_csv(synthetic / "detection_summary.csv", index=False)
    saturation, _ = saturation_deprojection_experiment(n_samples=3000, seed=23)
    saturation.to_csv(synthetic / "saturation_summary.csv", index=False)

    manifest = build_paper_outputs(root, root / "paper")
    assert Path(manifest["figure1"]).exists()
    assert Path(manifest["figure2"]).exists()
    assert Path(manifest["figure3"]).exists()
    table6 = root / "paper" / "tables" / "table6_concordance.csv"
    table7 = root / "paper" / "tables" / "table7_concordance_decomposition.csv"
    assert table6.exists()
    assert table7.exists()
    assert "pl_ccc" in table6.read_text(encoding="utf-8").splitlines()[0]
    assert "pl_pearson" in table7.read_text(encoding="utf-8").splitlines()[0]
    assert (root / "paper" / "tables" / "table9_pair_deprojection_transfer.csv").exists()
