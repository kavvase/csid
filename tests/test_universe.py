from pathlib import Path

import numpy as np
import pandas as pd

from csid.analysis import AnalysisConfig
from csid.data import SparseBasketDataset, load_instacart_universes
from csid.universe import (
    build_item_universe_coefficient_comparisons,
    run_item_universe_expansion,
)


def _nested_toy(seed: int = 31):
    rng = np.random.default_rng(seed)
    n, c, focus = 1400, 10, 6
    X = np.zeros((n, c), dtype=np.uint8)
    for row in range(n):
        focal_size = int(rng.integers(2, 6))
        X[row, rng.choice(focus, size=focal_size, replace=False)] = 1
        for item in range(focus, c):
            if rng.random() < 0.22 + 0.04 * (item - focus):
                X[row, item] = 1
    periods = np.where(np.arange(n) < n // 2, "H1", "H2")
    clusters = np.asarray([f"u{row // 4}" for row in range(n)])
    full = SparseBasketDataset.from_dense(
        X,
        name="nested-toy",
        category_names=[f"c{i}" for i in range(c)],
        periods=periods,
        cluster_ids=clusters,
    )
    return {
        "6": full.subset_columns(range(6), name="nested-6"),
        "8": full.subset_columns(range(8), name="nested-8"),
        "all": full,
    }


def test_item_universe_expansion_end_to_end(tmp_path):
    result = run_item_universe_expansion(
        _nested_toy(),
        tmp_path,
        focus_n=6,
        triple_min_count=0,
        config=AnalysisConfig(
            epsilon=0.5,
            ridge2=1.0,
            ridge3=1.0,
            min_rest_size=2,
            memory_limit_mb=64,
            block_size=1000,
        ),
        primary_z=0.0,
        ccc_thresholds=(0.0, 1.0),
    )
    assert (tmp_path / "item_universe_expansion.png").exists()
    assert (tmp_path / "item_universe_pair_comparison.csv").exists()
    assert (tmp_path / "item_universe_triple_comparison.csv").exists()
    assert (tmp_path / "item_universe_coefficient_stability.csv").exists()
    assert (tmp_path / "item_universe_transfer_primary.csv").exists()
    assert (tmp_path / "item_universe_deprojection_stability.csv").exists()
    assert not result.transfer_primary.empty
    reference = result.coefficient_stability[
        result.coefficient_stability["universe"].eq("all")
        & result.coefficient_stability["filter"].eq("all")
    ]
    assert np.allclose(reference["pearson"], 1.0)
    assert set(result.runtime["context_categories"]) == {6, 8, 10}


def test_instacart_universe_loader_preserves_baskets(tmp_path):
    root = Path(tmp_path)
    aisles = pd.DataFrame(
        {"aisle_id": np.arange(1, 9), "aisle": [f"aisle-{i}" for i in range(1, 9)]}
    )
    products = pd.DataFrame(
        {"product_id": np.arange(101, 109), "aisle_id": np.arange(1, 9)}
    )
    orders = []
    prior = []
    order_id = 1
    for user in range(1, 9):
        for order_number in (1, 2, 11):
            orders.append(
                {
                    "order_id": order_id,
                    "user_id": user,
                    "eval_set": "prior",
                    "order_number": order_number,
                }
            )
            aisle_ids = [1, 2, 3 + (user + order_number) % 2, 5 + user % 4]
            for aisle_id in aisle_ids:
                prior.append(
                    {"order_id": order_id, "product_id": 100 + int(aisle_id)}
                )
            order_id += 1
    aisles.to_csv(root / "aisles.csv", index=False)
    products.to_csv(root / "products.csv", index=False)
    pd.DataFrame(orders).to_csv(root / "orders.csv", index=False)
    pd.DataFrame(prior).to_csv(root / "order_products__prior.csv", index=False)

    collection = load_instacart_universes(
        root,
        n_users=8,
        seed=7,
        focus_n=4,
        universe_sizes=(4, 6, "all"),
        chunksize=20,
        min_focus_basket_size=2,
    )
    datasets = collection.as_dict()
    assert list(datasets) == ["4", "6", "all"]
    ids = [np.asarray(dataset.basket_ids, dtype=object) for dataset in datasets.values()]
    assert all(np.array_equal(ids[0], value) for value in ids[1:])
    assert all(dataset.n_baskets == datasets["4"].n_baskets for dataset in datasets.values())
    assert (datasets["4"].X != datasets["all"].X[:, :4]).nnz == 0


def test_item_universe_coefficient_comparisons():
    coefficients = pd.DataFrame(
        {
            "universe": ["20", "all", "20", "all", "20", "all"],
            "period": ["full"] * 6,
            "order": [2, 2, 2, 2, 3, 3],
            "candidate": ["0|1", "0|1", "0|2", "0|2", "0|1|2", "0|1|2"],
            "theta": [0.5, 0.55, -0.2, -0.15, 0.3, 0.25],
            "mu": [10.0, 10.0, 5.0, 5.0, 8.0, 9.0],
            "z": [1.0, 1.1, -0.5, -0.4, 2.0, 2.2],
        }
    )
    pairs, triples = build_item_universe_coefficient_comparisons(
        coefficients,
        small_universe="20",
        large_universe="all",
    )
    assert len(pairs) == 2
    assert len(triples) == 1
    assert np.isclose(float(np.corrcoef(pairs["theta_small"], pairs["theta_large"])[0, 1]), 1.0)
