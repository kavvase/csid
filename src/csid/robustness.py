"""Focused robustness checks used in the manuscript supplement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .analysis import AnalysisConfig, DatasetAnalysis, build_contrast_pair, fit_periods
from .candidates import generate_triple_candidates, pair_candidates_for_model
from .concordance import (
    _benchmark_metric_row,
    add_transfer_prediction,
    build_transfer_layers,
    full_transfer_summary,
)
from .correction import correction_transfer_summary
from .counts import count_for_candidates, filter_triples_by_count, scan_pairs
from .data import SparseBasketDataset
from .estimator import fit_csid
from .validation import temporal_stability


def run_sensitivity_scenario(
    dataset: SparseBasketDataset,
    *,
    scenario: str,
    config: AnalysisConfig,
    external_candidates: np.ndarray,
    external_theta: np.ndarray,
    cache_dir: str | Path,
) -> dict[str, object]:
    """Run one CSID sensitivity scenario using a fixed external comparator."""

    analysis = fit_periods(dataset, config=config, cache_dir=cache_dir)
    return summarize_sensitivity_analysis(
        analysis,
        scenario=scenario,
        config=config,
        external_candidates=external_candidates,
        external_theta=external_theta,
    )


def summarize_sensitivity_analysis(
    analysis: DatasetAnalysis,
    *,
    scenario: str,
    config: AnalysisConfig,
    external_candidates: np.ndarray,
    external_theta: np.ndarray,
) -> dict[str, object]:
    """Summarize an existing period fit as one sensitivity row."""

    dataset = analysis.dataset
    stability, _ = temporal_stability(analysis.h1_fit.fit3, analysis.h2_fit.fit3)
    stability_row = stability[stability["filter"].eq("top25%")].iloc[0]

    layers = build_transfer_layers(
        analysis.h1_fit.fit3,
        analysis.h2_fit.fit3,
        category_names=dataset.category_names,
        min_rest_size=config.min_rest_size,
    )
    layers = add_transfer_prediction(
        layers,
        candidates=external_candidates,
        theta=external_theta,
    )
    benchmark = _benchmark_metric_row(layers, information_fraction=0.25)
    full_transfer = full_transfer_summary(layers, information_fraction=0.25).iloc[0]

    correction = correction_transfer_summary(
        full_dataset=analysis.dataset,
        h2_dataset=analysis.h2,
        full_counts=analysis.full_evaluation_fit.counts,
        h2_counts=analysis.h2_fit.counts,
        full_pair_fit=analysis.full_evaluation_fit.fit2,
        h2_pair_fit=analysis.h2_fit.fit2,
        full_triple_fit=analysis.full_evaluation_fit.fit3,
        h1_triple_fit=analysis.h1_fit.fit3,
        h2_triple_fit=analysis.h2_fit.fit3,
        epsilon=config.epsilon,
        min_rest_size=config.min_rest_size,
        ridge2=config.ridge2,
    )
    correction_row = correction[
        correction["comparison"].eq("H1 theta on H2 vs H2 theta on H2")
    ].iloc[0]
    return {
        "dataset": dataset.name,
        "scenario": scenario,
        "baskets": dataset.n_baskets,
        **asdict(config),
        "temporal_pearson_top25": float(stability_row["pearson"]),
        "csid_minus_pl_ccc": float(benchmark["csid_minus_pl_ccc"]),
        "pair_correction_pearson": float(correction_row["pearson"]),
        "full_transfer_ccc": float(full_transfer["full_csid_ccc"]),
        "partial_transfer_ccc": float(full_transfer["partial_csid_ccc"]),
    }


def write_sensitivity_results(rows: list[dict[str, object]], path: str | Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def benchmark_candidate_scaling(
    datasets: Mapping[int, SparseBasketDataset],
    *,
    config: AnalysisConfig,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Measure candidate generation, counting, contrast, and fit scaling."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    retained: dict[tuple[int, str], set[tuple[int, int, int]]] = {}
    for requested_size, dataset in sorted(datasets.items()):
        modes = ("exhaustive", "sparse") if dataset.n_categories <= 40 else ("sparse",)
        for mode in modes:
            scenario = f"{requested_size}-{mode}"
            scenario_config = replace(config, candidate_mode=mode)

            started = perf_counter()
            scan = scan_pairs(dataset)
            scan_seconds = perf_counter() - started

            started = perf_counter()
            triples, report = generate_triple_candidates(
                scan.pair_totals,
                n_categories=dataset.n_categories,
                mode=mode,
                exhaustive_max_categories=max(dataset.n_categories, 40),
                pair_candidate_min_count=scenario_config.pair_candidate_min_count,
                top_l_neighbors=scenario_config.top_l_neighbors,
                max_candidates=scenario_config.max_candidates,
            )
            pairs = pair_candidates_for_model(
                scan.pair_totals,
                triples,
                n_categories=dataset.n_categories,
                min_count=scenario_config.pair_model_min_count,
            )
            candidate_seconds = perf_counter() - started

            started = perf_counter()
            counts = count_for_candidates(
                dataset,
                pair_candidates=pairs,
                triple_candidates=triples,
                base_scan=scan,
            )
            counts = filter_triples_by_count(counts, min_count=scenario_config.triple_min_count)
            counting_seconds = perf_counter() - started

            started = perf_counter()
            contrast2, contrast3 = build_contrast_pair(
                counts,
                config=scenario_config,
                work_dir=output / "cache" / scenario,
                split_k_dirs=True,
            )
            contrast_seconds = perf_counter() - started

            started = perf_counter()
            fit_csid(
                contrast2,
                ridge=scenario_config.ridge2,
                block_size=scenario_config.block_size,
            )
            fit_csid(
                contrast3,
                ridge=scenario_config.ridge3,
                block_size=scenario_config.block_size,
            )
            fit_seconds = perf_counter() - started

            retained[(dataset.n_categories, mode)] = {
                (int(triple[0]), int(triple[1]), int(triple[2]))
                for triple in counts.triple_candidates
            }
            storage_bytes = int(
                counts.pair_by_size.nbytes
                + counts.triple_by_size.nbytes
                + counts.singleton_by_size.nbytes
                + contrast2.weights.nbytes
                + contrast2.lambda_hat.nbytes
                + contrast3.weights.nbytes
                + contrast3.lambda_hat.nbytes
            )
            rows.append(
                {
                    "requested_categories": requested_size,
                    "categories": dataset.n_categories,
                    "mode": mode,
                    "baskets": dataset.n_baskets,
                    "generated_triples": report.candidates_after_cap,
                    "retained_triples": len(counts.triple_candidates),
                    "pairs": len(counts.pair_candidates),
                    "scan_seconds": scan_seconds,
                    "candidate_seconds": candidate_seconds,
                    "counting_seconds": counting_seconds,
                    "contrast_seconds": contrast_seconds,
                    "fit_seconds": fit_seconds,
                    "total_seconds": (
                        scan_seconds
                        + candidate_seconds
                        + counting_seconds
                        + contrast_seconds
                        + fit_seconds
                    ),
                    "storage_megabytes": storage_bytes / (1024.0**2),
                    "sparse_recall_vs_exhaustive": np.nan,
                }
            )

    frame = pd.DataFrame(rows)
    for categories in (20, 40):
        exhaustive = retained.get((categories, "exhaustive"), set())
        sparse = retained.get((categories, "sparse"), set())
        if exhaustive:
            recall = len(exhaustive.intersection(sparse)) / len(exhaustive)
            frame.loc[
                (frame["categories"] == categories) & frame["mode"].eq("sparse"),
                "sparse_recall_vs_exhaustive",
            ] = recall
    frame.to_csv(output / "candidate_scaling.csv", index=False)
    return frame
