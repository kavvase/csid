"""Core CSID fitting pipeline.

Builds candidate sets, counts size-stratified cells, and fits CSID-2/CSID-3
for the full sample and the H1/H2 split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .candidates import CandidateReport, generate_triple_candidates, pair_candidates_for_model
from .contrasts import ContrastMatrix, build_contrast_matrix
from .counts import (
    PairScan,
    SizeStratifiedCounts,
    count_for_candidates,
    filter_triples_by_count,
    scan_pairs,
)
from .data import SparseBasketDataset
from .estimator import CSIDFit, fit_csid


@dataclass(frozen=True)
class AnalysisConfig:
    candidate_mode: str = "exhaustive"
    exhaustive_max_categories: int = 40
    pair_candidate_min_count: int = 5
    pair_model_min_count: int = 0
    triple_min_count: int = 0
    top_l_neighbors: int = 20
    max_candidates: int | None = 250_000
    epsilon: float = 0.5
    ridge2: float = 1.0
    ridge3: float = 1.0
    min_rest_size: int = 2
    memory_limit_mb: float = 512.0
    block_size: int = 50_000


@dataclass(frozen=True)
class FitBundle:
    scan: PairScan
    counts: SizeStratifiedCounts
    contrast2: ContrastMatrix
    contrast3: ContrastMatrix
    fit2: CSIDFit
    fit3: CSIDFit
    candidate_report: CandidateReport | None
    elapsed_seconds: float


@dataclass(frozen=True)
class DatasetAnalysis:
    dataset: SparseBasketDataset
    h1: SparseBasketDataset
    h2: SparseBasketDataset
    full: FitBundle
    full_evaluation_fit: FitBundle
    h1_fit: FitBundle
    h2_fit: FitBundle


def build_contrast_pair(
    counts: SizeStratifiedCounts,
    *,
    config: AnalysisConfig,
    work_dir: str | Path,
    split_k_dirs: bool = False,
) -> tuple[ContrastMatrix, ContrastMatrix]:
    base = Path(work_dir)
    base.mkdir(parents=True, exist_ok=True)
    contrast2 = build_contrast_matrix(
        counts=counts,
        k=2,
        epsilon=config.epsilon,
        min_rest_size=config.min_rest_size,
        work_dir=base / "k2" if split_k_dirs else base,
        memory_limit_mb=config.memory_limit_mb,
    )
    contrast3 = build_contrast_matrix(
        counts=counts,
        k=3,
        epsilon=config.epsilon,
        min_rest_size=config.min_rest_size,
        work_dir=base / "k3" if split_k_dirs else base,
        memory_limit_mb=config.memory_limit_mb,
    )
    return contrast2, contrast3


def _fit_from_counts(
    counts: SizeStratifiedCounts,
    *,
    config: AnalysisConfig,
    cache_dir: str | Path,
    gauge_weights2: np.ndarray | None = None,
    gauge_weights3: np.ndarray | None = None,
) -> tuple[ContrastMatrix, ContrastMatrix, CSIDFit, CSIDFit]:
    contrast2, contrast3 = build_contrast_pair(
        counts,
        config=config,
        work_dir=cache_dir,
    )
    fit2 = fit_csid(
        contrast2,
        ridge=config.ridge2,
        block_size=config.block_size,
        gauge_weights=gauge_weights2,
    )
    fit3 = fit_csid(
        contrast3,
        ridge=config.ridge3,
        block_size=config.block_size,
        gauge_weights=gauge_weights3,
    )
    return contrast2, contrast3, fit2, fit3


def fit_generated_candidates(
    dataset: SparseBasketDataset,
    *,
    config: AnalysisConfig,
    cache_dir: str | Path,
) -> FitBundle:
    started = perf_counter()
    scan = scan_pairs(dataset)
    triples, report = generate_triple_candidates(
        scan.pair_totals,
        n_categories=dataset.n_categories,
        mode=config.candidate_mode,
        exhaustive_max_categories=config.exhaustive_max_categories,
        pair_candidate_min_count=config.pair_candidate_min_count,
        top_l_neighbors=config.top_l_neighbors,
        max_candidates=config.max_candidates,
    )
    pairs = pair_candidates_for_model(
        scan.pair_totals,
        triples,
        n_categories=dataset.n_categories,
        min_count=config.pair_model_min_count,
    )
    counts = count_for_candidates(
        dataset,
        pair_candidates=pairs,
        triple_candidates=triples,
        base_scan=scan,
    )
    counts = filter_triples_by_count(counts, min_count=config.triple_min_count)
    if len(counts.triple_candidates) == 0:
        raise ValueError("No triple candidates remain after the support filter")
    contrast2, contrast3, fit2, fit3 = _fit_from_counts(
        counts, config=config, cache_dir=cache_dir
    )
    return FitBundle(
        scan=scan,
        counts=counts,
        contrast2=contrast2,
        contrast3=contrast3,
        fit2=fit2,
        fit3=fit3,
        candidate_report=report,
        elapsed_seconds=perf_counter() - started,
    )


def fit_fixed_candidates(
    dataset: SparseBasketDataset,
    *,
    pair_candidates: np.ndarray,
    triple_candidates: np.ndarray,
    config: AnalysisConfig,
    cache_dir: str | Path,
    gauge_weights2: np.ndarray | None = None,
    gauge_weights3: np.ndarray | None = None,
) -> FitBundle:
    started = perf_counter()
    scan = scan_pairs(dataset)
    counts = count_for_candidates(
        dataset,
        pair_candidates=pair_candidates,
        triple_candidates=triple_candidates,
        base_scan=scan,
    )
    contrast2, contrast3, fit2, fit3 = _fit_from_counts(
        counts,
        config=config,
        cache_dir=cache_dir,
        gauge_weights2=gauge_weights2,
        gauge_weights3=gauge_weights3,
    )
    return FitBundle(
        scan=scan,
        counts=counts,
        contrast2=contrast2,
        contrast3=contrast3,
        fit2=fit2,
        fit3=fit3,
        candidate_report=None,
        elapsed_seconds=perf_counter() - started,
    )


def fit_periods(
    dataset: SparseBasketDataset,
    *,
    config: AnalysisConfig,
    cache_dir: str | Path,
) -> DatasetAnalysis:
    """Fit full, H1, and H2 data.

    H1 generates the evaluation candidate set.  H2 and the full-period
    evaluation reference are recounted on the same pair/triple candidates and
    fitted with the H1 gauge weights, preventing either the candidate family or
    gauge convention from changing across evaluation fits.
    """

    cache = Path(cache_dir)
    h1 = dataset.subset_period("H1")
    h2 = dataset.subset_period("H2")
    full = fit_generated_candidates(dataset, config=config, cache_dir=cache / "full")
    h1_fit = fit_generated_candidates(h1, config=config, cache_dir=cache / "h1")
    full_evaluation_fit = fit_fixed_candidates(
        dataset,
        pair_candidates=h1_fit.counts.pair_candidates,
        triple_candidates=h1_fit.counts.triple_candidates,
        config=config,
        cache_dir=cache / "full-evaluation",
        gauge_weights2=h1_fit.fit2.gauge_weights,
        gauge_weights3=h1_fit.fit3.gauge_weights,
    )
    h2_fit = fit_fixed_candidates(
        h2,
        pair_candidates=h1_fit.counts.pair_candidates,
        triple_candidates=h1_fit.counts.triple_candidates,
        config=config,
        cache_dir=cache / "h2",
        gauge_weights2=h1_fit.fit2.gauge_weights,
        gauge_weights3=h1_fit.fit3.gauge_weights,
    )
    return DatasetAnalysis(dataset, h1, h2, full, full_evaluation_fit, h1_fit, h2_fit)
