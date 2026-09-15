"""Paper reproduction pipeline.

One dataset run produces every dataset-specific quantity used in the manuscript:
CSID-2/3 estimates, H1/H2 stability, continuous H2-contrast agreement,
conditional-odds examples, third-order pair deprojection, and the Instacart
item-universe robustness audit.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import AnalysisConfig, DatasetAnalysis, fit_periods
from .concordance import ConcordanceResult, run_concordance_validation
from .correction import (
    corrected_pair_fit,
    correction_transfer_summary,
    pair_correction_table,
)
from .data import SparseBasketDataset, dataset_fingerprint, load_instacart_universes
from .estimator import CSIDFit, fit_to_dataframe
from .pseudolikelihood import (
    PseudolikelihoodFit,
    fit_cardinality_pseudolikelihood,
    pseudolikelihood_triples_frame,
)
from .robustness import benchmark_candidate_scaling
from .universe import run_item_universe_expansion
from .validation import stability_loco, temporal_stability


@dataclass(frozen=True)
class PaperConfig:
    analysis: AnalysisConfig
    ccc_thresholds: tuple[float, ...] = tuple(np.arange(0.0, 4.01, 0.5))
    primary_z: float = 2.0
    ccc_permutations: int = 199
    conditional_examples: int = 8
    seed: int = 42
    pl_ridge_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    pl_validation_fraction: float = 0.2
    pl_max_iter: int = 500
    benchmark_information_fractions: tuple[float, ...] = (1.0, 0.75, 0.50, 0.25, 0.10)
    benchmark_primary_fraction: float = 0.25


@dataclass(frozen=True)
class ItemUniverseAuditOptions:
    n_users: int = 30_000
    focus_n: int = 20
    universe_sizes: tuple[int | str, ...] = (20, 40, 80, "all")
    triple_min_count: int = 30
    chunksize: int = 2_000_000
    min_focus_basket_size: int = 2
    exponent_clip: float = 20.0


@dataclass(frozen=True)
class DatasetResult:
    analysis: DatasetAnalysis
    stability: pd.DataFrame
    stability_loco: pd.DataFrame
    concordance: ConcordanceResult
    pair_corrections: pd.DataFrame
    correction_transfer: pd.DataFrame
    pseudolikelihood: PseudolikelihoodFit
    output_dir: Path


def _exhaustive_universe_analysis(config: PaperConfig, *, triple_min_count: int) -> AnalysisConfig:
    base = config.analysis
    return AnalysisConfig(
        candidate_mode="exhaustive",
        triple_min_count=triple_min_count,
        epsilon=base.epsilon,
        ridge2=base.ridge2,
        ridge3=base.ridge3,
        min_rest_size=base.min_rest_size,
        memory_limit_mb=base.memory_limit_mb,
        block_size=base.block_size,
    )


def _beta_frame(fit: CSIDFit) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rest_size": fit.table.r_values,
            "beta": fit.beta,
            "information": np.asarray(fit.table.weights).sum(axis=0),
        }
    )


def _top_table(frame: pd.DataFrame, value: str, n: int = 10) -> pd.DataFrame:
    positive = frame.nlargest(n, value).assign(direction="positive")
    negative = frame.nsmallest(n, value).assign(direction="negative")
    return pd.concat([positive, negative], ignore_index=True)


def _metadata(
    dataset: SparseBasketDataset,
    analysis: DatasetAnalysis,
    config: PaperConfig,
    pseudolikelihood: PseudolikelihoodFit,
) -> dict[str, object]:
    sizes = dataset.basket_sizes
    return {
        "dataset": dataset.name,
        "fingerprint_sha256": dataset_fingerprint(dataset),
        "baskets": dataset.n_baskets,
        "h1_baskets": analysis.h1.n_baskets,
        "h2_baskets": analysis.h2.n_baskets,
        "categories": dataset.n_categories,
        "mean_basket_size": float(np.mean(sizes)),
        "min_basket_size": int(np.min(sizes)),
        "max_basket_size": int(np.max(sizes)),
        "full_pair_candidates": int(len(analysis.full.fit2.candidates)),
        "full_triple_candidates": int(len(analysis.full.fit3.candidates)),
        "full_evaluation_triple_candidates": int(len(analysis.full_evaluation_fit.fit3.candidates)),
        "h1_triple_candidates": int(len(analysis.h1_fit.fit3.candidates)),
        "h2_triple_candidates": int(len(analysis.h2_fit.fit3.candidates)),
        "analysis_config": asdict(config.analysis),
        "paper_config": {
            **asdict(config),
            "analysis": asdict(config.analysis),
        },
        "pseudolikelihood": {
            "selected_ridge": pseudolikelihood.selected_ridge,
            "validation_log_loss": pseudolikelihood.validation_log_loss,
            "converged": pseudolikelihood.converged,
            "symmetrization": "incident-node mean followed by H1-information-weighted centering",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def _summary_markdown(
    dataset: SparseBasketDataset,
    stability: pd.DataFrame,
    concordance: ConcordanceResult,
    corrections: pd.DataFrame,
    transfer: pd.DataFrame,
) -> str:
    primary = concordance.primary.copy()
    if not primary.empty and not concordance.permutation_summary.empty:
        primary = primary.assign(
            permutation_p=float(concordance.permutation_summary.iloc[0]["one_sided_mc_p"])
        )
    lines = [
        f"# {dataset.name}: CSID paper reproduction summary",
        "",
        "## H1/H2 stability",
        "",
        stability.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Continuous H2 contrast agreement",
        "",
        primary.to_markdown(index=False, floatfmt=".4g") if not primary.empty else "_No rows._",
        "",
        "### CCC decomposition",
        "",
        concordance.decomposition.to_markdown(index=False, floatfmt=".4g"),
        "",
        "### External cardinality-aware pseudolikelihood benchmark",
        "",
        (
            concordance.benchmark_primary.to_markdown(index=False, floatfmt=".4g")
            if not concordance.benchmark_primary.empty
            else "_No rows._"
        ),
        "",
        "### H1-only full transfer",
        "",
        (
            concordance.full_transfer.to_markdown(index=False, floatfmt=".4g")
            if not concordance.full_transfer.empty
            else "_No rows._"
        ),
        "",
        "## Conditional-odds examples",
        "",
        concordance.examples.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Largest pair deprojections",
        "",
        corrections.head(10).to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Cross-period pair deprojection",
        "",
        transfer.to_markdown(index=False, floatfmt=".4g"),
        "",
    ]
    return "\n".join(lines)


def run_dataset(
    dataset: SparseBasketDataset,
    output_dir: str | Path,
    *,
    config: PaperConfig,
) -> DatasetResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    analysis = fit_periods(dataset, config=config.analysis, cache_dir=output / "cache")
    pseudolikelihood = fit_cardinality_pseudolikelihood(
        analysis.h1,
        pair_candidates=analysis.h1_fit.counts.pair_candidates,
        triple_candidates=analysis.h1_fit.counts.triple_candidates,
        gauge_weights=analysis.h1_fit.fit3.gauge_weights,
        ridge_grid=config.pl_ridge_grid,
        validation_fraction=config.pl_validation_fraction,
        seed=config.seed,
        max_iter=config.pl_max_iter,
    )
    pseudolikelihood_triples_frame(
        pseudolikelihood,
        category_names=dataset.category_names,
    ).to_csv(output / "pseudolikelihood_h1_triples.csv", index=False)
    pseudolikelihood.diagnostics.to_csv(
        output / "pseudolikelihood_diagnostics.csv",
        index=False,
    )

    singleton_support = np.asarray(dataset.X.mean(axis=0)).ravel()
    full_pair = fit_to_dataframe(
        analysis.full.fit2,
        category_names=dataset.category_names,
        singleton_support=singleton_support,
    )
    full_triple = fit_to_dataframe(
        analysis.full.fit3,
        category_names=dataset.category_names,
        singleton_support=singleton_support,
    )
    h1_triple = fit_to_dataframe(
        analysis.h1_fit.fit3,
        category_names=dataset.category_names,
        singleton_support=np.asarray(analysis.h1.X.mean(axis=0)).ravel(),
    )
    h2_triple = fit_to_dataframe(
        analysis.h2_fit.fit3,
        category_names=dataset.category_names,
        singleton_support=np.asarray(analysis.h2.X.mean(axis=0)).ravel(),
    )
    full_pair.to_csv(output / "csid2_pairs.csv", index=False)
    full_triple.to_csv(output / "csid3_triples.csv", index=False)
    h1_triple.to_csv(output / "csid3_h1.csv", index=False)
    h2_triple.to_csv(output / "csid3_h2.csv", index=False)
    _beta_frame(analysis.full.fit2).to_csv(output / "beta2.csv", index=False)
    _beta_frame(analysis.full.fit3).to_csv(output / "beta3.csv", index=False)
    _top_table(full_pair, "J_hat").to_csv(output / "top_csid2.csv", index=False)
    _top_table(full_triple, "theta_hat").to_csv(output / "top_csid3.csv", index=False)

    stability, aligned = temporal_stability(analysis.h1_fit.fit3, analysis.h2_fit.fit3)
    stability_sensitivity = stability_loco(
        aligned, n_categories=dataset.n_categories, information_fraction=0.25
    )
    stability.to_csv(output / "temporal_stability.csv", index=False)
    aligned.to_csv(output / "temporal_stability_candidates.csv", index=False)
    stability_sensitivity.to_csv(output / "temporal_stability_loco.csv", index=False)

    concordance = run_concordance_validation(
        analysis.h1_fit.fit3,
        analysis.h2_fit.fit3,
        analysis.h1_fit.fit2,
        category_names=dataset.category_names,
        thresholds=config.ccc_thresholds,
        primary_threshold=config.primary_z,
        permutations=config.ccc_permutations,
        seed=config.seed,
        n_examples=config.conditional_examples,
        min_rest_size=config.analysis.min_rest_size,
        external_triple_candidates=pseudolikelihood.triple_candidates,
        external_theta=pseudolikelihood.triple_theta,
        benchmark_fractions=config.benchmark_information_fractions,
        benchmark_primary_fraction=config.benchmark_primary_fraction,
    )
    concordance.layers.to_csv(output / "concordance_layers.csv", index=False)
    concordance.sweep.to_csv(output / "concordance_threshold_sweep.csv", index=False)
    concordance.primary.to_csv(output / "concordance_primary.csv", index=False)
    concordance.decomposition.to_csv(output / "concordance_decomposition.csv", index=False)
    concordance.loco.to_csv(output / "concordance_loco.csv", index=False)
    concordance.permutations.to_csv(output / "concordance_permutations.csv", index=False)
    concordance.permutation_summary.to_csv(output / "concordance_permutation_summary.csv", index=False)
    concordance.examples.to_csv(output / "conditional_odds_examples.csv", index=False)
    concordance.benchmark_sweep.to_csv(
        output / "external_benchmark_sweep.csv",
        index=False,
    )
    concordance.benchmark_primary.to_csv(
        output / "external_benchmark_primary.csv",
        index=False,
    )
    concordance.benchmark_loco.to_csv(
        output / "external_benchmark_loco.csv",
        index=False,
    )
    concordance.full_transfer.to_csv(
        output / "full_transfer_summary.csv",
        index=False,
    )

    corrected = corrected_pair_fit(
        analysis.dataset,
        analysis.full.counts,
        analysis.full.fit2,
        analysis.full.fit3,
        epsilon=config.analysis.epsilon,
        min_rest_size=config.analysis.min_rest_size,
        ridge=config.analysis.ridge2,
    ).corrected_fit
    corrections = pair_correction_table(
        analysis.full.fit2, corrected, category_names=dataset.category_names
    )
    corrections.to_csv(output / "pair_deprojection.csv", index=False)
    transfer = correction_transfer_summary(
        full_dataset=analysis.dataset,
        h2_dataset=analysis.h2,
        full_counts=analysis.full_evaluation_fit.counts,
        h2_counts=analysis.h2_fit.counts,
        full_pair_fit=analysis.full_evaluation_fit.fit2,
        h2_pair_fit=analysis.h2_fit.fit2,
        full_triple_fit=analysis.full_evaluation_fit.fit3,
        h1_triple_fit=analysis.h1_fit.fit3,
        h2_triple_fit=analysis.h2_fit.fit3,
        epsilon=config.analysis.epsilon,
        min_rest_size=config.analysis.min_rest_size,
        ridge2=config.analysis.ridge2,
    )
    transfer.to_csv(output / "pair_deprojection_transfer.csv", index=False)

    (output / "metadata.json").write_text(
        json.dumps(_metadata(dataset, analysis, config, pseudolikelihood), indent=2),
        encoding="utf-8",
    )
    (output / "summary.md").write_text(
        _summary_markdown(dataset, stability, concordance, corrections, transfer),
        encoding="utf-8",
    )
    return DatasetResult(
        analysis=analysis,
        stability=stability,
        stability_loco=stability_sensitivity,
        concordance=concordance,
        pair_corrections=corrections,
        correction_transfer=transfer,
        pseudolikelihood=pseudolikelihood,
        output_dir=output,
    )


def run_item_universe_audit(
    output_dir: str | Path,
    instacart_root: str | Path,
    *,
    config: PaperConfig,
    options: ItemUniverseAuditOptions | None = None,
) -> None:
    """Instacart item-universe robustness audit used by the paper pipeline."""

    opts = options or ItemUniverseAuditOptions()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    collection = load_instacart_universes(
        instacart_root,
        n_users=opts.n_users,
        seed=config.seed,
        focus_n=opts.focus_n,
        universe_sizes=opts.universe_sizes,
        chunksize=opts.chunksize,
        min_focus_basket_size=opts.min_focus_basket_size,
    )
    collection.category_table.to_csv(output / "item_universe_categories.csv", index=False)
    analysis = _exhaustive_universe_analysis(config, triple_min_count=opts.triple_min_count)
    run_item_universe_expansion(
        collection.as_dict(),
        output,
        focus_n=opts.focus_n,
        triple_min_count=opts.triple_min_count,
        config=analysis,
        primary_z=config.primary_z,
        ccc_thresholds=config.ccc_thresholds,
        exponent_clip=opts.exponent_clip,
    )
    scaling_datasets = {
        dataset.n_categories if size == "all" else int(size): dataset
        for size, dataset in collection.as_dict().items()
    }
    benchmark_candidate_scaling(
        scaling_datasets,
        config=analysis,
        output_dir=output / "candidate-scaling",
    )
