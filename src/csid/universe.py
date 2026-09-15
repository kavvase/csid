"""Robustness audit for expanding the surrounding item universe.

The focal pair and triple candidates are kept inside a fixed top-ranked item
set.  Only the surrounding context used to define rest cardinality is expanded.
All universe sizes use the same baskets, candidate sets, and gauge weights, so
coefficient changes reflect the enlarged context rather than sample or centering
changes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .analysis import AnalysisConfig, build_contrast_pair
from .concordance import build_transfer_layers, threshold_sweep, weighted_ccc
from .contrasts import ContrastMatrix
from .correction import corrected_pair_fit, correction_delta
from .counts import SizeStratifiedCounts, count_for_candidates, scan_pairs
from .data import SparseBasketDataset
from .estimator import CSIDFit, fit_csid
from .stats import pearson_spearman


@dataclass(frozen=True)
class UniverseLevel:
    label: str
    dataset: SparseBasketDataset
    full_counts: SizeStratifiedCounts
    h1_counts: SizeStratifiedCounts
    h2_counts: SizeStratifiedCounts
    full_contrast2: ContrastMatrix
    full_contrast3: ContrastMatrix
    h1_contrast2: ContrastMatrix
    h1_contrast3: ContrastMatrix
    h2_contrast2: ContrastMatrix
    h2_contrast3: ContrastMatrix
    build_seconds: float


@dataclass(frozen=True)
class UniverseFittedLevel:
    source: UniverseLevel
    full_fit2: CSIDFit
    full_fit3: CSIDFit
    h1_fit2: CSIDFit
    h1_fit3: CSIDFit
    h2_fit2: CSIDFit
    h2_fit3: CSIDFit
    fit_seconds: float


@dataclass(frozen=True)
class ItemUniverseResult:
    coefficient_stability: pd.DataFrame
    transfer_sweep: pd.DataFrame
    transfer_primary: pd.DataFrame
    deprojection_stability: pd.DataFrame
    runtime: pd.DataFrame
    coefficients: pd.DataFrame
    output_dir: Path


def _candidate_arrays(focus_n: int) -> tuple[np.ndarray, np.ndarray]:
    if focus_n < 3:
        raise ValueError("focus_n must be at least 3")
    pairs = np.asarray(list(combinations(range(int(focus_n)), 2)), dtype=np.int32)
    triples = np.asarray(list(combinations(range(int(focus_n)), 3)), dtype=np.int32)
    return pairs.reshape(-1, 2), triples.reshape(-1, 3)


def _count_and_contrast(
    dataset: SparseBasketDataset,
    *,
    pairs: np.ndarray,
    triples: np.ndarray,
    config: AnalysisConfig,
    cache_dir: Path,
) -> tuple[SizeStratifiedCounts, ContrastMatrix, ContrastMatrix]:
    scan = scan_pairs(dataset, count_pair_totals=False)
    counts = count_for_candidates(
        dataset,
        pair_candidates=pairs,
        triple_candidates=triples,
        base_scan=scan,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    contrast2, contrast3 = build_contrast_pair(
        counts,
        config=config,
        work_dir=cache_dir,
        split_k_dirs=True,
    )
    return counts, contrast2, contrast3


def _build_level(
    label: str,
    dataset: SparseBasketDataset,
    *,
    pairs: np.ndarray,
    triples: np.ndarray,
    config: AnalysisConfig,
    cache_dir: Path,
) -> UniverseLevel:
    started = perf_counter()
    h1 = dataset.subset_period("H1")
    h2 = dataset.subset_period("H2")
    full_counts, full_c2, full_c3 = _count_and_contrast(
        dataset,
        pairs=pairs,
        triples=triples,
        config=config,
        cache_dir=cache_dir / "full",
    )
    h1_counts, h1_c2, h1_c3 = _count_and_contrast(
        h1,
        pairs=pairs,
        triples=triples,
        config=config,
        cache_dir=cache_dir / "h1",
    )
    h2_counts, h2_c2, h2_c3 = _count_and_contrast(
        h2,
        pairs=pairs,
        triples=triples,
        config=config,
        cache_dir=cache_dir / "h2",
    )
    return UniverseLevel(
        label=str(label),
        dataset=dataset,
        full_counts=full_counts,
        h1_counts=h1_counts,
        h2_counts=h2_counts,
        full_contrast2=full_c2,
        full_contrast3=full_c3,
        h1_contrast2=h1_c2,
        h1_contrast3=h1_c3,
        h2_contrast2=h2_c2,
        h2_contrast3=h2_c3,
        build_seconds=perf_counter() - started,
    )


def _fit_level(
    level: UniverseLevel,
    *,
    config: AnalysisConfig,
    pair_full_gauge: np.ndarray,
    triple_full_gauge: np.ndarray,
    pair_period_gauge: np.ndarray,
    triple_period_gauge: np.ndarray,
) -> UniverseFittedLevel:
    started = perf_counter()
    kwargs = {"block_size": config.block_size}
    return UniverseFittedLevel(
        source=level,
        full_fit2=fit_csid(
            level.full_contrast2,
            ridge=config.ridge2,
            gauge_weights=pair_full_gauge,
            **kwargs,
        ),
        full_fit3=fit_csid(
            level.full_contrast3,
            ridge=config.ridge3,
            gauge_weights=triple_full_gauge,
            **kwargs,
        ),
        h1_fit2=fit_csid(
            level.h1_contrast2,
            ridge=config.ridge2,
            gauge_weights=pair_period_gauge,
            **kwargs,
        ),
        h1_fit3=fit_csid(
            level.h1_contrast3,
            ridge=config.ridge3,
            gauge_weights=triple_period_gauge,
            **kwargs,
        ),
        h2_fit2=fit_csid(
            level.h2_contrast2,
            ridge=config.ridge2,
            gauge_weights=pair_period_gauge,
            **kwargs,
        ),
        h2_fit3=fit_csid(
            level.h2_contrast3,
            ridge=config.ridge3,
            gauge_weights=triple_period_gauge,
            **kwargs,
        ),
        fit_seconds=perf_counter() - started,
    )


def _top_overlap(a: np.ndarray, b: np.ndarray, n: int, *, largest: bool) -> int:
    n = min(int(n), len(a), len(b))
    if n <= 0:
        return 0
    order_a = np.argsort(a)
    order_b = np.argsort(b)
    set_a = set((order_a[-n:] if largest else order_a[:n]).tolist())
    set_b = set((order_b[-n:] if largest else order_b[:n]).tolist())
    return int(len(set_a.intersection(set_b)))


def _agreement_row(
    current: CSIDFit,
    reference: CSIDFit,
    mask: np.ndarray,
) -> dict[str, object]:
    keep = np.asarray(mask, dtype=bool)
    a = np.asarray(current.theta, dtype=float)[keep]
    b = np.asarray(reference.theta, dtype=float)[keep]
    weights = np.asarray(reference.gauge_weights, dtype=float)[keep]
    pearson, spearman = pearson_spearman(a, b)
    ccc = weighted_ccc(b, a, weights)["ccc"]
    error = a - b
    return {
        "n": int(len(a)),
        "pearson": pearson,
        "spearman": spearman,
        "ccc": float(ccc),
        "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))) if len(a) else float("nan"),
        "mae": float(np.mean(np.abs(error))) if len(a) else float("nan"),
        "rmse": float(np.sqrt(np.mean(error * error))) if len(a) else float("nan"),
        "top20_positive_overlap": _top_overlap(a, b, 20, largest=True),
        "top20_negative_overlap": _top_overlap(a, b, 20, largest=False),
    }


def _reference_masks(fit: CSIDFit) -> dict[str, np.ndarray]:
    n = len(fit.theta)
    mu = np.asarray(fit.mu, dtype=float)
    result = {"all": np.ones(n, dtype=bool)}
    if n:
        result["top50%"] = mu >= float(np.quantile(mu, 0.50))
        result["top25%"] = mu >= float(np.quantile(mu, 0.75))
        result["abs_z_ge_2"] = np.abs(np.asarray(fit.z, dtype=float)) >= 2.0
    return result


def _coefficient_frame(
    fitted: Mapping[str, UniverseFittedLevel],
    *,
    focus_names: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, level in fitted.items():
        for period, pair_fit, triple_fit in (
            ("full", level.full_fit2, level.full_fit3),
            ("H1", level.h1_fit2, level.h1_fit3),
            ("H2", level.h2_fit2, level.h2_fit3),
        ):
            for fit in (pair_fit, triple_fit):
                for q, candidate in enumerate(fit.candidates):
                    items = tuple(map(int, candidate))
                    row: dict[str, object] = {
                        "universe": label,
                        "context_categories": level.source.dataset.n_categories,
                        "period": period,
                        "order": fit.k,
                        "candidate": "|".join(map(str, items)),
                        "theta": float(fit.theta[q]),
                        "mu": float(fit.mu[q]),
                        "se": float(fit.se[q]),
                        "z": float(fit.z[q]),
                    }
                    for p, item in enumerate(items, start=1):
                        row[f"item_{p}_idx"] = item
                        row[f"item_{p}"] = str(focus_names[item])
                    rows.append(row)
    return pd.DataFrame(rows)


def build_item_universe_coefficient_comparisons(
    coefficients: pd.DataFrame,
    *,
    small_universe: str = "20",
    large_universe: str = "all",
    period: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge focal coefficients from the small and full aisle universes."""

    base = coefficients[
        coefficients["period"].astype(str).eq(period)
        & coefficients["universe"].isin([small_universe, large_universe])
    ].copy()
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()

    def merge_order(order: int) -> pd.DataFrame:
        subset = base[base["order"].eq(order)]
        small = subset[subset["universe"].astype(str).eq(small_universe)][
            ["candidate", "theta", "mu", "z"]
        ].rename(columns={"theta": "theta_small", "mu": "mu_small", "z": "z_small"})
        large = subset[subset["universe"].astype(str).eq(large_universe)][
            ["candidate", "theta", "mu", "z"]
        ].rename(columns={"theta": "theta_large", "mu": "mu_large", "z": "z_large"})
        if small.empty or large.empty:
            return pd.DataFrame()
        return small.merge(large, on="candidate", how="inner")

    pair_comparison = merge_order(2)
    triple_comparison = merge_order(3)
    if not triple_comparison.empty:
        cutoff = float(np.quantile(triple_comparison["mu_large"].to_numpy(float), 0.75))
        triple_comparison = triple_comparison[triple_comparison["mu_large"] >= cutoff].copy()
    return pair_comparison.reset_index(drop=True), triple_comparison.reset_index(drop=True)


def _scatter_panel(
    ax,
    frame: pd.DataFrame,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    x = frame["theta_large"].to_numpy(dtype=float)
    y = frame["theta_small"].to_numpy(dtype=float)
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")
    ax.scatter(x, y, s=28, alpha=0.65, edgecolors="none")
    if len(x):
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = max(0.05, 0.04 * (hi - lo if hi > lo else 1.0))
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, linestyle="--", linewidth=1.0, color="0.35")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(alpha=0.25)
    ax.text(
        0.04,
        0.96,
        rf"$r={pearson:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )


def plot_item_universe_coefficient_scatter(
    pair_comparison: pd.DataFrame,
    triple_comparison: pd.DataFrame,
    path: str | Path,
    *,
    small_label: str = "20",
    large_label: str = "all",
) -> None:
    """Scatter-compare focal coefficients between two context sizes."""

    if pair_comparison.empty or triple_comparison.empty:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), squeeze=False)
    _scatter_panel(
        axes[0, 0],
        pair_comparison,
        xlabel=rf"$\hat{{J}}_{{ij}}^{{({large_label})}}$",
        ylabel=rf"$\hat{{J}}_{{ij}}^{{({small_label})}}$",
        title=rf"Focal pair coefficients ($n={len(pair_comparison)}$)",
    )
    _scatter_panel(
        axes[0, 1],
        triple_comparison,
        xlabel=rf"$\hat{{\theta}}_T^{{({large_label})}}$",
        ylabel=rf"$\hat{{\theta}}_T^{{({small_label})}}$",
        title=rf"Top 25% information triples ($n={len(triple_comparison)}$)",
    )
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    if output.suffix.lower() == ".png":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_item_universe_expansion(
    universes: Mapping[str, SparseBasketDataset],
    output_dir: str | Path,
    *,
    focus_n: int = 20,
    triple_min_count: int = 30,
    config: AnalysisConfig | None = None,
    primary_z: float = 2.0,
    ccc_thresholds: Sequence[float] = (0.0, 1.0, 2.0, 3.0, 4.0),
    exponent_clip: float = 20.0,
) -> ItemUniverseResult:
    """Run the fixed-focal-set item-universe expansion audit.

    The smallest universe must contain ``focus_n`` categories and all supplied
    datasets must have identical basket IDs, periods, and focal columns.  The
    largest universe is the reference for common gauge weights and stability
    metrics.
    """

    if not universes:
        raise ValueError("At least one universe dataset is required")
    config = AnalysisConfig() if config is None else config
    ordered = sorted(universes.items(), key=lambda item: item[1].n_categories)
    labels = [label for label, _ in ordered]
    datasets = [dataset for _, dataset in ordered]
    smallest = datasets[0]
    if smallest.n_categories < focus_n:
        raise ValueError("The smallest universe does not contain the focal categories")
    base_ids = np.asarray(smallest.basket_ids, dtype=object)
    base_periods = np.asarray(smallest.periods, dtype=object)
    base_focus = smallest.X[:, :focus_n]
    for label, dataset in ordered[1:]:
        if not np.array_equal(base_ids, np.asarray(dataset.basket_ids, dtype=object)):
            raise ValueError(f"Universe {label} does not use the same baskets")
        if not np.array_equal(base_periods, np.asarray(dataset.periods, dtype=object)):
            raise ValueError(f"Universe {label} does not use the same periods")
        if (base_focus != dataset.X[:, :focus_n]).nnz:
            raise ValueError(f"Universe {label} does not preserve the focal columns")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairs, all_triples = _candidate_arrays(focus_n)

    focus_h1 = smallest.subset_period("H1")
    focus_counts = count_for_candidates(
        focus_h1,
        pair_candidates=pairs,
        triple_candidates=all_triples,
        base_scan=scan_pairs(focus_h1, count_pair_totals=False),
    )
    keep = focus_counts.triple_totals >= int(triple_min_count)
    triples = all_triples[keep]
    if len(triples) == 0:
        raise ValueError("No focal triples remain after the H1 support filter")

    built: dict[str, UniverseLevel] = {}
    for label, dataset in ordered:
        built[label] = _build_level(
            label,
            dataset,
            pairs=pairs,
            triples=triples,
            config=config,
            cache_dir=output / "cache" / str(label),
        )

    reference_label = labels[-1]
    reference_level = built[reference_label]
    pair_full_gauge = reference_level.full_contrast2.mu
    triple_full_gauge = reference_level.full_contrast3.mu
    pair_period_gauge = reference_level.h1_contrast2.mu
    triple_period_gauge = reference_level.h1_contrast3.mu

    fitted: dict[str, UniverseFittedLevel] = {}
    for label in labels:
        fitted[label] = _fit_level(
            built[label],
            config=config,
            pair_full_gauge=pair_full_gauge,
            triple_full_gauge=triple_full_gauge,
            pair_period_gauge=pair_period_gauge,
            triple_period_gauge=triple_period_gauge,
        )
    reference = fitted[reference_label]

    stability_rows: list[dict[str, object]] = []
    for label in labels:
        current = fitted[label]
        for order, current_fit, reference_fit in (
            (2, current.full_fit2, reference.full_fit2),
            (3, current.full_fit3, reference.full_fit3),
        ):
            for filter_name, mask in _reference_masks(reference_fit).items():
                row = _agreement_row(current_fit, reference_fit, mask)
                row.update(
                    {
                        "universe": label,
                        "context_categories": current.source.dataset.n_categories,
                        "reference_universe": reference_label,
                        "order": order,
                        "filter": filter_name,
                    }
                )
                stability_rows.append(row)
    coefficient_stability = pd.DataFrame(stability_rows)

    transfer_rows: list[pd.DataFrame] = []
    for label in labels:
        current = fitted[label]
        layers = build_transfer_layers(
            current.h1_fit3,
            current.h2_fit3,
            category_names=current.source.dataset.category_names,
            min_rest_size=config.min_rest_size,
        )
        sweep = threshold_sweep(layers, ccc_thresholds)
        sweep.insert(0, "context_categories", current.source.dataset.n_categories)
        sweep.insert(0, "universe", label)
        transfer_rows.append(sweep)
    transfer_sweep = pd.concat(transfer_rows, ignore_index=True)
    transfer_primary = transfer_sweep[
        np.isclose(transfer_sweep["h1_z_threshold"], float(primary_z))
    ].copy()

    reference_correction_started = perf_counter()
    reference_correction = corrected_pair_fit(
        reference.source.dataset,
        reference.source.full_counts,
        reference.full_fit2,
        reference.full_fit3,
        epsilon=config.epsilon,
        min_rest_size=config.min_rest_size,
        ridge=config.ridge2,
        exponent_clip=exponent_clip,
        block_size=config.block_size,
        gauge_weights=pair_full_gauge,
    ).corrected_fit
    reference_correction_seconds = perf_counter() - reference_correction_started
    _, reference_delta = correction_delta(reference.full_fit2, reference_correction)
    deprojection_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for label in labels:
        current = fitted[label]
        if label == reference_label:
            corrected = reference_correction
            correction_seconds = reference_correction_seconds
        else:
            correction_started = perf_counter()
            corrected = corrected_pair_fit(
                current.source.dataset,
                current.source.full_counts,
                current.full_fit2,
                current.full_fit3,
                epsilon=config.epsilon,
                min_rest_size=config.min_rest_size,
                ridge=config.ridge2,
                exponent_clip=exponent_clip,
                block_size=config.block_size,
                gauge_weights=pair_full_gauge,
            ).corrected_fit
            correction_seconds = perf_counter() - correction_started
        _, delta = correction_delta(current.full_fit2, corrected)
        pearson, spearman = pearson_spearman(delta, reference_delta)
        error = delta - reference_delta
        deprojection_rows.append(
            {
                "universe": label,
                "context_categories": current.source.dataset.n_categories,
                "reference_universe": reference_label,
                "n_pairs": len(delta),
                "pearson": pearson,
                "spearman": spearman,
                "sign_agreement": float(np.mean(np.sign(delta) == np.sign(reference_delta))),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error * error))),
            }
        )
        matrices = (
            current.source.full_contrast2,
            current.source.full_contrast3,
            current.source.h1_contrast2,
            current.source.h1_contrast3,
            current.source.h2_contrast2,
            current.source.h2_contrast3,
        )
        matrix_bytes = sum(
            int(np.asarray(table.lambda_hat).nbytes + np.asarray(table.weights).nbytes)
            for table in matrices
        )
        runtime_rows.append(
            {
                "universe": label,
                "context_categories": current.source.dataset.n_categories,
                "baskets": current.source.dataset.n_baskets,
                "mean_context_basket_size": float(np.mean(current.source.dataset.basket_sizes)),
                "max_context_basket_size": int(np.max(current.source.dataset.basket_sizes)),
                "focal_pairs": len(pairs),
                "focal_triples": len(triples),
                "count_and_contrast_seconds": current.source.build_seconds,
                "fit_seconds": current.fit_seconds,
                "deprojection_seconds": correction_seconds,
                "contrast_storage_mb": matrix_bytes / (1024.0 * 1024.0),
            }
        )
    deprojection_stability = pd.DataFrame(deprojection_rows)
    runtime = pd.DataFrame(runtime_rows)
    coefficients = _coefficient_frame(
        fitted,
        focus_names=smallest.category_names[:focus_n],
    )

    coefficient_stability.to_csv(output / "item_universe_coefficient_stability.csv", index=False)
    transfer_sweep.to_csv(output / "item_universe_transfer_sweep.csv", index=False)
    transfer_primary.to_csv(output / "item_universe_transfer_primary.csv", index=False)
    deprojection_stability.to_csv(output / "item_universe_deprojection_stability.csv", index=False)
    runtime.to_csv(output / "item_universe_runtime.csv", index=False)
    coefficients.to_csv(output / "item_universe_coefficients.csv", index=False)
    pair_comparison, triple_comparison = build_item_universe_coefficient_comparisons(
        coefficients,
        small_universe=labels[0],
        large_universe=reference_label,
    )
    if not pair_comparison.empty:
        pair_comparison.to_csv(output / "item_universe_pair_comparison.csv", index=False)
    if not triple_comparison.empty:
        triple_comparison.to_csv(output / "item_universe_triple_comparison.csv", index=False)
    plot_item_universe_coefficient_scatter(
        pair_comparison,
        triple_comparison,
        output / "item_universe_expansion.png",
        small_label=labels[0],
        large_label=reference_label,
    )

    summary = [
        "# Item-universe expansion audit",
        "",
        (
            f"Focal categories: {focus_n}; H1-supported focal triples: {len(triples)}; "
            f"reference universe: {reference_label}."
        ),
        "All levels use the same baskets, focal candidates, and reference-universe gauge weights.",
        "",
        "## Coefficient stability relative to the full universe",
        "",
        coefficient_stability[
            ((coefficient_stability["order"] == 2) & (coefficient_stability["filter"] == "all"))
            | ((coefficient_stability["order"] == 3) & (coefficient_stability["filter"] == "top25%"))
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Cross-period agreement at the primary H1 threshold",
        "",
        transfer_primary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Pair-deprojection stability",
        "",
        deprojection_stability.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Runtime",
        "",
        runtime.to_markdown(index=False, floatfmt=".4g"),
        "",
        "![Item-universe expansion](item_universe_expansion.png)",
        "",
    ]
    (output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    metadata = {
        "focus_n": int(focus_n),
        "triple_min_count_h1": int(triple_min_count),
        "primary_z": float(primary_z),
        "reference_universe": reference_label,
        "universe_labels": labels,
        "context_sizes": [int(fitted[label].source.dataset.n_categories) for label in labels],
        "analysis_config": config.__dict__,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ItemUniverseResult(
        coefficient_stability=coefficient_stability,
        transfer_sweep=transfer_sweep,
        transfer_primary=transfer_primary,
        deprojection_stability=deprojection_stability,
        runtime=runtime,
        coefficients=coefficients,
        output_dir=output,
    )
