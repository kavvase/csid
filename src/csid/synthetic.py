from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score

from .contrasts import build_contrast_matrix
from .counts import count_for_candidates, scan_pairs
from .data import SparseBasketDataset
from .estimator import fit_csid
from .ising import fit_pairwise_ising_exact
from .keys import as_pair, as_triple


@dataclass(frozen=True)
class SyntheticModel:
    n_categories: int
    alpha: np.ndarray
    h: np.ndarray
    pair: Mapping[tuple[int, int], float]
    triple: Mapping[tuple[int, int, int], float]
    min_size: int = 1


def _subset_zeta(coefficients: np.ndarray, n_categories: int) -> np.ndarray:
    values = np.asarray(coefficients, dtype=float).copy()
    for bit in range(n_categories):
        step = 1 << bit
        for start in range(0, 1 << n_categories, step << 1):
            values[start + step : start + (step << 1)] += values[start : start + step]
    return values


def _popcount(n_categories: int) -> np.ndarray:
    values = np.arange(1 << n_categories, dtype=np.uint64)
    # Exact sampler is used only for small C; a vectorized unpack is sufficient.
    return np.asarray([int(value).bit_count() for value in values], dtype=np.int16)


def sample_loglinear(model: SyntheticModel, n_samples: int, *, seed: int) -> np.ndarray:
    C = int(model.n_categories)
    if C > 20:
        raise ValueError("Exact synthetic sampling is intentionally limited to C <= 20")
    coefficients = np.zeros(1 << C, dtype=float)
    for i, value in enumerate(np.asarray(model.h, dtype=float)):
        coefficients[1 << i] += float(value)
    for pair_candidate, value in model.pair.items():
        mask = sum(1 << int(i) for i in pair_candidate)
        coefficients[mask] += float(value)
    for triple_candidate, value in model.triple.items():
        mask = sum(1 << int(i) for i in triple_candidate)
        coefficients[mask] += float(value)
    energy = _subset_zeta(coefficients, C)
    sizes = _popcount(C)
    energy += np.asarray(model.alpha, dtype=float)[sizes]
    allowed = sizes >= int(model.min_size)
    probabilities = np.zeros_like(energy)
    probabilities[allowed] = np.exp(energy[allowed] - logsumexp(energy[allowed]))
    rng = np.random.default_rng(seed)
    masks = rng.choice(len(probabilities), size=int(n_samples), p=probabilities).astype(np.uint64)
    bits = np.arange(C, dtype=np.uint64)
    return ((masks[:, None] >> bits[None, :]) & 1).astype(np.uint8)


def default_alpha(n_categories: int, *, strength: float = 0.7) -> np.ndarray:
    alpha = np.zeros(n_categories + 1, dtype=float)
    alpha[0] = -100.0
    for size in range(1, n_categories + 1):
        alpha[size] = -float(strength) * (size - 1) ** 1.3
    return alpha


def _gauge_align(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    return values.copy() if total <= 0 else values - float(np.dot(weights, values) / total)


def _mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"),) * 4
    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, float("nan"), float("nan"), float("nan")
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    critical = float(student_t.ppf(0.5 + confidence / 2.0, df=len(x) - 1))
    return mean, se, mean - critical * se, mean + critical * se


def _fit_csid2_dense(X: np.ndarray):
    """Fit CSID-2 exhaustively for a small dense synthetic matrix."""

    C = int(X.shape[1])
    pair_candidates = np.asarray(tuple(combinations(range(C), 2)), dtype=np.int32)
    dataset = SparseBasketDataset.from_dense(X, name="synthetic-cardinality")
    scan = scan_pairs(dataset)
    counts = count_for_candidates(
        dataset,
        pair_candidates=pair_candidates,
        triple_candidates=np.empty((0, 3), dtype=np.int32),
        base_scan=scan,
    )
    table = build_contrast_matrix(counts, k=2, min_rest_size=1, memory_limit_mb=256)
    return fit_csid(table, ridge=1.0)


def cardinality_confounding_experiment(
    *,
    cardinality_strengths: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    replicates: int = 20,
    n_baskets: int = 10_000,
    n_categories: int = 10,
    seed: int = 2_026,
    ising_l2: float = 1.0e-5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Demonstrate negative pair contamination from an omitted cardinality term.

    Every generating model has J_ij = 0 and no higher-order interactions.  Only
    the global cardinality potential is strengthened.  Each sample is fitted by
    an ordinary pairwise Ising model without alpha_|x| and by CSID-2.
    """

    rows: list[dict[str, float | int | str | bool]] = []
    for replicate in range(int(replicates)):
        rng = np.random.default_rng(int(seed) + replicate)
        # Reuse the same item propensities across strengths within a replicate,
        # so that the curve isolates the changed cardinality potential.
        h = rng.normal(-0.30, 0.25, int(n_categories))
        for strength_index, strength in enumerate(cardinality_strengths):
            X = sample_loglinear(
                SyntheticModel(
                    n_categories=int(n_categories),
                    alpha=default_alpha(int(n_categories), strength=float(strength)),
                    h=h,
                    pair={},
                    triple={},
                    min_size=1,
                ),
                int(n_baskets),
                seed=int(seed) + 50_000 + replicate * 100 + strength_index,
            )
            ordinary = fit_pairwise_ising_exact(
                X,
                min_size=1,
                l2=float(ising_l2),
            )
            csid2 = _fit_csid2_dense(X)
            for method, estimate, converged in (
                ("Ordinary pairwise Ising", ordinary.J, ordinary.success),
                ("CSID-2", csid2.theta, csid2.converged),
            ):
                estimate = np.asarray(estimate, dtype=float)
                rows.append(
                    {
                        "replicate": replicate,
                        "cardinality_strength": float(strength),
                        "method": method,
                        "mean_basket_size": float(np.mean(X.sum(axis=1))),
                        "mean_pair_coefficient": float(np.mean(estimate)),
                        "median_pair_coefficient": float(np.median(estimate)),
                        "negative_pair_fraction": float(np.mean(estimate < 0.0)),
                        "rmse_to_zero": float(np.sqrt(np.mean(estimate**2))),
                        "fit_converged": bool(converged),
                    }
                )

    replicate_frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, float | int | str]] = []
    metrics = (
        "mean_basket_size",
        "mean_pair_coefficient",
        "median_pair_coefficient",
        "negative_pair_fraction",
        "rmse_to_zero",
    )
    for (strength, method), group in replicate_frame.groupby(
        ["cardinality_strength", "method"], sort=True
    ):
        row: dict[str, float | int | str] = {
            "cardinality_strength": float(strength),
            "method": str(method),
            "replicates": int(group["replicate"].nunique()),
        }
        for metric in metrics:
            mean, se, low, high = _mean_ci(group[metric].to_numpy(float))
            row[metric] = mean
            row[f"{metric}_mc_se"] = se
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summary_rows.append(row)
    return replicate_frame, pd.DataFrame(summary_rows)


def plot_cardinality_confounding(summary: pd.DataFrame, path: str | Path) -> None:
    if summary.empty:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for method, frame in summary.groupby("method", sort=False):
        data = frame.sort_values("mean_basket_size", ascending=False, kind="stable")
        x = data["mean_basket_size"].to_numpy(float)
        for ax, metric, label in (
            (axes[0], "mean_pair_coefficient", r"Mean pairwise estimate $\widehat{\theta}_{ij}$"),
            (axes[1], "rmse_to_zero", "RMSE"),
        ):
            y = data[metric].to_numpy(float)
            low = data[f"{metric}_ci_low"].to_numpy(float)
            high = data[f"{metric}_ci_high"].to_numpy(float)
            ax.plot(x, y, marker="o", label=str(method))
            ax.fill_between(x, low, high, alpha=0.15)
            ax.set_xlabel("Mean basket size")
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
    axes[0].axhline(0.0, linestyle=":", linewidth=1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    if output.suffix.lower() == ".png":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def randomized_triple_power_experiment(
    *,
    effect_sizes: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
    replicates: int = 20,
    n_baskets: int = 10_000,
    n_categories: int = 10,
    n_planted: int = 8,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_planted < 2 or n_planted % 2:
        raise ValueError("n_planted must be a positive even integer")
    all_pairs = tuple(combinations(range(n_categories), 2))
    all_triples = tuple(combinations(range(n_categories), 3))
    pair_candidates = np.asarray(all_pairs, dtype=np.int32)
    triple_candidates = np.asarray(all_triples, dtype=np.int32)
    rows: list[dict[str, float | int | bool]] = []

    for replicate in range(int(replicates)):
        rng = np.random.default_rng(seed + 10_000 * replicate)
        h = rng.normal(-0.25, 0.35, n_categories)
        pair_mask = rng.random(len(all_pairs)) < 0.35
        pair_values = rng.normal(0.0, 0.30, len(all_pairs))
        pair = {
            candidate: float(value)
            for candidate, keep, value in zip(all_pairs, pair_mask, pair_values, strict=False)
            if keep
        }
        alpha = default_alpha(n_categories, strength=float(rng.uniform(0.50, 0.80)))
        selected = rng.choice(len(all_triples), size=n_planted, replace=False)
        planted = tuple(all_triples[int(index)] for index in selected)
        signs = np.asarray([1.0] * (n_planted // 2) + [-1.0] * (n_planted // 2))
        rng.shuffle(signs)

        for effect_index, effect_size in enumerate(effect_sizes):
            triple = {
                candidate: float(sign * float(effect_size))
                for candidate, sign in zip(planted, signs, strict=False)
            }
            X = sample_loglinear(
                SyntheticModel(n_categories, alpha, h, pair, triple, min_size=1),
                int(n_baskets),
                seed=seed + 1_000_000 + replicate * 100 + effect_index,
            )
            dataset = SparseBasketDataset.from_dense(X, name="synthetic")
            scan = scan_pairs(dataset)
            counts = count_for_candidates(
                dataset,
                pair_candidates=pair_candidates,
                triple_candidates=triple_candidates,
                base_scan=scan,
            )
            table = build_contrast_matrix(counts, k=3, min_rest_size=1, memory_limit_mb=256)
            fit = fit_csid(table, ridge=1.0)
            candidate_to_index = {
                as_triple(candidate): q for q, candidate in enumerate(fit.candidates)
            }
            labels = np.asarray(
                [int(as_triple(candidate) in triple) for candidate in fit.candidates],
                dtype=int,
            )
            scores = np.abs(fit.z)
            detected = scores >= 2.0
            positive = labels == 1
            negative = labels == 0
            raw_truth = np.asarray(
                [float(triple.get(as_triple(candidate), 0.0)) for candidate in fit.candidates],
                dtype=np.float64,
            )
            truth = _gauge_align(raw_truth, fit.mu)
            planted_indexes = np.asarray([candidate_to_index[t] for t in planted], dtype=int)
            prevalence = float(np.mean(labels))
            auprc = float(average_precision_score(labels, scores))
            rows.append(
                {
                    "replicate": replicate,
                    "effect_size": float(effect_size),
                    "fit_converged": bool(fit.converged),
                    "power_abs_z_gt_2": float(np.mean(detected[positive])),
                    "false_positive_rate_abs_z_gt_2": float(np.mean(detected[negative])),
                    "auprc_abs_z": auprc,
                    "random_auprc": prevalence,
                    "auroc_abs_z": float(roc_auc_score(labels, scores)),
                    "sign_recovery": float(
                        np.mean(np.sign(fit.theta[planted_indexes]) == np.sign(truth[planted_indexes]))
                    ),
                    "planted_rmse_gauge": float(
                        np.sqrt(np.mean((fit.theta[planted_indexes] - truth[planted_indexes]) ** 2))
                    ),
                }
            )

    replicate_frame = pd.DataFrame(rows)
    metrics = [
        "power_abs_z_gt_2",
        "false_positive_rate_abs_z_gt_2",
        "auprc_abs_z",
        "auroc_abs_z",
        "sign_recovery",
        "planted_rmse_gauge",
    ]
    summaries: list[dict[str, float | int]] = []
    for effect_size, group in replicate_frame.groupby("effect_size", sort=True):
        row: dict[str, float | int] = {
            "effect_size": float(effect_size),
            "replicates": int(group["replicate"].nunique()),
            "random_auprc": float(group["random_auprc"].mean()),
        }
        for metric in metrics:
            mean, se, low, high = _mean_ci(group[metric].to_numpy(float))
            row[metric] = mean
            row[f"{metric}_mc_se"] = se
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summaries.append(row)
    return replicate_frame, pd.DataFrame(summaries)


def plot_synthetic_detection_power(summary: pd.DataFrame, path: str | Path) -> None:
    if summary.empty:
        return
    import matplotlib.pyplot as plt

    frame = summary.sort_values("effect_size")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = frame["effect_size"].to_numpy(float)
    for metric, label in (
        ("power_abs_z_gt_2", "Power"),
        ("false_positive_rate_abs_z_gt_2", "Non-planted exceedance rate"),
    ):
        y = frame[metric].to_numpy(float)
        low = frame[f"{metric}_ci_low"].to_numpy(float)
        high = frame[f"{metric}_ci_high"].to_numpy(float)
        ax.plot(x, y, marker="o", label=label)
        ax.fill_between(x, low, high, alpha=0.15)
    ax.set_xlabel(r"Planted effect size $|\theta|$")
    ax.set_ylabel("Rate")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def saturation_deprojection_experiment(
    *,
    n_samples: int = 60_000,
    seed: int = 27,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paper saturation experiment with known pair and triple coefficients."""

    from tempfile import TemporaryDirectory

    from .analysis import AnalysisConfig, fit_generated_candidates
    from .correction import corrected_pair_fit

    C = 6
    pairs = tuple(combinations(range(C), 2))
    true_pairs = {pair: 0.0 for pair in pairs}
    for pair in ((0, 1), (0, 2), (1, 2)):
        true_pairs[pair] = 0.8
    true_pairs[(3, 4)] = 0.5
    true_pairs[(4, 5)] = 0.3
    model = SyntheticModel(
        n_categories=C,
        alpha=default_alpha(C, strength=0.55),
        h=np.asarray([-0.15, -0.10, -0.05, -0.20, -0.25, -0.30]),
        pair=true_pairs,
        triple={(0, 1, 2): -1.2},
        min_size=1,
    )
    X = sample_loglinear(model, int(n_samples), seed=seed)
    dataset = SparseBasketDataset.from_dense(X, name="saturation")
    config = AnalysisConfig(
        candidate_mode="exhaustive",
        pair_candidate_min_count=0,
        pair_model_min_count=0,
        triple_min_count=0,
        min_rest_size=1,
        memory_limit_mb=256,
    )
    with TemporaryDirectory(prefix="csid_saturation_") as temp_dir:
        bundle = fit_generated_candidates(dataset, config=config, cache_dir=temp_dir)
    corrected = corrected_pair_fit(
        dataset,
        bundle.counts,
        bundle.fit2,
        bundle.fit3,
        epsilon=config.epsilon,
        min_rest_size=config.min_rest_size,
        ridge=config.ridge2,
    ).corrected_fit

    raw_truth = np.asarray(
        [float(true_pairs.get(as_pair(pair), 0.0)) for pair in bundle.fit2.candidates],
        dtype=np.float64,
    )
    truth = _gauge_align(raw_truth, bundle.fit2.mu)

    def metric(name: str, estimate: np.ndarray) -> dict[str, float | str]:
        return {
            "method": name,
            "rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
            "pearson": float(np.corrcoef(estimate, truth)[0, 1]),
        }

    summary = pd.DataFrame(
        [
            metric("CSID-2", bundle.fit2.theta),
            metric("CSID-3 deprojected", corrected.theta),
        ]
    )
    detail = pd.DataFrame(
        {
            "pair": [f"{i}-{j}" for i, j in bundle.fit2.candidates],
            "J_true_gauge": truth,
            "CSID2": bundle.fit2.theta,
            "deprojected": corrected.theta,
            "delta_J": corrected.theta - bundle.fit2.theta,
        }
    )
    return summary, detail
