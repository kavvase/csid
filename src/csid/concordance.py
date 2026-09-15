"""Cross-period agreement diagnostics used in the paper.

The module evaluates whether the triple-specific component estimated in H1
improves agreement with the rest-size-stratified third-order contrasts observed
in H2.  H2 is used only for the common rest-size component and the evaluation
contrast; the triple-specific component comes from H1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .estimator import CSIDFit
from .keys import as_pair, as_triple, pair_key


@dataclass(frozen=True)
class ConcordanceResult:
    layers: pd.DataFrame
    sweep: pd.DataFrame
    primary: pd.DataFrame
    decomposition: pd.DataFrame
    loco: pd.DataFrame
    permutations: pd.DataFrame
    permutation_summary: pd.DataFrame
    examples: pd.DataFrame
    benchmark_sweep: pd.DataFrame
    benchmark_primary: pd.DataFrame
    benchmark_loco: pd.DataFrame
    full_transfer: pd.DataFrame


def weighted_ccc(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int]:
    """Information-weighted Lin concordance correlation statistics."""

    y = np.asarray(observed, dtype=float)
    x = np.asarray(predicted, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(y) & np.isfinite(x) & np.isfinite(w) & (w > 0)
    if keep.sum() < 2:
        return {
            "n": int(keep.sum()),
            "weight_sum": float(w[keep].sum()),
            "ccc": float("nan"),
            "pearson": float("nan"),
            "bias_correction": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "observed_mean": float("nan"),
            "predicted_mean": float("nan"),
            "observed_sd": float("nan"),
            "predicted_sd": float("nan"),
        }
    y, x, w = y[keep], x[keep], w[keep]
    wn = w / w.sum()
    mean_y = float(np.dot(wn, y))
    mean_x = float(np.dot(wn, x))
    dy, dx = y - mean_y, x - mean_x
    var_y = float(np.dot(wn, dy * dy))
    var_x = float(np.dot(wn, dx * dx))
    cov = float(np.dot(wn, dy * dx))
    scale = float(np.sqrt(max(var_y * var_x, 0.0)))
    denominator = var_y + var_x + (mean_y - mean_x) ** 2
    pearson = cov / scale if scale > 0 else float("nan")
    ccc = 2.0 * cov / denominator if denominator > 0 else float("nan")
    bias = 2.0 * scale / denominator if denominator > 0 else float("nan")
    error = x - y
    return {
        "n": int(len(y)),
        "weight_sum": float(w.sum()),
        "ccc": ccc,
        "pearson": pearson,
        "bias_correction": bias,
        "mae": float(np.dot(wn, np.abs(error))),
        "rmse": float(np.sqrt(np.dot(wn, error * error))),
        "observed_mean": mean_y,
        "predicted_mean": mean_x,
        "observed_sd": float(np.sqrt(max(var_y, 0.0))),
        "predicted_sd": float(np.sqrt(max(var_x, 0.0))),
    }


def _index(fit: CSIDFit) -> dict[tuple[int, int, int], int]:
    return {as_triple(candidate): q for q, candidate in enumerate(fit.candidates)}


def build_transfer_layers(
    fit_h1: CSIDFit,
    fit_h2: CSIDFit,
    *,
    category_names: Sequence[str] | None = None,
    min_rest_size: int = 2,
) -> pd.DataFrame:
    """Construct all common triple/rest-size rows used by the CCC audit."""

    index_h1, index_h2 = _index(fit_h1), _index(fit_h2)
    common = sorted(set(index_h1).intersection(index_h2))
    if not common:
        return pd.DataFrame()
    q1 = np.asarray([index_h1[key] for key in common], dtype=int)
    q2 = np.asarray([index_h2[key] for key in common], dtype=int)
    W1 = np.asarray(fit_h1.table.weights[q1], dtype=float)
    W2 = np.asarray(fit_h2.table.weights[q2], dtype=float)
    L1 = np.asarray(fit_h1.table.lambda_hat[q1], dtype=float)
    L2 = np.asarray(fit_h2.table.lambda_hat[q2], dtype=float)
    theta = np.asarray(fit_h1.theta[q1], dtype=float)
    z = np.asarray(fit_h1.z[q1], dtype=float)
    mu = np.asarray(fit_h1.mu[q1], dtype=float)

    r_count = min(W1.shape[1], W2.shape[1], len(fit_h1.beta))
    W1, W2, L1, L2 = W1[:, :r_count], W2[:, :r_count], L1[:, :r_count], L2[:, :r_count]
    sum_w = W2.sum(axis=0)
    sum_w_lambda = np.sum(W2 * L2, axis=0)
    sum_w_residual = np.sum(W2 * (L2 - theta[:, None]), axis=0)

    rows: list[dict[str, object]] = []
    for q, triple in enumerate(common):
        for r in range(max(int(min_rest_size), 0), r_count):
            w1, w2 = float(W1[q, r]), float(W2[q, r])
            if w2 <= 0:
                continue
            denom = float(sum_w[r] - w2)
            if denom <= 0:
                continue
            observed = float(L2[q, r])
            baseline = float((sum_w_lambda[r] - w2 * observed) / denom)
            own_residual = observed - float(theta[q])
            common_component = float((sum_w_residual[r] - w2 * own_residual) / denom)
            harmonic_weight = float(2.0 * w1 * w2 / (w1 + w2)) if w1 > 0 else 0.0
            row: dict[str, object] = {
                "candidate": "|".join(map(str, triple)),
                "item_1_idx": triple[0],
                "item_2_idx": triple[1],
                "item_3_idx": triple[2],
                "rest_size": r,
                "theta_h1": float(theta[q]),
                "z_h1": float(z[q]),
                "mu_h1": float(mu[q]),
                "h1_layer_weight": w1,
                "h2_layer_weight": w2,
                "information_weight": harmonic_weight,
                "observed_h1_lambda": float(L1[q, r]),
                "observed_h2_lambda": observed,
                "observed_h2_se": float(1.0 / np.sqrt(w2)),
                "baseline_prediction": baseline,
                "csid3_prediction": common_component + float(theta[q]),
                "h1_common_prediction": float(fit_h1.beta[r]),
                "full_transfer_prediction": float(fit_h1.beta[r] + theta[q]),
            }
            if category_names is not None:
                row.update(
                    {
                        "item_1": str(category_names[triple[0]]),
                        "item_2": str(category_names[triple[1]]),
                        "item_3": str(category_names[triple[2]]),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _metric_row(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    selected = frame[
        (frame["information_weight"].to_numpy(float) > 0)
        & (np.abs(frame["z_h1"].to_numpy(float)) >= float(threshold))
    ]
    if selected.empty:
        empty_row: dict[str, float | int] = {
            "h1_z_threshold": float(threshold),
            "selected_triples": 0,
            "n_layers": 0,
        }
        for prefix in ("baseline", "csid"):
            for key in (
                "n",
                "weight_sum",
                "ccc",
                "pearson",
                "bias_correction",
                "mae",
                "rmse",
                "observed_mean",
                "predicted_mean",
                "observed_sd",
                "predicted_sd",
            ):
                empty_row[f"{prefix}_{key}"] = 0 if key == "n" else float("nan")
        empty_row["delta_ccc"] = float("nan")
        empty_row["delta_pearson"] = float("nan")
        empty_row["delta_bias_correction"] = float("nan")
        return empty_row
    weight = selected["information_weight"].to_numpy(float)
    observed = selected["observed_h2_lambda"].to_numpy(float)
    base = weighted_ccc(observed, selected["baseline_prediction"], weight)
    csid = weighted_ccc(observed, selected["csid3_prediction"], weight)
    row: dict[str, float | int] = {
        "h1_z_threshold": float(threshold),
        "selected_triples": int(selected["candidate"].nunique()),
        "n_layers": int(len(selected)),
    }
    for prefix, metrics in (("baseline", base), ("csid", csid)):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    row["delta_ccc"] = float(csid["ccc"] - base["ccc"])
    row["delta_pearson"] = float(csid["pearson"] - base["pearson"])
    row["delta_bias_correction"] = float(csid["bias_correction"] - base["bias_correction"])
    return row


def threshold_sweep(
    layers: pd.DataFrame,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    return pd.DataFrame([_metric_row(layers, threshold) for threshold in thresholds])


def category_loco(
    layers: pd.DataFrame,
    *,
    threshold: float,
    n_categories: int,
) -> pd.DataFrame:
    point = _metric_row(layers, threshold)
    rows: list[dict[str, float | int]] = []
    for category in range(int(n_categories)):
        keep = ~(
            layers["item_1_idx"].eq(category) | layers["item_2_idx"].eq(category) | layers["item_3_idx"].eq(category)
        )
        row = _metric_row(layers.loc[keep], threshold)
        row["excluded_category"] = category
        rows.append(row)
    frame = pd.DataFrame(rows)
    for metric in ("delta_ccc", "delta_pearson", "delta_bias_correction"):
        frame.attrs[f"{metric}_point"] = float(point[metric])
        frame.attrs[f"{metric}_loco_min"] = float(frame[metric].min())
        frame.attrs[f"{metric}_loco_max"] = float(frame[metric].max())
    return frame


def _permuted_predictions(
    layers: pd.DataFrame,
    theta_map: Mapping[str, float],
) -> np.ndarray:
    frame = layers.copy()
    frame["permuted_theta"] = frame["candidate"].map(theta_map).astype(float)
    frame["residual"] = frame["observed_h2_lambda"] - frame["permuted_theta"]
    result = np.empty(len(frame), dtype=float)
    for _, positions in frame.groupby("rest_size", sort=False).indices.items():
        pos = np.asarray(positions, dtype=int)
        w = frame.iloc[pos]["h2_layer_weight"].to_numpy(float)
        residual = frame.iloc[pos]["residual"].to_numpy(float)
        theta = frame.iloc[pos]["permuted_theta"].to_numpy(float)
        total_w = float(w.sum())
        total_wr = float(np.dot(w, residual))
        denom = total_w - w
        common = np.divide(
            total_wr - w * residual,
            denom,
            out=np.zeros_like(w),
            where=denom > 0,
        )
        result[pos] = common + theta
    return result


def add_transfer_prediction(
    layers: pd.DataFrame,
    *,
    candidates: np.ndarray,
    theta: np.ndarray,
    column: str = "pl_prediction",
) -> pd.DataFrame:
    """Add a leave-one-out transfer reconstruction from external coefficients."""

    keys = ["|".join(map(str, map(int, candidate))) for candidate in np.asarray(candidates)]
    values = np.asarray(theta, dtype=float)
    if len(keys) != len(values):
        raise ValueError("candidates and theta must have the same length")
    theta_map = dict(zip(keys, values, strict=False))
    missing = sorted(set(layers["candidate"].astype(str)).difference(theta_map))
    if missing:
        raise ValueError(f"External coefficients are missing {len(missing)} layer candidates")
    result = layers.copy()
    result["theta_pl_h1"] = result["candidate"].map(theta_map).astype(float)
    result[column] = _permuted_predictions(result, theta_map)
    return result


def _benchmark_metric_row(
    layers: pd.DataFrame,
    *,
    information_fraction: float,
) -> dict[str, float | int]:
    if not 0 < information_fraction <= 1:
        raise ValueError("information_fraction must lie in (0, 1]")
    candidate_table = layers[["candidate", "mu_h1"]].drop_duplicates("candidate")
    cutoff = float(candidate_table["mu_h1"].quantile(1.0 - information_fraction))
    selected_candidates = set(
        candidate_table.loc[candidate_table["mu_h1"] >= cutoff, "candidate"].astype(str)
    )
    selected = layers[
        layers["candidate"].astype(str).isin(selected_candidates)
        & (layers["information_weight"].to_numpy(float) > 0)
    ]
    row: dict[str, float | int] = {
        "information_fraction": float(information_fraction),
        "information_cutoff": cutoff,
        "selected_triples": int(selected["candidate"].nunique()),
        "n_layers": int(len(selected)),
    }
    if selected.empty:
        for prefix in ("baseline", "csid", "pl"):
            row[f"{prefix}_ccc"] = float("nan")
            row[f"{prefix}_pearson"] = float("nan")
            row[f"{prefix}_bias_correction"] = float("nan")
            row[f"{prefix}_rmse"] = float("nan")
        row["csid_minus_pl_ccc"] = float("nan")
        row["csid_delta_ccc"] = float("nan")
        row["pl_delta_ccc"] = float("nan")
        return row

    observed = selected["observed_h2_lambda"].to_numpy(float)
    weights = selected["information_weight"].to_numpy(float)
    metrics = {
        "baseline": weighted_ccc(observed, selected["baseline_prediction"].to_numpy(float), weights),
        "csid": weighted_ccc(observed, selected["csid3_prediction"].to_numpy(float), weights),
        "pl": weighted_ccc(observed, selected["pl_prediction"].to_numpy(float), weights),
    }
    for prefix, values in metrics.items():
        for key in ("ccc", "pearson", "bias_correction", "rmse"):
            row[f"{prefix}_{key}"] = float(values[key])
    row["csid_minus_pl_ccc"] = float(metrics["csid"]["ccc"] - metrics["pl"]["ccc"])
    row["csid_delta_ccc"] = float(metrics["csid"]["ccc"] - metrics["baseline"]["ccc"])
    row["pl_delta_ccc"] = float(metrics["pl"]["ccc"] - metrics["baseline"]["ccc"])
    return row


def external_benchmark_sweep(
    layers: pd.DataFrame,
    fractions: Sequence[float],
) -> pd.DataFrame:
    if "pl_prediction" not in layers:
        return pd.DataFrame()
    return pd.DataFrame(
        [_benchmark_metric_row(layers, information_fraction=float(fraction)) for fraction in fractions]
    )


def external_benchmark_loco(
    layers: pd.DataFrame,
    *,
    information_fraction: float,
    n_categories: int,
) -> pd.DataFrame:
    if "pl_prediction" not in layers:
        return pd.DataFrame()
    point = _benchmark_metric_row(layers, information_fraction=information_fraction)
    rows: list[dict[str, float | int]] = []
    for category in range(int(n_categories)):
        keep = ~(
            layers["item_1_idx"].eq(category)
            | layers["item_2_idx"].eq(category)
            | layers["item_3_idx"].eq(category)
        )
        row = _benchmark_metric_row(
            layers.loc[keep],
            information_fraction=information_fraction,
        )
        row["excluded_category"] = category
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs["csid_minus_pl_ccc_point"] = float(point["csid_minus_pl_ccc"])
    frame.attrs["csid_minus_pl_ccc_loco_min"] = float(frame["csid_minus_pl_ccc"].min())
    frame.attrs["csid_minus_pl_ccc_loco_max"] = float(frame["csid_minus_pl_ccc"].max())
    return frame


def full_transfer_summary(
    layers: pd.DataFrame,
    *,
    information_fraction: float,
) -> pd.DataFrame:
    """Evaluate H1-only common and CSID reconstructions against H2 layers."""

    if layers.empty:
        return pd.DataFrame()
    candidate_table = layers[["candidate", "mu_h1"]].drop_duplicates("candidate")
    cutoff = float(candidate_table["mu_h1"].quantile(1.0 - information_fraction))
    selected_candidates = set(
        candidate_table.loc[candidate_table["mu_h1"] >= cutoff, "candidate"].astype(str)
    )
    selected = layers[
        layers["candidate"].astype(str).isin(selected_candidates)
        & (layers["information_weight"].to_numpy(float) > 0)
    ]
    if selected.empty:
        return pd.DataFrame()
    observed = selected["observed_h2_lambda"].to_numpy(float)
    weights = selected["information_weight"].to_numpy(float)
    row: dict[str, float | int] = {
        "information_fraction": float(information_fraction),
        "selected_triples": int(selected["candidate"].nunique()),
        "n_layers": int(len(selected)),
    }
    columns = {
        "h1_common": "h1_common_prediction",
        "full_csid": "full_transfer_prediction",
        "partial_csid": "csid3_prediction",
    }
    for prefix, column in columns.items():
        metrics = weighted_ccc(observed, selected[column].to_numpy(float), weights)
        for metric in ("ccc", "pearson", "bias_correction", "rmse"):
            row[f"{prefix}_{metric}"] = float(metrics[metric])
    row["full_csid_minus_h1_common_ccc"] = float(row["full_csid_ccc"]) - float(row["h1_common_ccc"])
    return pd.DataFrame([row])


def permutation_test(
    layers: pd.DataFrame,
    *,
    threshold: float,
    replicates: int = 199,
    information_bins: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_mask = (
        (layers["information_weight"].to_numpy(float) > 0)
        & (np.abs(layers["z_h1"].to_numpy(float)) >= float(threshold))
    )
    selected = layers.loc[selected_mask].copy()
    if selected.empty or replicates <= 0:
        return pd.DataFrame(), pd.DataFrame()
    candidate_table = layers[["candidate", "theta_h1", "mu_h1"]].drop_duplicates("candidate").reset_index(drop=True)
    try:
        candidate_table["info_bin"] = pd.qcut(
            candidate_table["mu_h1"], q=information_bins, labels=False, duplicates="drop"
        )
    except ValueError:
        candidate_table["info_bin"] = 0

    observed = selected["observed_h2_lambda"].to_numpy(float)
    weight = selected["information_weight"].to_numpy(float)
    baseline_ccc = weighted_ccc(observed, selected["baseline_prediction"].to_numpy(float), weight)["ccc"]
    observed_ccc = weighted_ccc(observed, selected["csid3_prediction"].to_numpy(float), weight)["ccc"]
    observed_delta = float(observed_ccc - baseline_ccc)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for replicate in range(int(replicates)):
        permuted = candidate_table.copy()
        for _, index in permuted.groupby("info_bin", dropna=False).groups.items():
            idx = np.asarray(list(index), dtype=int)
            values = permuted.loc[idx, "theta_h1"].to_numpy(float)
            permuted.loc[idx, "theta_h1"] = rng.permutation(values)
        theta_map = dict(zip(permuted["candidate"], permuted["theta_h1"], strict=False))
        prediction = _permuted_predictions(layers, theta_map)[selected_mask]
        ccc = weighted_ccc(observed, prediction, weight)["ccc"]
        rows.append(
            {
                "replicate": replicate,
                "permuted_ccc": float(ccc),
                "permuted_delta_ccc": float(ccc - baseline_ccc),
            }
        )
    frame = pd.DataFrame(rows)
    p_value = float((1 + np.sum(frame["permuted_delta_ccc"].to_numpy(float) >= observed_delta)) / (len(frame) + 1))
    summary = pd.DataFrame(
        [
            {
                "h1_z_threshold": float(threshold),
                "replicates": int(replicates),
                "observed_delta_ccc": observed_delta,
                "permutation_mean_delta_ccc": float(frame["permuted_delta_ccc"].mean()),
                "permutation_q95_delta_ccc": float(frame["permuted_delta_ccc"].quantile(0.95)),
                "one_sided_mc_p": p_value,
            }
        ]
    )
    return frame, summary


def _primary_decomposition(primary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame()
    row = primary.iloc[0]
    return pd.DataFrame(
        [
            {
                "selected_triples": int(row["selected_triples"]),
                "n_layers": int(row["n_layers"]),
                "baseline_ccc": float(row["baseline_ccc"]),
                "csid_ccc": float(row["csid_ccc"]),
                "delta_ccc": float(row["delta_ccc"]),
                "baseline_pearson": float(row["baseline_pearson"]),
                "csid_pearson": float(row["csid_pearson"]),
                "delta_pearson": float(row["delta_pearson"]),
                "baseline_bias_correction": float(row["baseline_bias_correction"]),
                "csid_bias_correction": float(row["csid_bias_correction"]),
                "delta_bias_correction": float(row["delta_bias_correction"]),
                "baseline_mae": float(row["baseline_mae"]),
                "csid_mae": float(row["csid_mae"]),
                "baseline_rmse": float(row["baseline_rmse"]),
                "csid_rmse": float(row["csid_rmse"]),
            }
        ]
    )


def select_conditional_odds_examples(
    layers: pd.DataFrame,
    pair_fit_h1: CSIDFit,
    *,
    category_names: Sequence[str],
    n_examples: int = 8,
    information_quantile: float = 0.75,
) -> pd.DataFrame:
    """Choose H1-ranked positive/negative examples among H2-evaluable layers."""

    if layers.empty:
        return pd.DataFrame()
    evaluable = layers[layers["information_weight"].to_numpy(float) > 0]
    candidate_table = (
        evaluable[["candidate", "item_1_idx", "item_2_idx", "item_3_idx", "theta_h1", "z_h1", "mu_h1"]]
        .drop_duplicates("candidate")
        .reset_index(drop=True)
    )
    cutoff = float(candidate_table["mu_h1"].quantile(information_quantile))
    candidate_table = candidate_table[candidate_table["mu_h1"] >= cutoff]
    half = max(int(n_examples) // 2, 1)
    negative = candidate_table[candidate_table["theta_h1"] < 0].nsmallest(half, "z_h1")
    positive = candidate_table[candidate_table["theta_h1"] > 0].nlargest(max(n_examples - len(negative), 0), "z_h1")
    chosen = pd.concat([negative, positive], ignore_index=True)
    if len(chosen) < n_examples:
        remaining = candidate_table[~candidate_table["candidate"].isin(chosen["candidate"])]
        remaining = remaining.assign(abs_z=remaining["z_h1"].abs()).nlargest(n_examples - len(chosen), "abs_z")
        chosen = pd.concat([chosen, remaining], ignore_index=True)

    pair_map = {
        as_pair(pair): float(value) for pair, value in zip(pair_fit_h1.candidates, pair_fit_h1.theta, strict=False)
    }
    rows: list[dict[str, object]] = []
    for selection_order, source in enumerate(chosen.itertuples(index=False), start=1):
        candidate_rows = evaluable[evaluable["candidate"].eq(source.candidate)]
        r_row = candidate_rows.loc[candidate_rows["h1_layer_weight"].idxmax()]
        triple = (int(source.item_1_idx), int(source.item_2_idx), int(source.item_3_idx))
        edge_pairs = (
            pair_key(triple[0], triple[1]),
            pair_key(triple[0], triple[2]),
            pair_key(triple[1], triple[2]),
        )
        focal = max(edge_pairs, key=lambda pair: abs(pair_map.get(pair, 0.0)))
        moderator = next(item for item in triple if item not in focal)
        observed = float(r_row["observed_h2_lambda"])
        se = float(r_row["observed_h2_se"])
        rows.append(
            {
                "selection_order_h1": selection_order,
                "candidate": source.candidate,
                "item_1": category_names[triple[0]],
                "item_2": category_names[triple[1]],
                "item_3": category_names[triple[2]],
                "focal_item_1": category_names[focal[0]],
                "focal_item_2": category_names[focal[1]],
                "moderator": category_names[moderator],
                "label": f"{category_names[focal[0]]} × {category_names[focal[1]]} | {category_names[moderator]}",
                "rest_size": int(r_row["rest_size"]),
                "theta_h1": float(source.theta_h1),
                "z_h1": float(source.z_h1),
                "observed_h2_lambda": observed,
                "observed_h2_ci_low": observed - 1.96 * se,
                "observed_h2_ci_high": observed + 1.96 * se,
                "baseline_prediction": float(r_row["baseline_prediction"]),
                "csid3_prediction": float(r_row["csid3_prediction"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values("observed_h2_lambda").reset_index(drop=True)
    frame["display_order_h2"] = np.arange(1, len(frame) + 1)
    return frame


def run_concordance_validation(
    fit_h1: CSIDFit,
    fit_h2: CSIDFit,
    pair_fit_h1: CSIDFit,
    *,
    category_names: Sequence[str],
    thresholds: Sequence[float] = tuple(np.arange(0.0, 4.01, 0.5)),
    primary_threshold: float = 2.0,
    permutations: int = 199,
    seed: int = 42,
    n_examples: int = 8,
    min_rest_size: int = 2,
    external_triple_candidates: np.ndarray | None = None,
    external_theta: np.ndarray | None = None,
    benchmark_fractions: Sequence[float] = (1.0, 0.75, 0.50, 0.25, 0.10),
    benchmark_primary_fraction: float = 0.25,
) -> ConcordanceResult:
    layers = build_transfer_layers(fit_h1, fit_h2, category_names=category_names, min_rest_size=min_rest_size)
    if external_triple_candidates is not None or external_theta is not None:
        if external_triple_candidates is None or external_theta is None:
            raise ValueError("external candidates and coefficients must be provided together")
        layers = add_transfer_prediction(
            layers,
            candidates=external_triple_candidates,
            theta=external_theta,
        )
    sweep = threshold_sweep(layers, thresholds)
    primary = sweep[np.isclose(sweep["h1_z_threshold"], primary_threshold)].copy()
    decomposition = _primary_decomposition(primary)
    loco = category_loco(layers, threshold=primary_threshold, n_categories=len(category_names))
    if not primary.empty:
        for metric in ("delta_ccc", "delta_pearson", "delta_bias_correction"):
            primary[f"{metric}_loco_min"] = loco.attrs.get(f"{metric}_loco_min", np.nan)
            primary[f"{metric}_loco_max"] = loco.attrs.get(f"{metric}_loco_max", np.nan)
    permutation_frame, permutation_summary = permutation_test(
        layers,
        threshold=primary_threshold,
        replicates=permutations,
        seed=seed,
    )
    examples = select_conditional_odds_examples(
        layers,
        pair_fit_h1,
        category_names=category_names,
        n_examples=n_examples,
    )
    benchmark_sweep = external_benchmark_sweep(layers, benchmark_fractions)
    benchmark_primary = (
        benchmark_sweep[
            np.isclose(
                benchmark_sweep["information_fraction"],
                float(benchmark_primary_fraction),
            )
        ].copy()
        if not benchmark_sweep.empty
        else pd.DataFrame()
    )
    benchmark_loco = external_benchmark_loco(
        layers,
        information_fraction=benchmark_primary_fraction,
        n_categories=len(category_names),
    )
    if not benchmark_primary.empty and not benchmark_loco.empty:
        benchmark_primary["csid_minus_pl_ccc_loco_min"] = benchmark_loco.attrs.get(
            "csid_minus_pl_ccc_loco_min",
            np.nan,
        )
        benchmark_primary["csid_minus_pl_ccc_loco_max"] = benchmark_loco.attrs.get(
            "csid_minus_pl_ccc_loco_max",
            np.nan,
        )
    full_transfer = full_transfer_summary(
        layers,
        information_fraction=benchmark_primary_fraction,
    )
    return ConcordanceResult(
        layers=layers,
        sweep=sweep,
        primary=primary,
        decomposition=decomposition,
        loco=loco,
        permutations=permutation_frame,
        permutation_summary=permutation_summary,
        examples=examples,
        benchmark_sweep=benchmark_sweep,
        benchmark_primary=benchmark_primary,
        benchmark_loco=benchmark_loco,
        full_transfer=full_transfer,
    )


def plot_concordance_thresholds(
    sweeps: Mapping[str, pd.DataFrame],
    path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    valid = [(name, frame) for name, frame in sweeps.items() if not frame.empty]
    if not valid:
        return
    fig, axes = plt.subplots(1, len(valid), figsize=(5.3 * len(valid), 4.5), sharey=True, squeeze=False)
    for ax, (name, frame) in zip(axes[0], valid, strict=False):
        benchmark = "information_fraction" in frame and "pl_ccc" in frame
        x_column = "information_fraction" if benchmark else "h1_z_threshold"
        data = frame.sort_values(x_column)
        ax.plot(
            data[x_column],
            data["baseline_ccc"],
            marker="o",
            linestyle="--",
            label="Common-effect baseline",
        )
        ax.plot(
            data[x_column],
            data["csid_ccc"],
            marker="o",
            label="CSID-3 transfer",
        )
        if benchmark:
            ax.plot(
                data[x_column],
                data["pl_ccc"],
                marker="o",
                linestyle="-.",
                label="Cardinality-aware pseudolikelihood",
            )
        ax.set_title(name, fontsize=18)
        ax.set_xlabel(
            "H1 information retained fraction"
            if benchmark
            else r"H1 screening threshold $|z_T|$",
            fontsize=16,
        )
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Lin's CCC", fontsize=16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3 if any("pl_ccc" in frame for _, frame in valid) else 2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=16,
        markerscale=1.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    if output.suffix.lower() == ".png":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _wrap_conditional_odds_label(item1: str, item2: str, moderator: str) -> str:
    return f"{item1}\n{item2}\n{moderator}"


def plot_conditional_odds(
    examples: Mapping[str, pd.DataFrame],
    path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    valid = [(name, frame) for name, frame in examples.items() if not frame.empty]
    if not valid:
        return

    colors = {"observed": "#0072B2", "m2": "#E69F00", "m3": "#009E73"}

    max_rows = max(len(frame) for _, frame in valid)
    fig, axes = plt.subplots(1, len(valid), figsize=(18.0, max(6.0, 0.58 * max_rows + 2.4)), squeeze=False)
    artists = None

    for ax, (name, frame) in zip(axes[0], valid, strict=False):
        data = frame.sort_values("observed_h2_lambda", ascending=True, kind="stable").reset_index(drop=True)
        y = np.arange(len(data), dtype=float)[::-1]
        observed = data["observed_h2_lambda"].to_numpy(dtype=float)
        ci_low = data["observed_h2_ci_low"].to_numpy(dtype=float)
        ci_high = data["observed_h2_ci_high"].to_numpy(dtype=float)
        xerr = np.vstack([observed - ci_low, ci_high - observed])
        if name in {"Complete Journey", "Instacart"}:
            labels = [
                _wrap_conditional_odds_label(
                    str(row.focal_item_1),
                    str(row.focal_item_2),
                    str(row.moderator),
                )
                for row in data.itertuples(index=False)
            ]
        else:
            labels = [
                f"{str(row.focal_item_1)}–{str(row.focal_item_2)} | {str(row.moderator)}"
                for row in data.itertuples(index=False)
            ]

        observed_artist = ax.errorbar(
            observed,
            y,
            xerr=xerr,
            marker="o",
            linestyle="none",
            capsize=3,
            color=colors["observed"],
            ecolor=colors["observed"],
            label="Observed H2 (95% interval)",
        )
        m2_artist = ax.scatter(
            data["baseline_prediction"],
            y + 0.16,
            marker="^",
            color=colors["m2"],
            label="Common-effect baseline",
        )
        m3_artist = ax.scatter(
            data["csid3_prediction"],
            y - 0.16,
            marker="s",
            color=colors["m3"],
            label="CSID-3 transfer",
        )
        artists = (m2_artist, m3_artist, observed_artist)
        values = np.concatenate([ci_low, ci_high, data["baseline_prediction"], data["csid3_prediction"], [0.0]])
        finite = values[np.isfinite(values)]
        if finite.size:
            margin = max(0.08, 0.08 * max(float(finite.max() - finite.min()), 1.0))
            ax.set_xlim(float(finite.min()) - margin, float(finite.max()) + margin)
        ax.axvline(0.0, linestyle="--", linewidth=1.0)
        ax.set_yticks(y, labels)
        ax.set_title(name, fontsize=18)
        ax.set_xlabel(r"$\Lambda_{ij\ell}$", fontsize=16)
        ax.grid(axis="x", alpha=0.25)

    if artists is not None:
        fig.legend(
            list(artists),
            [
                "Common-effect baseline",
                "CSID-3 transfer",
                "Observed H2 (95% interval)",
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
            ncol=3,
            fontsize=16,
            markerscale=1.5,
        )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    if output.suffix.lower() == ".png":
        fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
