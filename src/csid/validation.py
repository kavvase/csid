"""Period stability statistics for CSID estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .estimator import CSIDFit
from .stats import pearson_correlation


def _aligned(fit_h1: CSIDFit, fit_h2: CSIDFit) -> pd.DataFrame:
    index_h2 = {tuple(map(int, c)): q for q, c in enumerate(fit_h2.candidates)}
    rows: list[dict[str, float | int | str]] = []
    for q1, candidate in enumerate(fit_h1.candidates):
        key = tuple(map(int, candidate))
        q2 = index_h2.get(key)
        if q2 is None or fit_h1.mu[q1] <= 0 or fit_h2.mu[q2] <= 0:
            continue
        rows.append(
            {
                "candidate": "|".join(map(str, key)),
                "item_1_idx": key[0],
                "item_2_idx": key[1],
                "item_3_idx": key[2],
                "theta_h1": float(fit_h1.theta[q1]),
                "theta_h2": float(fit_h2.theta[q2]),
                "mu_h1": float(fit_h1.mu[q1]),
                "mu_h2": float(fit_h2.mu[q2]),
                "information": float(min(fit_h1.mu[q1], fit_h2.mu[q2])),
            }
        )
    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    a = frame["theta_h1"].to_numpy(float)
    b = frame["theta_h2"].to_numpy(float)
    n = len(frame)
    k = min(20, n)
    neg_h1 = set(frame.nsmallest(k, "theta_h1")["candidate"])
    neg_h2 = set(frame.nsmallest(k, "theta_h2")["candidate"])
    return {
        "filter": label,
        "N": n,
        "pearson": pearson_correlation(a, b),
        "spearman": float(spearmanr(a, b).statistic) if n > 1 else float("nan"),
        "sign_agree": float(np.mean(np.sign(a) == np.sign(b))) if n else float("nan"),
        "negative_top20_overlap": int(len(neg_h1 & neg_h2)),
        "negative_top20_random_expectation": float(k * k / n) if n else float("nan"),
    }


def temporal_stability(fit_h1: CSIDFit, fit_h2: CSIDFit) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return threshold summaries and aligned candidate rows."""

    aligned = _aligned(fit_h1, fit_h2)
    if aligned.empty:
        return pd.DataFrame(), aligned
    rows = []
    for label, fraction in (("all", 1.0), ("top50%", 0.50), ("top25%", 0.25), ("top10%", 0.10)):
        if fraction == 1.0:
            subset = aligned
        else:
            cutoff = float(aligned["information"].quantile(1.0 - fraction))
            subset = aligned[aligned["information"] >= cutoff]
        rows.append(_summary(subset, label))
    return pd.DataFrame(rows), aligned


def stability_loco(
    aligned: pd.DataFrame,
    *,
    information_fraction: float = 0.25,
    n_categories: int,
) -> pd.DataFrame:
    if aligned.empty:
        return pd.DataFrame()
    cutoff = float(aligned["information"].quantile(1.0 - information_fraction))
    primary = aligned[aligned["information"] >= cutoff]
    point = _summary(primary, "top25%")
    rows = []
    for category in range(int(n_categories)):
        keep = ~(
            aligned["item_1_idx"].eq(category) | aligned["item_2_idx"].eq(category) | aligned["item_3_idx"].eq(category)
        )
        reduced = aligned.loc[keep]
        if reduced.empty:
            continue
        reduced_cutoff = float(reduced["information"].quantile(1.0 - information_fraction))
        row = _summary(reduced[reduced["information"] >= reduced_cutoff], "top25%")
        row["excluded_category"] = category
        rows.append(row)
    frame = pd.DataFrame(rows)
    for metric in ("pearson", "sign_agree"):
        frame.attrs[f"{metric}_point"] = float(point[metric])
        frame.attrs[f"{metric}_loco_min"] = float(frame[metric].min())
        frame.attrs[f"{metric}_loco_max"] = float(frame[metric].max())
    return frame
