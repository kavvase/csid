"""Shared correlation and resampling helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pearson_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    return float(np.corrcoef(x, y)[0, 1]), float(spearmanr(x, y).statistic)


def jackknife_interval(values: np.ndarray, point: float) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    pseudo = len(x) * point - (len(x) - 1) * x
    se = float(np.std(pseudo, ddof=1) / np.sqrt(len(x)))
    return point - 1.96 * se, point + 1.96 * se
