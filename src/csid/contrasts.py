from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .counts import SizeStratifiedCounts
from .keys import pair_key


@dataclass(frozen=True)
class ContrastMatrix:
    k: int
    candidates: np.ndarray
    r_values: np.ndarray
    lambda_hat: np.ndarray
    weights: np.ndarray
    n_all_total: np.ndarray
    epsilon: float
    min_rest_size: int
    n_baskets: int
    storage_paths: tuple[str, ...] = ()

    @property
    def mu(self) -> np.ndarray:
        return np.asarray(self.weights.sum(axis=1), dtype=float)

    @property
    def n_candidates(self) -> int:
        return int(len(self.candidates))


def _allocate_matrix(
    shape: tuple[int, int],
    *,
    name: str,
    work_dir: str | Path | None,
    memory_limit_mb: float,
) -> tuple[np.ndarray, str | None]:
    nbytes = int(np.prod(shape)) * np.dtype(np.float64).itemsize
    if nbytes <= float(memory_limit_mb) * 1024 * 1024:
        return np.zeros(shape, dtype=np.float64), None
    directory = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="csid_"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.dat"
    array = np.memmap(path, mode="w+", dtype=np.float64, shape=shape)
    array[:] = 0.0
    return array, str(path)


def _shift(values: np.ndarray, offset: int, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.float64)
    if offset >= length:
        return out
    available = min(length - offset, len(values) - offset)
    if available > 0:
        out[:available] = values[offset : offset + available]
    return out


def _clip_counts(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.maximum(np.asarray(value, dtype=float), 0.0) for value in arrays)


def build_contrast_matrix(
    counts: SizeStratifiedCounts,
    *,
    k: int,
    epsilon: float = 0.5,
    min_rest_size: int = 1,
    memory_limit_mb: float = 512.0,
    work_dir: str | Path | None = None,
) -> ContrastMatrix:
    """Build size-stratified log-odds contrasts for selected candidates only.

    Only lambda and inverse-variance weights are retained.  The full 2^k cell
    tensor is reconstructed row by row and discarded, which keeps memory linear
    in ``n_candidates * observed_basket_sizes``.
    """

    if k not in {2, 3}:
        raise ValueError("Only k=2 and k=3 are implemented")
    candidates = counts.pair_candidates if k == 2 else counts.triple_candidates
    candidate_counts = counts.pair_by_size if k == 2 else counts.triple_by_size
    Q = len(candidates)
    R = counts.max_basket_size + 1
    lambda_hat, lambda_path = _allocate_matrix(
        (Q, R), name=f"lambda_k{k}", work_dir=work_dir, memory_limit_mb=memory_limit_mb / 2
    )
    weights, weight_path = _allocate_matrix(
        (Q, R), name=f"weights_k{k}", work_dir=work_dir, memory_limit_mb=memory_limit_mb / 2
    )
    n_all_total = candidate_counts.sum(axis=1).astype(float)
    r_values = np.arange(R, dtype=np.int32)
    N = np.asarray(counts.N_by_size, dtype=float)
    singleton = np.asarray(counts.singleton_by_size, dtype=float)
    pair_map = {(int(i), int(j)): idx for idx, (i, j) in enumerate(counts.pair_candidates)}

    for q, candidate in enumerate(candidates):
        if k == 2:
            i, j = map(int, candidate)
            pij = np.asarray(counts.pair_by_size[q], dtype=float)
            ci, cj = singleton[i], singleton[j]
            n00, n10, n01, n11 = _clip_counts(
                _shift(N, 0, R) - _shift(ci, 0, R) - _shift(cj, 0, R) + _shift(pij, 0, R),
                _shift(ci, 1, R) - _shift(pij, 1, R),
                _shift(cj, 1, R) - _shift(pij, 1, R),
                _shift(pij, 2, R),
            )
            cells = np.vstack([n00, n10, n01, n11]).T
            signs = np.asarray([1.0, -1.0, -1.0, 1.0])
        else:
            i, j, k_item = map(int, candidate)
            try:
                cij = counts.pair_by_size[pair_map[pair_key(i, j)]]
                cik = counts.pair_by_size[pair_map[pair_key(i, k_item)]]
                cjk = counts.pair_by_size[pair_map[pair_key(j, k_item)]]
            except KeyError as exc:
                raise ValueError(f"Missing pair edge for triple {tuple(candidate)}") from exc
            cijk = np.asarray(counts.triple_by_size[q], dtype=float)
            ci, cj, ck = singleton[i], singleton[j], singleton[k_item]
            n000, n100, n010, n001, n110, n101, n011, n111 = _clip_counts(
                _shift(N, 0, R)
                - _shift(ci, 0, R)
                - _shift(cj, 0, R)
                - _shift(ck, 0, R)
                + _shift(cij, 0, R)
                + _shift(cik, 0, R)
                + _shift(cjk, 0, R)
                - _shift(cijk, 0, R),
                _shift(ci, 1, R) - _shift(cij, 1, R) - _shift(cik, 1, R) + _shift(cijk, 1, R),
                _shift(cj, 1, R) - _shift(cij, 1, R) - _shift(cjk, 1, R) + _shift(cijk, 1, R),
                _shift(ck, 1, R) - _shift(cik, 1, R) - _shift(cjk, 1, R) + _shift(cijk, 1, R),
                _shift(cij, 2, R) - _shift(cijk, 2, R),
                _shift(cik, 2, R) - _shift(cijk, 2, R),
                _shift(cjk, 2, R) - _shift(cijk, 2, R),
                _shift(cijk, 3, R),
            )
            cells = np.vstack([n000, n100, n010, n001, n110, n101, n011, n111]).T
            signs = np.asarray([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0])

        raw_total = cells.sum(axis=1)
        smooth = cells + float(epsilon)
        row_lambda = np.sum(np.log(smooth) * signs[None, :], axis=1)
        row_weight = 1.0 / np.sum(1.0 / smooth, axis=1)
        valid = (r_values >= int(min_rest_size)) & (raw_total > 0)
        lambda_hat[q] = np.where(valid, row_lambda, 0.0)
        weights[q] = np.where(valid, row_weight, 0.0)

    paths = tuple(path for path in (lambda_path, weight_path) if path is not None)
    return ContrastMatrix(
        k=k,
        candidates=np.asarray(candidates, dtype=np.int32),
        r_values=r_values,
        lambda_hat=lambda_hat,
        weights=weights,
        n_all_total=n_all_total,
        epsilon=float(epsilon),
        min_rest_size=int(min_rest_size),
        n_baskets=int(counts.n_baskets),
        storage_paths=paths,
    )
