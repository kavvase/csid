from itertools import combinations

import numpy as np

from csid.contrasts import build_contrast_matrix
from csid.counts import count_for_candidates, scan_pairs
from csid.data import SparseBasketDataset


def direct_cells(X, candidate, rest_size):
    candidate = tuple(candidate)
    rows = []
    for pattern in range(1 << len(candidate)):
        count = 0
        bits = [(pattern >> p) & 1 for p in range(len(candidate))]
        for row in X:
            if [int(row[i]) for i in candidate] != bits:
                continue
            rest = int(row.sum() - sum(bits))
            if rest == rest_size:
                count += 1
        rows.append(count)
    return np.asarray(rows)


def test_inclusion_exclusion_contrasts_match_direct_cells():
    X = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    dataset = SparseBasketDataset.from_dense(X)
    pairs = np.asarray(list(combinations(range(4), 2)), dtype=np.int32)
    triples = np.asarray(list(combinations(range(4), 3)), dtype=np.int32)
    counts = count_for_candidates(
        dataset,
        pair_candidates=pairs,
        triple_candidates=triples,
        base_scan=scan_pairs(dataset),
    )
    eps = 0.5
    pair_table = build_contrast_matrix(counts, k=2, epsilon=eps, min_rest_size=0)
    triple_table = build_contrast_matrix(counts, k=3, epsilon=eps, min_rest_size=0)

    pair = tuple(pair_table.candidates[0])
    for r in range(pair_table.lambda_hat.shape[1]):
        cells = direct_cells(X, pair, r)
        expected = np.log((cells[3] + eps) * (cells[0] + eps) / ((cells[1] + eps) * (cells[2] + eps)))
        assert np.isclose(pair_table.lambda_hat[0, r], expected)

    triple = tuple(triple_table.candidates[0])
    signs = np.asarray([-1, 1, 1, -1, 1, -1, -1, 1])
    for r in range(triple_table.lambda_hat.shape[1]):
        cells = direct_cells(X, triple, r)
        expected = float(np.dot(signs, np.log(cells + eps)))
        assert np.isclose(triple_table.lambda_hat[0, r], expected)
