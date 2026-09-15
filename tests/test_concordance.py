import numpy as np
import pandas as pd

from csid.concordance import (
    _permuted_predictions,
    add_transfer_prediction,
    external_benchmark_sweep,
    full_transfer_summary,
    weighted_ccc,
)


def test_weighted_ccc_identity_and_scale_penalty():
    y = np.asarray([-2.0, -1.0, 1.0, 2.0])
    w = np.asarray([1.0, 2.0, 2.0, 1.0])
    same = weighted_ccc(y, y, w)
    scaled = weighted_ccc(y, 2.0 * y, w)
    reversed_ = weighted_ccc(y, -y, w)
    assert np.isclose(same["ccc"], 1.0)
    assert scaled["pearson"] > 0.999
    assert scaled["ccc"] < 1.0
    assert reversed_["ccc"] < 0.0


def test_permuted_predictions_use_all_reference_candidates():
    layers = pd.DataFrame(
        {
            "candidate": ["a", "b", "c"],
            "rest_size": [2, 2, 2],
            "h2_layer_weight": [1.0, 1.0, 1.0],
            "observed_h2_lambda": [1.0, 2.0, 9.0],
        }
    )
    prediction = _permuted_predictions(layers, {"a": 0.0, "b": 0.0, "c": 0.0})
    assert np.isclose(prediction[0], (2.0 + 9.0) / 2.0)


def test_external_transfer_is_invariant_to_common_coefficient_shift():
    layers = pd.DataFrame(
        {
            "candidate": ["0|1|2", "0|1|3", "0|2|3"],
            "rest_size": [2, 2, 2],
            "h2_layer_weight": [2.0, 1.0, 3.0],
            "observed_h2_lambda": [0.4, -0.2, 0.8],
            "information_weight": [1.0, 1.0, 1.0],
            "mu_h1": [5.0, 4.0, 3.0],
            "baseline_prediction": [0.3, 0.5, 0.1],
            "csid3_prediction": [0.35, -0.1, 0.7],
        }
    )
    candidates = np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3]])
    theta = np.asarray([0.2, -0.1, 0.4])
    original = add_transfer_prediction(layers, candidates=candidates, theta=theta)
    shifted = add_transfer_prediction(layers, candidates=candidates, theta=theta + 7.0)
    assert np.allclose(original["pl_prediction"], shifted["pl_prediction"])

    sweep = external_benchmark_sweep(original, (1.0,))
    assert len(sweep) == 1
    assert np.isfinite(float(sweep.iloc[0]["csid_minus_pl_ccc"]))


def test_full_transfer_summary_uses_h1_only_predictions():
    layers = pd.DataFrame(
        {
            "candidate": ["a", "b", "c"],
            "mu_h1": [3.0, 2.0, 1.0],
            "information_weight": [1.0, 1.0, 1.0],
            "observed_h2_lambda": [0.0, 1.0, 2.0],
            "h1_common_prediction": [0.0, 0.0, 0.0],
            "full_transfer_prediction": [0.0, 1.0, 2.0],
            "csid3_prediction": [0.0, 1.0, 2.0],
        }
    )
    result = full_transfer_summary(layers, information_fraction=1.0)
    assert len(result) == 1
    assert np.isclose(result.iloc[0]["full_csid_ccc"], 1.0)
    assert result.iloc[0]["full_csid_minus_h1_common_ccc"] > 0
