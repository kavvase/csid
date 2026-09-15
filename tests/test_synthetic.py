from inspect import signature

from csid.cli import build_parser
from csid.synthetic import randomized_triple_power_experiment, saturation_deprojection_experiment


def test_synthetic_experiments_smoke():
    replicates, summary = randomized_triple_power_experiment(
        effect_sizes=(0.2,), replicates=2, n_baskets=1200, n_categories=7, n_planted=4, seed=11
    )
    assert len(replicates) == 2
    assert len(summary) == 1
    assert 0 <= summary.iloc[0]["power_abs_z_gt_2"] <= 1

    saturation, detail = saturation_deprojection_experiment(n_samples=5000, seed=23)
    assert set(saturation["method"]) == {"CSID-2", "CSID-3 deprojected"}
    assert len(detail) == 15


def test_synthetic_function_and_cli_seed_defaults_match():
    args = build_parser().parse_args(["synthetic", "--output", "unused"])
    assert signature(randomized_triple_power_experiment).parameters["seed"].default == args.seed
    assert signature(saturation_deprojection_experiment).parameters["seed"].default == args.saturation_seed
