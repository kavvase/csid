"""Command-line interface for the CSID paper reproduction package."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from . import __version__
from .analysis import AnalysisConfig
from .data import (
    default_cj_category_file,
    load_complete_journey,
    load_generic,
    load_instacart,
    load_ta_feng,
    resolve_dataset_root,
)
from .figures import build_paper_outputs
from .paper import ItemUniverseAuditOptions, PaperConfig, run_dataset, run_item_universe_audit
from .robustness import (
    run_sensitivity_scenario,
    summarize_sensitivity_analysis,
    write_sensitivity_results,
)
from .synthetic import (
    cardinality_confounding_experiment,
    plot_cardinality_confounding,
    plot_synthetic_detection_power,
    randomized_triple_power_experiment,
    saturation_deprojection_experiment,
)


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _universe_sizes(value: str) -> tuple[int | str, ...]:
    result: list[int | str] = []
    for part in value.split(","):
        text = part.strip().lower()
        if not text:
            continue
        result.append("all" if text == "all" else int(text))
    if not result:
        raise argparse.ArgumentTypeError("universe sizes must not be empty")
    return tuple(result)


def _default_category_file() -> Path | None:
    path = default_cj_category_file()
    return path if path.exists() else None


def _analysis_config(args: argparse.Namespace, *, triple_min_count: int) -> AnalysisConfig:
    return AnalysisConfig(
        candidate_mode=args.candidate_mode,
        exhaustive_max_categories=args.exhaustive_max_categories,
        pair_candidate_min_count=args.pair_candidate_min_count,
        pair_model_min_count=args.pair_model_min_count,
        triple_min_count=triple_min_count,
        top_l_neighbors=args.top_l_neighbors,
        max_candidates=args.max_candidates,
        epsilon=args.epsilon,
        ridge2=args.ridge2,
        ridge3=args.ridge3,
        min_rest_size=args.min_rest_size,
        memory_limit_mb=args.memory_limit_mb,
        block_size=args.block_size,
    )


def _paper_config(args: argparse.Namespace, *, triple_min_count: int) -> PaperConfig:
    return PaperConfig(
        analysis=_analysis_config(args, triple_min_count=triple_min_count),
        ccc_thresholds=_floats(args.ccc_thresholds),
        primary_z=args.primary_z,
        ccc_permutations=args.ccc_permutations,
        conditional_examples=args.conditional_examples,
        seed=args.seed,
        pl_ridge_grid=_floats(getattr(args, "pl_ridge_grid", "0.1,1,10,100,1000")),
        pl_validation_fraction=getattr(args, "pl_validation_fraction", 0.2),
        pl_max_iter=getattr(args, "pl_max_iter", 500),
        benchmark_information_fractions=_floats(
            getattr(args, "benchmark_information_fractions", "1,0.75,0.5,0.25,0.1")
        ),
        benchmark_primary_fraction=getattr(args, "benchmark_primary_fraction", 0.25),
    )


def _load_dataset(args: argparse.Namespace):
    if args.dataset == "complete-journey":
        return load_complete_journey(
            resolve_dataset_root(args.data_root, "complete-journey"),
            category_file=args.category_file or _default_category_file(),
            top_n=args.top_n,
        )
    if args.dataset == "ta-feng":
        return load_ta_feng(
            resolve_dataset_root(args.data_root, "ta-feng"),
            top_n=args.top_n,
            min_category_support=args.ta_feng_min_category_support,
        )
    if args.dataset == "instacart":
        return load_instacart(
            resolve_dataset_root(args.data_root, "instacart"),
            n_users=args.n_users,
            seed=args.seed,
            top_n=args.top_n,
        )
    return load_generic(
        args.input,
        basket_col=args.basket_col,
        item_col=args.item_col,
        period_col=args.period_col,
        cluster_col=args.cluster_col,
        top_n=args.top_n,
    )


def run_synthetic(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": __version__,
        "cardinality": {
            "strengths": _floats(args.cardinality_strengths),
            "replicates": args.cardinality_replicates,
            "baskets": args.cardinality_baskets,
            "categories": args.cardinality_categories,
            "seed": args.cardinality_seed,
            "ising_l2": args.cardinality_ising_l2,
            "csid_epsilon": 0.5,
            "csid_ridge": 1.0,
            "min_rest_size": 1,
        },
        "detection": {
            "effect_sizes": _floats(args.effect_sizes),
            "replicates": args.power_replicates,
            "baskets": args.power_baskets,
            "categories": args.power_categories,
            "planted": args.power_planted,
            "seed": args.seed,
            "csid_epsilon": 0.5,
            "csid_ridge": 1.0,
            "min_rest_size": 1,
        },
        "saturation": {
            "baskets": args.saturation_baskets,
            "seed": args.saturation_seed,
            "csid_epsilon": 0.5,
            "csid_ridge2": 1.0,
            "csid_ridge3": 1.0,
            "min_rest_size": 1,
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    cardinality_replicates, cardinality_summary = cardinality_confounding_experiment(
        cardinality_strengths=_floats(args.cardinality_strengths),
        replicates=args.cardinality_replicates,
        n_baskets=args.cardinality_baskets,
        n_categories=args.cardinality_categories,
        seed=args.cardinality_seed,
        ising_l2=args.cardinality_ising_l2,
    )
    cardinality_replicates.to_csv(output / "cardinality_confounding_replicates.csv", index=False)
    cardinality_summary.to_csv(output / "cardinality_confounding_summary.csv", index=False)
    plot_cardinality_confounding(cardinality_summary, output / "cardinality_confounding.png")
    replicates, summary = randomized_triple_power_experiment(
        effect_sizes=_floats(args.effect_sizes),
        replicates=args.power_replicates,
        n_baskets=args.power_baskets,
        n_categories=args.power_categories,
        n_planted=args.power_planted,
        seed=args.seed,
    )
    replicates.to_csv(output / "detection_replicates.csv", index=False)
    summary.to_csv(output / "detection_summary.csv", index=False)
    plot_synthetic_detection_power(summary, output / "synthetic_detection_power.png")
    plot_synthetic_detection_power(summary, output / "synthetic_detection_power.pdf")
    saturation, detail = saturation_deprojection_experiment(
        n_samples=args.saturation_baskets,
        seed=args.saturation_seed,
    )
    saturation.to_csv(output / "saturation_summary.csv", index=False)
    detail.to_csv(output / "saturation_detail.csv", index=False)
    text = [
        "# Synthetic validation summary",
        "",
        "## Pairwise Ising contamination under a cardinality-only model",
        "",
        cardinality_summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "![Cardinality confounding](cardinality_confounding.png)",
        "",
        "## Randomized sparse-triple detection",
        "",
        summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "![Synthetic detection power](synthetic_detection_power.png)",
        "",
        "## Pair recovery under third-order saturation",
        "",
        saturation.to_markdown(index=False, floatfmt=".4g"),
        "",
    ]
    (output / "summary.md").write_text("\n".join(text), encoding="utf-8")


def run_analyze(args: argparse.Namespace) -> None:
    dataset = _load_dataset(args)
    default_min = {"complete-journey": 0, "ta-feng": 10, "instacart": 30}.get(args.dataset, 0)
    triple_min = default_min if args.triple_min_count is None else args.triple_min_count
    run_dataset(dataset, args.output, config=_paper_config(args, triple_min_count=triple_min))


def run_universe(args: argparse.Namespace) -> None:
    run_item_universe_audit(
        args.output,
        resolve_dataset_root(args.data_root, "instacart"),
        config=_paper_config(args, triple_min_count=args.triple_min_count),
        options=ItemUniverseAuditOptions(
            n_users=args.n_users,
            focus_n=args.focus_n,
            universe_sizes=_universe_sizes(args.universe_sizes),
            triple_min_count=args.triple_min_count,
            chunksize=args.chunksize,
            min_focus_basket_size=args.min_focus_basket_size,
            exponent_clip=args.exponent_clip,
        ),
    )


def run_paper(args: argparse.Namespace) -> None:
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    synthetic_args = argparse.Namespace(**vars(args))
    synthetic_args.output = root / "synthetic"
    synthetic_args.seed = args.synthetic_seed
    run_synthetic(synthetic_args)

    category_file = args.category_file or _default_category_file()
    datasets = [
        (
            "complete-journey",
            load_complete_journey(
                resolve_dataset_root(args.data_root, "complete-journey"),
                category_file=category_file,
                top_n=20,
            ),
            0,
        ),
        (
            "ta-feng",
            load_ta_feng(
                resolve_dataset_root(args.data_root, "ta-feng"),
                top_n=20,
                min_category_support=1000,
            ),
            10,
        ),
        (
            "instacart",
            load_instacart(
                resolve_dataset_root(args.data_root, "instacart"),
                n_users=30000,
                seed=args.seed,
                top_n=20,
            ),
            30,
        ),
    ]
    sensitivity_rows: list[dict[str, object]] = []
    for name, dataset, triple_min in datasets:
        base_config = _paper_config(args, triple_min_count=triple_min)
        result = run_dataset(
            dataset,
            root / name,
            config=base_config,
        )
        sensitivity_rows.append(
            summarize_sensitivity_analysis(
                result.analysis,
                scenario="default",
                config=base_config.analysis,
                external_candidates=result.pseudolikelihood.triple_candidates,
                external_theta=result.pseudolikelihood.triple_theta,
            )
        )
        variants = (
            ("epsilon_0.25", replace(base_config.analysis, epsilon=0.25)),
            ("epsilon_1", replace(base_config.analysis, epsilon=1.0)),
            ("ridge_0.1", replace(base_config.analysis, ridge2=0.1, ridge3=0.1)),
            ("ridge_10", replace(base_config.analysis, ridge2=10.0, ridge3=10.0)),
        )
        for scenario, analysis_config in variants:
            sensitivity_rows.append(
                run_sensitivity_scenario(
                    dataset,
                    scenario=scenario,
                    config=analysis_config,
                    external_candidates=result.pseudolikelihood.triple_candidates,
                    external_theta=result.pseudolikelihood.triple_theta,
                    cache_dir=root / "robustness" / "cache" / name / scenario,
                )
            )

    singleton_datasets = [
        (
            "complete-journey",
            load_complete_journey(
                resolve_dataset_root(args.data_root, "complete-journey"),
                category_file=category_file,
                top_n=20,
                min_basket_size=1,
            ),
            0,
        ),
        (
            "ta-feng",
            load_ta_feng(
                resolve_dataset_root(args.data_root, "ta-feng"),
                top_n=20,
                min_category_support=1000,
                min_basket_size=1,
            ),
            10,
        ),
        (
            "instacart",
            load_instacart(
                resolve_dataset_root(args.data_root, "instacart"),
                n_users=30000,
                seed=args.seed,
                top_n=20,
                min_basket_size=1,
            ),
            30,
        ),
    ]
    for name, dataset, triple_min in singleton_datasets:
        base_config = _paper_config(args, triple_min_count=triple_min)
        r1_config = replace(
            base_config,
            analysis=replace(
                base_config.analysis,
                min_rest_size=1,
            ),
        )
        result = run_dataset(
            dataset,
            root / "robustness" / "r1" / name,
            config=r1_config,
        )
        sensitivity_rows.append(
            summarize_sensitivity_analysis(
                result.analysis,
                scenario="rest1_singletons",
                config=r1_config.analysis,
                external_candidates=result.pseudolikelihood.triple_candidates,
                external_theta=result.pseudolikelihood.triple_theta,
            )
        )
    write_sensitivity_results(
        sensitivity_rows,
        root / "robustness" / "sensitivity_summary.csv",
    )
    run_item_universe_audit(
        root / "item-universe-expansion",
        resolve_dataset_root(args.data_root, "instacart"),
        config=_paper_config(args, triple_min_count=30),
    )
    build_paper_outputs(root, root / "paper")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-mode", choices=["exhaustive", "sparse", "auto"], default="auto")
    parser.add_argument("--exhaustive-max-categories", type=int, default=40)
    parser.add_argument("--pair-candidate-min-count", type=int, default=5)
    parser.add_argument("--pair-model-min-count", type=int, default=0)
    parser.add_argument("--triple-min-count", type=int)
    parser.add_argument("--top-l-neighbors", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=250000)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--ridge2", type=float, default=1.0)
    parser.add_argument("--ridge3", type=float, default=1.0)
    parser.add_argument("--min-rest-size", type=int, default=2)
    parser.add_argument("--memory-limit-mb", type=float, default=512.0)
    parser.add_argument("--block-size", type=int, default=50000)
    parser.add_argument("--ccc-thresholds", default="0,0.5,1,1.5,2,2.5,3,3.5,4")
    parser.add_argument("--primary-z", type=float, default=2.0)
    parser.add_argument("--ccc-permutations", type=int, default=199)
    parser.add_argument("--conditional-examples", type=int, default=8)
    parser.add_argument("--pl-ridge-grid", default="0.1,1,10,100,1000")
    parser.add_argument("--pl-validation-fraction", type=float, default=0.2)
    parser.add_argument("--pl-max-iter", type=int, default=500)
    parser.add_argument("--benchmark-information-fractions", default="1,0.75,0.5,0.25,0.1")
    parser.add_argument("--benchmark-primary-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)


def add_synthetic_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cardinality-strengths", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--cardinality-replicates", type=int, default=20)
    parser.add_argument("--cardinality-baskets", type=int, default=10000)
    parser.add_argument("--cardinality-categories", type=int, default=10)
    parser.add_argument("--cardinality-seed", type=int, default=2026)
    parser.add_argument("--cardinality-ising-l2", type=float, default=1.0e-5)
    parser.add_argument("--effect-sizes", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--power-replicates", type=int, default=20)
    parser.add_argument("--power-baskets", type=int, default=10000)
    parser.add_argument("--power-categories", type=int, default=10)
    parser.add_argument("--power-planted", type=int, default=8)
    parser.add_argument("--saturation-baskets", type=int, default=60000)
    parser.add_argument("--saturation-seed", type=int, default=27)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="csid", description="Cardinality-Stratified Interaction Decomposition")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="run every paper analysis for one dataset")
    analyze.add_argument("--dataset", choices=["complete-journey", "ta-feng", "instacart", "generic"], required=True)
    analyze.add_argument("--data-root", default="data")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--category-file")
    analyze.add_argument("--top-n", type=int, default=20)
    analyze.add_argument("--n-users", type=int, default=30000)
    analyze.add_argument("--ta-feng-min-category-support", type=int, default=1000)
    analyze.add_argument("--input")
    analyze.add_argument("--basket-col", default="basket_id")
    analyze.add_argument("--item-col", default="item")
    analyze.add_argument("--period-col", default="period")
    analyze.add_argument("--cluster-col")
    add_common(analyze)
    analyze.set_defaults(func=run_analyze)

    synthetic = sub.add_parser("synthetic", help="run the paper's synthetic experiments")
    synthetic.add_argument("--output", required=True)
    add_synthetic_options(synthetic)
    synthetic.add_argument("--seed", type=int, default=17)
    synthetic.set_defaults(func=run_synthetic)

    universe = sub.add_parser(
        "universe",
        help="audit robustness while expanding the Instacart aisle universe",
    )
    universe.add_argument("--data-root", default="data/instacart")
    universe.add_argument("--output", required=True)
    universe.add_argument("--n-users", type=int, default=30000)
    universe.add_argument("--focus-n", type=int, default=20)
    universe.add_argument("--universe-sizes", default="20,40,80,all")
    universe.add_argument("--chunksize", type=int, default=2000000)
    universe.add_argument("--min-focus-basket-size", type=int, default=2)
    universe.add_argument("--triple-min-count", type=int, default=30)
    universe.add_argument("--epsilon", type=float, default=0.5)
    universe.add_argument("--ridge2", type=float, default=1.0)
    universe.add_argument("--ridge3", type=float, default=1.0)
    universe.add_argument("--min-rest-size", type=int, default=2)
    universe.add_argument("--memory-limit-mb", type=float, default=512.0)
    universe.add_argument("--block-size", type=int, default=50000)
    universe.add_argument("--primary-z", type=float, default=2.0)
    universe.add_argument("--ccc-thresholds", default="0,1,2,3,4")
    universe.add_argument("--exponent-clip", type=float, default=20.0)
    universe.add_argument("--seed", type=int, default=42)
    universe.set_defaults(func=run_universe)

    figures = sub.add_parser("figures", help="build paper figures and tables from a results root")
    figures.add_argument("--results-root", required=True)
    figures.add_argument("--output", required=True)
    figures.set_defaults(func=lambda a: build_paper_outputs(a.results_root, a.output))

    paper = sub.add_parser("paper", help="run all three public datasets, synthetic experiments, figures, and tables")
    paper.add_argument("--data-root", default="data")
    paper.add_argument("--output", required=True)
    paper.add_argument("--category-file")
    paper.add_argument("--synthetic-seed", type=int, default=17)
    add_common(paper)
    add_synthetic_options(paper)
    paper.set_defaults(func=run_paper)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
