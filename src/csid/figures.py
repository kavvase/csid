"""Consolidated paper figures and tables."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .concordance import plot_concordance_thresholds, plot_conditional_odds
from .synthetic import plot_cardinality_confounding, plot_synthetic_detection_power
from .universe import build_item_universe_coefficient_comparisons, plot_item_universe_coefficient_scatter

DATASET_DIRS = {
    "Complete Journey": "complete-journey",
    "Ta-Feng": "ta-feng",
    "Instacart": "instacart",
}

CATEGORY_DESCRIPTIONS = {
    "Complete Journey": "20 predefined product_category values",
    "Ta-Feng": "Top 20 two-digit PRODUCT_SUBCLASS codes",
    "Instacart": "Top 20 aisles (30,000-user subsample)",
}

BENCHMARK_DECOMPOSITION_COLUMNS = [
    "baseline_pearson",
    "pl_pearson",
    "csid_pearson",
    "baseline_bias_correction",
    "pl_bias_correction",
    "csid_bias_correction",
]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _item_universe_table(universe: Path) -> pd.DataFrame:
    stability = _read(universe / "item_universe_coefficient_stability.csv")
    deprojection = _read(universe / "item_universe_deprojection_stability.csv")
    transfer = _read(universe / "item_universe_transfer_primary.csv")
    if stability.empty or deprojection.empty or transfer.empty:
        return pd.DataFrame()

    pair = stability[(stability["order"] == 2) & (stability["filter"] == "all")][
        ["context_categories", "pearson"]
    ].rename(columns={"pearson": "pair_pearson"})
    triple = stability[(stability["order"] == 3) & (stability["filter"] == "top25%")][
        ["context_categories", "pearson"]
    ].rename(columns={"pearson": "triple_pearson_top25"})
    deprojection = deprojection[["context_categories", "pearson"]].rename(
        columns={"pearson": "deprojection_pearson"}
    )
    transfer = transfer[["context_categories", "delta_ccc"]].rename(
        columns={"delta_ccc": "csid_minus_baseline_ccc"}
    )
    return (
        pair.merge(triple, on="context_categories", validate="one_to_one")
        .merge(deprojection, on="context_categories", validate="one_to_one")
        .merge(transfer, on="context_categories", validate="one_to_one")
        .sort_values("context_categories")
        .reset_index(drop=True)
    )


def _pair_example(frame: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    keep = (
        (frame["item_1"].astype(str).eq(a) & frame["item_2"].astype(str).eq(b))
        | (frame["item_1"].astype(str).eq(b) & frame["item_2"].astype(str).eq(a))
    )
    return frame[keep].head(1)


def build_paper_outputs(results_root: str | Path, output_dir: str | Path) -> dict[str, str]:
    root, output = Path(results_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    synthetic = root / "synthetic"
    cardinality = _read(synthetic / "cardinality_confounding_summary.csv")
    power = _read(synthetic / "detection_summary.csv")
    saturation = _read(synthetic / "saturation_summary.csv")
    if not cardinality.empty:
        plot_cardinality_confounding(cardinality, output / "cardinality_confounding.png")
        cardinality.to_csv(tables / "table1_cardinality_confounding.csv", index=False)
    if not power.empty:
        plot_synthetic_detection_power(power, output / "synthetic_detection_power.png")
        power.to_csv(tables / "table2_synthetic_detection.csv", index=False)
    if not saturation.empty:
        saturation.assign(
            method=saturation["method"].replace({"CSID-3 deprojected": "Reweighted CSID-2"})
        ).to_csv(tables / "table3_saturation_recovery.csv", index=False)

    sweeps: dict[str, pd.DataFrame] = {}
    examples: dict[str, pd.DataFrame] = {}
    metadata_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    concordance_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    full_transfer_rows: list[dict[str, object]] = []
    correction_rows: list[pd.DataFrame] = []
    transfer_rows: list[pd.DataFrame] = []

    for label, directory in DATASET_DIRS.items():
        ds = root / directory
        benchmark_sweep = _read(ds / "external_benchmark_sweep.csv")
        sweep = (
            benchmark_sweep
            if not benchmark_sweep.empty
            else _read(ds / "concordance_threshold_sweep.csv")
        )
        example = _read(ds / "conditional_odds_examples.csv")
        if not sweep.empty:
            sweeps[label] = sweep
        if not example.empty:
            examples[label] = example

        metadata_path = ds / "metadata.json"
        if metadata_path.exists():
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_rows.append(
                {
                    "dataset": label,
                    "baskets": meta["baskets"],
                    "H1": meta["h1_baskets"],
                    "H2": meta["h2_baskets"],
                    "categories": meta["categories"],
                    "selected_categories": CATEGORY_DESCRIPTIONS[label],
                    "triple_min_count": meta["analysis_config"]["triple_min_count"],
                    "mean_basket_size": meta["mean_basket_size"],
                }
            )

        stability = _read(ds / "temporal_stability.csv")
        loco = _read(ds / "temporal_stability_loco.csv")
        if not stability.empty:
            row = stability[stability["filter"].eq("top25%")].iloc[0].to_dict()
            row["dataset"] = label
            if not loco.empty:
                row["pearson_loco_min"] = float(loco["pearson"].min())
                row["pearson_loco_max"] = float(loco["pearson"].max())
            stability_rows.append(row)

        benchmark_primary = _read(ds / "external_benchmark_primary.csv")
        benchmark_loco = _read(ds / "external_benchmark_loco.csv")
        if not benchmark_primary.empty:
            row = benchmark_primary.iloc[0].to_dict()
            loco_min = (
                float(benchmark_loco["csid_minus_pl_ccc"].min())
                if not benchmark_loco.empty
                else row.get("csid_minus_pl_ccc_loco_min", float("nan"))
            )
            loco_max = (
                float(benchmark_loco["csid_minus_pl_ccc"].max())
                if not benchmark_loco.empty
                else row.get("csid_minus_pl_ccc_loco_max", float("nan"))
            )
            concordance_rows.append(
                {
                    "dataset": label,
                    "selected_triples": int(row["selected_triples"]),
                    "n_layers": int(row["n_layers"]),
                    "baseline_ccc": row["baseline_ccc"],
                    "pl_ccc": row["pl_ccc"],
                    "csid_ccc": row["csid_ccc"],
                    "csid_minus_pl_ccc": row["csid_minus_pl_ccc"],
                    "loco_min": loco_min,
                    "loco_max": loco_max,
                }
            )
            decomposition_rows.append(
                {
                    "dataset": label,
                    **{column: row[column] for column in BENCHMARK_DECOMPOSITION_COLUMNS},
                }
            )

        full_transfer = _read(ds / "full_transfer_summary.csv")
        if not full_transfer.empty:
            row = full_transfer.iloc[0].to_dict()
            row["dataset"] = label
            full_transfer_rows.append(row)

        corrections = _read(ds / "pair_deprojection.csv")
        if not corrections.empty:
            if label == "Complete Journey":
                chosen = pd.concat(
                    [
                        _pair_example(corrections, "FLUID MILK PRODUCTS", "COLD CEREAL"),
                        _pair_example(corrections, "SOUP", "CRACKERS/MISC BKD FD"),
                    ]
                )
            elif label == "Ta-Feng":
                chosen = pd.concat(
                    [_pair_example(corrections, "51", "54"), _pair_example(corrections, "73", "47")]
                )
            else:
                chosen = pd.concat(
                    [
                        _pair_example(corrections, "fresh herbs", "cereal"),
                        _pair_example(corrections, "packaged cheese", "lunch meat"),
                    ]
                )
            if chosen.empty:
                chosen = corrections.head(2)
            chosen = chosen.copy()
            chosen.insert(0, "dataset", label)
            correction_rows.append(chosen)

        transfer = _read(ds / "pair_deprojection_transfer.csv")
        if not transfer.empty:
            transfer = transfer.copy()
            transfer.insert(0, "dataset", label)
            transfer_rows.append(transfer)

    if sweeps:
        plot_concordance_thresholds(sweeps, output / "concordance_threshold_validation.png")
    if examples:
        plot_conditional_odds(examples, output / "conditional_odds_lambda_validation.png")

    pd.DataFrame(metadata_rows).to_csv(tables / "table4_datasets.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(tables / "table5_stability.csv", index=False)
    if concordance_rows:
        pd.DataFrame(concordance_rows).to_csv(tables / "table6_concordance.csv", index=False)
    if decomposition_rows:
        pd.DataFrame(decomposition_rows).to_csv(
            tables / "table7_concordance_decomposition.csv",
            index=False,
        )
    if correction_rows:
        pd.concat(correction_rows, ignore_index=True).to_csv(tables / "table8_pair_deprojection.csv", index=False)
    if transfer_rows:
        pd.concat(transfer_rows, ignore_index=True).to_csv(
            tables / "table9_pair_deprojection_transfer.csv",
            index=False,
        )
    if full_transfer_rows:
        pd.DataFrame(full_transfer_rows).to_csv(
            tables / "table11_full_transfer.csv",
            index=False,
        )

    universe = root / "item-universe-expansion"
    audit_png = universe / "item_universe_expansion.png"
    if audit_png.exists():
        shutil.copy2(audit_png, output / "item_universe_expansion.png")
    else:
        coefficients = _read(universe / "item_universe_coefficients.csv")
        if not coefficients.empty:
            pair_comparison, triple_comparison = build_item_universe_coefficient_comparisons(coefficients)
            if not pair_comparison.empty and not triple_comparison.empty:
                plot_item_universe_coefficient_scatter(
                    pair_comparison,
                    triple_comparison,
                    output / "item_universe_expansion.png",
                )
    universe_table = _item_universe_table(universe)
    if not universe_table.empty:
        universe_table.to_csv(tables / "table10_item_universe_stability.csv", index=False)
    sensitivity = _read(root / "robustness" / "sensitivity_summary.csv")
    if not sensitivity.empty:
        sensitivity.to_csv(tables / "table12_sensitivity.csv", index=False)
    scaling = _read(universe / "candidate-scaling" / "candidate_scaling.csv")
    if not scaling.empty:
        scaling.to_csv(tables / "table13_candidate_scaling.csv", index=False)

    manifest = {
        "figure1": str(output / "cardinality_confounding.png"),
        "figure2": str(output / "synthetic_detection_power.png"),
        "figure3": str(output / "concordance_threshold_validation.png"),
        "figure4": str(output / "conditional_odds_lambda_validation.png"),
        "figure5": str(output / "item_universe_expansion.png"),
        "tables": str(tables),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
