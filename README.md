# csid

Reproduction code for **Cardinality-Stratified Interaction Decomposition (CSID)** on transactional basket data.

## Modules

| Module | Role |
| --- | --- |
| `data.py` | Dataset loaders and `SparseBasketDataset` |
| `candidates.py` | Pair/triple candidate construction |
| `counts.py` | Size-stratified subset counts |
| `contrasts.py` | Rest-size contrasts and information weights |
| `estimator.py` | Gauge-constrained weighted ridge fit (CSID-2/3) |
| `pseudolikelihood.py` | Cardinality-aware higher-order pseudolikelihood benchmark |
| `analysis.py` | Core fit pipeline (full sample and H1/H2 split) |
| `validation.py` | H1/H2 temporal stability |
| `concordance.py` | Cross-period CCC validation and plot helpers |
| `correction.py` | Third-order deprojection of pair effects |
| `synthetic.py` | Cardinality confounding, detection, and saturation experiments |
| `ising.py` | Small exact pairwise-Ising benchmark for synthetic audits |
| `paper.py` | One-dataset paper analysis and CSV exports |
| `universe.py` | Instacart item-universe expansion robustness audit |
| `robustness.py` | Focused parameter sensitivity and candidate-scaling audits |
| `figures.py` | Consolidated paper figures and tables |
| `cli.py` | `csid` CLI entry point |
| `keys.py` | Typed `(i, j)` / `(i, j, k)` key helpers |

## Setup

```bash
uv sync --group dev
```

Python 3.10 or newer and [`uv`](https://docs.astral.sh/uv/) are required.

## Data

Raw datasets are not included in this repository. Download them separately and place the prepared files under `data/` (not tracked in git):

```text
data/
├── complete-journey/transactions.parquet
├── complete-journey/products.csv
├── ta-feng/ta_feng_all_months_merged.csv
└── instacart/{orders,products,aisles,order_products__prior}.csv
```

- **Complete Journey:** obtain via the [`completejourney`](https://bradleyboehmke.github.io/completejourney/) R package. Export `get_transactions()` as `transactions.parquet` with columns `basket_id`, `product_id`, and `week`, and export the products table as `products.csv` with columns `product_id` and `product_category`. The bundled category list fixes the 20 categories used in the paper.
- **Ta-Feng:** [Kaggle](https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset)
- **Instacart:** [Kaggle](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)

Complete Journey category list: `configs/complete_journey_categories.txt`

## Full paper run

```bash
uv run csid paper --data-root data --output outputs/paper-run
```

Expected outputs include:

```text
outputs/paper-run/
├── synthetic/
├── complete-journey/
├── ta-feng/
├── instacart/
├── robustness/
├── item-universe-expansion/
└── paper/
```

The full command includes the one-factor sensitivity suite, singleton-basket
analysis, and Instacart candidate-scaling benchmark and can require several
hours on a laptop. Each dataset directory also contains
`full_transfer_summary.csv`; consolidated sensitivity and scaling results are
written under `robustness/` and `item-universe-expansion/candidate-scaling/`.

To refresh the tracked manuscript figures and table CSVs directly from this run:

```bash
uv run csid figures --results-root outputs/paper-run --output figures
```

## Quality checks

```bash
uv run pytest
uv run ruff check src tests
uv run mypy
```
