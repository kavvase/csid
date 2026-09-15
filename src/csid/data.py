from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_cj_category_file() -> Path:
    bundled = repo_root() / "configs" / "complete_journey_categories.txt"
    if bundled.exists():
        return bundled
    packaged = Path(__file__).with_name("complete_journey_categories.txt")
    if packaged.exists():
        return packaged
    return repo_root() / "configs" / "complete_journey_categories.example.txt"


def resolve_dataset_root(data_root: str | Path, dataset: str) -> Path:
    root = Path(data_root)
    subdirs = {
        "complete-journey": "complete-journey",
        "ta-feng": "ta-feng",
        "instacart": "instacart",
    }
    nested = root / subdirs[dataset]
    if nested.exists():
        return nested
    markers = {
        "complete-journey": ("products.csv",),
        "ta-feng": ("ta_feng_all_months_merged.csv",),
        "instacart": ("orders.csv",),
    }
    if any((root / name).exists() for name in markers[dataset]):
        return root
    return nested


@dataclass(frozen=True)
class SparseBasketDataset:
    """CSR-backed transactional basket data.

    Rows are baskets and columns are selected product groups.  ``X`` is binary,
    sorted by column within each row, and stores no explicit zeros.
    """

    name: str
    X: sparse.csr_matrix
    category_names: tuple[str, ...]
    basket_ids: np.ndarray
    periods: np.ndarray
    cluster_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        X = self.X.tocsr(copy=False)
        X.sum_duplicates()
        X.sort_indices()
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if X.shape[1] != len(self.category_names):
            raise ValueError("category_names length does not match X columns")
        n = X.shape[0]
        if len(self.basket_ids) != n or len(self.periods) != n:
            raise ValueError("basket_ids/periods length does not match X rows")
        if self.cluster_ids is not None and len(self.cluster_ids) != n:
            raise ValueError("cluster_ids length does not match X rows")
        if X.nnz and not np.all(X.data == 1):
            raise ValueError("X must be binary")
        object.__setattr__(self, "X", X)

    @property
    def n_baskets(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_categories(self) -> int:
        return int(self.X.shape[1])

    @property
    def basket_sizes(self) -> np.ndarray:
        return cast(np.ndarray, np.diff(self.X.indptr).astype(np.int32, copy=False))

    def row_items(self, row: int) -> np.ndarray:
        start, end = int(self.X.indptr[row]), int(self.X.indptr[row + 1])
        return cast(np.ndarray, self.X.indices[start:end])

    def iter_baskets(self) -> Iterable[np.ndarray]:
        for row in range(self.n_baskets):
            yield self.row_items(row)

    def subset_rows(self, keep: np.ndarray, *, name: str | None = None) -> SparseBasketDataset:
        mask = np.asarray(keep, dtype=bool)
        if mask.shape != (self.n_baskets,):
            raise ValueError("row mask must have one value per basket")
        if not np.any(mask):
            raise ValueError("row subset is empty")
        return SparseBasketDataset(
            name=self.name if name is None else str(name),
            X=self.X[mask].tocsr(),
            category_names=self.category_names,
            basket_ids=np.asarray(self.basket_ids)[mask],
            periods=np.asarray(self.periods)[mask],
            cluster_ids=(None if self.cluster_ids is None else np.asarray(self.cluster_ids)[mask]),
        )

    def subset_period(self, period: str) -> SparseBasketDataset:
        keep = np.asarray(self.periods).astype(str) == str(period)
        return self.subset_rows(keep, name=f"{self.name}-{period}")

    def subset_columns(
        self, columns: Sequence[int], *, name: str | None = None
    ) -> SparseBasketDataset:
        indexes = np.asarray(columns, dtype=np.int32)
        if indexes.ndim != 1 or len(indexes) == 0:
            raise ValueError("columns must be a non-empty one-dimensional sequence")
        if len(np.unique(indexes)) != len(indexes):
            raise ValueError("columns must not contain duplicates")
        if np.any(indexes < 0) or np.any(indexes >= self.n_categories):
            raise ValueError("column index out of range")
        return SparseBasketDataset(
            name=self.name if name is None else str(name),
            X=self.X[:, indexes].tocsr(),
            category_names=tuple(self.category_names[int(i)] for i in indexes),
            basket_ids=np.asarray(self.basket_ids, dtype=object).copy(),
            periods=np.asarray(self.periods, dtype=object).copy(),
            cluster_ids=(
                None
                if self.cluster_ids is None
                else np.asarray(self.cluster_ids, dtype=object).copy()
            ),
        )

    @classmethod
    def from_dense(
        cls,
        X: np.ndarray,
        *,
        name: str = "dense",
        category_names: Sequence[str] | None = None,
        periods: Sequence[str] | None = None,
        basket_ids: Sequence[object] | None = None,
        cluster_ids: Sequence[object] | None = None,
    ) -> SparseBasketDataset:
        arr = np.asarray(X, dtype=np.uint8)
        if arr.ndim != 2 or not np.all((arr == 0) | (arr == 1)):
            raise ValueError("X must be a binary two-dimensional array")
        n, c = arr.shape
        names = tuple(str(i) for i in range(c)) if category_names is None else tuple(map(str, category_names))
        per = np.asarray(["ALL"] * n if periods is None else periods, dtype=object)
        ids = np.asarray(np.arange(n) if basket_ids is None else basket_ids, dtype=object)
        clusters = None if cluster_ids is None else np.asarray(cluster_ids, dtype=object)
        return cls(
            name=name,
            X=sparse.csr_matrix(arr),
            category_names=names,
            basket_ids=ids,
            periods=per,
            cluster_ids=clusters,
        )


@dataclass(frozen=True)
class InstacartUniverseCollection:
    """Nested Instacart aisle universes sharing baskets and focal columns."""

    datasets: tuple[tuple[str, SparseBasketDataset], ...]
    category_table: pd.DataFrame
    focus_n: int

    def as_dict(self) -> dict[str, SparseBasketDataset]:
        return dict(self.datasets)


def dataset_fingerprint(dataset: SparseBasketDataset) -> str:
    digest = hashlib.sha256()
    digest.update(dataset.name.encode("utf-8"))
    digest.update(np.ascontiguousarray(dataset.X.indptr, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(dataset.X.indices, dtype=np.int32).tobytes())
    for values in (
        dataset.category_names,
        np.asarray(dataset.basket_ids, dtype=object),
        np.asarray(dataset.periods, dtype=object),
        () if dataset.cluster_ids is None else np.asarray(dataset.cluster_ids, dtype=object),
    ):
        digest.update(b"\xff")
        for value in values:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def _normalize_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def _read_category_file(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    p = Path(path)
    values = [
        line.strip()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values:
        raise ValueError(f"Category file is empty: {p}")
    if len(values) != len(set(values)):
        raise ValueError(f"Category file contains duplicates: {p}")
    return values


def _top_categories(
    presence: pd.DataFrame,
    *,
    basket_col: str,
    category_col: str,
    top_n: int | None,
    min_support: int = 1,
) -> list[object]:
    support = (
        presence[[basket_col, category_col]]
        .drop_duplicates()
        .groupby(category_col, sort=False)[basket_col]
        .nunique()
        .sort_values(ascending=False)
    )
    support = support[support >= int(min_support)]
    if top_n is not None and top_n > 0:
        support = support.head(int(top_n))
    return cast(list[object], support.index.tolist())


def _presence_to_sparse_dataset(
    *,
    name: str,
    presence: pd.DataFrame,
    basket_col: str,
    category_col: str,
    categories: Sequence[object],
    metadata: pd.DataFrame,
    period_col: str,
    cluster_col: str | None = None,
    min_basket_size: int = 2,
    display_names: Mapping[Any, str] | None = None,
) -> SparseBasketDataset:
    if not categories:
        raise ValueError(f"No categories selected for {name}")
    category_list = list(categories)
    category_index = {value: idx for idx, value in enumerate(category_list)}

    meta_cols = [basket_col, period_col] + ([] if cluster_col is None else [cluster_col])
    meta = metadata[meta_cols].drop_duplicates(subset=[basket_col]).copy()
    meta = meta[meta[period_col].notna()].reset_index(drop=True)
    basket_index = {value: idx for idx, value in enumerate(meta[basket_col].tolist())}

    dedup = presence[[basket_col, category_col]].dropna().drop_duplicates()
    dedup = dedup[dedup[basket_col].isin(basket_index) & dedup[category_col].isin(category_index)]
    rows = dedup[basket_col].map(basket_index).to_numpy(np.int64)
    cols = dedup[category_col].map(category_index).to_numpy(np.int32)
    values = np.ones(len(dedup), dtype=np.uint8)
    X = sparse.coo_matrix((values, (rows, cols)), shape=(len(meta), len(category_list)), dtype=np.uint8).tocsr()
    X.sum_duplicates()
    X.data[:] = 1
    X.sort_indices()

    keep = np.diff(X.indptr) >= int(min_basket_size)
    if not np.any(keep):
        raise ValueError(f"No baskets of size >= {min_basket_size} remain for {name}")
    X = X[keep].tocsr()
    meta = meta.loc[keep].reset_index(drop=True)

    names = tuple(str(value if display_names is None else display_names.get(value, value)) for value in category_list)
    return SparseBasketDataset(
        name=name,
        X=X,
        category_names=names,
        basket_ids=meta[basket_col].astype(str).to_numpy(),
        periods=meta[period_col].astype(str).to_numpy(),
        cluster_ids=(None if cluster_col is None else meta[cluster_col].astype(str).to_numpy()),
    )


def _load_complete_journey_transactions(root: Path) -> pd.DataFrame:
    products_path = root / "products.csv"
    if not products_path.exists():
        raise FileNotFoundError(f"Expected {products_path}")

    parquet_path = root / "transactions.parquet"
    sample_path = root / "transactions_sample.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path, columns=["basket_id", "product_id", "week"])
        except ImportError as exc:
            raise ImportError(
                "Reading transactions.parquet requires pyarrow. Install with: pip install -e '.[parquet]'"
            ) from exc
    if sample_path.exists():
        warnings.warn(
            f"Using {sample_path.name} because transactions.parquet is absent. "
            "Results are a sample-subset smoke validation, not full Complete Journey.",
            stacklevel=2,
        )
        return pd.read_csv(sample_path, usecols=["basket_id", "product_id", "week"])
    raise FileNotFoundError(f"Expected {parquet_path} or {sample_path} alongside {products_path}")


def load_complete_journey(
    data_root: str | Path = "data/complete-journey",
    *,
    category_file: str | Path | None = None,
    top_n: int | None = 20,
    min_basket_size: int = 2,
) -> SparseBasketDataset:
    root = Path(data_root)
    tx = _load_complete_journey_transactions(root)
    products = pd.read_csv(root / "products.csv", usecols=["product_id", "product_category"])
    products["product_id"] = _normalize_id(products["product_id"])
    tx["product_id"] = _normalize_id(tx["product_id"])
    merged = tx.merge(products, on="product_id", how="inner").dropna(subset=["basket_id", "product_category", "week"])
    if category_file is None:
        bundled = default_cj_category_file()
        if bundled.exists():
            category_file = bundled
    categories = _read_category_file(category_file)
    if categories is None:
        categories = [
            str(value)
            for value in _top_categories(
                merged,
                basket_col="basket_id",
                category_col="product_category",
                top_n=top_n,
            )
        ]
        if top_n == 20:
            warnings.warn(
                "CAT_MAP was not supplied; selected categories by basket support. ",
                stacklevel=2,
            )
    meta = merged.groupby("basket_id", as_index=False)["week"].first()
    week = meta["week"].to_numpy()
    period = np.empty(len(meta), dtype=object)
    period[week <= 26] = "H1"
    period[week >= 27] = "H2"
    meta["period"] = period
    return _presence_to_sparse_dataset(
        name="complete-journey",
        presence=merged,
        basket_col="basket_id",
        category_col="product_category",
        categories=categories,
        metadata=meta[["basket_id", "period"]],
        period_col="period",
        min_basket_size=min_basket_size,
    )


def load_ta_feng(
    data_root: str | Path = "data/ta-feng",
    *,
    top_n: int | None = 20,
    min_category_support: int = 1000,
    min_basket_size: int = 2,
) -> SparseBasketDataset:
    path = Path(data_root) / "ta_feng_all_months_merged.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(
        path,
        usecols=["TRANSACTION_DT", "CUSTOMER_ID", "PRODUCT_SUBCLASS"],
        dtype={"CUSTOMER_ID": "string", "PRODUCT_SUBCLASS": "string"},
        low_memory=False,
    )
    df["DATE"] = pd.to_datetime(df["TRANSACTION_DT"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["DATE", "CUSTOMER_ID", "PRODUCT_SUBCLASS"])
    df["basket_id"] = df["CUSTOMER_ID"].astype(str) + "_" + df["DATE"].dt.strftime("%Y%m%d")
    df["CAT"] = df["PRODUCT_SUBCLASS"].astype(str).str[:2]
    categories = _top_categories(
        df,
        basket_col="basket_id",
        category_col="CAT",
        top_n=top_n,
        min_support=min_category_support,
    )
    meta = df.groupby("basket_id", as_index=False).agg(DATE=("DATE", "first"), CUSTOMER_ID=("CUSTOMER_ID", "first"))
    year, month = meta["DATE"].dt.year, meta["DATE"].dt.month
    year_values = year.to_numpy()
    month_values = month.to_numpy()
    period = np.empty(len(meta), dtype=object)
    period[(year_values == 2000) & np.isin(month_values, [11, 12])] = "H1"
    period[(year_values == 2001) & np.isin(month_values, [1, 2])] = "H2"
    meta["period"] = period
    return _presence_to_sparse_dataset(
        name="ta-feng",
        presence=df,
        basket_col="basket_id",
        category_col="CAT",
        categories=categories,
        metadata=meta[["basket_id", "period", "CUSTOMER_ID"]],
        period_col="period",
        cluster_col="CUSTOMER_ID",
        min_basket_size=min_basket_size,
    )


def _read_instacart_presence(
    data_root: str | Path,
    *,
    n_users: int,
    seed: int,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    root = Path(data_root)
    paths = {
        "orders": root / "orders.csv",
        "products": root / "products.csv",
        "aisles": root / "aisles.csv",
        "prior": root / "order_products__prior.csv",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    orders = pd.read_csv(
        paths["orders"],
        usecols=["order_id", "user_id", "eval_set", "order_number"],
    )
    orders = orders.loc[
        orders["eval_set"].eq("prior"), ["order_id", "user_id", "order_number"]
    ]
    users = np.sort(orders["user_id"].dropna().unique())
    rng = np.random.default_rng(seed)
    sampled_users = rng.choice(users, size=min(n_users, len(users)), replace=False)
    sampled_orders = orders[orders["user_id"].isin(sampled_users)].copy()
    order_set = set(sampled_orders["order_id"].astype(int).tolist())

    products = pd.read_csv(paths["products"], usecols=["product_id", "aisle_id"])
    product_to_aisle = products.set_index("product_id")["aisle_id"]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        paths["prior"], usecols=["order_id", "product_id"], chunksize=chunksize
    ):
        chunk = chunk[chunk["order_id"].isin(order_set)]
        if chunk.empty:
            continue
        chunk["aisle_id"] = chunk["product_id"].map(product_to_aisle)
        parts.append(chunk[["order_id", "aisle_id"]].dropna().drop_duplicates())
    if not parts:
        raise ValueError("No prior rows matched sampled users")
    presence = pd.concat(parts, ignore_index=True).drop_duplicates()
    presence["aisle_id"] = presence["aisle_id"].astype(int)
    display = (
        pd.read_csv(paths["aisles"], usecols=["aisle_id", "aisle"]).set_index("aisle_id")["aisle"].astype(str).to_dict()
    )
    sampled_orders["period"] = np.where(sampled_orders["order_number"] <= 10, "H1", "H2")
    return presence, sampled_orders, display


def load_instacart(
    data_root: str | Path = "data/instacart",
    *,
    n_users: int = 30_000,
    seed: int = 42,
    top_n: int | None = 20,
    chunksize: int = 2_000_000,
    min_basket_size: int = 2,
) -> SparseBasketDataset:
    presence, sampled_orders, display = _read_instacart_presence(
        data_root, n_users=n_users, seed=seed, chunksize=chunksize
    )
    categories = _top_categories(
        presence,
        basket_col="order_id",
        category_col="aisle_id",
        top_n=top_n,
    )
    return _presence_to_sparse_dataset(
        name="instacart",
        presence=presence,
        basket_col="order_id",
        category_col="aisle_id",
        categories=categories,
        metadata=sampled_orders[["order_id", "period", "user_id"]],
        period_col="period",
        cluster_col="user_id",
        min_basket_size=min_basket_size,
        display_names=display,
    )


def load_instacart_universes(
    data_root: str | Path = "data/instacart",
    *,
    n_users: int = 30_000,
    seed: int = 42,
    focus_n: int = 20,
    universe_sizes: Sequence[int | str] = (20, 40, 80, "all"),
    chunksize: int = 2_000_000,
    min_focus_basket_size: int = 2,
) -> InstacartUniverseCollection:
    """Load nested aisle universes on one fixed basket sample."""

    presence, sampled_orders, display = _read_instacart_presence(
        data_root, n_users=n_users, seed=seed, chunksize=chunksize
    )
    support = (
        presence[["order_id", "aisle_id"]]
        .drop_duplicates()
        .groupby("aisle_id", sort=False)["order_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    ordered_categories = [int(value) for value in support.index.tolist()]
    if focus_n < 3 or focus_n > len(ordered_categories):
        raise ValueError("focus_n must be between 3 and the number of observed aisles")

    focus_categories = set(ordered_categories[: int(focus_n)])
    focus_counts = (
        presence[presence["aisle_id"].isin(focus_categories)]
        .drop_duplicates(["order_id", "aisle_id"])
        .groupby("order_id")["aisle_id"]
        .nunique()
    )
    fixed_ids = set(
        focus_counts[focus_counts >= int(min_focus_basket_size)].index.tolist()
    )
    if not fixed_ids:
        raise ValueError("No baskets satisfy the focal-universe size restriction")
    fixed_orders = sampled_orders[sampled_orders["order_id"].isin(fixed_ids)].copy()
    fixed_presence = presence[presence["order_id"].isin(fixed_ids)].copy()

    normalized: list[tuple[str, int]] = []
    for raw in universe_sizes:
        text = str(raw).strip().lower()
        if text == "all" or text == "0":
            size = len(ordered_categories)
            label = "all"
        else:
            size = int(raw)
            if size < focus_n:
                raise ValueError("Every universe size must be at least focus_n")
            size = min(size, len(ordered_categories))
            label = str(size)
        if size == len(ordered_categories):
            label = "all"
        if not any(existing_size == size for _, existing_size in normalized):
            normalized.append((label, size))
    if not any(size == focus_n for _, size in normalized):
        normalized.insert(0, (str(focus_n), int(focus_n)))
    normalized.sort(key=lambda item: item[1])

    datasets: list[tuple[str, SparseBasketDataset]] = []
    reference_ids: np.ndarray | None = None
    for label, size in normalized:
        categories = ordered_categories[:size]
        dataset = _presence_to_sparse_dataset(
            name=f"instacart-universe-{label}",
            presence=fixed_presence,
            basket_col="order_id",
            category_col="aisle_id",
            categories=categories,
            metadata=fixed_orders[["order_id", "period", "user_id"]],
            period_col="period",
            cluster_col="user_id",
            min_basket_size=min_focus_basket_size,
            display_names=display,
        )
        if reference_ids is None:
            reference_ids = np.asarray(dataset.basket_ids, dtype=object)
        elif not np.array_equal(reference_ids, np.asarray(dataset.basket_ids, dtype=object)):
            raise RuntimeError("Nested universes did not preserve a common basket order")
        datasets.append((label, dataset))

    category_table = pd.DataFrame(
        {
            "rank": np.arange(1, len(ordered_categories) + 1),
            "aisle_id": ordered_categories,
            "aisle": [display.get(value, str(value)) for value in ordered_categories],
            "basket_support": [int(support.loc[value]) for value in ordered_categories],
            "is_focal": [rank <= focus_n for rank in range(1, len(ordered_categories) + 1)],
        }
    )
    return InstacartUniverseCollection(
        datasets=tuple(datasets),
        category_table=category_table,
        focus_n=int(focus_n),
    )


def load_generic(
    path: str | Path,
    *,
    basket_col: str,
    item_col: str,
    period_col: str,
    cluster_col: str | None = None,
    top_n: int | None = None,
    min_category_support: int = 1,
    min_basket_size: int = 2,
) -> SparseBasketDataset:
    p = Path(path)
    columns = [basket_col, item_col, period_col] + ([] if cluster_col is None else [cluster_col])
    if p.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(p, columns=columns)
    else:
        frame = pd.read_csv(p, usecols=columns)
    frame = frame.dropna(subset=[basket_col, item_col, period_col])
    categories = _top_categories(
        frame,
        basket_col=basket_col,
        category_col=item_col,
        top_n=top_n,
        min_support=min_category_support,
    )
    metadata = frame[[basket_col, period_col] + ([] if cluster_col is None else [cluster_col])]
    return _presence_to_sparse_dataset(
        name=p.stem,
        presence=frame,
        basket_col=basket_col,
        category_col=item_col,
        categories=categories,
        metadata=metadata,
        period_col=period_col,
        cluster_col=cluster_col,
        min_basket_size=min_basket_size,
    )
