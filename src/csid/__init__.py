"""Cardinality-Stratified Interaction Decomposition."""

__version__ = "1.0.0"

from .analysis import AnalysisConfig, DatasetAnalysis, FitBundle, fit_periods
from .data import (
    InstacartUniverseCollection,
    SparseBasketDataset,
    load_instacart_universes,
)
from .estimator import CSIDFit, fit_csid
from .universe import ItemUniverseResult, run_item_universe_expansion

__all__ = [
    "__version__",
    "AnalysisConfig",
    "DatasetAnalysis",
    "FitBundle",
    "SparseBasketDataset",
    "InstacartUniverseCollection",
    "load_instacart_universes",
    "ItemUniverseResult",
    "run_item_universe_expansion",
    "CSIDFit",
    "fit_csid",
    "fit_periods",
]
