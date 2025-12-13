"""
Preprocessing module for credit scoring model development.

This module provides tools for feature transformation including:
- WOE (Weight of Evidence) encoding
- Binning strategies
- Missing value handling
- Feature engineering (time series, static, missing patterns)
"""

from .woe_transformer import (
    # Enums
    BinningMethod,
    MissingStrategy,
    # Constants
    IV_THRESHOLDS,
    SMOOTHING_CONSTANT,
    # Dataclasses
    WOEBinStats,
    FeatureWOEResult,
    # Classes
    WOEBinner,
    WOETransformer,
)

from .feature_engineer import (
    # Constants
    DEFAULT_WINDOWS,
    AGE_BINS,
    AGE_LABELS,
    INCOME_THRESHOLDS,
    CREDIT_HISTORY_THRESHOLDS,
    TREND_THRESHOLDS,
    # Enums
    TrendDirection,
    IncomeLevel,
    CreditHistoryDepth,
    # Classes
    TimeSeriesFeatureEngineer,
    StaticFeatureEngineer,
    MissingFeatureEngineer,
    CreditFeatureEngineer,
)

__all__ = [
    # WOE Transformer Enums
    "BinningMethod",
    "MissingStrategy",
    # WOE Constants
    "IV_THRESHOLDS",
    "SMOOTHING_CONSTANT",
    # WOE Dataclasses
    "WOEBinStats",
    "FeatureWOEResult",
    # WOE Classes
    "WOEBinner",
    "WOETransformer",
    # Feature Engineer Constants
    "DEFAULT_WINDOWS",
    "AGE_BINS",
    "AGE_LABELS",
    "INCOME_THRESHOLDS",
    "CREDIT_HISTORY_THRESHOLDS",
    "TREND_THRESHOLDS",
    # Feature Engineer Enums
    "TrendDirection",
    "IncomeLevel",
    "CreditHistoryDepth",
    # Feature Engineer Classes
    "TimeSeriesFeatureEngineer",
    "StaticFeatureEngineer",
    "MissingFeatureEngineer",
    "CreditFeatureEngineer",
]
