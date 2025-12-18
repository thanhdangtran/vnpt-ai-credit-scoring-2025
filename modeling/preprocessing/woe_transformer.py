from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# sklearn compatibility
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_is_fitted
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Create dummy classes if sklearn not available
    class BaseEstimator:
        pass
    class TransformerMixin:
        pass
    def check_is_fitted(estimator, attributes):
        for attr in attributes:
            if not hasattr(estimator, attr):
                raise ValueError(f"{estimator} is not fitted")


# ENUMS AND CONSTANTS

class BinningMethod(Enum):
    EQUAL_FREQUENCY = "equal_frequency"   # Equal number of observations
    EQUAL_WIDTH = "equal_width"           # Equal range width
    OPTIMAL = "optimal"                   # Optimal binning (maximize IV)
    MONOTONIC = "monotonic"               # Monotonic WOE constraint
    CUSTOM = "custom"                     # User-defined bins
    QUANTILE = "quantile"                 # Quantile-based


class MissingStrategy(Enum):
    SEPARATE_BIN = "separate_bin"     # Create separate bin for missing
    WOE_ZERO = "woe_zero"             # Assign WOE = 0 (neutral)
    WORST_WOE = "worst_woe"           # Assign worst (lowest) WOE
    BEST_WOE = "best_woe"             # Assign best (highest) WOE
    MODE_BIN = "mode_bin"             # Assign to most frequent bin


# IV interpretation thresholds
IV_THRESHOLDS = {
    "not_useful": 0.02,
    "weak": 0.10,
    "medium": 0.30,
    "strong": 0.50,
}

# Smoothing constant for zero counts
SMOOTHING_CONSTANT = 0.5


# WOE BIN STATISTICS

@dataclass
class WOEBinStats:
    bin_id: int
    bin_label: str
    bin_min: Optional[float]
    bin_max: Optional[float]
    count: int
    count_pct: float
    good_count: int
    bad_count: int
    good_pct: float
    bad_pct: float
    bad_rate: float
    distr_good: float
    distr_bad: float
    woe: float
    iv_contribution: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_id": self.bin_id,
            "bin_label": self.bin_label,
            "bin_min": self.bin_min,
            "bin_max": self.bin_max,
            "count": self.count,
            "count_pct": self.count_pct,
            "good_count": self.good_count,
            "bad_count": self.bad_count,
            "good_pct": self.good_pct,
            "bad_pct": self.bad_pct,
            "bad_rate": self.bad_rate,
            "distr_good": self.distr_good,
            "distr_bad": self.distr_bad,
            "woe": self.woe,
            "iv_contribution": self.iv_contribution,
        }


@dataclass
class FeatureWOEResult:
    feature_name: str
    feature_type: str  # 'continuous' or 'categorical'
    n_bins: int
    iv: float
    iv_interpretation: str
    bins: List[WOEBinStats]
    bin_edges: Optional[List[float]] = None
    is_monotonic: bool = True
    missing_woe: Optional[float] = None

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([b.to_dict() for b in self.bins])


# WOE BINNER

class WOEBinner:
    def __init__(
        self,
        method: BinningMethod = BinningMethod.MONOTONIC,
        n_bins: int = 10,
        min_bin_pct: float = 0.05,
        monotonic: bool = True,
        handle_missing: MissingStrategy = MissingStrategy.SEPARATE_BIN,
        initial_bins: int = 50,
    ):
        self.method = method
        self.n_bins = n_bins
        self.min_bin_pct = min_bin_pct
        self.monotonic = monotonic
        self.handle_missing = handle_missing
        self.initial_bins = initial_bins

    def equal_frequency_binning(
        self,
        x: np.ndarray,
        n_bins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(x, percentiles)
        # Remove duplicate edges
        bin_edges = np.unique(bin_edges)
        bin_assignments = np.digitize(x, bin_edges[1:-1])
        return bin_edges, bin_assignments

    def equal_width_binning(
        self,
        x: np.ndarray,
        n_bins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_min, x_max = x.min(), x.max()
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        bin_assignments = np.digitize(x, bin_edges[1:-1])
        return bin_edges, bin_assignments

    def quantile_binning(
        self,
        x: np.ndarray,
        n_bins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            bin_assignments, bin_edges = pd.qcut(
                x, q=n_bins, labels=False, retbins=True, duplicates='drop'
            )
            return bin_edges, bin_assignments
        except ValueError:
            # Fall back to equal frequency if qcut fails
            return self.equal_frequency_binning(x, n_bins)

    def optimal_binning(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_bins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Step 1: Create initial fine bins
        initial_bins = min(self.initial_bins, len(np.unique(x)))
        bin_edges, _ = self.equal_frequency_binning(x, initial_bins)

        # Step 2: Calculate initial statistics
        bin_assignments = np.digitize(x, bin_edges[1:-1])

        # Step 3: Iteratively merge bins
        current_edges = bin_edges.copy()
        min_samples = int(len(x) * self.min_bin_pct)

        while True:
            # Assign to current bins
            assignments = np.digitize(x, current_edges[1:-1])
            n_current_bins = len(current_edges) - 1

            if n_current_bins <= max_bins:
                break

            # Calculate WOE for each bin
            bin_stats = self._calculate_bin_stats(x, y, assignments, current_edges)

            # Find bins to merge
            merge_idx = self._find_merge_candidate(
                bin_stats, min_samples, self.monotonic
            )

            if merge_idx is None:
                break

            # Merge bins
            current_edges = np.delete(current_edges, merge_idx + 1)

        # Final assignment
        final_assignments = np.digitize(x, current_edges[1:-1])
        return current_edges, final_assignments

    def _calculate_bin_stats(
        self,
        x: np.ndarray,
        y: np.ndarray,
        assignments: np.ndarray,
        edges: np.ndarray
    ) -> List[Dict]:
        stats_list = []
        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()

        for bin_id in range(len(edges) - 1):
            mask = assignments == bin_id
            count = mask.sum()
            good = ((y == 0) & mask).sum()
            bad = ((y == 1) & mask).sum()

            # Calculate WOE with smoothing
            distr_good = (good + SMOOTHING_CONSTANT) / (total_good + 1)
            distr_bad = (bad + SMOOTHING_CONSTANT) / (total_bad + 1)
            woe = np.log(distr_good / distr_bad)

            stats_list.append({
                'bin_id': bin_id,
                'count': count,
                'good': good,
                'bad': bad,
                'woe': woe,
                'edge_low': edges[bin_id],
                'edge_high': edges[bin_id + 1],
            })

        return stats_list

    def _find_merge_candidate(
        self,
        bin_stats: List[Dict],
        min_samples: int,
        enforce_monotonic: bool
    ) -> Optional[int]:
        n_bins = len(bin_stats)
        if n_bins <= 2:
            return None

        # Check for bins below minimum size
        for i in range(n_bins):
            if bin_stats[i]['count'] < min_samples:
                if i == 0:
                    return 0
                elif i == n_bins - 1:
                    return i - 1
                else:
                    # Merge with neighbor that has fewer samples
                    if bin_stats[i-1]['count'] <= bin_stats[i+1]['count']:
                        return i - 1
                    else:
                        return i

        # Check for monotonicity violations
        if enforce_monotonic:
            woes = [s['woe'] for s in bin_stats]
            # Check if increasing or decreasing
            diffs = np.diff(woes)
            if np.all(diffs >= 0) or np.all(diffs <= 0):
                pass  # Already monotonic
            else:
                # Find violation points
                for i in range(len(diffs) - 1):
                    if (diffs[i] > 0 and diffs[i + 1] < 0) or \
                       (diffs[i] < 0 and diffs[i + 1] > 0):
                        return i + 1

        # Merge bins with smallest WOE difference
        woe_diffs = [abs(bin_stats[i+1]['woe'] - bin_stats[i]['woe'])
                     for i in range(n_bins - 1)]
        return int(np.argmin(woe_diffs))

    def custom_binning(
        self,
        x: np.ndarray,
        cuts: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_min, x_max = x.min(), x.max()
        bin_edges = np.array([x_min] + sorted(cuts) + [x_max])
        bin_assignments = np.digitize(x, bin_edges[1:-1])
        return bin_edges, bin_assignments

    def fit_bins(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: Optional[BinningMethod] = None,
        custom_cuts: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        method = method or self.method

        # Handle missing values first
        mask_valid = ~pd.isna(x)
        x_valid = x[mask_valid]
        y_valid = y[mask_valid]

        if len(x_valid) == 0:
            raise ValueError("No valid (non-null) values to bin")

        # Apply binning method
        if method == BinningMethod.EQUAL_FREQUENCY:
            edges, assigns = self.equal_frequency_binning(x_valid, self.n_bins)
        elif method == BinningMethod.EQUAL_WIDTH:
            edges, assigns = self.equal_width_binning(x_valid, self.n_bins)
        elif method == BinningMethod.QUANTILE:
            edges, assigns = self.quantile_binning(x_valid, self.n_bins)
        elif method == BinningMethod.OPTIMAL:
            edges, assigns = self.optimal_binning(x_valid, y_valid, self.n_bins)
        elif method == BinningMethod.MONOTONIC:
            edges, assigns = self.optimal_binning(x_valid, y_valid, self.n_bins)
        elif method == BinningMethod.CUSTOM:
            if custom_cuts is None:
                raise ValueError("custom_cuts required for CUSTOM method")
            edges, assigns = self.custom_binning(x_valid, custom_cuts)
        else:
            raise ValueError(f"Unknown binning method: {method}")

        return edges, assigns


# WOE TRANSFORMER

class WOETransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_bins: int = 10,
        min_bin_pct: float = 0.05,
        monotonic: bool = True,
        handle_missing: MissingStrategy = MissingStrategy.SEPARATE_BIN,
        min_iv: float = 0.02,
        max_iv: float = 0.50,
        binning_method: BinningMethod = BinningMethod.MONOTONIC,
        categorical_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.monotonic = monotonic
        self.handle_missing = handle_missing
        self.min_iv = min_iv
        self.max_iv = max_iv
        self.binning_method = binning_method
        # Store None instead of empty list for sklearn clone compatibility
        self.categorical_features = categorical_features
        self.exclude_features = exclude_features
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray]
    ) -> 'WOETransformer':
        # Validate input
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        y = np.asarray(y).ravel()
        if not np.all(np.isin(y[~np.isnan(y)], [0, 1])):
            raise ValueError("y must be binary (0/1)")

        # Initialize storage
        self.woe_dict_: Dict[str, Dict[Any, float]] = {}
        self.iv_dict_: Dict[str, float] = {}
        self.bin_edges_: Dict[str, np.ndarray] = {}
        self.feature_results_: Dict[str, FeatureWOEResult] = {}
        self.feature_types_: Dict[str, str] = {}

        # Total good and bad
        self.total_good_ = (y == 0).sum()
        self.total_bad_ = (y == 1).sum()

        if self.total_bad_ == 0 or self.total_good_ == 0:
            raise ValueError("y must have both positive and negative cases")

        # Process each feature
        exclude = self.exclude_features or []
        features_to_process = [
            col for col in X.columns
            if col not in exclude
        ]

        for feature in features_to_process:
            if self.verbose:
                print(f"Processing feature: {feature}")

            try:
                result = self._fit_feature(X[feature], y, feature)
                if result is not None:
                    self.feature_results_[feature] = result
                    self.woe_dict_[feature] = self._create_woe_mapping(result)
                    self.iv_dict_[feature] = result.iv
                    if result.bin_edges is not None:
                        self.bin_edges_[feature] = result.bin_edges
                    self.feature_types_[feature] = result.feature_type
            except Exception as e:
                if self.verbose:
                    print(f"  Error processing {feature}: {e}")
                continue

        # Select features based on IV
        self.selected_features_ = [
            f for f, iv in self.iv_dict_.items()
            if iv >= self.min_iv
        ]

        self.is_fitted_ = True
        return self

    def _fit_feature(
        self,
        x: pd.Series,
        y: np.ndarray,
        feature_name: str
    ) -> Optional[FeatureWOEResult]:
        x_values = x.values

        # Determine feature type
        categorical = self.categorical_features or []
        if feature_name in categorical or x.dtype == 'object':
            feature_type = 'categorical'
            return self._fit_categorical(x_values, y, feature_name)
        else:
            feature_type = 'continuous'
            return self._fit_continuous(x_values, y, feature_name)

    def _fit_continuous(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_name: str
    ) -> Optional[FeatureWOEResult]:
        # Separate missing values
        mask_missing = pd.isna(x)
        x_valid = x[~mask_missing]
        y_valid = y[~mask_missing]

        if len(x_valid) == 0:
            return None

        # Create binner and get bins
        binner = WOEBinner(
            method=self.binning_method,
            n_bins=self.max_bins,
            min_bin_pct=self.min_bin_pct,
            monotonic=self.monotonic,
        )

        bin_edges, bin_assignments = binner.fit_bins(x_valid, y_valid)

        # Calculate WOE for each bin
        bins_stats = []
        total_iv = 0.0
        woe_values = []

        for bin_id in range(len(bin_edges) - 1):
            mask = bin_assignments == bin_id
            count = mask.sum()
            good = ((y_valid == 0) & mask).sum()
            bad = ((y_valid == 1) & mask).sum()

            # Calculate WOE
            woe, iv_contrib, distr_good, distr_bad = self._calculate_woe(
                good, bad, self.total_good_, self.total_bad_
            )

            total_iv += iv_contrib
            woe_values.append(woe)

            # Create bin stats
            bin_min = bin_edges[bin_id]
            bin_max = bin_edges[bin_id + 1]

            if bin_id == 0:
                bin_label = f"<= {bin_max:.4g}"
            elif bin_id == len(bin_edges) - 2:
                bin_label = f"> {bin_min:.4g}"
            else:
                bin_label = f"({bin_min:.4g}, {bin_max:.4g}]"

            stats = WOEBinStats(
                bin_id=bin_id,
                bin_label=bin_label,
                bin_min=bin_min,
                bin_max=bin_max,
                count=int(count),
                count_pct=count / len(x_valid),
                good_count=int(good),
                bad_count=int(bad),
                good_pct=good / max(count, 1),
                bad_pct=bad / max(count, 1),
                bad_rate=bad / max(count, 1),
                distr_good=distr_good,
                distr_bad=distr_bad,
                woe=woe,
                iv_contribution=iv_contrib,
            )
            bins_stats.append(stats)

        # Handle missing values
        missing_woe = None
        if mask_missing.any():
            missing_count = mask_missing.sum()
            missing_good = ((y == 0) & mask_missing).sum()
            missing_bad = ((y == 1) & mask_missing).sum()

            if self.handle_missing == MissingStrategy.SEPARATE_BIN:
                missing_woe, iv_missing, dg, db = self._calculate_woe(
                    missing_good, missing_bad,
                    self.total_good_, self.total_bad_
                )
                total_iv += iv_missing

                # Add missing bin
                bins_stats.append(WOEBinStats(
                    bin_id=len(bins_stats),
                    bin_label="Missing",
                    bin_min=None,
                    bin_max=None,
                    count=int(missing_count),
                    count_pct=missing_count / len(x),
                    good_count=int(missing_good),
                    bad_count=int(missing_bad),
                    good_pct=missing_good / max(missing_count, 1),
                    bad_pct=missing_bad / max(missing_count, 1),
                    bad_rate=missing_bad / max(missing_count, 1),
                    distr_good=dg,
                    distr_bad=db,
                    woe=missing_woe,
                    iv_contribution=iv_missing,
                ))
            elif self.handle_missing == MissingStrategy.WOE_ZERO:
                missing_woe = 0.0
            elif self.handle_missing == MissingStrategy.WORST_WOE:
                missing_woe = min(woe_values) if woe_values else 0.0
            elif self.handle_missing == MissingStrategy.BEST_WOE:
                missing_woe = max(woe_values) if woe_values else 0.0

        # Check monotonicity
        is_monotonic = self._check_monotonicity(woe_values)

        return FeatureWOEResult(
            feature_name=feature_name,
            feature_type='continuous',
            n_bins=len(bins_stats),
            iv=total_iv,
            iv_interpretation=self._interpret_iv(total_iv),
            bins=bins_stats,
            bin_edges=bin_edges,
            is_monotonic=is_monotonic,
            missing_woe=missing_woe,
        )

    def _fit_categorical(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_name: str
    ) -> Optional[FeatureWOEResult]:
        # Get unique categories
        categories = pd.Series(x).fillna('_MISSING_').astype(str).unique()

        bins_stats = []
        total_iv = 0.0

        for bin_id, cat in enumerate(sorted(categories)):
            if cat == '_MISSING_':
                mask = pd.isna(x)
            else:
                mask = (x.astype(str) == cat)

            count = mask.sum()
            good = ((y == 0) & mask).sum()
            bad = ((y == 1) & mask).sum()

            woe, iv_contrib, distr_good, distr_bad = self._calculate_woe(
                good, bad, self.total_good_, self.total_bad_
            )
            total_iv += iv_contrib

            stats = WOEBinStats(
                bin_id=bin_id,
                bin_label=str(cat),
                bin_min=None,
                bin_max=None,
                count=int(count),
                count_pct=count / len(x),
                good_count=int(good),
                bad_count=int(bad),
                good_pct=good / max(count, 1),
                bad_pct=bad / max(count, 1),
                bad_rate=bad / max(count, 1),
                distr_good=distr_good,
                distr_bad=distr_bad,
                woe=woe,
                iv_contribution=iv_contrib,
            )
            bins_stats.append(stats)

        # Sort by WOE for potential monotonicity check
        bins_stats.sort(key=lambda x: x.woe)

        return FeatureWOEResult(
            feature_name=feature_name,
            feature_type='categorical',
            n_bins=len(bins_stats),
            iv=total_iv,
            iv_interpretation=self._interpret_iv(total_iv),
            bins=bins_stats,
            bin_edges=None,
            is_monotonic=True,  # N/A for categorical
            missing_woe=next(
                (b.woe for b in bins_stats if b.bin_label == '_MISSING_'),
                None
            ),
        )

    def _calculate_woe(
        self,
        good: int,
        bad: int,
        total_good: int,
        total_bad: int
    ) -> Tuple[float, float, float, float]:
        # Add smoothing to avoid log(0)
        distr_good = (good + SMOOTHING_CONSTANT) / (total_good + 1)
        distr_bad = (bad + SMOOTHING_CONSTANT) / (total_bad + 1)

        woe = np.log(distr_good / distr_bad)
        iv_contrib = (distr_good - distr_bad) * woe

        return woe, iv_contrib, distr_good, distr_bad

    def _create_woe_mapping(
        self,
        result: FeatureWOEResult
    ) -> Dict[Any, float]:
        if result.feature_type == 'categorical':
            # Map category label to WOE
            return {bin.bin_label: bin.woe for bin in result.bins}
        else:
            # For continuous, we'll use bin_id as key
            # Actual mapping happens in transform
            return {bin.bin_id: bin.woe for bin in result.bins}

    def _check_monotonicity(self, woe_values: List[float]) -> bool:
        if len(woe_values) <= 1:
            return True

        diffs = np.diff(woe_values)
        return np.all(diffs >= 0) or np.all(diffs <= 0)

    def _interpret_iv(self, iv: float) -> str:
        if iv < IV_THRESHOLDS['not_useful']:
            return "Not useful for prediction"
        elif iv < IV_THRESHOLDS['weak']:
            return "Weak predictive power"
        elif iv < IV_THRESHOLDS['medium']:
            return "Medium predictive power"
        elif iv < IV_THRESHOLDS['strong']:
            return "Strong predictive power"
        else:
            return "Very strong (possible overfit)"

    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        check_is_fitted(self, ['woe_dict_', 'iv_dict_', 'is_fitted_'])

        X_woe = pd.DataFrame(index=X.index)

        for feature in self.woe_dict_.keys():
            if feature not in X.columns:
                continue

            result = self.feature_results_[feature]

            if result.feature_type == 'categorical':
                X_woe[feature] = self._transform_categorical(
                    X[feature], result
                )
            else:
                X_woe[feature] = self._transform_continuous(
                    X[feature], result
                )

        return X_woe

    def _transform_continuous(
        self,
        x: pd.Series,
        result: FeatureWOEResult
    ) -> pd.Series:
        x_values = x.values
        bin_edges = result.bin_edges
        woe_values = [b.woe for b in result.bins if b.bin_label != 'Missing']

        # Initialize with missing WOE
        x_woe = np.full(len(x), result.missing_woe or 0.0)

        # Assign WOE based on bins
        mask_valid = ~pd.isna(x_values)
        if mask_valid.any():
            bin_assignments = np.digitize(x_values[mask_valid], bin_edges[1:-1])
            # Clip to valid bin range
            bin_assignments = np.clip(bin_assignments, 0, len(woe_values) - 1)
            x_woe[mask_valid] = [woe_values[i] for i in bin_assignments]

        return pd.Series(x_woe, index=x.index, name=x.name)

    def _transform_categorical(
        self,
        x: pd.Series,
        result: FeatureWOEResult
    ) -> pd.Series:
        woe_mapping = {b.bin_label: b.woe for b in result.bins}

        # Handle missing and unknown categories
        default_woe = result.missing_woe or 0.0

        x_woe = x.astype(str).map(woe_mapping).fillna(default_woe)
        return x_woe

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # REPORTING METHODS

    def get_woe_table(
        self,
        feature: str
    ) -> pd.DataFrame:
        check_is_fitted(self, ['feature_results_'])

        if feature not in self.feature_results_:
            raise ValueError(f"Feature '{feature}' not found")

        return self.feature_results_[feature].to_dataframe()

    def get_iv_summary(self) -> pd.DataFrame:
        check_is_fitted(self, ['iv_dict_', 'feature_results_'])

        data = []
        for feature, iv in sorted(self.iv_dict_.items(), key=lambda x: -x[1]):
            result = self.feature_results_[feature]
            data.append({
                'feature': feature,
                'iv': iv,
                'interpretation': result.iv_interpretation,
                'n_bins': result.n_bins,
                'feature_type': result.feature_type,
                'is_monotonic': result.is_monotonic,
                'selected': feature in self.selected_features_,
            })

        return pd.DataFrame(data)

    def get_all_woe_tables(self) -> Dict[str, pd.DataFrame]:
        check_is_fitted(self, ['feature_results_'])

        return {
            feature: result.to_dataframe()
            for feature, result in self.feature_results_.items()
        }

    def plot_woe(
        self,
        feature: str,
        ax=None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        check_is_fitted(self, ['feature_results_'])

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if feature not in self.feature_results_:
            raise ValueError(f"Feature '{feature}' not found")

        result = self.feature_results_[feature]
        df = result.to_dataframe()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Bar plot for WOE
        bars = ax.bar(df['bin_label'], df['woe'], color='steelblue', alpha=0.7)

        # Color bars based on WOE sign
        for bar, woe in zip(bars, df['woe']):
            if woe < 0:
                bar.set_color('indianred')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Bin')
        ax.set_ylabel('WOE')
        ax.set_title(f"WOE Distribution: {feature}\nIV = {result.iv:.4f}")

        # Rotate x labels if too many bins
        if len(df) > 5:
            plt.xticks(rotation=45, ha='right')

        # Add bad rate as secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(df['bin_label'], df['bad_rate'], 'go-', label='Bad Rate')
        ax2.set_ylabel('Bad Rate', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        plt.tight_layout()
        return ax

    def plot_iv_summary(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8)
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        check_is_fitted(self, ['iv_dict_'])

        # Sort by IV
        sorted_iv = sorted(self.iv_dict_.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [x[0] for x in sorted_iv]
        ivs = [x[1] for x in sorted_iv]

        fig, ax = plt.subplots(figsize=figsize)

        # Color by IV threshold
        colors = []
        for iv in ivs:
            if iv >= IV_THRESHOLDS['strong']:
                colors.append('darkred')
            elif iv >= IV_THRESHOLDS['medium']:
                colors.append('orange')
            elif iv >= IV_THRESHOLDS['weak']:
                colors.append('steelblue')
            else:
                colors.append('gray')

        bars = ax.barh(features[::-1], ivs[::-1], color=colors[::-1])

        # Add threshold lines
        for threshold_name, threshold_val in IV_THRESHOLDS.items():
            if threshold_val <= max(ivs):
                ax.axvline(x=threshold_val, color='red', linestyle='--',
                          alpha=0.5, label=f'{threshold_name}: {threshold_val}')

        ax.set_xlabel('Information Value (IV)')
        ax.set_title(f'Top {top_n} Features by Information Value')
        ax.legend(loc='lower right')

        plt.tight_layout()
        return ax


# MODULE EXPORTS

__all__ = [
    # Enums
    "BinningMethod",
    "MissingStrategy",
    # Constants
    "IV_THRESHOLDS",
    "SMOOTHING_CONSTANT",
    # Dataclasses
    "WOEBinStats",
    "FeatureWOEResult",
    # Classes
    "WOEBinner",
    "WOETransformer",
]
