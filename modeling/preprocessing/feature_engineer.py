from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# CONSTANTS AND CONFIGURATIONS

# Rolling window sizes in months
DEFAULT_WINDOWS = [3, 6, 12]

# Age group boundaries
AGE_BINS = [18, 25, 35, 45, 55, 100]
AGE_LABELS = ['18-25', '26-35', '36-45', '46-55', '55+']

# Income level thresholds (in VND millions per month)
INCOME_THRESHOLDS = {
    'low': 10,           # < 10M
    'medium': 20,        # 10M - 20M
    'high': 50,          # 20M - 50M
    'very_high': np.inf  # > 50M
}

# Credit history depth thresholds (in months)
CREDIT_HISTORY_THRESHOLDS = {
    'thin': 12,      # < 12 months
    'moderate': 36,  # 12 - 36 months
    'deep': np.inf   # > 36 months
}

# Trend interpretation thresholds
TREND_THRESHOLDS = {
    'improving': -0.05,    # Negative slope (decreasing risk)
    'deteriorating': 0.05,  # Positive slope (increasing risk)
}


# ENUMS

class TrendDirection(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"


class IncomeLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class CreditHistoryDepth(Enum):
    THIN = "thin"
    MODERATE = "moderate"
    DEEP = "deep"


# TIME SERIES FEATURE ENGINEER

class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        windows: List[int] = None,
        customer_id_col: str = 'customer_id',
        date_col: str = 'period',
        dpd_col: str = 'dpd',
        utilization_col: str = 'credit_utilization',
        balance_col: str = 'balance',
        payment_col: str = 'payment_amount',
        due_date_col: str = 'due_date',
        payment_date_col: str = 'payment_date',
        arpu_col: str = 'arpu',
        include_telecom: bool = True,
    ):
        self.windows = windows
        self.customer_id_col = customer_id_col
        self.date_col = date_col
        self.dpd_col = dpd_col
        self.utilization_col = utilization_col
        self.balance_col = balance_col
        self.payment_col = payment_col
        self.due_date_col = due_date_col
        self.payment_date_col = payment_date_col
        self.arpu_col = arpu_col
        self.include_telecom = include_telecom

    def fit(self, X: pd.DataFrame, y=None) -> 'TimeSeriesFeatureEngineer':
        # Use default windows if not specified
        self._windows = self.windows or DEFAULT_WINDOWS

        # Detect available columns
        self.available_cols_ = set(X.columns)

        # Initialize feature names list
        self.feature_names_ = []

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, 'available_cols_'):
            raise ValueError("Must call fit before transform")

        # Group by customer
        grouped = X.groupby(self.customer_id_col)

        # Generate all features
        features_list = []

        # A. Rolling Window Features
        rolling_features = self._create_rolling_features(grouped)
        features_list.append(rolling_features)

        # B. Trend Features
        trend_features = self._create_trend_features(grouped)
        features_list.append(trend_features)

        # C. Behavioral Pattern Features
        behavioral_features = self._create_behavioral_features(grouped, X)
        features_list.append(behavioral_features)

        # D. Velocity Features
        velocity_features = self._create_velocity_features(grouped)
        features_list.append(velocity_features)

        # E. VNPT Telecom Features
        if self.include_telecom:
            telecom_features = self._create_telecom_features(grouped, X)
            features_list.append(telecom_features)

        # Combine all features
        result = features_list[0]
        for df in features_list[1:]:
            result = result.join(df, how='outer')

        # Store feature names
        self.feature_names_ = list(result.columns)

        return result.reset_index()

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # A. ROLLING WINDOW FEATURES

    def _create_rolling_features(self, grouped) -> pd.DataFrame:
        features = {}

        for window in self._windows:
            window_suffix = f"_{window}m"

            # DPD features
            if self.dpd_col in self.available_cols_:
                features[f'max_dpd{window_suffix}'] = grouped[self.dpd_col].apply(
                    lambda x: x.tail(window).max() if len(x) >= 1 else np.nan
                )
                features[f'avg_dpd{window_suffix}'] = grouped[self.dpd_col].apply(
                    lambda x: x.tail(window).mean() if len(x) >= 1 else np.nan
                )
                features[f'count_dpd30plus{window_suffix}'] = grouped[self.dpd_col].apply(
                    lambda x: (x.tail(window) > 30).sum() if len(x) >= 1 else 0
                )

            # Utilization features
            if self.utilization_col in self.available_cols_:
                features[f'max_utilization{window_suffix}'] = grouped[self.utilization_col].apply(
                    lambda x: x.tail(window).max() if len(x) >= 1 else np.nan
                )
                features[f'avg_utilization{window_suffix}'] = grouped[self.utilization_col].apply(
                    lambda x: x.tail(window).mean() if len(x) >= 1 else np.nan
                )

            # Balance features
            if self.balance_col in self.available_cols_:
                features[f'min_balance{window_suffix}'] = grouped[self.balance_col].apply(
                    lambda x: x.tail(window).min() if len(x) >= 1 else np.nan
                )
                features[f'avg_balance{window_suffix}'] = grouped[self.balance_col].apply(
                    lambda x: x.tail(window).mean() if len(x) >= 1 else np.nan
                )
                features[f'balance_volatility{window_suffix}'] = grouped[self.balance_col].apply(
                    lambda x: self._coefficient_of_variation(x.tail(window))
                )

        return pd.DataFrame(features)

    def _coefficient_of_variation(self, series: pd.Series) -> float:
        mean_val = series.mean()
        if pd.isna(mean_val) or mean_val == 0:
            return np.nan
        return series.std() / abs(mean_val)

    # B. TREND FEATURES

    def _create_trend_features(self, grouped) -> pd.DataFrame:
        features = {}

        # DPD trend
        if self.dpd_col in self.available_cols_:
            features['dpd_trend'] = grouped[self.dpd_col].apply(self._calculate_slope)

        # Utilization trend
        if self.utilization_col in self.available_cols_:
            features['utilization_trend'] = grouped[self.utilization_col].apply(
                self._calculate_slope
            )

        # Balance trend
        if self.balance_col in self.available_cols_:
            features['balance_trend'] = grouped[self.balance_col].apply(
                self._calculate_slope_normalized
            )

        # Income trend (if available)
        if 'income' in self.available_cols_:
            features['income_trend'] = grouped['income'].apply(
                self._calculate_slope_normalized
            )

        # Overall trend direction (based on DPD if available)
        if 'dpd_trend' in features:
            features['trend_direction'] = features['dpd_trend'].apply(
                self._interpret_trend
            )

        return pd.DataFrame(features)

    def _calculate_slope(self, series: pd.Series) -> float:
        if len(series) < 2:
            return 0.0

        series = series.dropna()
        if len(series) < 2:
            return 0.0

        x = np.arange(len(series))
        y = series.values

        try:
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope if not np.isnan(slope) else 0.0
        except Exception:
            return 0.0

    def _calculate_slope_normalized(self, series: pd.Series) -> float:
        if len(series) < 2:
            return 0.0

        series = series.dropna()
        if len(series) < 2:
            return 0.0

        mean_val = series.mean()
        if mean_val == 0:
            return 0.0

        slope = self._calculate_slope(series)
        return slope / abs(mean_val)

    def _interpret_trend(self, slope: float) -> str:
        if slope < TREND_THRESHOLDS['improving']:
            return TrendDirection.IMPROVING.value
        elif slope > TREND_THRESHOLDS['deteriorating']:
            return TrendDirection.DETERIORATING.value
        else:
            return TrendDirection.STABLE.value

    # C. BEHAVIORAL PATTERN FEATURES

    def _create_behavioral_features(
        self,
        grouped,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        features = {}

        # Payment consistency (% of on-time payments)
        if self.dpd_col in self.available_cols_:
            features['payment_consistency'] = grouped[self.dpd_col].apply(
                lambda x: (x <= 0).mean() if len(x) > 0 else np.nan
            )

        # Early payment ratio
        if self.payment_date_col in self.available_cols_ and self.due_date_col in self.available_cols_:
            features['early_payment_ratio'] = grouped.apply(
                lambda g: self._calculate_early_payment_ratio(g)
            )
        elif self.dpd_col in self.available_cols_:
            # Approximate using DPD
            features['early_payment_ratio'] = grouped[self.dpd_col].apply(
                lambda x: (x < 0).mean() if len(x) > 0 else np.nan
            )

        # Late payment severity (weighted average of late days)
        if self.dpd_col in self.available_cols_:
            features['late_payment_severity'] = grouped[self.dpd_col].apply(
                self._calculate_late_payment_severity
            )

        # Recovery flag (improvement after DPD event)
        if self.dpd_col in self.available_cols_:
            features['recovery_flag'] = grouped[self.dpd_col].apply(
                self._detect_recovery
            )

        # Seasonal pattern detection
        if self.dpd_col in self.available_cols_ and self.date_col in self.available_cols_:
            features['seasonal_pattern'] = grouped.apply(
                lambda g: self._detect_seasonal_pattern(g),
                include_groups=False
            )

        return pd.DataFrame(features)

    def _calculate_early_payment_ratio(self, group: pd.DataFrame) -> float:
        if self.payment_date_col not in group.columns or self.due_date_col not in group.columns:
            return np.nan

        try:
            payment_dates = pd.to_datetime(group[self.payment_date_col])
            due_dates = pd.to_datetime(group[self.due_date_col])
            early = (payment_dates < due_dates).sum()
            total = len(group)
            return early / total if total > 0 else np.nan
        except Exception:
            return np.nan

    def _calculate_late_payment_severity(self, dpd_series: pd.Series) -> float:
        dpd_series = dpd_series.dropna()
        if len(dpd_series) == 0:
            return 0.0

        # Only consider late payments
        late = dpd_series[dpd_series > 0]
        if len(late) == 0:
            return 0.0

        # Calculate weighted severity
        weights = np.where(late > 90, 3,
                          np.where(late > 60, 2,
                                  np.where(late > 30, 1, 0.5)))

        weighted_sum = (late * weights).sum()
        return weighted_sum / len(dpd_series)

    def _detect_recovery(self, dpd_series: pd.Series) -> int:
        dpd_series = dpd_series.dropna()
        if len(dpd_series) < 3:
            return 0

        # Find peak DPD
        peak_idx = dpd_series.idxmax()
        peak_val = dpd_series[peak_idx]

        if peak_val <= 0:
            return 0  # No DPD event

        # Check if DPD improved after peak
        try:
            peak_pos = dpd_series.index.get_loc(peak_idx)
            if peak_pos >= len(dpd_series) - 1:
                return 0  # Peak is at end

            after_peak = dpd_series.iloc[peak_pos + 1:]
            if len(after_peak) > 0 and after_peak.mean() < peak_val * 0.5:
                return 1  # Significant improvement
        except Exception:
            pass

        return 0

    def _detect_seasonal_pattern(self, group: pd.DataFrame) -> int:
        if self.dpd_col not in group.columns or self.date_col not in group.columns:
            return 0

        if len(group) < 12:
            return 0  # Need at least 12 months

        try:
            # Extract month from date
            dates = pd.to_datetime(group[self.date_col])
            months = dates.dt.month
            dpd = group[self.dpd_col]

            # Calculate mean DPD by month
            monthly_avg = pd.DataFrame({'month': months, 'dpd': dpd}).groupby('month')['dpd'].mean()

            if len(monthly_avg) < 4:
                return 0

            # Check for significant variation (CV > 0.3)
            cv = monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() != 0 else 0
            return 1 if cv > 0.3 else 0
        except Exception:
            return 0

    # D. VELOCITY FEATURES

    def _create_velocity_features(self, grouped) -> pd.DataFrame:
        features = {}

        # DPD velocity (month-over-month change)
        if self.dpd_col in self.available_cols_:
            features['dpd_velocity'] = grouped[self.dpd_col].apply(
                self._calculate_velocity
            )
            features['dpd_acceleration'] = grouped[self.dpd_col].apply(
                self._calculate_acceleration
            )

        # Utilization velocity
        if self.utilization_col in self.available_cols_:
            features['util_velocity'] = grouped[self.utilization_col].apply(
                self._calculate_velocity
            )

        # Inquiry velocity (if available)
        if 'num_inquiries' in self.available_cols_:
            features['num_inquiries_velocity'] = grouped['num_inquiries'].apply(
                self._calculate_velocity
            )

        # Balance velocity
        if self.balance_col in self.available_cols_:
            features['balance_velocity'] = grouped[self.balance_col].apply(
                self._calculate_velocity_normalized
            )

        return pd.DataFrame(features)

    def _calculate_velocity(self, series: pd.Series) -> float:
        series = series.dropna()
        if len(series) < 2:
            return 0.0

        diffs = series.diff().dropna()
        return diffs.mean() if len(diffs) > 0 else 0.0

    def _calculate_velocity_normalized(self, series: pd.Series) -> float:
        series = series.dropna()
        if len(series) < 2:
            return 0.0

        mean_val = series.mean()
        if mean_val == 0:
            return 0.0

        velocity = self._calculate_velocity(series)
        return velocity / abs(mean_val)

    def _calculate_acceleration(self, series: pd.Series) -> float:
        series = series.dropna()
        if len(series) < 3:
            return 0.0

        # First derivative (velocity)
        velocity = series.diff().dropna()

        # Second derivative (acceleration)
        acceleration = velocity.diff().dropna()

        return acceleration.mean() if len(acceleration) > 0 else 0.0

    # E. VNPT TELECOM FEATURES

    def _create_telecom_features(
        self,
        grouped,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        features = {}

        # Telecom payment consistency
        if 'telecom_dpd' in self.available_cols_:
            features['telecom_payment_consistency'] = grouped['telecom_dpd'].apply(
                lambda x: (x <= 0).mean() if len(x) > 0 else np.nan
            )
        elif 'bill_payment_status' in self.available_cols_:
            features['telecom_payment_consistency'] = grouped['bill_payment_status'].apply(
                lambda x: (x == 'on_time').mean() if len(x) > 0 else np.nan
            )

        # ARPU stability
        if self.arpu_col in self.available_cols_:
            features['arpu_stability'] = grouped[self.arpu_col].apply(
                lambda x: 1 - self._coefficient_of_variation(x)
                if self._coefficient_of_variation(x) is not np.nan else np.nan
            )
            features['avg_arpu'] = grouped[self.arpu_col].mean()
            features['arpu_trend'] = grouped[self.arpu_col].apply(
                self._calculate_slope_normalized
            )

        # Usage growth
        if 'data_usage' in self.available_cols_:
            features['usage_growth'] = grouped['data_usage'].apply(
                self._calculate_slope_normalized
            )
        if 'voice_minutes' in self.available_cols_:
            features['voice_usage_trend'] = grouped['voice_minutes'].apply(
                self._calculate_slope_normalized
            )

        # Cross-sell score (aggregate from static data if available)
        if 'num_services' in self.available_cols_:
            features['cross_sell_score'] = grouped['num_services'].last()

        # Digital engagement
        digital_cols = [
            'app_logins', 'online_payments', 'digital_transactions',
            'app_sessions', 'web_sessions'
        ]
        available_digital = [c for c in digital_cols if c in self.available_cols_]

        if available_digital:
            # Calculate average digital engagement across available columns
            for col in available_digital:
                features[f'{col}_avg'] = grouped[col].mean()

            # Overall digital engagement score
            features['digital_engagement'] = grouped.apply(
                lambda g: self._calculate_digital_engagement(g, available_digital)
            )

        # Tenure and loyalty indicators
        if 'tenure_months' in self.available_cols_:
            features['telecom_tenure'] = grouped['tenure_months'].last()

        if 'contract_type' in self.available_cols_:
            features['is_postpaid'] = grouped['contract_type'].apply(
                lambda x: 1 if x.iloc[-1] == 'postpaid' else 0
            )

        return pd.DataFrame(features)

    def _calculate_digital_engagement(
        self,
        group: pd.DataFrame,
        digital_cols: List[str]
    ) -> float:
        if not digital_cols:
            return np.nan

        scores = []
        for col in digital_cols:
            if col in group.columns:
                val = group[col].mean()
                if not pd.isna(val):
                    scores.append(val)

        if not scores:
            return np.nan

        # Normalize to 0-1 scale (assuming reasonable max values)
        return min(sum(scores) / (len(scores) * 100), 1.0)


# STATIC FEATURE ENGINEER

class StaticFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        create_ratios: bool = True,
        create_interactions: bool = True,
        create_buckets: bool = True,
        income_col: str = 'monthly_income',
        debt_col: str = 'total_debt',
        savings_col: str = 'savings_balance',
        expense_col: str = 'monthly_expenses',
        credit_limit_col: str = 'total_credit_limit',
        credit_used_col: str = 'credit_used',
        loan_amount_col: str = 'loan_amount',
        collateral_col: str = 'collateral_value',
        age_col: str = 'age',
        job_tenure_col: str = 'employment_years',
        address_tenure_col: str = 'address_stability_years',
        num_accounts_col: str = 'num_active_accounts',
        credit_history_col: str = 'credit_history_months',
    ):
        self.create_ratios = create_ratios
        self.create_interactions = create_interactions
        self.create_buckets = create_buckets
        self.income_col = income_col
        self.debt_col = debt_col
        self.savings_col = savings_col
        self.expense_col = expense_col
        self.credit_limit_col = credit_limit_col
        self.credit_used_col = credit_used_col
        self.loan_amount_col = loan_amount_col
        self.collateral_col = collateral_col
        self.age_col = age_col
        self.job_tenure_col = job_tenure_col
        self.address_tenure_col = address_tenure_col
        self.num_accounts_col = num_accounts_col
        self.credit_history_col = credit_history_col

    def fit(self, X: pd.DataFrame, y=None) -> 'StaticFeatureEngineer':
        self.available_cols_ = set(X.columns)
        self.feature_names_ = []

        # Calculate income percentiles for bucketing
        if self.income_col in self.available_cols_:
            self.income_percentiles_ = X[self.income_col].quantile([0.25, 0.5, 0.75]).values

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, 'available_cols_'):
            raise ValueError("Must call fit before transform")

        # Start with copy of original data
        result = X.copy()

        # A. Ratio Features
        if self.create_ratios:
            ratio_features = self._create_ratio_features(X)
            result = pd.concat([result, ratio_features], axis=1)

        # B. Interaction Features
        if self.create_interactions:
            interaction_features = self._create_interaction_features(X)
            result = pd.concat([result, interaction_features], axis=1)

        # C. Bucketing Features
        if self.create_buckets:
            bucket_features = self._create_bucket_features(X)
            result = pd.concat([result, bucket_features], axis=1)

        self.feature_names_ = list(result.columns)
        return result

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # A. RATIO FEATURES

    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = {}

        # Debt-to-Income ratio
        if self.debt_col in self.available_cols_ and self.income_col in self.available_cols_:
            income = df[self.income_col].replace(0, np.nan)
            features['dti_ratio'] = df[self.debt_col] / income
            features['dti_ratio'] = features['dti_ratio'].clip(0, 10)  # Cap at 1000%

        # Savings ratio
        if self.savings_col in self.available_cols_ and self.income_col in self.available_cols_:
            income = df[self.income_col].replace(0, np.nan)
            features['savings_ratio'] = df[self.savings_col] / income
            features['savings_ratio'] = features['savings_ratio'].clip(0, 100)

        # Expense ratio
        if self.expense_col in self.available_cols_ and self.income_col in self.available_cols_:
            income = df[self.income_col].replace(0, np.nan)
            features['expense_ratio'] = df[self.expense_col] / income
            features['expense_ratio'] = features['expense_ratio'].clip(0, 2)

        # Credit utilization
        if self.credit_used_col in self.available_cols_ and self.credit_limit_col in self.available_cols_:
            limit = df[self.credit_limit_col].replace(0, np.nan)
            features['credit_utilization_ratio'] = df[self.credit_used_col] / limit
            features['credit_utilization_ratio'] = features['credit_utilization_ratio'].clip(0, 2)

        # Loan-to-Value ratio
        if self.loan_amount_col in self.available_cols_ and self.collateral_col in self.available_cols_:
            collateral = df[self.collateral_col].replace(0, np.nan)
            features['loan_to_value'] = df[self.loan_amount_col] / collateral
            features['loan_to_value'] = features['loan_to_value'].clip(0, 2)

        # Disposable income ratio
        if self.income_col in self.available_cols_ and self.expense_col in self.available_cols_:
            features['disposable_income_ratio'] = (
                (df[self.income_col] - df[self.expense_col]) /
                df[self.income_col].replace(0, np.nan)
            ).clip(-1, 1)

        # Payment capacity (income - expenses - debt service)
        if all(col in self.available_cols_
               for col in [self.income_col, self.expense_col, self.debt_col]):
            # Assume 10% of debt as monthly payment
            debt_service = df[self.debt_col] * 0.10
            payment_capacity = df[self.income_col] - df[self.expense_col] - debt_service
            features['payment_capacity'] = payment_capacity
            features['payment_capacity_ratio'] = (
                payment_capacity / df[self.income_col].replace(0, np.nan)
            ).clip(-2, 2)

        return pd.DataFrame(features, index=df.index)

    # B. INTERACTION FEATURES

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = {}

        # Age-Income interaction
        if self.age_col in self.available_cols_ and self.income_col in self.available_cols_:
            # Normalize both to 0-1 scale
            age_norm = (df[self.age_col] - 18) / (70 - 18)  # 18-70 range
            income_norm = df[self.income_col] / df[self.income_col].quantile(0.95)
            features['age_income_interaction'] = age_norm * income_norm.clip(0, 2)

        # Tenure stability (job + address)
        if self.job_tenure_col in self.available_cols_ and self.address_tenure_col in self.available_cols_:
            job_score = np.log1p(df[self.job_tenure_col])
            address_score = np.log1p(df[self.address_tenure_col])
            features['tenure_stability'] = job_score * address_score

        # Credit depth (accounts * history)
        if self.num_accounts_col in self.available_cols_ and self.credit_history_col in self.available_cols_:
            features['credit_depth'] = (
                df[self.num_accounts_col] * np.log1p(df[self.credit_history_col])
            )

        # Income per dependent (if available)
        if self.income_col in self.available_cols_ and 'num_dependents' in self.available_cols_:
            dependents = df['num_dependents'].replace(0, 1)  # Avoid division by zero
            features['income_per_dependent'] = df[self.income_col] / dependents

        # Employment income ratio
        if self.income_col in self.available_cols_ and self.job_tenure_col in self.available_cols_:
            tenure = df[self.job_tenure_col].replace(0, 0.5)
            features['income_tenure_ratio'] = df[self.income_col] / tenure

        # Age-employment interaction
        if self.age_col in self.available_cols_ and self.job_tenure_col in self.available_cols_:
            # Ratio of job tenure to working years since 18
            working_years = (df[self.age_col] - 18).clip(1, None)
            features['job_stability_ratio'] = df[self.job_tenure_col] / working_years

        return pd.DataFrame(features, index=df.index)

    # C. BUCKETING FEATURES

    def _create_bucket_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = {}

        # Age group
        if self.age_col in self.available_cols_:
            features['age_group'] = pd.cut(
                df[self.age_col],
                bins=AGE_BINS,
                labels=AGE_LABELS,
                include_lowest=True
            )
            # Also create numeric version
            features['age_group_code'] = pd.cut(
                df[self.age_col],
                bins=AGE_BINS,
                labels=list(range(len(AGE_LABELS))),
                include_lowest=True
            ).astype(float)

        # Income level
        if self.income_col in self.available_cols_:
            income = df[self.income_col] / 1_000_000  # Convert to millions
            features['income_level'] = pd.cut(
                income,
                bins=[0, INCOME_THRESHOLDS['low'], INCOME_THRESHOLDS['medium'],
                      INCOME_THRESHOLDS['high'], np.inf],
                labels=['low', 'medium', 'high', 'very_high'],
                include_lowest=True
            )
            features['income_level_code'] = pd.cut(
                income,
                bins=[0, INCOME_THRESHOLDS['low'], INCOME_THRESHOLDS['medium'],
                      INCOME_THRESHOLDS['high'], np.inf],
                labels=[0, 1, 2, 3],
                include_lowest=True
            ).astype(float)

        # Credit history depth
        if self.credit_history_col in self.available_cols_:
            features['credit_history_depth'] = pd.cut(
                df[self.credit_history_col],
                bins=[0, CREDIT_HISTORY_THRESHOLDS['thin'],
                      CREDIT_HISTORY_THRESHOLDS['moderate'], np.inf],
                labels=['thin', 'moderate', 'deep'],
                include_lowest=True
            )
            features['credit_history_depth_code'] = pd.cut(
                df[self.credit_history_col],
                bins=[0, CREDIT_HISTORY_THRESHOLDS['thin'],
                      CREDIT_HISTORY_THRESHOLDS['moderate'], np.inf],
                labels=[0, 1, 2],
                include_lowest=True
            ).astype(float)

        # DTI bucket (if DTI can be calculated)
        if self.debt_col in self.available_cols_ and self.income_col in self.available_cols_:
            income = df[self.income_col].replace(0, np.nan)
            dti = df[self.debt_col] / income
            features['dti_bucket'] = pd.cut(
                dti,
                bins=[0, 0.2, 0.4, 0.6, np.inf],
                labels=['low', 'moderate', 'high', 'very_high'],
                include_lowest=True
            )

        # Utilization bucket
        if self.credit_used_col in self.available_cols_ and self.credit_limit_col in self.available_cols_:
            limit = df[self.credit_limit_col].replace(0, np.nan)
            util = df[self.credit_used_col] / limit
            features['utilization_bucket'] = pd.cut(
                util,
                bins=[0, 0.3, 0.5, 0.7, 0.9, np.inf],
                labels=['low', 'moderate', 'high', 'very_high', 'maxed'],
                include_lowest=True
            )

        return pd.DataFrame(features, index=df.index)


# MISSING FEATURE ENGINEER

class MissingFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        create_indicators: bool = True,
        create_counts: bool = True,
        create_clusters: bool = True,
        create_thin_file_score: bool = True,
        n_clusters: int = 5,
        indicator_suffix: str = '_missing',
        thin_file_threshold: float = 0.3,
        critical_features: List[str] = None,
    ):
        self.create_indicators = create_indicators
        self.create_counts = create_counts
        self.create_clusters = create_clusters
        self.create_thin_file_score = create_thin_file_score
        self.n_clusters = n_clusters
        self.indicator_suffix = indicator_suffix
        self.thin_file_threshold = thin_file_threshold
        self.critical_features = critical_features

    def fit(self, X: pd.DataFrame, y=None) -> 'MissingFeatureEngineer':
        # Store column info
        self.columns_ = list(X.columns)
        self.n_features_ = len(self.columns_)

        # Default critical features for credit scoring
        self._critical_features = self.critical_features or [
            'income', 'monthly_income', 'credit_score', 'cic_score',
            'employment_years', 'job_tenure', 'age', 'dpd', 'max_dpd',
            'credit_history_months', 'num_active_accounts'
        ]

        # Find which critical features are present
        self.critical_present_ = [
            col for col in self._critical_features
            if col in self.columns_
        ]

        # Calculate baseline missing rates
        self.missing_rates_ = X.isna().mean()

        # Fit cluster model on missing patterns
        if self.create_clusters:
            missing_matrix = X.isna().astype(int)

            # Only cluster if there's variation
            if missing_matrix.sum().sum() > 0:
                self.cluster_model_ = KMeans(
                    n_clusters=min(self.n_clusters, len(X)),
                    random_state=42,
                    n_init=10
                )
                self.cluster_model_.fit(missing_matrix)
            else:
                self.cluster_model_ = None

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, 'columns_'):
            raise ValueError("Must call fit before transform")

        result = X.copy()

        # Create missing indicators
        if self.create_indicators:
            indicator_features = self._create_missing_indicators(X)
            result = pd.concat([result, indicator_features], axis=1)

        # Create missing counts
        if self.create_counts:
            count_features = self._create_missing_counts(X)
            result = pd.concat([result, count_features], axis=1)

        # Create pattern clusters
        if self.create_clusters and self.cluster_model_ is not None:
            cluster_features = self._create_pattern_clusters(X)
            result = pd.concat([result, cluster_features], axis=1)

        # Create thin-file score
        if self.create_thin_file_score:
            thin_file_features = self._create_thin_file_score(X)
            result = pd.concat([result, thin_file_features], axis=1)

        return result

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _create_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        indicators = {}

        for col in df.columns:
            missing_rate = df[col].isna().mean()
            # Only create indicator if there are some missing values
            # but not all missing (which would be useless)
            if 0 < missing_rate < 1:
                indicators[f'{col}{self.indicator_suffix}'] = df[col].isna().astype(int)

        return pd.DataFrame(indicators, index=df.index)

    def _create_missing_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        features = {}

        # Total missing count
        features['missing_count'] = df.isna().sum(axis=1)

        # Missing percentage
        features['missing_pct'] = features['missing_count'] / len(df.columns)

        # Critical features missing count
        critical_cols = [c for c in self.critical_present_ if c in df.columns]
        if critical_cols:
            features['critical_missing_count'] = df[critical_cols].isna().sum(axis=1)
            features['critical_missing_pct'] = (
                features['critical_missing_count'] / len(critical_cols)
            )

        # Missing by category
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(numeric_cols) > 0:
            features['numeric_missing_count'] = df[numeric_cols].isna().sum(axis=1)
        if len(categorical_cols) > 0:
            features['categorical_missing_count'] = df[categorical_cols].isna().sum(axis=1)

        return pd.DataFrame(features, index=df.index)

    def _create_pattern_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        features = {}

        # Create missing pattern matrix
        missing_matrix = df.isna().astype(int)

        # Predict clusters
        features['missing_pattern_cluster'] = self.cluster_model_.predict(missing_matrix)

        # Distance to cluster center (indicates unusual missing pattern)
        distances = self.cluster_model_.transform(missing_matrix)
        features['missing_pattern_distance'] = distances.min(axis=1)

        return pd.DataFrame(features, index=df.index)

    def _create_thin_file_score(self, df: pd.DataFrame) -> pd.DataFrame:
        features = {}

        # Base thin-file score on missing percentage
        missing_pct = df.isna().mean(axis=1)

        # Weighted score based on critical features
        critical_cols = [c for c in self.critical_present_ if c in df.columns]

        if critical_cols:
            critical_missing = df[critical_cols].isna().mean(axis=1)
            # Combine general and critical missing
            thin_file_raw = 0.3 * missing_pct + 0.7 * critical_missing
        else:
            thin_file_raw = missing_pct

        # Normalize to 0-1 scale
        features['thin_file_score'] = thin_file_raw.clip(0, 1)

        # Binary thin-file flag
        features['is_thin_file'] = (
            features['thin_file_score'] >= self.thin_file_threshold
        ).astype(int)

        # Thin-file category
        features['thin_file_category'] = pd.cut(
            features['thin_file_score'],
            bins=[0, 0.1, 0.3, 0.5, 1.0],
            labels=['full_file', 'moderate_file', 'thin_file', 'very_thin_file'],
            include_lowest=True
        )

        return pd.DataFrame(features, index=df.index)


# MAIN CREDIT FEATURE ENGINEER

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        # Time series parameters
        windows: List[int] = None,
        include_telecom: bool = True,
        # Static parameters
        create_ratios: bool = True,
        create_interactions: bool = True,
        create_buckets: bool = True,
        # Missing parameters
        create_missing_indicators: bool = True,
        create_missing_clusters: bool = True,
        create_thin_file_score: bool = True,
        # Column mappings
        customer_id_col: str = 'customer_id',
        date_col: str = 'period',
    ):
        self.windows = windows
        self.include_telecom = include_telecom
        self.create_ratios = create_ratios
        self.create_interactions = create_interactions
        self.create_buckets = create_buckets
        self.create_missing_indicators = create_missing_indicators
        self.create_missing_clusters = create_missing_clusters
        self.create_thin_file_score = create_thin_file_score
        self.customer_id_col = customer_id_col
        self.date_col = date_col

    def fit(
        self,
        X_static: pd.DataFrame,
        X_timeseries: Optional[pd.DataFrame] = None,
        y=None
    ) -> 'CreditFeatureEngineer':
        # Initialize engineers
        self.ts_engineer_ = TimeSeriesFeatureEngineer(
            windows=self.windows,
            customer_id_col=self.customer_id_col,
            date_col=self.date_col,
            include_telecom=self.include_telecom,
        )

        self.static_engineer_ = StaticFeatureEngineer(
            create_ratios=self.create_ratios,
            create_interactions=self.create_interactions,
            create_buckets=self.create_buckets,
        )

        self.missing_engineer_ = MissingFeatureEngineer(
            create_indicators=self.create_missing_indicators,
            create_clusters=self.create_missing_clusters,
            create_thin_file_score=self.create_thin_file_score,
        )

        # Fit static engineer
        self.static_engineer_.fit(X_static)

        # Fit time series engineer if data provided
        if X_timeseries is not None and len(X_timeseries) > 0:
            self.ts_engineer_.fit(X_timeseries)
            self.has_timeseries_ = True
        else:
            self.has_timeseries_ = False

        return self

    def transform(
        self,
        X_static: pd.DataFrame,
        X_timeseries: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        # Transform static features
        static_features = self.static_engineer_.transform(X_static)

        # Transform time series if available
        if self.has_timeseries_ and X_timeseries is not None:
            ts_features = self.ts_engineer_.transform(X_timeseries)

            # Merge on customer_id
            if self.customer_id_col in static_features.columns:
                combined = static_features.merge(
                    ts_features,
                    on=self.customer_id_col,
                    how='left'
                )
            else:
                # Assume same index
                combined = pd.concat([static_features, ts_features], axis=1)
        else:
            combined = static_features

        # Add missing features (after combining)
        # Fit missing engineer on combined data first
        self.missing_engineer_.fit(combined)
        result = self.missing_engineer_.transform(combined)

        # Store feature names
        self.feature_names_ = list(result.columns)

        return result

    def fit_transform(
        self,
        X_static: pd.DataFrame,
        X_timeseries: Optional[pd.DataFrame] = None,
        y=None
    ) -> pd.DataFrame:
        return self.fit(X_static, X_timeseries, y).transform(X_static, X_timeseries)

    def get_feature_names(self) -> List[str]:
        if hasattr(self, 'feature_names_'):
            return self.feature_names_
        return []

    def get_feature_groups(self) -> Dict[str, List[str]]:
        groups = {
            'static_original': [],
            'static_ratios': [],
            'static_interactions': [],
            'static_buckets': [],
            'timeseries_rolling': [],
            'timeseries_trend': [],
            'timeseries_behavioral': [],
            'timeseries_velocity': [],
            'timeseries_telecom': [],
            'missing_indicators': [],
            'missing_scores': [],
        }

        for feature in self.feature_names_:
            if feature.endswith('_missing'):
                groups['missing_indicators'].append(feature)
            elif 'thin_file' in feature or 'missing_count' in feature:
                groups['missing_scores'].append(feature)
            elif any(f'_{w}m' in feature for w in (self.windows or DEFAULT_WINDOWS)):
                groups['timeseries_rolling'].append(feature)
            elif 'trend' in feature:
                groups['timeseries_trend'].append(feature)
            elif any(kw in feature for kw in ['consistency', 'recovery', 'severity', 'seasonal']):
                groups['timeseries_behavioral'].append(feature)
            elif 'velocity' in feature or 'acceleration' in feature:
                groups['timeseries_velocity'].append(feature)
            elif any(kw in feature for kw in ['telecom', 'arpu', 'digital', 'cross_sell']):
                groups['timeseries_telecom'].append(feature)
            elif 'ratio' in feature or 'to_value' in feature:
                groups['static_ratios'].append(feature)
            elif 'interaction' in feature or 'stability' in feature or 'depth' in feature:
                groups['static_interactions'].append(feature)
            elif 'group' in feature or 'level' in feature or 'bucket' in feature:
                groups['static_buckets'].append(feature)
            else:
                groups['static_original'].append(feature)

        return {k: v for k, v in groups.items() if v}


# MODULE EXPORTS

__all__ = [
    # Constants
    "DEFAULT_WINDOWS",
    "AGE_BINS",
    "AGE_LABELS",
    "INCOME_THRESHOLDS",
    "CREDIT_HISTORY_THRESHOLDS",
    "TREND_THRESHOLDS",
    # Enums
    "TrendDirection",
    "IncomeLevel",
    "CreditHistoryDepth",
    # Classes
    "TimeSeriesFeatureEngineer",
    "StaticFeatureEngineer",
    "MissingFeatureEngineer",
    "CreditFeatureEngineer",
]
