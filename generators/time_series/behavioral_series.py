from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import SyntheticDataConfig
from generators.base import BaseDataGenerator, CorrelationMixin, TimeSeriesMixin
from generators.financial import truncated_normal, truncated_lognormal


# ENUMS AND CONSTANTS

class BehaviorPattern(Enum):
    STABLE_GOOD = "stable_good"           # Consistently good behavior
    STABLE_BAD = "stable_bad"             # Consistently bad behavior
    DETERIORATION = "deterioration"       # Good -> Bad over time
    RECOVERY = "recovery"                 # Bad -> Good over time
    CYCLICAL = "cyclical"                 # Good/Bad in cycles
    SUDDEN_DEFAULT = "sudden_default"     # Good then sudden bad
    VOLATILE = "volatile"                 # Random ups and downs


class DPDCategory(Enum):
    CURRENT = "current"          # 0 days - Nhóm 1
    DPD_1_30 = "dpd_1_30"        # 1-30 days
    DPD_31_60 = "dpd_31_60"      # 31-60 days
    DPD_61_90 = "dpd_61_90"      # 61-90 days - Nhóm 2 boundary
    DPD_91_180 = "dpd_91_180"    # 91-180 days - Nhóm 3
    DPD_181_360 = "dpd_181_360"  # 181-360 days - Nhóm 4
    DPD_360_PLUS = "dpd_360_plus"  # 360+ days - Nhóm 5


class TrendDirection(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"


# DPD category to days range mapping
DPD_RANGES: Dict[str, Tuple[int, int]] = {
    "current": (0, 0),
    "dpd_1_30": (1, 30),
    "dpd_31_60": (31, 60),
    "dpd_61_90": (61, 90),
    "dpd_91_180": (91, 180),
    "dpd_181_360": (181, 360),
    "dpd_360_plus": (361, 720),
}

# Pattern to default probability mapping
PATTERN_DEFAULT_PROB: Dict[str, float] = {
    "stable_good": 0.02,
    "stable_bad": 0.60,
    "deterioration": 0.45,
    "recovery": 0.15,
    "cyclical": 0.25,
    "sudden_default": 0.70,
    "volatile": 0.30,
}


# CUSTOMER BEHAVIOR PROFILE

@dataclass
class CustomerBehaviorProfile:
    customer_id: str
    monthly_income: float
    existing_debt: float
    credit_limit: float
    cic_score: int
    has_credit_history: bool
    is_vnpt_customer: bool
    monthly_arpu: float
    contract_type: str
    is_risky: bool = False
    behavior_pattern: BehaviorPattern = BehaviorPattern.STABLE_GOOD

    def __post_init__(self):
        if self.has_credit_history and self.cic_score > 0:
            if self.cic_score >= 700:
                self.is_risky = False
                self.behavior_pattern = BehaviorPattern.STABLE_GOOD
            elif self.cic_score >= 600:
                self.is_risky = False
                # Random pattern selection weighted towards good
                patterns = [
                    BehaviorPattern.STABLE_GOOD,
                    BehaviorPattern.RECOVERY,
                    BehaviorPattern.CYCLICAL,
                ]
                self.behavior_pattern = patterns[hash(self.customer_id) % 3]
            elif self.cic_score >= 500:
                self.is_risky = True
                patterns = [
                    BehaviorPattern.CYCLICAL,
                    BehaviorPattern.DETERIORATION,
                    BehaviorPattern.VOLATILE,
                ]
                self.behavior_pattern = patterns[hash(self.customer_id) % 3]
            else:
                self.is_risky = True
                patterns = [
                    BehaviorPattern.STABLE_BAD,
                    BehaviorPattern.DETERIORATION,
                    BehaviorPattern.SUDDEN_DEFAULT,
                ]
                self.behavior_pattern = patterns[hash(self.customer_id) % 3]
        else:
            # Thin file - assign based on income
            if self.monthly_income > 20_000_000:
                self.is_risky = False
                self.behavior_pattern = BehaviorPattern.STABLE_GOOD
            elif self.monthly_income > 10_000_000:
                self.is_risky = False
                self.behavior_pattern = BehaviorPattern.CYCLICAL
            else:
                self.is_risky = True
                self.behavior_pattern = BehaviorPattern.VOLATILE


# BEHAVIORAL SERIES GENERATOR

class BehavioralSeriesGenerator(BaseDataGenerator, CorrelationMixin, TimeSeriesMixin):
    def __init__(
        self,
        config: SyntheticDataConfig,
        seed: Optional[int] = None,
        n_months: int = 24,
        start_date: Optional[date] = None
    ) -> None:
        super().__init__(config, seed)
        self.n_months = n_months

        # Set start date
        if start_date is None:
            today = date.today()
            year = today.year - (n_months // 12)
            month = today.month - (n_months % 12)
            if month <= 0:
                year -= 1
                month += 12
            self.start_date = date(year, month, 1)
        else:
            self.start_date = start_date

        self.months = self._generate_month_list()

    def _generate_month_list(self) -> List[Tuple[int, int, str]]:
        months = []
        year = self.start_date.year
        month = self.start_date.month

        for _ in range(self.n_months):
            month_id = f"{year}{month:02d}"
            months.append((year, month, month_id))
            month += 1
            if month > 12:
                month = 1
                year += 1

        return months

    # BEHAVIOR PATTERN GENERATORS

    def _generate_pattern_multiplier(
        self,
        pattern: BehaviorPattern,
        n: int
    ) -> np.ndarray:
        t = np.arange(n)

        if pattern == BehaviorPattern.STABLE_GOOD:
            # Low and stable
            base = 0.1 + self.rng.normal(0, 0.05, n)
            return np.clip(base, 0.01, 0.3)

        elif pattern == BehaviorPattern.STABLE_BAD:
            # High and stable
            base = 0.7 + self.rng.normal(0, 0.1, n)
            return np.clip(base, 0.5, 0.95)

        elif pattern == BehaviorPattern.DETERIORATION:
            # Start low, end high (linear increase)
            base = 0.1 + 0.7 * (t / (n - 1))
            noise = self.rng.normal(0, 0.08, n)
            return np.clip(base + noise, 0.01, 0.95)

        elif pattern == BehaviorPattern.RECOVERY:
            # Start high, end low (linear decrease)
            base = 0.8 - 0.6 * (t / (n - 1))
            noise = self.rng.normal(0, 0.08, n)
            return np.clip(base + noise, 0.05, 0.95)

        elif pattern == BehaviorPattern.CYCLICAL:
            # Sinusoidal pattern
            cycle = 0.4 + 0.3 * np.sin(2 * np.pi * t / 6)  # 6-month cycle
            noise = self.rng.normal(0, 0.1, n)
            return np.clip(cycle + noise, 0.1, 0.8)

        elif pattern == BehaviorPattern.SUDDEN_DEFAULT:
            # Good then suddenly bad
            change_point = int(n * 0.7)  # Last 30% is bad
            base = np.zeros(n)
            base[:change_point] = 0.1 + self.rng.normal(0, 0.05, change_point)
            base[change_point:] = 0.8 + self.rng.normal(0, 0.1, n - change_point)
            return np.clip(base, 0.01, 0.95)

        elif pattern == BehaviorPattern.VOLATILE:
            # High variance, random
            base = 0.4 + self.rng.normal(0, 0.25, n)
            return np.clip(base, 0.05, 0.9)

        return np.full(n, 0.3)

    def apply_behavior_pattern(
        self,
        series: np.ndarray,
        pattern: BehaviorPattern,
        invert: bool = False
    ) -> np.ndarray:
        n = len(series)
        multiplier = self._generate_pattern_multiplier(pattern, n)

        if invert:
            multiplier = 1 - multiplier

        return series * multiplier

    # CREDIT BEHAVIOR SERIES

    def generate_dpd_trajectory(
        self,
        profile: CustomerBehaviorProfile
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n_months
        pattern = profile.behavior_pattern

        # Generate base risk multiplier
        risk_mult = self._generate_pattern_multiplier(pattern, n)

        dpd_days = np.zeros(n, dtype=int)
        dpd_category = np.empty(n, dtype=object)

        for i in range(n):
            risk = risk_mult[i]

            # Determine DPD based on risk level
            if risk < 0.15:
                # Current (no DPD)
                dpd_days[i] = 0
                dpd_category[i] = "current"
            elif risk < 0.30:
                # Possibly light DPD
                if self.rng.random() < risk:
                    dpd_days[i] = int(self.rng.integers(1, 15))
                    dpd_category[i] = "dpd_1_30"
                else:
                    dpd_days[i] = 0
                    dpd_category[i] = "current"
            elif risk < 0.50:
                # Light to moderate DPD
                if self.rng.random() < 0.7:
                    dpd_days[i] = int(self.rng.integers(1, 45))
                    dpd_category[i] = "dpd_1_30" if dpd_days[i] <= 30 else "dpd_31_60"
                else:
                    dpd_days[i] = 0
                    dpd_category[i] = "current"
            elif risk < 0.70:
                # Moderate to severe DPD
                dpd_days[i] = int(self.rng.integers(15, 120))
                if dpd_days[i] <= 30:
                    dpd_category[i] = "dpd_1_30"
                elif dpd_days[i] <= 60:
                    dpd_category[i] = "dpd_31_60"
                elif dpd_days[i] <= 90:
                    dpd_category[i] = "dpd_61_90"
                else:
                    dpd_category[i] = "dpd_91_180"
            else:
                # Severe DPD
                dpd_days[i] = int(self.rng.integers(60, 360))
                if dpd_days[i] <= 60:
                    dpd_category[i] = "dpd_31_60"
                elif dpd_days[i] <= 90:
                    dpd_category[i] = "dpd_61_90"
                elif dpd_days[i] <= 180:
                    dpd_category[i] = "dpd_91_180"
                elif dpd_days[i] <= 360:
                    dpd_category[i] = "dpd_181_360"
                else:
                    dpd_category[i] = "dpd_360_plus"

        return dpd_days, dpd_category

    def _generate_credit_series(
        self,
        profile: CustomerBehaviorProfile
    ) -> pd.DataFrame:
        n = self.n_months
        month_ids = [mid for _, _, mid in self.months]

        if not profile.has_credit_history or profile.credit_limit <= 0:
            # No credit history - return empty series
            return pd.DataFrame({
                'customer_id': [profile.customer_id] * n,
                'month_id': month_ids,
                'dpd_days': [0] * n,
                'dpd_category': ['no_credit'] * n,
                'payment_amount': [0.0] * n,
                'due_amount': [0.0] * n,
                'payment_ratio': [1.0] * n,
                'credit_utilization': [0.0] * n,
                'new_credit_inquiries': [0] * n,
                'credit_limit_change': [0.0] * n,
            })

        # Generate DPD trajectory
        dpd_days, dpd_category = self.generate_dpd_trajectory(profile)

        # Generate payment amounts
        monthly_payment_due = profile.existing_debt / 60 * 1.2  # Estimated payment
        due_amount = np.full(n, monthly_payment_due)

        # Payment ratio inversely related to DPD
        payment_ratio = np.zeros(n)
        payment_amount = np.zeros(n)
        for i in range(n):
            if dpd_days[i] == 0:
                payment_ratio[i] = self.rng.uniform(1.0, 1.2)  # Full or more
            elif dpd_days[i] <= 30:
                payment_ratio[i] = self.rng.uniform(0.7, 1.0)
            elif dpd_days[i] <= 60:
                payment_ratio[i] = self.rng.uniform(0.5, 0.8)
            elif dpd_days[i] <= 90:
                payment_ratio[i] = self.rng.uniform(0.3, 0.6)
            else:
                payment_ratio[i] = self.rng.uniform(0.0, 0.4)

            payment_amount[i] = due_amount[i] * payment_ratio[i]

        # Credit utilization (apply pattern)
        base_utilization = profile.existing_debt / profile.credit_limit
        utilization = self.apply_behavior_pattern(
            np.full(n, base_utilization),
            profile.behavior_pattern
        )
        utilization = np.clip(utilization + self.rng.normal(0, 0.1, n), 0.01, 0.99)

        # Credit inquiries (higher when desperate)
        risk_mult = self._generate_pattern_multiplier(profile.behavior_pattern, n)
        inquiries = np.zeros(n, dtype=int)
        for i in range(n):
            if risk_mult[i] > 0.5:
                inquiries[i] = int(self.rng.poisson(risk_mult[i] * 2))
            else:
                inquiries[i] = int(self.rng.poisson(0.3))

        # Credit limit changes
        limit_changes = np.zeros(n)
        for i in range(n):
            if i > 0 and i % 6 == 0:  # Review every 6 months
                if dpd_days[i-3:i].max() == 0 and utilization[i-3:i].mean() < 0.5:
                    # Good behavior - possible increase
                    if self.rng.random() < 0.3:
                        limit_changes[i] = self.rng.uniform(0.1, 0.2)  # 10-20% increase
                elif dpd_days[i-3:i].max() > 60:
                    # Bad behavior - possible decrease
                    if self.rng.random() < 0.5:
                        limit_changes[i] = self.rng.uniform(-0.3, -0.1)  # 10-30% decrease

        # Round amounts
        payment_amount = np.round(payment_amount / 1000) * 1000
        due_amount = np.round(due_amount / 1000) * 1000

        return pd.DataFrame({
            'customer_id': [profile.customer_id] * n,
            'month_id': month_ids,
            'dpd_days': dpd_days,
            'dpd_category': dpd_category,
            'payment_amount': payment_amount,
            'due_amount': due_amount,
            'payment_ratio': np.round(payment_ratio, 4),
            'credit_utilization': np.round(utilization, 4),
            'new_credit_inquiries': inquiries,
            'credit_limit_change': np.round(limit_changes, 4),
        })

    # TELECOM BEHAVIOR SERIES

    def generate_telecom_trajectory(
        self,
        profile: CustomerBehaviorProfile
    ) -> pd.DataFrame:
        n = self.n_months
        month_ids = [mid for _, _, mid in self.months]

        if not profile.is_vnpt_customer or profile.monthly_arpu <= 0:
            # Not VNPT customer - return empty series
            return pd.DataFrame({
                'customer_id': [profile.customer_id] * n,
                'month_id': month_ids,
                'bill_amount': [0.0] * n,
                'payment_date_day': [0] * n,
                'days_to_payment': [0] * n,
                'payment_method': ['none'] * n,
                'data_usage_gb': [0.0] * n,
                'voice_minutes': [0] * n,
                'sms_count': [0] * n,
                'roaming_usage': [False] * n,
                'service_calls': [0] * n,
                'complaint_flag': [False] * n,
            })

        # Generate bill amounts with seasonality
        base_bill = profile.monthly_arpu
        bills = np.zeros(n)
        for i, (year, month, _) in enumerate(self.months):
            # Seasonal adjustment
            seasonal = 1.0
            if month in [1, 2]:
                seasonal = 1.15  # Tet - more usage
            elif month in [6, 7, 8]:
                seasonal = 1.1  # Summer
            elif month == 12:
                seasonal = 1.2  # Year end

            noise = self.rng.normal(1, 0.1)
            bills[i] = base_bill * seasonal * noise

        # Days to payment (correlated with credit behavior)
        pattern = profile.behavior_pattern
        risk_mult = self._generate_pattern_multiplier(pattern, n)

        days_to_payment = np.zeros(n, dtype=int)
        payment_methods = np.empty(n, dtype=object)

        # Payment method probabilities
        if profile.contract_type == "tra_sau":
            method_options = ["bank_transfer", "vi_dien_tu", "auto_debit", "tien_mat", "vnpt_pay"]
            method_probs = [0.25, 0.25, 0.20, 0.15, 0.15]
        else:
            method_options = ["tien_mat", "vi_dien_tu", "bank_transfer", "vnpt_pay"]
            method_probs = [0.35, 0.30, 0.20, 0.15]

        for i in range(n):
            risk = risk_mult[i]

            # Days to payment - bill due around day 20, deadline day 30
            if risk < 0.2:
                days_to_payment[i] = int(self.rng.integers(1, 10))  # Early
            elif risk < 0.4:
                days_to_payment[i] = int(self.rng.integers(5, 20))  # On time
            elif risk < 0.6:
                days_to_payment[i] = int(self.rng.integers(15, 35))  # Near deadline
            elif risk < 0.8:
                days_to_payment[i] = int(self.rng.integers(25, 45))  # Late
            else:
                days_to_payment[i] = int(self.rng.integers(30, 60))  # Very late

            payment_methods[i] = self.rng.choice(method_options, p=method_probs)

        # Usage patterns
        # Base usage determined by ARPU
        if base_bill > 500_000:
            data_base, voice_base = 30, 500
        elif base_bill > 200_000:
            data_base, voice_base = 15, 300
        else:
            data_base, voice_base = 5, 100

        data_usage = truncated_normal(self.rng, data_base, data_base * 0.3, 0.1, data_base * 3, n)
        voice_minutes = truncated_normal(self.rng, voice_base, voice_base * 0.3, 0, voice_base * 3, n).astype(int)
        sms_count = self.rng.poisson(10, n)

        # Roaming (occasional)
        roaming = self.rng.random(n) < 0.05

        # Service calls (correlated with complaints and issues)
        service_calls = np.zeros(n, dtype=int)
        complaint_flag = np.zeros(n, dtype=bool)

        for i in range(n):
            # More service calls when late payer (frustrated)
            if days_to_payment[i] > 30:
                service_calls[i] = int(self.rng.poisson(1.5))
                complaint_flag[i] = self.rng.random() < 0.3
            else:
                service_calls[i] = int(self.rng.poisson(0.3))
                complaint_flag[i] = self.rng.random() < 0.05

        # Round values
        bills = np.round(bills / 1000) * 1000
        data_usage = np.round(data_usage, 2)

        # Payment date day (day of month)
        payment_date_day = np.clip(10 + days_to_payment, 1, 28).astype(int)

        return pd.DataFrame({
            'customer_id': [profile.customer_id] * n,
            'month_id': month_ids,
            'bill_amount': bills,
            'payment_date_day': payment_date_day,
            'days_to_payment': days_to_payment,
            'payment_method': payment_methods,
            'data_usage_gb': data_usage,
            'voice_minutes': voice_minutes,
            'sms_count': sms_count,
            'roaming_usage': roaming,
            'service_calls': service_calls,
            'complaint_flag': complaint_flag,
        })

    # ROLLING / AGGREGATE FEATURES

    def calculate_rolling_features(
        self,
        credit_df: pd.DataFrame,
        telecom_df: pd.DataFrame,
        windows: List[int] = [3, 6, 12]
    ) -> pd.DataFrame:
        customers = credit_df['customer_id'].unique()
        features = []

        for customer_id in customers:
            cust_credit = credit_df[credit_df['customer_id'] == customer_id].copy()
            cust_telecom = telecom_df[telecom_df['customer_id'] == customer_id].copy()

            feature_row = {'customer_id': customer_id}

            # Sort by month
            cust_credit = cust_credit.sort_values('month_id')
            cust_telecom = cust_telecom.sort_values('month_id')

            # ===== Credit Features =====
            dpd_values = cust_credit['dpd_days'].values
            utilization = cust_credit['credit_utilization'].values
            payment_ratio = cust_credit['payment_ratio'].values
            inquiries = cust_credit['new_credit_inquiries'].values

            for w in windows:
                suffix = f"_{w}m"

                # Max DPD in window
                if len(dpd_values) >= w:
                    feature_row[f'max_dpd{suffix}'] = int(dpd_values[-w:].max())
                    feature_row[f'avg_dpd{suffix}'] = float(dpd_values[-w:].mean())
                    feature_row[f'count_dpd_30plus{suffix}'] = int((dpd_values[-w:] > 30).sum())
                    feature_row[f'count_dpd_60plus{suffix}'] = int((dpd_values[-w:] > 60).sum())
                    feature_row[f'count_dpd_90plus{suffix}'] = int((dpd_values[-w:] > 90).sum())
                else:
                    feature_row[f'max_dpd{suffix}'] = int(dpd_values.max()) if len(dpd_values) > 0 else 0
                    feature_row[f'avg_dpd{suffix}'] = float(dpd_values.mean()) if len(dpd_values) > 0 else 0
                    feature_row[f'count_dpd_30plus{suffix}'] = int((dpd_values > 30).sum())
                    feature_row[f'count_dpd_60plus{suffix}'] = int((dpd_values > 60).sum())
                    feature_row[f'count_dpd_90plus{suffix}'] = int((dpd_values > 90).sum())

                # Average utilization
                if len(utilization) >= w:
                    feature_row[f'avg_utilization{suffix}'] = float(utilization[-w:].mean())
                    feature_row[f'max_utilization{suffix}'] = float(utilization[-w:].max())
                else:
                    feature_row[f'avg_utilization{suffix}'] = float(utilization.mean()) if len(utilization) > 0 else 0
                    feature_row[f'max_utilization{suffix}'] = float(utilization.max()) if len(utilization) > 0 else 0

                # Payment ratio
                if len(payment_ratio) >= w:
                    feature_row[f'avg_payment_ratio{suffix}'] = float(payment_ratio[-w:].mean())
                    feature_row[f'min_payment_ratio{suffix}'] = float(payment_ratio[-w:].min())
                else:
                    feature_row[f'avg_payment_ratio{suffix}'] = float(payment_ratio.mean()) if len(payment_ratio) > 0 else 1
                    feature_row[f'min_payment_ratio{suffix}'] = float(payment_ratio.min()) if len(payment_ratio) > 0 else 1

                # Inquiries
                if len(inquiries) >= w:
                    feature_row[f'total_inquiries{suffix}'] = int(inquiries[-w:].sum())
                else:
                    feature_row[f'total_inquiries{suffix}'] = int(inquiries.sum())

            # ===== Telecom Features =====
            days_late = cust_telecom['days_to_payment'].values
            complaints = cust_telecom['complaint_flag'].values
            service_calls = cust_telecom['service_calls'].values
            data_usage = cust_telecom['data_usage_gb'].values

            for w in windows:
                suffix = f"_{w}m"

                if len(days_late) >= w:
                    feature_row[f'avg_days_to_payment{suffix}'] = float(days_late[-w:].mean())
                    feature_row[f'max_days_to_payment{suffix}'] = int(days_late[-w:].max())
                    feature_row[f'count_late_30plus{suffix}'] = int((days_late[-w:] > 30).sum())
                else:
                    feature_row[f'avg_days_to_payment{suffix}'] = float(days_late.mean()) if len(days_late) > 0 else 0
                    feature_row[f'max_days_to_payment{suffix}'] = int(days_late.max()) if len(days_late) > 0 else 0
                    feature_row[f'count_late_30plus{suffix}'] = int((days_late > 30).sum())

                if len(complaints) >= w:
                    feature_row[f'complaint_count{suffix}'] = int(complaints[-w:].sum())
                    feature_row[f'service_calls{suffix}'] = int(service_calls[-w:].sum())
                else:
                    feature_row[f'complaint_count{suffix}'] = int(complaints.sum())
                    feature_row[f'service_calls{suffix}'] = int(service_calls.sum())

            # ===== Trend and Stability Features =====
            # Payment consistency score (0-1, higher = more consistent)
            if len(payment_ratio) > 3:
                consistency = 1 - np.std(payment_ratio) / max(np.mean(payment_ratio), 0.01)
                feature_row['payment_consistency_score'] = float(np.clip(consistency, 0, 1))
            else:
                feature_row['payment_consistency_score'] = 0.5

            # Trend direction
            if len(dpd_values) >= 6:
                first_half = dpd_values[:len(dpd_values)//2].mean()
                second_half = dpd_values[len(dpd_values)//2:].mean()
                diff = second_half - first_half

                if diff < -5:
                    feature_row['trend_direction'] = TrendDirection.IMPROVING.value
                elif diff > 5:
                    feature_row['trend_direction'] = TrendDirection.DETERIORATING.value
                else:
                    feature_row['trend_direction'] = TrendDirection.STABLE.value
            else:
                feature_row['trend_direction'] = TrendDirection.STABLE.value

            # Volatility score (0-1, higher = more volatile)
            if len(dpd_values) > 3:
                dpd_volatility = np.std(dpd_values) / max(np.mean(dpd_values) + 1, 1)
            else:
                dpd_volatility = 0

            if len(utilization) > 3:
                util_volatility = np.std(utilization) / max(np.mean(utilization) + 0.01, 0.01)
            else:
                util_volatility = 0

            feature_row['volatility_score'] = float(np.clip(
                (dpd_volatility * 0.6 + util_volatility * 0.4), 0, 1
            ))

            # Recent behavior flags
            if len(dpd_values) >= 3:
                feature_row['recent_dpd_flag'] = bool(dpd_values[-3:].max() > 30)
                feature_row['recent_severe_dpd_flag'] = bool(dpd_values[-3:].max() > 90)
            else:
                feature_row['recent_dpd_flag'] = False
                feature_row['recent_severe_dpd_flag'] = False

            features.append(feature_row)

        return pd.DataFrame(features)

    def _determine_default_label(
        self,
        profile: CustomerBehaviorProfile,
        agg_features: Dict[str, Any]
    ) -> Tuple[bool, float]:
        pattern = profile.behavior_pattern.value
        base_prob = PATTERN_DEFAULT_PROB.get(pattern, 0.15)

        # Adjust based on aggregate features
        max_dpd = agg_features.get('max_dpd_12m', 0)
        trend = agg_features.get('trend_direction', 'stable')
        volatility = agg_features.get('volatility_score', 0)

        # DPD adjustment
        if max_dpd > 90:
            base_prob = max(base_prob, 0.5)
        elif max_dpd > 60:
            base_prob *= 1.3
        elif max_dpd > 30:
            base_prob *= 1.1

        # Trend adjustment
        if trend == 'deteriorating':
            base_prob *= 1.3
        elif trend == 'improving':
            base_prob *= 0.7

        # Volatility adjustment
        if volatility > 0.5:
            base_prob *= 1.2

        base_prob = np.clip(base_prob, 0.01, 0.95)

        is_default = self.rng.random() < base_prob

        return is_default, round(base_prob, 4)

    # MAIN GENERATE METHOD

    def generate(
        self,
        demographic_df: pd.DataFrame,
        financial_df: pd.DataFrame,
        credit_df: Optional[pd.DataFrame] = None,
        telecom_df: Optional[pd.DataFrame] = None,
        sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Merge all data
        merged = demographic_df[['customer_id', 'age']].merge(
            financial_df[['customer_id', 'monthly_income', 'existing_debt']],
            on='customer_id'
        )

        # Add credit data
        if credit_df is not None:
            credit_cols = ['customer_id', 'has_credit_history', 'cic_score', 'total_credit_limit']
            available_cols = [c for c in credit_cols if c in credit_df.columns]
            merged = merged.merge(credit_df[available_cols], on='customer_id', how='left')
        else:
            merged['has_credit_history'] = False
            merged['cic_score'] = 0
            merged['total_credit_limit'] = 0

        # Add telecom data
        if telecom_df is not None:
            telecom_cols = ['customer_id', 'is_vnpt_customer', 'monthly_arpu', 'contract_type_code']
            available_cols = [c for c in telecom_cols if c in telecom_df.columns]
            merged = merged.merge(telecom_df[available_cols], on='customer_id', how='left')
        else:
            merged['is_vnpt_customer'] = False
            merged['monthly_arpu'] = 0
            merged['contract_type_code'] = 'none'

        # Fill NaN
        merged['has_credit_history'] = merged['has_credit_history'].fillna(False)
        merged['cic_score'] = merged['cic_score'].fillna(0)
        merged['total_credit_limit'] = merged['total_credit_limit'].fillna(0)
        merged['is_vnpt_customer'] = merged['is_vnpt_customer'].fillna(False)
        merged['monthly_arpu'] = merged['monthly_arpu'].fillna(0)
        merged['contract_type_code'] = merged['contract_type_code'].fillna('none')

        # Sample if specified
        if sample_size is not None and sample_size < len(merged):
            merged = merged.sample(n=sample_size, random_state=self.seed)

        # Generate series for each customer
        all_credit_series = []
        all_telecom_series = []

        for _, row in merged.iterrows():
            profile = CustomerBehaviorProfile(
                customer_id=row['customer_id'],
                monthly_income=row['monthly_income'],
                existing_debt=row['existing_debt'],
                credit_limit=row['total_credit_limit'],
                cic_score=int(row['cic_score']),
                has_credit_history=bool(row['has_credit_history']),
                is_vnpt_customer=bool(row['is_vnpt_customer']),
                monthly_arpu=row['monthly_arpu'],
                contract_type=row['contract_type_code'],
            )

            # Generate credit series
            credit_series = self._generate_credit_series(profile)
            all_credit_series.append(credit_series)

            # Generate telecom series
            telecom_series = self.generate_telecom_trajectory(profile)
            all_telecom_series.append(telecom_series)

        # Combine series
        credit_series_df = pd.concat(all_credit_series, ignore_index=True)
        telecom_series_df = pd.concat(all_telecom_series, ignore_index=True)

        # Calculate aggregate features
        agg_features_df = self.calculate_rolling_features(
            credit_series_df, telecom_series_df, windows=[3, 6, 12]
        )

        # Add default labels
        default_labels = []
        default_probs = []

        for _, row in merged.iterrows():
            profile = CustomerBehaviorProfile(
                customer_id=row['customer_id'],
                monthly_income=row['monthly_income'],
                existing_debt=row['existing_debt'],
                credit_limit=row['total_credit_limit'],
                cic_score=int(row['cic_score']),
                has_credit_history=bool(row['has_credit_history']),
                is_vnpt_customer=bool(row['is_vnpt_customer']),
                monthly_arpu=row['monthly_arpu'],
                contract_type=row['contract_type_code'],
            )

            # Get aggregate features for this customer
            cust_agg = agg_features_df[
                agg_features_df['customer_id'] == row['customer_id']
            ].iloc[0].to_dict() if len(agg_features_df[
                agg_features_df['customer_id'] == row['customer_id']
            ]) > 0 else {}

            is_default, default_prob = self._determine_default_label(profile, cust_agg)
            default_labels.append(is_default)
            default_probs.append(default_prob)

        agg_features_df['is_default'] = default_labels
        agg_features_df['default_probability'] = default_probs

        # Store generated data
        self._credit_series = credit_series_df
        self._telecom_series = telecom_series_df
        self._agg_features = agg_features_df

        return credit_series_df, telecom_series_df, agg_features_df

    def get_behavior_summary(self) -> Dict[str, Any]:
        if self._agg_features is None:
            return {"error": "No data generated"}

        agg = self._agg_features

        return {
            "n_customers": len(agg),
            "default_rate": float(agg['is_default'].mean()),
            "avg_default_prob": float(agg['default_probability'].mean()),
            "trend_distribution": agg['trend_direction'].value_counts(normalize=True).to_dict(),
            "dpd_stats": {
                "avg_max_dpd_12m": float(agg['max_dpd_12m'].mean()),
                "pct_dpd_30plus": float((agg['max_dpd_12m'] > 30).mean()),
                "pct_dpd_90plus": float((agg['max_dpd_12m'] > 90).mean()),
            },
            "volatility_stats": {
                "avg_volatility": float(agg['volatility_score'].mean()),
                "pct_high_volatility": float((agg['volatility_score'] > 0.5).mean()),
            },
        }


# MODULE EXPORTS

__all__ = [
    "BehavioralSeriesGenerator",
    "BehaviorPattern",
    "DPDCategory",
    "TrendDirection",
    "CustomerBehaviorProfile",
    "DPD_RANGES",
    "PATTERN_DEFAULT_PROB",
]
