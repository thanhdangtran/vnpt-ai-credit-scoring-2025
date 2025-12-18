from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit  # Sigmoid function

from config.settings import SyntheticDataConfig
from generators.base import BaseDataGenerator


# ENUMS AND CONSTANTS

class RiskGrade(Enum):
    A = "A"  # Very Low Risk: PD < 1%
    B = "B"  # Low Risk: 1% <= PD < 3%
    C = "C"  # Medium Risk: 3% <= PD < 7%
    D = "D"  # High Risk: 7% <= PD < 15%
    E = "E"  # Very High Risk: PD >= 15%


class TrendDirection(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"


class CustomerSegment(Enum):
    PRIME = "prime"              # Good credit, stable income
    NEAR_PRIME = "near_prime"    # Moderate credit, some risk signals
    SUBPRIME = "subprime"        # Poor credit or high risk
    THIN_FILE = "thin_file"      # No/limited credit history
    NEW_TO_CREDIT = "new_to_credit"  # Young, first-time borrowers


# Risk grade PD boundaries
RISK_GRADE_BOUNDARIES: Dict[str, Tuple[float, float]] = {
    "A": (0.0, 0.01),
    "B": (0.01, 0.03),
    "C": (0.03, 0.07),
    "D": (0.07, 0.15),
    "E": (0.15, 1.0),
}

# Target default rates by segment (Vietnamese market)
TARGET_DEFAULT_RATES: Dict[str, Tuple[float, float]] = {
    "prime": (0.01, 0.02),           # 1-2%
    "near_prime": (0.03, 0.05),      # 3-5%
    "subprime": (0.08, 0.12),        # 8-12%
    "thin_file": (0.06, 0.08),       # 6-8%
    "new_to_credit": (0.04, 0.06),   # 4-6%
    "overall": (0.03, 0.05),         # 3-5%
}

# Factor weights for PD calculation
DEFAULT_FACTOR_WEIGHTS: Dict[str, float] = {
    "credit_history": 0.25,
    "financial": 0.25,
    "demographic": 0.10,
    "telecom": 0.15,
    "time_series": 0.25,
}

# Factor weights when thin-file (no credit history)
THIN_FILE_FACTOR_WEIGHTS: Dict[str, float] = {
    "credit_history": 0.05,  # Very low - no history
    "financial": 0.30,       # Higher weight
    "demographic": 0.15,     # Higher weight
    "telecom": 0.35,         # Much higher - alternative data
    "time_series": 0.15,     # Lower - limited history
}


# TIME SERIES FEATURES FOR LABELING

@dataclass
class TimeSeriesSignals:
    customer_id: str

    # DPD (Days Past Due) signals
    dpd_trend: TrendDirection = TrendDirection.STABLE
    max_dpd_3m: int = 0
    max_dpd_6m: int = 0
    max_dpd_12m: int = 0
    avg_dpd_12m: float = 0.0
    count_dpd_30plus_12m: int = 0
    count_dpd_90plus_12m: int = 0

    # Payment signals
    payment_consistency: float = 1.0  # 0-1, higher = more consistent
    recent_late_payment: bool = False
    consecutive_late_payments: int = 0

    # Balance and utilization signals
    balance_volatility: float = 0.0  # 0-1, higher = more volatile
    utilization_trend: TrendDirection = TrendDirection.STABLE
    avg_utilization_6m: float = 0.0
    max_utilization_12m: float = 0.0

    # Income signals
    income_stability: float = 1.0  # 0-1, higher = more stable
    income_trend: TrendDirection = TrendDirection.STABLE

    # Telecom signals
    telecom_payment_trend: TrendDirection = TrendDirection.STABLE
    telecom_late_rate: float = 0.0  # Rate of late telecom payments
    avg_days_to_payment: float = 0.0

    # Aggregate risk score from time series
    ts_risk_score: float = 0.0  # 0-1, higher = riskier


@dataclass
class LabelingConfig:
    # Target rates
    target_overall_default_rate: float = 0.04  # 4%
    target_thin_file_rate: float = 0.07        # 7%
    target_prime_rate: float = 0.015           # 1.5%
    target_deteriorating_rate: float = 0.10    # 10%

    # Observation and outcome windows
    observation_months: int = 12  # Feature observation window
    outcome_months: int = 12      # Default outcome window

    # Factor weights
    factor_weights: Dict[str, float] = field(
        default_factory=lambda: DEFAULT_FACTOR_WEIGHTS.copy()
    )
    thin_file_weights: Dict[str, float] = field(
        default_factory=lambda: THIN_FILE_FACTOR_WEIGHTS.copy()
    )

    # PD adjustment factors
    deteriorating_multiplier: float = 1.5
    improving_multiplier: float = 0.7
    high_volatility_multiplier: float = 1.3
    recent_late_multiplier: float = 1.4
    missing_income_multiplier: float = 1.2

    # Noise for non-determinism
    noise_std: float = 0.02  # Standard deviation of noise

    # Calibration
    calibrate_to_target: bool = True


# LABEL GENERATOR CLASS

class LabelGenerator(BaseDataGenerator):
    def __init__(
        self,
        config: SyntheticDataConfig,
        seed: Optional[int] = None,
        labeling_config: Optional[LabelingConfig] = None
    ) -> None:
        super().__init__(config, seed)
        self.labeling_config = labeling_config or LabelingConfig()

        # Storage for intermediate calculations
        self._factor_scores: Dict[str, pd.DataFrame] = {}
        self._time_series_signals: Dict[str, TimeSeriesSignals] = {}
        self._segment_assignments: Dict[str, CustomerSegment] = {}

    # FACTOR SCORE CALCULATORS

    def _calculate_credit_history_score(
        self,
        row: pd.Series
    ) -> float:
        score = 0.0

        # CIC score (if available)
        cic_score = row.get('cic_score', np.nan)
        if pd.notna(cic_score) and cic_score > 0:
            # Higher CIC = lower risk
            # CIC range: 300-900
            cic_normalized = (900 - cic_score) / 600
            score += 0.35 * np.clip(cic_normalized, 0, 1)
        else:
            # No CIC score - assign moderate risk
            score += 0.35 * 0.5

        # NHNN loan group
        loan_group = row.get('nhnn_loan_group', 1)
        if pd.notna(loan_group):
            # Group 1=best, 5=worst
            group_score = (loan_group - 1) / 4
            score += 0.25 * group_score

        # Max DPD ever
        max_dpd = row.get('max_dpd_ever', 0)
        if pd.notna(max_dpd):
            if max_dpd == 0:
                score += 0.15 * 0.0
            elif max_dpd <= 30:
                score += 0.15 * 0.2
            elif max_dpd <= 60:
                score += 0.15 * 0.4
            elif max_dpd <= 90:
                score += 0.15 * 0.6
            else:
                score += 0.15 * 1.0

        # Number of active loans
        num_loans = row.get('num_active_loans', 0)
        if pd.notna(num_loans):
            loan_score = min(num_loans / 5, 1.0)  # Cap at 5 loans
            score += 0.10 * loan_score

        # Credit history length (longer = better)
        history_months = row.get('credit_history_months', 0)
        if pd.notna(history_months) and history_months > 0:
            # More history = lower risk
            history_score = max(0, 1 - history_months / 120)  # 10 years = 0 risk
            score += 0.10 * history_score
        else:
            score += 0.10 * 0.7  # No history = higher risk

        # Credit utilization
        utilization = row.get('credit_utilization', 0)
        if pd.notna(utilization):
            if utilization > 0.9:
                score += 0.05 * 1.0
            elif utilization > 0.7:
                score += 0.05 * 0.6
            elif utilization > 0.5:
                score += 0.05 * 0.3
            else:
                score += 0.05 * 0.1

        return np.clip(score, 0, 1)

    def _calculate_financial_score(
        self,
        row: pd.Series
    ) -> float:
        score = 0.0

        # DTI ratio
        dti = row.get('dti_ratio', 0)
        if pd.notna(dti):
            if dti > 0.6:
                score += 0.30 * 1.0
            elif dti > 0.45:
                score += 0.30 * 0.7
            elif dti > 0.30:
                score += 0.30 * 0.4
            else:
                score += 0.30 * 0.1
        else:
            # Missing DTI - moderate risk
            score += 0.30 * 0.5

        # Monthly income level
        income = row.get('monthly_income', 0)
        if pd.notna(income) and income > 0:
            if income >= 30_000_000:
                score += 0.25 * 0.1
            elif income >= 15_000_000:
                score += 0.25 * 0.25
            elif income >= 8_000_000:
                score += 0.25 * 0.4
            else:
                score += 0.25 * 0.7
        else:
            # Missing income - higher risk (MNAR signal)
            score += 0.25 * 0.8

        # Employment stability
        employment_type = row.get('employment_type_code', '')
        if pd.notna(employment_type):
            stable_employment = ['chinh_thuc', 'cong_chuc', 'quan_doi']
            unstable_employment = ['that_nghiep', 'thoi_vu', 'lao_dong_pho_thong']

            if employment_type in stable_employment:
                score += 0.20 * 0.1
            elif employment_type in unstable_employment:
                score += 0.20 * 0.8
            else:
                score += 0.20 * 0.4
        else:
            score += 0.20 * 0.5

        # Job tenure
        tenure = row.get('job_tenure_months', 0)
        if pd.notna(tenure):
            if tenure >= 60:  # 5+ years
                score += 0.15 * 0.1
            elif tenure >= 24:  # 2+ years
                score += 0.15 * 0.3
            elif tenure >= 12:  # 1+ year
                score += 0.15 * 0.5
            else:
                score += 0.15 * 0.8
        else:
            score += 0.15 * 0.6

        # Savings ratio
        savings_ratio = row.get('savings_ratio', 0)
        if pd.notna(savings_ratio):
            if savings_ratio >= 0.3:
                score += 0.10 * 0.1
            elif savings_ratio >= 0.15:
                score += 0.10 * 0.3
            elif savings_ratio >= 0.05:
                score += 0.10 * 0.5
            else:
                score += 0.10 * 0.8
        else:
            score += 0.10 * 0.5

        return np.clip(score, 0, 1)

    def _calculate_demographic_score(
        self,
        row: pd.Series
    ) -> float:
        score = 0.0

        # Age factor
        age = row.get('age', 30)
        if pd.notna(age):
            if 30 <= age <= 50:
                score += 0.30 * 0.2  # Prime age
            elif 25 <= age < 30 or 50 < age <= 60:
                score += 0.30 * 0.4
            elif age < 25:
                score += 0.30 * 0.7  # Young - higher risk
            else:
                score += 0.30 * 0.5  # Elderly

        # Education
        education = row.get('education_code', '')
        if pd.notna(education):
            high_edu = ['dai_hoc', 'thac_si', 'tien_si']
            if education in high_edu:
                score += 0.25 * 0.2
            elif education == 'cao_dang':
                score += 0.25 * 0.35
            elif education == 'trung_cap':
                score += 0.25 * 0.5
            else:
                score += 0.25 * 0.7
        else:
            score += 0.25 * 0.5

        # Marital status
        marital = row.get('marital_status_code', '')
        if pd.notna(marital):
            if marital == 'da_ket_hon':
                score += 0.20 * 0.3  # Married - more stable
            elif marital == 'doc_than':
                score += 0.20 * 0.5
            else:
                score += 0.20 * 0.6
        else:
            score += 0.20 * 0.5

        # Geographic (urban/rural)
        is_urban = row.get('is_urban', True)
        if pd.notna(is_urban):
            if is_urban:
                score += 0.15 * 0.35  # Urban - better access
            else:
                score += 0.15 * 0.55  # Rural - more risk
        else:
            score += 0.15 * 0.45

        # Number of dependents (if available)
        dependents = row.get('num_dependents', None)
        if dependents is not None and pd.notna(dependents):
            dep_score = min(dependents / 5, 1.0)  # More dependents = more burden
            score += 0.10 * dep_score
        else:
            # No dependents info - add small neutral contribution
            score += 0.10 * 0.4

        return np.clip(score, 0, 1)

    def _calculate_telecom_score(
        self,
        row: pd.Series
    ) -> float:
        score = 0.0

        # Check if VNPT customer
        is_vnpt = row.get('is_vnpt_customer', False)
        if not is_vnpt or pd.isna(is_vnpt):
            # No telecom data - return neutral score
            return 0.5

        # Telecom credit score (if available)
        telecom_score = row.get('telecom_credit_score', np.nan)
        if pd.notna(telecom_score):
            # Higher telecom score = lower risk
            # Range: 0-100
            score += 0.35 * (1 - telecom_score / 100)
        else:
            score += 0.35 * 0.5

        # Payment rate
        payment_rate = row.get('payment_rate', 1.0)
        if pd.notna(payment_rate):
            if payment_rate >= 0.95:
                score += 0.25 * 0.1
            elif payment_rate >= 0.85:
                score += 0.25 * 0.3
            elif payment_rate >= 0.70:
                score += 0.25 * 0.6
            else:
                score += 0.25 * 0.9
        else:
            score += 0.25 * 0.5

        # Contract type
        contract_type = row.get('contract_type_code', '')
        if pd.notna(contract_type):
            if contract_type == 'tra_sau':
                score += 0.15 * 0.3  # Postpaid - better credit
            else:
                score += 0.15 * 0.5  # Prepaid - less info
        else:
            score += 0.15 * 0.5

        # ARPU level
        arpu = row.get('monthly_arpu', 0)
        if pd.notna(arpu) and arpu > 0:
            if arpu >= 500_000:
                score += 0.15 * 0.2  # High spender
            elif arpu >= 200_000:
                score += 0.15 * 0.35
            elif arpu >= 100_000:
                score += 0.15 * 0.5
            else:
                score += 0.15 * 0.7  # Low spender
        else:
            score += 0.15 * 0.5

        # Tenure with telecom
        tenure = row.get('tenure_months', 0)
        if pd.notna(tenure):
            if tenure >= 36:  # 3+ years
                score += 0.10 * 0.2
            elif tenure >= 12:
                score += 0.10 * 0.4
            else:
                score += 0.10 * 0.7
        else:
            score += 0.10 * 0.5

        return np.clip(score, 0, 1)

    def _calculate_time_series_score(
        self,
        signals: TimeSeriesSignals
    ) -> float:
        score = 0.0

        # DPD trend (most important)
        if signals.dpd_trend == TrendDirection.WORSENING:
            score += 0.25 * 1.0
        elif signals.dpd_trend == TrendDirection.IMPROVING:
            score += 0.25 * 0.2
        else:
            score += 0.25 * 0.4

        # Max DPD in 12 months
        if signals.max_dpd_12m > 90:
            score += 0.20 * 1.0
        elif signals.max_dpd_12m > 60:
            score += 0.20 * 0.7
        elif signals.max_dpd_12m > 30:
            score += 0.20 * 0.4
        else:
            score += 0.20 * 0.1

        # Payment consistency
        score += 0.15 * (1 - signals.payment_consistency)

        # Recent late payment flag
        if signals.recent_late_payment:
            score += 0.10 * 0.9
        else:
            score += 0.10 * 0.1

        # Balance volatility
        score += 0.10 * signals.balance_volatility

        # Utilization trend
        if signals.utilization_trend == TrendDirection.WORSENING:
            score += 0.10 * 0.8
        elif signals.utilization_trend == TrendDirection.IMPROVING:
            score += 0.10 * 0.2
        else:
            score += 0.10 * 0.4

        # Telecom payment trend
        if signals.telecom_payment_trend == TrendDirection.WORSENING:
            score += 0.10 * 0.7
        elif signals.telecom_payment_trend == TrendDirection.IMPROVING:
            score += 0.10 * 0.2
        else:
            score += 0.10 * 0.35

        return np.clip(score, 0, 1)

    # TIME SERIES SIGNAL EXTRACTION

    def extract_time_series_signals(
        self,
        customer_id: str,
        credit_series: Optional[pd.DataFrame] = None,
        telecom_series: Optional[pd.DataFrame] = None,
        transaction_series: Optional[pd.DataFrame] = None
    ) -> TimeSeriesSignals:
        signals = TimeSeriesSignals(customer_id=customer_id)

        # Extract from credit series
        if credit_series is not None and len(credit_series) > 0:
            cust_credit = credit_series[
                credit_series['customer_id'] == customer_id
            ].sort_values('month_id')

            if len(cust_credit) > 0:
                dpd_values = cust_credit['dpd_days'].values

                # DPD statistics
                signals.max_dpd_12m = int(dpd_values[-12:].max()) if len(dpd_values) >= 12 else int(dpd_values.max())
                signals.max_dpd_6m = int(dpd_values[-6:].max()) if len(dpd_values) >= 6 else int(dpd_values.max())
                signals.max_dpd_3m = int(dpd_values[-3:].max()) if len(dpd_values) >= 3 else int(dpd_values.max())
                signals.avg_dpd_12m = float(dpd_values[-12:].mean()) if len(dpd_values) >= 12 else float(dpd_values.mean())
                signals.count_dpd_30plus_12m = int((dpd_values[-12:] > 30).sum()) if len(dpd_values) >= 12 else int((dpd_values > 30).sum())
                signals.count_dpd_90plus_12m = int((dpd_values[-12:] > 90).sum()) if len(dpd_values) >= 12 else int((dpd_values > 90).sum())

                # DPD trend
                if len(dpd_values) >= 6:
                    first_half = dpd_values[:len(dpd_values)//2].mean()
                    second_half = dpd_values[len(dpd_values)//2:].mean()
                    diff = second_half - first_half
                    if diff > 10:
                        signals.dpd_trend = TrendDirection.WORSENING
                    elif diff < -10:
                        signals.dpd_trend = TrendDirection.IMPROVING
                    else:
                        signals.dpd_trend = TrendDirection.STABLE

                # Payment consistency
                if 'payment_ratio' in cust_credit.columns:
                    payment_ratios = cust_credit['payment_ratio'].values
                    if len(payment_ratios) > 3:
                        consistency = 1 - np.std(payment_ratios) / max(np.mean(payment_ratios), 0.01)
                        signals.payment_consistency = float(np.clip(consistency, 0, 1))
                        signals.recent_late_payment = bool(payment_ratios[-3:].min() < 0.9)

                # Utilization
                if 'credit_utilization' in cust_credit.columns:
                    utilization = cust_credit['credit_utilization'].values
                    signals.avg_utilization_6m = float(utilization[-6:].mean()) if len(utilization) >= 6 else float(utilization.mean())
                    signals.max_utilization_12m = float(utilization[-12:].max()) if len(utilization) >= 12 else float(utilization.max())

                    if len(utilization) >= 6:
                        first_half = utilization[:len(utilization)//2].mean()
                        second_half = utilization[len(utilization)//2:].mean()
                        diff = second_half - first_half
                        if diff > 0.1:
                            signals.utilization_trend = TrendDirection.WORSENING
                        elif diff < -0.1:
                            signals.utilization_trend = TrendDirection.IMPROVING

        # Extract from telecom series
        if telecom_series is not None and len(telecom_series) > 0:
            cust_telecom = telecom_series[
                telecom_series['customer_id'] == customer_id
            ].sort_values('month_id')

            if len(cust_telecom) > 0 and 'days_to_payment' in cust_telecom.columns:
                days_late = cust_telecom['days_to_payment'].values
                signals.avg_days_to_payment = float(days_late.mean())
                signals.telecom_late_rate = float((days_late > 30).mean())

                if len(days_late) >= 6:
                    first_half = days_late[:len(days_late)//2].mean()
                    second_half = days_late[len(days_late)//2:].mean()
                    diff = second_half - first_half
                    if diff > 5:
                        signals.telecom_payment_trend = TrendDirection.WORSENING
                    elif diff < -5:
                        signals.telecom_payment_trend = TrendDirection.IMPROVING

        # Extract from transaction series
        if transaction_series is not None and len(transaction_series) > 0:
            cust_trans = transaction_series[
                transaction_series['customer_id'] == customer_id
            ].sort_values('month_id')

            if len(cust_trans) > 0 and 'ending_balance' in cust_trans.columns:
                balances = cust_trans['ending_balance'].values
                if len(balances) > 3:
                    volatility = np.std(balances) / max(np.mean(balances), 1)
                    signals.balance_volatility = float(np.clip(volatility, 0, 1))

                # Income stability (from deposits)
                if 'total_deposits' in cust_trans.columns:
                    deposits = cust_trans['total_deposits'].values
                    if len(deposits) > 3 and deposits.mean() > 0:
                        stability = 1 - np.std(deposits) / max(np.mean(deposits), 1)
                        signals.income_stability = float(np.clip(stability, 0, 1))

        # Calculate aggregate time series risk score
        signals.ts_risk_score = self._calculate_time_series_score(signals)

        return signals

    # SEGMENT ASSIGNMENT

    def _assign_customer_segment(
        self,
        row: pd.Series,
        credit_score: float,
        financial_score: float
    ) -> CustomerSegment:
        has_credit = row.get('has_credit_history', False)
        cic_score = row.get('cic_score', 0)
        age = row.get('age', 30)

        # Thin file check
        if not has_credit or pd.isna(has_credit):
            if age < 25:
                return CustomerSegment.NEW_TO_CREDIT
            return CustomerSegment.THIN_FILE

        # Based on credit and financial scores
        combined_score = (credit_score + financial_score) / 2

        if combined_score < 0.25 and cic_score >= 700:
            return CustomerSegment.PRIME
        elif combined_score < 0.45:
            return CustomerSegment.NEAR_PRIME
        else:
            return CustomerSegment.SUBPRIME

    # PROBABILITY OF DEFAULT CALCULATION

    def calculate_pd(
        self,
        row: pd.Series,
        ts_signals: Optional[TimeSeriesSignals] = None
    ) -> float:
        config = self.labeling_config

        # Calculate factor scores
        credit_score = self._calculate_credit_history_score(row)
        financial_score = self._calculate_financial_score(row)
        demographic_score = self._calculate_demographic_score(row)
        telecom_score = self._calculate_telecom_score(row)

        # Time series score
        if ts_signals is not None:
            ts_score = ts_signals.ts_risk_score
        else:
            ts_score = 0.5  # Neutral if no time series

        # Determine if thin-file
        has_credit = row.get('has_credit_history', False)
        is_thin_file = not has_credit or pd.isna(has_credit)

        # Select weights
        if is_thin_file:
            weights = config.thin_file_weights
        else:
            weights = config.factor_weights

        # Weighted combination of risk scores (0-1 range)
        composite_risk = (
            weights['credit_history'] * credit_score +
            weights['financial'] * financial_score +
            weights['demographic'] * demographic_score +
            weights['telecom'] * telecom_score +
            weights['time_series'] * ts_score
        )

        # Transform risk score to realistic PD using logistic function
        # Maps 0-1 risk score to 0.005-0.25 PD range (0.5% to 25%)
        # At risk_score=0.3 -> PD ~2%, risk_score=0.5 -> PD ~5%, risk_score=0.7 -> PD ~12%
        base_pd = 0.005 + 0.245 * expit(6 * (composite_risk - 0.5))

        # Apply adjustments based on time series signals
        if ts_signals is not None:
            # Deteriorating trend
            if ts_signals.dpd_trend == TrendDirection.WORSENING:
                base_pd *= config.deteriorating_multiplier
            elif ts_signals.dpd_trend == TrendDirection.IMPROVING:
                base_pd *= config.improving_multiplier

            # High volatility
            if ts_signals.balance_volatility > 0.5:
                base_pd *= config.high_volatility_multiplier

            # Recent late payments
            if ts_signals.recent_late_payment:
                base_pd *= config.recent_late_multiplier

        # MNAR adjustment - missing income as risk signal
        if pd.isna(row.get('monthly_income')):
            base_pd *= config.missing_income_multiplier

        # Add noise for non-determinism
        noise = self.rng.normal(0, config.noise_std)
        pd_with_noise = base_pd + noise

        # Clip to valid range
        return float(np.clip(pd_with_noise, 0.001, 0.95))

    # LABEL GENERATION METHODS

    def generate_default_label(
        self,
        pd_score: float
    ) -> int:
        return int(self.rng.random() < pd_score)

    def generate_pd_score(
        self,
        row: pd.Series,
        ts_signals: Optional[TimeSeriesSignals] = None
    ) -> float:
        return self.calculate_pd(row, ts_signals)

    def generate_risk_grade(
        self,
        pd_score: float
    ) -> str:
        for grade, (low, high) in RISK_GRADE_BOUNDARIES.items():
            if low <= pd_score < high:
                return grade
        return "E"  # Default to highest risk

    # DEFAULT RATE CALIBRATION

    def _calibrate_default_rates(
        self,
        df: pd.DataFrame,
        target_rate: float
    ) -> pd.DataFrame:
        current_rate = df['is_default'].mean()

        if abs(current_rate - target_rate) < 0.005:
            return df  # Close enough

        df = df.copy()

        if current_rate > target_rate:
            # Too many defaults - flip some to non-default
            defaults = df[df['is_default'] == 1]
            n_flip = int((current_rate - target_rate) * len(df))

            # Flip those with lowest PD among defaults
            flip_indices = defaults.nsmallest(n_flip, 'pd_score').index
            df.loc[flip_indices, 'is_default'] = 0

        else:
            # Too few defaults - flip some to default
            non_defaults = df[df['is_default'] == 0]
            n_flip = int((target_rate - current_rate) * len(df))

            # Flip those with highest PD among non-defaults
            flip_indices = non_defaults.nlargest(n_flip, 'pd_score').index
            df.loc[flip_indices, 'is_default'] = 1

        return df

    # IMBALANCED SAMPLING

    def generate_imbalanced_sample(
        self,
        df: pd.DataFrame,
        target_default_ratio: float = 0.5,
        sample_type: str = 'oversample'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        defaults = df[df['is_default'] == 1]
        non_defaults = df[df['is_default'] == 0]

        n_defaults = len(defaults)
        n_non_defaults = len(non_defaults)

        if sample_type == 'oversample':
            # Oversample defaults to achieve target ratio
            target_n_defaults = int(n_non_defaults * target_default_ratio / (1 - target_default_ratio))

            if target_n_defaults > n_defaults:
                # Sample with replacement
                oversampled_defaults = defaults.sample(
                    n=target_n_defaults,
                    replace=True,
                    random_state=self.seed
                )
                training_df = pd.concat([non_defaults, oversampled_defaults], ignore_index=True)
            else:
                training_df = df.copy()

        else:  # undersample
            # Undersample non-defaults
            target_n_non_defaults = int(n_defaults * (1 - target_default_ratio) / target_default_ratio)

            if target_n_non_defaults < n_non_defaults:
                undersampled_non_defaults = non_defaults.sample(
                    n=target_n_non_defaults,
                    random_state=self.seed
                )
                training_df = pd.concat([defaults, undersampled_non_defaults], ignore_index=True)
            else:
                training_df = df.copy()

        # Shuffle
        training_df = training_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # Validation keeps original distribution
        validation_df = df.copy()

        return training_df, validation_df

    # MAIN GENERATE METHOD

    def generate(
        self,
        demographic_df: pd.DataFrame,
        financial_df: pd.DataFrame,
        credit_df: Optional[pd.DataFrame] = None,
        telecom_df: Optional[pd.DataFrame] = None,
        credit_series: Optional[pd.DataFrame] = None,
        telecom_series: Optional[pd.DataFrame] = None,
        transaction_series: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        # Merge all static data - use available columns
        demo_cols = ['customer_id', 'age']
        optional_demo_cols = ['education_level_code', 'marital_status_code', 'urban_rural', 'address_stability_years']
        for col in optional_demo_cols:
            if col in demographic_df.columns:
                demo_cols.append(col)
        merged = demographic_df[demo_cols].copy()

        # Rename columns for consistency
        if 'education_level_code' in merged.columns:
            merged['education_code'] = merged['education_level_code']
        if 'urban_rural' in merged.columns:
            merged['is_urban'] = merged['urban_rural'].apply(lambda x: x == 'thanh_thi' if pd.notna(x) else True)

        # Merge financial
        fin_cols = ['customer_id', 'monthly_income', 'employment_type_code',
                    'job_tenure_months', 'existing_debt', 'dti_ratio', 'savings_ratio']
        available_fin = [c for c in fin_cols if c in financial_df.columns]
        merged = merged.merge(financial_df[available_fin], on='customer_id', how='left')

        # Merge credit
        if credit_df is not None:
            credit_cols = ['customer_id', 'has_credit_history', 'cic_score',
                          'nhnn_loan_group', 'max_dpd_ever', 'num_active_loans',
                          'credit_history_months', 'credit_utilization', 'total_credit_limit']
            available_credit = [c for c in credit_cols if c in credit_df.columns]
            merged = merged.merge(credit_df[available_credit], on='customer_id', how='left')
        else:
            merged['has_credit_history'] = False

        # Merge telecom
        if telecom_df is not None:
            telecom_cols = ['customer_id', 'is_vnpt_customer', 'telecom_credit_score',
                           'payment_rate', 'contract_type_code', 'monthly_arpu', 'tenure_months']
            available_telecom = [c for c in telecom_cols if c in telecom_df.columns]
            merged = merged.merge(telecom_df[available_telecom], on='customer_id', how='left')

        # Fill NaN for boolean columns
        merged['has_credit_history'] = merged['has_credit_history'].fillna(False)
        merged['is_vnpt_customer'] = merged.get('is_vnpt_customer', pd.Series([False] * len(merged))).fillna(False)

        # Generate labels
        results = []

        for idx, row in merged.iterrows():
            customer_id = row['customer_id']

            # Extract time series signals
            ts_signals = self.extract_time_series_signals(
                customer_id,
                credit_series,
                telecom_series,
                transaction_series
            )

            # Store signals
            self._time_series_signals[customer_id] = ts_signals

            # Calculate PD
            pd_score = self.calculate_pd(row, ts_signals)

            # Generate default label
            is_default = self.generate_default_label(pd_score)

            # Generate risk grade
            risk_grade = self.generate_risk_grade(pd_score)

            # Assign segment
            credit_score = self._calculate_credit_history_score(row)
            financial_score = self._calculate_financial_score(row)
            segment = self._assign_customer_segment(row, credit_score, financial_score)

            results.append({
                'customer_id': customer_id,
                'pd_score': round(pd_score, 6),
                'is_default': is_default,
                'risk_grade': risk_grade,
                'segment': segment.value,
                'credit_factor_score': round(credit_score, 4),
                'financial_factor_score': round(financial_score, 4),
                'demographic_factor_score': round(self._calculate_demographic_score(row), 4),
                'telecom_factor_score': round(self._calculate_telecom_score(row), 4),
                'time_series_factor_score': round(ts_signals.ts_risk_score, 4),
                'dpd_trend': ts_signals.dpd_trend.value,
                'max_dpd_12m': ts_signals.max_dpd_12m,
                'payment_consistency': round(ts_signals.payment_consistency, 4),
            })

        labels_df = pd.DataFrame(results)

        # Calibrate if needed
        if self.labeling_config.calibrate_to_target:
            labels_df = self._calibrate_default_rates(
                labels_df,
                self.labeling_config.target_overall_default_rate
            )

        return labels_df

    def get_label_statistics(
        self,
        labels_df: pd.DataFrame
    ) -> Dict[str, Any]:
        stats = {
            'total_customers': len(labels_df),
            'overall_default_rate': float(labels_df['is_default'].mean()),
            'avg_pd_score': float(labels_df['pd_score'].mean()),
            'pd_std': float(labels_df['pd_score'].std()),
        }

        # Default rate by segment
        stats['default_rate_by_segment'] = {}
        for segment in labels_df['segment'].unique():
            seg_df = labels_df[labels_df['segment'] == segment]
            stats['default_rate_by_segment'][segment] = {
                'count': len(seg_df),
                'default_rate': float(seg_df['is_default'].mean()),
                'avg_pd': float(seg_df['pd_score'].mean()),
            }

        # Default rate by risk grade
        stats['default_rate_by_grade'] = {}
        for grade in ['A', 'B', 'C', 'D', 'E']:
            grade_df = labels_df[labels_df['risk_grade'] == grade]
            if len(grade_df) > 0:
                stats['default_rate_by_grade'][grade] = {
                    'count': len(grade_df),
                    'pct': float(len(grade_df) / len(labels_df)),
                    'default_rate': float(grade_df['is_default'].mean()),
                }

        # Default rate by trend
        stats['default_rate_by_trend'] = {}
        for trend in labels_df['dpd_trend'].unique():
            trend_df = labels_df[labels_df['dpd_trend'] == trend]
            stats['default_rate_by_trend'][trend] = {
                'count': len(trend_df),
                'default_rate': float(trend_df['is_default'].mean()),
            }

        return stats


# MODULE EXPORTS

__all__ = [
    # Enums
    "RiskGrade",
    "TrendDirection",
    "CustomerSegment",
    # Dataclasses
    "TimeSeriesSignals",
    "LabelingConfig",
    # Constants
    "RISK_GRADE_BOUNDARIES",
    "TARGET_DEFAULT_RATES",
    "DEFAULT_FACTOR_WEIGHTS",
    "THIN_FILE_FACTOR_WEIGHTS",
    # Generator
    "LabelGenerator",
]
