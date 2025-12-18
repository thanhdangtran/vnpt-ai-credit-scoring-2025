from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '/home/thanhdang/Desktop/vnpt-ai')

from config.settings import SyntheticDataConfig
from generators.base import (
    BaseDataGenerator,
    CorrelationMixin,
    weighted_random_choice,
)
from generators.financial import truncated_normal, truncated_lognormal


# ENUMS AND CONSTANTS

class ContractType(Enum):
    TRA_TRUOC = "tra_truoc"      # Prepaid
    TRA_SAU = "tra_sau"          # Postpaid


class ServiceBundle(Enum):
    MOBILE_ONLY = "mobile_only"
    FIBER_ONLY = "fiber_only"
    TV_ONLY = "tv_only"
    MOBILE_FIBER = "mobile_fiber"
    FIBER_TV = "fiber_tv"
    MOBILE_TV = "mobile_tv"
    FULL_COMBO = "full_combo"  # Mobile + Fiber + TV


class PaymentMethod(Enum):
    TIEN_MAT = "tien_mat"                    # Cash
    BANK_TRANSFER = "bank_transfer"          # Bank transfer
    VI_DIEN_TU = "vi_dien_tu"               # E-wallet (MoMo, ZaloPay, etc.)
    AUTO_DEBIT = "auto_debit"               # Auto debit from bank
    VNPT_PAY = "vnpt_pay"                   # VNPT Pay app
    CREDIT_CARD = "credit_card"             # Credit card


class LoyaltyTier(Enum):
    STANDARD = "standard"        # Basic
    SILVER = "silver"           # 1-2 years
    GOLD = "gold"               # 2-5 years
    PLATINUM = "platinum"       # 5-10 years
    DIAMOND = "diamond"         # 10+ years


class UsagePattern(Enum):
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    POWER_USER = "power_user"


# Labels in Vietnamese
CONTRACT_LABELS: Dict[str, str] = {
    "tra_truoc": "Trả trước",
    "tra_sau": "Trả sau",
}

SERVICE_LABELS: Dict[str, str] = {
    "mobile_only": "Di động",
    "fiber_only": "Internet cáp quang",
    "tv_only": "Truyền hình",
    "mobile_fiber": "Di động + Internet",
    "fiber_tv": "Internet + Truyền hình",
    "mobile_tv": "Di động + Truyền hình",
    "full_combo": "Combo đầy đủ",
}

PAYMENT_LABELS: Dict[str, str] = {
    "tien_mat": "Tiền mặt",
    "bank_transfer": "Chuyển khoản",
    "vi_dien_tu": "Ví điện tử",
    "auto_debit": "Trích nợ tự động",
    "vnpt_pay": "VNPT Pay",
    "credit_card": "Thẻ tín dụng",
}

LOYALTY_LABELS: Dict[str, str] = {
    "standard": "Tiêu chuẩn",
    "silver": "Bạc",
    "gold": "Vàng",
    "platinum": "Bạch kim",
    "diamond": "Kim cương",
}


# ARPU AND USAGE CONFIGURATIONS

@dataclass
class ARPUConfig:
    min_arpu: int
    max_arpu: int
    mean_arpu: int


# ARPU ranges by service bundle (VND/month)
ARPU_BY_SERVICE: Dict[str, ARPUConfig] = {
    "mobile_only": ARPUConfig(50_000, 500_000, 150_000),
    "fiber_only": ARPUConfig(150_000, 500_000, 250_000),
    "tv_only": ARPUConfig(80_000, 200_000, 120_000),
    "mobile_fiber": ARPUConfig(200_000, 800_000, 400_000),
    "fiber_tv": ARPUConfig(230_000, 600_000, 350_000),
    "mobile_tv": ARPUConfig(130_000, 600_000, 280_000),
    "full_combo": ARPUConfig(300_000, 1_500_000, 600_000),
}

# Data usage ranges by pattern (GB/month)
DATA_USAGE_BY_PATTERN: Dict[str, Tuple[float, float, float]] = {
    # (min, max, mean)
    "light": (0.5, 5.0, 2.0),
    "moderate": (5.0, 20.0, 12.0),
    "heavy": (20.0, 50.0, 35.0),
    "power_user": (50.0, 200.0, 80.0),
}

# Call minutes by pattern (minutes/month)
CALL_MINUTES_BY_PATTERN: Dict[str, Tuple[int, int, int]] = {
    # (min, max, mean)
    "light": (10, 100, 50),
    "moderate": (100, 500, 250),
    "heavy": (500, 1500, 800),
    "power_user": (1500, 5000, 2500),
}


# ALTERNATIVE CREDIT SCORE CALCULATION

@dataclass
class TelecomCreditSignal:
    payment_reliability_score: float  # 0-100
    usage_stability_score: float      # 0-100
    loyalty_score: float              # 0-100
    digital_engagement_score: float   # 0-100
    overall_telecom_score: float      # 0-100 (weighted average)


# VNPT BEHAVIOR GENERATOR

class VNPTBehaviorGenerator(BaseDataGenerator, CorrelationMixin):
    def __init__(
        self,
        config: SyntheticDataConfig,
        seed: Optional[int] = None
    ) -> None:
        super().__init__(config, seed)
        self._set_default_schema()

    def _set_default_schema(self) -> None:
        self.set_schema({
            'customer_id': str,
            'is_vnpt_customer': bool,
            'vnpt_customer_tenure_months': int,
            'contract_type': str,
            'contract_type_code': str,
            'service_bundle': str,
            'service_bundle_code': str,
            'monthly_arpu': float,
            'payment_method': str,
            'payment_method_code': str,
            'on_time_payment_rate': float,
            'num_late_payments_telecom': int,
            'avg_days_late': float,
            'data_usage_gb': float,
            'call_minutes': int,
            'sms_count': int,
            'service_complaints': int,
            'loyalty_program_tier': str,
            'loyalty_program_tier_code': str,
            'num_service_upgrades': int,
            'num_service_downgrades': int,
            'churn_risk_score': float,
            'telecom_credit_score': float,
        })

    def _get_age_group(self, age: int) -> str:
        if age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        else:
            return "55-65"

    def _generate_is_vnpt_customer(
        self,
        n: int,
        province_codes: np.ndarray
    ) -> np.ndarray:
        is_customer = np.zeros(n, dtype=bool)

        # VNPT market share varies by region
        # Stronger in Northern provinces and state-affiliated areas
        strong_vnpt_provinces = {
            "HN", "HP", "QN", "BN", "HD", "TB", "ND", "NB",  # North
            "TH", "NA", "HT", "QB", "QT", "TTH",  # North Central
        }

        for i in range(n):
            province = province_codes[i]
            if province in strong_vnpt_provinces:
                prob = 0.85  # Higher VNPT penetration
            elif province in {"HCM", "DN", "CT"}:
                prob = 0.70  # Major cities, competitive market
            else:
                prob = 0.75  # Other provinces

            is_customer[i] = self.rng.random() < prob

        return is_customer

    def _generate_customer_tenure(
        self,
        is_customer: np.ndarray,
        ages: np.ndarray
    ) -> np.ndarray:
        n = len(is_customer)
        tenure = np.zeros(n, dtype=int)

        for i in range(n):
            if not is_customer[i]:
                continue

            age = ages[i]
            max_tenure = max(1, (age - 16) * 12)  # Could have phone since 16

            if age < 25:
                mean_tenure = min(max_tenure * 0.4, 36)
            elif age < 35:
                mean_tenure = min(max_tenure * 0.3, 60)
            elif age < 50:
                mean_tenure = min(max_tenure * 0.4, 120)
            else:
                mean_tenure = min(max_tenure * 0.5, 180)

            mean_tenure = max(mean_tenure, 3)
            std_tenure = max(1, mean_tenure * 0.5)

            t = truncated_normal(
                self.rng, mean_tenure, std_tenure,
                1, max_tenure, 1
            )[0]
            tenure[i] = int(max(1, t))

        return tenure

    def _generate_contract_type(
        self,
        is_customer: np.ndarray,
        monthly_income: np.ndarray,
        ages: np.ndarray,
        employment_codes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(is_customer)
        codes = np.empty(n, dtype=object)
        labels = np.empty(n, dtype=object)

        for i in range(n):
            if not is_customer[i]:
                codes[i] = None
                labels[i] = None
                continue

            income = monthly_income[i]
            age = ages[i]
            emp_type = employment_codes[i]

            # Base postpaid probability
            postpaid_prob = 0.45

            # Income adjustment
            if income > 20_000_000:
                postpaid_prob += 0.25
            elif income > 10_000_000:
                postpaid_prob += 0.15
            elif income < 5_000_000:
                postpaid_prob -= 0.20

            # Employment adjustment
            if emp_type in ["cong_chuc", "nhan_vien"]:
                postpaid_prob += 0.15
            elif emp_type in ["that_nghiep"]:
                postpaid_prob -= 0.30
            elif emp_type in ["freelancer", "nghe_tu_do"]:
                postpaid_prob -= 0.10

            # Age adjustment
            if age < 25:
                postpaid_prob -= 0.15
            elif age > 35:
                postpaid_prob += 0.10

            postpaid_prob = np.clip(postpaid_prob, 0.1, 0.9)

            if self.rng.random() < postpaid_prob:
                codes[i] = "tra_sau"
                labels[i] = "Trả sau"
            else:
                codes[i] = "tra_truoc"
                labels[i] = "Trả trước"

        return codes, labels

    def _generate_service_bundle(
        self,
        is_customer: np.ndarray,
        monthly_income: np.ndarray,
        ages: np.ndarray,
        property_codes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(is_customer)
        codes = np.empty(n, dtype=object)
        labels = np.empty(n, dtype=object)

        for i in range(n):
            if not is_customer[i]:
                codes[i] = None
                labels[i] = None
                continue

            income = monthly_income[i]
            age = ages[i]
            property_status = property_codes[i] if property_codes is not None else "thue"

            # Base distribution
            distribution = {
                "mobile_only": 0.35,
                "fiber_only": 0.15,
                "tv_only": 0.05,
                "mobile_fiber": 0.20,
                "fiber_tv": 0.08,
                "mobile_tv": 0.05,
                "full_combo": 0.12,
            }

            # Property ownership affects fiber/TV likelihood
            if property_status == "so_huu":
                distribution["fiber_only"] *= 1.5
                distribution["mobile_fiber"] *= 1.5
                distribution["fiber_tv"] *= 2.0
                distribution["full_combo"] *= 2.0
            elif property_status in ["thue", "ky_tuc_xa"]:
                distribution["mobile_only"] *= 1.5
                distribution["fiber_only"] *= 0.5
                distribution["full_combo"] *= 0.3

            # Income adjustment
            if income > 30_000_000:
                distribution["full_combo"] *= 2.0
                distribution["mobile_fiber"] *= 1.5
            elif income < 8_000_000:
                distribution["mobile_only"] *= 1.5
                distribution["full_combo"] *= 0.3

            # Age adjustment
            if age < 25:
                distribution["mobile_only"] *= 1.5
                distribution["tv_only"] *= 0.3
            elif age > 50:
                distribution["tv_only"] *= 1.5
                distribution["fiber_tv"] *= 1.3

            # Normalize
            total = sum(distribution.values())
            distribution = {k: v / total for k, v in distribution.items()}

            bundle = weighted_random_choice(
                self.rng,
                list(distribution.keys()),
                list(distribution.values())
            )
            codes[i] = bundle
            labels[i] = SERVICE_LABELS[bundle]

        return codes, labels

    def _generate_monthly_arpu(
        self,
        is_customer: np.ndarray,
        service_bundles: np.ndarray,
        monthly_income: np.ndarray,
        contract_types: np.ndarray
    ) -> np.ndarray:
        n = len(is_customer)
        arpu = np.zeros(n)

        for i in range(n):
            if not is_customer[i] or service_bundles[i] is None:
                continue

            bundle = service_bundles[i]
            income = monthly_income[i]
            contract = contract_types[i]

            config = ARPU_BY_SERVICE.get(bundle)
            if config is None:
                continue

            # Base ARPU
            base_arpu = config.mean_arpu

            # Income correlation
            if income > 30_000_000:
                income_mult = 1.4
            elif income > 15_000_000:
                income_mult = 1.2
            elif income < 8_000_000:
                income_mult = 0.7
            else:
                income_mult = 1.0

            # Contract type adjustment
            if contract == "tra_sau":
                contract_mult = 1.2  # Postpaid usually higher ARPU
            else:
                contract_mult = 0.85

            adjusted_mean = base_arpu * income_mult * contract_mult

            arpu[i] = truncated_lognormal(
                self.rng,
                adjusted_mean,
                adjusted_mean * 0.3,
                config.min_arpu,
                config.max_arpu * income_mult,
                1
            )[0]

        # Round to nearest 1,000 VND
        arpu = np.round(arpu / 1000) * 1000
        return arpu

    def _generate_payment_method(
        self,
        is_customer: np.ndarray,
        contract_types: np.ndarray,
        ages: np.ndarray,
        monthly_income: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(is_customer)
        codes = np.empty(n, dtype=object)
        labels = np.empty(n, dtype=object)

        for i in range(n):
            if not is_customer[i]:
                codes[i] = None
                labels[i] = None
                continue

            contract = contract_types[i]
            age = ages[i]
            income = monthly_income[i]

            # Base distribution
            if contract == "tra_truoc":
                # Prepaid - mostly top-up
                distribution = {
                    "tien_mat": 0.40,
                    "bank_transfer": 0.20,
                    "vi_dien_tu": 0.25,
                    "vnpt_pay": 0.10,
                    "auto_debit": 0.03,
                    "credit_card": 0.02,
                }
            else:
                # Postpaid - more payment options
                distribution = {
                    "tien_mat": 0.15,
                    "bank_transfer": 0.25,
                    "vi_dien_tu": 0.20,
                    "vnpt_pay": 0.15,
                    "auto_debit": 0.18,
                    "credit_card": 0.07,
                }

            # Age adjustment (younger = more digital)
            if age < 35:
                distribution["vi_dien_tu"] *= 1.5
                distribution["vnpt_pay"] *= 1.3
                distribution["tien_mat"] *= 0.6
            elif age > 50:
                distribution["tien_mat"] *= 1.5
                distribution["vi_dien_tu"] *= 0.5

            # Income adjustment
            if income > 20_000_000:
                distribution["auto_debit"] *= 1.8
                distribution["credit_card"] *= 2.0
                distribution["tien_mat"] *= 0.5
            elif income < 8_000_000:
                distribution["tien_mat"] *= 1.5
                distribution["credit_card"] *= 0.2

            # Normalize
            total = sum(distribution.values())
            distribution = {k: v / total for k, v in distribution.items()}

            method = weighted_random_choice(
                self.rng,
                list(distribution.keys()),
                list(distribution.values())
            )
            codes[i] = method
            labels[i] = PAYMENT_LABELS[method]

        return codes, labels

    def _generate_payment_behavior(
        self,
        is_customer: np.ndarray,
        contract_types: np.ndarray,
        tenure_months: np.ndarray,
        cic_scores: np.ndarray,
        has_credit_history: np.ndarray,
        max_dpd_credit: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(is_customer)
        on_time_rate = np.ones(n)
        late_count = np.zeros(n, dtype=int)
        avg_days_late = np.zeros(n)

        for i in range(n):
            if not is_customer[i]:
                continue

            contract = contract_types[i]
            tenure = tenure_months[i]
            cic_score = cic_scores[i] if cic_scores[i] > 0 else 600
            has_history = has_credit_history[i]
            credit_dpd = max_dpd_credit[i]

            # Prepaid customers - payment at point of use
            if contract == "tra_truoc":
                on_time_rate[i] = 1.0  # Always "on time" for prepaid
                late_count[i] = 0
                avg_days_late[i] = 0
                continue

            # Postpaid - payment behavior matters
            # Base on-time rate correlated with CIC score
            if cic_score >= 750:
                base_rate = 0.95
            elif cic_score >= 650:
                base_rate = 0.88
            elif cic_score >= 550:
                base_rate = 0.78
            elif cic_score >= 450:
                base_rate = 0.65
            else:
                base_rate = 0.50

            # Correlation with credit DPD
            if credit_dpd > 90:
                base_rate *= 0.7
            elif credit_dpd > 30:
                base_rate *= 0.85

            # Tenure effect - longer tenure often means better behavior
            if tenure > 60:
                base_rate *= 1.05
            elif tenure < 12:
                base_rate *= 0.95

            base_rate = np.clip(base_rate, 0.3, 0.99)

            # Add noise
            on_time_rate[i] = np.clip(
                base_rate + self.rng.normal(0, 0.05),
                0.2, 0.99
            )

            # Calculate late payments based on tenure and on-time rate
            expected_payments = tenure
            late_prob = 1 - on_time_rate[i]
            late_count[i] = int(self.rng.binomial(
                min(expected_payments, 24),  # Last 24 months
                late_prob
            ))

            # Average days late
            if late_count[i] > 0:
                if cic_score < 500:
                    mean_days = 25
                elif cic_score < 600:
                    mean_days = 15
                else:
                    mean_days = 7

                avg_days_late[i] = max(1, self.rng.normal(mean_days, 5))

        return on_time_rate, late_count, avg_days_late

    def _generate_usage_metrics(
        self,
        is_customer: np.ndarray,
        service_bundles: np.ndarray,
        ages: np.ndarray,
        monthly_income: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(is_customer)
        data_gb = np.zeros(n)
        call_mins = np.zeros(n, dtype=int)
        sms = np.zeros(n, dtype=int)

        for i in range(n):
            if not is_customer[i]:
                continue

            bundle = service_bundles[i]
            age = ages[i]
            income = monthly_income[i]

            # Determine usage pattern
            if income > 30_000_000 and age < 45:
                pattern = "power_user"
            elif income > 15_000_000 or age < 30:
                pattern = "heavy"
            elif income > 8_000_000:
                pattern = "moderate"
            else:
                pattern = "light"

            # Data usage (only if has mobile/internet)
            if bundle and bundle != "tv_only":
                data_config = DATA_USAGE_BY_PATTERN[pattern]
                data_gb[i] = truncated_normal(
                    self.rng, data_config[2], (data_config[1] - data_config[0]) / 4,
                    data_config[0], data_config[1], 1
                )[0]

            # Call minutes (only if has mobile)
            if bundle and "mobile" in bundle or bundle == "mobile_only":
                call_config = CALL_MINUTES_BY_PATTERN[pattern]
                call_mins[i] = int(truncated_normal(
                    self.rng, call_config[2], (call_config[1] - call_config[0]) / 4,
                    call_config[0], call_config[1], 1
                )[0])

                # SMS (declining, mostly older users)
                if age > 40:
                    sms[i] = int(self.rng.poisson(20))
                elif age > 30:
                    sms[i] = int(self.rng.poisson(10))
                else:
                    sms[i] = int(self.rng.poisson(5))

        return np.round(data_gb, 2), call_mins, sms

    def _generate_service_complaints(
        self,
        is_customer: np.ndarray,
        tenure_months: np.ndarray,
        on_time_rate: np.ndarray
    ) -> np.ndarray:
        n = len(is_customer)
        complaints = np.zeros(n, dtype=int)

        for i in range(n):
            if not is_customer[i]:
                continue

            tenure = tenure_months[i]
            payment_rate = on_time_rate[i]

            # Base complaint rate per year
            base_rate = 0.5  # Average 0.5 complaints per year

            # Bad payers tend to complain more
            if payment_rate < 0.7:
                base_rate *= 1.5

            # Calculate expected complaints based on tenure
            expected = base_rate * (tenure / 12)

            complaints[i] = int(self.rng.poisson(expected))
            complaints[i] = min(complaints[i], 20)  # Cap

        return complaints

    def _generate_loyalty_tier(
        self,
        is_customer: np.ndarray,
        tenure_months: np.ndarray,
        monthly_arpu: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(is_customer)
        codes = np.empty(n, dtype=object)
        labels = np.empty(n, dtype=object)

        for i in range(n):
            if not is_customer[i]:
                codes[i] = None
                labels[i] = None
                continue

            tenure = tenure_months[i]
            arpu = monthly_arpu[i]

            # Tier based on tenure and ARPU
            if tenure >= 120 and arpu >= 500_000:
                tier = "diamond"
            elif tenure >= 60 and arpu >= 300_000:
                tier = "platinum"
            elif tenure >= 24 and arpu >= 200_000:
                tier = "gold"
            elif tenure >= 12 and arpu >= 100_000:
                tier = "silver"
            else:
                tier = "standard"

            codes[i] = tier
            labels[i] = LOYALTY_LABELS[tier]

        return codes, labels

    def _generate_service_changes(
        self,
        is_customer: np.ndarray,
        tenure_months: np.ndarray,
        monthly_income: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(is_customer)
        upgrades = np.zeros(n, dtype=int)
        downgrades = np.zeros(n, dtype=int)

        for i in range(n):
            if not is_customer[i]:
                continue

            tenure = tenure_months[i]
            income = monthly_income[i]

            # Upgrades more likely with higher income
            upgrade_rate = 0.3 if income > 15_000_000 else 0.15
            upgrade_rate *= tenure / 12

            # Downgrades less likely but possible
            downgrade_rate = 0.05 if income > 15_000_000 else 0.12
            downgrade_rate *= tenure / 24

            upgrades[i] = int(self.rng.poisson(upgrade_rate))
            downgrades[i] = int(self.rng.poisson(downgrade_rate))

        return upgrades, downgrades

    def _generate_churn_risk_score(
        self,
        is_customer: np.ndarray,
        on_time_rate: np.ndarray,
        complaints: np.ndarray,
        tenure_months: np.ndarray,
        num_downgrades: np.ndarray
    ) -> np.ndarray:
        n = len(is_customer)
        churn_risk = np.zeros(n)

        for i in range(n):
            if not is_customer[i]:
                continue

            # Factors contributing to churn risk
            # 1. Payment behavior (poor payment = higher churn)
            payment_risk = 1 - on_time_rate[i]

            # 2. Complaints
            complaint_risk = min(complaints[i] / 5, 1.0)

            # 3. Tenure (shorter = higher churn)
            tenure_risk = max(0, 1 - (tenure_months[i] / 60))

            # 4. Downgrades
            downgrade_risk = min(num_downgrades[i] / 3, 1.0)

            # Weighted combination
            churn_risk[i] = (
                0.30 * payment_risk +
                0.25 * complaint_risk +
                0.30 * tenure_risk +
                0.15 * downgrade_risk
            )

            # Add some noise
            churn_risk[i] = np.clip(
                churn_risk[i] + self.rng.normal(0, 0.05),
                0.01, 0.99
            )

        return np.round(churn_risk, 4)

    def _calculate_telecom_credit_score(
        self,
        is_customer: np.ndarray,
        on_time_rate: np.ndarray,
        tenure_months: np.ndarray,
        churn_risk: np.ndarray,
        payment_method_codes: np.ndarray,
        loyalty_tiers: np.ndarray,
        monthly_arpu: np.ndarray
    ) -> np.ndarray:
        n = len(is_customer)
        scores = np.zeros(n)

        for i in range(n):
            if not is_customer[i]:
                scores[i] = 0
                continue

            # 1. Payment reliability score (0-40)
            payment_score = on_time_rate[i] * 40

            # 2. Loyalty score (0-25)
            tenure = tenure_months[i]
            loyalty_tier = loyalty_tiers[i]

            tenure_score = min(tenure / 60, 1.0) * 15  # Up to 15 points for 5+ years

            tier_bonus = {
                "standard": 0,
                "silver": 2,
                "gold": 5,
                "platinum": 8,
                "diamond": 10,
            }
            loyalty_score = tenure_score + tier_bonus.get(loyalty_tier, 0)

            # 3. Financial capacity proxy (0-20)
            arpu = monthly_arpu[i]
            if arpu >= 1_000_000:
                arpu_score = 20
            elif arpu >= 500_000:
                arpu_score = 16
            elif arpu >= 300_000:
                arpu_score = 12
            elif arpu >= 150_000:
                arpu_score = 8
            else:
                arpu_score = 4

            # 4. Digital engagement score (0-15)
            payment_method = payment_method_codes[i]
            digital_methods = {"vi_dien_tu", "vnpt_pay", "auto_debit", "credit_card"}
            if payment_method in digital_methods:
                digital_score = 15
            elif payment_method == "bank_transfer":
                digital_score = 10
            else:
                digital_score = 5

            # Stability bonus (inverse of churn risk)
            stability_bonus = (1 - churn_risk[i]) * 5

            # Total score
            total = payment_score + loyalty_score + arpu_score + digital_score + stability_bonus

            # Add small noise
            total = np.clip(total + self.rng.normal(0, 2), 10, 100)

            scores[i] = round(total, 1)

        return scores

    def generate(
        self,
        demographic_df: Optional[pd.DataFrame] = None,
        financial_df: Optional[pd.DataFrame] = None,
        credit_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if demographic_df is None or financial_df is None:
            raise ValueError(
                "demographic_df and financial_df are required."
            )

        # Validate required columns
        demo_required = ['customer_id', 'age', 'province_code']
        fin_required = ['customer_id', 'monthly_income', 'employment_type_code']

        for col in demo_required:
            if col not in demographic_df.columns:
                raise ValueError(f"Missing required column in demographic_df: {col}")
        for col in fin_required:
            if col not in financial_df.columns:
                raise ValueError(f"Missing required column in financial_df: {col}")

        # Merge data
        merged = demographic_df[['customer_id', 'age', 'province_code']].merge(
            financial_df[['customer_id', 'monthly_income', 'employment_type_code', 'property_ownership_code']],
            on='customer_id'
        )

        # Add credit data if available
        if credit_df is not None:
            credit_cols = ['customer_id', 'has_credit_history', 'cic_score', 'max_dpd_12m']
            available_cols = [c for c in credit_cols if c in credit_df.columns]
            merged = merged.merge(credit_df[available_cols], on='customer_id', how='left')
        else:
            merged['has_credit_history'] = False
            merged['cic_score'] = 0
            merged['max_dpd_12m'] = 0

        n = len(merged)

        # Extract arrays
        customer_ids = merged['customer_id'].values
        ages = merged['age'].values
        province_codes = merged['province_code'].values
        monthly_income = merged['monthly_income'].values
        employment_codes = merged['employment_type_code'].values
        property_codes = merged.get('property_ownership_code', pd.Series([None] * n)).values
        has_credit_history = merged['has_credit_history'].fillna(False).values
        cic_scores = merged['cic_score'].fillna(0).values
        max_dpd = merged['max_dpd_12m'].fillna(0).values

        # Step 1: Determine VNPT customer status
        is_customer = self._generate_is_vnpt_customer(n, province_codes)

        # Step 2: Generate tenure
        tenure_months = self._generate_customer_tenure(is_customer, ages)

        # Step 3: Generate contract type
        contract_codes, contract_labels = self._generate_contract_type(
            is_customer, monthly_income, ages, employment_codes
        )

        # Step 4: Generate service bundle
        bundle_codes, bundle_labels = self._generate_service_bundle(
            is_customer, monthly_income, ages, property_codes
        )

        # Step 5: Generate ARPU
        monthly_arpu = self._generate_monthly_arpu(
            is_customer, bundle_codes, monthly_income, contract_codes
        )

        # Step 6: Generate payment method
        payment_codes, payment_labels = self._generate_payment_method(
            is_customer, contract_codes, ages, monthly_income
        )

        # Step 7: Generate payment behavior
        on_time_rate, late_count, avg_days_late = self._generate_payment_behavior(
            is_customer, contract_codes, tenure_months,
            cic_scores, has_credit_history, max_dpd
        )

        # Step 8: Generate usage metrics
        data_gb, call_mins, sms = self._generate_usage_metrics(
            is_customer, bundle_codes, ages, monthly_income
        )

        # Step 9: Generate complaints
        complaints = self._generate_service_complaints(
            is_customer, tenure_months, on_time_rate
        )

        # Step 10: Generate loyalty tier
        loyalty_codes, loyalty_labels = self._generate_loyalty_tier(
            is_customer, tenure_months, monthly_arpu
        )

        # Step 11: Generate service changes
        upgrades, downgrades = self._generate_service_changes(
            is_customer, tenure_months, monthly_income
        )

        # Step 12: Generate churn risk
        churn_risk = self._generate_churn_risk_score(
            is_customer, on_time_rate, complaints, tenure_months, downgrades
        )

        # Step 13: Calculate telecom credit score
        telecom_score = self._calculate_telecom_credit_score(
            is_customer, on_time_rate, tenure_months, churn_risk,
            payment_codes, loyalty_codes, monthly_arpu
        )

        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'is_vnpt_customer': is_customer,
            'vnpt_customer_tenure_months': tenure_months,
            'contract_type': contract_labels,
            'contract_type_code': contract_codes,
            'service_bundle': bundle_labels,
            'service_bundle_code': bundle_codes,
            'monthly_arpu': monthly_arpu,
            'payment_method': payment_labels,
            'payment_method_code': payment_codes,
            'on_time_payment_rate': np.round(on_time_rate, 4),
            'num_late_payments_telecom': late_count,
            'avg_days_late': np.round(avg_days_late, 1),
            'data_usage_gb': data_gb,
            'call_minutes': call_mins,
            'sms_count': sms,
            'service_complaints': complaints,
            'loyalty_program_tier': loyalty_labels,
            'loyalty_program_tier_code': loyalty_codes,
            'num_service_upgrades': upgrades,
            'num_service_downgrades': downgrades,
            'churn_risk_score': churn_risk,
            'telecom_credit_score': telecom_score,
        })

        self._generated_data = df
        return df

    def get_alternative_credit_analysis(
        self,
        data: Optional[pd.DataFrame] = None,
        credit_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        if data is None:
            data = self._generated_data

        if data is None:
            return {"error": "No data available"}

        vnpt_customers = data[data['is_vnpt_customer']]

        analysis = {
            "total_records": len(data),
            "vnpt_customer_rate": float(data['is_vnpt_customer'].mean()),
            "telecom_credit_stats": {},
            "payment_behavior": {},
            "segment_analysis": {},
        }

        if len(vnpt_customers) > 0:
            # Telecom credit score stats
            analysis["telecom_credit_stats"] = {
                "mean_score": float(vnpt_customers['telecom_credit_score'].mean()),
                "median_score": float(vnpt_customers['telecom_credit_score'].median()),
                "std_score": float(vnpt_customers['telecom_credit_score'].std()),
                "score_distribution": {
                    "excellent (80-100)": float((vnpt_customers['telecom_credit_score'] >= 80).mean()),
                    "good (60-80)": float((
                        (vnpt_customers['telecom_credit_score'] >= 60) &
                        (vnpt_customers['telecom_credit_score'] < 80)
                    ).mean()),
                    "fair (40-60)": float((
                        (vnpt_customers['telecom_credit_score'] >= 40) &
                        (vnpt_customers['telecom_credit_score'] < 60)
                    ).mean()),
                    "poor (<40)": float((vnpt_customers['telecom_credit_score'] < 40).mean()),
                },
            }

            # Payment behavior analysis
            postpaid = vnpt_customers[vnpt_customers['contract_type_code'] == 'tra_sau']
            if len(postpaid) > 0:
                analysis["payment_behavior"] = {
                    "postpaid_rate": float((vnpt_customers['contract_type_code'] == 'tra_sau').mean()),
                    "avg_on_time_rate": float(postpaid['on_time_payment_rate'].mean()),
                    "pct_perfect_payment": float((postpaid['on_time_payment_rate'] >= 0.95).mean()),
                    "avg_late_payments": float(postpaid['num_late_payments_telecom'].mean()),
                }

            # Correlation with credit (if available)
            if credit_df is not None:
                merged = data.merge(
                    credit_df[['customer_id', 'has_credit_history', 'cic_score']],
                    on='customer_id',
                    how='left'
                )
                thin_file = merged[~merged['has_credit_history'].fillna(False)]
                thin_vnpt = thin_file[thin_file['is_vnpt_customer']]

                if len(thin_vnpt) > 0:
                    analysis["thin_file_alternative_credit"] = {
                        "thin_file_count": int(len(thin_file)),
                        "thin_file_with_vnpt": int(len(thin_vnpt)),
                        "coverage_rate": float(len(thin_vnpt) / len(thin_file)) if len(thin_file) > 0 else 0,
                        "avg_telecom_score": float(thin_vnpt['telecom_credit_score'].mean()),
                    }

        return analysis


# MODULE EXPORTS

__all__ = [
    "VNPTBehaviorGenerator",
    "ContractType",
    "ServiceBundle",
    "PaymentMethod",
    "LoyaltyTier",
    "UsagePattern",
    "CONTRACT_LABELS",
    "SERVICE_LABELS",
    "PAYMENT_LABELS",
    "LOYALTY_LABELS",
    "ARPU_BY_SERVICE",
    "TelecomCreditSignal",
]
