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

class CICGrade(Enum):
    AAA = "AAA"  # Excellent
    AA = "AA"    # Very Good
    A = "A"      # Good
    BBB = "BBB"  # Above Average
    BB = "BB"    # Average
    B = "B"      # Below Average
    CCC = "CCC"  # Poor
    CC = "CC"    # Very Poor
    C = "C"      # Extremely Poor
    D = "D"      # Default


class NHNNLoanGroup(Enum):
    NHOM_1 = 1  # Nợ đủ tiêu chuẩn
    NHOM_2 = 2  # Nợ cần chú ý
    NHOM_3 = 3  # Nợ dưới tiêu chuẩn
    NHOM_4 = 4  # Nợ nghi ngờ
    NHOM_5 = 5  # Nợ có khả năng mất vốn


class LoanStatus(Enum):
    CURRENT = "current"              # Đang vay, trả đúng hạn
    DELINQUENT_30 = "delinquent_30"  # Quá hạn 1-30 ngày
    DELINQUENT_60 = "delinquent_60"  # Quá hạn 31-60 ngày
    DELINQUENT_90 = "delinquent_90"  # Quá hạn 61-90 ngày
    DELINQUENT_180 = "delinquent_180"  # Quá hạn 91-180 ngày
    DELINQUENT_360 = "delinquent_360"  # Quá hạn 181-360 ngày
    DEFAULT = "default"              # Nợ xấu (>360 ngày)
    CLOSED = "closed"                # Đã tất toán


# Labels in Vietnamese
CIC_GRADE_LABELS: Dict[str, str] = {
    "AAA": "Xuất sắc",
    "AA": "Rất tốt",
    "A": "Tốt",
    "BBB": "Khá",
    "BB": "Trung bình",
    "B": "Dưới trung bình",
    "CCC": "Yếu",
    "CC": "Rất yếu",
    "C": "Cực kỳ yếu",
    "D": "Vỡ nợ",
}

NHNN_GROUP_LABELS: Dict[int, str] = {
    1: "Nợ đủ tiêu chuẩn",
    2: "Nợ cần chú ý",
    3: "Nợ dưới tiêu chuẩn",
    4: "Nợ nghi ngờ",
    5: "Nợ có khả năng mất vốn",
}

# NHNN provision rates by group
NHNN_PROVISION_RATES: Dict[int, float] = {
    1: 0.00,   # 0%
    2: 0.05,   # 5%
    3: 0.20,   # 20%
    4: 0.50,   # 50%
    5: 1.00,   # 100%
}

# DPD (Days Past Due) ranges by NHNN group
NHNN_DPD_RANGES: Dict[int, Tuple[int, int]] = {
    1: (0, 10),
    2: (10, 90),
    3: (90, 180),
    4: (180, 360),
    5: (360, 720),
}

# CIC Grade to approximate PD (Probability of Default) mapping
CIC_GRADE_PD: Dict[str, Tuple[float, float]] = {
    "AAA": (0.0001, 0.001),
    "AA": (0.001, 0.005),
    "A": (0.005, 0.01),
    "BBB": (0.01, 0.03),
    "BB": (0.03, 0.06),
    "B": (0.06, 0.15),
    "CCC": (0.15, 0.30),
    "CC": (0.30, 0.50),
    "C": (0.50, 0.80),
    "D": (0.80, 1.0),
}


# CONFIGURATION

@dataclass
class ThinFileConfig:
    # Probability of being thin-file by age group
    thin_file_rate_by_age: Dict[str, float] = None

    def __post_init__(self):
        if self.thin_file_rate_by_age is None:
            self.thin_file_rate_by_age = {
                "18-24": 0.65,  # Young people often have no credit history
                "25-34": 0.35,
                "35-44": 0.25,
                "45-54": 0.20,
                "55-65": 0.30,  # Older generation may not use formal credit
            }


# CREDIT HISTORY GENERATOR

class CreditHistoryGenerator(BaseDataGenerator, CorrelationMixin):
    def __init__(
        self,
        config: SyntheticDataConfig,
        seed: Optional[int] = None
    ) -> None:
        super().__init__(config, seed)
        self.thin_file_config = ThinFileConfig()
        self._set_default_schema()

    def _set_default_schema(self) -> None:
        self.set_schema({
            'customer_id': str,
            'has_credit_history': bool,
            'credit_history_months': int,
            'cic_grade': str,
            'cic_grade_label': str,
            'cic_score': int,
            'num_active_loans': int,
            'num_closed_loans': int,
            'total_credit_limit': float,
            'current_balance': float,
            'credit_utilization': float,
            'num_credit_cards': int,
            'max_dpd_12m': int,
            'num_late_payments_12m': int,
            'num_inquiries_6m': int,
            'worst_status_12m': str,
            'nhnn_loan_group': int,
            'nhnn_loan_group_label': str,
            'bankruptcy_flag': bool,
            'estimated_pd': float,
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

    def _generate_has_credit_history(
        self,
        ages: np.ndarray,
        monthly_income: np.ndarray,
        employment_codes: np.ndarray
    ) -> np.ndarray:
        n = len(ages)
        has_history = np.zeros(n, dtype=bool)

        for i in range(n):
            age_group = self._get_age_group(ages[i])
            income = monthly_income[i]
            emp_type = employment_codes[i]

            # Base thin-file rate
            base_thin_rate = self.thin_file_config.thin_file_rate_by_age[age_group]

            # Income adjustment - higher income = more likely to have credit
            if income > 30_000_000:
                income_adj = 0.6
            elif income > 15_000_000:
                income_adj = 0.8
            elif income > 8_000_000:
                income_adj = 1.0
            else:
                income_adj = 1.3

            # Employment adjustment
            if emp_type in ["cong_chuc", "nhan_vien"]:
                emp_adj = 0.8  # Formal employment = more credit access
            elif emp_type in ["tu_kinh_doanh"]:
                emp_adj = 0.9
            elif emp_type in ["that_nghiep"]:
                emp_adj = 1.5
            else:
                emp_adj = 1.0

            # Final thin-file probability
            thin_rate = base_thin_rate * income_adj * emp_adj
            thin_rate = min(thin_rate, 0.9)

            has_history[i] = self.rng.random() > thin_rate

        return has_history

    def _generate_credit_history_months(
        self,
        has_history: np.ndarray,
        ages: np.ndarray
    ) -> np.ndarray:
        n = len(has_history)
        history_months = np.zeros(n, dtype=int)

        for i in range(n):
            if not has_history[i]:
                continue

            age = ages[i]
            max_history = max(3, (age - 18) * 12)  # Max possible history, at least 3

            if max_history <= 6:
                # Very young - simple random
                history_months[i] = int(self.rng.integers(3, max(4, max_history + 1)))
                continue

            if age < 25:
                mean_history = min(max_history * 0.3, 24)
            elif age < 35:
                mean_history = min(max_history * 0.4, 60)
            elif age < 45:
                mean_history = min(max_history * 0.5, 120)
            else:
                mean_history = min(max_history * 0.6, 180)

            mean_history = max(mean_history, 6)
            std_history = max(1, mean_history * 0.4)  # Ensure std is positive

            history = truncated_normal(
                self.rng, mean_history, std_history,
                3, max_history, 1
            )[0]
            history_months[i] = int(max(3, history))

        return history_months

    def _generate_cic_grade_and_score(
        self,
        has_history: np.ndarray,
        ages: np.ndarray,
        monthly_income: np.ndarray,
        existing_debt: np.ndarray,
        employment_codes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(has_history)
        grades = np.empty(n, dtype=object)
        grade_labels = np.empty(n, dtype=object)
        scores = np.zeros(n, dtype=int)

        # Score ranges for each grade
        grade_score_ranges = {
            "AAA": (820, 900),
            "AA": (760, 819),
            "A": (700, 759),
            "BBB": (640, 699),
            "BB": (580, 639),
            "B": (520, 579),
            "CCC": (460, 519),
            "CC": (400, 459),
            "C": (350, 399),
            "D": (300, 349),
        }

        for i in range(n):
            if not has_history[i]:
                grades[i] = None
                grade_labels[i] = "Không có"
                scores[i] = 0
                continue

            income = monthly_income[i]
            debt = existing_debt[i]
            age = ages[i]
            emp_type = employment_codes[i]

            # Calculate base score
            base_score = 600

            # Income factor (+/- 100 points)
            if income > 50_000_000:
                base_score += 80
            elif income > 30_000_000:
                base_score += 50
            elif income > 15_000_000:
                base_score += 20
            elif income < 8_000_000:
                base_score -= 30

            # DTI factor (+/- 80 points)
            if income > 0:
                monthly_debt_payment = debt / 60  # Assume 5-year term
                dti = monthly_debt_payment / income
                if dti < 0.2:
                    base_score += 50
                elif dti < 0.3:
                    base_score += 20
                elif dti > 0.5:
                    base_score -= 60
                elif dti > 0.4:
                    base_score -= 30

            # Age factor (+/- 40 points)
            if 30 <= age <= 50:
                base_score += 30
            elif age < 25:
                base_score -= 20
            elif age > 60:
                base_score -= 10

            # Employment factor (+/- 50 points)
            if emp_type in ["cong_chuc"]:
                base_score += 40
            elif emp_type in ["nhan_vien"]:
                base_score += 20
            elif emp_type in ["that_nghiep"]:
                base_score -= 80
            elif emp_type in ["freelancer", "nghe_tu_do"]:
                base_score -= 20

            # Add some randomness
            noise = self.rng.normal(0, 40)
            final_score = int(np.clip(base_score + noise, 300, 900))
            scores[i] = final_score

            # Determine grade from score
            for grade, (low, high) in grade_score_ranges.items():
                if low <= final_score <= high:
                    grades[i] = grade
                    grade_labels[i] = CIC_GRADE_LABELS[grade]
                    break

        return grades, grade_labels, scores

    def _generate_loan_counts(
        self,
        has_history: np.ndarray,
        credit_history_months: np.ndarray,
        ages: np.ndarray,
        monthly_income: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(has_history)
        active_loans = np.zeros(n, dtype=int)
        closed_loans = np.zeros(n, dtype=int)

        for i in range(n):
            if not has_history[i]:
                continue

            history = credit_history_months[i]
            income = monthly_income[i]
            age = ages[i]

            # Higher income and longer history = more loans
            base_total = history / 24  # Expect ~1 loan per 2 years

            # Income adjustment
            if income > 30_000_000:
                income_mult = 1.5
            elif income > 15_000_000:
                income_mult = 1.2
            else:
                income_mult = 0.8

            # Age adjustment
            if 30 <= age <= 50:
                age_mult = 1.3
            elif age < 25:
                age_mult = 0.6
            else:
                age_mult = 1.0

            expected_total = base_total * income_mult * age_mult

            # Generate total loans (Poisson-like distribution)
            total_loans = int(self.rng.poisson(max(0.5, expected_total)))
            total_loans = min(total_loans, 15)  # Cap at 15

            if total_loans > 0:
                # Split between active and closed
                active_rate = 0.3 + self.rng.uniform(-0.1, 0.2)
                active_loans[i] = max(0, int(total_loans * active_rate))
                closed_loans[i] = total_loans - active_loans[i]

        return active_loans, closed_loans

    def _generate_credit_limits_and_balance(
        self,
        has_history: np.ndarray,
        num_active_loans: np.ndarray,
        monthly_income: np.ndarray,
        cic_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(has_history)
        credit_limit = np.zeros(n)
        current_balance = np.zeros(n)
        utilization = np.zeros(n)

        for i in range(n):
            if not has_history[i] or num_active_loans[i] == 0:
                continue

            income = monthly_income[i]
            score = cic_scores[i]
            active = num_active_loans[i]

            # Credit limit based on income and score
            # Typically 12-60 months of income depending on score
            if score >= 750:
                limit_months = self.rng.uniform(36, 60)
            elif score >= 650:
                limit_months = self.rng.uniform(24, 48)
            elif score >= 550:
                limit_months = self.rng.uniform(12, 36)
            else:
                limit_months = self.rng.uniform(6, 24)

            base_limit = income * limit_months * active * 0.5

            # Add variance
            credit_limit[i] = truncated_lognormal(
                self.rng, base_limit, base_limit * 0.3,
                income * 3, income * 120, 1
            )[0]

            # Utilization - inversely correlated with score
            if score >= 750:
                util_mean = 0.2
            elif score >= 650:
                util_mean = 0.35
            elif score >= 550:
                util_mean = 0.5
            else:
                util_mean = 0.7

            util = truncated_normal(
                self.rng, util_mean, 0.15,
                0.01, 0.95, 1
            )[0]
            utilization[i] = round(util, 4)

            current_balance[i] = credit_limit[i] * util

        # Round to nearest 100,000 VND
        credit_limit = np.round(credit_limit / 100_000) * 100_000
        current_balance = np.round(current_balance / 100_000) * 100_000

        return credit_limit, current_balance, utilization

    def _generate_credit_cards(
        self,
        has_history: np.ndarray,
        monthly_income: np.ndarray,
        ages: np.ndarray,
        province_codes: np.ndarray
    ) -> np.ndarray:
        n = len(has_history)
        cards = np.zeros(n, dtype=int)

        # Tier 1 provinces for credit card penetration
        tier1_provinces = {"HN", "HCM"}

        for i in range(n):
            if not has_history[i]:
                continue

            income = monthly_income[i]
            age = ages[i]
            province = province_codes[i]

            # Base probability of having credit card
            if income > 20_000_000:
                base_prob = 0.7
            elif income > 10_000_000:
                base_prob = 0.4
            else:
                base_prob = 0.15

            # Location adjustment
            if province in tier1_provinces:
                base_prob *= 1.3

            # Age adjustment
            if age < 25:
                base_prob *= 0.5
            elif 25 <= age <= 45:
                base_prob *= 1.2
            elif age > 55:
                base_prob *= 0.7

            base_prob = min(base_prob, 0.9)

            if self.rng.random() < base_prob:
                # Number of cards (most have 1-2)
                if income > 50_000_000:
                    cards[i] = int(self.rng.choice([1, 2, 2, 3, 3, 4]))
                elif income > 20_000_000:
                    cards[i] = int(self.rng.choice([1, 1, 2, 2, 3]))
                else:
                    cards[i] = int(self.rng.choice([1, 1, 1, 2]))

        return cards

    def _generate_delinquency_data(
        self,
        has_history: np.ndarray,
        cic_scores: np.ndarray,
        num_active_loans: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(has_history)
        max_dpd = np.zeros(n, dtype=int)
        late_payments = np.zeros(n, dtype=int)
        worst_status = np.empty(n, dtype=object)
        nhnn_group = np.ones(n, dtype=int)  # Default to group 1

        for i in range(n):
            if not has_history[i]:
                worst_status[i] = None
                continue

            score = cic_scores[i]
            active = num_active_loans[i]

            # Probability of delinquency based on score
            if score >= 750:
                delq_prob = 0.02
                severe_prob = 0.001
            elif score >= 650:
                delq_prob = 0.08
                severe_prob = 0.01
            elif score >= 550:
                delq_prob = 0.20
                severe_prob = 0.05
            elif score >= 450:
                delq_prob = 0.40
                severe_prob = 0.15
            else:
                delq_prob = 0.60
                severe_prob = 0.30

            if self.rng.random() < delq_prob:
                # Has some delinquency
                if self.rng.random() < severe_prob:
                    # Severe delinquency (90+ DPD)
                    group = self.rng.choice([3, 4, 5], p=[0.5, 0.35, 0.15])
                    dpd_range = NHNN_DPD_RANGES[group]
                    max_dpd[i] = int(self.rng.integers(dpd_range[0], dpd_range[1]))
                    nhnn_group[i] = group

                    if group >= 4:
                        worst_status[i] = "default"
                    elif group == 3:
                        worst_status[i] = "delinquent_180"
                    else:
                        worst_status[i] = "delinquent_90"
                else:
                    # Mild delinquency (10-90 DPD)
                    nhnn_group[i] = 2
                    max_dpd[i] = int(self.rng.integers(10, 90))

                    if max_dpd[i] > 60:
                        worst_status[i] = "delinquent_90"
                    elif max_dpd[i] > 30:
                        worst_status[i] = "delinquent_60"
                    else:
                        worst_status[i] = "delinquent_30"

                # Late payments count
                late_payments[i] = int(self.rng.integers(1, max(2, max_dpd[i] // 30 + 2)))
            else:
                # No delinquency
                nhnn_group[i] = 1
                max_dpd[i] = int(self.rng.integers(0, 10))  # 0-10 DPD is still group 1
                worst_status[i] = "current"
                late_payments[i] = 0 if max_dpd[i] == 0 else 1

        return max_dpd, late_payments, worst_status, nhnn_group

    def _generate_inquiries(
        self,
        has_history: np.ndarray,
        ages: np.ndarray,
        monthly_income: np.ndarray
    ) -> np.ndarray:
        n = len(has_history)
        inquiries = np.zeros(n, dtype=int)

        for i in range(n):
            if not has_history[i]:
                continue

            age = ages[i]
            income = monthly_income[i]

            # Base inquiry rate
            if 25 <= age <= 45:
                base_rate = 1.5  # Peak borrowing years
            elif age < 25:
                base_rate = 1.0
            else:
                base_rate = 0.8

            # Income adjustment
            if income > 20_000_000:
                base_rate *= 1.3
            elif income < 8_000_000:
                base_rate *= 0.6

            # Generate inquiries (Poisson)
            inquiries[i] = int(self.rng.poisson(base_rate))
            inquiries[i] = min(inquiries[i], 10)  # Cap at 10

        return inquiries

    def _generate_bankruptcy_flag(
        self,
        has_history: np.ndarray,
        nhnn_groups: np.ndarray,
        cic_scores: np.ndarray
    ) -> np.ndarray:
        n = len(has_history)
        bankruptcy = np.zeros(n, dtype=bool)

        for i in range(n):
            if not has_history[i]:
                continue

            group = nhnn_groups[i]
            score = cic_scores[i]

            # Very low probability, higher for group 5
            if group == 5:
                prob = 0.15
            elif group == 4:
                prob = 0.05
            elif score < 400:
                prob = 0.02
            else:
                prob = 0.001

            bankruptcy[i] = self.rng.random() < prob

        return bankruptcy

    def _calculate_estimated_pd(
        self,
        has_history: np.ndarray,
        cic_grades: np.ndarray,
        nhnn_groups: np.ndarray
    ) -> np.ndarray:
        n = len(has_history)
        pd_values = np.zeros(n)

        for i in range(n):
            if not has_history[i]:
                pd_values[i] = 0.15  # Default PD for thin-file
                continue

            grade = cic_grades[i]
            group = nhnn_groups[i]

            # Base PD from grade
            if grade and grade in CIC_GRADE_PD:
                pd_range = CIC_GRADE_PD[grade]
                base_pd = self.rng.uniform(pd_range[0], pd_range[1])
            else:
                base_pd = 0.15

            # Adjust for NHNN group
            if group >= 3:
                # Already delinquent
                base_pd = max(base_pd, 0.3)
            if group >= 4:
                base_pd = max(base_pd, 0.5)
            if group == 5:
                base_pd = max(base_pd, 0.8)

            pd_values[i] = round(base_pd, 4)

        return pd_values

    def generate(
        self,
        demographic_df: Optional[pd.DataFrame] = None,
        financial_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if demographic_df is None or financial_df is None:
            raise ValueError(
                "Both demographic_df and financial_df are required. "
                "Generate them first using DemographicGenerator and FinancialGenerator."
            )

        # Validate required columns
        demo_required = ['customer_id', 'age', 'province_code']
        fin_required = ['customer_id', 'monthly_income', 'existing_debt', 'employment_type_code']

        for col in demo_required:
            if col not in demographic_df.columns:
                raise ValueError(f"Missing required column in demographic_df: {col}")
        for col in fin_required:
            if col not in financial_df.columns:
                raise ValueError(f"Missing required column in financial_df: {col}")

        # Merge data
        merged = demographic_df.merge(financial_df, on='customer_id')
        n = len(merged)

        # Extract arrays
        customer_ids = merged['customer_id'].values
        ages = merged['age'].values
        province_codes = merged['province_code'].values
        monthly_income = merged['monthly_income'].values
        existing_debt = merged['existing_debt'].values
        employment_codes = merged['employment_type_code'].values

        # Step 1: Determine credit history presence
        has_history = self._generate_has_credit_history(
            ages, monthly_income, employment_codes
        )

        # Step 2: Generate credit history length
        history_months = self._generate_credit_history_months(has_history, ages)

        # Step 3: Generate CIC grade and score
        cic_grades, cic_labels, cic_scores = self._generate_cic_grade_and_score(
            has_history, ages, monthly_income, existing_debt, employment_codes
        )

        # Step 4: Generate loan counts
        active_loans, closed_loans = self._generate_loan_counts(
            has_history, history_months, ages, monthly_income
        )

        # Step 5: Generate credit limits and utilization
        credit_limit, current_balance, utilization = self._generate_credit_limits_and_balance(
            has_history, active_loans, monthly_income, cic_scores
        )

        # Step 6: Generate credit cards
        credit_cards = self._generate_credit_cards(
            has_history, monthly_income, ages, province_codes
        )

        # Step 7: Generate delinquency data
        max_dpd, late_payments, worst_status, nhnn_groups = self._generate_delinquency_data(
            has_history, cic_scores, active_loans
        )

        # Step 8: Generate inquiries
        inquiries = self._generate_inquiries(has_history, ages, monthly_income)

        # Step 9: Generate bankruptcy flag
        bankruptcy = self._generate_bankruptcy_flag(has_history, nhnn_groups, cic_scores)

        # Step 10: Calculate estimated PD
        estimated_pd = self._calculate_estimated_pd(has_history, cic_grades, nhnn_groups)

        # Generate NHNN group labels
        nhnn_labels = np.array([
            NHNN_GROUP_LABELS.get(g, "Không xác định") if h else "Không có"
            for g, h in zip(nhnn_groups, has_history)
        ])

        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'has_credit_history': has_history,
            'credit_history_months': history_months,
            'cic_grade': cic_grades,
            'cic_grade_label': cic_labels,
            'cic_score': cic_scores,
            'num_active_loans': active_loans,
            'num_closed_loans': closed_loans,
            'total_credit_limit': credit_limit,
            'current_balance': current_balance,
            'credit_utilization': utilization,
            'num_credit_cards': credit_cards,
            'max_dpd_12m': max_dpd,
            'num_late_payments_12m': late_payments,
            'num_inquiries_6m': inquiries,
            'worst_status_12m': worst_status,
            'nhnn_loan_group': nhnn_groups,
            'nhnn_loan_group_label': nhnn_labels,
            'bankruptcy_flag': bankruptcy,
            'estimated_pd': estimated_pd,
        })

        # Set null values for thin-file customers
        thin_file_mask = ~df['has_credit_history']
        thin_file_cols = [
            'credit_history_months', 'cic_score', 'num_active_loans',
            'num_closed_loans', 'total_credit_limit', 'current_balance',
            'credit_utilization', 'num_credit_cards', 'max_dpd_12m',
            'num_late_payments_12m', 'num_inquiries_6m'
        ]
        for col in thin_file_cols:
            df.loc[thin_file_mask, col] = 0

        self._generated_data = df
        return df

    def get_credit_summary(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        if data is None:
            data = self._generated_data

        if data is None:
            return {"error": "No data available"}

        has_history = data['has_credit_history']

        summary = {
            "sample_size": len(data),
            "thin_file_rate": float((~has_history).mean()),
            "with_history": {},
            "cic_distribution": {},
            "nhnn_distribution": {},
            "delinquency_stats": {},
        }

        # Stats for those with history
        with_history = data[has_history]
        if len(with_history) > 0:
            summary["with_history"] = {
                "count": int(len(with_history)),
                "avg_history_months": float(with_history['credit_history_months'].mean()),
                "avg_cic_score": float(with_history['cic_score'].mean()),
                "avg_active_loans": float(with_history['num_active_loans'].mean()),
                "avg_utilization": float(with_history['credit_utilization'].mean()),
                "credit_card_penetration": float((with_history['num_credit_cards'] > 0).mean()),
            }

            # CIC grade distribution
            summary["cic_distribution"] = with_history['cic_grade'].value_counts(
                normalize=True
            ).to_dict()

            # NHNN group distribution
            summary["nhnn_distribution"] = with_history['nhnn_loan_group'].value_counts(
                normalize=True
            ).to_dict()

            # Delinquency stats
            summary["delinquency_stats"] = {
                "pct_ever_delinquent": float((with_history['max_dpd_12m'] > 10).mean()),
                "pct_severe_delinquent": float((with_history['max_dpd_12m'] > 90).mean()),
                "avg_max_dpd": float(with_history['max_dpd_12m'].mean()),
                "bankruptcy_rate": float(with_history['bankruptcy_flag'].mean()),
            }

        return summary


# MODULE EXPORTS

__all__ = [
    "CreditHistoryGenerator",
    "CICGrade",
    "NHNNLoanGroup",
    "LoanStatus",
    "CIC_GRADE_LABELS",
    "NHNN_GROUP_LABELS",
    "NHNN_PROVISION_RATES",
    "NHNN_DPD_RANGES",
    "CIC_GRADE_PD",
    "ThinFileConfig",
]
