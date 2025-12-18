from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, '/home/thanhdang/Desktop/vnpt-ai')

from config.settings import SyntheticDataConfig, Region
from generators.base import (
    BaseDataGenerator,
    CorrelationMixin,
    weighted_random_choice,
)


# ENUMS AND CONSTANTS

class EmploymentType(Enum):
    CONG_CHUC = "cong_chuc"              # Government employee
    NHAN_VIEN = "nhan_vien"              # Office worker/Employee
    TU_KINH_DOANH = "tu_kinh_doanh"      # Self-employed/Business owner
    FREELANCER = "freelancer"            # Freelancer
    CONG_NHAN = "cong_nhan"              # Factory worker
    NGHE_TU_DO = "nghe_tu_do"            # Gig worker
    HUU_TRI = "huu_tri"                  # Retired
    THAT_NGHIEP = "that_nghiep"          # Unemployed


class EmployerType(Enum):
    NHA_NUOC = "nha_nuoc"                # State-owned
    TU_NHAN = "tu_nhan"                  # Private Vietnamese
    FDI = "fdi"                          # Foreign Direct Investment
    STARTUP = "startup"                  # Startup
    LIEN_DOANH = "lien_doanh"            # Joint venture
    TU_DO = "tu_do"                      # Self/No employer
    KHAC = "khac"                        # Other


class PropertyOwnership(Enum):
    SO_HUU = "so_huu"                    # Own property
    THUE = "thue"                        # Renting
    O_CUNG_GIA_DINH = "o_cung_gia_dinh"  # Living with family
    NHA_CONG_TY = "nha_cong_ty"          # Company housing
    KY_TUC_XA = "ky_tuc_xa"              # Dormitory


# Employment type labels
EMPLOYMENT_LABELS: Dict[str, str] = {
    "cong_chuc": "Công chức/Viên chức",
    "nhan_vien": "Nhân viên văn phòng",
    "tu_kinh_doanh": "Tự kinh doanh",
    "freelancer": "Freelancer",
    "cong_nhan": "Công nhân",
    "nghe_tu_do": "Nghề tự do",
    "huu_tri": "Hưu trí",
    "that_nghiep": "Thất nghiệp",
}

# Employer type labels
EMPLOYER_LABELS: Dict[str, str] = {
    "nha_nuoc": "Nhà nước",
    "tu_nhan": "Tư nhân Việt Nam",
    "fdi": "Công ty FDI",
    "startup": "Startup",
    "lien_doanh": "Liên doanh",
    "tu_do": "Tự do/Không có",
    "khac": "Khác",
}

# Property ownership labels
PROPERTY_LABELS: Dict[str, str] = {
    "so_huu": "Sở hữu",
    "thue": "Thuê",
    "o_cung_gia_dinh": "Ở cùng gia đình",
    "nha_cong_ty": "Nhà công ty",
    "ky_tuc_xa": "Ký túc xá",
}


# INCOME CONFIGURATION BY REGION AND EDUCATION

@dataclass
class IncomeConfig:
    min_income: int          # VND/month
    max_income: int          # VND/month
    mean_income: int         # VND/month
    std_income: int          # Standard deviation


# Income by province tier
INCOME_BY_TIER: Dict[str, IncomeConfig] = {
    "tier1": IncomeConfig(  # Hà Nội, HCM
        min_income=8_000_000,
        max_income=150_000_000,
        mean_income=18_000_000,
        std_income=15_000_000
    ),
    "tier2": IncomeConfig(  # Đà Nẵng, Hải Phòng, Cần Thơ, Bình Dương, Đồng Nai
        min_income=6_000_000,
        max_income=80_000_000,
        mean_income=12_000_000,
        std_income=8_000_000
    ),
    "tier3": IncomeConfig(  # Other provinces
        min_income=4_000_000,
        max_income=50_000_000,
        mean_income=8_000_000,
        std_income=5_000_000
    ),
}

# Province to tier mapping
PROVINCE_TIER: Dict[str, str] = {
    "HN": "tier1", "HCM": "tier1",
    "DN": "tier2", "HP": "tier2", "CT": "tier2", "BDg": "tier2", "DNa": "tier2",
    "QN": "tier2", "BN": "tier2", "BR": "tier2",
}

# Education income multiplier
EDUCATION_INCOME_MULTIPLIER: Dict[str, float] = {
    "tieu_hoc": 0.6,
    "thcs": 0.7,
    "thpt": 0.85,
    "trung_cap": 0.95,
    "cao_dang": 1.0,
    "dai_hoc": 1.3,
    "thac_si": 1.7,
    "tien_si": 2.2,
}

# Employment type income multiplier
EMPLOYMENT_INCOME_MULTIPLIER: Dict[str, Tuple[float, float]] = {
    "cong_chuc": (0.8, 1.2),       # Stable but moderate
    "nhan_vien": (0.9, 1.3),       # Average
    "tu_kinh_doanh": (0.5, 2.5),   # High variance
    "freelancer": (0.6, 2.0),      # High variance
    "cong_nhan": (0.7, 1.0),       # Lower range
    "nghe_tu_do": (0.4, 1.5),      # Very variable
    "huu_tri": (0.3, 0.7),         # Pension
    "that_nghiep": (0.0, 0.3),     # Minimal/none
}


# HELPER FUNCTIONS

def truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    size: int = 1
) -> np.ndarray:
    if std <= 0:
        return np.full(size, mean)

    a = (low - mean) / std
    b = (high - mean) / std

    # Use scipy's truncnorm
    samples = stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size, random_state=rng)
    return samples


def truncated_lognormal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    size: int = 1
) -> np.ndarray:
    # Convert to log-space parameters
    if mean <= 0:
        return np.full(size, low)

    # Calculate lognormal parameters
    variance = std ** 2
    mu = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
    sigma = np.sqrt(np.log(1 + variance / mean ** 2))

    samples = rng.lognormal(mu, sigma, size=size)
    samples = np.clip(samples, low, high)

    return samples


# FINANCIAL GENERATOR

class FinancialGenerator(BaseDataGenerator, CorrelationMixin):
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
            'employment_type': str,
            'employment_type_code': str,
            'employer_type': str,
            'employer_type_code': str,
            'job_tenure_months': int,
            'monthly_income': float,
            'income_source_count': int,
            'monthly_expenses': float,
            'savings_amount': float,
            'existing_debt': float,
            'dti_ratio': float,
            'has_insurance': bool,
            'property_ownership': str,
            'property_ownership_code': str,
        })

    def _get_province_tier(self, province_code: str) -> str:
        return PROVINCE_TIER.get(province_code, "tier3")

    def _generate_employment_type(
        self,
        ages: np.ndarray,
        education_codes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(ages)
        emp_codes = np.empty(n, dtype=object)
        emp_labels = np.empty(n, dtype=object)

        for i in range(n):
            age = ages[i]
            education = education_codes[i]

            # Base distribution
            distribution = {
                "cong_chuc": 0.12,
                "nhan_vien": 0.30,
                "tu_kinh_doanh": 0.15,
                "freelancer": 0.08,
                "cong_nhan": 0.20,
                "nghe_tu_do": 0.10,
                "huu_tri": 0.03,
                "that_nghiep": 0.02,
            }

            # Age adjustments
            if age < 25:
                distribution["cong_chuc"] *= 0.3
                distribution["huu_tri"] = 0.0
                distribution["freelancer"] *= 1.5
                distribution["that_nghiep"] *= 2.0
            elif age >= 55:
                distribution["huu_tri"] *= 5.0
                distribution["freelancer"] *= 0.5
                distribution["cong_nhan"] *= 0.6
            elif age >= 45:
                distribution["tu_kinh_doanh"] *= 1.3
                distribution["cong_chuc"] *= 1.2

            # Education adjustments
            if education in ["dai_hoc", "thac_si", "tien_si"]:
                distribution["nhan_vien"] *= 1.5
                distribution["cong_chuc"] *= 1.4
                distribution["freelancer"] *= 1.3
                distribution["cong_nhan"] *= 0.3
            elif education in ["tieu_hoc", "thcs"]:
                distribution["cong_nhan"] *= 2.0
                distribution["nghe_tu_do"] *= 1.5
                distribution["nhan_vien"] *= 0.4
                distribution["cong_chuc"] *= 0.2

            # Normalize
            total = sum(distribution.values())
            distribution = {k: v / total for k, v in distribution.items()}

            emp_code = weighted_random_choice(
                self.rng,
                list(distribution.keys()),
                list(distribution.values())
            )
            emp_codes[i] = emp_code
            emp_labels[i] = EMPLOYMENT_LABELS[emp_code]

        return emp_codes, emp_labels

    def _generate_employer_type(
        self,
        employment_codes: np.ndarray,
        province_codes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(employment_codes)
        employer_codes = np.empty(n, dtype=object)
        employer_labels = np.empty(n, dtype=object)

        for i in range(n):
            emp_type = employment_codes[i]
            province = province_codes[i]
            tier = self._get_province_tier(province)

            # Employment-specific distributions
            if emp_type == "cong_chuc":
                distribution = {"nha_nuoc": 1.0}
            elif emp_type in ["tu_kinh_doanh", "freelancer", "nghe_tu_do"]:
                distribution = {"tu_do": 1.0}
            elif emp_type in ["huu_tri", "that_nghiep"]:
                distribution = {"khac": 1.0}
            else:
                # Regular employees
                distribution = {
                    "nha_nuoc": 0.15,
                    "tu_nhan": 0.45,
                    "fdi": 0.20,
                    "startup": 0.10,
                    "lien_doanh": 0.08,
                    "khac": 0.02,
                }

                # Location adjustments
                if tier == "tier1":
                    distribution["fdi"] *= 1.5
                    distribution["startup"] *= 2.0
                elif tier == "tier3":
                    distribution["fdi"] *= 0.3
                    distribution["startup"] *= 0.3
                    distribution["tu_nhan"] *= 1.3

            # Normalize
            total = sum(distribution.values())
            distribution = {k: v / total for k, v in distribution.items()}

            employer_code = weighted_random_choice(
                self.rng,
                list(distribution.keys()),
                list(distribution.values())
            )
            employer_codes[i] = employer_code
            employer_labels[i] = EMPLOYER_LABELS[employer_code]

        return employer_codes, employer_labels

    def _generate_job_tenure(
        self,
        ages: np.ndarray,
        employment_codes: np.ndarray
    ) -> np.ndarray:
        n = len(ages)
        tenure = np.zeros(n, dtype=int)

        for i in range(n):
            age = ages[i]
            emp_type = employment_codes[i]
            max_tenure = max(1, (age - 18) * 12)  # Max possible tenure, at least 1

            if emp_type in ["that_nghiep"]:
                tenure[i] = 0
            elif max_tenure <= 6:
                # Very young workers - simple random
                tenure[i] = int(self.rng.integers(1, max(2, max_tenure + 1)))
            elif emp_type == "huu_tri":
                # Retired - use last job tenure
                mean_tenure = max(12, min(max_tenure * 0.6, 300))
                std_tenure = max(1, mean_tenure * 0.3)
                tenure[i] = int(truncated_normal(
                    self.rng, mean_tenure, std_tenure,
                    12, max_tenure, 1
                )[0])
            elif emp_type == "cong_chuc":
                # Government - longer tenure
                mean_tenure = max(6, min(max_tenure * 0.5, 240))
                std_tenure = max(1, mean_tenure * 0.4)
                tenure[i] = int(truncated_normal(
                    self.rng, mean_tenure, std_tenure,
                    6, max_tenure, 1
                )[0])
            elif emp_type in ["freelancer", "nghe_tu_do"]:
                # Freelancers - shorter tenure
                mean_tenure = min(24, max_tenure * 0.5)
                std_tenure = max(1, min(18, mean_tenure * 0.5))
                tenure[i] = int(truncated_normal(
                    self.rng, mean_tenure, std_tenure,
                    1, min(120, max_tenure), 1
                )[0])
            else:
                # Regular employees
                mean_tenure = min(36, max_tenure * 0.5)
                std_tenure = max(1, min(30, mean_tenure * 0.6))
                tenure[i] = int(truncated_normal(
                    self.rng, mean_tenure, std_tenure,
                    1, min(180, max_tenure), 1
                )[0])

        return tenure

    def _generate_monthly_income(
        self,
        province_codes: np.ndarray,
        education_codes: np.ndarray,
        employment_codes: np.ndarray,
        ages: np.ndarray
    ) -> np.ndarray:
        n = len(province_codes)
        income = np.zeros(n)

        for i in range(n):
            tier = self._get_province_tier(province_codes[i])
            education = education_codes[i]
            emp_type = employment_codes[i]
            age = ages[i]

            # Base income from tier
            config = INCOME_BY_TIER[tier]

            # Education multiplier
            edu_mult = EDUCATION_INCOME_MULTIPLIER.get(education, 1.0)

            # Employment multiplier (random within range)
            emp_range = EMPLOYMENT_INCOME_MULTIPLIER.get(emp_type, (0.8, 1.2))
            emp_mult = self.rng.uniform(emp_range[0], emp_range[1])

            # Age/experience multiplier
            if age < 25:
                age_mult = 0.7
            elif age < 35:
                age_mult = 1.0
            elif age < 45:
                age_mult = 1.15
            elif age < 55:
                age_mult = 1.1
            else:
                age_mult = 0.9

            # Calculate adjusted parameters
            adjusted_mean = config.mean_income * edu_mult * emp_mult * age_mult
            adjusted_std = config.std_income * edu_mult

            # Generate income
            income[i] = truncated_lognormal(
                self.rng,
                adjusted_mean,
                adjusted_std,
                config.min_income * 0.5,  # Allow some below minimum
                config.max_income * emp_mult,
                1
            )[0]

        # Round to nearest 100,000 VND
        income = np.round(income / 100_000) * 100_000
        return income

    def _generate_income_source_count(
        self,
        employment_codes: np.ndarray,
        monthly_income: np.ndarray
    ) -> np.ndarray:
        n = len(employment_codes)
        counts = np.ones(n, dtype=int)

        for i in range(n):
            emp_type = employment_codes[i]
            income = monthly_income[i]

            # Base probabilities for multiple income sources
            if emp_type == "that_nghiep":
                prob_multiple = 0.1
            elif emp_type in ["tu_kinh_doanh", "freelancer"]:
                prob_multiple = 0.6
            elif emp_type == "huu_tri":
                prob_multiple = 0.3
            else:
                prob_multiple = 0.25

            # Higher income = more likely to have multiple sources
            if income > 30_000_000:
                prob_multiple *= 1.5
            elif income > 50_000_000:
                prob_multiple *= 2.0

            prob_multiple = min(prob_multiple, 0.8)

            if self.rng.random() < prob_multiple:
                # 2-4 income sources
                counts[i] = int(self.rng.choice([2, 2, 2, 3, 3, 4]))

        return counts

    def _generate_monthly_expenses(
        self,
        monthly_income: np.ndarray,
        ages: np.ndarray,
        marital_status_codes: np.ndarray,
        province_codes: np.ndarray
    ) -> np.ndarray:
        n = len(monthly_income)
        expenses = np.zeros(n)

        for i in range(n):
            income = monthly_income[i]
            age = ages[i]
            marital = marital_status_codes[i]
            tier = self._get_province_tier(province_codes[i])

            # Base expense ratio (expenses as % of income)
            if tier == "tier1":
                base_ratio = 0.75  # Higher living costs
            elif tier == "tier2":
                base_ratio = 0.70
            else:
                base_ratio = 0.65

            # Family status adjustment
            if marital == "da_ket_hon":
                family_mult = 1.2  # Family expenses
            elif marital == "doc_than" and age < 30:
                family_mult = 0.85  # Young singles spend less
            else:
                family_mult = 1.0

            # Age adjustment
            if age < 25:
                age_mult = 0.8
            elif age > 50:
                age_mult = 0.9  # Lower expenses in older age
            else:
                age_mult = 1.0

            # Calculate expense ratio with some randomness
            expense_ratio = base_ratio * family_mult * age_mult
            expense_ratio *= self.rng.uniform(0.8, 1.2)  # Add variance
            expense_ratio = min(expense_ratio, 0.95)  # Cap at 95%

            expenses[i] = income * expense_ratio

        # Round to nearest 100,000 VND
        expenses = np.round(expenses / 100_000) * 100_000
        return expenses

    def _generate_savings(
        self,
        monthly_income: np.ndarray,
        monthly_expenses: np.ndarray,
        ages: np.ndarray,
        employment_codes: np.ndarray
    ) -> np.ndarray:
        n = len(monthly_income)
        savings = np.zeros(n)

        for i in range(n):
            income = monthly_income[i]
            expenses = monthly_expenses[i]
            age = ages[i]
            emp_type = employment_codes[i]

            # Monthly surplus
            monthly_surplus = income - expenses

            # Savings potential based on employment type
            if emp_type == "that_nghiep":
                savings_mult = 0.1
            elif emp_type == "huu_tri":
                savings_mult = 1.5  # Accumulated savings
            elif emp_type == "cong_chuc":
                savings_mult = 1.2  # Stable income, good savers
            elif emp_type in ["tu_kinh_doanh"]:
                savings_mult = 1.3  # Business owners
            else:
                savings_mult = 1.0

            # Age factor - older people have more accumulated savings
            if age < 25:
                age_months = (age - 18) * 12 * 0.3
            elif age < 35:
                age_months = (age - 18) * 12 * 0.5
            elif age < 45:
                age_months = (age - 18) * 12 * 0.7
            else:
                age_months = (age - 18) * 12 * 0.8

            # Base savings estimate
            base_savings = max(0, monthly_surplus * age_months * savings_mult * 0.3)

            # Add randomness
            if base_savings > 0:
                savings[i] = truncated_lognormal(
                    self.rng,
                    base_savings,
                    base_savings * 0.5,
                    0,
                    base_savings * 5,
                    1
                )[0]
            else:
                # Some people have no savings
                if self.rng.random() < 0.3:
                    savings[i] = self.rng.uniform(0, income * 2)
                else:
                    savings[i] = 0

        # Round to nearest million VND
        savings = np.round(savings / 1_000_000) * 1_000_000
        return savings

    def _generate_existing_debt(
        self,
        monthly_income: np.ndarray,
        ages: np.ndarray,
        property_ownership_codes: np.ndarray
    ) -> np.ndarray:
        n = len(monthly_income)
        debt = np.zeros(n)

        for i in range(n):
            income = monthly_income[i]
            age = ages[i]
            property_status = property_ownership_codes[i]

            # Probability of having debt
            if property_status == "so_huu":
                # Homeowners often have mortgages
                prob_debt = 0.6
                max_debt_months = 240  # 20 years of income
            elif property_status == "thue":
                prob_debt = 0.4
                max_debt_months = 48
            else:
                prob_debt = 0.3
                max_debt_months = 24

            # Age adjustments
            if age < 25:
                prob_debt *= 0.5
                max_debt_months *= 0.3
            elif 30 <= age <= 45:
                prob_debt *= 1.2  # Peak borrowing years
            elif age > 55:
                prob_debt *= 0.7

            if self.rng.random() < prob_debt:
                # Generate debt amount
                mean_debt = income * max_debt_months * 0.3
                debt[i] = truncated_lognormal(
                    self.rng,
                    mean_debt,
                    mean_debt * 0.6,
                    income,  # Minimum 1 month income
                    income * max_debt_months,
                    1
                )[0]

        # Round to nearest million VND
        debt = np.round(debt / 1_000_000) * 1_000_000
        return debt

    def _generate_property_ownership(
        self,
        ages: np.ndarray,
        monthly_income: np.ndarray,
        marital_status_codes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(ages)
        prop_codes = np.empty(n, dtype=object)
        prop_labels = np.empty(n, dtype=object)

        for i in range(n):
            age = ages[i]
            income = monthly_income[i]
            marital = marital_status_codes[i]

            # Base distribution
            distribution = {
                "so_huu": 0.35,
                "thue": 0.30,
                "o_cung_gia_dinh": 0.30,
                "nha_cong_ty": 0.03,
                "ky_tuc_xa": 0.02,
            }

            # Age adjustments
            if age < 25:
                distribution["so_huu"] *= 0.1
                distribution["o_cung_gia_dinh"] *= 2.5
                distribution["ky_tuc_xa"] *= 3.0
            elif age < 35:
                distribution["so_huu"] *= 0.5
                distribution["thue"] *= 1.5
            elif age >= 45:
                distribution["so_huu"] *= 1.8
                distribution["o_cung_gia_dinh"] *= 0.5
                distribution["ky_tuc_xa"] *= 0.1

            # Income adjustments
            if income > 30_000_000:
                distribution["so_huu"] *= 1.5
                distribution["thue"] *= 0.8
            elif income < 10_000_000:
                distribution["so_huu"] *= 0.5
                distribution["o_cung_gia_dinh"] *= 1.3

            # Marital status
            if marital == "da_ket_hon":
                distribution["so_huu"] *= 1.4
                distribution["ky_tuc_xa"] *= 0.1
            elif marital == "doc_than":
                distribution["thue"] *= 1.2
                distribution["o_cung_gia_dinh"] *= 1.1

            # Normalize
            total = sum(distribution.values())
            distribution = {k: v / total for k, v in distribution.items()}

            prop_code = weighted_random_choice(
                self.rng,
                list(distribution.keys()),
                list(distribution.values())
            )
            prop_codes[i] = prop_code
            prop_labels[i] = PROPERTY_LABELS[prop_code]

        return prop_codes, prop_labels

    def _generate_has_insurance(
        self,
        employment_codes: np.ndarray,
        employer_codes: np.ndarray,
        monthly_income: np.ndarray
    ) -> np.ndarray:
        n = len(employment_codes)
        has_insurance = np.zeros(n, dtype=bool)

        for i in range(n):
            emp_type = employment_codes[i]
            employer = employer_codes[i]
            income = monthly_income[i]

            # Base probability
            if emp_type == "cong_chuc":
                prob = 0.95  # Almost all have insurance
            elif emp_type == "that_nghiep":
                prob = 0.15
            elif employer in ["nha_nuoc", "fdi", "lien_doanh"]:
                prob = 0.90
            elif employer == "tu_nhan":
                prob = 0.70
            elif emp_type in ["tu_kinh_doanh", "freelancer"]:
                prob = 0.50
            else:
                prob = 0.60

            # Income adjustment
            if income > 20_000_000:
                prob = min(prob * 1.2, 0.98)
            elif income < 8_000_000:
                prob *= 0.8

            has_insurance[i] = self.rng.random() < prob

        return has_insurance

    def _calculate_dti_ratio(
        self,
        existing_debt: np.ndarray,
        monthly_income: np.ndarray
    ) -> np.ndarray:
        # Estimate monthly debt payment (assuming 5-year term, ~10% interest)
        monthly_payment = existing_debt / 60 * 1.25  # Simplified calculation

        # Calculate DTI
        dti = np.where(
            monthly_income > 0,
            monthly_payment / monthly_income,
            0
        )

        # Cap at reasonable values
        dti = np.clip(dti, 0, 1.5)

        return np.round(dti, 4)

    def generate(
        self,
        demographic_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if demographic_df is None:
            raise ValueError(
                "demographic_df is required. Generate demographic data first "
                "using DemographicGenerator."
            )

        required_cols = [
            'customer_id', 'age', 'education_level_code',
            'marital_status_code', 'province_code'
        ]
        missing_cols = [c for c in required_cols if c not in demographic_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        n = len(demographic_df)

        # Extract demographic data
        customer_ids = demographic_df['customer_id'].values
        ages = demographic_df['age'].values
        education_codes = demographic_df['education_level_code'].values
        marital_codes = demographic_df['marital_status_code'].values
        province_codes = demographic_df['province_code'].values

        # Step 1: Generate employment
        employment_codes, employment_labels = self._generate_employment_type(
            ages, education_codes
        )
        employer_codes, employer_labels = self._generate_employer_type(
            employment_codes, province_codes
        )
        job_tenure = self._generate_job_tenure(ages, employment_codes)

        # Step 2: Generate income (depends on everything above)
        monthly_income = self._generate_monthly_income(
            province_codes, education_codes, employment_codes, ages
        )
        income_source_count = self._generate_income_source_count(
            employment_codes, monthly_income
        )

        # Step 3: Generate property ownership (needed for debt calculation)
        property_codes, property_labels = self._generate_property_ownership(
            ages, monthly_income, marital_codes
        )

        # Step 4: Generate expenses and savings (depend on income)
        monthly_expenses = self._generate_monthly_expenses(
            monthly_income, ages, marital_codes, province_codes
        )
        savings = self._generate_savings(
            monthly_income, monthly_expenses, ages, employment_codes
        )
        existing_debt = self._generate_existing_debt(
            monthly_income, ages, property_codes
        )

        # Step 5: Calculate DTI
        dti_ratio = self._calculate_dti_ratio(existing_debt, monthly_income)

        # Step 6: Generate insurance status
        has_insurance = self._generate_has_insurance(
            employment_codes, employer_codes, monthly_income
        )

        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'employment_type': employment_labels,
            'employment_type_code': employment_codes,
            'employer_type': employer_labels,
            'employer_type_code': employer_codes,
            'job_tenure_months': job_tenure,
            'monthly_income': monthly_income,
            'income_source_count': income_source_count,
            'monthly_expenses': monthly_expenses,
            'savings_amount': savings,
            'existing_debt': existing_debt,
            'dti_ratio': dti_ratio,
            'has_insurance': has_insurance,
            'property_ownership': property_labels,
            'property_ownership_code': property_codes,
        })

        self._generated_data = df
        return df

    def get_financial_summary(
        self,
        data: Optional[pd.DataFrame] = None,
        demographic_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        if data is None:
            data = self._generated_data

        if data is None:
            return {"error": "No data available"}

        summary = {
            "sample_size": len(data),
            "income_stats": {},
            "employment_distribution": {},
            "debt_stats": {},
            "property_distribution": {},
        }

        # Income statistics
        summary["income_stats"] = {
            "mean": float(data['monthly_income'].mean()),
            "median": float(data['monthly_income'].median()),
            "std": float(data['monthly_income'].std()),
            "min": float(data['monthly_income'].min()),
            "max": float(data['monthly_income'].max()),
            "quartiles": data['monthly_income'].quantile([0.25, 0.5, 0.75]).to_dict(),
        }

        # Employment distribution
        summary["employment_distribution"] = data['employment_type'].value_counts(
            normalize=True
        ).to_dict()

        # Debt statistics
        has_debt = data['existing_debt'] > 0
        summary["debt_stats"] = {
            "pct_with_debt": float(has_debt.mean()),
            "mean_debt_if_has": float(data.loc[has_debt, 'existing_debt'].mean()) if has_debt.any() else 0,
            "mean_dti": float(data['dti_ratio'].mean()),
            "median_dti": float(data['dti_ratio'].median()),
        }

        # Property distribution
        summary["property_distribution"] = data['property_ownership'].value_counts(
            normalize=True
        ).to_dict()

        # Insurance rate
        summary["insurance_rate"] = float(data['has_insurance'].mean())

        # Savings rate (% with savings > 0)
        summary["savings_rate"] = float((data['savings_amount'] > 0).mean())

        return summary


# MODULE EXPORTS

__all__ = [
    "FinancialGenerator",
    "EmploymentType",
    "EmployerType",
    "PropertyOwnership",
    "EMPLOYMENT_LABELS",
    "EMPLOYER_LABELS",
    "PROPERTY_LABELS",
    "INCOME_BY_TIER",
    "truncated_normal",
    "truncated_lognormal",
]
