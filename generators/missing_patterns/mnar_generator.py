from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config.settings import SyntheticDataConfig
from generators.base import BaseDataGenerator


# ENUMS AND CONSTANTS

class MissingMechanism(Enum):
    MCAR = "mcar"  # Missing Completely At Random
    MAR = "mar"    # Missing At Random (depends on observed data)
    MNAR = "mnar"  # Missing Not At Random (depends on unobserved/missing value)


class MissingCategory(Enum):
    INCOME = "income"
    EMPLOYMENT = "employment"
    CREDIT_HISTORY = "credit_history"
    DEBT = "debt"
    TELECOM = "telecom"
    DEMOGRAPHIC = "demographic"
    ASSET = "asset"


# MNAR RULE DATACLASS

@dataclass
class MNARRule:
    rule_id: str
    target_column: str
    mechanism: MissingMechanism
    missing_probability: float
    category: MissingCategory
    description: str
    description_en: str
    condition_column: Optional[str] = None
    condition_func: Optional[Callable[[Any], bool]] = None
    max_missing_rate: float = 0.30

    def __post_init__(self):
        if not 0 <= self.missing_probability <= 1:
            raise ValueError("missing_probability must be between 0 and 1")
        if not 0 <= self.max_missing_rate <= 1:
            raise ValueError("max_missing_rate must be between 0 and 1")
        if self.mechanism in [MissingMechanism.MAR, MissingMechanism.MNAR]:
            if self.condition_column is None and self.condition_func is None:
                raise ValueError(
                    f"MAR/MNAR rules require condition_column or condition_func"
                )


# MISSING REPORT DATACLASS

@dataclass
class MissingReport:
    column_missing_rates: Dict[str, float]
    mechanism_breakdown: Dict[str, Dict[str, float]]
    rule_application_counts: Dict[str, int]
    correlation_with_target: Dict[str, float]
    total_cells: int
    total_missing: int
    overall_missing_rate: float
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_missing_rates": self.column_missing_rates,
            "mechanism_breakdown": self.mechanism_breakdown,
            "rule_application_counts": self.rule_application_counts,
            "correlation_with_target": self.correlation_with_target,
            "total_cells": self.total_cells,
            "total_missing": self.total_missing,
            "overall_missing_rate": self.overall_missing_rate,
            "recommendations": self.recommendations,
        }

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "MISSING DATA REPORT",
            "=" * 60,
            f"\nOverall Statistics:",
            f"  Total cells: {self.total_cells:,}",
            f"  Total missing: {self.total_missing:,}",
            f"  Overall missing rate: {self.overall_missing_rate*100:.2f}%",
            f"\nMissing Rate by Column:",
        ]

        for col, rate in sorted(
            self.column_missing_rates.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if rate > 0:
                lines.append(f"  {col}: {rate*100:.2f}%")

        if self.recommendations:
            lines.append(f"\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


# VIETNAMESE CREDIT MNAR RULES

def _create_vietnamese_credit_mnar_rules() -> List[MNARRule]:
    rules = []

    # RULE 1: Income MNAR - Low income underreporting
    rules.append(MNARRule(
        rule_id="INCOME_LOW_MNAR",
        target_column="monthly_income",
        mechanism=MissingMechanism.MNAR,
        condition_column="monthly_income",
        condition_func=lambda x: x < 8_000_000,  # < 8 triệu
        missing_probability=0.30,
        max_missing_rate=0.25,
        category=MissingCategory.INCOME,
        description="Người thu nhập thấp (<8 triệu) có xu hướng không khai báo thu nhập đầy đủ",
        description_en="Low income earners (<8M VND) tend to underreport income",
    ))

    rules.append(MNARRule(
        rule_id="INCOME_HIGH_MNAR",
        target_column="monthly_income",
        mechanism=MissingMechanism.MNAR,
        condition_column="monthly_income",
        condition_func=lambda x: x > 50_000_000,  # > 50 triệu
        missing_probability=0.05,
        max_missing_rate=0.25,
        category=MissingCategory.INCOME,
        description="Người thu nhập cao (>50 triệu) thường khai báo đầy đủ",
        description_en="High income earners (>50M VND) usually report complete income",
    ))

    rules.append(MNARRule(
        rule_id="INCOME_SELF_EMPLOYED_MAR",
        target_column="monthly_income",
        mechanism=MissingMechanism.MAR,
        condition_column="employment_type_code",
        condition_func=lambda x: x in ["tu_kinh_doanh", "freelancer", "nghe_tu_do"],
        missing_probability=0.40,
        max_missing_rate=0.25,
        category=MissingCategory.INCOME,
        description="Người tự kinh doanh/freelancer thường thiếu chứng từ thu nhập",
        description_en="Self-employed/freelancers often lack income documentation",
    ))

    # RULE 2: Credit History MNAR - Bad credit missing
    rules.append(MNARRule(
        rule_id="CREDIT_BAD_MNAR",
        target_column="cic_score",
        mechanism=MissingMechanism.MNAR,
        condition_column="cic_score",
        condition_func=lambda x: x < 500,  # CIC score thấp
        missing_probability=0.25,
        max_missing_rate=0.20,
        category=MissingCategory.CREDIT_HISTORY,
        description="Người có nợ xấu (CIC<500) thường thiếu thông tin tín dụng",
        description_en="Bad credit customers (CIC<500) often have missing credit data",
    ))

    rules.append(MNARRule(
        rule_id="CREDIT_THIN_FILE_MAR",
        target_column="cic_grade",
        mechanism=MissingMechanism.MAR,
        condition_column="has_credit_history",
        condition_func=lambda x: not x,  # No credit history
        missing_probability=0.80,
        max_missing_rate=0.35,
        category=MissingCategory.CREDIT_HISTORY,
        description="Khách hàng thin-file không có dữ liệu CIC",
        description_en="Thin-file customers have no CIC data",
    ))

    rules.append(MNARRule(
        rule_id="CREDIT_DPD_MNAR",
        target_column="max_dpd_ever",
        mechanism=MissingMechanism.MNAR,
        condition_column="max_dpd_ever",
        condition_func=lambda x: x > 90,  # DPD > 90 days
        missing_probability=0.20,
        max_missing_rate=0.15,
        category=MissingCategory.CREDIT_HISTORY,
        description="Thông tin DPD cao có thể bị thiếu do khách hàng che giấu",
        description_en="High DPD information may be missing due to customer concealment",
    ))

    # RULE 3: Employment MNAR - Unstable job missing
    rules.append(MNARRule(
        rule_id="EMPLOYMENT_UNSTABLE_MAR",
        target_column="job_tenure_months",
        mechanism=MissingMechanism.MAR,
        condition_column="employment_type_code",
        condition_func=lambda x: x in ["that_nghiep", "lao_dong_pho_thong", "thoi_vu"],
        missing_probability=0.35,
        max_missing_rate=0.25,
        category=MissingCategory.EMPLOYMENT,
        description="Người thất nghiệp/việc không ổn định thiếu thông tin việc làm",
        description_en="Unemployed/unstable job holders lack employment information",
    ))

    rules.append(MNARRule(
        rule_id="EMPLOYMENT_SENSITIVE_MAR",
        target_column="employer_name",
        mechanism=MissingMechanism.MAR,
        condition_column="employer_type_code",
        condition_func=lambda x: x in ["ca_nhan", "ho_kinh_doanh", "khong_xac_dinh"],
        missing_probability=0.45,
        max_missing_rate=0.30,
        category=MissingCategory.EMPLOYMENT,
        description="Người làm việc cho cá nhân/hộ kinh doanh không khai báo chi tiết",
        description_en="Those working for individuals/households don't report details",
    ))

    rules.append(MNARRule(
        rule_id="EMPLOYMENT_TENURE_MNAR",
        target_column="job_tenure_months",
        mechanism=MissingMechanism.MNAR,
        condition_column="job_tenure_months",
        condition_func=lambda x: x < 6,  # < 6 tháng
        missing_probability=0.30,
        max_missing_rate=0.20,
        category=MissingCategory.EMPLOYMENT,
        description="Người mới đi làm (<6 tháng) có xu hướng không khai báo",
        description_en="New workers (<6 months) tend to not report tenure",
    ))

    # RULE 4: Debt MNAR - High debt underreporting
    rules.append(MNARRule(
        rule_id="DEBT_HIGH_MNAR",
        target_column="existing_debt",
        mechanism=MissingMechanism.MNAR,
        condition_column="existing_debt",
        condition_func=lambda x: x > 500_000_000,  # > 500 triệu
        missing_probability=0.30,
        max_missing_rate=0.20,
        category=MissingCategory.DEBT,
        description="Người nợ nhiều (>500 triệu) có xu hướng underreport",
        description_en="High debt customers (>500M) tend to underreport",
    ))

    rules.append(MNARRule(
        rule_id="DEBT_INFORMAL_MNAR",
        target_column="num_active_loans",
        mechanism=MissingMechanism.MNAR,
        condition_column="num_active_loans",
        condition_func=lambda x: x > 3,  # Nhiều khoản vay
        missing_probability=0.25,
        max_missing_rate=0.20,
        category=MissingCategory.DEBT,
        description="Người có nhiều khoản vay không khai báo đầy đủ",
        description_en="Customers with many loans don't report all",
    ))

    rules.append(MNARRule(
        rule_id="DEBT_DTI_MNAR",
        target_column="dti_ratio",
        mechanism=MissingMechanism.MNAR,
        condition_column="dti_ratio",
        condition_func=lambda x: x > 0.6,  # DTI > 60%
        missing_probability=0.25,
        max_missing_rate=0.20,
        category=MissingCategory.DEBT,
        description="Người có DTI cao (>60%) thường thiếu dữ liệu",
        description_en="High DTI (>60%) customers often have missing data",
    ))

    # RULE 5: Age-related MAR
    rules.append(MNARRule(
        rule_id="AGE_ELDERLY_TELECOM_MAR",
        target_column="monthly_arpu",
        mechanism=MissingMechanism.MAR,
        condition_column="age",
        condition_func=lambda x: x > 55,
        missing_probability=0.20,
        max_missing_rate=0.15,
        category=MissingCategory.TELECOM,
        description="Người cao tuổi (>55) thiếu digital footprint",
        description_en="Elderly (>55) lack digital footprint",
    ))

    rules.append(MNARRule(
        rule_id="AGE_YOUNG_CREDIT_MAR",
        target_column="credit_history_months",
        mechanism=MissingMechanism.MAR,
        condition_column="age",
        condition_func=lambda x: x < 25,
        missing_probability=0.50,
        max_missing_rate=0.40,
        category=MissingCategory.CREDIT_HISTORY,
        description="Người trẻ (<25 tuổi) thiếu lịch sử tín dụng",
        description_en="Young people (<25) lack credit history",
    ))

    rules.append(MNARRule(
        rule_id="AGE_ELDERLY_DIGITAL_MAR",
        target_column="data_usage_gb",
        mechanism=MissingMechanism.MAR,
        condition_column="age",
        condition_func=lambda x: x > 60,
        missing_probability=0.35,
        max_missing_rate=0.25,
        category=MissingCategory.TELECOM,
        description="Người già (>60) ít sử dụng dịch vụ số",
        description_en="Elderly (>60) use fewer digital services",
    ))

    # RULE 6: Geographic MAR
    rules.append(MNARRule(
        rule_id="GEO_RURAL_MAR",
        target_column="cic_score",
        mechanism=MissingMechanism.MAR,
        condition_column="is_urban",
        condition_func=lambda x: not x,  # Rural
        missing_probability=0.25,
        max_missing_rate=0.20,
        category=MissingCategory.CREDIT_HISTORY,
        description="Vùng nông thôn thiếu kết nối CIC",
        description_en="Rural areas lack CIC connectivity",
    ))

    rules.append(MNARRule(
        rule_id="GEO_SMALL_PROVINCE_MAR",
        target_column="bank_account_count",
        mechanism=MissingMechanism.MAR,
        condition_column="province_tier",
        condition_func=lambda x: x == 3,  # Tier 3 provinces
        missing_probability=0.20,
        max_missing_rate=0.15,
        category=MissingCategory.ASSET,
        description="Tỉnh nhỏ có ít dữ liệu ngân hàng",
        description_en="Small provinces have less banking data",
    ))

    # RULE 7: Asset MNAR
    rules.append(MNARRule(
        rule_id="ASSET_NO_PROPERTY_MNAR",
        target_column="property_value",
        mechanism=MissingMechanism.MNAR,
        condition_column="property_ownership_code",
        condition_func=lambda x: x in ["khong_co", "thue", "o_nho"],
        missing_probability=0.60,
        max_missing_rate=0.50,
        category=MissingCategory.ASSET,
        description="Người không có BĐS không khai báo tài sản",
        description_en="Non-property owners don't report assets",
    ))

    rules.append(MNARRule(
        rule_id="ASSET_HIGH_VALUE_MNAR",
        target_column="property_value",
        mechanism=MissingMechanism.MNAR,
        condition_column="property_value",
        condition_func=lambda x: x > 5_000_000_000,  # > 5 tỷ
        missing_probability=0.15,
        max_missing_rate=0.10,
        category=MissingCategory.ASSET,
        description="Người có BĐS giá trị cao đôi khi không khai báo đầy đủ",
        description_en="High-value property owners sometimes underreport",
    ))

    return rules


def _create_telecom_mnar_rules() -> List[MNARRule]:
    rules = []

    rules.append(MNARRule(
        rule_id="TELECOM_PREPAID_MAR",
        target_column="bill_amount",
        mechanism=MissingMechanism.MAR,
        condition_column="contract_type_code",
        condition_func=lambda x: x == "tra_truoc",
        missing_probability=0.40,
        max_missing_rate=0.35,
        category=MissingCategory.TELECOM,
        description="Khách trả trước thiếu dữ liệu bill",
        description_en="Prepaid customers lack billing data",
    ))

    rules.append(MNARRule(
        rule_id="TELECOM_LOW_ARPU_MNAR",
        target_column="monthly_arpu",
        mechanism=MissingMechanism.MNAR,
        condition_column="monthly_arpu",
        condition_func=lambda x: x < 50_000,
        missing_probability=0.30,
        max_missing_rate=0.25,
        category=MissingCategory.TELECOM,
        description="Khách ARPU thấp (<50k) có dữ liệu không đầy đủ",
        description_en="Low ARPU (<50k) customers have incomplete data",
    ))

    rules.append(MNARRule(
        rule_id="TELECOM_CHURN_MAR",
        target_column="tenure_months",
        mechanism=MissingMechanism.MAR,
        condition_column="tenure_months",
        condition_func=lambda x: x < 3,
        missing_probability=0.35,
        max_missing_rate=0.30,
        category=MissingCategory.TELECOM,
        description="Khách mới (<3 tháng) thiếu dữ liệu hành vi",
        description_en="New customers (<3 months) lack behavior data",
    ))

    rules.append(MNARRule(
        rule_id="TELECOM_LATE_PAYER_MNAR",
        target_column="days_to_payment",
        mechanism=MissingMechanism.MNAR,
        condition_column="days_to_payment",
        condition_func=lambda x: x > 30,
        missing_probability=0.25,
        max_missing_rate=0.20,
        category=MissingCategory.TELECOM,
        description="Người trả chậm (>30 ngày) có dữ liệu thiếu",
        description_en="Late payers (>30 days) have missing data",
    ))

    return rules


def _create_thin_file_mnar_rules() -> List[MNARRule]:
    rules = []

    rules.append(MNARRule(
        rule_id="THIN_NO_CREDIT_HISTORY",
        target_column="cic_score",
        mechanism=MissingMechanism.MAR,
        condition_column="has_credit_history",
        condition_func=lambda x: not x,
        missing_probability=0.95,
        max_missing_rate=0.95,
        category=MissingCategory.CREDIT_HISTORY,
        description="Thin-file không có CIC score",
        description_en="Thin-file customers have no CIC score",
    ))

    rules.append(MNARRule(
        rule_id="THIN_NO_LOAN_HISTORY",
        target_column="num_active_loans",
        mechanism=MissingMechanism.MAR,
        condition_column="has_credit_history",
        condition_func=lambda x: not x,
        missing_probability=0.90,
        max_missing_rate=0.90,
        category=MissingCategory.CREDIT_HISTORY,
        description="Thin-file không có lịch sử vay",
        description_en="Thin-file customers have no loan history",
    ))

    rules.append(MNARRule(
        rule_id="THIN_NO_DPD_HISTORY",
        target_column="max_dpd_ever",
        mechanism=MissingMechanism.MAR,
        condition_column="has_credit_history",
        condition_func=lambda x: not x,
        missing_probability=0.95,
        max_missing_rate=0.95,
        category=MissingCategory.CREDIT_HISTORY,
        description="Thin-file không có lịch sử DPD",
        description_en="Thin-file customers have no DPD history",
    ))

    rules.append(MNARRule(
        rule_id="THIN_NO_CREDIT_LIMIT",
        target_column="total_credit_limit",
        mechanism=MissingMechanism.MAR,
        condition_column="has_credit_history",
        condition_func=lambda x: not x,
        missing_probability=0.90,
        max_missing_rate=0.90,
        category=MissingCategory.CREDIT_HISTORY,
        description="Thin-file không có credit limit",
        description_en="Thin-file customers have no credit limit",
    ))

    return rules


# Predefined rule sets
VIETNAMESE_CREDIT_MNAR_RULES = _create_vietnamese_credit_mnar_rules()
TELECOM_MNAR_RULES = _create_telecom_mnar_rules()
THIN_FILE_MNAR_RULES = _create_thin_file_mnar_rules()

# Combined rule set
ALL_MNAR_RULES = VIETNAMESE_CREDIT_MNAR_RULES + TELECOM_MNAR_RULES + THIN_FILE_MNAR_RULES


# MNAR PATTERN GENERATOR

class MNARPatternGenerator(BaseDataGenerator):
    def __init__(
        self,
        config: SyntheticDataConfig,
        seed: Optional[int] = None,
        global_max_missing: float = 0.30
    ) -> None:
        super().__init__(config, seed)
        self.global_max_missing = global_max_missing
        self._missing_masks: Dict[str, np.ndarray] = {}
        self._applied_rules: List[str] = []

    # MISSING MECHANISM METHODS

    def apply_mcar(
        self,
        df: pd.DataFrame,
        column: str,
        missing_rate: float,
        max_missing: Optional[float] = None
    ) -> pd.DataFrame:
        if column not in df.columns:
            return df

        df = df.copy()
        n = len(df)
        max_rate = max_missing or self.global_max_missing

        # Calculate current missing rate
        current_missing = df[column].isna().mean()
        remaining_capacity = max(0, max_rate - current_missing)

        # Adjust rate if needed
        effective_rate = min(missing_rate, remaining_capacity)

        if effective_rate <= 0:
            return df

        # Generate random mask
        mask = self.rng.random(n) < effective_rate

        # Only apply to non-missing values
        mask = mask & df[column].notna()

        df.loc[mask, column] = np.nan

        # Store mask
        self._missing_masks[f"{column}_mcar"] = mask

        return df

    def apply_mar(
        self,
        df: pd.DataFrame,
        target_column: str,
        condition_column: str,
        condition_func: Callable[[Any], bool],
        missing_probability: float,
        max_missing: Optional[float] = None
    ) -> pd.DataFrame:
        if target_column not in df.columns or condition_column not in df.columns:
            return df

        df = df.copy()
        n = len(df)
        max_rate = max_missing or self.global_max_missing

        # Calculate current missing rate
        current_missing = df[target_column].isna().mean()
        remaining_capacity = max(0, max_rate - current_missing)

        if remaining_capacity <= 0:
            return df

        # Apply condition function
        try:
            condition_mask = df[condition_column].apply(
                lambda x: condition_func(x) if pd.notna(x) else False
            ).values
        except Exception:
            return df

        # Generate random mask for those meeting condition
        random_mask = self.rng.random(n) < missing_probability

        # Combine: condition AND random AND not already missing
        mask = condition_mask & random_mask & df[target_column].notna().values

        # Check if this would exceed max missing rate
        new_missing_rate = mask.sum() / n
        if current_missing + new_missing_rate > max_rate:
            # Reduce number of missing values
            excess = int((current_missing + new_missing_rate - max_rate) * n)
            true_indices = np.where(mask)[0]
            if len(true_indices) > excess:
                indices_to_flip = self.rng.choice(
                    true_indices, size=excess, replace=False
                )
                mask[indices_to_flip] = False

        df.loc[mask, target_column] = np.nan

        # Store mask
        self._missing_masks[f"{target_column}_mar_{condition_column}"] = mask

        return df

    def apply_mnar(
        self,
        df: pd.DataFrame,
        column: str,
        condition_func: Callable[[Any], bool],
        missing_probability: float,
        max_missing: Optional[float] = None
    ) -> pd.DataFrame:
        if column not in df.columns:
            return df

        df = df.copy()
        n = len(df)
        max_rate = max_missing or self.global_max_missing

        # Calculate current missing rate
        current_missing = df[column].isna().mean()
        remaining_capacity = max(0, max_rate - current_missing)

        if remaining_capacity <= 0:
            return df

        # Apply condition function to the column itself
        try:
            condition_mask = df[column].apply(
                lambda x: condition_func(x) if pd.notna(x) else False
            ).values
        except Exception:
            return df

        # Generate random mask for those meeting condition
        random_mask = self.rng.random(n) < missing_probability

        # Combine: condition AND random AND not already missing
        mask = condition_mask & random_mask & df[column].notna().values

        # Check if this would exceed max missing rate
        new_missing_rate = mask.sum() / n
        if current_missing + new_missing_rate > max_rate:
            # Reduce number of missing values
            excess = int((current_missing + new_missing_rate - max_rate) * n)
            true_indices = np.where(mask)[0]
            if len(true_indices) > excess:
                indices_to_flip = self.rng.choice(
                    true_indices, size=excess, replace=False
                )
                mask[indices_to_flip] = False

        df.loc[mask, column] = np.nan

        # Store mask
        self._missing_masks[f"{column}_mnar"] = mask

        return df

    def apply_rule(
        self,
        df: pd.DataFrame,
        rule: MNARRule
    ) -> pd.DataFrame:
        if rule.target_column not in df.columns:
            return df

        if rule.mechanism == MissingMechanism.MCAR:
            df = self.apply_mcar(
                df,
                rule.target_column,
                rule.missing_probability,
                rule.max_missing_rate
            )

        elif rule.mechanism == MissingMechanism.MAR:
            if rule.condition_column and rule.condition_func:
                df = self.apply_mar(
                    df,
                    rule.target_column,
                    rule.condition_column,
                    rule.condition_func,
                    rule.missing_probability,
                    rule.max_missing_rate
                )

        elif rule.mechanism == MissingMechanism.MNAR:
            if rule.condition_func:
                df = self.apply_mnar(
                    df,
                    rule.target_column,
                    rule.condition_func,
                    rule.missing_probability,
                    rule.max_missing_rate
                )

        self._applied_rules.append(rule.rule_id)
        return df

    def apply_all_mnar_rules(
        self,
        df: pd.DataFrame,
        rules: List[MNARRule],
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, MissingReport]:
        df = df.copy()
        self._missing_masks = {}
        self._applied_rules = []
        rule_counts: Dict[str, int] = {}

        # Apply each rule
        for rule in rules:
            if rule.target_column in df.columns:
                before_missing = df[rule.target_column].isna().sum()
                df = self.apply_rule(df, rule)
                after_missing = df[rule.target_column].isna().sum()
                rule_counts[rule.rule_id] = after_missing - before_missing

        # Generate report
        report = self._generate_missing_report(df, rules, rule_counts, target_column)

        return df, report

    # MISSING INDICATORS

    def generate_missing_indicator(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        df = df.copy()
        columns = columns or df.columns.tolist()

        for col in columns:
            if col in df.columns:
                df[f"{col}_missing"] = df[col].isna().astype(int)

        return df

    def get_missing_statistics(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        stats = {}

        for col in df.columns:
            missing_count = df[col].isna().sum()
            total_count = len(df)
            missing_rate = missing_count / total_count if total_count > 0 else 0

            stats[col] = {
                "missing_count": int(missing_count),
                "total_count": int(total_count),
                "missing_rate": float(missing_rate),
                "missing_pct": f"{missing_rate * 100:.2f}%",
                "dtype": str(df[col].dtype),
            }

        return stats

    # REPORT GENERATION

    def _generate_missing_report(
        self,
        df: pd.DataFrame,
        rules: List[MNARRule],
        rule_counts: Dict[str, int],
        target_column: Optional[str] = None
    ) -> MissingReport:
        # Column missing rates
        column_missing_rates = {}
        for col in df.columns:
            column_missing_rates[col] = float(df[col].isna().mean())

        # Mechanism breakdown per column
        mechanism_breakdown: Dict[str, Dict[str, float]] = {}
        for rule in rules:
            col = rule.target_column
            if col not in mechanism_breakdown:
                mechanism_breakdown[col] = {
                    "mcar": 0.0, "mar": 0.0, "mnar": 0.0
                }
            # This is approximate - actual breakdown requires tracking
            mechanism_breakdown[col][rule.mechanism.value] += rule.missing_probability * 0.5

        # Correlation with target
        correlation_with_target = {}
        if target_column and target_column in df.columns:
            for col in df.columns:
                if col != target_column and df[col].isna().any():
                    missing_indicator = df[col].isna().astype(int)
                    if df[target_column].dtype in ['int64', 'float64', 'bool']:
                        target_values = df[target_column].fillna(0)
                        if missing_indicator.std() > 0 and target_values.std() > 0:
                            corr = np.corrcoef(missing_indicator, target_values)[0, 1]
                            if not np.isnan(corr):
                                correlation_with_target[col] = float(corr)

        # Total statistics
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isna().sum().sum()
        overall_missing_rate = total_missing / total_cells if total_cells > 0 else 0

        # Recommendations
        recommendations = []
        for col, rate in column_missing_rates.items():
            if rate > 0.3:
                recommendations.append(
                    f"Column '{col}' has high missing rate ({rate*100:.1f}%). "
                    f"Consider imputation or feature engineering."
                )
            elif rate > 0.15:
                recommendations.append(
                    f"Column '{col}' has moderate missing rate ({rate*100:.1f}%). "
                    f"Review data collection process."
                )

        if overall_missing_rate > 0.2:
            recommendations.append(
                f"Overall missing rate is high ({overall_missing_rate*100:.1f}%). "
                f"Consider data quality improvements."
            )

        return MissingReport(
            column_missing_rates=column_missing_rates,
            mechanism_breakdown=mechanism_breakdown,
            rule_application_counts=rule_counts,
            correlation_with_target=correlation_with_target,
            total_cells=int(total_cells),
            total_missing=int(total_missing),
            overall_missing_rate=float(overall_missing_rate),
            recommendations=recommendations,
        )

    # MAIN GENERATE METHOD (required by BaseDataGenerator)

    def generate(
        self,
        df: pd.DataFrame,
        rules: Optional[List[MNARRule]] = None
    ) -> Tuple[pd.DataFrame, MissingReport]:
        rules = rules or VIETNAMESE_CREDIT_MNAR_RULES
        return self.apply_all_mnar_rules(df, rules)


# MODULE EXPORTS

__all__ = [
    # Enums
    "MissingMechanism",
    "MissingCategory",
    # Dataclasses
    "MNARRule",
    "MissingReport",
    # Generator
    "MNARPatternGenerator",
    # Predefined rules
    "VIETNAMESE_CREDIT_MNAR_RULES",
    "TELECOM_MNAR_RULES",
    "THIN_FILE_MNAR_RULES",
    "ALL_MNAR_RULES",
]
