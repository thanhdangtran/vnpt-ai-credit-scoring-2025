from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, '/home/thanhdang/Desktop/vnpt-ai')

from config.settings import SyntheticDataConfig
from generators.base import BaseDataGenerator, CorrelationMixin, TimeSeriesMixin
from generators.financial import truncated_normal, truncated_lognormal


# ENUMS AND CONSTANTS

class SeasonalityType(Enum):
    NONE = "none"
    MONTHLY = "monthly"          # Within-month patterns
    QUARTERLY = "quarterly"      # Q1-Q4 patterns
    ANNUAL = "annual"            # Year-round patterns
    VIETNAMESE = "vietnamese"    # Vietnamese calendar (Tet, holidays)


class TrendType(Enum):
    NONE = "none"
    LINEAR_UP = "linear_up"
    LINEAR_DOWN = "linear_down"
    EXPONENTIAL_UP = "exponential_up"
    EXPONENTIAL_DOWN = "exponential_down"
    CYCLICAL = "cyclical"
    RANDOM_WALK = "random_walk"


class OutputFormat(Enum):
    WIDE = "wide"    # [customer_id, month_id, feature1, feature2, ...]
    LONG = "long"    # [customer_id, month_id, feature_name, value]


# Vietnamese holidays and special periods
VIETNAMESE_HOLIDAYS: Dict[str, List[Tuple[int, int]]] = {
    # (month, day) tuples for fixed holidays
    "tet_duong_lich": [(1, 1)],                    # New Year
    "giai_phong": [(4, 30)],                       # Liberation Day
    "quoc_te_lao_dong": [(5, 1)],                  # Labor Day
    "quoc_khanh": [(9, 2)],                        # National Day
}

# Tet Nguyen Dan typically falls in Jan-Feb (lunar calendar)
# These are approximate Gregorian months affected
TET_MONTHS: Dict[int, List[int]] = {
    2020: [1],       # Tet in late Jan 2020
    2021: [2],       # Tet in Feb 2021
    2022: [2],       # Tet in Feb 2022
    2023: [1],       # Tet in late Jan 2023
    2024: [2],       # Tet in Feb 2024
    2025: [1, 2],    # Tet in late Jan 2025
}

# Spending multipliers by month (1.0 = baseline)
MONTHLY_SPENDING_MULTIPLIER: Dict[int, float] = {
    1: 1.4,    # Tet preparation & spending
    2: 1.3,    # Post-Tet, some years Tet falls here
    3: 0.9,    # Post-Tet recovery
    4: 0.95,
    5: 1.0,
    6: 1.1,    # Summer/vacation
    7: 1.15,   # Summer peak
    8: 1.1,    # Back to school
    9: 1.0,
    10: 1.0,
    11: 1.05,  # Year-end shopping begins
    12: 1.25,  # Year-end, Christmas shopping
}


# VIETNAMESE CALENDAR HELPER

@dataclass
class VietnameseCalendar:
    base_year: int = 2023

    def get_tet_effect(self, year: int, month: int) -> float:
        tet_months = TET_MONTHS.get(year, [1, 2])

        if month in tet_months:
            return 1.5  # 50% increase during Tet
        elif month == tet_months[0] - 1 if tet_months[0] > 1 else 12:
            return 1.2  # Pre-Tet preparation
        elif month == (tet_months[-1] % 12) + 1:
            return 0.8  # Post-Tet recovery
        return 1.0

    def get_13th_month_salary(self, month: int) -> bool:
        # Most companies pay 13th month salary in December or before Tet
        return month in [12, 1]

    def get_bonus_probability(self, month: int) -> float:
        bonus_probs = {
            1: 0.4,   # Tet bonus
            2: 0.2,   # Tet bonus (some companies)
            3: 0.05,
            4: 0.1,   # Q1 bonus
            5: 0.05,
            6: 0.1,   # Mid-year bonus
            7: 0.15,  # Q2 bonus
            8: 0.05,
            9: 0.1,   # Q3 bonus
            10: 0.1,  # Q3 bonus
            11: 0.15,
            12: 0.5,  # Year-end bonus
        }
        return bonus_probs.get(month, 0.05)

    def get_spending_multiplier(self, year: int, month: int) -> float:
        base_mult = MONTHLY_SPENDING_MULTIPLIER.get(month, 1.0)
        tet_mult = self.get_tet_effect(year, month)

        # Combine multiplicatively with dampening
        return base_mult * (0.5 + 0.5 * tet_mult)


# CUSTOMER PROFILE FOR TIME SERIES

@dataclass
class CustomerTransactionProfile:
    customer_id: str
    monthly_income: float
    monthly_expenses: float
    employment_type: str
    age: int
    has_credit_history: bool
    cic_score: int
    existing_debt: float
    savings_amount: float
    is_risky: bool = False

    # Derived attributes
    income_volatility: float = 0.1
    expense_volatility: float = 0.15
    salary_regularity: float = 0.95  # How regular is salary (0-1)

    def __post_init__(self):
        # Income volatility based on employment type
        if self.employment_type in ["cong_chuc", "nhan_vien"]:
            self.income_volatility = 0.05
            self.salary_regularity = 0.98
        elif self.employment_type in ["tu_kinh_doanh"]:
            self.income_volatility = 0.3
            self.salary_regularity = 0.7
        elif self.employment_type in ["freelancer", "nghe_tu_do"]:
            self.income_volatility = 0.4
            self.salary_regularity = 0.5
        elif self.employment_type in ["that_nghiep"]:
            self.income_volatility = 0.5
            self.salary_regularity = 0.1

        # Risk flag based on CIC score
        if self.has_credit_history:
            self.is_risky = self.cic_score < 550
        else:
            self.is_risky = self.monthly_income < 8_000_000


# TRANSACTION SERIES GENERATOR

class TransactionSeriesGenerator(BaseDataGenerator, CorrelationMixin, TimeSeriesMixin):
    def __init__(
        self,
        config: SyntheticDataConfig,
        seed: Optional[int] = None,
        n_months: int = 24,
        start_date: Optional[date] = None,
        output_format: OutputFormat = OutputFormat.WIDE
    ) -> None:
        super().__init__(config, seed)
        self.n_months = n_months
        self.output_format = output_format
        self.calendar = VietnameseCalendar()

        # Set start date
        if start_date is None:
            today = date.today()
            # Go back n_months from today
            year = today.year - (n_months // 12)
            month = today.month - (n_months % 12)
            if month <= 0:
                year -= 1
                month += 12
            self.start_date = date(year, month, 1)
        else:
            self.start_date = start_date

        # Generate month list
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

    def _create_customer_profile(
        self,
        row: pd.Series,
        financial_row: pd.Series,
        credit_row: Optional[pd.Series] = None
    ) -> CustomerTransactionProfile:
        has_credit = False
        cic_score = 0
        if credit_row is not None:
            has_credit = credit_row.get('has_credit_history', False)
            cic_score = credit_row.get('cic_score', 0) if has_credit else 0

        return CustomerTransactionProfile(
            customer_id=row['customer_id'],
            monthly_income=financial_row['monthly_income'],
            monthly_expenses=financial_row['monthly_expenses'],
            employment_type=financial_row['employment_type_code'],
            age=row['age'],
            has_credit_history=has_credit,
            cic_score=cic_score,
            existing_debt=financial_row['existing_debt'],
            savings_amount=financial_row['savings_amount'],
        )

    def add_seasonality(
        self,
        series: np.ndarray,
        seasonality_type: SeasonalityType,
        amplitude: float = 0.2,
        year_months: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        n = len(series)

        if seasonality_type == SeasonalityType.NONE:
            return series

        elif seasonality_type == SeasonalityType.MONTHLY:
            # Monthly cycle within a year
            seasonal = amplitude * np.sin(2 * np.pi * np.arange(n) / 12)
            return series * (1 + seasonal)

        elif seasonality_type == SeasonalityType.QUARTERLY:
            # Quarterly patterns
            seasonal = amplitude * np.sin(2 * np.pi * np.arange(n) / 3)
            return series * (1 + seasonal)

        elif seasonality_type == SeasonalityType.ANNUAL:
            # Full year cycle
            seasonal = amplitude * np.sin(2 * np.pi * np.arange(n) / 12)
            return series * (1 + seasonal)

        elif seasonality_type == SeasonalityType.VIETNAMESE:
            # Vietnamese calendar effects
            if year_months is None:
                return series

            multipliers = np.array([
                self.calendar.get_spending_multiplier(y, m)
                for y, m in year_months
            ])
            # Normalize and apply amplitude
            multipliers = 1 + amplitude * (multipliers - 1)
            return series * multipliers

        return series

    def add_trend(
        self,
        series: np.ndarray,
        trend_type: TrendType,
        strength: float = 0.02
    ) -> np.ndarray:
        n = len(series)
        t = np.arange(n)

        if trend_type == TrendType.NONE:
            return series

        elif trend_type == TrendType.LINEAR_UP:
            trend = 1 + strength * t
            return series * trend

        elif trend_type == TrendType.LINEAR_DOWN:
            trend = 1 - strength * t / (n * 2)  # Cap at 50% decline
            trend = np.maximum(trend, 0.5)
            return series * trend

        elif trend_type == TrendType.EXPONENTIAL_UP:
            trend = np.exp(strength * t / 12)  # Annualized
            return series * trend

        elif trend_type == TrendType.EXPONENTIAL_DOWN:
            trend = np.exp(-strength * t / 12)
            trend = np.maximum(trend, 0.5)
            return series * trend

        elif trend_type == TrendType.CYCLICAL:
            # Business cycle (3-5 year period)
            cycle = 1 + strength * np.sin(2 * np.pi * t / 48)
            return series * cycle

        elif trend_type == TrendType.RANDOM_WALK:
            # Cumulative random walk
            steps = self.rng.normal(0, strength, n)
            walk = np.cumsum(steps)
            walk = walk - walk.mean()  # Center
            return series * (1 + walk)

        return series

    def add_noise(
        self,
        series: np.ndarray,
        noise_level: float = 0.1,
        noise_type: str = "gaussian"
    ) -> np.ndarray:
        n = len(series)
        mean_val = np.mean(series[series > 0]) if np.any(series > 0) else 1.0

        if noise_type == "gaussian":
            noise = self.rng.normal(1, noise_level, n)
        elif noise_type == "uniform":
            noise = self.rng.uniform(1 - noise_level, 1 + noise_level, n)
        elif noise_type == "lognormal":
            sigma = np.sqrt(np.log(1 + noise_level**2))
            mu = -sigma**2 / 2
            noise = self.rng.lognormal(mu, sigma, n)
        else:
            noise = np.ones(n)

        return series * noise

    def introduce_anomalies(
        self,
        series: np.ndarray,
        anomaly_rate: float = 0.05,
        anomaly_magnitude: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(series)
        result = series.copy()
        anomaly_flags = np.zeros(n, dtype=bool)

        for i in range(n):
            if self.rng.random() < anomaly_rate:
                anomaly_flags[i] = True
                # Random direction
                direction = self.rng.choice([-1, 1])
                magnitude = self.rng.uniform(1.5, anomaly_magnitude)
                if direction > 0:
                    result[i] *= magnitude
                else:
                    result[i] /= magnitude

        return result, anomaly_flags

    def _generate_salary_series(
        self,
        profile: CustomerTransactionProfile,
        year_months: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(year_months)
        salary_count = np.zeros(n, dtype=int)
        salary_amount = np.zeros(n)
        has_bonus = np.zeros(n, dtype=bool)
        bonus_amount = np.zeros(n)

        base_salary = profile.monthly_income
        regularity = profile.salary_regularity

        for i, (year, month) in enumerate(year_months):
            # Salary deposit
            if self.rng.random() < regularity:
                salary_count[i] = 1

                # Add some variation to salary
                variation = self.rng.normal(1, profile.income_volatility * 0.3)
                salary_amount[i] = base_salary * variation

                # Apply annual raise (1-5% per year)
                years_passed = i / 12
                annual_raise = 1 + self.rng.uniform(0.01, 0.05) * years_passed
                salary_amount[i] *= annual_raise

            # Bonus check
            bonus_prob = self.calendar.get_bonus_probability(month)
            if self.rng.random() < bonus_prob:
                has_bonus[i] = True
                # Bonus typically 1-3 months salary
                if self.calendar.get_13th_month_salary(month):
                    bonus_multiplier = self.rng.uniform(1.0, 2.0)
                else:
                    bonus_multiplier = self.rng.uniform(0.5, 1.5)
                bonus_amount[i] = base_salary * bonus_multiplier

        return salary_count, salary_amount, has_bonus, bonus_amount

    def _generate_transaction_counts(
        self,
        profile: CustomerTransactionProfile,
        year_months: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(year_months)

        # Base transaction counts based on income level
        if profile.monthly_income > 30_000_000:
            base_credit = 8
            base_debit = 40
        elif profile.monthly_income > 15_000_000:
            base_credit = 5
            base_debit = 30
        elif profile.monthly_income > 8_000_000:
            base_credit = 3
            base_debit = 20
        else:
            base_credit = 2
            base_debit = 15

        # Age adjustment (older = fewer transactions)
        if profile.age > 50:
            base_debit *= 0.7
        elif profile.age < 30:
            base_debit *= 1.2

        credit_counts = np.zeros(n, dtype=int)
        debit_counts = np.zeros(n, dtype=int)

        for i, (year, month) in enumerate(year_months):
            # Seasonal adjustment for spending
            seasonal_mult = self.calendar.get_spending_multiplier(year, month)

            credit_counts[i] = int(self.rng.poisson(base_credit))
            debit_counts[i] = int(self.rng.poisson(base_debit * seasonal_mult))

        return credit_counts, debit_counts

    def _generate_balance_series(
        self,
        profile: CustomerTransactionProfile,
        total_credit: np.ndarray,
        total_debit: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(total_credit)

        # Starting balance based on savings
        starting_balance = profile.savings_amount * 0.3  # Only portion in checking

        avg_balance = np.zeros(n)
        min_balance = np.zeros(n)
        max_balance = np.zeros(n)
        days_zero = np.zeros(n, dtype=int)
        overdraft_count = np.zeros(n, dtype=int)
        volatility = np.zeros(n)

        current_balance = starting_balance

        for i in range(n):
            # Net flow
            net_flow = total_credit[i] - total_debit[i]

            # Simulate daily balance fluctuations
            daily_balances = []
            temp_balance = current_balance

            # Distribute transactions across month (simplified)
            n_days = 30
            daily_credit = total_credit[i] / max(1, n_days // 3)  # Credits early
            daily_debit = total_debit[i] / n_days

            for day in range(n_days):
                # Credits typically come early (salary)
                if day < 5:
                    temp_balance += daily_credit * 3
                # Debits spread throughout
                temp_balance -= daily_debit
                temp_balance = max(temp_balance, -profile.monthly_income * 0.2)  # Overdraft limit
                daily_balances.append(temp_balance)

            daily_balances = np.array(daily_balances)

            # Calculate metrics
            avg_balance[i] = max(0, np.mean(daily_balances))
            min_balance[i] = np.min(daily_balances)
            max_balance[i] = np.max(daily_balances)
            days_zero[i] = int(np.sum(daily_balances <= 0))
            overdraft_count[i] = int(np.sum(daily_balances < 0))
            volatility[i] = np.std(daily_balances) if len(daily_balances) > 1 else 0

            # Update balance for next month
            current_balance = daily_balances[-1]

        return avg_balance, min_balance, max_balance, days_zero, overdraft_count, volatility

    def _generate_loan_payments(
        self,
        profile: CustomerTransactionProfile,
        n_months: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        payment_count = np.zeros(n_months, dtype=int)
        payment_amount = np.zeros(n_months)

        if profile.existing_debt <= 0:
            return payment_count, payment_amount

        # Estimate monthly payment (assuming 5-year term, 10% interest)
        monthly_payment = profile.existing_debt / 60 * 1.25

        for i in range(n_months):
            # Most people pay once per month
            if self.rng.random() < 0.95:
                payment_count[i] = 1
                # Some variation in payment amount
                variation = self.rng.normal(1, 0.05)
                payment_amount[i] = monthly_payment * variation
            # Risk profile affects payment regularity
            elif profile.is_risky and self.rng.random() < 0.2:
                # Risky customers may miss payments
                payment_count[i] = 0
                payment_amount[i] = 0

        return payment_count, payment_amount

    def generate_single_customer_series(
        self,
        profile: CustomerTransactionProfile
    ) -> pd.DataFrame:
        n = self.n_months
        year_months = [(y, m) for y, m, _ in self.months]
        month_ids = [mid for _, _, mid in self.months]

        # Generate salary and bonus
        salary_count, salary_amount, has_bonus, bonus_amount = \
            self._generate_salary_series(profile, year_months)

        # Generate transaction counts
        credit_counts, debit_counts = \
            self._generate_transaction_counts(profile, year_months)

        # Generate total amounts
        total_credit = salary_amount + bonus_amount

        # Add other income sources
        other_income = np.zeros(n)
        for i in range(n):
            if self.rng.random() < 0.3:  # 30% chance of other income
                other_income[i] = self.rng.uniform(0.1, 0.3) * profile.monthly_income
        total_credit += other_income

        # Generate debit amounts (spending)
        base_spending = profile.monthly_expenses
        total_debit = np.zeros(n)
        for i, (year, month) in enumerate(year_months):
            seasonal_mult = self.calendar.get_spending_multiplier(year, month)
            noise = self.rng.normal(1, profile.expense_volatility)
            total_debit[i] = base_spending * seasonal_mult * noise

        # Apply trend (gradual income/expense increase)
        if profile.employment_type in ["cong_chuc", "nhan_vien"]:
            total_credit = self.add_trend(total_credit, TrendType.LINEAR_UP, 0.005)
            total_debit = self.add_trend(total_debit, TrendType.LINEAR_UP, 0.004)

        # Generate loan payments
        loan_count, loan_amount = self._generate_loan_payments(profile, n)
        total_debit += loan_amount

        # Generate balance metrics
        avg_balance, min_balance, max_balance, days_zero, overdraft, volatility = \
            self._generate_balance_series(profile, total_credit, total_debit)

        # Add noise to all series
        total_credit = self.add_noise(total_credit, 0.05)
        total_debit = self.add_noise(total_debit, 0.08)

        # Round amounts
        total_credit = np.round(total_credit / 1000) * 1000
        total_debit = np.round(total_debit / 1000) * 1000
        salary_amount = np.round(salary_amount / 1000) * 1000
        bonus_amount = np.round(bonus_amount / 1000) * 1000
        loan_amount = np.round(loan_amount / 1000) * 1000
        avg_balance = np.round(avg_balance / 1000) * 1000
        min_balance = np.round(min_balance / 1000) * 1000
        max_balance = np.round(max_balance / 1000) * 1000

        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': profile.customer_id,
            'month_id': month_ids,
            'year': [y for y, _, _ in self.months],
            'month': [m for _, m, _ in self.months],
            'num_credit_transactions': credit_counts,
            'num_debit_transactions': debit_counts,
            'total_credit_amount': total_credit,
            'total_debit_amount': total_debit,
            'avg_balance': avg_balance,
            'min_balance': min_balance,
            'max_balance': max_balance,
            'num_salary_deposits': salary_count,
            'salary_amount': salary_amount,
            'has_bonus': has_bonus,
            'bonus_amount': bonus_amount,
            'num_loan_payments': loan_count,
            'loan_payment_amount': loan_amount,
            'days_with_zero_balance': days_zero,
            'overdraft_count': overdraft,
            'balance_volatility': np.round(volatility, 2),
        })

        return df

    def generate(
        self,
        demographic_df: pd.DataFrame,
        financial_df: pd.DataFrame,
        credit_df: Optional[pd.DataFrame] = None,
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        # Merge data
        merged = demographic_df[['customer_id', 'age']].merge(
            financial_df[[
                'customer_id', 'monthly_income', 'monthly_expenses',
                'employment_type_code', 'existing_debt', 'savings_amount'
            ]],
            on='customer_id'
        )

        if credit_df is not None:
            credit_cols = ['customer_id', 'has_credit_history', 'cic_score']
            available_cols = [c for c in credit_cols if c in credit_df.columns]
            merged = merged.merge(credit_df[available_cols], on='customer_id', how='left')
            merged['has_credit_history'] = merged['has_credit_history'].fillna(False)
            merged['cic_score'] = merged['cic_score'].fillna(0)
        else:
            merged['has_credit_history'] = False
            merged['cic_score'] = 0

        # Sample if specified
        if sample_size is not None and sample_size < len(merged):
            merged = merged.sample(n=sample_size, random_state=self.seed)

        # Generate series for each customer
        all_series = []
        for idx, row in merged.iterrows():
            # Create profile
            profile = CustomerTransactionProfile(
                customer_id=row['customer_id'],
                monthly_income=row['monthly_income'],
                monthly_expenses=row['monthly_expenses'],
                employment_type=row['employment_type_code'],
                age=row['age'],
                has_credit_history=row['has_credit_history'],
                cic_score=int(row['cic_score']),
                existing_debt=row['existing_debt'],
                savings_amount=row['savings_amount'],
            )

            # Generate series
            customer_series = self.generate_single_customer_series(profile)
            all_series.append(customer_series)

        # Combine all series
        result = pd.concat(all_series, ignore_index=True)

        # Convert to long format if requested
        if self.output_format == OutputFormat.LONG:
            result = self._convert_to_long_format(result)

        self._generated_data = result
        return result

    def _convert_to_long_format(self, wide_df: pd.DataFrame) -> pd.DataFrame:
        id_vars = ['customer_id', 'month_id', 'year', 'month']
        value_vars = [col for col in wide_df.columns if col not in id_vars]

        long_df = pd.melt(
            wide_df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='feature_name',
            value_name='value'
        )

        return long_df

    def get_series_summary(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        if data is None:
            data = self._generated_data

        if data is None:
            return {"error": "No data available"}

        # Handle long format
        if 'feature_name' in data.columns:
            # Convert back to wide for analysis
            data = data.pivot_table(
                index=['customer_id', 'month_id'],
                columns='feature_name',
                values='value'
            ).reset_index()

        summary = {
            "n_customers": data['customer_id'].nunique(),
            "n_months": data['month_id'].nunique(),
            "total_records": len(data),
            "date_range": {
                "start": data['month_id'].min(),
                "end": data['month_id'].max(),
            },
            "transaction_stats": {},
            "balance_stats": {},
            "income_stats": {},
        }

        # Transaction statistics
        if 'total_credit_amount' in data.columns:
            summary["transaction_stats"] = {
                "avg_monthly_credit": float(data['total_credit_amount'].mean()),
                "avg_monthly_debit": float(data['total_debit_amount'].mean()),
                "avg_credit_transactions": float(data['num_credit_transactions'].mean()),
                "avg_debit_transactions": float(data['num_debit_transactions'].mean()),
            }

        # Balance statistics
        if 'avg_balance' in data.columns:
            summary["balance_stats"] = {
                "avg_balance": float(data['avg_balance'].mean()),
                "pct_zero_balance_days": float(
                    (data['days_with_zero_balance'] > 0).mean()
                ),
                "pct_overdraft": float((data['overdraft_count'] > 0).mean()),
            }

        # Income statistics
        if 'salary_amount' in data.columns:
            has_salary = data['salary_amount'] > 0
            summary["income_stats"] = {
                "avg_salary": float(data.loc[has_salary, 'salary_amount'].mean()),
                "pct_with_salary": float(has_salary.mean()),
                "pct_with_bonus": float(data['has_bonus'].mean()),
            }

        return summary


# MODULE EXPORTS

__all__ = [
    "TransactionSeriesGenerator",
    "VietnameseCalendar",
    "CustomerTransactionProfile",
    "SeasonalityType",
    "TrendType",
    "OutputFormat",
    "VIETNAMESE_HOLIDAYS",
    "TET_MONTHS",
    "MONTHLY_SPENDING_MULTIPLIER",
]
