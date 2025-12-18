from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import Generator

import sys
sys.path.insert(0, '/home/thanhdang/Desktop/vnpt-ai')

from config.settings import SyntheticDataConfig


# ABSTRACT BASE CLASS

class BaseDataGenerator(ABC):
    def __init__(
        self,
        config: SyntheticDataConfig,
        seed: Optional[int] = None
    ) -> None:
        self.config = config
        self.seed = seed if seed is not None else config.credit_scoring.random_seed
        self.rng: Generator = self._create_rng()
        self._schema: Optional[Dict[str, type]] = None
        self._generated_data: Optional[pd.DataFrame] = None

    def _create_rng(self) -> Generator:
        return np.random.default_rng(self.seed)

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.rng = self._create_rng()

    def get_rng_state(self) -> Dict[str, Any]:
        return self.rng.bit_generator.state

    def set_rng_state(self, state: Dict[str, Any]) -> None:
        self.rng.bit_generator.state = state

    @abstractmethod
    def generate(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate()")

    def set_schema(self, schema: Dict[str, type]) -> None:
        self._schema = schema

    def validate_output(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        if data is None or data.empty:
            errors.append("Generated data is empty or None")
            return False, errors

        # Schema validation
        if self._schema:
            for col, expected_type in self._schema.items():
                if col not in data.columns:
                    errors.append(f"Missing required column: {col}")
                    continue

                # Check dtype compatibility
                actual_dtype = data[col].dtype
                if expected_type == int:
                    if not np.issubdtype(actual_dtype, np.integer):
                        # Allow float columns that could be int (check for decimals)
                        if np.issubdtype(actual_dtype, np.floating):
                            if not data[col].dropna().apply(float.is_integer).all():
                                errors.append(
                                    f"Column {col}: expected int, got {actual_dtype}"
                                )
                        else:
                            errors.append(
                                f"Column {col}: expected int, got {actual_dtype}"
                            )
                elif expected_type == float:
                    if not np.issubdtype(actual_dtype, np.number):
                        errors.append(
                            f"Column {col}: expected float, got {actual_dtype}"
                        )
                elif expected_type == str:
                    if actual_dtype != object and actual_dtype.name != 'string':
                        errors.append(
                            f"Column {col}: expected str, got {actual_dtype}"
                        )
                elif expected_type == bool:
                    if actual_dtype != bool and actual_dtype.name != 'boolean':
                        errors.append(
                            f"Column {col}: expected bool, got {actual_dtype}"
                        )

        # Basic data quality checks
        total_rows = len(data)
        expected_rows = self.config.credit_scoring.n_samples

        if total_rows != expected_rows:
            errors.append(
                f"Row count mismatch: expected {expected_rows}, got {total_rows}"
            )

        # Check for duplicate IDs if 'customer_id' exists
        if 'customer_id' in data.columns:
            duplicates = data['customer_id'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate customer_id values")

        return len(errors) == 0, errors

    def generate_and_validate(self) -> Tuple[pd.DataFrame, bool, List[str]]:
        data = self.generate()
        is_valid, errors = self.validate_output(data)
        self._generated_data = data
        return data, is_valid, errors

    def get_summary_statistics(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if data is None:
            data = self._generated_data

        if data is None:
            return {"error": "No data available"}

        stats = {
            "n_rows": len(data),
            "n_columns": len(data.columns),
            "columns": list(data.columns),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "missing_counts": data.isnull().sum().to_dict(),
            "missing_rates": (data.isnull().sum() / len(data)).to_dict(),
            "numeric_summary": {},
            "categorical_summary": {},
        }

        # Numeric columns summary
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats["numeric_summary"][col] = {
                "mean": float(data[col].mean()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "max": float(data[col].max()),
                "median": float(data[col].median()),
            }

        # Categorical columns summary
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            value_counts = data[col].value_counts()
            stats["categorical_summary"][col] = {
                "n_unique": int(data[col].nunique()),
                "top_values": value_counts.head(5).to_dict(),
            }

        return stats


# MIXIN CLASSES

class CorrelationMixin:
    rng: Generator  # Type hint for mixin compatibility

    def add_noise(
        self,
        data: Union[np.ndarray, pd.Series],
        noise_level: float,
        noise_type: str = "gaussian"
    ) -> Union[np.ndarray, pd.Series]:
        is_series = isinstance(data, pd.Series)
        arr = data.values if is_series else data

        # Calculate noise scale based on data standard deviation
        data_std = np.std(arr[~np.isnan(arr)]) if np.any(~np.isnan(arr)) else 1.0
        scale = noise_level * data_std

        # Generate noise based on type
        if noise_type == "gaussian":
            noise = self.rng.normal(0, scale, size=arr.shape)
        elif noise_type == "uniform":
            noise = self.rng.uniform(-scale * np.sqrt(3), scale * np.sqrt(3), size=arr.shape)
        elif noise_type == "laplacian":
            noise = self.rng.laplace(0, scale / np.sqrt(2), size=arr.shape)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        result = arr + noise

        if is_series:
            return pd.Series(result, index=data.index, name=data.name)
        return result

    def introduce_correlation(
        self,
        col1_data: np.ndarray,
        col2_data: np.ndarray,
        target_correlation: float,
        method: str = "cholesky"
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not -1 <= target_correlation <= 1:
            raise ValueError("target_correlation must be between -1 and 1")

        n = len(col1_data)

        if method == "cholesky":
            # Standardize the data
            mean1, std1 = np.mean(col1_data), np.std(col1_data)
            mean2, std2 = np.mean(col2_data), np.std(col2_data)

            z1 = (col1_data - mean1) / std1 if std1 > 0 else col1_data - mean1
            z2 = (col2_data - mean2) / std2 if std2 > 0 else col2_data - mean2

            # Create correlation matrix and Cholesky decomposition
            corr_matrix = np.array([
                [1.0, target_correlation],
                [target_correlation, 1.0]
            ])

            # Ensure positive semi-definite
            eigvals = np.linalg.eigvalsh(corr_matrix)
            if np.any(eigvals < 0):
                corr_matrix = corr_matrix + np.eye(2) * (abs(min(eigvals)) + 0.01)

            L = np.linalg.cholesky(corr_matrix)

            # Apply transformation
            uncorrelated = np.column_stack([z1, self.rng.standard_normal(n)])
            correlated = uncorrelated @ L.T

            # Restore original scale
            new_col1 = correlated[:, 0] * std1 + mean1
            new_col2 = correlated[:, 1] * std2 + mean2

            return new_col1, new_col2

        elif method == "copula":
            # Rank-based copula approach (preserves marginals better)
            ranks1 = np.argsort(np.argsort(col1_data))
            ranks2 = np.argsort(np.argsort(col2_data))

            # Generate correlated uniform ranks
            u1 = (ranks1 + 0.5) / n
            u2_independent = (ranks2 + 0.5) / n

            # Transform to normal, apply correlation, transform back
            from scipy import stats
            z1 = stats.norm.ppf(u1)
            z2_ind = stats.norm.ppf(u2_independent)

            # Apply correlation
            z2_corr = target_correlation * z1 + np.sqrt(1 - target_correlation**2) * z2_ind
            u2_corr = stats.norm.cdf(z2_corr)

            # Map back to original values using ranks
            sorted_col2 = np.sort(col2_data)
            new_col2_indices = np.clip(
                (u2_corr * n).astype(int), 0, n - 1
            )
            new_col2 = sorted_col2[new_col2_indices]

            return col1_data.copy(), new_col2

        else:
            raise ValueError(f"Unknown method: {method}")

    def create_conditional_distribution(
        self,
        condition_col: pd.Series,
        target_col: pd.Series,
        conditions: Dict[Any, Callable[[Generator, int], np.ndarray]]
    ) -> pd.Series:
        result = pd.Series(index=target_col.index, dtype=float)

        for condition_value, generator_func in conditions.items():
            mask = condition_col == condition_value
            count = mask.sum()
            if count > 0:
                result.loc[mask] = generator_func(self.rng, count)

        # Handle any unmatched conditions with default (uniform random from target)
        unmatched = result.isna()
        if unmatched.any():
            result.loc[unmatched] = self.rng.choice(
                target_col.dropna().values,
                size=unmatched.sum()
            )

        return result

    def create_correlation_matrix(
        self,
        n_features: int,
        base_correlation: float = 0.3,
        correlation_structure: str = "random"
    ) -> np.ndarray:
        if correlation_structure == "random":
            # Generate random correlations
            A = self.rng.uniform(
                -base_correlation, base_correlation,
                size=(n_features, n_features)
            )
            corr = (A + A.T) / 2
            np.fill_diagonal(corr, 1.0)

        elif correlation_structure == "block":
            # Block diagonal structure
            corr = np.eye(n_features)
            block_size = max(2, n_features // 3)
            for i in range(0, n_features, block_size):
                end = min(i + block_size, n_features)
                corr[i:end, i:end] = base_correlation
                np.fill_diagonal(corr[i:end, i:end], 1.0)

        elif correlation_structure == "decay":
            # Correlation decays with distance
            corr = np.eye(n_features)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr[i, j] = base_correlation * np.exp(-0.5 * (j - i))
                    corr[j, i] = corr[i, j]

        else:
            raise ValueError(f"Unknown correlation structure: {correlation_structure}")

        # Ensure positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 0.01)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Normalize to correlation matrix
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

        return corr


class TimeSeriesMixin:
    rng: Generator

    def generate_time_series(
        self,
        n_periods: int,
        base_value: float,
        trend: float = 0.0,
        seasonality_amplitude: float = 0.0,
        seasonality_period: int = 12,
        noise_level: float = 0.1
    ) -> np.ndarray:
        t = np.arange(n_periods)

        # Trend component
        trend_component = base_value + trend * t

        # Seasonality component
        seasonality = seasonality_amplitude * np.sin(2 * np.pi * t / seasonality_period)

        # Noise component
        noise = self.rng.normal(0, noise_level * base_value, n_periods)

        return trend_component + seasonality + noise

    def generate_ar_process(
        self,
        n_periods: int,
        ar_coefficients: List[float],
        initial_values: Optional[List[float]] = None,
        noise_std: float = 1.0
    ) -> np.ndarray:
        p = len(ar_coefficients)
        series = np.zeros(n_periods)

        if initial_values is not None:
            series[:p] = initial_values[:p]
        else:
            series[:p] = self.rng.normal(0, noise_std, p)

        for t in range(p, n_periods):
            ar_term = sum(
                ar_coefficients[i] * series[t - i - 1]
                for i in range(p)
            )
            series[t] = ar_term + self.rng.normal(0, noise_std)

        return series


# UTILITY FUNCTIONS

def weighted_random_choice(
    rng: Generator,
    options: List[Any],
    weights: List[float],
    size: Optional[int] = None
) -> Union[Any, np.ndarray]:
    # Normalize weights
    weights_arr = np.array(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()

    indices = rng.choice(len(options), size=size, p=weights_arr)

    if size is None:
        return options[indices]
    return np.array([options[i] for i in indices])


def generate_vietnamese_name(
    rng: Generator,
    gender: Optional[str] = None
) -> str:
    # Common Vietnamese family names (họ)
    family_names = [
        "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ",
        "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý"
    ]
    family_weights = [
        0.38, 0.11, 0.10, 0.07, 0.05, 0.05, 0.04, 0.04,
        0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.015, 0.015
    ]

    # Middle names (tên đệm)
    middle_names_male = [
        "Văn", "Hữu", "Đức", "Công", "Quốc", "Minh", "Thanh", "Hoàng",
        "Xuân", "Quang", "Ngọc", "Anh"
    ]
    middle_names_female = [
        "Thị", "Ngọc", "Thanh", "Thu", "Hoàng", "Minh", "Phương", "Thùy",
        "Bích", "Kim", "Mỹ", "Hồng"
    ]

    # Given names (tên)
    given_names_male = [
        "An", "Bình", "Cường", "Dũng", "Đức", "Hải", "Hùng", "Khoa",
        "Long", "Minh", "Nam", "Phong", "Quân", "Sơn", "Tùng", "Vinh",
        "Anh", "Bảo", "Hiếu", "Hoàng", "Huy", "Khánh", "Lâm", "Nghĩa",
        "Phúc", "Quang", "Tâm", "Thành", "Toàn", "Trung", "Tuấn", "Việt"
    ]
    given_names_female = [
        "Anh", "Châu", "Chi", "Dung", "Hà", "Hạnh", "Hoa", "Hương",
        "Lan", "Linh", "Mai", "My", "Ngân", "Ngọc", "Nhung", "Phương",
        "Quỳnh", "Thảo", "Thu", "Thủy", "Trang", "Trinh", "Vân", "Yến",
        "Diệu", "Giang", "Hằng", "Hiền", "Loan", "Ly", "Nhi", "Oanh"
    ]

    # Determine gender
    if gender is None:
        gender = rng.choice(['male', 'female'])

    # Select name components
    family = weighted_random_choice(rng, family_names, family_weights)

    if gender == 'male':
        middle = rng.choice(middle_names_male)
        given = rng.choice(given_names_male)
    else:
        middle = rng.choice(middle_names_female)
        given = rng.choice(given_names_female)

    return f"{family} {middle} {given}"


def generate_vietnamese_phone(
    rng: Generator,
    carrier: Optional[str] = None
) -> str:
    # Vietnamese mobile prefixes by carrier
    prefixes = {
        'vnpt': ['091', '094', '088', '083', '084', '085', '081', '082'],
        'viettel': ['096', '097', '098', '086', '032', '033', '034', '035', '036', '037', '038', '039'],
        'mobifone': ['090', '093', '089', '070', '076', '077', '078', '079'],
        'vietnamobile': ['092', '056', '058'],
        'gmobile': ['099', '059'],
    }

    if carrier is None:
        # Weight towards major carriers
        carrier = weighted_random_choice(
            rng,
            list(prefixes.keys()),
            [0.30, 0.45, 0.20, 0.03, 0.02]
        )

    prefix = rng.choice(prefixes[carrier])
    suffix = ''.join([str(rng.integers(0, 10)) for _ in range(7)])

    return f"{prefix}{suffix}"


def generate_vietnamese_id_number(
    rng: Generator,
    birth_year: Optional[int] = None,
    gender: Optional[str] = None,
    province_code: Optional[str] = None
) -> str:
    # Province codes (sample of common ones)
    province_codes = [
        '001',  # Hà Nội
        '079',  # Hồ Chí Minh
        '048',  # Đà Nẵng
        '092',  # Cần Thơ
        '031',  # Hải Phòng
        '024',  # Bắc Ninh
        '027',  # Hải Dương
        '036',  # Nam Định
        '038',  # Thanh Hóa
        '040',  # Nghệ An
        '060',  # Bình Dương
        '062',  # Đồng Nai
        '080',  # Long An
        '082',  # Tiền Giang
        '089',  # An Giang
    ]

    if province_code is None:
        province_code = rng.choice(province_codes)

    if birth_year is None:
        birth_year = int(rng.integers(1950, 2006))

    if gender is None:
        gender = rng.choice(['male', 'female'])

    # Determine century and gender digit (digit 4)
    # 0: Male, 20th century (1900-1999)
    # 1: Female, 20th century
    # 2: Male, 21st century (2000-2099)
    # 3: Female, 21st century
    century = 0 if birth_year < 2000 else 2
    gender_offset = 0 if gender == 'male' else 1
    century_gender_digit = century + gender_offset

    # Birth year last 2 digits
    year_digits = str(birth_year)[-2:]

    # Random sequence (6 digits)
    random_sequence = ''.join([str(rng.integers(0, 10)) for _ in range(6)])

    return f"{province_code}{century_gender_digit}{year_digits}{random_sequence}"


def generate_bank_account_number(
    rng: Generator,
    bank_code: Optional[str] = None
) -> str:
    # Vietnamese bank account formats
    bank_formats = {
        'vcb': {'prefix': '', 'length': 13},        # Vietcombank
        'tcb': {'prefix': '', 'length': 14},        # Techcombank
        'bidv': {'prefix': '', 'length': 14},       # BIDV
        'agribank': {'prefix': '', 'length': 13},   # Agribank
        'mb': {'prefix': '', 'length': 13},         # MB Bank
        'vpbank': {'prefix': '', 'length': 12},     # VPBank
        'acb': {'prefix': '', 'length': 10},        # ACB
        'tpbank': {'prefix': '', 'length': 12},     # TPBank
    }

    if bank_code is None:
        bank_code = rng.choice(list(bank_formats.keys()))

    format_info = bank_formats.get(bank_code, {'prefix': '', 'length': 12})
    length = format_info['length']

    account_number = ''.join([str(rng.integers(0, 10)) for _ in range(length)])

    # Ensure first digit is not 0
    if account_number[0] == '0':
        account_number = str(rng.integers(1, 10)) + account_number[1:]

    return account_number


def validate_vietnamese_id(id_number: str) -> bool:
    if not id_number or not id_number.isdigit():
        return False

    # New CCCD format: 12 digits
    if len(id_number) == 12:
        # Check century/gender digit (must be 0-9)
        if not 0 <= int(id_number[3]) <= 9:
            return False
        return True

    # Old CMND format: 9 digits
    if len(id_number) == 9:
        return True

    return False


# MODULE EXPORTS

__all__ = [
    # Base classes
    "BaseDataGenerator",
    # Mixins
    "CorrelationMixin",
    "TimeSeriesMixin",
    # Utility functions
    "weighted_random_choice",
    "generate_vietnamese_name",
    "generate_vietnamese_phone",
    "generate_vietnamese_id_number",
    "generate_bank_account_number",
    "validate_vietnamese_id",
]
