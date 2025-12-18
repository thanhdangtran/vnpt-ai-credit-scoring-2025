from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union


# ENUMS FOR CATEGORICAL VALUES

class OutputFormat(Enum):
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"


class Region(Enum):
    HA_NOI = "ha_noi"
    HO_CHI_MINH = "ho_chi_minh"
    DA_NANG = "da_nang"
    CAN_THO = "can_tho"
    HAI_PHONG = "hai_phong"
    OTHER_PROVINCE = "other_province"


class ContractType(Enum):
    TRA_TRUOC = "tra_truoc"      # Prepaid
    TRA_SAU = "tra_sau"          # Postpaid


class ServiceType(Enum):
    MOBILE = "mobile"
    FIBER = "fiber"
    COMBO = "combo"              # Mobile + Fiber bundle
    IPTV = "iptv"
    MOBILE_FIBER_IPTV = "mobile_fiber_iptv"


class UsagePattern(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TimeGranularity(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class MissingMechanism(Enum):
    MCAR = "MCAR"    # Missing Completely At Random
    MAR = "MAR"      # Missing At Random
    MNAR = "MNAR"    # Missing Not At Random


class ConditionOperator(Enum):
    LESS_THAN = "<"
    GREATER_THAN = ">"
    EQUAL = "=="
    NOT_EQUAL = "!="
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"


class BaselComplianceLevel(Enum):
    BASEL_II = "II"
    BASEL_III = "III"


class JobCategory(Enum):
    CONG_NHAN = "cong_nhan"                      # Factory worker
    NHAN_VIEN_VAN_PHONG = "nhan_vien_van_phong"  # Office worker
    KINH_DOANH_TU_DO = "kinh_doanh_tu_do"        # Self-employed/Business
    CONG_CHUC_NHA_NUOC = "cong_chuc_nha_nuoc"    # Government employee
    GIAO_VIEN = "giao_vien"                      # Teacher
    BAC_SI_Y_TA = "bac_si_y_ta"                  # Doctor/Nurse
    KY_SU_IT = "ky_su_it"                        # IT Engineer
    NONG_DAN = "nong_dan"                        # Farmer
    LAI_XE = "lai_xe"                            # Driver
    SINH_VIEN = "sinh_vien"                      # Student
    HUU_TRI = "huu_tri"                          # Retired
    NOI_TRO = "noi_tro"                          # Homemaker
    THAT_NGHIEP = "that_nghiep"                  # Unemployed
    KHAC = "khac"                                # Other


class LoanPurpose(Enum):
    MUA_NHA = "mua_nha"                          # Home purchase
    MUA_XE = "mua_xe"                            # Vehicle purchase
    KINH_DOANH = "kinh_doanh"                    # Business
    TIEU_DUNG = "tieu_dung"                      # Consumer goods
    HOC_TAP = "hoc_tap"                          # Education
    CHUA_BENH = "chua_benh"                      # Medical treatment
    SUA_NHA = "sua_nha"                          # Home renovation
    DU_LICH = "du_lich"                          # Travel
    DAM_CUOI = "dam_cuoi"                        # Wedding
    TRA_NO = "tra_no"                            # Debt consolidation
    KHAC = "khac"                                # Other


# DATACLASSES

@dataclass
class MNARRule:
    target_column: str
    condition_column: str
    condition_operator: ConditionOperator
    condition_value: Any
    missing_probability: float
    description: Optional[str] = None

    def __post_init__(self):
        if not 0 <= self.missing_probability <= 1:
            raise ValueError("missing_probability must be between 0 and 1")


@dataclass
class CreditScoringConfig:
    n_samples: int = 10_000
    random_seed: int = 42
    output_format: OutputFormat = OutputFormat.PARQUET
    include_time_series: bool = True
    time_series_months: int = 24
    target_default_rate: float = 0.15
    train_test_split: float = 0.8
    include_demographics: bool = True
    include_financial: bool = True
    include_telecom: bool = True

    def __post_init__(self):
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if not 0 < self.target_default_rate < 1:
            raise ValueError("target_default_rate must be between 0 and 1")
        if not 0 < self.train_test_split < 1:
            raise ValueError("train_test_split must be between 0 and 1")


@dataclass
class IncomeRange:
    min_income: int          # VND monthly
    max_income: int          # VND monthly
    median_income: int       # VND monthly
    std_dev: int             # Standard deviation


@dataclass
class VietnameseMarketConfig:
    min_age: int = 18
    max_age: int = 65

    # Income ranges by region (VND/month)
    income_ranges: Dict[Region, IncomeRange] = field(default_factory=lambda: {
        Region.HA_NOI: IncomeRange(
            min_income=5_000_000,
            max_income=200_000_000,
            median_income=15_000_000,
            std_dev=12_000_000
        ),
        Region.HO_CHI_MINH: IncomeRange(
            min_income=5_500_000,
            max_income=250_000_000,
            median_income=17_000_000,
            std_dev=15_000_000
        ),
        Region.DA_NANG: IncomeRange(
            min_income=4_500_000,
            max_income=100_000_000,
            median_income=12_000_000,
            std_dev=8_000_000
        ),
        Region.CAN_THO: IncomeRange(
            min_income=4_000_000,
            max_income=80_000_000,
            median_income=10_000_000,
            std_dev=6_000_000
        ),
        Region.HAI_PHONG: IncomeRange(
            min_income=4_500_000,
            max_income=90_000_000,
            median_income=11_000_000,
            std_dev=7_000_000
        ),
        Region.OTHER_PROVINCE: IncomeRange(
            min_income=3_500_000,
            max_income=60_000_000,
            median_income=8_000_000,
            std_dev=5_000_000
        ),
    })

    # Job categories with probability weights
    job_categories: Dict[JobCategory, float] = field(default_factory=lambda: {
        JobCategory.CONG_NHAN: 0.20,
        JobCategory.NHAN_VIEN_VAN_PHONG: 0.18,
        JobCategory.KINH_DOANH_TU_DO: 0.15,
        JobCategory.CONG_CHUC_NHA_NUOC: 0.08,
        JobCategory.GIAO_VIEN: 0.06,
        JobCategory.BAC_SI_Y_TA: 0.04,
        JobCategory.KY_SU_IT: 0.05,
        JobCategory.NONG_DAN: 0.10,
        JobCategory.LAI_XE: 0.05,
        JobCategory.SINH_VIEN: 0.03,
        JobCategory.HUU_TRI: 0.03,
        JobCategory.NOI_TRO: 0.02,
        JobCategory.THAT_NGHIEP: 0.01,
    })

    # Loan purposes with probability weights
    loan_purposes: Dict[LoanPurpose, float] = field(default_factory=lambda: {
        LoanPurpose.MUA_NHA: 0.15,
        LoanPurpose.MUA_XE: 0.12,
        LoanPurpose.KINH_DOANH: 0.20,
        LoanPurpose.TIEU_DUNG: 0.25,
        LoanPurpose.HOC_TAP: 0.05,
        LoanPurpose.CHUA_BENH: 0.03,
        LoanPurpose.SUA_NHA: 0.08,
        LoanPurpose.DU_LICH: 0.02,
        LoanPurpose.DAM_CUOI: 0.03,
        LoanPurpose.TRA_NO: 0.05,
        LoanPurpose.KHAC: 0.02,
    })

    # Education levels with probability weights
    education_levels: Dict[str, float] = field(default_factory=lambda: {
        "tieu_hoc": 0.05,           # Primary school
        "thcs": 0.10,               # Secondary school
        "thpt": 0.25,               # High school
        "trung_cap": 0.15,          # Vocational
        "cao_dang": 0.12,           # College
        "dai_hoc": 0.28,            # University
        "thac_si": 0.04,            # Master's
        "tien_si": 0.01,            # PhD
    })

    # Marital status distribution
    marital_status: Dict[str, float] = field(default_factory=lambda: {
        "doc_than": 0.30,           # Single
        "da_ket_hon": 0.55,         # Married
        "ly_hon": 0.10,             # Divorced
        "goa": 0.05,                # Widowed
    })

    # Region distribution
    region_distribution: Dict[Region, float] = field(default_factory=lambda: {
        Region.HA_NOI: 0.20,
        Region.HO_CHI_MINH: 0.25,
        Region.DA_NANG: 0.08,
        Region.CAN_THO: 0.05,
        Region.HAI_PHONG: 0.07,
        Region.OTHER_PROVINCE: 0.35,
    })

    def __post_init__(self):
        if self.min_age < 18:
            raise ValueError("min_age must be at least 18")
        if self.max_age > 100:
            raise ValueError("max_age must be at most 100")
        if self.min_age >= self.max_age:
            raise ValueError("min_age must be less than max_age")


@dataclass
class VNPTTelecomConfig:
    # Contract type distribution
    contract_types: Dict[ContractType, float] = field(default_factory=lambda: {
        ContractType.TRA_TRUOC: 0.55,
        ContractType.TRA_SAU: 0.45,
    })

    # Service type distribution
    service_types: Dict[ServiceType, float] = field(default_factory=lambda: {
        ServiceType.MOBILE: 0.40,
        ServiceType.FIBER: 0.20,
        ServiceType.COMBO: 0.25,
        ServiceType.IPTV: 0.05,
        ServiceType.MOBILE_FIBER_IPTV: 0.10,
    })

    # Usage pattern distribution
    usage_patterns: Dict[UsagePattern, float] = field(default_factory=lambda: {
        UsagePattern.LOW: 0.25,
        UsagePattern.MEDIUM: 0.45,
        UsagePattern.HIGH: 0.25,
        UsagePattern.VERY_HIGH: 0.05,
    })

    # Customer tenure range (months)
    tenure_min_months: int = 1
    tenure_max_months: int = 120  # 10 years

    # ARPU (Average Revenue Per User) ranges by usage pattern (VND/month)
    arpu_ranges: Dict[UsagePattern, Tuple[int, int]] = field(default_factory=lambda: {
        UsagePattern.LOW: (50_000, 150_000),
        UsagePattern.MEDIUM: (150_000, 400_000),
        UsagePattern.HIGH: (400_000, 1_000_000),
        UsagePattern.VERY_HIGH: (1_000_000, 5_000_000),
    })

    # Late payment probability by usage pattern
    late_payment_probability: Dict[UsagePattern, float] = field(default_factory=lambda: {
        UsagePattern.LOW: 0.25,
        UsagePattern.MEDIUM: 0.15,
        UsagePattern.HIGH: 0.10,
        UsagePattern.VERY_HIGH: 0.05,
    })

    # Data usage ranges (GB/month)
    data_usage_ranges: Dict[UsagePattern, Tuple[float, float]] = field(default_factory=lambda: {
        UsagePattern.LOW: (0.5, 5.0),
        UsagePattern.MEDIUM: (5.0, 20.0),
        UsagePattern.HIGH: (20.0, 50.0),
        UsagePattern.VERY_HIGH: (50.0, 200.0),
    })

    # Call minutes per month ranges
    call_minutes_ranges: Dict[UsagePattern, Tuple[int, int]] = field(default_factory=lambda: {
        UsagePattern.LOW: (10, 100),
        UsagePattern.MEDIUM: (100, 500),
        UsagePattern.HIGH: (500, 1500),
        UsagePattern.VERY_HIGH: (1500, 5000),
    })

    # SMS per month ranges
    sms_ranges: Dict[UsagePattern, Tuple[int, int]] = field(default_factory=lambda: {
        UsagePattern.LOW: (0, 20),
        UsagePattern.MEDIUM: (20, 100),
        UsagePattern.HIGH: (100, 300),
        UsagePattern.VERY_HIGH: (300, 1000),
    })


@dataclass
class TimeSeriesConfig:
    observation_months: int = 24
    granularity: TimeGranularity = TimeGranularity.MONTHLY
    include_seasonality: bool = True
    include_trend: bool = True
    include_noise: bool = True

    # Seasonal periods (in terms of granularity units)
    seasonality_periods: List[int] = field(default_factory=lambda: [12])  # Annual

    # Trend parameters
    trend_strength: float = 0.3
    trend_direction: str = "increasing"  # "increasing", "decreasing", "stable"

    # Noise parameters
    noise_level: float = 0.1

    # Features to generate time series for
    time_series_features: List[str] = field(default_factory=lambda: [
        "monthly_income",
        "monthly_spending",
        "telecom_arpu",
        "data_usage_gb",
        "call_minutes",
        "payment_timeliness_score",
        "account_balance",
        "transaction_count",
    ])

    # Special events (holidays, Tet, etc.)
    include_special_events: bool = True
    special_events: Dict[str, List[int]] = field(default_factory=lambda: {
        "tet_nguyen_dan": [1, 2],      # Lunar New Year (Jan-Feb)
        "summer_holiday": [6, 7, 8],    # Summer months
        "year_end": [11, 12],           # Year-end shopping
    })

    def __post_init__(self):
        if self.observation_months <= 0:
            raise ValueError("observation_months must be positive")
        if not 0 <= self.trend_strength <= 1:
            raise ValueError("trend_strength must be between 0 and 1")
        if not 0 <= self.noise_level <= 1:
            raise ValueError("noise_level must be between 0 and 1")


@dataclass
class MNARConfig:
    enable_mnar: bool = True
    overall_missing_rate: float = 0.15

    # Missing mechanism for each feature
    missing_mechanisms: Dict[str, MissingMechanism] = field(default_factory=lambda: {
        # Demographic features
        "income": MissingMechanism.MNAR,          # High earners hide income
        "age": MissingMechanism.MCAR,             # Random missing
        "education": MissingMechanism.MAR,        # Related to job type
        "marital_status": MissingMechanism.MCAR,

        # Financial features
        "bank_balance": MissingMechanism.MNAR,    # Low balance = less likely to report
        "existing_loans": MissingMechanism.MNAR,  # Many loans = hide info
        "credit_history_months": MissingMechanism.MAR,

        # Telecom features
        "telecom_tenure": MissingMechanism.MCAR,
        "monthly_arpu": MissingMechanism.MAR,
        "late_payments_count": MissingMechanism.MNAR,  # Bad behavior hidden
    })

    # Per-feature missing rates (overrides overall rate)
    feature_missing_rates: Dict[str, float] = field(default_factory=lambda: {
        "income": 0.20,
        "bank_balance": 0.15,
        "existing_loans": 0.10,
        "late_payments_count": 0.25,
    })

    # MNAR rules - business logic for missing data
    mnar_rules: List[MNARRule] = field(default_factory=lambda: [
        # High income individuals tend to hide exact income
        MNARRule(
            target_column="income",
            condition_column="income",
            condition_operator=ConditionOperator.GREATER_THAN,
            condition_value=50_000_000,
            missing_probability=0.35,
            description="High earners less likely to report exact income"
        ),
        # Low bank balance leads to missing bank info
        MNARRule(
            target_column="bank_balance",
            condition_column="bank_balance",
            condition_operator=ConditionOperator.LESS_THAN,
            condition_value=5_000_000,
            missing_probability=0.40,
            description="Low balance customers avoid reporting"
        ),
        # Multiple existing loans leads to hiding loan info
        MNARRule(
            target_column="existing_loans",
            condition_column="num_existing_loans",
            condition_operator=ConditionOperator.GREATER_THAN,
            condition_value=3,
            missing_probability=0.45,
            description="Heavy borrowers hide loan information"
        ),
        # Many late payments leads to missing payment history
        MNARRule(
            target_column="late_payments_count",
            condition_column="late_payments_count",
            condition_operator=ConditionOperator.GREATER_THAN,
            condition_value=5,
            missing_probability=0.50,
            description="Bad payers hide payment history"
        ),
        # Young people without credit history
        MNARRule(
            target_column="credit_history_months",
            condition_column="age",
            condition_operator=ConditionOperator.LESS_THAN,
            condition_value=25,
            missing_probability=0.60,
            description="Young applicants lack credit history"
        ),
    ])

    def __post_init__(self):
        if not 0 <= self.overall_missing_rate <= 1:
            raise ValueError("overall_missing_rate must be between 0 and 1")


@dataclass
class NHNNCircularRequirement:
    circular_number: str
    name: str
    required_fields: List[str]
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    effective_date: Optional[str] = None


@dataclass
class RegulatoryConfig:
    basel_compliance_level: BaselComplianceLevel = BaselComplianceLevel.BASEL_II

    # NHNN Circular requirements
    nhnn_circulars: List[NHNNCircularRequirement] = field(default_factory=lambda: [
        NHNNCircularRequirement(
            circular_number="11/2021/TT-NHNN",
            name="Thông tư về phân loại tài sản có và trích lập dự phòng rủi ro",
            required_fields=[
                "customer_id",
                "loan_amount",
                "loan_term",
                "interest_rate",
                "collateral_value",
                "credit_rating",
                "overdue_days",
                "loan_group",  # Nhóm nợ 1-5
            ],
            validation_rules={
                "loan_group_range": (1, 5),
                "overdue_classification": {
                    1: (0, 10),      # Nhóm 1: Nợ đủ tiêu chuẩn
                    2: (10, 90),     # Nhóm 2: Nợ cần chú ý
                    3: (90, 180),    # Nhóm 3: Nợ dưới tiêu chuẩn
                    4: (180, 360),   # Nhóm 4: Nợ nghi ngờ
                    5: (360, None),  # Nhóm 5: Nợ có khả năng mất vốn
                },
                "provision_rates": {
                    1: 0.0,
                    2: 0.05,
                    3: 0.20,
                    4: 0.50,
                    5: 1.00,
                },
            },
            effective_date="2021-10-01"
        ),
        NHNNCircularRequirement(
            circular_number="13/2023/TT-NHNN",
            name="Thông tư về hệ thống xếp hạng tín dụng nội bộ",
            required_fields=[
                "customer_id",
                "financial_indicators",
                "non_financial_indicators",
                "credit_score",
                "rating_grade",
                "pd_estimate",
            ],
            validation_rules={
                "rating_grades": ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"],
                "pd_ranges": {
                    "AAA": (0.0, 0.001),
                    "AA": (0.001, 0.005),
                    "A": (0.005, 0.01),
                    "BBB": (0.01, 0.03),
                    "BB": (0.03, 0.06),
                    "B": (0.06, 0.15),
                    "CCC": (0.15, 0.30),
                    "CC": (0.30, 0.50),
                    "C": (0.50, 0.80),
                    "D": (0.80, 1.0),
                },
            },
            effective_date="2023-07-01"
        ),
        NHNNCircularRequirement(
            circular_number="41/2016/TT-NHNN",
            name="Thông tư quy định tỷ lệ an toàn vốn",
            required_fields=[
                "tier1_capital",
                "tier2_capital",
                "risk_weighted_assets",
                "capital_adequacy_ratio",
            ],
            validation_rules={
                "min_car": 0.08,  # Minimum 8% CAR
                "tier1_min": 0.045,  # Minimum 4.5% Tier 1
            },
            effective_date="2016-12-31"
        ),
    ])

    # Basel requirements
    pd_calculation_method: str = "through_the_cycle"  # "point_in_time" or "through_the_cycle"
    lgd_floor: float = 0.10  # 10% LGD floor for Basel II
    ead_approach: str = "ccf"  # Credit Conversion Factor approach

    # Risk weight approaches
    credit_risk_approach: str = "standardized"  # "standardized" or "irb"
    operational_risk_approach: str = "basic_indicator"  # "basic_indicator", "standardized", "advanced"

    # Stress testing parameters
    stress_testing_scenarios: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "baseline": {
            "pd_multiplier": 1.0,
            "lgd_multiplier": 1.0,
            "gdp_shock": 0.0,
        },
        "adverse": {
            "pd_multiplier": 1.5,
            "lgd_multiplier": 1.2,
            "gdp_shock": -0.03,
        },
        "severe": {
            "pd_multiplier": 2.5,
            "lgd_multiplier": 1.5,
            "gdp_shock": -0.08,
        },
    })


# MASTER CONFIG

@dataclass
class SyntheticDataConfig:
    credit_scoring: CreditScoringConfig = field(default_factory=CreditScoringConfig)
    vietnamese_market: VietnameseMarketConfig = field(default_factory=VietnameseMarketConfig)
    vnpt_telecom: VNPTTelecomConfig = field(default_factory=VNPTTelecomConfig)
    time_series: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)
    mnar: MNARConfig = field(default_factory=MNARConfig)
    regulatory: RegulatoryConfig = field(default_factory=RegulatoryConfig)

    # Output settings
    output_dir: str = "./output"
    save_intermediate: bool = True
    verbose: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SyntheticDataConfig":
        return cls(
            credit_scoring=CreditScoringConfig(**config_dict.get("credit_scoring", {})),
            vietnamese_market=VietnameseMarketConfig(**config_dict.get("vietnamese_market", {})),
            vnpt_telecom=VNPTTelecomConfig(**config_dict.get("vnpt_telecom", {})),
            time_series=TimeSeriesConfig(**config_dict.get("time_series", {})),
            mnar=MNARConfig(**config_dict.get("mnar", {})),
            regulatory=RegulatoryConfig(**config_dict.get("regulatory", {})),
            output_dir=config_dict.get("output_dir", "./output"),
            save_intermediate=config_dict.get("save_intermediate", True),
            verbose=config_dict.get("verbose", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


# DEFAULT CONFIGURATIONS

def get_default_config() -> SyntheticDataConfig:
    return SyntheticDataConfig()


def get_small_sample_config() -> SyntheticDataConfig:
    return SyntheticDataConfig(
        credit_scoring=CreditScoringConfig(
            n_samples=1000,
            time_series_months=12,
        ),
        time_series=TimeSeriesConfig(
            observation_months=12,
        ),
    )


def get_production_config() -> SyntheticDataConfig:
    return SyntheticDataConfig(
        credit_scoring=CreditScoringConfig(
            n_samples=100_000,
            time_series_months=36,
            output_format=OutputFormat.PARQUET,
        ),
        time_series=TimeSeriesConfig(
            observation_months=36,
            granularity=TimeGranularity.MONTHLY,
        ),
    )


# MODULE EXPORTS

__all__ = [
    # Enums
    "OutputFormat",
    "Region",
    "ContractType",
    "ServiceType",
    "UsagePattern",
    "TimeGranularity",
    "MissingMechanism",
    "ConditionOperator",
    "BaselComplianceLevel",
    "JobCategory",
    "LoanPurpose",
    # Dataclasses
    "MNARRule",
    "CreditScoringConfig",
    "IncomeRange",
    "VietnameseMarketConfig",
    "VNPTTelecomConfig",
    "TimeSeriesConfig",
    "MNARConfig",
    "NHNNCircularRequirement",
    "RegulatoryConfig",
    "SyntheticDataConfig",
    # Factory functions
    "get_default_config",
    "get_small_sample_config",
    "get_production_config",
]
