from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math


# ENUMS

class SegmentationMethod(Enum):
    CHAID = "chaid"           # Chi-squared Automatic Interaction Detection
    CART = "cart"             # Classification and Regression Trees
    BOTH = "both"             # Compare both methods
    MANUAL = "manual"         # Manual/expert-defined segments
    KMEANS = "kmeans"         # K-means clustering
    RFM = "rfm"               # Recency-Frequency-Monetary


class SegmentationType(Enum):
    RISK_BASED = "risk_based"           # Segment by risk level
    PRODUCT_BASED = "product_based"     # Segment by product type
    BEHAVIORAL = "behavioral"           # Segment by behavior patterns
    DEMOGRAPHIC = "demographic"         # Segment by demographic features
    CHANNEL = "channel"                 # Segment by acquisition channel
    VINTAGE = "vintage"                 # Segment by origination period
    HYBRID = "hybrid"                   # Combination of multiple types


class MissingHandling(Enum):
    SEPARATE_BIN = "separate_bin"   # Create separate bin for missing
    MODE = "mode"                   # Replace with mode
    MEDIAN = "median"               # Replace with median (continuous)
    WOE_ZERO = "woe_zero"           # Assign WOE = 0 (neutral)
    WORST_BIN = "worst_bin"         # Assign to worst WOE bin
    BEST_BIN = "best_bin"           # Assign to best WOE bin


class RegularizationType(Enum):
    L1 = "l1"                   # Lasso (sparse solutions)
    L2 = "l2"                   # Ridge (shrinkage)
    ELASTICNET = "elasticnet"  # Combination of L1 and L2
    NONE = "none"               # No regularization


class FeatureSelectionMethod(Enum):
    STEPWISE = "stepwise"           # Stepwise selection
    FORWARD = "forward"             # Forward selection
    BACKWARD = "backward"           # Backward elimination
    LASSO = "lasso"                 # Lasso-based selection
    IV_THRESHOLD = "iv_threshold"   # Information Value threshold
    CORRELATION = "correlation"     # Correlation-based
    RFE = "rfe"                     # Recursive Feature Elimination
    MANUAL = "manual"               # Manual selection


class ValidationStrategy(Enum):
    HOLDOUT = "holdout"             # Simple train/test split
    KFOLD = "kfold"                 # K-fold cross validation
    STRATIFIED_KFOLD = "stratified_kfold"  # Stratified K-fold
    TIME_SERIES = "time_series"     # Time-based split
    OUT_OF_TIME = "out_of_time"     # Out-of-time validation
    BOOTSTRAP = "bootstrap"         # Bootstrap validation


class MetricType(Enum):
    GINI = "gini"                   # Gini coefficient
    AUC = "auc"                     # Area Under ROC Curve
    KS = "ks"                       # Kolmogorov-Smirnov statistic
    PSI = "psi"                     # Population Stability Index
    AR = "ar"                       # Accuracy Ratio
    BRIER = "brier"                 # Brier Score
    LOG_LOSS = "log_loss"           # Log Loss


# SPECIAL CODES

# Standard special codes for credit scoring
DEFAULT_SPECIAL_CODES: Dict[int, str] = {
    -9999: "missing",               # Missing value
    -8888: "not_applicable",        # Not applicable
    -7777: "refused",               # Customer refused to provide
    -6666: "unknown",               # Unknown/unverified
    -5555: "new_customer",          # New customer (no history)
}


# SEGMENTATION CONFIG

@dataclass
class SegmentationConfig:
    method: SegmentationMethod = SegmentationMethod.CHAID
    segment_type: SegmentationType = SegmentationType.RISK_BASED

    # Tree parameters
    max_depth: int = 4
    min_samples_leaf: int = 500
    min_samples_split: int = 1000

    # CHAID specific
    significance_level: float = 0.05
    bonferroni_adjustment: bool = True

    # CART specific
    criterion: str = "gini"  # 'gini' or 'entropy'
    max_features: Optional[Union[int, float, str]] = None

    # Target and features
    target_column: str = "default_flag"
    categorical_features: List[str] = field(default_factory=list)
    continuous_features: List[str] = field(default_factory=list)

    # Segment constraints
    max_segments: int = 8
    min_segment_size: float = 0.05  # 5% minimum
    merge_small_segments: bool = True

    # Output
    segment_column_name: str = "segment_id"

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        errors = []

        if self.max_depth < 1 or self.max_depth > 10:
            errors.append("max_depth must be between 1 and 10")

        if self.min_samples_leaf < 1:
            errors.append("min_samples_leaf must be positive")

        if self.min_samples_split < 2:
            errors.append("min_samples_split must be at least 2")

        if self.min_samples_split <= self.min_samples_leaf:
            errors.append("min_samples_split must be greater than min_samples_leaf")

        if not 0 < self.significance_level < 1:
            errors.append("significance_level must be between 0 and 1")

        if not 0 < self.min_segment_size < 0.5:
            errors.append("min_segment_size must be between 0 and 0.5")

        if self.max_segments < 2 or self.max_segments > 20:
            errors.append("max_segments must be between 2 and 20")

        if errors:
            raise ValueError(f"SegmentationConfig validation errors: {errors}")

    def get_feature_list(self) -> List[str]:
        return self.categorical_features + self.continuous_features


# WOE CONFIG

@dataclass
class WOEConfig:
    # Binning parameters
    min_bin_size: float = 0.05  # 5% minimum per bin
    max_bins: int = 10
    initial_bins: int = 20  # Initial fine bins before coarse binning

    # Monotonicity
    monotonic_constraint: bool = True
    allow_u_shape: bool = False  # Allow U-shaped WOE pattern

    # Missing value handling
    handle_missing: MissingHandling = MissingHandling.SEPARATE_BIN
    missing_as_separate: bool = True

    # Special codes
    special_codes: Dict[int, str] = field(
        default_factory=lambda: DEFAULT_SPECIAL_CODES.copy()
    )

    # IV thresholds
    min_iv: float = 0.02      # Minimum IV to include feature
    max_iv: float = 0.50      # Maximum IV (may indicate overfitting)
    suspicious_iv: float = 0.30  # Flag for manual review

    # Bin merging
    merge_bins_by: str = "chi2"  # 'chi2', 'woe_diff', 'iv_loss'
    min_woe_diff: float = 0.05   # Minimum WOE difference between bins

    # Output
    output_woe_table: bool = True
    output_iv_report: bool = True

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        errors = []

        if not 0.01 <= self.min_bin_size <= 0.25:
            errors.append("min_bin_size must be between 0.01 and 0.25")

        if not 2 <= self.max_bins <= 20:
            errors.append("max_bins must be between 2 and 20")

        if self.initial_bins < self.max_bins:
            errors.append("initial_bins must be >= max_bins")

        if not 0 <= self.min_iv <= 0.1:
            errors.append("min_iv must be between 0 and 0.1")

        if self.max_iv <= self.min_iv:
            errors.append("max_iv must be greater than min_iv")

        if errors:
            raise ValueError(f"WOEConfig validation errors: {errors}")

    @staticmethod
    def calculate_woe(good: int, bad: int, total_good: int, total_bad: int) -> float:
        # Add small constant to avoid log(0)
        dist_good = (good + 0.5) / (total_good + 1)
        dist_bad = (bad + 0.5) / (total_bad + 1)

        return math.log(dist_good / dist_bad)

    @staticmethod
    def calculate_iv(woe_table: List[Dict]) -> float:
        iv = 0.0
        for row in woe_table:
            iv += (row['dist_good'] - row['dist_bad']) * row['woe']
        return iv

    @staticmethod
    def interpret_iv(iv: float) -> str:
        if iv < 0.02:
            return "Not useful for prediction"
        elif iv < 0.1:
            return "Weak predictive power"
        elif iv < 0.3:
            return "Medium predictive power"
        elif iv < 0.5:
            return "Strong predictive power"
        else:
            return "Suspicious (possible overfitting)"


# LOGISTIC REGRESSION CONFIG

@dataclass
class LogisticConfig:
    # Regularization
    regularization: RegularizationType = RegularizationType.L2
    C: float = 1.0
    l1_ratio: float = 0.5  # For elastic net (0=L2, 1=L1)

    # Solver settings
    solver: str = "lbfgs"  # 'lbfgs', 'liblinear', 'saga'
    max_iter: int = 1000
    tol: float = 1e-4
    warm_start: bool = False

    # Class weighting
    class_weight: Optional[Union[str, Dict[int, float]]] = "balanced"

    # Feature selection
    feature_selection_method: FeatureSelectionMethod = FeatureSelectionMethod.STEPWISE
    iv_threshold: float = 0.02        # Minimum IV to include
    vif_threshold: float = 5.0        # Maximum VIF allowed
    p_value_threshold: float = 0.05   # Maximum p-value
    correlation_threshold: float = 0.7  # Maximum correlation between features

    # Stepwise parameters
    stepwise_direction: str = "both"  # 'forward', 'backward', 'both'
    stepwise_criterion: str = "aic"   # 'aic', 'bic', 'pvalue'

    # Coefficient constraints
    force_positive_coefficients: bool = False  # For interpretability
    max_features: Optional[int] = 15  # Maximum features in final model

    # Output
    output_coefficients: bool = True
    output_odds_ratios: bool = True
    output_marginal_effects: bool = True

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        errors = []

        if self.C <= 0:
            errors.append("C must be positive")

        if not 0 <= self.l1_ratio <= 1:
            errors.append("l1_ratio must be between 0 and 1")

        if self.max_iter < 100:
            errors.append("max_iter should be at least 100")

        if not 0 < self.iv_threshold < 0.1:
            errors.append("iv_threshold must be between 0 and 0.1")

        if not 1 < self.vif_threshold < 20:
            errors.append("vif_threshold must be between 1 and 20")

        if not 0 < self.p_value_threshold <= 0.1:
            errors.append("p_value_threshold must be between 0 and 0.1")

        if not 0.5 <= self.correlation_threshold <= 1:
            errors.append("correlation_threshold must be between 0.5 and 1")

        if errors:
            raise ValueError(f"LogisticConfig validation errors: {errors}")

    def get_sklearn_params(self) -> Dict[str, Any]:
        params = {
            "C": self.C,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "warm_start": self.warm_start,
            "class_weight": self.class_weight,
        }

        # Set penalty based on regularization
        if self.regularization == RegularizationType.L1:
            params["penalty"] = "l1"
            params["solver"] = "saga"  # Required for L1
        elif self.regularization == RegularizationType.L2:
            params["penalty"] = "l2"
            params["solver"] = self.solver
        elif self.regularization == RegularizationType.ELASTICNET:
            params["penalty"] = "elasticnet"
            params["solver"] = "saga"
            params["l1_ratio"] = self.l1_ratio
        else:
            params["penalty"] = None
            params["solver"] = "lbfgs"

        return params


# SCORECARD CONFIG

@dataclass
class ScorecardConfig:
    # Scoring scale
    base_score: int = 600
    base_odds: float = 50.0    # 50:1 good:bad at base score
    pdo: int = 20              # Points to double the odds

    # Score range
    score_range: Tuple[int, int] = (300, 850)
    clip_scores: bool = True   # Clip to score range

    # Rounding
    round_points: bool = True
    round_to: int = 1          # Round to nearest integer

    # Variable points
    min_points_per_variable: int = -100
    max_points_per_variable: int = 100

    # Output options
    output_points_table: bool = True
    output_reason_codes: bool = True
    top_reason_codes: int = 4   # Number of top reasons to return

    # Vietnamese market specific
    vn_score_interpretation: Dict[Tuple[int, int], str] = field(
        default_factory=lambda: {
            (750, 850): "Excellent - Rất tốt",
            (700, 749): "Good - Tốt",
            (650, 699): "Fair - Khá",
            (600, 649): "Average - Trung bình",
            (550, 599): "Below Average - Dưới trung bình",
            (400, 549): "Poor - Kém",
            (300, 399): "Very Poor - Rất kém",
        }
    )

    def __post_init__(self):
        self.validate()
        self._calculate_scaling_factors()

    def validate(self) -> None:
        errors = []

        if not 0 < self.base_odds < 1000:
            errors.append("base_odds must be between 0 and 1000")

        if not 10 <= self.pdo <= 50:
            errors.append("pdo must be between 10 and 50")

        if self.score_range[0] >= self.score_range[1]:
            errors.append("score_range min must be less than max")

        if not self.score_range[0] <= self.base_score <= self.score_range[1]:
            errors.append("base_score must be within score_range")

        if errors:
            raise ValueError(f"ScorecardConfig validation errors: {errors}")

    def _calculate_scaling_factors(self) -> None:
        self.factor = self.pdo / math.log(2)
        self.offset = self.base_score - self.factor * math.log(self.base_odds)

    def calculate_score(self, log_odds: float) -> int:
        score = self.offset + self.factor * log_odds

        if self.round_points:
            score = round(score / self.round_to) * self.round_to

        if self.clip_scores:
            score = max(self.score_range[0], min(self.score_range[1], score))

        return int(score)

    def calculate_points(
        self,
        woe: float,
        coefficient: float,
        n_features: int
    ) -> float:
        points = (woe * coefficient - self.offset / n_features) * self.factor

        if self.round_points:
            points = round(points / self.round_to) * self.round_to

        return points

    def score_to_pd(self, score: int) -> float:
        log_odds = (score - self.offset) / self.factor
        odds = math.exp(log_odds)
        pd = 1 / (1 + odds)
        return pd

    def pd_to_score(self, pd: float) -> int:
        if pd <= 0 or pd >= 1:
            raise ValueError("PD must be between 0 and 1")

        odds = (1 - pd) / pd
        log_odds = math.log(odds)
        score = self.offset + self.factor * log_odds

        return self.calculate_score(log_odds)

    def interpret_score(self, score: int) -> str:
        for (min_score, max_score), interpretation in self.vn_score_interpretation.items():
            if min_score <= score <= max_score:
                return interpretation
        return "Unknown - Không xác định"


# MODEL VALIDATION CONFIG

@dataclass
class ModelValidationConfig:
    # Data splitting
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    # Cross validation
    cv_folds: int = 5
    stratify: bool = True

    # Temporal validation
    temporal_validation: bool = False
    oot_months: int = 6              # Out-of-time validation months
    time_column: str = "application_date"

    # Validation strategy
    validation_strategy: ValidationStrategy = ValidationStrategy.STRATIFIED_KFOLD

    # Performance metrics
    primary_metric: MetricType = MetricType.GINI
    secondary_metrics: List[MetricType] = field(
        default_factory=lambda: [MetricType.KS, MetricType.AUC, MetricType.PSI]
    )

    # Metric thresholds
    min_gini: float = 0.30          # Minimum acceptable Gini
    min_ks: float = 0.20            # Minimum acceptable KS
    max_psi: float = 0.25           # Maximum PSI for stability
    psi_warning: float = 0.10       # PSI warning threshold

    # Overfit detection
    max_train_test_gini_diff: float = 0.05  # Max difference train vs test
    monitor_overfit: bool = True

    # Bootstrap settings
    bootstrap_iterations: int = 100
    bootstrap_ci: float = 0.95      # Confidence interval

    # Random state
    random_state: int = 42
    shuffle: bool = True

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        errors = []

        total_ratio = self.train_ratio + self.validation_ratio + self.test_ratio
        if not 0.99 <= total_ratio <= 1.01:
            errors.append(f"Ratios must sum to 1.0, got {total_ratio}")

        if not 0.5 <= self.train_ratio <= 0.9:
            errors.append("train_ratio must be between 0.5 and 0.9")

        if not 2 <= self.cv_folds <= 20:
            errors.append("cv_folds must be between 2 and 20")

        if self.oot_months < 1 or self.oot_months > 24:
            errors.append("oot_months must be between 1 and 24")

        if not 0.1 <= self.min_gini <= 0.6:
            errors.append("min_gini must be between 0.1 and 0.6")

        if errors:
            raise ValueError(f"ModelValidationConfig validation errors: {errors}")

    def get_split_sizes(self, n_samples: int) -> Dict[str, int]:
        train_size = int(n_samples * self.train_ratio)
        validation_size = int(n_samples * self.validation_ratio)
        test_size = n_samples - train_size - validation_size

        return {
            "train": train_size,
            "validation": validation_size,
            "test": test_size,
        }


# COMBINED MODEL CONFIG

@dataclass
class CreditScoringModelConfig:
    # Component configs
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    woe: WOEConfig = field(default_factory=WOEConfig)
    logistic: LogisticConfig = field(default_factory=LogisticConfig)
    scorecard: ScorecardConfig = field(default_factory=ScorecardConfig)
    validation: ModelValidationConfig = field(default_factory=ModelValidationConfig)

    # Model metadata
    model_name: str = "vnpt_credit_score_v1"
    model_version: str = "1.0.0"
    model_type: str = "application_scorecard"  # 'application', 'behavioral', 'collection'

    # Target definition
    target_column: str = "is_default"
    target_definition: str = "90+ DPD within 12 months"
    observation_window_months: int = 12
    outcome_window_months: int = 12

    # Feature groups
    demographic_features: List[str] = field(default_factory=list)
    financial_features: List[str] = field(default_factory=list)
    credit_features: List[str] = field(default_factory=list)
    telecom_features: List[str] = field(default_factory=list)
    behavioral_features: List[str] = field(default_factory=list)

    # Regulatory compliance
    nhnn_compliant: bool = True      # NHNN regulation compliance
    basel_compliant: bool = True     # Basel II/III compliance

    def validate_all(self) -> List[str]:
        errors = []

        try:
            self.segmentation.validate()
        except ValueError as e:
            errors.append(f"Segmentation: {e}")

        try:
            self.woe.validate()
        except ValueError as e:
            errors.append(f"WOE: {e}")

        try:
            self.logistic.validate()
        except ValueError as e:
            errors.append(f"Logistic: {e}")

        try:
            self.scorecard.validate()
        except ValueError as e:
            errors.append(f"Scorecard: {e}")

        try:
            self.validation.validate()
        except ValueError as e:
            errors.append(f"Validation: {e}")

        return errors

    def get_all_features(self) -> List[str]:
        return (
            self.demographic_features +
            self.financial_features +
            self.credit_features +
            self.telecom_features +
            self.behavioral_features
        )


# FACTORY FUNCTIONS

def get_default_model_config() -> CreditScoringModelConfig:
    return CreditScoringModelConfig(
        segmentation=SegmentationConfig(
            method=SegmentationMethod.CHAID,
            max_depth=4,
            min_samples_leaf=500,
            categorical_features=[
                'gender_code', 'education_level_code', 'marital_status_code',
                'employment_type_code', 'property_ownership_code', 'region',
            ],
            continuous_features=[
                'age', 'monthly_income', 'dti_ratio', 'job_tenure_months',
            ],
        ),
        woe=WOEConfig(
            min_bin_size=0.05,
            max_bins=10,
            monotonic_constraint=True,
        ),
        logistic=LogisticConfig(
            regularization=RegularizationType.L2,
            C=1.0,
            feature_selection_method=FeatureSelectionMethod.STEPWISE,
            iv_threshold=0.02,
            vif_threshold=5.0,
        ),
        scorecard=ScorecardConfig(
            base_score=600,
            base_odds=50.0,
            pdo=20,
            score_range=(300, 850),
        ),
        validation=ModelValidationConfig(
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            cv_folds=5,
            stratify=True,
            primary_metric=MetricType.GINI,
            min_gini=0.30,
        ),
        model_name="vnpt_credit_score_v1",
        demographic_features=[
            'age', 'gender_code', 'education_level_code', 'marital_status_code',
            'province_code', 'is_urban',
        ],
        financial_features=[
            'monthly_income', 'employment_type_code', 'job_tenure_months',
            'dti_ratio', 'savings_ratio', 'property_ownership_code',
        ],
        credit_features=[
            'has_credit_history', 'cic_score', 'nhnn_loan_group',
            'num_active_loans', 'credit_utilization', 'max_dpd_ever',
        ],
        telecom_features=[
            'is_vnpt_customer', 'telecom_credit_score', 'payment_rate',
            'tenure_months', 'monthly_arpu', 'contract_type_code',
        ],
        behavioral_features=[
            'max_dpd_12m', 'payment_consistency', 'dpd_trend',
            'balance_volatility', 'utilization_trend',
        ],
    )


def get_thin_file_model_config() -> CreditScoringModelConfig:
    config = get_default_model_config()

    # Reduce WOE requirements
    config.woe.min_iv = 0.01
    config.woe.min_bin_size = 0.03

    # Adjust validation
    config.validation.min_gini = 0.25

    # Focus on telecom features
    config.telecom_features = [
        'is_vnpt_customer', 'telecom_credit_score', 'payment_rate',
        'tenure_months', 'monthly_arpu', 'contract_type_code',
        'late_payment_count', 'avg_days_to_payment', 'data_usage_trend',
    ]

    config.model_name = "vnpt_thin_file_score_v1"

    return config


def get_behavioral_model_config() -> CreditScoringModelConfig:
    config = get_default_model_config()

    config.model_type = "behavioral_scorecard"
    config.target_definition = "30+ DPD in next 3 months"
    config.observation_window_months = 12
    config.outcome_window_months = 3

    # Focus on behavioral features
    config.behavioral_features = [
        'max_dpd_3m', 'max_dpd_6m', 'max_dpd_12m',
        'avg_utilization_6m', 'utilization_trend',
        'payment_consistency', 'balance_volatility',
        'recent_late_payment', 'consecutive_late_payments',
        'telecom_payment_trend', 'telecom_late_rate',
    ]

    config.model_name = "vnpt_behavioral_score_v1"

    return config


# MODULE EXPORTS

__all__ = [
    # Enums
    "SegmentationMethod",
    "SegmentationType",
    "MissingHandling",
    "RegularizationType",
    "FeatureSelectionMethod",
    "ValidationStrategy",
    "MetricType",
    # Configs
    "SegmentationConfig",
    "WOEConfig",
    "LogisticConfig",
    "ScorecardConfig",
    "ModelValidationConfig",
    "CreditScoringModelConfig",
    # Constants
    "DEFAULT_SPECIAL_CODES",
    # Factory functions
    "get_default_model_config",
    "get_thin_file_model_config",
    "get_behavioral_model_config",
]
