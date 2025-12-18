from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import warnings

import numpy as np
import pandas as pd

# sklearn compatibility
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# CONSTANTS AND DEFAULTS

# Industry standard scorecard parameters
DEFAULT_BASE_SCORE = 600
DEFAULT_BASE_ODDS = 50  # 50:1 good:bad ratio
DEFAULT_PDO = 20  # Points to double odds

# Score range
DEFAULT_MIN_SCORE = 300
DEFAULT_MAX_SCORE = 850

# Rounding precision for points
DEFAULT_POINT_PRECISION = 1


# ENUMS

class RiskRating(Enum):
    A = "A"  # Excellent
    B = "B"  # Good
    C = "C"  # Fair
    D = "D"  # Poor
    E = "E"  # Very Poor


class DecisionType(Enum):
    AUTO_APPROVE = "auto_approve"
    APPROVE = "approve"
    REVIEW = "review"
    ENHANCED_REVIEW = "enhanced_review"
    DECLINE = "decline"


# DATACLASSES

@dataclass
class ScorecardBin:
    feature: str
    bin_id: int
    bin_label: str
    bin_min: Optional[float]
    bin_max: Optional[float]
    woe: float
    coefficient: float
    points: float
    points_rounded: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature': self.feature,
            'bin_id': self.bin_id,
            'bin_label': self.bin_label,
            'bin_min': self.bin_min,
            'bin_max': self.bin_max,
            'woe': self.woe,
            'coefficient': self.coefficient,
            'points': self.points,
            'points_rounded': self.points_rounded,
        }


@dataclass
class ScoreBand:
    band_id: int
    band_name: str
    score_min: float
    score_max: float
    expected_bad_rate: float
    rating: str
    decision: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'band_id': self.band_id,
            'band_name': self.band_name,
            'score_min': self.score_min,
            'score_max': self.score_max,
            'expected_bad_rate': self.expected_bad_rate,
            'rating': self.rating,
            'decision': self.decision,
        }


@dataclass
class ReasonCode:
    code: str
    feature: str
    description: str
    impact_points: float
    priority: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'feature': self.feature,
            'description': self.description,
            'impact_points': self.impact_points,
            'priority': self.priority,
        }


# SCORECARD TABLE

class ScorecardTable:
    def __init__(
        self,
        bins: List[ScorecardBin],
        offset: float,
        base_score: float,
        pdo: float,
        base_odds: float,
    ):
        self.bins = bins
        self.offset = offset
        self.base_score = base_score
        self.pdo = pdo
        self.base_odds = base_odds

        # Build lookup dictionary
        self._build_lookup()

    def _build_lookup(self):
        self.lookup = {}
        self.features = []

        for bin_obj in self.bins:
            if bin_obj.feature not in self.lookup:
                self.lookup[bin_obj.feature] = {}
                self.features.append(bin_obj.feature)

            self.lookup[bin_obj.feature][bin_obj.bin_label] = bin_obj

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([b.to_dict() for b in self.bins])

    def get_feature_points(self, feature: str) -> pd.DataFrame:
        feature_bins = [b for b in self.bins if b.feature == feature]
        return pd.DataFrame([b.to_dict() for b in feature_bins])

    def get_min_max_points(self) -> Dict[str, Dict[str, float]]:
        result = {}

        for feature in self.features:
            feature_bins = [b for b in self.bins if b.feature == feature]
            points = [b.points_rounded for b in feature_bins]
            result[feature] = {
                'min_points': min(points),
                'max_points': max(points),
            }

        return result

    def get_total_score_range(self) -> Tuple[float, float]:
        min_max = self.get_min_max_points()

        min_score = self.offset + sum(v['min_points'] for v in min_max.values())
        max_score = self.offset + sum(v['max_points'] for v in min_max.values())

        return min_score, max_score

    def validate(self) -> Dict[str, Any]:
        min_score, max_score = self.get_total_score_range()

        return {
            'min_possible_score': min_score,
            'max_possible_score': max_score,
            'offset': self.offset,
            'n_features': len(self.features),
            'n_bins': len(self.bins),
            'is_valid': True,
        }


# SCORECARD BUILDER

class ScorecardBuilder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        base_score: float = DEFAULT_BASE_SCORE,
        base_odds: float = DEFAULT_BASE_ODDS,
        pdo: float = DEFAULT_PDO,
        round_points: bool = True,
        point_precision: int = DEFAULT_POINT_PRECISION,
        min_score: Optional[float] = DEFAULT_MIN_SCORE,
        max_score: Optional[float] = DEFAULT_MAX_SCORE,
    ):
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.round_points = round_points
        self.point_precision = point_precision
        self.min_score = min_score
        self.max_score = max_score

    def fit(
        self,
        woe_transformer,
        logistic_model,
    ) -> 'ScorecardBuilder':
        # Calculate scaling factor
        self.factor_ = self.pdo / np.log(2)

        # Calculate offset
        # Offset = Base_Score - Factor * ln(Base_Odds) - Factor * Intercept
        intercept = getattr(logistic_model, 'intercept_', 0)
        self.offset_ = self.base_score - self.factor_ * np.log(self.base_odds) - self.factor_ * intercept

        # Get coefficients
        if hasattr(logistic_model, 'coefficients_'):
            coefficients = logistic_model.coefficients_
        elif hasattr(logistic_model, 'coef_'):
            coef_array = logistic_model.coef_[0]
            feature_names = getattr(logistic_model, 'feature_names_', None)
            if feature_names is None:
                feature_names = getattr(woe_transformer, 'selected_features_',
                                       list(woe_transformer.feature_results_.keys()))
            coefficients = dict(zip(feature_names, coef_array))
        else:
            raise ValueError("Cannot extract coefficients from model")

        # Get WOE values from transformer
        woe_results = woe_transformer.feature_results_

        # Build scorecard bins
        bins = []
        self.feature_points_ = {}

        for feature, result in woe_results.items():
            if feature not in coefficients:
                continue

            coef = coefficients[feature]
            self.feature_points_[feature] = {}

            for bin_stat in result.bins:
                # Calculate points: -(WOE * Coefficient * Factor)
                points = -(bin_stat.woe * coef * self.factor_)

                # Round points if requested
                if self.round_points:
                    points_rounded = int(round(points / self.point_precision) * self.point_precision)
                else:
                    points_rounded = int(round(points))

                scorecard_bin = ScorecardBin(
                    feature=feature,
                    bin_id=bin_stat.bin_id,
                    bin_label=bin_stat.bin_label,
                    bin_min=bin_stat.bin_min,
                    bin_max=bin_stat.bin_max,
                    woe=bin_stat.woe,
                    coefficient=coef,
                    points=points,
                    points_rounded=points_rounded,
                )
                bins.append(scorecard_bin)
                self.feature_points_[feature][bin_stat.bin_label] = points_rounded

        # Create scorecard table
        self.scorecard_table_ = ScorecardTable(
            bins=bins,
            offset=self.offset_,
            base_score=self.base_score,
            pdo=self.pdo,
            base_odds=self.base_odds,
        )

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ['scorecard_table_', 'feature_points_', 'is_fitted_'])

        scores = np.full(len(X), self.offset_)

        for feature in self.scorecard_table_.features:
            if feature not in X.columns:
                continue

            # Get points for each value
            feature_lookup = self.feature_points_[feature]

            for idx in range(len(X)):
                value = X[feature].iloc[idx]

                # Find matching bin
                points = self._get_points_for_value(feature, value)
                scores[idx] += points

        # Clip to range if specified
        if self.min_score is not None:
            scores = np.maximum(scores, self.min_score)
        if self.max_score is not None:
            scores = np.minimum(scores, self.max_score)

        return scores

    def fit_transform(
        self,
        woe_transformer,
        logistic_model,
        X: pd.DataFrame
    ) -> np.ndarray:
        self.fit(woe_transformer, logistic_model)
        return self.transform(X)

    def _get_points_for_value(self, feature: str, value) -> float:
        feature_lookup = self.feature_points_.get(feature, {})

        # Try exact match first
        if str(value) in feature_lookup:
            return feature_lookup[str(value)]

        # Handle missing
        if pd.isna(value):
            if 'Missing' in feature_lookup:
                return feature_lookup['Missing']
            elif '_MISSING_' in feature_lookup:
                return feature_lookup['_MISSING_']
            else:
                # Return average points for feature
                return np.mean(list(feature_lookup.values()))

        # For continuous features, find matching bin by range
        for bin_obj in self.scorecard_table_.bins:
            if bin_obj.feature != feature:
                continue

            if bin_obj.bin_min is not None and bin_obj.bin_max is not None:
                try:
                    val = float(value)
                    if bin_obj.bin_min <= val <= bin_obj.bin_max:
                        return bin_obj.points_rounded
                except (ValueError, TypeError):
                    pass

        # Default: return first bin's points
        if feature_lookup:
            return list(feature_lookup.values())[0]

        return 0

    def get_scorecard_table(self) -> ScorecardTable:
        check_is_fitted(self, ['scorecard_table_'])
        return self.scorecard_table_

    def get_scorecard_dataframe(self) -> pd.DataFrame:
        check_is_fitted(self, ['scorecard_table_'])
        return self.scorecard_table_.to_dataframe()

    # SCALING METHODS

    def scale_to_range(
        self,
        target_min: float,
        target_max: float
    ) -> 'ScorecardBuilder':
        check_is_fitted(self, ['scorecard_table_'])

        current_min, current_max = self.scorecard_table_.get_total_score_range()
        current_range = current_max - current_min
        target_range = target_max - target_min

        if current_range == 0:
            warnings.warn("Current score range is 0, cannot scale")
            return self

        # Calculate scaling factor and new offset
        scale = target_range / current_range
        new_offset = target_min - current_min * scale

        # Update offset
        self.offset_ = new_offset

        # Update all bin points
        for bin_obj in self.scorecard_table_.bins:
            bin_obj.points = bin_obj.points * scale
            bin_obj.points_rounded = int(round(bin_obj.points / self.point_precision) * self.point_precision)

        # Update lookup
        self._rebuild_feature_points()

        # Update scorecard table
        self.scorecard_table_.offset = self.offset_

        # Update score range
        self.min_score = target_min
        self.max_score = target_max

        return self

    def align_to_target(
        self,
        target_score: float,
        target_odds: float
    ) -> 'ScorecardBuilder':
        check_is_fitted(self, ['factor_'])

        # Recalculate offset for new target
        # Score = Offset + Points
        # target_score = new_offset - Factor * ln(target_odds)
        # new_offset = target_score + Factor * ln(target_odds)

        adjustment = target_score - self.base_score + self.factor_ * (
            np.log(target_odds) - np.log(self.base_odds)
        )

        self.offset_ += adjustment
        self.scorecard_table_.offset = self.offset_

        # Update base parameters
        self.base_score = target_score
        self.base_odds = target_odds

        return self

    def round_points_to_precision(
        self,
        precision: int = 5
    ) -> 'ScorecardBuilder':
        check_is_fitted(self, ['scorecard_table_'])

        self.point_precision = precision

        # Round offset
        self.offset_ = round(self.offset_ / precision) * precision

        # Round all bin points
        for bin_obj in self.scorecard_table_.bins:
            bin_obj.points_rounded = int(round(bin_obj.points / precision) * precision)

        self._rebuild_feature_points()
        self.scorecard_table_.offset = self.offset_

        return self

    def _rebuild_feature_points(self):
        self.feature_points_ = {}

        for bin_obj in self.scorecard_table_.bins:
            if bin_obj.feature not in self.feature_points_:
                self.feature_points_[bin_obj.feature] = {}
            self.feature_points_[bin_obj.feature][bin_obj.bin_label] = bin_obj.points_rounded


# SCORE INTERPRETER

class ScoreInterpreter:
    # Default score bands for Vietnamese credit market
    DEFAULT_SCORE_BANDS = [
        ScoreBand(1, "Excellent", 750, 850, 0.01, "A", DecisionType.AUTO_APPROVE.value),
        ScoreBand(2, "Good", 700, 749, 0.02, "B", DecisionType.APPROVE.value),
        ScoreBand(3, "Fair", 650, 699, 0.04, "C", DecisionType.REVIEW.value),
        ScoreBand(4, "Poor", 600, 649, 0.08, "D", DecisionType.ENHANCED_REVIEW.value),
        ScoreBand(5, "Very Poor", 300, 599, 0.15, "E", DecisionType.DECLINE.value),
    ]

    def __init__(
        self,
        base_score: float = DEFAULT_BASE_SCORE,
        pdo: float = DEFAULT_PDO,
        base_odds: float = DEFAULT_BASE_ODDS,
        score_bands: Optional[List[ScoreBand]] = None,
    ):
        self.base_score = base_score
        self.pdo = pdo
        self.base_odds = base_odds
        self.factor = pdo / np.log(2)
        self.score_bands = score_bands or self.DEFAULT_SCORE_BANDS.copy()

    @classmethod
    def from_scorecard(cls, scorecard: ScorecardBuilder) -> 'ScoreInterpreter':
        return cls(
            base_score=scorecard.base_score,
            pdo=scorecard.pdo,
            base_odds=scorecard.base_odds,
        )

    def score_to_odds(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.base_odds * np.power(2, (score - self.base_score) / self.pdo)

    def score_to_pd(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        odds = self.score_to_odds(score)
        return 1 / (1 + odds)

    def pd_to_score(self, pd: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # PD = 1 / (1 + Odds)
        # Odds = (1 - PD) / PD
        # Score = Base_Score + PDO * log2(Odds / Base_Odds)
        odds = (1 - pd) / pd
        return self.base_score + self.pdo * np.log2(odds / self.base_odds)

    def odds_to_score(self, odds: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.base_score + self.pdo * np.log2(odds / self.base_odds)

    def score_to_rating(self, score: Union[float, np.ndarray]) -> Union[str, List[str]]:
        if isinstance(score, (int, float)):
            for band in self.score_bands:
                if band.score_min <= score <= band.score_max:
                    return band.rating
            return "E"  # Default to worst rating

        # Array input
        ratings = []
        for s in score:
            rating = "E"
            for band in self.score_bands:
                if band.score_min <= s <= band.score_max:
                    rating = band.rating
                    break
            ratings.append(rating)
        return ratings

    def score_to_decision(self, score: Union[float, np.ndarray]) -> Union[str, List[str]]:
        if isinstance(score, (int, float)):
            for band in self.score_bands:
                if band.score_min <= score <= band.score_max:
                    return band.decision
            return DecisionType.DECLINE.value

        # Array input
        decisions = []
        for s in score:
            decision = DecisionType.DECLINE.value
            for band in self.score_bands:
                if band.score_min <= s <= band.score_max:
                    decision = band.decision
                    break
            decisions.append(decision)
        return decisions

    def get_score_bands(self) -> pd.DataFrame:
        return pd.DataFrame([b.to_dict() for b in self.score_bands])

    def set_custom_bands(
        self,
        bands: List[Dict[str, Any]]
    ) -> 'ScoreInterpreter':
        self.score_bands = [
            ScoreBand(**band) for band in bands
        ]
        return self

    def calibrate_bands_from_data(
        self,
        scores: np.ndarray,
        y: np.ndarray,
        n_bands: int = 5
    ) -> 'ScoreInterpreter':
        # Create quantile-based bands
        percentiles = np.linspace(0, 100, n_bands + 1)
        edges = np.percentile(scores, percentiles)

        ratings = ['A', 'B', 'C', 'D', 'E'][:n_bands]
        decisions = [
            DecisionType.AUTO_APPROVE.value,
            DecisionType.APPROVE.value,
            DecisionType.REVIEW.value,
            DecisionType.ENHANCED_REVIEW.value,
            DecisionType.DECLINE.value,
        ][:n_bands]
        names = ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor'][:n_bands]

        new_bands = []
        for i in range(n_bands):
            mask = (scores >= edges[n_bands - i - 1]) & (scores <= edges[n_bands - i])
            bad_rate = y[mask].mean() if mask.sum() > 0 else 0

            band = ScoreBand(
                band_id=i + 1,
                band_name=names[i] if i < len(names) else f"Band {i+1}",
                score_min=edges[n_bands - i - 1],
                score_max=edges[n_bands - i],
                expected_bad_rate=bad_rate,
                rating=ratings[i] if i < len(ratings) else f"R{i+1}",
                decision=decisions[i] if i < len(decisions) else DecisionType.REVIEW.value,
            )
            new_bands.append(band)

        self.score_bands = new_bands
        return self


# REASON CODE GENERATOR

class ReasonCodeGenerator:
    # Default reason code descriptions
    DEFAULT_DESCRIPTIONS = {
        'age': "Age-related risk factors",
        'income': "Income level insufficient",
        'credit_history': "Limited credit history length",
        'credit_history_months': "Insufficient credit history duration",
        'dpd': "Recent late payment history",
        'dpd_max': "History of significant payment delays",
        'utilization': "High credit utilization ratio",
        'credit_utilization': "Credit lines highly utilized",
        'num_accounts': "Limited number of credit accounts",
        'inquiries': "Too many recent credit inquiries",
        'employment': "Employment stability concerns",
        'employment_years': "Limited employment tenure",
        'debt': "High debt levels",
        'dti': "Debt-to-income ratio too high",
        'default': "Factor contributed negatively to score",
    }

    def __init__(
        self,
        scorecard: ScorecardBuilder,
        reason_descriptions: Optional[Dict[str, str]] = None,
    ):
        self.scorecard = scorecard
        self.reason_descriptions = {
            **self.DEFAULT_DESCRIPTIONS,
            **(reason_descriptions or {})
        }

    def generate_reason_codes(
        self,
        X: pd.Series,
        n_reasons: int = 4,
    ) -> List[ReasonCode]:
        check_is_fitted(self.scorecard, ['scorecard_table_', 'feature_points_'])

        # Calculate point contribution for each feature
        contributions = []

        for feature in self.scorecard.scorecard_table_.features:
            if feature not in X.index:
                continue

            value = X[feature]
            points = self.scorecard._get_points_for_value(feature, value)

            # Get max possible points for this feature
            max_points = max(self.scorecard.feature_points_[feature].values())

            # Negative impact = points below maximum
            impact = points - max_points

            if impact < 0:
                contributions.append({
                    'feature': feature,
                    'impact': impact,
                    'actual_points': points,
                    'max_points': max_points,
                })

        # Sort by impact (most negative first)
        contributions.sort(key=lambda x: x['impact'])

        # Generate reason codes
        reasons = []
        for i, contrib in enumerate(contributions[:n_reasons]):
            feature = contrib['feature']

            # Find description
            description = self._get_description(feature)

            reason = ReasonCode(
                code=f"R{i+1:02d}",
                feature=feature,
                description=description,
                impact_points=contrib['impact'],
                priority=i + 1,
            )
            reasons.append(reason)

        return reasons

    def get_adverse_action_reasons(
        self,
        X: pd.Series,
        n_reasons: int = 4,
    ) -> List[str]:
        reasons = self.generate_reason_codes(X, n_reasons)
        return [r.description for r in reasons]

    def get_all_reason_codes(
        self,
        X: pd.DataFrame,
        n_reasons: int = 4,
    ) -> pd.DataFrame:
        all_reasons = []

        for idx in range(len(X)):
            row = X.iloc[idx]
            reasons = self.generate_reason_codes(row, n_reasons)

            for reason in reasons:
                all_reasons.append({
                    'index': idx,
                    'code': reason.code,
                    'feature': reason.feature,
                    'description': reason.description,
                    'impact_points': reason.impact_points,
                    'priority': reason.priority,
                })

        return pd.DataFrame(all_reasons)

    def _get_description(self, feature: str) -> str:
        # Try exact match
        if feature in self.reason_descriptions:
            return self.reason_descriptions[feature]

        # Try partial match
        feature_lower = feature.lower()
        for key, desc in self.reason_descriptions.items():
            if key in feature_lower:
                return desc

        return self.reason_descriptions.get('default', f"Factor: {feature}")


# EXPORT METHODS

class ScorecardExporter:
    def __init__(self, scorecard: ScorecardBuilder):
        self.scorecard = scorecard

    def to_excel(self, filepath: str):
        check_is_fitted(self.scorecard, ['scorecard_table_'])

        try:
            import openpyxl
        except ImportError:
            # Fallback to pandas Excel writer
            pass

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary = pd.DataFrame([{
                'Parameter': 'Base Score',
                'Value': self.scorecard.base_score,
            }, {
                'Parameter': 'PDO',
                'Value': self.scorecard.pdo,
            }, {
                'Parameter': 'Base Odds',
                'Value': self.scorecard.base_odds,
            }, {
                'Parameter': 'Offset',
                'Value': self.scorecard.offset_,
            }])
            summary.to_excel(writer, sheet_name='Summary', index=False)

            # Full scorecard
            scorecard_df = self.scorecard.get_scorecard_dataframe()
            scorecard_df.to_excel(writer, sheet_name='Scorecard', index=False)

            # Feature-by-feature sheets
            for feature in self.scorecard.scorecard_table_.features:
                feature_df = self.scorecard.scorecard_table_.get_feature_points(feature)
                # Truncate sheet name to 31 chars (Excel limit)
                sheet_name = feature[:31]
                feature_df.to_excel(writer, sheet_name=sheet_name, index=False)

    def to_json(self, filepath: Optional[str] = None) -> str:
        check_is_fitted(self.scorecard, ['scorecard_table_'])

        data = {
            'parameters': {
                'base_score': self.scorecard.base_score,
                'pdo': self.scorecard.pdo,
                'base_odds': self.scorecard.base_odds,
                'offset': self.scorecard.offset_,
            },
            'features': {},
        }

        for feature in self.scorecard.scorecard_table_.features:
            feature_bins = self.scorecard.scorecard_table_.get_feature_points(feature)
            data['features'][feature] = feature_bins.to_dict(orient='records')

        json_str = json.dumps(data, indent=2, default=str)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str

    def to_sql(
        self,
        filepath: Optional[str] = None,
        table_alias: str = "t"
    ) -> str:
        check_is_fitted(self.scorecard, ['scorecard_table_'])

        lines = [
            "-- Credit Scorecard SQL Implementation",
            f"-- Base Score: {self.scorecard.base_score}",
            f"-- PDO: {self.scorecard.pdo}",
            f"-- Offset: {self.scorecard.offset_:.2f}",
            "",
            "SELECT",
            f"    {self.scorecard.offset_:.2f}  -- Offset",
        ]

        for feature in self.scorecard.scorecard_table_.features:
            lines.append(f"    + CASE  -- {feature}")

            feature_bins = [b for b in self.scorecard.scorecard_table_.bins
                          if b.feature == feature]

            for bin_obj in feature_bins:
                if bin_obj.bin_min is not None and bin_obj.bin_max is not None:
                    # Numeric range
                    if bin_obj.bin_label.startswith("<="):
                        lines.append(
                            f"        WHEN {table_alias}.{feature} <= {bin_obj.bin_max} "
                            f"THEN {bin_obj.points_rounded}"
                        )
                    elif bin_obj.bin_label.startswith(">"):
                        lines.append(
                            f"        WHEN {table_alias}.{feature} > {bin_obj.bin_min} "
                            f"THEN {bin_obj.points_rounded}"
                        )
                    else:
                        lines.append(
                            f"        WHEN {table_alias}.{feature} > {bin_obj.bin_min} "
                            f"AND {table_alias}.{feature} <= {bin_obj.bin_max} "
                            f"THEN {bin_obj.points_rounded}"
                        )
                else:
                    # Categorical
                    lines.append(
                        f"        WHEN {table_alias}.{feature} = '{bin_obj.bin_label}' "
                        f"THEN {bin_obj.points_rounded}"
                    )

            lines.append("        ELSE 0")
            lines.append("      END")

        lines.append("AS credit_score")
        lines.append(f"FROM table_name {table_alias}")

        sql = "\n".join(lines)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(sql)

        return sql

    def to_python_function(
        self,
        filepath: Optional[str] = None,
        function_name: str = "calculate_credit_score"
    ) -> str:
        check_is_fitted(self.scorecard, ['scorecard_table_'])

        lines = [
            '"""Auto-generated credit scoring function."""',
            "",
            f"def {function_name}(data: dict) -> float:",
            '    """',
            "    Calculate credit score from input data.",
            "",
            "    Args:",
            "        data: Dictionary with feature values",
            "",
            "    Returns:",
            "        Credit score",
            '    """',
            f"    score = {self.scorecard.offset_:.2f}  # Offset",
            "",
        ]

        for feature in self.scorecard.scorecard_table_.features:
            lines.append(f"    # {feature}")
            lines.append(f"    value = data.get('{feature}')")

            feature_bins = [b for b in self.scorecard.scorecard_table_.bins
                          if b.feature == feature]

            first = True
            for bin_obj in feature_bins:
                keyword = "if" if first else "elif"
                first = False

                if bin_obj.bin_min is not None and bin_obj.bin_max is not None:
                    # Numeric range
                    if bin_obj.bin_label.startswith("<="):
                        lines.append(
                            f"    {keyword} value is not None and value <= {bin_obj.bin_max}:"
                        )
                    elif bin_obj.bin_label.startswith(">"):
                        lines.append(
                            f"    {keyword} value is not None and value > {bin_obj.bin_min}:"
                        )
                    else:
                        lines.append(
                            f"    {keyword} value is not None and {bin_obj.bin_min} < value <= {bin_obj.bin_max}:"
                        )
                else:
                    # Categorical or missing
                    if bin_obj.bin_label in ['Missing', '_MISSING_']:
                        lines.append(f"    {keyword} value is None:")
                    else:
                        lines.append(f"    {keyword} value == '{bin_obj.bin_label}':")

                lines.append(f"        score += {bin_obj.points_rounded}")

            lines.append("")

        lines.append("    return score")

        python_code = "\n".join(lines)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(python_code)

        return python_code


# MODULE EXPORTS

__all__ = [
    # Enums
    "RiskRating",
    "DecisionType",
    # Dataclasses
    "ScorecardBin",
    "ScoreBand",
    "ReasonCode",
    # Classes
    "ScorecardTable",
    "ScorecardBuilder",
    "ScoreInterpreter",
    "ReasonCodeGenerator",
    "ScorecardExporter",
    # Constants
    "DEFAULT_BASE_SCORE",
    "DEFAULT_BASE_ODDS",
    "DEFAULT_PDO",
]
