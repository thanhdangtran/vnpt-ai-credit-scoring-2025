from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# sklearn imports
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# statsmodels imports
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available, some features will be limited")


# CONSTANTS

# IV thresholds
IV_MIN_THRESHOLD = 0.02  # Below this, feature is not useful
IV_MAX_THRESHOLD = 0.50  # Above this, potential overfit

# Correlation threshold
CORRELATION_THRESHOLD = 0.70

# VIF threshold
VIF_THRESHOLD = 5.0

# Stepwise p-value thresholds
P_VALUE_ENTER = 0.05
P_VALUE_REMOVE = 0.10


# ENUMS

class SelectionMethod(Enum):
    NONE = "none"
    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"
    STEPWISE = "stepwise"  # Alias for bidirectional


class SelectionCriterion(Enum):
    AIC = "aic"
    BIC = "bic"
    PVALUE = "pvalue"
    LIKELIHOOD = "likelihood"


# MULTICOLLINEARITY CHECKER

class MulticollinearityChecker:
    def __init__(
        self,
        correlation_threshold: float = CORRELATION_THRESHOLD,
        vif_threshold: float = VIF_THRESHOLD,
    ):
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold

    def calculate_correlation_matrix(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        return X.corr()

    def calculate_vif(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        if not STATSMODELS_AVAILABLE:
            # Manual VIF calculation
            return self._calculate_vif_manual(X)

        # Add constant for VIF calculation
        X_with_const = sm.add_constant(X)

        vif_data = []
        for i, col in enumerate(X.columns):
            try:
                vif = variance_inflation_factor(X_with_const.values, i + 1)
                vif_data.append({'feature': col, 'vif': vif})
            except Exception:
                vif_data.append({'feature': col, 'vif': np.nan})

        return pd.DataFrame(vif_data).sort_values('vif', ascending=False)

    def _calculate_vif_manual(self, X: pd.DataFrame) -> pd.DataFrame:
        from sklearn.linear_model import LinearRegression

        vif_data = []
        for col in X.columns:
            other_cols = [c for c in X.columns if c != col]
            if len(other_cols) == 0:
                vif_data.append({'feature': col, 'vif': 1.0})
                continue

            X_others = X[other_cols].values
            y_col = X[col].values

            # Handle missing values
            mask = ~(np.isnan(X_others).any(axis=1) | np.isnan(y_col))
            if mask.sum() < 10:
                vif_data.append({'feature': col, 'vif': np.nan})
                continue

            lr = LinearRegression()
            lr.fit(X_others[mask], y_col[mask])
            r_squared = lr.score(X_others[mask], y_col[mask])

            vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
            vif_data.append({'feature': col, 'vif': vif})

        return pd.DataFrame(vif_data).sort_values('vif', ascending=False)

    def get_redundant_features(
        self,
        X: pd.DataFrame,
        iv_scores: Optional[Dict[str, float]] = None
    ) -> List[str]:
        corr_matrix = self.calculate_correlation_matrix(X)
        features_to_remove = set()

        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i >= j:
                    continue

                if abs(corr_matrix.loc[col1, col2]) > self.correlation_threshold:
                    # Decide which to remove
                    if iv_scores:
                        iv1 = iv_scores.get(col1, 0)
                        iv2 = iv_scores.get(col2, 0)
                        remove = col1 if iv1 < iv2 else col2
                    else:
                        remove = col2  # Remove second one

                    features_to_remove.add(remove)

        return list(features_to_remove)

    def iterative_vif_removal(
        self,
        X: pd.DataFrame,
        iv_scores: Optional[Dict[str, float]] = None,
        max_iterations: int = 100
    ) -> Tuple[List[str], pd.DataFrame]:
        features = list(X.columns)
        history = []

        for iteration in range(max_iterations):
            if len(features) <= 1:
                break

            X_subset = X[features]
            vif_df = self.calculate_vif(X_subset)

            # Record history
            history.append({
                'iteration': iteration,
                'n_features': len(features),
                'max_vif': vif_df['vif'].max(),
                'features': features.copy(),
            })

            # Find feature with highest VIF
            max_vif_row = vif_df.iloc[0]

            if max_vif_row['vif'] <= self.vif_threshold:
                break

            # Remove feature with highest VIF
            # If tie, remove one with lower IV
            high_vif_features = vif_df[vif_df['vif'] > self.vif_threshold]

            if iv_scores:
                # Sort by IV (ascending) to remove lowest IV first
                high_vif_features = high_vif_features.copy()
                high_vif_features['iv'] = high_vif_features['feature'].map(
                    lambda x: iv_scores.get(x, 0)
                )
                high_vif_features = high_vif_features.sort_values('iv')

            feature_to_remove = high_vif_features.iloc[0]['feature']
            features.remove(feature_to_remove)

        return features, pd.DataFrame(history)

    def get_multicollinearity_report(
        self,
        X: pd.DataFrame,
        iv_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        corr_matrix = self.calculate_correlation_matrix(X)
        vif_df = self.calculate_vif(X)
        redundant = self.get_redundant_features(X, iv_scores)

        # Find highly correlated pairs
        high_corr_pairs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i >= j:
                    continue
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > self.correlation_threshold:
                    high_corr_pairs.append({
                        'feature_1': col1,
                        'feature_2': col2,
                        'correlation': corr,
                    })

        return {
            'correlation_matrix': corr_matrix,
            'vif': vif_df,
            'high_correlation_pairs': pd.DataFrame(high_corr_pairs),
            'redundant_features': redundant,
            'high_vif_features': vif_df[vif_df['vif'] > self.vif_threshold]['feature'].tolist(),
        }


# STEPWISE SELECTOR

class StepwiseSelector:
    def __init__(
        self,
        method: str = "bidirectional",
        criterion: str = "aic",
        p_value_enter: float = P_VALUE_ENTER,
        p_value_remove: float = P_VALUE_REMOVE,
        max_features: Optional[int] = None,
        verbose: bool = False,
    ):
        self.method = method
        self.criterion = criterion
        self.p_value_enter = p_value_enter
        self.p_value_remove = p_value_remove
        self.max_features = max_features
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> List[str]:
        if self.method == "forward":
            return self.forward_selection(X, y)
        elif self.method == "backward":
            return self.backward_elimination(X, y)
        elif self.method in ["bidirectional", "stepwise"]:
            return self.bidirectional_selection(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def forward_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> List[str]:
        if not STATSMODELS_AVAILABLE:
            return self._forward_selection_sklearn(X, y)

        remaining = set(X.columns)
        selected = []
        current_score = np.inf

        while remaining:
            best_feature = None
            best_score = current_score

            for feature in remaining:
                # Try adding this feature
                features_to_try = selected + [feature]
                X_try = sm.add_constant(X[features_to_try])

                try:
                    model = sm.Logit(y, X_try).fit(disp=0)

                    if self.criterion == "aic":
                        score = model.aic
                    elif self.criterion == "bic":
                        score = model.bic
                    elif self.criterion == "pvalue":
                        # Use p-value of the new feature
                        score = model.pvalues[feature]
                    else:
                        score = model.aic

                    if score < best_score:
                        best_score = score
                        best_feature = feature

                except Exception:
                    continue

            # Check stopping criteria
            if best_feature is None:
                break

            if self.criterion == "pvalue" and best_score > self.p_value_enter:
                break

            if self.max_features and len(selected) >= self.max_features:
                break

            # Add best feature
            selected.append(best_feature)
            remaining.remove(best_feature)
            current_score = best_score

            if self.verbose:
                print(f"Added: {best_feature}, Score: {best_score:.4f}")

        self.selected_features_ = selected
        return selected

    def backward_elimination(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> List[str]:
        if not STATSMODELS_AVAILABLE:
            return self._backward_elimination_sklearn(X, y)

        selected = list(X.columns)

        while len(selected) > 1:
            X_current = sm.add_constant(X[selected])

            try:
                model = sm.Logit(y, X_current).fit(disp=0)
            except Exception:
                break

            # Find feature with highest p-value (worst)
            pvalues = model.pvalues.drop('const', errors='ignore')
            worst_feature = pvalues.idxmax()
            worst_pvalue = pvalues[worst_feature]

            # Check if should remove
            if self.criterion == "pvalue":
                if worst_pvalue <= self.p_value_remove:
                    break
            else:
                # Try removing and check if score improves
                features_without = [f for f in selected if f != worst_feature]
                X_try = sm.add_constant(X[features_without])

                try:
                    model_without = sm.Logit(y, X_try).fit(disp=0)

                    if self.criterion == "aic":
                        if model_without.aic >= model.aic:
                            break
                    elif self.criterion == "bic":
                        if model_without.bic >= model.bic:
                            break
                except Exception:
                    break

            # Remove worst feature
            selected.remove(worst_feature)

            if self.verbose:
                print(f"Removed: {worst_feature}, P-value: {worst_pvalue:.4f}")

        self.selected_features_ = selected
        return selected

    def bidirectional_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> List[str]:
        if not STATSMODELS_AVAILABLE:
            return self._bidirectional_sklearn(X, y)

        remaining = set(X.columns)
        selected = []

        changed = True
        while changed:
            changed = False

            # Forward step: try to add best feature
            best_feature = None
            best_score = np.inf

            for feature in remaining:
                features_to_try = selected + [feature]
                X_try = sm.add_constant(X[features_to_try])

                try:
                    model = sm.Logit(y, X_try).fit(disp=0)

                    if self.criterion == "pvalue":
                        score = model.pvalues[feature]
                        if score < self.p_value_enter and score < best_score:
                            best_score = score
                            best_feature = feature
                    else:
                        score = model.aic if self.criterion == "aic" else model.bic
                        if score < best_score:
                            best_score = score
                            best_feature = feature
                except Exception:
                    continue

            if best_feature:
                selected.append(best_feature)
                remaining.remove(best_feature)
                changed = True

                if self.verbose:
                    print(f"Added: {best_feature}")

            # Backward step: try to remove worst feature
            if len(selected) > 1:
                X_current = sm.add_constant(X[selected])

                try:
                    model = sm.Logit(y, X_current).fit(disp=0)
                    pvalues = model.pvalues.drop('const', errors='ignore')

                    for feature in list(selected):
                        if pvalues[feature] > self.p_value_remove:
                            selected.remove(feature)
                            remaining.add(feature)
                            changed = True

                            if self.verbose:
                                print(f"Removed: {feature}")
                except Exception:
                    pass

            # Check max features
            if self.max_features and len(selected) >= self.max_features:
                break

        self.selected_features_ = selected
        return selected

    def _forward_selection_sklearn(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> List[str]:
        from sklearn.feature_selection import SequentialFeatureSelector

        lr = LogisticRegression(max_iter=1000, solver='lbfgs')
        n_features = self.max_features or min(10, X.shape[1] - 1)
        n_features = max(1, min(n_features, X.shape[1] - 1))  # Ensure valid range

        try:
            sfs = SequentialFeatureSelector(
                lr, n_features_to_select=n_features,
                direction='forward', scoring='roc_auc'
            )
            sfs.fit(X, y)
            return list(X.columns[sfs.get_support()])
        except Exception:
            # Return all features if selection fails
            return list(X.columns)

    def _backward_elimination_sklearn(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> List[str]:
        from sklearn.feature_selection import SequentialFeatureSelector

        lr = LogisticRegression(max_iter=1000, solver='lbfgs')
        n_features = self.max_features or max(1, X.shape[1] - 3)
        n_features = max(1, min(n_features, X.shape[1] - 1))  # Ensure valid range

        try:
            sfs = SequentialFeatureSelector(
                lr, n_features_to_select=n_features,
                direction='backward', scoring='roc_auc'
            )
            sfs.fit(X, y)
            return list(X.columns[sfs.get_support()])
        except Exception:
            # Return all features if selection fails
            return list(X.columns)

    def _bidirectional_sklearn(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> List[str]:
        return self._forward_selection_sklearn(X, y)


# MODEL OUTPUT DATACLASS

@dataclass
class ModelCoefficient:
    feature: str
    coefficient: float
    std_error: float
    z_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    odds_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature': self.feature,
            'coefficient': self.coefficient,
            'std_error': self.std_error,
            'z_stat': self.z_stat,
            'p_value': self.p_value,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'odds_ratio': self.odds_ratio,
        }


@dataclass
class ModelSummary:
    n_observations: int
    n_features: int
    log_likelihood: float
    aic: float
    bic: float
    pseudo_r2: float
    coefficients: List[ModelCoefficient]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([c.to_dict() for c in self.coefficients])


# CREDIT LOGISTIC MODEL

class CreditLogisticModel(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        # IV thresholds
        iv_min_threshold: float = IV_MIN_THRESHOLD,
        iv_max_threshold: float = IV_MAX_THRESHOLD,
        # Multicollinearity
        correlation_threshold: float = CORRELATION_THRESHOLD,
        vif_threshold: float = VIF_THRESHOLD,
        check_vif: bool = True,
        check_correlation: bool = True,
        # Stepwise selection
        stepwise_method: Optional[str] = "bidirectional",
        stepwise_criterion: str = "aic",
        p_value_enter: float = P_VALUE_ENTER,
        p_value_remove: float = P_VALUE_REMOVE,
        max_features: Optional[int] = None,
        # Regularization
        regularization: Optional[str] = "l2",
        C: float = 1.0,
        # Constraints
        force_positive_coefficients: bool = False,
        # Other
        fit_intercept: bool = True,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.iv_min_threshold = iv_min_threshold
        self.iv_max_threshold = iv_max_threshold
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.check_vif = check_vif
        self.check_correlation = check_correlation
        self.stepwise_method = stepwise_method
        self.stepwise_criterion = stepwise_criterion
        self.p_value_enter = p_value_enter
        self.p_value_remove = p_value_remove
        self.max_features = max_features
        self.regularization = regularization
        self.C = C
        self.force_positive_coefficients = force_positive_coefficients
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        iv_scores: Optional[Dict[str, float]] = None,
    ) -> 'CreditLogisticModel':
        # Convert to numpy array if needed
        y_arr = np.asarray(y).ravel()

        # Store original features
        self.original_features_ = list(X.columns)
        self.iv_scores_ = iv_scores or {}

        # Step 1: IV Screening
        features = self._iv_screening(X, iv_scores)

        if self.verbose:
            print(f"After IV screening: {len(features)} features")

        # Step 2: Correlation check
        if self.check_correlation and len(features) > 1:
            features = self._correlation_screening(X[features], iv_scores)

            if self.verbose:
                print(f"After correlation check: {len(features)} features")

        # Step 3: VIF check
        if self.check_vif and len(features) > 1:
            features = self._vif_screening(X[features], iv_scores)

            if self.verbose:
                print(f"After VIF check: {len(features)} features")

        # Step 4: Stepwise selection
        if self.stepwise_method and len(features) > 1:
            selector = StepwiseSelector(
                method=self.stepwise_method,
                criterion=self.stepwise_criterion,
                p_value_enter=self.p_value_enter,
                p_value_remove=self.p_value_remove,
                max_features=self.max_features,
                verbose=self.verbose,
            )
            features = selector.fit(X[features], y_arr)

            if self.verbose:
                print(f"After stepwise selection: {len(features)} features")

        self.selected_features_ = features

        # Step 5: Final model fitting
        X_final = X[features]

        # Fit with statsmodels for detailed statistics
        if STATSMODELS_AVAILABLE:
            self._fit_statsmodels(X_final, y_arr)
        else:
            self._fit_sklearn(X_final, y_arr)

        self.is_fitted_ = True
        return self

    def _iv_screening(
        self,
        X: pd.DataFrame,
        iv_scores: Optional[Dict[str, float]]
    ) -> List[str]:
        if not iv_scores:
            return list(X.columns)

        selected = []
        flagged_overfit = []

        for feature in X.columns:
            iv = iv_scores.get(feature, 0)

            if iv < self.iv_min_threshold:
                if self.verbose:
                    print(f"Removed (low IV): {feature} (IV={iv:.4f})")
                continue

            if iv > self.iv_max_threshold:
                flagged_overfit.append(feature)
                if self.verbose:
                    print(f"Warning (high IV): {feature} (IV={iv:.4f})")

            selected.append(feature)

        self.flagged_overfit_features_ = flagged_overfit
        return selected

    def _correlation_screening(
        self,
        X: pd.DataFrame,
        iv_scores: Optional[Dict[str, float]]
    ) -> List[str]:
        checker = MulticollinearityChecker(
            correlation_threshold=self.correlation_threshold
        )
        redundant = checker.get_redundant_features(X, iv_scores)

        return [f for f in X.columns if f not in redundant]

    def _vif_screening(
        self,
        X: pd.DataFrame,
        iv_scores: Optional[Dict[str, float]]
    ) -> List[str]:
        checker = MulticollinearityChecker(vif_threshold=self.vif_threshold)
        selected, _ = checker.iterative_vif_removal(X, iv_scores)

        return selected

    def _fit_statsmodels(self, X: pd.DataFrame, y: np.ndarray):
        X_with_const = sm.add_constant(X)

        try:
            self.sm_model_ = sm.Logit(y, X_with_const).fit(disp=0)

            # Extract coefficients
            self.coefficients_ = {}
            self.p_values_ = {}
            self.std_errors_ = {}
            self.z_stats_ = {}
            self.confidence_intervals_ = {}

            for feature in X.columns:
                self.coefficients_[feature] = self.sm_model_.params[feature]
                self.p_values_[feature] = self.sm_model_.pvalues[feature]
                self.std_errors_[feature] = self.sm_model_.bse[feature]
                self.z_stats_[feature] = self.sm_model_.tvalues[feature]

                ci = self.sm_model_.conf_int().loc[feature]
                self.confidence_intervals_[feature] = (ci[0], ci[1])

            self.intercept_ = self.sm_model_.params.get('const', 0)

            # Model fit statistics
            self.log_likelihood_ = self.sm_model_.llf
            self.aic_ = self.sm_model_.aic
            self.bic_ = self.sm_model_.bic
            self.pseudo_r2_ = self.sm_model_.prsquared

        except Exception as e:
            warnings.warn(f"statsmodels fitting failed: {e}, falling back to sklearn")
            self._fit_sklearn(X, y)

    def _fit_sklearn(self, X: pd.DataFrame, y: np.ndarray):
        # Determine penalty
        if self.regularization:
            penalty = self.regularization
            solver = 'lbfgs' if penalty == 'l2' else 'saga'
        else:
            penalty = None
            solver = 'lbfgs'

        self.sklearn_model_ = LogisticRegression(
            penalty=penalty,
            C=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            solver=solver,
            random_state=self.random_state,
        )

        self.sklearn_model_.fit(X, y)

        # Extract coefficients
        self.coefficients_ = {
            feature: coef
            for feature, coef in zip(X.columns, self.sklearn_model_.coef_[0])
        }
        self.intercept_ = self.sklearn_model_.intercept_[0]

        # Estimate p-values using Wald test approximation
        self._estimate_pvalues_sklearn(X, y)

    def _estimate_pvalues_sklearn(self, X: pd.DataFrame, y: np.ndarray):
        n = len(y)
        k = len(self.selected_features_) + 1  # +1 for intercept

        # Get predictions
        proba = self.sklearn_model_.predict_proba(X)[:, 1]

        # Calculate Hessian diagonal approximation
        W = np.diag(proba * (1 - proba))

        X_with_const = np.column_stack([np.ones(n), X.values])

        try:
            # Fisher information matrix
            fisher = X_with_const.T @ W @ X_with_const
            cov_matrix = np.linalg.inv(fisher)

            # Standard errors
            std_errors = np.sqrt(np.diag(cov_matrix))

            self.std_errors_ = {}
            self.z_stats_ = {}
            self.p_values_ = {}
            self.confidence_intervals_ = {}

            # Intercept
            intercept_se = std_errors[0]
            intercept_z = self.intercept_ / intercept_se
            intercept_p = 2 * (1 - stats.norm.cdf(abs(intercept_z)))

            for i, feature in enumerate(self.selected_features_):
                coef = self.coefficients_[feature]
                se = std_errors[i + 1]
                z = coef / se if se > 0 else 0
                p = 2 * (1 - stats.norm.cdf(abs(z)))

                self.std_errors_[feature] = se
                self.z_stats_[feature] = z
                self.p_values_[feature] = p
                self.confidence_intervals_[feature] = (
                    coef - 1.96 * se,
                    coef + 1.96 * se
                )

        except Exception:
            # Fallback: set all to NaN
            for feature in self.selected_features_:
                self.std_errors_[feature] = np.nan
                self.z_stats_[feature] = np.nan
                self.p_values_[feature] = np.nan
                self.confidence_intervals_[feature] = (np.nan, np.nan)

        # Approximate fit statistics
        self.log_likelihood_ = -self.sklearn_model_.score(X, y) * len(y)
        self.aic_ = 2 * k - 2 * self.log_likelihood_
        self.bic_ = k * np.log(n) - 2 * self.log_likelihood_
        self.pseudo_r2_ = np.nan  # Cannot compute without null model

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ['coefficients_', 'intercept_', 'is_fitted_'])

        X_selected = X[self.selected_features_]

        if hasattr(self, 'sm_model_'):
            X_with_const = sm.add_constant(X_selected)
            proba_1 = self.sm_model_.predict(X_with_const)
            proba_0 = 1 - proba_1
            return np.column_stack([proba_0, proba_1])
        else:
            return self.sklearn_model_.predict_proba(X_selected)

    def calculate_predicted_probability(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    def get_odds_ratios(self) -> Dict[str, float]:
        check_is_fitted(self, ['coefficients_'])

        return {
            feature: np.exp(coef)
            for feature, coef in self.coefficients_.items()
        }

    def summary(self) -> str:
        check_is_fitted(self, ['coefficients_', 'is_fitted_'])

        lines = []
        lines.append("=" * 80)
        lines.append("Credit Logistic Model Summary")
        lines.append("=" * 80)

        # Model fit statistics
        lines.append(f"\nObservations: {getattr(self, 'n_observations_', 'N/A')}")
        lines.append(f"Features: {len(self.selected_features_)}")
        lines.append(f"Log-Likelihood: {self.log_likelihood_:.4f}")
        lines.append(f"AIC: {self.aic_:.4f}")
        lines.append(f"BIC: {self.bic_:.4f}")

        if hasattr(self, 'pseudo_r2_') and not np.isnan(self.pseudo_r2_):
            lines.append(f"Pseudo R-squared: {self.pseudo_r2_:.4f}")

        # Coefficient table
        lines.append("\n" + "-" * 80)
        lines.append(f"{'Feature':<25} {'Coef':>10} {'Std Err':>10} {'z':>8} {'P>|z|':>8} {'[95% CI]':>20}")
        lines.append("-" * 80)

        # Intercept
        lines.append(f"{'const':<25} {self.intercept_:>10.4f} {'':>10} {'':>8} {'':>8} {'':>20}")

        # Features
        for feature in self.selected_features_:
            coef = self.coefficients_.get(feature, np.nan)
            se = self.std_errors_.get(feature, np.nan)
            z = self.z_stats_.get(feature, np.nan)
            p = self.p_values_.get(feature, np.nan)
            ci = self.confidence_intervals_.get(feature, (np.nan, np.nan))

            p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if not np.isnan(ci[0]) else "N/A"

            lines.append(
                f"{feature:<25} {coef:>10.4f} {se:>10.4f} {z:>8.2f} {p_str:>8} {ci_str:>20}"
            )

        lines.append("=" * 80)

        return "\n".join(lines)

    def get_coefficient_table(self) -> pd.DataFrame:
        check_is_fitted(self, ['coefficients_'])

        data = []
        for feature in self.selected_features_:
            ci = self.confidence_intervals_.get(feature, (np.nan, np.nan))

            data.append({
                'feature': feature,
                'coefficient': self.coefficients_.get(feature, np.nan),
                'std_error': self.std_errors_.get(feature, np.nan),
                'z_stat': self.z_stats_.get(feature, np.nan),
                'p_value': self.p_values_.get(feature, np.nan),
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'odds_ratio': np.exp(self.coefficients_.get(feature, 0)),
            })

        return pd.DataFrame(data)

    def plot_coefficients(
        self,
        figsize: Tuple[int, int] = (10, 8)
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        check_is_fitted(self, ['coefficients_', 'confidence_intervals_'])

        fig, ax = plt.subplots(figsize=figsize)

        features = list(self.selected_features_)
        coefficients = [self.coefficients_[f] for f in features]
        errors_lower = [
            self.coefficients_[f] - self.confidence_intervals_[f][0]
            for f in features
        ]
        errors_upper = [
            self.confidence_intervals_[f][1] - self.coefficients_[f]
            for f in features
        ]

        # Sort by coefficient value
        sorted_idx = np.argsort(coefficients)
        features = [features[i] for i in sorted_idx]
        coefficients = [coefficients[i] for i in sorted_idx]
        errors = [[errors_lower[i] for i in sorted_idx],
                  [errors_upper[i] for i in sorted_idx]]

        y_pos = range(len(features))

        ax.barh(y_pos, coefficients, xerr=errors, capsize=3,
                color=['steelblue' if c >= 0 else 'indianred' for c in coefficients])
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Coefficient')
        ax.set_title('Logistic Regression Coefficients with 95% CI')

        plt.tight_layout()
        return fig

    def get_model_summary(self) -> ModelSummary:
        check_is_fitted(self, ['coefficients_'])

        coefficients = []
        for feature in self.selected_features_:
            ci = self.confidence_intervals_.get(feature, (np.nan, np.nan))

            coef_obj = ModelCoefficient(
                feature=feature,
                coefficient=self.coefficients_.get(feature, np.nan),
                std_error=self.std_errors_.get(feature, np.nan),
                z_stat=self.z_stats_.get(feature, np.nan),
                p_value=self.p_values_.get(feature, np.nan),
                ci_lower=ci[0],
                ci_upper=ci[1],
                odds_ratio=np.exp(self.coefficients_.get(feature, 0)),
            )
            coefficients.append(coef_obj)

        return ModelSummary(
            n_observations=getattr(self, 'n_observations_', 0),
            n_features=len(self.selected_features_),
            log_likelihood=self.log_likelihood_,
            aic=self.aic_,
            bic=self.bic_,
            pseudo_r2=getattr(self, 'pseudo_r2_', np.nan),
            coefficients=coefficients,
        )


# MODULE EXPORTS

__all__ = [
    # Enums
    "SelectionMethod",
    "SelectionCriterion",
    # Dataclasses
    "ModelCoefficient",
    "ModelSummary",
    # Classes
    "MulticollinearityChecker",
    "StepwiseSelector",
    "CreditLogisticModel",
    # Constants
    "IV_MIN_THRESHOLD",
    "IV_MAX_THRESHOLD",
    "CORRELATION_THRESHOLD",
    "VIF_THRESHOLD",
    "P_VALUE_ENTER",
    "P_VALUE_REMOVE",
]
