"""
CHAID Segmentation for Vietnamese Credit Scoring.

CHAID (Chi-square Automatic Interaction Detection) is a decision tree
algorithm that uses chi-square tests for splitting and merging categories.

Key Features:
    - Automatic category merging based on chi-square significance
    - Multi-way splits (not just binary)
    - Bonferroni correction for multiple comparisons
    - Segment rule extraction and profiling

Example:
    >>> from modeling.segmentation import CHAIDSegmenter
    >>> segmenter = CHAIDSegmenter(max_depth=4, min_samples_leaf=100)
    >>> segmenter.fit(X, y)
    >>> segments = segmenter.predict(X)
    >>> rules = segmenter.get_segment_rules()
"""

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

# sklearn compatibility
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.utils.validation import check_is_fitted
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    class BaseEstimator:
        pass
    class ClassifierMixin:
        pass
    def check_is_fitted(estimator, attributes):
        for attr in attributes:
            if not hasattr(estimator, attr):
                raise ValueError(f"{estimator} is not fitted")


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Default significance level for chi-square tests
DEFAULT_ALPHA = 0.05

# Minimum expected frequency for chi-square test validity
MIN_EXPECTED_FREQ = 5

# Maximum number of categories before auto-binning
MAX_CATEGORIES = 20

# Vietnamese credit segment names
VIETNAMESE_SEGMENT_NAMES = {
    'established': 'Khách hàng có lịch sử tín dụng tốt',
    'young_professional': 'Chuyên gia trẻ',
    'thin_file': 'Khách hàng mới/ít thông tin',
    'self_employed': 'Tự kinh doanh',
    'high_risk': 'Rủi ro cao',
    'stable_income': 'Thu nhập ổn định',
    'new_credit': 'Mới sử dụng tín dụng',
}


# =============================================================================
# CHAID NODE DATACLASS
# =============================================================================

@dataclass
class CHAIDNode:
    """
    Node in the CHAID decision tree.

    Attributes:
        node_id: Unique identifier for the node
        depth: Depth level in the tree (root = 0)
        split_feature: Feature used to split this node
        split_values: Values/categories that lead to this node
        n_samples: Number of samples in this node
        n_bad: Number of bad (default) samples
        bad_rate: Default rate in this node
        chi_square: Chi-square statistic for the split
        p_value: P-value for the split
        children: List of child nodes
        is_leaf: Whether this is a terminal node
        parent_id: ID of parent node
        segment_name: Descriptive name for the segment
    """
    node_id: int
    depth: int
    split_feature: Optional[str] = None
    split_values: Optional[List[Any]] = None
    n_samples: int = 0
    n_bad: int = 0
    bad_rate: float = 0.0
    chi_square: float = 0.0
    p_value: float = 1.0
    children: List['CHAIDNode'] = field(default_factory=list)
    is_leaf: bool = True
    parent_id: Optional[int] = None
    segment_name: Optional[str] = None

    def __post_init__(self):
        """Calculate bad rate if not provided."""
        if self.n_samples > 0 and self.bad_rate == 0.0:
            self.bad_rate = self.n_bad / self.n_samples

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            'node_id': self.node_id,
            'depth': self.depth,
            'split_feature': self.split_feature,
            'split_values': self.split_values,
            'n_samples': self.n_samples,
            'n_bad': self.n_bad,
            'bad_rate': self.bad_rate,
            'chi_square': self.chi_square,
            'p_value': self.p_value,
            'is_leaf': self.is_leaf,
            'parent_id': self.parent_id,
            'n_children': len(self.children),
            'segment_name': self.segment_name,
        }


@dataclass
class SegmentProfile:
    """
    Profile of a segment for reporting.

    Attributes:
        segment_id: Unique segment identifier
        segment_name: Descriptive name
        segment_rules: List of rules defining the segment
        segment_size: Number of customers
        segment_pct: Percentage of total population
        segment_bad_rate: Default rate
        segment_risk_rank: Risk ranking (1=lowest risk)
        feature_summary: Summary statistics per feature
    """
    segment_id: int
    segment_name: str
    segment_rules: List[str]
    segment_size: int
    segment_pct: float
    segment_bad_rate: float
    segment_risk_rank: int
    feature_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'segment_id': self.segment_id,
            'segment_name': self.segment_name,
            'segment_rules': self.segment_rules,
            'segment_size': self.segment_size,
            'segment_pct': self.segment_pct,
            'segment_bad_rate': self.segment_bad_rate,
            'segment_risk_rank': self.segment_risk_rank,
        }


# =============================================================================
# CHAID SEGMENTER
# =============================================================================

class CHAIDSegmenter(BaseEstimator, ClassifierMixin):
    """
    CHAID (Chi-square Automatic Interaction Detection) Segmenter.

    Implements the CHAID algorithm for customer segmentation in credit scoring.
    Uses chi-square tests to determine optimal splits and automatically merges
    non-significant categories.

    Attributes:
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples in each leaf node
        alpha_split: Significance level for splitting
        alpha_merge: Significance level for merging categories
        max_categories: Maximum categories per feature before auto-binning
        use_bonferroni: Apply Bonferroni correction

    Fitted Attributes:
        tree_: Root node of the fitted tree
        n_segments_: Number of leaf segments
        segment_map_: Mapping of leaf node IDs to segment IDs
        feature_importances_: Feature importance based on chi-square

    Example:
        >>> segmenter = CHAIDSegmenter(max_depth=4, min_samples_leaf=100)
        >>> segmenter.fit(X, y)
        >>> segments = segmenter.predict(X)
        >>> print(segmenter.get_segment_rules())
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 100,
        min_samples_leaf: int = 50,
        alpha_split: float = 0.05,
        alpha_merge: float = 0.05,
        max_categories: int = 20,
        use_bonferroni: bool = True,
        n_bins_continuous: int = 10,
        random_state: Optional[int] = None,
    ):
        """
        Initialize CHAID Segmenter.

        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to attempt a split
            min_samples_leaf: Minimum samples per leaf
            alpha_split: P-value threshold for splitting
            alpha_merge: P-value threshold for category merging
            max_categories: Max categories before binning continuous
            use_bonferroni: Apply Bonferroni correction
            n_bins_continuous: Number of bins for continuous variables
            random_state: Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.alpha_split = alpha_split
        self.alpha_merge = alpha_merge
        self.max_categories = max_categories
        self.use_bonferroni = use_bonferroni
        self.n_bins_continuous = n_bins_continuous
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray]
    ) -> 'CHAIDSegmenter':
        """
        Fit the CHAID tree.

        Args:
            X: Feature DataFrame
            y: Binary target variable (0=good, 1=bad)

        Returns:
            self
        """
        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        y = np.asarray(y).ravel()
        if not np.all(np.isin(y[~np.isnan(y)], [0, 1])):
            raise ValueError("y must be binary (0/1)")

        # Store feature names and types
        self.feature_names_ = list(X.columns)
        self.feature_types_ = self._detect_feature_types(X)

        # Store bin edges for continuous variables during fit
        self.bin_edges_ = {}

        # Preprocess: bin continuous variables
        X_processed = self._preprocess_features(X, fit=True)

        # Initialize counters
        self._node_counter = 0
        self._chi_square_importance = {f: 0.0 for f in self.feature_names_}

        # Build tree recursively
        self.tree_ = self._build_tree(
            X_processed, y,
            depth=0,
            parent_id=None,
            split_values=None
        )

        # Create segment mapping from leaves
        self._create_segment_mapping()

        # Calculate feature importances
        self._calculate_feature_importance()

        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict segment for each sample.

        Args:
            X: Feature DataFrame

        Returns:
            Array of segment IDs
        """
        check_is_fitted(self, ['tree_', 'segment_map_', 'is_fitted_'])

        X_processed = self._preprocess_features(X)
        segments = np.zeros(len(X), dtype=int)

        for i in range(len(X)):
            row = X_processed.iloc[i]
            leaf = self._traverse_tree(row, self.tree_)
            segments[i] = self.segment_map_.get(leaf.node_id, 0)

        return segments

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of default for each segment.

        Args:
            X: Feature DataFrame

        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        check_is_fitted(self, ['tree_', 'is_fitted_'])

        X_processed = self._preprocess_features(X)
        proba = np.zeros((len(X), 2))

        for i in range(len(X)):
            row = X_processed.iloc[i]
            leaf = self._traverse_tree(row, self.tree_)
            bad_rate = leaf.bad_rate
            proba[i] = [1 - bad_rate, bad_rate]

        return proba

    def get_segment_rules(self) -> Dict[int, List[str]]:
        """
        Extract rules for each segment.

        Returns:
            Dictionary mapping segment ID to list of rules
        """
        check_is_fitted(self, ['tree_', 'segment_map_', 'is_fitted_'])

        rules = {}
        leaves = self._get_all_leaves(self.tree_)

        for leaf in leaves:
            segment_id = self.segment_map_.get(leaf.node_id)
            if segment_id is not None:
                rules[segment_id] = self._extract_path_rules(leaf)

        return rules

    def get_segment_profiles(self) -> List[SegmentProfile]:
        """
        Get detailed profiles for all segments.

        Returns:
            List of SegmentProfile objects
        """
        check_is_fitted(self, ['tree_', 'segment_map_', 'is_fitted_'])

        profiles = []
        leaves = self._get_all_leaves(self.tree_)
        total_samples = self.tree_.n_samples

        # Sort leaves by bad rate for ranking
        leaves_sorted = sorted(leaves, key=lambda x: x.bad_rate)

        for rank, leaf in enumerate(leaves_sorted, 1):
            segment_id = self.segment_map_.get(leaf.node_id)
            if segment_id is None:
                continue

            rules = self._extract_path_rules(leaf)

            profile = SegmentProfile(
                segment_id=segment_id,
                segment_name=leaf.segment_name or f"Segment_{segment_id}",
                segment_rules=rules,
                segment_size=leaf.n_samples,
                segment_pct=leaf.n_samples / total_samples,
                segment_bad_rate=leaf.bad_rate,
                segment_risk_rank=rank,
            )
            profiles.append(profile)

        return profiles

    # =========================================================================
    # TREE BUILDING METHODS
    # =========================================================================

    def _build_tree(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        depth: int,
        parent_id: Optional[int],
        split_values: Optional[List]
    ) -> CHAIDNode:
        """
        Recursively build the CHAID tree.

        Algorithm:
        1. Create node with current statistics
        2. Check stopping conditions
        3. For each feature, merge non-significant categories
        4. Select best split based on chi-square
        5. Create child nodes recursively
        """
        # Create node
        node_id = self._node_counter
        self._node_counter += 1

        n_samples = len(y)
        n_bad = y.sum()
        bad_rate = n_bad / n_samples if n_samples > 0 else 0

        node = CHAIDNode(
            node_id=node_id,
            depth=depth,
            split_values=split_values,
            n_samples=n_samples,
            n_bad=int(n_bad),
            bad_rate=bad_rate,
            parent_id=parent_id,
            is_leaf=True,
        )

        # Check stopping conditions
        if self._should_stop(X, y, depth):
            node.segment_name = self._generate_segment_name(node, split_values)
            return node

        # Find best split
        best_split = self._find_best_split(X, y)

        if best_split is None:
            node.segment_name = self._generate_segment_name(node, split_values)
            return node

        feature, merged_categories, chi_sq, p_value = best_split

        # Check if split is significant
        adjusted_alpha = self._get_adjusted_alpha(len(self.feature_names_))
        if p_value > adjusted_alpha:
            node.segment_name = self._generate_segment_name(node, split_values)
            return node

        # Update node with split info
        node.split_feature = feature
        node.chi_square = chi_sq
        node.p_value = p_value
        node.is_leaf = False

        # Update feature importance
        self._chi_square_importance[feature] += chi_sq

        # Create child nodes
        for category_group in merged_categories:
            # Get samples for this category group
            mask = X[feature].isin(category_group)
            X_child = X[mask]
            y_child = y[mask]

            if len(y_child) >= self.min_samples_leaf:
                child_node = self._build_tree(
                    X_child, y_child,
                    depth=depth + 1,
                    parent_id=node_id,
                    split_values=category_group
                )
                child_node.split_feature = feature
                node.children.append(child_node)

        # If no valid children, make this a leaf
        if len(node.children) == 0:
            node.is_leaf = True
            node.segment_name = self._generate_segment_name(node, split_values)

        return node

    def _should_stop(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        depth: int
    ) -> bool:
        """Check if we should stop splitting."""
        # Max depth reached
        if depth >= self.max_depth:
            return True

        # Not enough samples to split
        if len(y) < self.min_samples_split:
            return True

        # Pure node (all same class)
        if y.sum() == 0 or y.sum() == len(y):
            return True

        # Not enough features
        if len(X.columns) == 0:
            return True

        return False

    def _find_best_split(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Optional[Tuple[str, List[List], float, float]]:
        """
        Find the best feature and split point.

        Returns:
            Tuple of (feature_name, merged_categories, chi_square, p_value)
            or None if no valid split found
        """
        best_feature = None
        best_categories = None
        best_chi_sq = 0
        best_p_value = 1.0

        for feature in X.columns:
            # Merge categories for this feature
            merged = self._merge_categories(X[feature], y)

            if len(merged) < 2:
                continue

            # Calculate chi-square for this split
            chi_sq, p_value = self._calculate_split_chi_square(
                X[feature], y, merged
            )

            # Update best if better
            if chi_sq > best_chi_sq:
                best_feature = feature
                best_categories = merged
                best_chi_sq = chi_sq
                best_p_value = p_value

        if best_feature is None:
            return None

        return best_feature, best_categories, best_chi_sq, best_p_value

    # =========================================================================
    # CHI-SQUARE METHODS
    # =========================================================================

    def _merge_categories(
        self,
        x: pd.Series,
        y: np.ndarray
    ) -> List[List]:
        """
        Merge non-significant categories.

        Algorithm:
        1. Start with all unique categories
        2. For each pair of adjacent categories, calculate chi-square
        3. If p-value > alpha_merge, merge them
        4. Repeat until no more merges
        """
        # Get unique categories
        categories = list(x.dropna().unique())

        if len(categories) <= 1:
            return [categories]

        # Sort categories if ordinal (numeric-like)
        try:
            categories = sorted(categories, key=lambda c: float(str(c).split('_')[0]) if '_' in str(c) else float(c))
        except (ValueError, TypeError):
            categories = sorted(categories, key=str)

        # Initialize each category as its own group
        groups = [[c] for c in categories]

        # Iteratively merge non-significant pairs
        merged = True
        while merged and len(groups) > 1:
            merged = False
            min_chi_sq = float('inf')
            merge_idx = None

            # Find pair with lowest chi-square (highest p-value)
            for i in range(len(groups) - 1):
                # Create contingency table for adjacent groups
                group1 = groups[i]
                group2 = groups[i + 1]

                chi_sq, p_value = self._chi_square_test(
                    x, y, group1, group2
                )

                if p_value > self.alpha_merge and chi_sq < min_chi_sq:
                    min_chi_sq = chi_sq
                    merge_idx = i

            # Merge if found non-significant pair
            if merge_idx is not None:
                groups[merge_idx] = groups[merge_idx] + groups[merge_idx + 1]
                groups.pop(merge_idx + 1)
                merged = True

        return groups

    def _chi_square_test(
        self,
        x: pd.Series,
        y: np.ndarray,
        group1: List,
        group2: List
    ) -> Tuple[float, float]:
        """
        Perform chi-square test between two category groups.

        Args:
            x: Feature values
            y: Target values
            group1: First group of categories
            group2: Second group of categories

        Returns:
            Tuple of (chi_square, p_value)
        """
        # Create masks
        mask1 = x.isin(group1)
        mask2 = x.isin(group2)

        # Count good/bad for each group
        good1 = ((y == 0) & mask1).sum()
        bad1 = ((y == 1) & mask1).sum()
        good2 = ((y == 0) & mask2).sum()
        bad2 = ((y == 1) & mask2).sum()

        # Create contingency table
        contingency = np.array([[good1, bad1], [good2, bad2]])

        # Check if valid for chi-square test
        if contingency.sum() == 0 or (contingency == 0).all(axis=0).any():
            return 0.0, 1.0

        try:
            chi_sq, p_value, dof, expected = chi2_contingency(contingency)

            # Check minimum expected frequency
            if expected.min() < MIN_EXPECTED_FREQ:
                # Use Fisher's exact test for small samples
                try:
                    _, p_value = stats.fisher_exact(contingency)
                    chi_sq = -2 * np.log(p_value) if p_value > 0 else 0
                except Exception:
                    pass

            return chi_sq, p_value
        except Exception:
            return 0.0, 1.0

    def _calculate_split_chi_square(
        self,
        x: pd.Series,
        y: np.ndarray,
        merged_categories: List[List]
    ) -> Tuple[float, float]:
        """
        Calculate chi-square for a split with merged categories.

        Creates a contingency table with rows=categories, cols=good/bad
        """
        if len(merged_categories) < 2:
            return 0.0, 1.0

        # Build contingency table
        n_groups = len(merged_categories)
        contingency = np.zeros((n_groups, 2), dtype=int)

        for i, group in enumerate(merged_categories):
            mask = x.isin(group)
            contingency[i, 0] = ((y == 0) & mask).sum()  # Good
            contingency[i, 1] = ((y == 1) & mask).sum()  # Bad

        # Remove empty rows
        row_sums = contingency.sum(axis=1)
        contingency = contingency[row_sums > 0]

        if len(contingency) < 2:
            return 0.0, 1.0

        try:
            chi_sq, p_value, dof, expected = chi2_contingency(contingency)
            return chi_sq, p_value
        except Exception:
            return 0.0, 1.0

    def _get_adjusted_alpha(self, n_comparisons: int) -> float:
        """Get adjusted alpha with Bonferroni correction if enabled."""
        if self.use_bonferroni and n_comparisons > 1:
            return self.alpha_split / n_comparisons
        return self.alpha_split

    # =========================================================================
    # PREPROCESSING METHODS
    # =========================================================================

    def _detect_feature_types(self, X: pd.DataFrame) -> Dict[str, str]:
        """Detect whether features are continuous or categorical."""
        types = {}
        for col in X.columns:
            if X[col].dtype in ['object', 'category', 'bool']:
                types[col] = 'categorical'
            elif X[col].nunique() <= self.max_categories:
                types[col] = 'categorical'
            else:
                types[col] = 'continuous'
        return types

    def _preprocess_features(
        self,
        X: pd.DataFrame,
        fit: bool = False
    ) -> pd.DataFrame:
        """Preprocess features: bin continuous, handle missing."""
        X_processed = X.copy()

        for col in X.columns:
            if col not in self.feature_types_:
                # New column, detect type
                if X[col].dtype in ['object', 'category', 'bool']:
                    continue
                elif X[col].nunique() <= self.max_categories:
                    continue
                else:
                    X_processed[col] = self._bin_continuous(X[col], col, fit)
            elif self.feature_types_.get(col) == 'continuous':
                X_processed[col] = self._bin_continuous(X[col], col, fit)

            # Handle missing as category
            if X_processed[col].isna().any():
                X_processed[col] = X_processed[col].fillna('_MISSING_')

        return X_processed

    def _bin_continuous(
        self,
        x: pd.Series,
        col_name: str,
        fit: bool = False
    ) -> pd.Series:
        """Bin continuous variable into categories."""
        if fit:
            # During fit: create and store bin edges
            try:
                _, bins = pd.qcut(
                    x.dropna(), q=self.n_bins_continuous,
                    labels=False, retbins=True, duplicates='drop'
                )
                # Extend edges to handle values outside training range
                bins[0] = -np.inf
                bins[-1] = np.inf
                self.bin_edges_[col_name] = bins
            except Exception:
                # Fall back to equal width
                x_clean = x.dropna()
                bins = np.linspace(x_clean.min(), x_clean.max(), self.n_bins_continuous + 1)
                bins[0] = -np.inf
                bins[-1] = np.inf
                self.bin_edges_[col_name] = bins

        # Use stored bin edges
        bins = self.bin_edges_.get(col_name)
        if bins is None:
            # Fallback if no stored edges
            return pd.cut(x, bins=self.n_bins_continuous, include_lowest=True)

        # Create readable labels
        labels = []
        for i in range(len(bins) - 1):
            if i == 0:
                labels.append(f"<= {bins[i+1]:.2f}")
            elif i == len(bins) - 2:
                labels.append(f"> {bins[i]:.2f}")
            else:
                labels.append(f"{bins[i]:.2f} - {bins[i+1]:.2f}")

        return pd.cut(x, bins=bins, labels=labels, include_lowest=True)

    # =========================================================================
    # TREE TRAVERSAL METHODS
    # =========================================================================

    def _traverse_tree(self, row: pd.Series, node: CHAIDNode) -> CHAIDNode:
        """Traverse tree to find leaf node for a sample."""
        if node.is_leaf or len(node.children) == 0:
            return node

        feature = node.split_feature
        value = row[feature]

        # Find matching child
        for child in node.children:
            if child.split_values is not None and value in child.split_values:
                return self._traverse_tree(row, child)

        # If no match (shouldn't happen), return current node
        return node

    def _get_all_leaves(self, node: CHAIDNode) -> List[CHAIDNode]:
        """Get all leaf nodes in the tree."""
        leaves = []

        if node.is_leaf:
            leaves.append(node)
        else:
            for child in node.children:
                leaves.extend(self._get_all_leaves(child))

        return leaves

    def _extract_path_rules(self, leaf: CHAIDNode) -> List[str]:
        """Extract rules from root to leaf."""
        rules = []
        current = leaf

        # Traverse up to root collecting rules
        path = []
        while current is not None:
            if current.split_feature is not None and current.split_values is not None:
                path.append((current.split_feature, current.split_values))

            # Find parent
            current = self._find_node_by_id(current.parent_id)

        # Reverse to get root-to-leaf order
        path.reverse()

        # Format rules
        for feature, values in path:
            if len(values) == 1:
                rules.append(f"{feature} = {values[0]}")
            else:
                values_str = ", ".join(str(v) for v in values)
                rules.append(f"{feature} IN ({values_str})")

        return rules

    def _find_node_by_id(self, node_id: Optional[int]) -> Optional[CHAIDNode]:
        """Find node by ID in the tree."""
        if node_id is None:
            return None
        return self._find_node_recursive(self.tree_, node_id)

    def _find_node_recursive(
        self,
        node: CHAIDNode,
        target_id: int
    ) -> Optional[CHAIDNode]:
        """Recursively find node by ID."""
        if node.node_id == target_id:
            return node

        for child in node.children:
            result = self._find_node_recursive(child, target_id)
            if result is not None:
                return result

        return None

    # =========================================================================
    # SEGMENT MAPPING AND NAMING
    # =========================================================================

    def _create_segment_mapping(self):
        """Create mapping from leaf node IDs to segment IDs."""
        leaves = self._get_all_leaves(self.tree_)

        # Sort by bad rate for consistent ordering
        leaves_sorted = sorted(leaves, key=lambda x: x.bad_rate)

        self.segment_map_ = {}
        for i, leaf in enumerate(leaves_sorted):
            self.segment_map_[leaf.node_id] = i + 1

        self.n_segments_ = len(leaves)

    def _generate_segment_name(
        self,
        node: CHAIDNode,
        split_values: Optional[List]
    ) -> str:
        """Generate descriptive segment name."""
        if split_values is None:
            return f"Root_BadRate_{node.bad_rate:.1%}"

        # Create name based on path
        parts = []
        current = node
        while current is not None and current.split_feature is not None:
            feature = current.split_feature
            values = current.split_values

            # Simplify feature name
            feature_short = feature.replace('_', '').title()[:8]

            if values:
                if len(values) == 1:
                    val_short = str(values[0])[:10]
                else:
                    val_short = f"{len(values)}cats"
                parts.append(f"{feature_short}_{val_short}")

            current = self._find_node_by_id(current.parent_id)

        parts.reverse()
        name = "_".join(parts[-3:]) if parts else "Root"

        return f"{name}_BR{node.bad_rate:.0%}"

    def _calculate_feature_importance(self):
        """Calculate feature importances based on chi-square contributions."""
        total_chi_sq = sum(self._chi_square_importance.values())

        if total_chi_sq > 0:
            self.feature_importances_ = {
                f: chi_sq / total_chi_sq
                for f, chi_sq in self._chi_square_importance.items()
            }
        else:
            self.feature_importances_ = {
                f: 0.0 for f in self.feature_names_
            }

    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================

    def plot_tree(
        self,
        max_depth: Optional[int] = None,
        figsize: Tuple[int, int] = (20, 12),
        fontsize: int = 10
    ):
        """
        Visualize the CHAID tree.

        Args:
            max_depth: Maximum depth to display
            figsize: Figure size
            fontsize: Font size for labels

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        check_is_fitted(self, ['tree_', 'is_fitted_'])

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate tree dimensions
        leaves = self._get_all_leaves(self.tree_)
        n_leaves = len(leaves)
        actual_depth = max(leaf.depth for leaf in leaves)
        display_depth = min(max_depth or actual_depth, actual_depth)

        # Draw tree
        self._draw_node(
            ax, self.tree_,
            x=0.5, y=0.95,
            width=0.9,
            depth=0,
            max_depth=display_depth,
            fontsize=fontsize
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('CHAID Decision Tree', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def _draw_node(
        self,
        ax,
        node: CHAIDNode,
        x: float,
        y: float,
        width: float,
        depth: int,
        max_depth: int,
        fontsize: int
    ):
        """Recursively draw tree nodes."""
        import matplotlib.patches as patches

        # Node box dimensions
        box_width = 0.08
        box_height = 0.06

        # Color based on bad rate
        bad_rate = node.bad_rate
        color = plt.cm.RdYlGn(1 - bad_rate)

        # Draw node box
        rect = patches.FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.01",
            facecolor=color,
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)

        # Node text
        if node.is_leaf:
            text = f"n={node.n_samples}\nBR={node.bad_rate:.1%}"
        else:
            text = f"{node.split_feature}\nn={node.n_samples}\nBR={node.bad_rate:.1%}"

        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize-1, wrap=True)

        # Draw children
        if not node.is_leaf and depth < max_depth and len(node.children) > 0:
            n_children = len(node.children)
            child_width = width / n_children
            start_x = x - width/2 + child_width/2
            child_y = y - 0.12

            for i, child in enumerate(node.children):
                child_x = start_x + i * child_width

                # Draw edge
                ax.plot([x, child_x], [y - box_height/2, child_y + box_height/2],
                       'k-', linewidth=0.5)

                # Draw edge label
                if child.split_values:
                    label = str(child.split_values[0])[:15]
                    if len(child.split_values) > 1:
                        label += f"..."
                    mid_x = (x + child_x) / 2
                    mid_y = (y - box_height/2 + child_y + box_height/2) / 2
                    ax.text(mid_x, mid_y, label, fontsize=fontsize-2,
                           ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                # Recurse
                self._draw_node(
                    ax, child, child_x, child_y,
                    child_width * 0.9,
                    depth + 1, max_depth, fontsize
                )

    def plot_segment_distribution(
        self,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot distribution of samples across segments.

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        check_is_fitted(self, ['tree_', 'segment_map_', 'is_fitted_'])

        profiles = self.get_segment_profiles()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Segment sizes
        segment_ids = [p.segment_id for p in profiles]
        sizes = [p.segment_size for p in profiles]
        bad_rates = [p.segment_bad_rate for p in profiles]

        colors = plt.cm.RdYlGn([1 - br for br in bad_rates])

        axes[0].barh(segment_ids, sizes, color=colors)
        axes[0].set_xlabel('Number of Customers')
        axes[0].set_ylabel('Segment ID')
        axes[0].set_title('Segment Size Distribution')

        # Bad rates
        axes[1].barh(segment_ids, bad_rates, color=colors)
        axes[1].set_xlabel('Bad Rate')
        axes[1].set_ylabel('Segment ID')
        axes[1].set_title('Segment Bad Rates')
        axes[1].axvline(x=np.mean(bad_rates), color='red',
                       linestyle='--', label='Average')
        axes[1].legend()

        plt.tight_layout()
        return fig

    def plot_segment_risk(
        self,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot segment risk ranking.

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        check_is_fitted(self, ['tree_', 'segment_map_', 'is_fitted_'])

        profiles = self.get_segment_profiles()
        profiles_sorted = sorted(profiles, key=lambda x: x.segment_bad_rate)

        fig, ax = plt.subplots(figsize=figsize)

        names = [f"Seg {p.segment_id}\n({p.segment_size:,})" for p in profiles_sorted]
        bad_rates = [p.segment_bad_rate for p in profiles_sorted]
        colors = plt.cm.RdYlGn([1 - br for br in bad_rates])

        bars = ax.bar(names, bad_rates, color=colors, edgecolor='black')

        # Add value labels
        for bar, br in zip(bars, bad_rates):
            height = bar.get_height()
            ax.annotate(f'{br:.1%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Segment')
        ax.set_ylabel('Bad Rate')
        ax.set_title('Segment Risk Ranking (Low to High)')
        ax.axhline(y=np.mean(bad_rates), color='red',
                  linestyle='--', label=f'Avg: {np.mean(bad_rates):.1%}')
        ax.legend()

        plt.tight_layout()
        return fig

    def get_tree_summary(self) -> pd.DataFrame:
        """Get summary of all nodes in the tree."""
        check_is_fitted(self, ['tree_', 'is_fitted_'])

        nodes_data = []
        self._collect_nodes(self.tree_, nodes_data)

        return pd.DataFrame(nodes_data)

    def _collect_nodes(self, node: CHAIDNode, data: List[Dict]):
        """Recursively collect node data."""
        data.append(node.to_dict())
        for child in node.children:
            self._collect_nodes(child, data)


# =============================================================================
# VIETNAMESE CREDIT SEGMENTS
# =============================================================================

class VietnameseCreditSegmenter(CHAIDSegmenter):
    """
    Specialized CHAID segmenter for Vietnamese credit market.

    Includes predefined segment definitions and Vietnamese naming.
    """

    # Predefined segment rules for Vietnamese market
    SEGMENT_DEFINITIONS = {
        'established': {
            'rules': [
                ('credit_history_months', '>', 24),
                ('monthly_income', '>', 15_000_000),
            ],
            'name_vi': 'Khách hàng ổn định',
            'description': 'Có lịch sử tín dụng > 24 tháng, thu nhập > 15M',
        },
        'young_professional': {
            'rules': [
                ('age', '<', 30),
                ('employment_years', '>', 1),
            ],
            'name_vi': 'Chuyên gia trẻ',
            'description': 'Tuổi < 30, có việc làm ổn định > 1 năm',
        },
        'thin_file': {
            'rules': [
                ('credit_history_months', '<', 6),
            ],
            'name_vi': 'Hồ sơ mỏng',
            'description': 'Lịch sử tín dụng < 6 tháng',
        },
        'self_employed': {
            'rules': [
                ('employment_type', '==', 'self_employed'),
            ],
            'name_vi': 'Tự kinh doanh',
            'description': 'Khách hàng tự kinh doanh',
        },
        'high_risk': {
            'rules': [
                ('max_dpd', '>', 60),
            ],
            'name_vi': 'Rủi ro cao',
            'description': 'Có nợ quá hạn > 60 ngày',
        },
    }

    def assign_predefined_segments(
        self,
        X: pd.DataFrame
    ) -> pd.Series:
        """
        Assign samples to predefined Vietnamese credit segments.

        Args:
            X: Feature DataFrame

        Returns:
            Series with segment labels
        """
        segments = pd.Series('other', index=X.index)

        for segment_name, definition in self.SEGMENT_DEFINITIONS.items():
            mask = pd.Series(True, index=X.index)

            for feature, op, value in definition['rules']:
                if feature not in X.columns:
                    continue

                if op == '>':
                    mask &= X[feature] > value
                elif op == '<':
                    mask &= X[feature] < value
                elif op == '>=':
                    mask &= X[feature] >= value
                elif op == '<=':
                    mask &= X[feature] <= value
                elif op == '==':
                    mask &= X[feature] == value
                elif op == '!=':
                    mask &= X[feature] != value

            segments[mask] = segment_name

        return segments

    def get_segment_descriptions(self) -> Dict[str, str]:
        """Get Vietnamese descriptions for segments."""
        return {
            name: defn['description']
            for name, defn in self.SEGMENT_DEFINITIONS.items()
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Dataclasses
    "CHAIDNode",
    "SegmentProfile",
    # Classes
    "CHAIDSegmenter",
    "VietnameseCreditSegmenter",
    # Constants
    "DEFAULT_ALPHA",
    "VIETNAMESE_SEGMENT_NAMES",
]
