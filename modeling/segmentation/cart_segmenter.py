from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# sklearn imports
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.utils.validation import check_is_fitted
from sklearn.inspection import permutation_importance as sklearn_permutation_importance


# CONSTANTS AND CONFIGURATION

# Default parameters
DEFAULT_MAX_DEPTH = 5
DEFAULT_MIN_SAMPLES_LEAF = 50
DEFAULT_MIN_SAMPLES_SPLIT = 100

# Bad rate similarity threshold for merging
BAD_RATE_MERGE_THRESHOLD = 0.01

# Minimum segment size (percentage)
MIN_SEGMENT_PCT = 0.05


# ENUMS

class SplitCriterion(Enum):
    GINI = "gini"
    ENTROPY = "entropy"
    LOG_LOSS = "log_loss"


class PruningStrategy(Enum):
    NONE = "none"
    PRE_PRUNING = "pre_pruning"
    POST_PRUNING = "post_pruning"
    RISK_BASED = "risk_based"


# CART NODE DATACLASS

@dataclass
class CARTNode:
    node_id: int
    depth: int
    feature: Optional[str] = None
    threshold: Optional[float] = None
    operator: str = "<="
    n_samples: int = 0
    n_bad: int = 0
    bad_rate: float = 0.0
    gini: float = 0.0
    left_child: Optional['CARTNode'] = None
    right_child: Optional['CARTNode'] = None
    is_leaf: bool = True
    segment_id: Optional[int] = None
    feature_idx: Optional[int] = None

    def __post_init__(self):
        if self.n_samples > 0 and self.bad_rate == 0.0 and self.n_bad > 0:
            self.bad_rate = self.n_bad / self.n_samples

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'depth': self.depth,
            'feature': self.feature,
            'threshold': self.threshold,
            'operator': self.operator,
            'n_samples': self.n_samples,
            'n_bad': self.n_bad,
            'bad_rate': self.bad_rate,
            'gini': self.gini,
            'is_leaf': self.is_leaf,
            'segment_id': self.segment_id,
            'has_left': self.left_child is not None,
            'has_right': self.right_child is not None,
        }

    def get_rule(self) -> str:
        if self.feature is None or self.threshold is None:
            return "ROOT"
        return f"{self.feature} {self.operator} {self.threshold:.4g}"


@dataclass
class SegmentStats:
    segment_id: int
    node_id: int
    n_samples: int
    n_bad: int
    bad_rate: float
    pct_of_total: float
    rules: List[str]
    depth: int
    gini: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'segment_id': self.segment_id,
            'node_id': self.node_id,
            'n_samples': self.n_samples,
            'n_bad': self.n_bad,
            'bad_rate': self.bad_rate,
            'pct_of_total': self.pct_of_total,
            'rules': self.rules,
            'depth': self.depth,
            'gini': self.gini,
        }


# CART SEGMENTER

class CARTSegmenter(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_depth: int = DEFAULT_MAX_DEPTH,
        min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT,
        min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
        criterion: str = "gini",
        ccp_alpha: float = 0.0,
        class_weight: Optional[Union[str, Dict]] = "balanced",
        max_features: Optional[Union[int, float, str]] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = None,
        # Custom parameters
        merge_similar_segments: bool = False,
        bad_rate_merge_threshold: float = BAD_RATE_MERGE_THRESHOLD,
        min_segment_pct: float = MIN_SEGMENT_PCT,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.ccp_alpha = ccp_alpha
        self.class_weight = class_weight
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.merge_similar_segments = merge_similar_segments
        self.bad_rate_merge_threshold = bad_rate_merge_threshold
        self.min_segment_pct = min_segment_pct

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray]
    ) -> 'CARTSegmenter':
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_arr = X

        y_arr = np.asarray(y).ravel()

        # Validate target
        if not np.all(np.isin(y_arr[~np.isnan(y_arr)], [0, 1])):
            raise ValueError("y must be binary (0/1)")

        # Store total statistics
        self.n_samples_ = len(y_arr)
        self.n_bad_ = int(y_arr.sum())
        self.overall_bad_rate_ = self.n_bad_ / self.n_samples_

        # Create and fit sklearn tree
        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            ccp_alpha=self.ccp_alpha,
            class_weight=self.class_weight,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state,
        )

        self.tree_.fit(X_arr, y_arr)

        # Build custom tree structure
        self.cart_tree_ = self._build_cart_tree(X_arr, y_arr)

        # Count segments and assign IDs
        self._assign_segment_ids()

        # Calculate segment statistics
        self._calculate_segment_stats(X_arr, y_arr)

        # Merge similar segments if requested
        if self.merge_similar_segments:
            self._merge_similar_segments()

        # Store feature importances
        self.feature_importances_dict_ = {
            name: imp for name, imp in
            zip(self.feature_names_, self.tree_.feature_importances_)
        }

        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ['tree_', 'cart_tree_', 'is_fitted_'])

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        # Get leaf indices from sklearn tree
        leaf_indices = self.tree_.apply(X_arr)

        # Map to segment IDs
        segments = np.array([
            self.leaf_to_segment_.get(leaf_idx, 0)
            for leaf_idx in leaf_indices
        ])

        return segments

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ['tree_', 'is_fitted_'])

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        return self.tree_.predict_proba(X_arr)

    def get_decision_path(self, X: pd.DataFrame) -> Dict[int, List[str]]:
        check_is_fitted(self, ['tree_', 'cart_tree_', 'is_fitted_'])

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        # Get decision path from sklearn
        decision_path = self.tree_.decision_path(X_arr)

        paths = {}
        for i in range(X_arr.shape[0]):
            # Get nodes in path
            node_indices = decision_path[i].indices
            rules = self._extract_path_rules(X_arr[i], node_indices)
            paths[i] = rules

        return paths

    # TREE BUILDING METHODS

    def _build_cart_tree(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> CARTNode:
        sklearn_tree = self.tree_.tree_

        def build_node(node_idx: int, depth: int) -> CARTNode:
            # Get node statistics
            n_samples = int(sklearn_tree.n_node_samples[node_idx])

            # Get class distribution
            values = sklearn_tree.value[node_idx].flatten()
            n_good = int(values[0]) if len(values) > 0 else 0
            n_bad = int(values[1]) if len(values) > 1 else 0
            bad_rate = n_bad / n_samples if n_samples > 0 else 0

            # Gini impurity
            gini = sklearn_tree.impurity[node_idx]

            # Check if leaf
            is_leaf = sklearn_tree.children_left[node_idx] == -1

            # Get split info
            feature_idx = sklearn_tree.feature[node_idx]
            threshold = sklearn_tree.threshold[node_idx]

            feature_name = None
            if feature_idx >= 0:
                feature_name = self.feature_names_[feature_idx]

            node = CARTNode(
                node_id=node_idx,
                depth=depth,
                feature=feature_name,
                threshold=threshold if not is_leaf else None,
                n_samples=n_samples,
                n_bad=n_bad,
                bad_rate=bad_rate,
                gini=gini,
                is_leaf=is_leaf,
                feature_idx=feature_idx if feature_idx >= 0 else None,
            )

            # Build children
            if not is_leaf:
                left_idx = sklearn_tree.children_left[node_idx]
                right_idx = sklearn_tree.children_right[node_idx]

                if left_idx != -1:
                    node.left_child = build_node(left_idx, depth + 1)
                    node.left_child.operator = "<="

                if right_idx != -1:
                    node.right_child = build_node(right_idx, depth + 1)
                    node.right_child.operator = ">"

            return node

        return build_node(0, 0)

    def _assign_segment_ids(self):
        leaves = self.get_leaf_nodes()

        # Sort by bad rate for consistent ordering
        leaves_sorted = sorted(leaves, key=lambda x: x.bad_rate)

        self.leaf_to_segment_ = {}
        for i, leaf in enumerate(leaves_sorted, 1):
            leaf.segment_id = i
            self.leaf_to_segment_[leaf.node_id] = i

        self.n_segments_ = len(leaves)

    def _calculate_segment_stats(self, X: np.ndarray, y: np.ndarray):
        leaves = self.get_leaf_nodes()

        self.segment_stats_ = {}
        for leaf in leaves:
            if leaf.segment_id is None:
                continue

            # Get rules to reach this leaf
            rules = self._get_node_rules(leaf)

            stats = SegmentStats(
                segment_id=leaf.segment_id,
                node_id=leaf.node_id,
                n_samples=leaf.n_samples,
                n_bad=leaf.n_bad,
                bad_rate=leaf.bad_rate,
                pct_of_total=leaf.n_samples / self.n_samples_,
                rules=rules,
                depth=leaf.depth,
                gini=leaf.gini,
            )
            self.segment_stats_[leaf.segment_id] = stats

    def _get_node_rules(self, target_node: CARTNode) -> List[str]:
        rules = []

        def find_path(node: CARTNode, path: List[Tuple[str, str, float]]) -> bool:
            if node.node_id == target_node.node_id:
                return True

            if node.is_leaf:
                return False

            # Try left child
            if node.left_child:
                new_path = path + [(node.feature, "<=", node.threshold)]
                if find_path(node.left_child, new_path):
                    rules.extend([f"{f} {op} {t:.4g}" for f, op, t in new_path])
                    return True

            # Try right child
            if node.right_child:
                new_path = path + [(node.feature, ">", node.threshold)]
                if find_path(node.right_child, new_path):
                    rules.extend([f"{f} {op} {t:.4g}" for f, op, t in new_path])
                    return True

            return False

        find_path(self.cart_tree_, [])
        return rules

    def _extract_path_rules(
        self,
        sample: np.ndarray,
        node_indices: np.ndarray
    ) -> List[str]:
        rules = []
        sklearn_tree = self.tree_.tree_

        for i, node_idx in enumerate(node_indices[:-1]):
            feature_idx = sklearn_tree.feature[node_idx]
            threshold = sklearn_tree.threshold[node_idx]

            if feature_idx >= 0:
                feature_name = self.feature_names_[feature_idx]
                value = sample[feature_idx]

                if value <= threshold:
                    rules.append(f"{feature_name} <= {threshold:.4g}")
                else:
                    rules.append(f"{feature_name} > {threshold:.4g}")

        return rules

    # LEAF NODE METHODS

    def get_leaf_nodes(self) -> List[CARTNode]:
        check_is_fitted(self, ['cart_tree_'])

        leaves = []

        def collect_leaves(node: CARTNode):
            if node.is_leaf:
                leaves.append(node)
            else:
                if node.left_child:
                    collect_leaves(node.left_child)
                if node.right_child:
                    collect_leaves(node.right_child)

        collect_leaves(self.cart_tree_)
        return leaves

    def calculate_node_statistics(self) -> pd.DataFrame:
        check_is_fitted(self, ['cart_tree_'])

        nodes_data = []

        def collect_nodes(node: CARTNode):
            nodes_data.append(node.to_dict())
            if node.left_child:
                collect_nodes(node.left_child)
            if node.right_child:
                collect_nodes(node.right_child)

        collect_nodes(self.cart_tree_)
        return pd.DataFrame(nodes_data)

    # SEGMENT REFINEMENT

    def _merge_similar_segments(self):
        # Sort segments by bad rate
        segments = sorted(
            self.segment_stats_.values(),
            key=lambda x: x.bad_rate
        )

        merged_segments = []
        current_group = [segments[0]]

        for seg in segments[1:]:
            # Check if should merge with current group
            avg_rate = np.mean([s.bad_rate for s in current_group])
            if abs(seg.bad_rate - avg_rate) <= self.bad_rate_merge_threshold:
                current_group.append(seg)
            else:
                merged_segments.append(current_group)
                current_group = [seg]

        merged_segments.append(current_group)

        # Update segment IDs
        self.merged_segment_map_ = {}
        new_segment_stats = {}

        for new_id, group in enumerate(merged_segments, 1):
            for seg in group:
                self.merged_segment_map_[seg.segment_id] = new_id

            # Create combined stats
            total_samples = sum(s.n_samples for s in group)
            total_bad = sum(s.n_bad for s in group)

            combined = SegmentStats(
                segment_id=new_id,
                node_id=group[0].node_id,  # Use first node's ID
                n_samples=total_samples,
                n_bad=total_bad,
                bad_rate=total_bad / total_samples if total_samples > 0 else 0,
                pct_of_total=total_samples / self.n_samples_,
                rules=group[0].rules,  # Use first segment's rules
                depth=min(s.depth for s in group),
                gini=np.mean([s.gini for s in group]),
            )
            new_segment_stats[new_id] = combined

        self.segment_stats_ = new_segment_stats
        self.n_segments_ = len(new_segment_stats)

    def merge_similar_segments_manual(
        self,
        threshold: float = 0.01
    ) -> 'CARTSegmenter':
        self.bad_rate_merge_threshold = threshold
        self._merge_similar_segments()
        return self

    def split_large_segments(
        self,
        max_size: float = 0.3,
        X: pd.DataFrame = None,
        y: np.ndarray = None
    ) -> 'CARTSegmenter':
        warnings.warn(
            "split_large_segments requires re-fitting with different parameters. "
            "Consider using max_leaf_nodes or reducing max_depth instead."
        )
        return self

    def ensure_minimum_size(
        self,
        min_pct: float = 0.05
    ) -> 'CARTSegmenter':
        # Get segments below threshold
        small_segments = [
            seg for seg in self.segment_stats_.values()
            if seg.pct_of_total < min_pct
        ]

        if not small_segments:
            return self

        # Sort all segments by bad rate
        all_segments = sorted(
            self.segment_stats_.values(),
            key=lambda x: x.bad_rate
        )

        # Merge small segments with nearest neighbor
        for small_seg in small_segments:
            # Find nearest neighbor by bad rate
            nearest = None
            min_diff = float('inf')

            for seg in all_segments:
                if seg.segment_id != small_seg.segment_id:
                    diff = abs(seg.bad_rate - small_seg.bad_rate)
                    if diff < min_diff:
                        min_diff = diff
                        nearest = seg

            if nearest:
                # Update merged segment map
                if not hasattr(self, 'merged_segment_map_'):
                    self.merged_segment_map_ = {}
                self.merged_segment_map_[small_seg.segment_id] = nearest.segment_id

        return self

    # FEATURE IMPORTANCE

    def get_feature_importance(
        self,
        method: str = "gini"
    ) -> Dict[str, float]:
        check_is_fitted(self, ['tree_', 'feature_importances_dict_'])

        if method == "gini":
            return self.feature_importances_dict_

        elif method == "split_count":
            # Count how many times each feature is used
            counts = {f: 0 for f in self.feature_names_}

            def count_splits(node: CARTNode):
                if node.feature:
                    counts[node.feature] += 1
                if node.left_child:
                    count_splits(node.left_child)
                if node.right_child:
                    count_splits(node.right_child)

            count_splits(self.cart_tree_)

            total = sum(counts.values())
            return {f: c / total if total > 0 else 0 for f, c in counts.items()}

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_permutation_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_repeats: int = 10,
        scoring: str = "roc_auc"
    ) -> Dict[str, float]:
        check_is_fitted(self, ['tree_'])

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        result = sklearn_permutation_importance(
            self.tree_, X_arr, y,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=self.random_state
        )

        return {
            name: imp for name, imp in
            zip(self.feature_names_, result.importances_mean)
        }

    # COMPARISON WITH CHAID

    def compare_with_chaid(
        self,
        chaid_model: Any,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> pd.DataFrame:
        check_is_fitted(self, ['tree_'])

        cart_segments = self.predict(X)
        chaid_segments = chaid_model.predict(X)

        comparison = []
        for cart_seg in np.unique(cart_segments):
            cart_mask = cart_segments == cart_seg
            cart_rate = y[cart_mask].mean()
            cart_size = cart_mask.sum()

            # Find overlapping CHAID segments
            overlapping_chaid = chaid_segments[cart_mask]
            chaid_dist = pd.Series(overlapping_chaid).value_counts(normalize=True)

            comparison.append({
                'cart_segment': cart_seg,
                'cart_size': cart_size,
                'cart_bad_rate': cart_rate,
                'main_chaid_segment': chaid_dist.index[0] if len(chaid_dist) > 0 else None,
                'chaid_overlap_pct': chaid_dist.iloc[0] if len(chaid_dist) > 0 else 0,
                'n_chaid_segments': len(chaid_dist),
            })

        return pd.DataFrame(comparison)

    # VISUALIZATION

    def plot_tree(
        self,
        max_depth: Optional[int] = 3,
        figsize: Tuple[int, int] = (20, 10),
        fontsize: int = 10,
        filled: bool = True,
        rounded: bool = True,
    ):
        """
        Plot the decision tree.

        Args:
            max_depth: Maximum depth to display
            figsize: Figure size
            fontsize: Font size
            filled: Fill nodes with colors
            rounded: Use rounded boxes

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        check_is_fitted(self, ['tree_'])

        fig, ax = plt.subplots(figsize=figsize)

        plot_tree(
            self.tree_,
            feature_names=self.feature_names_,
            class_names=['Good', 'Bad'],
            filled=filled,
            rounded=rounded,
            fontsize=fontsize,
            max_depth=max_depth,
            ax=ax
        )

        ax.set_title('CART Decision Tree', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_feature_importance(
        self,
        top_n: int = 10,
        figsize: Tuple[int, int] = (10, 6)
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        check_is_fitted(self, ['feature_importances_dict_'])

        # Sort by importance
        sorted_imp = sorted(
            self.feature_importances_dict_.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        features = [x[0] for x in sorted_imp]
        importances = [x[1] for x in sorted_imp]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(features[::-1], importances[::-1], color='steelblue')

        ax.set_xlabel('Importance (Gini)')
        ax.set_title('Feature Importance')
        plt.tight_layout()
        return fig

    def plot_segment_distribution(
        self,
        figsize: Tuple[int, int] = (12, 6)
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        check_is_fitted(self, ['segment_stats_'])

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        segments = sorted(self.segment_stats_.values(), key=lambda x: x.segment_id)
        seg_ids = [s.segment_id for s in segments]
        sizes = [s.n_samples for s in segments]
        bad_rates = [s.bad_rate for s in segments]

        colors = plt.cm.RdYlGn([1 - br for br in bad_rates])

        # Size distribution
        axes[0].barh(seg_ids, sizes, color=colors)
        axes[0].set_xlabel('Number of Samples')
        axes[0].set_ylabel('Segment ID')
        axes[0].set_title('Segment Size Distribution')

        # Bad rate distribution
        axes[1].barh(seg_ids, bad_rates, color=colors)
        axes[1].set_xlabel('Bad Rate')
        axes[1].set_ylabel('Segment ID')
        axes[1].set_title('Segment Bad Rates')
        axes[1].axvline(
            x=self.overall_bad_rate_, color='red',
            linestyle='--', label=f'Overall: {self.overall_bad_rate_:.2%}'
        )
        axes[1].legend()

        plt.tight_layout()
        return fig

    # EXPORT METHODS

    def to_sql_rules(
        self,
        table_alias: str = "t",
        segment_column: str = "segment_id"
    ) -> str:
        check_is_fitted(self, ['segment_stats_'])

        sql_lines = [f"CASE"]

        for seg_id, stats in sorted(self.segment_stats_.items()):
            conditions = []
            for rule in stats.rules:
                # Parse rule
                parts = rule.split()
                if len(parts) >= 3:
                    feature = parts[0]
                    op = parts[1]
                    value = parts[2]
                    conditions.append(f"{table_alias}.{feature} {op} {value}")

            if conditions:
                condition_str = " AND ".join(conditions)
                sql_lines.append(f"    WHEN {condition_str} THEN {seg_id}")

        sql_lines.append(f"    ELSE 0")
        sql_lines.append(f"END AS {segment_column}")

        return "\n".join(sql_lines)

    def to_python_function(
        self,
        function_name: str = "get_segment"
    ) -> str:
        check_is_fitted(self, ['segment_stats_'])

        lines = [
            f"def {function_name}(row):",
            '    """Auto-generated segmentation function."""',
        ]

        first = True
        for seg_id, stats in sorted(
            self.segment_stats_.items(),
            key=lambda x: x[1].bad_rate
        ):
            conditions = []
            for rule in stats.rules:
                parts = rule.split()
                if len(parts) >= 3:
                    feature = parts[0]
                    op = parts[1]
                    value = parts[2]
                    conditions.append(f"row['{feature}'] {op} {value}")

            if conditions:
                condition_str = " and ".join(conditions)
                keyword = "if" if first else "elif"
                lines.append(f"    {keyword} {condition_str}:")
                lines.append(f"        return {seg_id}")
                first = False

        lines.append("    else:")
        lines.append("        return 0")

        return "\n".join(lines)

    def to_pmml(self) -> str:
        check_is_fitted(self, ['tree_'])

        # Basic PMML structure
        pmml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<PMML version="4.4" xmlns="http://www.dmg.org/PMML-4_4">',
            '  <Header>',
            '    <Application name="CARTSegmenter" version="1.0"/>',
            '  </Header>',
            '  <DataDictionary>',
        ]

        # Add feature definitions
        for feature in self.feature_names_:
            pmml_lines.append(
                f'    <DataField name="{feature}" optype="continuous" dataType="double"/>'
            )

        pmml_lines.extend([
            '    <DataField name="segment" optype="categorical" dataType="integer"/>',
            '  </DataDictionary>',
            '  <TreeModel modelName="CARTSegmenter" functionName="classification">',
            '    <MiningSchema>',
        ])

        for feature in self.feature_names_:
            pmml_lines.append(f'      <MiningField name="{feature}"/>')

        pmml_lines.extend([
            '      <MiningField name="segment" usageType="target"/>',
            '    </MiningSchema>',
        ])

        # Add tree nodes
        pmml_lines.append(self._node_to_pmml(self.cart_tree_, indent=4))

        pmml_lines.extend([
            '  </TreeModel>',
            '</PMML>',
        ])

        return "\n".join(pmml_lines)

    def _node_to_pmml(self, node: CARTNode, indent: int = 0) -> str:
        spaces = "  " * indent
        lines = []

        if node.is_leaf:
            lines.append(
                f'{spaces}<Node id="{node.node_id}" score="{node.segment_id}">'
            )
            lines.append(f'{spaces}  <True/>')
            lines.append(f'{spaces}</Node>')
        else:
            lines.append(f'{spaces}<Node id="{node.node_id}">')

            # Add predicate
            if node.feature and node.threshold is not None:
                lines.append(f'{spaces}  <True/>')

                # Left child (<=)
                if node.left_child:
                    lines.append(
                        f'{spaces}  <Node id="{node.left_child.node_id}">'
                    )
                    lines.append(
                        f'{spaces}    <SimplePredicate field="{node.feature}" '
                        f'operator="lessOrEqual" value="{node.threshold}"/>'
                    )
                    if node.left_child.is_leaf:
                        lines.append(
                            f'{spaces}    <Node score="{node.left_child.segment_id}">'
                        )
                        lines.append(f'{spaces}      <True/>')
                        lines.append(f'{spaces}    </Node>')
                    else:
                        lines.append(
                            self._node_to_pmml(node.left_child, indent + 2)
                        )
                    lines.append(f'{spaces}  </Node>')

                # Right child (>)
                if node.right_child:
                    lines.append(
                        f'{spaces}  <Node id="{node.right_child.node_id}">'
                    )
                    lines.append(
                        f'{spaces}    <SimplePredicate field="{node.feature}" '
                        f'operator="greaterThan" value="{node.threshold}"/>'
                    )
                    if node.right_child.is_leaf:
                        lines.append(
                            f'{spaces}    <Node score="{node.right_child.segment_id}">'
                        )
                        lines.append(f'{spaces}      <True/>')
                        lines.append(f'{spaces}    </Node>')
                    else:
                        lines.append(
                            self._node_to_pmml(node.right_child, indent + 2)
                        )
                    lines.append(f'{spaces}  </Node>')

            lines.append(f'{spaces}</Node>')

        return "\n".join(lines)

    def get_tree_text(self) -> str:
        check_is_fitted(self, ['tree_'])

        return export_text(
            self.tree_,
            feature_names=self.feature_names_
        )

    def get_segment_rules(self) -> Dict[int, List[str]]:
        check_is_fitted(self, ['segment_stats_'])

        return {
            seg_id: stats.rules
            for seg_id, stats in self.segment_stats_.items()
        }

    def get_segment_profiles(self) -> List[SegmentStats]:
        check_is_fitted(self, ['segment_stats_'])

        return sorted(
            self.segment_stats_.values(),
            key=lambda x: x.bad_rate
        )


# CUSTOM CRITERIA CART SEGMENTER

class CustomCARTSegmenter(CARTSegmenter):
    def __init__(
        self,
        custom_criterion: str = "bad_rate_reduction",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.custom_criterion = custom_criterion

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray]
    ) -> 'CustomCARTSegmenter':
        # For now, use standard fitting
        # Custom criteria would require reimplementing the tree building
        # which is complex - sklearn doesn't support custom split criteria directly

        # Use gini as proxy (most similar to bad_rate_reduction)
        self.criterion = "gini"

        return super().fit(X, y)

    def calculate_bad_rate_reduction(
        self,
        y_parent: np.ndarray,
        y_left: np.ndarray,
        y_right: np.ndarray
    ) -> float:
        parent_rate = y_parent.mean()
        left_rate = y_left.mean() if len(y_left) > 0 else 0
        right_rate = y_right.mean() if len(y_right) > 0 else 0

        # Weighted variance of bad rates
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        if n_total == 0:
            return 0

        variance_reduction = (
            (n_left / n_total) * (left_rate - parent_rate) ** 2 +
            (n_right / n_total) * (right_rate - parent_rate) ** 2
        )

        return variance_reduction

    def calculate_ks_improvement(
        self,
        y_left: np.ndarray,
        y_right: np.ndarray
    ) -> float:
        if len(y_left) == 0 or len(y_right) == 0:
            return 0

        # Calculate cumulative distributions
        left_bad_rate = y_left.mean()
        right_bad_rate = y_right.mean()

        # KS is the maximum separation
        ks = abs(left_bad_rate - right_bad_rate)

        return ks


# MODULE EXPORTS

__all__ = [
    # Enums
    "SplitCriterion",
    "PruningStrategy",
    # Dataclasses
    "CARTNode",
    "SegmentStats",
    # Classes
    "CARTSegmenter",
    "CustomCARTSegmenter",
    # Constants
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MIN_SAMPLES_LEAF",
    "DEFAULT_MIN_SAMPLES_SPLIT",
]
