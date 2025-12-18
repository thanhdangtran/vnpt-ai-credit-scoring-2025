from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtri
import warnings

# Sklearn metrics
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    log_loss,
)
from sklearn.calibration import calibration_curve

# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

@dataclass
class ROCResult:

    auc: float
    gini: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    optimal_threshold: float
    optimal_fpr: float
    optimal_tpr: float

@dataclass
class KSResult:

    ks_statistic: float
    ks_threshold: float
    ks_decile: int
    cumulative_good: np.ndarray
    cumulative_bad: np.ndarray
    thresholds: np.ndarray

@dataclass
class DecileRow:

    decile: int
    min_score: float
    max_score: float
    total_count: int
    bad_count: int
    good_count: int
    bad_rate: float
    good_rate: float
    cumulative_bad_pct: float
    cumulative_good_pct: float
    ks: float
    lift: float

@dataclass
class DecileTable:

    rows: List[DecileRow]
    total_count: int
    total_bad: int
    total_good: int
    overall_bad_rate: float
    max_ks: float
    max_ks_decile: int

@dataclass
class CalibrationResult:

    brier_score: float
    hosmer_lemeshow_stat: float
    hosmer_lemeshow_pvalue: float
    mean_predicted: np.ndarray
    fraction_positives: np.ndarray
    bin_counts: np.ndarray

@dataclass
class PSIResult:

    psi: float
    interpretation: str
    bin_details: pd.DataFrame
    is_stable: bool

@dataclass
class CSIResult:

    feature: str
    csi: float
    interpretation: str
    bin_details: pd.DataFrame
    is_stable: bool

@dataclass
class ModelComparisonResult:

    model_name: str
    auc: float
    gini: float
    ks: float
    brier_score: float
    log_loss: float
    precision_at_10pct: float
    recall_at_10pct: float

@dataclass
class ECLResult:

    pd_12m: float
    pd_lifetime: float
    lgd: float
    ead: float
    ecl_12m: float
    ecl_lifetime: float
    stage: int  # IFRS9 stage 1, 2, or 3

@dataclass
class BaselIRBResult:

    pd: float
    lgd: float
    ead: float
    maturity: float
    risk_weight: float
    rwa: float
    capital_requirement: float

class DiscriminationMetrics:

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize

    # A. ROC Analysis

    def calculate_auc(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> ROCResult:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        # Calculate AUC
        auc = roc_auc_score(y_true, y_prob)

        # Calculate Gini coefficient
        gini = 2 * auc - 1

        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]

        return ROCResult(
            auc=auc,
            gini=gini,
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            optimal_threshold=optimal_threshold,
            optimal_fpr=optimal_fpr,
            optimal_tpr=optimal_tpr
        )

    def calculate_gini(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        auc = roc_auc_score(y_true, y_prob)
        return 2 * auc - 1

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "ROC Curve",
        show_optimal: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> Figure:
        roc_result = self.calculate_auc(y_true, y_prob)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Plot ROC curve
        ax.plot(
            roc_result.fpr,
            roc_result.tpr,
            color='blue',
            lw=2,
            label=f'ROC (AUC = {roc_result.auc:.4f}, Gini = {roc_result.gini:.4f})'
        )

        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')

        # Plot optimal threshold point
        if show_optimal:
            ax.scatter(
                [roc_result.optimal_fpr],
                [roc_result.optimal_tpr],
                color='red',
                s=100,
                zorder=5,
                label=f'Optimal (threshold={roc_result.optimal_threshold:.3f})'
            )

        # Fill area under curve
        ax.fill_between(
            roc_result.fpr,
            roc_result.tpr,
            alpha=0.2,
            color='blue'
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # B. KS Statistic

    def calculate_ks(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 100
    ) -> KSResult:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Create DataFrame for easier manipulation
        df = pd.DataFrame({'prob': y_prob, 'target': y_true})
        df = df.sort_values('prob', ascending=False).reset_index(drop=True)

        # Calculate cumulative distributions
        total_bad = df['target'].sum()
        total_good = len(df) - total_bad

        df['cum_bad'] = df['target'].cumsum() / total_bad
        df['cum_good'] = (1 - df['target']).cumsum() / total_good
        df['ks'] = np.abs(df['cum_bad'] - df['cum_good'])

        # Find max KS
        max_ks_idx = df['ks'].idxmax()
        ks_statistic = df.loc[max_ks_idx, 'ks']
        ks_threshold = df.loc[max_ks_idx, 'prob']

        # Determine KS decile (1-10)
        ks_decile = int(np.ceil((max_ks_idx + 1) / len(df) * 10))

        # Sample for output arrays
        sample_indices = np.linspace(0, len(df) - 1, n_bins, dtype=int)

        return KSResult(
            ks_statistic=ks_statistic,
            ks_threshold=ks_threshold,
            ks_decile=ks_decile,
            cumulative_good=df['cum_good'].values[sample_indices],
            cumulative_bad=df['cum_bad'].values[sample_indices],
            thresholds=df['prob'].values[sample_indices]
        )

    def plot_ks_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "KS Curve",
        ax: Optional[plt.Axes] = None
    ) -> Figure:
        ks_result = self.calculate_ks(y_true, y_prob)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # X-axis: population percentage
        x = np.linspace(0, 100, len(ks_result.cumulative_bad))

        # Plot cumulative distributions
        ax.plot(x, ks_result.cumulative_bad * 100, 'r-', lw=2, label='Cumulative Bad %')
        ax.plot(x, ks_result.cumulative_good * 100, 'g-', lw=2, label='Cumulative Good %')

        # Find and mark KS point
        ks_idx = np.argmax(np.abs(ks_result.cumulative_bad - ks_result.cumulative_good))
        ks_x = x[ks_idx]
        ks_bad = ks_result.cumulative_bad[ks_idx] * 100
        ks_good = ks_result.cumulative_good[ks_idx] * 100

        # Draw KS line
        ax.vlines(
            ks_x, ks_good, ks_bad,
            colors='blue', linestyles='--', lw=2,
            label=f'KS = {ks_result.ks_statistic:.4f} at decile {ks_result.ks_decile}'
        )

        ax.scatter([ks_x, ks_x], [ks_bad, ks_good], color='blue', s=50, zorder=5)

        ax.set_xlim([0, 100])
        ax.set_ylim([0, 105])
        ax.set_xlabel('Population %', fontsize=12)
        ax.set_ylabel('Cumulative %', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_ks_decile_table(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_deciles: int = 10
    ) -> DecileTable:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Create DataFrame
        df = pd.DataFrame({'prob': y_prob, 'target': y_true})

        # Create decile bins based on probability (higher prob = higher risk = lower decile number)
        df['decile'] = pd.qcut(
            df['prob'].rank(method='first'),
            q=n_deciles,
            labels=range(n_deciles, 0, -1)
        )

        # Aggregate by decile
        total_bad = df['target'].sum()
        total_good = len(df) - total_bad
        total_count = len(df)

        rows = []
        cum_bad = 0
        cum_good = 0

        for decile in range(1, n_deciles + 1):
            decile_df = df[df['decile'] == decile]

            n_total = len(decile_df)
            n_bad = decile_df['target'].sum()
            n_good = n_total - n_bad

            bad_rate = n_bad / n_total if n_total > 0 else 0
            good_rate = n_good / n_total if n_total > 0 else 0

            cum_bad += n_bad
            cum_good += n_good

            cum_bad_pct = cum_bad / total_bad if total_bad > 0 else 0
            cum_good_pct = cum_good / total_good if total_good > 0 else 0

            ks = cum_bad_pct - cum_good_pct

            # Calculate lift
            overall_bad_rate = total_bad / total_count
            lift = bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0

            rows.append(DecileRow(
                decile=decile,
                min_score=decile_df['prob'].min(),
                max_score=decile_df['prob'].max(),
                total_count=n_total,
                bad_count=int(n_bad),
                good_count=int(n_good),
                bad_rate=bad_rate,
                good_rate=good_rate,
                cumulative_bad_pct=cum_bad_pct,
                cumulative_good_pct=cum_good_pct,
                ks=ks,
                lift=lift
            ))

        # Find max KS
        max_ks = max(row.ks for row in rows)
        max_ks_decile = next(row.decile for row in rows if row.ks == max_ks)

        return DecileTable(
            rows=rows,
            total_count=total_count,
            total_bad=int(total_bad),
            total_good=int(total_good),
            overall_bad_rate=total_bad / total_count,
            max_ks=max_ks,
            max_ks_decile=max_ks_decile
        )

    def decile_table_to_dataframe(self, decile_table: DecileTable) -> pd.DataFrame:
        data = []
        for row in decile_table.rows:
            data.append({
                'Decile': row.decile,
                'Min Score': f'{row.min_score:.4f}',
                'Max Score': f'{row.max_score:.4f}',
                '#Total': row.total_count,
                '#Bad': row.bad_count,
                '#Good': row.good_count,
                'Bad Rate': f'{row.bad_rate:.2%}',
                'Cum Bad%': f'{row.cumulative_bad_pct:.2%}',
                'Cum Good%': f'{row.cumulative_good_pct:.2%}',
                'KS': f'{row.ks:.2%}',
                'Lift': f'{row.lift:.2f}x'
            })
        return pd.DataFrame(data)

    # C. Precision-Recall

    def calculate_precision_recall(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        results = []
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)

            # Calculate metrics
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results.append({
                'threshold': thresh,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            })

        return pd.DataFrame(results)

    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "Precision-Recall Curve",
        ax: Optional[plt.Axes] = None
    ) -> Figure:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Plot PR curve
        ax.plot(
            recall, precision,
            color='blue', lw=2,
            label=f'PR Curve (AP = {avg_precision:.4f})'
        )

        # Plot baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.2%})')

        # Fill area under curve
        ax.fill_between(recall, precision, alpha=0.2, color='blue')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def calculate_average_precision(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        return average_precision_score(y_true, y_prob)

    # D. Lift Analysis

    def calculate_lift(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_deciles: int = 10
    ) -> pd.DataFrame:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Create DataFrame
        df = pd.DataFrame({'prob': y_prob, 'target': y_true})

        # Create deciles (higher prob = lower decile number)
        df['decile'] = pd.qcut(
            df['prob'].rank(method='first'),
            q=n_deciles,
            labels=range(n_deciles, 0, -1)
        )

        overall_bad_rate = df['target'].mean()

        results = []
        cum_total = 0
        cum_bad = 0

        for decile in range(1, n_deciles + 1):
            decile_df = df[df['decile'] == decile]

            n_total = len(decile_df)
            n_bad = decile_df['target'].sum()
            bad_rate = n_bad / n_total if n_total > 0 else 0
            lift = bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0

            cum_total += n_total
            cum_bad += n_bad
            cum_bad_rate = cum_bad / cum_total if cum_total > 0 else 0
            cum_lift = cum_bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0

            results.append({
                'decile': decile,
                'count': n_total,
                'bad_count': int(n_bad),
                'bad_rate': bad_rate,
                'lift': lift,
                'cumulative_count': cum_total,
                'cumulative_bad': int(cum_bad),
                'cumulative_bad_rate': cum_bad_rate,
                'cumulative_lift': cum_lift
            })

        return pd.DataFrame(results)

    def plot_lift_chart(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_deciles: int = 10,
        title: str = "Lift Chart",
        ax: Optional[plt.Axes] = None
    ) -> Figure:
        lift_df = self.calculate_lift(y_true, y_prob, n_deciles)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Bar plot for lift
        bars = ax.bar(
            lift_df['decile'],
            lift_df['lift'],
            color='steelblue',
            alpha=0.7,
            edgecolor='black'
        )

        # Add baseline
        ax.axhline(y=1.0, color='red', linestyle='--', lw=2, label='Baseline (Lift=1)')

        # Add value labels
        for bar, lift in zip(bars, lift_df['lift']):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f'{lift:.2f}x',
                ha='center',
                va='bottom',
                fontsize=9
            )

        ax.set_xlabel('Decile', fontsize=12)
        ax.set_ylabel('Lift', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(lift_df['decile'])
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def plot_cumulative_gains(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_deciles: int = 10,
        title: str = "Cumulative Gains Chart",
        ax: Optional[plt.Axes] = None
    ) -> Figure:
        lift_df = self.calculate_lift(y_true, y_prob, n_deciles)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        total_bad = lift_df['cumulative_bad'].iloc[-1]

        # Create points for cumulative gains
        x_points = [0] + [d * 10 for d in lift_df['decile']]
        y_points = [0] + [b / total_bad * 100 for b in lift_df['cumulative_bad']]

        # Model curve
        ax.plot(x_points, y_points, 'b-', lw=2, marker='o', label='Model')

        # Perfect model (captures all bads first)
        perfect_x = [0, lift_df['cumulative_bad'].iloc[-1] / lift_df['cumulative_count'].iloc[-1] * 100, 100]
        perfect_y = [0, 100, 100]
        ax.plot(perfect_x, perfect_y, 'g--', lw=2, label='Perfect Model')

        # Random model (diagonal)
        ax.plot([0, 100], [0, 100], 'r--', lw=2, label='Random Model')

        # Fill area between model and random
        ax.fill_between(x_points, y_points, x_points, alpha=0.2, color='blue')

        ax.set_xlim([0, 100])
        ax.set_ylim([0, 105])
        ax.set_xlabel('Population % (Targeted)', fontsize=12)
        ax.set_ylabel('Cumulative Bad % (Captured)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

class CalibrationMetrics:

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize

    # A. Calibration Analysis

    def hosmer_lemeshow_test(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_groups: int = 10
    ) -> CalibrationResult:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Create groups based on predicted probability
        df = pd.DataFrame({'prob': y_prob, 'target': y_true})

        # Use quantile-based binning to ensure roughly equal group sizes
        try:
            df['group'] = pd.qcut(df['prob'], q=n_groups, duplicates='drop')
        except ValueError:
            # If too few unique values, use simple cut
            df['group'] = pd.cut(df['prob'], bins=n_groups)

        # Calculate observed and expected for each group
        grouped = df.groupby('group', observed=True).agg({
            'target': ['sum', 'count'],
            'prob': 'mean'
        }).reset_index()

        grouped.columns = ['group', 'observed', 'n', 'predicted_prob']
        grouped['expected'] = grouped['predicted_prob'] * grouped['n']
        grouped['expected_0'] = (1 - grouped['predicted_prob']) * grouped['n']
        grouped['observed_0'] = grouped['n'] - grouped['observed']

        # Calculate Hosmer-Lemeshow statistic
        # HL = sum((O - E)^2 / (E * (1 - p)))
        hl_stat = 0
        for _, row in grouped.iterrows():
            if row['expected'] > 0 and row['expected_0'] > 0:
                hl_stat += (row['observed'] - row['expected'])**2 / row['expected']
                hl_stat += (row['observed_0'] - row['expected_0'])**2 / row['expected_0']

        # Degrees of freedom = n_groups - 2
        df_hl = len(grouped) - 2
        if df_hl <= 0:
            df_hl = 1

        # p-value from chi-square distribution
        p_value = 1 - stats.chi2.cdf(hl_stat, df_hl)

        # Calculate Brier score
        brier = brier_score_loss(y_true, y_prob)

        return CalibrationResult(
            brier_score=brier,
            hosmer_lemeshow_stat=hl_stat,
            hosmer_lemeshow_pvalue=p_value,
            mean_predicted=grouped['predicted_prob'].values,
            fraction_positives=grouped['observed'].values / grouped['n'].values,
            bin_counts=grouped['n'].values
        )

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        title: str = "Calibration Curve",
        ax: Optional[plt.Axes] = None
    ) -> Figure:
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )

        # Calculate Brier score
        brier = brier_score_loss(y_true, y_prob)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Plot calibration curve
        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            'b-o',
            lw=2,
            label=f'Model (Brier={brier:.4f})'
        )

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Calibration')

        # Add histogram of predictions
        ax2 = ax.twinx()
        ax2.hist(y_prob, bins=n_bins, alpha=0.3, color='gray')
        ax2.set_ylabel('Count', fontsize=10, color='gray')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def calculate_brier_score(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        return brier_score_loss(y_true, y_prob)

    # B. Score Distribution

    def plot_score_distribution(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        title: str = "Score Distribution by Class",
        n_bins: int = 50,
        ax: Optional[plt.Axes] = None
    ) -> Figure:
        y_true = np.asarray(y_true)
        scores = np.asarray(scores)

        good_scores = scores[y_true == 0]
        bad_scores = scores[y_true == 1]

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Plot histograms
        ax.hist(
            good_scores,
            bins=n_bins,
            alpha=0.6,
            color='green',
            label=f'Good (n={len(good_scores)})',
            density=True
        )
        ax.hist(
            bad_scores,
            bins=n_bins,
            alpha=0.6,
            color='red',
            label=f'Bad (n={len(bad_scores)})',
            density=True
        )

        # Add vertical lines for means
        ax.axvline(
            good_scores.mean(),
            color='darkgreen',
            linestyle='--',
            lw=2,
            label=f'Good Mean ({good_scores.mean():.3f})'
        )
        ax.axvline(
            bad_scores.mean(),
            color='darkred',
            linestyle='--',
            lw=2,
            label=f'Bad Mean ({bad_scores.mean():.3f})'
        )

        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
        min_bin_pct: float = 0.05
    ) -> PSIResult:
        expected = np.asarray(expected)
        actual = np.asarray(actual)

        # Create bins based on expected distribution
        _, bin_edges = np.histogram(expected, bins=n_bins)

        # Count frequencies
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        # Convert to percentages
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)

        # Apply minimum to avoid log(0)
        expected_pct = np.maximum(expected_pct, min_bin_pct)
        actual_pct = np.maximum(actual_pct, min_bin_pct)

        # Calculate PSI
        psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        psi = np.sum(psi_values)

        # Interpretation
        if psi < 0.1:
            interpretation = "Không thay đổi đáng kể (Stable)"
            is_stable = True
        elif psi < 0.25:
            interpretation = "Thay đổi nhẹ (Minor shift - investigate)"
            is_stable = True
        else:
            interpretation = "Thay đổi đáng kể (Significant shift - action required)"
            is_stable = False

        # Create bin details
        bin_details = pd.DataFrame({
            'bin': range(1, n_bins + 1),
            'bin_min': bin_edges[:-1],
            'bin_max': bin_edges[1:],
            'expected_count': expected_counts,
            'actual_count': actual_counts,
            'expected_pct': expected_pct,
            'actual_pct': actual_pct,
            'psi_contribution': psi_values
        })

        return PSIResult(
            psi=psi,
            interpretation=interpretation,
            bin_details=bin_details,
            is_stable=is_stable
        )

    def calculate_csi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10
    ) -> float:
        result = self.calculate_psi(expected, actual, n_bins)
        return result.psi

class StabilityMetrics:

    PSI_THRESHOLDS = {
        'stable': 0.1,
        'minor_shift': 0.25,
    }

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize

    # A. PSI - Population Stability Index

    def calculate_psi(
        self,
        base_dist: np.ndarray,
        current_dist: np.ndarray,
        n_bins: int = 10,
        bin_edges: Optional[np.ndarray] = None
    ) -> PSIResult:
        base_dist = np.asarray(base_dist)
        current_dist = np.asarray(current_dist)

        # Define bins
        if bin_edges is None:
            _, bin_edges = np.histogram(base_dist, bins=n_bins)

        # Calculate frequencies
        base_counts, _ = np.histogram(base_dist, bins=bin_edges)
        current_counts, _ = np.histogram(current_dist, bins=bin_edges)

        # Convert to percentages with floor to avoid division by zero
        eps = 1e-10
        base_pct = np.maximum(base_counts / len(base_dist), eps)
        current_pct = np.maximum(current_counts / len(current_dist), eps)

        # Calculate PSI per bin
        psi_per_bin = (current_pct - base_pct) * np.log(current_pct / base_pct)
        total_psi = np.sum(psi_per_bin)

        # Determine interpretation
        if total_psi < self.PSI_THRESHOLDS['stable']:
            interpretation = "Không thay đổi (Stable - no action needed)"
            is_stable = True
        elif total_psi < self.PSI_THRESHOLDS['minor_shift']:
            interpretation = "Thay đổi nhẹ (Minor shift - monitor closely)"
            is_stable = True
        else:
            interpretation = "Thay đổi đáng kể (Significant shift - recalibration recommended)"
            is_stable = False

        # Bin details DataFrame
        bin_details = pd.DataFrame({
            'bin': range(1, len(bin_edges)),
            'bin_min': bin_edges[:-1],
            'bin_max': bin_edges[1:],
            'base_count': base_counts,
            'current_count': current_counts,
            'base_pct': base_pct,
            'current_pct': current_pct,
            'psi_contribution': psi_per_bin,
            'psi_pct': psi_per_bin / total_psi * 100 if total_psi > 0 else 0
        })

        return PSIResult(
            psi=total_psi,
            interpretation=interpretation,
            bin_details=bin_details,
            is_stable=is_stable
        )

    def plot_psi_distribution(
        self,
        base_dist: np.ndarray,
        current_dist: np.ndarray,
        n_bins: int = 10,
        title: str = "PSI Distribution Comparison",
        ax: Optional[plt.Axes] = None
    ) -> Figure:
        psi_result = self.calculate_psi(base_dist, current_dist, n_bins)

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig = ax.figure
            axes = [ax, ax.twinx()]

        # Left plot: Distribution comparison
        ax1 = axes[0]
        bin_centers = (psi_result.bin_details['bin_min'] + psi_result.bin_details['bin_max']) / 2
        width = (bin_centers.iloc[1] - bin_centers.iloc[0]) * 0.4

        ax1.bar(
            bin_centers - width/2,
            psi_result.bin_details['base_pct'],
            width=width,
            alpha=0.7,
            label='Base',
            color='blue'
        )
        ax1.bar(
            bin_centers + width/2,
            psi_result.bin_details['current_pct'],
            width=width,
            alpha=0.7,
            label='Current',
            color='orange'
        )

        ax1.set_xlabel('Score', fontsize=12)
        ax1.set_ylabel('Percentage', fontsize=12)
        ax1.set_title(f'{title}\nPSI = {psi_result.psi:.4f} - {psi_result.interpretation}', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right plot: PSI contribution by bin
        ax2 = axes[1]
        colors = ['green' if p < 0.01 else 'yellow' if p < 0.02 else 'red'
                  for p in psi_result.bin_details['psi_contribution']]

        ax2.bar(
            psi_result.bin_details['bin'],
            psi_result.bin_details['psi_contribution'],
            color=colors,
            alpha=0.7,
            edgecolor='black'
        )

        ax2.axhline(y=0.01, color='yellow', linestyle='--', label='Warning (0.01)')
        ax2.axhline(y=0.02, color='red', linestyle='--', label='Alert (0.02)')

        ax2.set_xlabel('Bin', fontsize=12)
        ax2.set_ylabel('PSI Contribution', fontsize=12)
        ax2.set_title('PSI Contribution by Bin', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # B. CSI - Characteristic Stability Index

    def calculate_csi_per_feature(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_bins: int = 10
    ) -> Dict[str, CSIResult]:
        if features is None:
            features = base_df.select_dtypes(include=[np.number]).columns.tolist()

        results = {}

        for feature in features:
            if feature not in base_df.columns or feature not in current_df.columns:
                continue

            base_values = base_df[feature].dropna().values
            current_values = current_df[feature].dropna().values

            if len(base_values) == 0 or len(current_values) == 0:
                continue

            # Calculate PSI for this feature
            psi_result = self.calculate_psi(base_values, current_values, n_bins)

            results[feature] = CSIResult(
                feature=feature,
                csi=psi_result.psi,
                interpretation=psi_result.interpretation,
                bin_details=psi_result.bin_details,
                is_stable=psi_result.is_stable
            )

        return results

    def identify_drifted_features(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None,
        threshold: float = 0.25
    ) -> pd.DataFrame:
        csi_results = self.calculate_csi_per_feature(base_df, current_df, features)

        drift_data = []
        for feature, result in csi_results.items():
            drift_data.append({
                'feature': feature,
                'csi': result.csi,
                'is_drifted': result.csi >= threshold,
                'interpretation': result.interpretation
            })

        df = pd.DataFrame(drift_data)
        df = df.sort_values('csi', ascending=False)

        return df

    def plot_csi_heatmap(
        self,
        csi_results: Dict[str, CSIResult],
        title: str = "Feature CSI Heatmap"
    ) -> Figure:
        features = list(csi_results.keys())
        csi_values = [csi_results[f].csi for f in features]

        # Sort by CSI value
        sorted_indices = np.argsort(csi_values)[::-1]
        features = [features[i] for i in sorted_indices]
        csi_values = [csi_values[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))

        # Color coding
        colors = []
        for csi in csi_values:
            if csi < 0.1:
                colors.append('green')
            elif csi < 0.25:
                colors.append('yellow')
            else:
                colors.append('red')

        # Horizontal bar chart
        bars = ax.barh(features, csi_values, color=colors, alpha=0.7, edgecolor='black')

        # Add threshold lines
        ax.axvline(x=0.1, color='yellow', linestyle='--', lw=2, label='Minor Shift (0.1)')
        ax.axvline(x=0.25, color='red', linestyle='--', lw=2, label='Significant Shift (0.25)')

        # Add value labels
        for bar, csi in zip(bars, csi_values):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{csi:.3f}',
                va='center',
                fontsize=9
            )

        ax.set_xlabel('CSI Value', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

class DecileAnalysis:

    def __init__(self, n_deciles: int = 10):
        self.n_deciles = n_deciles

    def generate_decile_table(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        score_column_name: str = "PD"
    ) -> DecileTable:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Create DataFrame
        df = pd.DataFrame({
            'score': y_prob,
            'target': y_true
        })

        # Assign deciles (1 = highest risk/score, 10 = lowest)
        df['decile'] = pd.qcut(
            df['score'].rank(method='first'),
            q=self.n_deciles,
            labels=range(self.n_deciles, 0, -1)
        ).astype(int)

        # Calculate totals
        total_count = len(df)
        total_bad = df['target'].sum()
        total_good = total_count - total_bad
        overall_bad_rate = total_bad / total_count

        # Build decile rows
        rows = []
        cum_bad = 0
        cum_good = 0

        for decile in range(1, self.n_deciles + 1):
            decile_df = df[df['decile'] == decile]

            n_total = len(decile_df)
            n_bad = int(decile_df['target'].sum())
            n_good = n_total - n_bad

            bad_rate = n_bad / n_total if n_total > 0 else 0
            good_rate = n_good / n_total if n_total > 0 else 0

            cum_bad += n_bad
            cum_good += n_good

            cum_bad_pct = cum_bad / total_bad if total_bad > 0 else 0
            cum_good_pct = cum_good / total_good if total_good > 0 else 0

            ks = cum_bad_pct - cum_good_pct
            lift = bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0

            rows.append(DecileRow(
                decile=decile,
                min_score=decile_df['score'].min(),
                max_score=decile_df['score'].max(),
                total_count=n_total,
                bad_count=n_bad,
                good_count=n_good,
                bad_rate=bad_rate,
                good_rate=good_rate,
                cumulative_bad_pct=cum_bad_pct,
                cumulative_good_pct=cum_good_pct,
                ks=ks,
                lift=lift
            ))

        # Find max KS
        max_ks = max(row.ks for row in rows)
        max_ks_decile = next(row.decile for row in rows if row.ks == max_ks)

        return DecileTable(
            rows=rows,
            total_count=total_count,
            total_bad=int(total_bad),
            total_good=int(total_good),
            overall_bad_rate=overall_bad_rate,
            max_ks=max_ks,
            max_ks_decile=max_ks_decile
        )

    def to_dataframe(
        self,
        decile_table: DecileTable,
        format_percentages: bool = True
    ) -> pd.DataFrame:
        data = []
        for row in decile_table.rows:
            if format_percentages:
                data.append({
                    'Decile': row.decile,
                    'Min Score': f'{row.min_score:.4f}',
                    'Max Score': f'{row.max_score:.4f}',
                    '#Total': f'{row.total_count:,}',
                    '#Bad': f'{row.bad_count:,}',
                    '#Good': f'{row.good_count:,}',
                    'Bad Rate': f'{row.bad_rate:.2%}',
                    'Cum Bad%': f'{row.cumulative_bad_pct:.2%}',
                    'Cum Good%': f'{row.cumulative_good_pct:.2%}',
                    'KS': f'{row.ks:.2%}',
                    'Lift': f'{row.lift:.2f}x'
                })
            else:
                data.append({
                    'Decile': row.decile,
                    'Min Score': row.min_score,
                    'Max Score': row.max_score,
                    'Total': row.total_count,
                    'Bad': row.bad_count,
                    'Good': row.good_count,
                    'Bad Rate': row.bad_rate,
                    'Cum Bad%': row.cumulative_bad_pct,
                    'Cum Good%': row.cumulative_good_pct,
                    'KS': row.ks,
                    'Lift': row.lift
                })

        return pd.DataFrame(data)

    def plot_decile_analysis(
        self,
        decile_table: DecileTable,
        title: str = "Decile Analysis"
    ) -> Figure:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        deciles = [row.decile for row in decile_table.rows]
        bad_rates = [row.bad_rate for row in decile_table.rows]
        cum_bad = [row.cumulative_bad_pct for row in decile_table.rows]
        cum_good = [row.cumulative_good_pct for row in decile_table.rows]
        ks_values = [row.ks for row in decile_table.rows]
        lifts = [row.lift for row in decile_table.rows]

        # Plot 1: Bad Rate by Decile
        ax1 = axes[0, 0]
        bars = ax1.bar(deciles, bad_rates, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axhline(
            y=decile_table.overall_bad_rate,
            color='red',
            linestyle='--',
            lw=2,
            label=f'Overall Bad Rate ({decile_table.overall_bad_rate:.2%})'
        )
        for bar, rate in zip(bars, bad_rates):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{rate:.1%}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        ax1.set_xlabel('Decile')
        ax1.set_ylabel('Bad Rate')
        ax1.set_title('Bad Rate by Decile')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative Capture Curves
        ax2 = axes[0, 1]
        ax2.plot(deciles, cum_bad, 'r-o', lw=2, label='Cumulative Bad %')
        ax2.plot(deciles, cum_good, 'g-o', lw=2, label='Cumulative Good %')
        ax2.fill_between(deciles, cum_bad, cum_good, alpha=0.2, color='blue')
        ax2.set_xlabel('Decile')
        ax2.set_ylabel('Cumulative %')
        ax2.set_title('Cumulative Capture Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: KS by Decile
        ax3 = axes[1, 0]
        bars = ax3.bar(deciles, ks_values, color='orange', alpha=0.7, edgecolor='black')
        max_ks_idx = ks_values.index(decile_table.max_ks)
        bars[max_ks_idx].set_color('red')
        for bar, ks in zip(bars, ks_values):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{ks:.1%}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        ax3.set_xlabel('Decile')
        ax3.set_ylabel('KS')
        ax3.set_title(f'KS by Decile (Max KS = {decile_table.max_ks:.2%} at Decile {decile_table.max_ks_decile})')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Lift by Decile
        ax4 = axes[1, 1]
        bars = ax4.bar(deciles, lifts, color='purple', alpha=0.7, edgecolor='black')
        ax4.axhline(y=1.0, color='red', linestyle='--', lw=2, label='Baseline (Lift=1)')
        for bar, lift in zip(bars, lifts):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f'{lift:.2f}x',
                ha='center',
                va='bottom',
                fontsize=9
            )
        ax4.set_xlabel('Decile')
        ax4.set_ylabel('Lift')
        ax4.set_title('Lift by Decile')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

class ModelComparer:

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.disc_metrics = DiscriminationMetrics()
        self.calib_metrics = CalibrationMetrics()

    def compare_models(
        self,
        models_dict: Dict[str, Any],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: np.ndarray
    ) -> Dict[str, ModelComparisonResult]:
        results = {}

        for name, model in models_dict.items():
            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict(X_test)

            # Calculate metrics
            auc = roc_auc_score(y_test, y_prob)
            gini = 2 * auc - 1
            ks_result = self.disc_metrics.calculate_ks(y_test, y_prob)
            brier = brier_score_loss(y_test, y_prob)
            ll = log_loss(y_test, y_prob)

            # Precision and recall at 10% threshold
            top_10_pct = int(len(y_test) * 0.1)
            top_indices = np.argsort(y_prob)[-top_10_pct:]
            precision_at_10 = y_test.iloc[top_indices].mean() if hasattr(y_test, 'iloc') else y_test[top_indices].mean()
            recall_at_10 = y_test.iloc[top_indices].sum() / y_test.sum() if hasattr(y_test, 'iloc') else y_test[top_indices].sum() / y_test.sum()

            results[name] = ModelComparisonResult(
                model_name=name,
                auc=auc,
                gini=gini,
                ks=ks_result.ks_statistic,
                brier_score=brier,
                log_loss=ll,
                precision_at_10pct=precision_at_10,
                recall_at_10pct=recall_at_10
            )

        return results

    def compare_from_predictions(
        self,
        predictions_dict: Dict[str, np.ndarray],
        y_test: np.ndarray
    ) -> Dict[str, ModelComparisonResult]:
        results = {}

        for name, y_prob in predictions_dict.items():
            auc = roc_auc_score(y_test, y_prob)
            gini = 2 * auc - 1
            ks_result = self.disc_metrics.calculate_ks(y_test, y_prob)
            brier = brier_score_loss(y_test, y_prob)
            ll = log_loss(y_test, y_prob)

            # Precision and recall at 10%
            top_10_pct = int(len(y_test) * 0.1)
            top_indices = np.argsort(y_prob)[-top_10_pct:]
            y_test_arr = np.asarray(y_test)
            precision_at_10 = y_test_arr[top_indices].mean()
            recall_at_10 = y_test_arr[top_indices].sum() / y_test_arr.sum() if y_test_arr.sum() > 0 else 0

            results[name] = ModelComparisonResult(
                model_name=name,
                auc=auc,
                gini=gini,
                ks=ks_result.ks_statistic,
                brier_score=brier,
                log_loss=ll,
                precision_at_10pct=precision_at_10,
                recall_at_10pct=recall_at_10
            )

        return results

    def statistical_significance_test(
        self,
        y_true: np.ndarray,
        y_prob1: np.ndarray,
        y_prob2: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        y_true = np.asarray(y_true)
        y_prob1 = np.asarray(y_prob1)
        y_prob2 = np.asarray(y_prob2)

        # Calculate AUCs
        auc1 = roc_auc_score(y_true, y_prob1)
        auc2 = roc_auc_score(y_true, y_prob2)

        # DeLong variance estimation
        n1 = np.sum(y_true == 1)  # Number of positives
        n0 = np.sum(y_true == 0)  # Number of negatives

        # Compute placement values for each model
        def compute_placement_values(y_true, y_prob):
            pos_probs = y_prob[y_true == 1]
            neg_probs = y_prob[y_true == 0]

            # For each positive, count negatives with lower score
            v10 = np.array([np.mean(neg_probs < p) + 0.5 * np.mean(neg_probs == p) for p in pos_probs])
            # For each negative, count positives with higher score
            v01 = np.array([np.mean(pos_probs > n) + 0.5 * np.mean(pos_probs == n) for n in neg_probs])

            return v10, v01

        v10_1, v01_1 = compute_placement_values(y_true, y_prob1)
        v10_2, v01_2 = compute_placement_values(y_true, y_prob2)

        # Compute covariance matrix
        s10_1 = np.var(v10_1, ddof=1)
        s10_2 = np.var(v10_2, ddof=1)
        s10_12 = np.cov(v10_1, v10_2, ddof=1)[0, 1]

        s01_1 = np.var(v01_1, ddof=1)
        s01_2 = np.var(v01_2, ddof=1)
        s01_12 = np.cov(v01_1, v01_2, ddof=1)[0, 1]

        # Variance of difference
        var_diff = (s10_1 + s01_1) / n1 + (s10_2 + s01_2) / n0 - 2 * (s10_12 / n1 + s01_12 / n0)

        # Ensure positive variance
        var_diff = max(var_diff, 1e-10)

        # Z-statistic
        z_stat = (auc1 - auc2) / np.sqrt(var_diff)

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Interpretation
        is_significant = p_value < alpha

        return {
            'auc1': auc1,
            'auc2': auc2,
            'auc_diff': auc1 - auc2,
            'z_statistic': z_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': alpha,
            'interpretation': (
                f"AUC difference is {'statistically significant' if is_significant else 'not statistically significant'} "
                f"at α={alpha} (p={p_value:.4f})"
            )
        }

    def plot_roc_comparison(
        self,
        predictions_dict: Dict[str, np.ndarray],
        y_test: np.ndarray,
        title: str = "ROC Curve Comparison"
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))

        for (name, y_prob), color in zip(predictions_dict.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)

            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.4f})')

        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_comparison_report(
        self,
        comparison_results: Dict[str, ModelComparisonResult]
    ) -> pd.DataFrame:
        data = []
        for name, result in comparison_results.items():
            data.append({
                'Model': name,
                'AUC': f'{result.auc:.4f}',
                'Gini': f'{result.gini:.4f}',
                'KS': f'{result.ks:.4f}',
                'Brier Score': f'{result.brier_score:.4f}',
                'Log Loss': f'{result.log_loss:.4f}',
                'Precision@10%': f'{result.precision_at_10pct:.4f}',
                'Recall@10%': f'{result.recall_at_10pct:.4f}'
            })

        df = pd.DataFrame(data)

        # Highlight best values
        return df

# REGULATORY METRICS (NHNN/BASEL)

class RegulatoryMetrics:

    # Default parameters for Vietnamese market
    DEFAULT_LGD = 0.45  # 45% LGD for unsecured
    SECURED_LGD = 0.25  # 25% LGD for secured

    def __init__(
        self,
        correlation_retail: float = 0.04,
        correlation_corporate: float = 0.12,
        confidence_level: float = 0.999
    ):
        self.correlation_retail = correlation_retail
        self.correlation_corporate = correlation_corporate
        self.confidence_level = confidence_level

    # A. IFRS9 ECL Metrics

    def calculate_12m_pd(
        self,
        scores: np.ndarray,
        calibration_func: Optional[callable] = None
    ) -> np.ndarray:
        scores = np.asarray(scores)

        if calibration_func is not None:
            pd_12m = calibration_func(scores)
        else:
            # Assume scores are already PD values
            pd_12m = np.clip(scores, 0.0001, 0.9999)

        return pd_12m

    def calculate_lifetime_pd(
        self,
        pd_12m: np.ndarray,
        term_months: int,
        survival_curve: Optional[np.ndarray] = None
    ) -> np.ndarray:
        pd_12m = np.asarray(pd_12m)
        term_years = term_months / 12

        if survival_curve is not None:
            # Use provided survival curve
            lifetime_pd = 1 - np.prod(1 - survival_curve[:term_months] * pd_12m.reshape(-1, 1), axis=1)
        else:
            # Simple approach: assume constant annual PD
            lifetime_pd = 1 - np.power(1 - pd_12m, term_years)

        return np.clip(lifetime_pd, 0.0001, 0.9999)

    def calculate_lgd(
        self,
        collateral_value: Optional[np.ndarray] = None,
        exposure: Optional[np.ndarray] = None,
        is_secured: bool = False,
        recovery_rate: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        if recovery_rate is not None:
            return 1 - recovery_rate

        if is_secured and collateral_value is not None and exposure is not None:
            collateral_value = np.asarray(collateral_value)
            exposure = np.asarray(exposure)
            lgd = np.maximum(0, 1 - collateral_value / exposure)
            return np.clip(lgd, 0.1, 1.0)  # Floor at 10% LGD

        return self.SECURED_LGD if is_secured else self.DEFAULT_LGD

    def calculate_ead(
        self,
        outstanding: np.ndarray,
        undrawn_limit: Optional[np.ndarray] = None,
        ccf: float = 0.75
    ) -> np.ndarray:
        outstanding = np.asarray(outstanding)

        if undrawn_limit is not None:
            undrawn_limit = np.asarray(undrawn_limit)
            ead = outstanding + ccf * undrawn_limit
        else:
            ead = outstanding

        return ead

    def calculate_ecl(
        self,
        pd: Union[float, np.ndarray],
        lgd: Union[float, np.ndarray],
        ead: Union[float, np.ndarray],
        discount_rate: float = 0.0
    ) -> Union[float, np.ndarray]:
        pd = np.asarray(pd)
        lgd = np.asarray(lgd)
        ead = np.asarray(ead)

        discount_factor = 1 / (1 + discount_rate) if discount_rate > 0 else 1

        ecl = pd * lgd * ead * discount_factor

        return ecl

    def calculate_ecl_by_stage(
        self,
        pd_12m: np.ndarray,
        lgd: Union[float, np.ndarray],
        ead: np.ndarray,
        stages: np.ndarray,
        term_months: np.ndarray
    ) -> Dict[str, np.ndarray]:
        pd_12m = np.asarray(pd_12m)
        lgd = np.asarray(lgd) if hasattr(lgd, '__len__') else np.full(len(pd_12m), lgd)
        ead = np.asarray(ead)
        stages = np.asarray(stages)
        term_months = np.asarray(term_months)

        # Calculate lifetime PD for each loan
        lifetime_pd = np.array([
            self.calculate_lifetime_pd(np.array([pd]), term)[0]
            for pd, term in zip(pd_12m, term_months)
        ])

        # Initialize ECL array
        ecl = np.zeros(len(pd_12m))

        # Stage 1: 12-month ECL
        stage1_mask = stages == 1
        ecl[stage1_mask] = self.calculate_ecl(
            pd_12m[stage1_mask],
            lgd[stage1_mask] if len(lgd) > 1 else lgd,
            ead[stage1_mask]
        )

        # Stage 2 & 3: Lifetime ECL
        stage23_mask = stages >= 2
        ecl[stage23_mask] = self.calculate_ecl(
            lifetime_pd[stage23_mask],
            lgd[stage23_mask] if len(lgd) > 1 else lgd,
            ead[stage23_mask]
        )

        return {
            'ecl': ecl,
            'ecl_stage1': np.sum(ecl[stage1_mask]),
            'ecl_stage2': np.sum(ecl[stages == 2]),
            'ecl_stage3': np.sum(ecl[stages == 3]),
            'total_ecl': np.sum(ecl)
        }

    # B. Basel IRB Metrics

    def calculate_risk_weight(
        self,
        pd: Union[float, np.ndarray],
        lgd: Union[float, np.ndarray],
        maturity: float = 2.5,
        is_retail: bool = True,
        size_adjustment: float = 0.0
    ) -> Union[float, np.ndarray]:
        pd = np.asarray(pd)
        lgd = np.asarray(lgd)

        # Correlation (R) - simplified retail formula
        if is_retail:
            R = self.correlation_retail
        else:
            # Corporate correlation formula
            R = (0.12 * (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)) +
                 0.24 * (1 - (1 - np.exp(-50 * pd)) / (1 - np.exp(-50))))
            R = R - size_adjustment

        # Maturity adjustment (b)
        b = (0.11852 - 0.05478 * np.log(pd)) ** 2

        # Capital requirement (K)
        G = stats.norm.ppf  # Inverse normal CDF

        # Avoid numerical issues
        pd_safe = np.clip(pd, 0.0003, 0.9999)

        K = (lgd * stats.norm.cdf(
            (1 - R) ** (-0.5) * G(pd_safe) +
            (R / (1 - R)) ** 0.5 * G(self.confidence_level)
        ) - pd_safe * lgd) * (1 - 1.5 * b) ** (-1) * (1 + (maturity - 2.5) * b)

        # Risk weight with 1.06 scaling factor
        risk_weight = K * 12.5 * 1.06

        return np.clip(risk_weight, 0, 12.5)  # Cap at 1250%

    def calculate_capital_requirement(
        self,
        rwa: Union[float, np.ndarray],
        capital_ratio: float = 0.08
    ) -> Union[float, np.ndarray]:
        return np.asarray(rwa) * capital_ratio

    def calculate_rwa(
        self,
        ead: np.ndarray,
        risk_weights: np.ndarray
    ) -> float:
        return np.sum(np.asarray(ead) * np.asarray(risk_weights))

    def generate_irb_report(
        self,
        pd: np.ndarray,
        lgd: np.ndarray,
        ead: np.ndarray,
        is_retail: bool = True
    ) -> pd.DataFrame:
        risk_weights = self.calculate_risk_weight(pd, lgd, is_retail=is_retail)
        rwa = ead * risk_weights
        capital = self.calculate_capital_requirement(rwa)

        return pd.DataFrame({
            'PD': pd,
            'LGD': lgd,
            'EAD': ead,
            'Risk Weight': risk_weights,
            'RWA': rwa,
            'Capital': capital
        })

class ModelEvaluationReport:

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.disc_metrics = DiscriminationMetrics(figsize)
        self.calib_metrics = CalibrationMetrics(figsize)
        self.stability_metrics = StabilityMetrics(figsize)
        self.decile_analysis = DecileAnalysis()

    def generate_executive_summary(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, Any]:
        # Discrimination
        roc_result = self.disc_metrics.calculate_auc(y_true, y_prob)
        ks_result = self.disc_metrics.calculate_ks(y_true, y_prob)

        # Calibration
        calib_result = self.calib_metrics.hosmer_lemeshow_test(y_true, y_prob)

        # Sample stats
        n_total = len(y_true)
        n_bad = np.sum(y_true)
        bad_rate = n_bad / n_total

        return {
            'sample_size': n_total,
            'bad_count': int(n_bad),
            'bad_rate': bad_rate,
            'auc': roc_result.auc,
            'gini': roc_result.gini,
            'ks': ks_result.ks_statistic,
            'ks_decile': ks_result.ks_decile,
            'brier_score': calib_result.brier_score,
            'hosmer_lemeshow_stat': calib_result.hosmer_lemeshow_stat,
            'hosmer_lemeshow_pvalue': calib_result.hosmer_lemeshow_pvalue,
            'optimal_threshold': roc_result.optimal_threshold
        }

    def generate_model_report(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        feature_importance: Optional[Dict[str, float]] = None,
        base_distribution: Optional[np.ndarray] = None,
        model_name: str = "Credit Scoring Model",
        output_format: str = "dict"
    ) -> Dict[str, Any]:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        report = {
            'model_name': model_name,
            'report_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # 1. Executive Summary
        report['executive_summary'] = self.generate_executive_summary(y_true, y_prob)

        # 2. Discrimination Metrics
        roc_result = self.disc_metrics.calculate_auc(y_true, y_prob)
        ks_result = self.disc_metrics.calculate_ks(y_true, y_prob)
        ap = self.disc_metrics.calculate_average_precision(y_true, y_prob)

        report['discrimination'] = {
            'auc': roc_result.auc,
            'gini': roc_result.gini,
            'ks': ks_result.ks_statistic,
            'ks_decile': ks_result.ks_decile,
            'average_precision': ap,
            'optimal_threshold': roc_result.optimal_threshold,
            'optimal_tpr': roc_result.optimal_tpr,
            'optimal_fpr': roc_result.optimal_fpr
        }

        # 3. Calibration Metrics
        calib_result = self.calib_metrics.hosmer_lemeshow_test(y_true, y_prob)

        report['calibration'] = {
            'brier_score': calib_result.brier_score,
            'hosmer_lemeshow_stat': calib_result.hosmer_lemeshow_stat,
            'hosmer_lemeshow_pvalue': calib_result.hosmer_lemeshow_pvalue,
            'is_well_calibrated': calib_result.hosmer_lemeshow_pvalue > 0.05
        }

        # 4. Decile Analysis
        decile_table = self.decile_analysis.generate_decile_table(y_true, y_prob)
        report['decile_analysis'] = {
            'table': self.decile_analysis.to_dataframe(decile_table, format_percentages=False).to_dict('records'),
            'max_ks': decile_table.max_ks,
            'max_ks_decile': decile_table.max_ks_decile,
            'total_count': decile_table.total_count,
            'total_bad': decile_table.total_bad,
            'overall_bad_rate': decile_table.overall_bad_rate
        }

        # 5. Stability Analysis (if baseline provided)
        if base_distribution is not None:
            psi_result = self.stability_metrics.calculate_psi(base_distribution, y_prob)
            report['stability'] = {
                'psi': psi_result.psi,
                'interpretation': psi_result.interpretation,
                'is_stable': psi_result.is_stable
            }
        else:
            report['stability'] = None

        # 6. Feature Importance
        if feature_importance is not None:
            sorted_importance = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            report['feature_importance'] = {
                'ranking': [{'feature': f, 'importance': i} for f, i in sorted_importance],
                'top_10': sorted_importance[:10]
            }
        else:
            report['feature_importance'] = None

        # 7. Recommendations
        report['recommendations'] = self._generate_recommendations(report)

        # 8. Model Rating
        report['model_rating'] = self._calculate_model_rating(report)

        return report

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        recommendations = []

        summary = report['executive_summary']

        # AUC recommendations
        if summary['auc'] < 0.65:
            recommendations.append(
                "⚠️ AUC thấp (<0.65): Model có khả năng phân biệt yếu. "
                "Xem xét thêm features mới hoặc thử thuật toán khác."
            )
        elif summary['auc'] < 0.70:
            recommendations.append(
                "⚡ AUC trung bình (0.65-0.70): Model có thể cải thiện. "
                "Xem xét feature engineering hoặc hyperparameter tuning."
            )
        else:
            recommendations.append(
                "✅ AUC tốt (>0.70): Model có khả năng phân biệt tốt."
            )

        # KS recommendations
        if summary['ks'] < 0.20:
            recommendations.append(
                "⚠️ KS thấp (<0.20): Khả năng tách biệt Good/Bad yếu."
            )
        elif summary['ks'] >= 0.40:
            recommendations.append(
                "✅ KS xuất sắc (>0.40): Khả năng tách biệt Good/Bad rất tốt."
            )

        # Calibration recommendations
        if report['calibration']['hosmer_lemeshow_pvalue'] < 0.05:
            recommendations.append(
                "⚠️ Hosmer-Lemeshow test thất bại (p<0.05): Model cần calibration lại. "
                "Xem xét Platt scaling hoặc isotonic regression."
            )

        # Stability recommendations
        if report['stability'] is not None:
            if not report['stability']['is_stable']:
                recommendations.append(
                    f"🔴 PSI cao ({report['stability']['psi']:.3f}): "
                    "Phân phối điểm số thay đổi đáng kể. Cần xem xét rebuild model."
                )

        return recommendations

    def _calculate_model_rating(self, report: Dict[str, Any]) -> Dict[str, Any]:
        summary = report['executive_summary']

        # Score components (0-100 scale)
        auc_score = min(100, max(0, (summary['auc'] - 0.5) * 200))
        ks_score = min(100, max(0, summary['ks'] * 200))

        calib_score = 100 if report['calibration']['hosmer_lemeshow_pvalue'] > 0.05 else 50

        # Overall score
        overall_score = 0.4 * auc_score + 0.3 * ks_score + 0.3 * calib_score

        # Rating
        if overall_score >= 80:
            rating = "Excellent"
            rating_vn = "Xuất sắc"
        elif overall_score >= 65:
            rating = "Good"
            rating_vn = "Tốt"
        elif overall_score >= 50:
            rating = "Acceptable"
            rating_vn = "Chấp nhận được"
        else:
            rating = "Poor"
            rating_vn = "Kém"

        return {
            'overall_score': overall_score,
            'auc_score': auc_score,
            'ks_score': ks_score,
            'calibration_score': calib_score,
            'rating': rating,
            'rating_vn': rating_vn
        }

    def plot_full_report(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "Model Evaluation Report"
    ) -> Figure:
        fig = plt.figure(figsize=(16, 12))

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. ROC Curve
        ax1 = fig.add_subplot(gs[0, 0])
        self.disc_metrics.plot_roc_curve(y_true, y_prob, ax=ax1)

        # 2. KS Curve
        ax2 = fig.add_subplot(gs[0, 1])
        self.disc_metrics.plot_ks_curve(y_true, y_prob, ax=ax2)

        # 3. PR Curve
        ax3 = fig.add_subplot(gs[0, 2])
        self.disc_metrics.plot_pr_curve(y_true, y_prob, ax=ax3)

        # 4. Calibration Curve
        ax4 = fig.add_subplot(gs[1, 0])
        self.calib_metrics.plot_calibration_curve(y_true, y_prob, ax=ax4)

        # 5. Score Distribution
        ax5 = fig.add_subplot(gs[1, 1])
        self.calib_metrics.plot_score_distribution(y_true, y_prob, ax=ax5)

        # 6. Lift Chart
        ax6 = fig.add_subplot(gs[1, 2])
        self.disc_metrics.plot_lift_chart(y_true, y_prob, ax=ax6)

        # 7. Cumulative Gains
        ax7 = fig.add_subplot(gs[2, 0])
        self.disc_metrics.plot_cumulative_gains(y_true, y_prob, ax=ax7)

        # 8. Summary metrics text
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')

        summary = self.generate_executive_summary(y_true, y_prob)
        summary_text = f"""
        MODEL EVALUATION SUMMARY
        ========================

        Sample Size: {summary['sample_size']:,}
        Bad Rate: {summary['bad_rate']:.2%}

        DISCRIMINATION METRICS
        ----------------------
        AUC-ROC: {summary['auc']:.4f}
        Gini: {summary['gini']:.4f}
        KS Statistic: {summary['ks']:.4f} (Decile {summary['ks_decile']})

        CALIBRATION METRICS
        -------------------
        Brier Score: {summary['brier_score']:.4f}
        H-L Test: χ²={summary['hosmer_lemeshow_stat']:.2f}, p={summary['hosmer_lemeshow_pvalue']:.4f}

        Optimal Threshold: {summary['optimal_threshold']:.4f}
        """

        ax8.text(
            0.1, 0.9, summary_text,
            transform=ax8.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        fig.suptitle(title, fontsize=16, fontweight='bold')

        return fig

    def export_report_to_html(
        self,
        report: Dict[str, Any],
        output_path: str,
        include_plots: bool = True
    ) -> str:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report['model_name']} - Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .good {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .bad {{ color: #e74c3c; }}
                .summary-box {{ background-color: #ecf0f1; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>{report['model_name']}</h1>
            <p>Report Generated: {report['report_date']}</p>

            <div class="summary-box">
                <h2>Executive Summary</h2>
                <p>Sample Size: <span class="metric">{report['executive_summary']['sample_size']:,}</span></p>
                <p>Bad Rate: <span class="metric">{report['executive_summary']['bad_rate']:.2%}</span></p>
                <p>AUC: <span class="metric">{report['executive_summary']['auc']:.4f}</span></p>
                <p>Gini: <span class="metric">{report['executive_summary']['gini']:.4f}</span></p>
                <p>KS: <span class="metric">{report['executive_summary']['ks']:.4f}</span></p>
            </div>

            <h2>Model Rating</h2>
            <p>Overall Score: <span class="metric">{report['model_rating']['overall_score']:.1f}/100</span></p>
            <p>Rating: <span class="metric">{report['model_rating']['rating_vn']}</span></p>

            <h2>Recommendations</h2>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in report['recommendations'])}
            </ul>

        </body>
        </html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

def quick_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    plot: bool = True
) -> Dict[str, float]:
    disc = DiscriminationMetrics()
    calib = CalibrationMetrics()

    roc_result = disc.calculate_auc(y_true, y_prob)
    ks_result = disc.calculate_ks(y_true, y_prob)
    brier = calib.calculate_brier_score(y_true, y_prob)

    metrics = {
        'auc': roc_result.auc,
        'gini': roc_result.gini,
        'ks': ks_result.ks_statistic,
        'brier_score': brier
    }

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        disc.plot_roc_curve(y_true, y_prob, ax=axes[0])
        disc.plot_ks_curve(y_true, y_prob, ax=axes[1])
        calib.plot_score_distribution(y_true, y_prob, ax=axes[2])
        plt.tight_layout()
        plt.show()

    return metrics

def compare_two_models(
    y_true: np.ndarray,
    y_prob1: np.ndarray,
    y_prob2: np.ndarray,
    names: Tuple[str, str] = ("Model 1", "Model 2")
) -> Dict[str, Any]:
    comparer = ModelComparer()

    results = comparer.compare_from_predictions(
        {names[0]: y_prob1, names[1]: y_prob2},
        y_true
    )

    significance = comparer.statistical_significance_test(
        y_true, y_prob1, y_prob2
    )

    return {
        'results': results,
        'significance_test': significance
    }
