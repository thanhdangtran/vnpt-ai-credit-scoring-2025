#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    SyntheticDataConfig,
    CreditScoringConfig,
    OutputFormat,
    get_default_config,
    get_small_sample_config,
    get_production_config,
)

from generators import (
    # Base generators
    DemographicGenerator,
    FinancialGenerator,
    CreditHistoryGenerator,
    VNPTBehaviorGenerator,
    # Time series generators
    TransactionSeriesGenerator,
    BehavioralSeriesGenerator,
    # Missing patterns
    MNARPatternGenerator,
    VIETNAMESE_CREDIT_MNAR_RULES,
    TELECOM_MNAR_RULES,
    # Label generator
    LabelGenerator,
    LabelingConfig,
)


# LOGGING SETUP

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("synthetic_data")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# PIPELINE CONFIGURATION

@dataclass
class PipelineConfig:
    # Core settings
    n_samples: int = 10_000
    seed: int = 42
    output_dir: str = "./output"
    output_format: str = "parquet"

    # Time series settings
    time_series_months: int = 24
    generate_transaction_series: bool = True
    generate_behavioral_series: bool = True

    # Missing data settings
    apply_mnar_patterns: bool = True
    mnar_global_max: float = 0.30

    # Label settings
    target_default_rate: float = 0.04
    calibrate_labels: bool = True

    # Export settings
    export_summary: bool = True
    export_metadata: bool = True
    compress: bool = True

    # Performance settings
    chunk_size: int = 10_000  # For large datasets

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# PIPELINE RESULT

@dataclass
class PipelineResult:
    # Generated DataFrames
    demographic_df: pd.DataFrame
    financial_df: pd.DataFrame
    credit_df: pd.DataFrame
    telecom_df: pd.DataFrame
    labels_df: pd.DataFrame

    # Optional time series
    transaction_series_df: Optional[pd.DataFrame] = None
    credit_series_df: Optional[pd.DataFrame] = None
    telecom_series_df: Optional[pd.DataFrame] = None

    # Combined dataset
    master_df: Optional[pd.DataFrame] = None

    # Metadata
    generation_time: float = 0.0
    n_samples: int = 0
    summary: Optional[Dict[str, Any]] = None

    def get_all_dataframes(self) -> Dict[str, pd.DataFrame]:
        dfs = {
            "demographic": self.demographic_df,
            "financial": self.financial_df,
            "credit": self.credit_df,
            "telecom": self.telecom_df,
            "labels": self.labels_df,
        }

        if self.transaction_series_df is not None:
            dfs["transaction_series"] = self.transaction_series_df
        if self.credit_series_df is not None:
            dfs["credit_series"] = self.credit_series_df
        if self.telecom_series_df is not None:
            dfs["telecom_series"] = self.telecom_series_df
        if self.master_df is not None:
            dfs["master"] = self.master_df

        return dfs


# SYNTHETIC DATA PIPELINE

class SyntheticDataPipeline:
    def __init__(
        self,
        n_samples: int = 10_000,
        seed: int = 42,
        output_dir: str = "./output",
        output_format: str = "parquet",
        time_series_months: int = 24,
        target_default_rate: float = 0.04,
        apply_mnar: bool = True,
        config: Optional[PipelineConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # Use provided config or create from parameters
        if config is not None:
            self.config = config
        else:
            self.config = PipelineConfig(
                n_samples=n_samples,
                seed=seed,
                output_dir=output_dir,
                output_format=output_format,
                time_series_months=time_series_months,
                target_default_rate=target_default_rate,
                apply_mnar_patterns=apply_mnar,
            )

        # Setup logger
        self.logger = logger or setup_logging()

        # Create synthetic data config
        self.synth_config = get_default_config()
        self.synth_config.credit_scoring.n_samples = self.config.n_samples
        self.synth_config.credit_scoring.random_seed = self.config.seed
        self.synth_config.credit_scoring.time_series_months = self.config.time_series_months

        # Initialize generators (lazy)
        self._generators_initialized = False

    def _initialize_generators(self) -> None:
        seed = self.config.seed

        self.logger.info("Initializing generators...")

        # Core generators
        self.demo_gen = DemographicGenerator(self.synth_config, seed=seed)
        self.fin_gen = FinancialGenerator(self.synth_config, seed=seed)
        self.credit_gen = CreditHistoryGenerator(self.synth_config, seed=seed)
        self.telecom_gen = VNPTBehaviorGenerator(self.synth_config, seed=seed)

        # Time series generators
        self.transaction_gen = TransactionSeriesGenerator(
            self.synth_config,
            seed=seed,
            n_months=self.config.time_series_months
        )
        self.behavioral_gen = BehavioralSeriesGenerator(
            self.synth_config,
            seed=seed,
            n_months=self.config.time_series_months
        )

        # MNAR generator
        self.mnar_gen = MNARPatternGenerator(
            self.synth_config,
            seed=seed,
            global_max_missing=self.config.mnar_global_max
        )

        # Label generator
        labeling_config = LabelingConfig(
            target_overall_default_rate=self.config.target_default_rate,
            calibrate_to_target=self.config.calibrate_labels,
        )
        self.label_gen = LabelGenerator(
            self.synth_config,
            seed=seed,
            labeling_config=labeling_config
        )

        self._generators_initialized = True

    # PIPELINE STAGES

    def _generate_demographic(self) -> pd.DataFrame:
        self.logger.info("Stage 1/7: Generating demographic data...")
        start = time.time()

        df = self.demo_gen.generate()

        elapsed = time.time() - start
        self.logger.info(f"  Generated {len(df)} demographic records in {elapsed:.2f}s")

        return df

    def _generate_financial(self, demographic_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Stage 2/7: Generating financial data...")
        start = time.time()

        df = self.fin_gen.generate(demographic_df=demographic_df)

        elapsed = time.time() - start
        self.logger.info(f"  Generated {len(df)} financial records in {elapsed:.2f}s")

        return df

    def _generate_credit_history(
        self,
        demographic_df: pd.DataFrame,
        financial_df: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.info("Stage 3/7: Generating credit history...")
        start = time.time()

        df = self.credit_gen.generate(
            demographic_df=demographic_df,
            financial_df=financial_df
        )

        elapsed = time.time() - start
        has_history_pct = df['has_credit_history'].mean() * 100
        self.logger.info(
            f"  Generated {len(df)} credit records in {elapsed:.2f}s "
            f"({has_history_pct:.1f}% with credit history)"
        )

        return df

    def _generate_telecom(
        self,
        demographic_df: pd.DataFrame,
        financial_df: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.info("Stage 4/7: Generating VNPT telecom data...")
        start = time.time()

        df = self.telecom_gen.generate(
            demographic_df=demographic_df,
            financial_df=financial_df
        )

        elapsed = time.time() - start
        vnpt_pct = df['is_vnpt_customer'].mean() * 100
        self.logger.info(
            f"  Generated {len(df)} telecom records in {elapsed:.2f}s "
            f"({vnpt_pct:.1f}% VNPT customers)"
        )

        return df

    def _generate_time_series(
        self,
        demographic_df: pd.DataFrame,
        financial_df: pd.DataFrame,
        credit_df: pd.DataFrame,
        telecom_df: pd.DataFrame
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        transaction_df = None
        credit_series_df = None
        telecom_series_df = None

        # Transaction series
        if self.config.generate_transaction_series:
            self.logger.info("Stage 5a/7: Generating transaction series...")
            start = time.time()

            transaction_df = self.transaction_gen.generate(
                demographic_df=demographic_df,
                financial_df=financial_df
            )

            elapsed = time.time() - start
            n_months = transaction_df['month_id'].nunique()
            self.logger.info(
                f"  Generated {len(transaction_df)} transaction records "
                f"({n_months} months) in {elapsed:.2f}s"
            )

        # Behavioral series (credit + telecom)
        if self.config.generate_behavioral_series:
            self.logger.info("Stage 5b/7: Generating behavioral series...")
            start = time.time()

            credit_series_df, telecom_series_df, _ = self.behavioral_gen.generate(
                demographic_df=demographic_df,
                financial_df=financial_df,
                credit_df=credit_df,
                telecom_df=telecom_df
            )

            elapsed = time.time() - start
            self.logger.info(
                f"  Generated {len(credit_series_df)} credit series + "
                f"{len(telecom_series_df)} telecom series records in {elapsed:.2f}s"
            )

        return transaction_df, credit_series_df, telecom_series_df

    def _apply_mnar_patterns(
        self,
        demographic_df: pd.DataFrame,
        financial_df: pd.DataFrame,
        credit_df: pd.DataFrame,
        telecom_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not self.config.apply_mnar_patterns:
            return demographic_df, financial_df, credit_df, telecom_df

        self.logger.info("Stage 6/7: Applying MNAR missing patterns...")
        start = time.time()

        # Merge for MNAR application
        merged = demographic_df.merge(financial_df, on='customer_id')
        merged = merged.merge(credit_df, on='customer_id', how='left')
        merged = merged.merge(telecom_df, on='customer_id', how='left')

        # Apply Vietnamese credit MNAR rules
        merged, credit_report = self.mnar_gen.apply_all_mnar_rules(
            merged, VIETNAMESE_CREDIT_MNAR_RULES
        )

        # Apply telecom MNAR rules
        merged, telecom_report = self.mnar_gen.apply_all_mnar_rules(
            merged, TELECOM_MNAR_RULES
        )

        elapsed = time.time() - start
        overall_missing = credit_report.overall_missing_rate * 100
        self.logger.info(
            f"  Applied MNAR patterns in {elapsed:.2f}s "
            f"(overall missing: {overall_missing:.2f}%)"
        )

        # Split back into separate DataFrames
        demo_cols = demographic_df.columns.tolist()
        fin_cols = financial_df.columns.tolist()
        credit_cols = credit_df.columns.tolist()
        telecom_cols = telecom_df.columns.tolist()

        # Update original DataFrames with missing values
        for col in fin_cols:
            if col in merged.columns and col != 'customer_id':
                financial_df[col] = merged[col].values
        for col in credit_cols:
            if col in merged.columns and col != 'customer_id':
                credit_df[col] = merged[col].values
        for col in telecom_cols:
            if col in merged.columns and col != 'customer_id':
                telecom_df[col] = merged[col].values

        return demographic_df, financial_df, credit_df, telecom_df

    def _generate_labels(
        self,
        demographic_df: pd.DataFrame,
        financial_df: pd.DataFrame,
        credit_df: pd.DataFrame,
        telecom_df: pd.DataFrame,
        credit_series_df: Optional[pd.DataFrame] = None,
        telecom_series_df: Optional[pd.DataFrame] = None,
        transaction_series_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        self.logger.info("Stage 7/7: Generating labels...")
        start = time.time()

        labels_df = self.label_gen.generate(
            demographic_df=demographic_df,
            financial_df=financial_df,
            credit_df=credit_df,
            telecom_df=telecom_df,
            credit_series=credit_series_df,
            telecom_series=telecom_series_df,
            transaction_series=transaction_series_df
        )

        elapsed = time.time() - start
        default_rate = labels_df['is_default'].mean() * 100
        self.logger.info(
            f"  Generated {len(labels_df)} labels in {elapsed:.2f}s "
            f"(default rate: {default_rate:.2f}%)"
        )

        return labels_df

    def _create_master_dataset(
        self,
        demographic_df: pd.DataFrame,
        financial_df: pd.DataFrame,
        credit_df: pd.DataFrame,
        telecom_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.info("Creating master dataset...")

        master = demographic_df.merge(financial_df, on='customer_id')
        master = master.merge(credit_df, on='customer_id', how='left')
        master = master.merge(telecom_df, on='customer_id', how='left')
        master = master.merge(labels_df, on='customer_id')

        self.logger.info(f"  Master dataset: {len(master)} rows x {len(master.columns)} columns")

        return master

    # MAIN RUN METHOD

    def run(self) -> PipelineResult:
        self.logger.info("=" * 60)
        self.logger.info("Vietnamese Credit Scoring Synthetic Data Generator")
        self.logger.info("=" * 60)
        self.logger.info(f"Samples: {self.config.n_samples:,}")
        self.logger.info(f"Seed: {self.config.seed}")
        self.logger.info(f"Time series months: {self.config.time_series_months}")
        self.logger.info(f"Target default rate: {self.config.target_default_rate*100:.1f}%")
        self.logger.info("=" * 60)

        start_time = time.time()

        # Initialize generators
        if not self._generators_initialized:
            self._initialize_generators()

        # Stage 1: Demographic
        demographic_df = self._generate_demographic()

        # Stage 2: Financial
        financial_df = self._generate_financial(demographic_df)

        # Stage 3: Credit History
        credit_df = self._generate_credit_history(demographic_df, financial_df)

        # Stage 4: Telecom
        telecom_df = self._generate_telecom(demographic_df, financial_df)

        # Stage 5: Time Series
        transaction_df, credit_series_df, telecom_series_df = self._generate_time_series(
            demographic_df, financial_df, credit_df, telecom_df
        )

        # Stage 6: MNAR Patterns
        demographic_df, financial_df, credit_df, telecom_df = self._apply_mnar_patterns(
            demographic_df, financial_df, credit_df, telecom_df
        )

        # Stage 7: Labels
        labels_df = self._generate_labels(
            demographic_df, financial_df, credit_df, telecom_df,
            credit_series_df, telecom_series_df, transaction_df
        )

        # Create master dataset
        master_df = self._create_master_dataset(
            demographic_df, financial_df, credit_df, telecom_df, labels_df
        )

        # Calculate total time
        total_time = time.time() - start_time

        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline completed in {total_time:.2f}s")
        self.logger.info("=" * 60)

        # Generate summary
        summary = self._generate_summary(
            demographic_df, financial_df, credit_df, telecom_df,
            labels_df, transaction_df, credit_series_df, telecom_series_df
        )

        return PipelineResult(
            demographic_df=demographic_df,
            financial_df=financial_df,
            credit_df=credit_df,
            telecom_df=telecom_df,
            labels_df=labels_df,
            transaction_series_df=transaction_df,
            credit_series_df=credit_series_df,
            telecom_series_df=telecom_series_df,
            master_df=master_df,
            generation_time=total_time,
            n_samples=len(demographic_df),
            summary=summary,
        )

    # SUMMARY AND EXPORT

    def _generate_summary(
        self,
        demographic_df: pd.DataFrame,
        financial_df: pd.DataFrame,
        credit_df: pd.DataFrame,
        telecom_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        transaction_df: Optional[pd.DataFrame],
        credit_series_df: Optional[pd.DataFrame],
        telecom_series_df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        summary = {
            "generation_info": {
                "n_samples": len(demographic_df),
                "seed": self.config.seed,
                "timestamp": datetime.now().isoformat(),
                "time_series_months": self.config.time_series_months,
            },
            "dataset_sizes": {
                "demographic": {"rows": len(demographic_df), "cols": len(demographic_df.columns)},
                "financial": {"rows": len(financial_df), "cols": len(financial_df.columns)},
                "credit": {"rows": len(credit_df), "cols": len(credit_df.columns)},
                "telecom": {"rows": len(telecom_df), "cols": len(telecom_df.columns)},
                "labels": {"rows": len(labels_df), "cols": len(labels_df.columns)},
            },
            "demographic_stats": {
                "age_mean": float(demographic_df['age'].mean()),
                "age_std": float(demographic_df['age'].std()),
                "gender_distribution": demographic_df['gender_code'].value_counts(normalize=True).to_dict(),
            },
            "financial_stats": {
                "income_mean": float(financial_df['monthly_income'].mean()),
                "income_median": float(financial_df['monthly_income'].median()),
                "dti_mean": float(financial_df['dti_ratio'].mean()) if 'dti_ratio' in financial_df.columns else None,
            },
            "credit_stats": {
                "has_credit_history_pct": float(credit_df['has_credit_history'].mean()) * 100,
                "avg_cic_score": float(credit_df[credit_df['cic_score'] > 0]['cic_score'].mean()) if (credit_df['cic_score'] > 0).any() else None,
            },
            "telecom_stats": {
                "vnpt_customer_pct": float(telecom_df['is_vnpt_customer'].mean()) * 100,
                "avg_arpu": float(telecom_df[telecom_df['monthly_arpu'] > 0]['monthly_arpu'].mean()) if (telecom_df['monthly_arpu'] > 0).any() else None,
            },
            "label_stats": {
                "default_rate": float(labels_df['is_default'].mean()) * 100,
                "avg_pd_score": float(labels_df['pd_score'].mean()) * 100,
                "risk_grade_distribution": labels_df['risk_grade'].value_counts(normalize=True).to_dict(),
                "segment_distribution": labels_df['segment'].value_counts(normalize=True).to_dict(),
            },
        }

        # Add time series info if generated
        if transaction_df is not None:
            summary["dataset_sizes"]["transaction_series"] = {
                "rows": len(transaction_df),
                "cols": len(transaction_df.columns)
            }
        if credit_series_df is not None:
            summary["dataset_sizes"]["credit_series"] = {
                "rows": len(credit_series_df),
                "cols": len(credit_series_df.columns)
            }
        if telecom_series_df is not None:
            summary["dataset_sizes"]["telecom_series"] = {
                "rows": len(telecom_series_df),
                "cols": len(telecom_series_df.columns)
            }

        return summary

    def export(
        self,
        result: PipelineResult,
        output_dir: Optional[str] = None,
        output_format: Optional[str] = None
    ) -> Dict[str, str]:
        output_dir = output_dir or self.config.output_dir
        output_format = output_format or self.config.output_format

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Exporting data to {output_dir} ({output_format} format)...")

        exported_files = {}
        dataframes = result.get_all_dataframes()

        for name, df in dataframes.items():
            if output_format in ["parquet", "both"]:
                filepath = os.path.join(output_dir, f"{name}.parquet")
                df.to_parquet(filepath, index=False, compression='snappy')
                exported_files[f"{name}_parquet"] = filepath
                self.logger.info(f"  Exported {name}.parquet ({len(df):,} rows)")

            if output_format in ["csv", "both"]:
                filepath = os.path.join(output_dir, f"{name}.csv")
                df.to_csv(filepath, index=False)
                exported_files[f"{name}_csv"] = filepath
                self.logger.info(f"  Exported {name}.csv ({len(df):,} rows)")

        # Export summary
        if self.config.export_summary and result.summary:
            summary_path = os.path.join(output_dir, "summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(result.summary, f, indent=2, ensure_ascii=False, default=str)
            exported_files["summary"] = summary_path
            self.logger.info(f"  Exported summary.json")

        # Export metadata
        if self.config.export_metadata:
            metadata = {
                "config": self.config.to_dict(),
                "generation_time": result.generation_time,
                "n_samples": result.n_samples,
            }
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            exported_files["metadata"] = metadata_path
            self.logger.info(f"  Exported metadata.json")

        self.logger.info(f"Export completed: {len(exported_files)} files")

        return exported_files

    def print_summary(self, result: PipelineResult) -> None:
        if result.summary is None:
            return

        s = result.summary
        print("\n" + "=" * 60)
        print("SYNTHETIC DATA GENERATION SUMMARY")
        print("=" * 60)

        print(f"\nGeneration Info:")
        print(f"  Samples: {s['generation_info']['n_samples']:,}")
        print(f"  Seed: {s['generation_info']['seed']}")
        print(f"  Time series months: {s['generation_info']['time_series_months']}")
        print(f"  Generation time: {result.generation_time:.2f}s")

        print(f"\nDataset Sizes:")
        for name, size in s['dataset_sizes'].items():
            print(f"  {name}: {size['rows']:,} rows x {size['cols']} cols")

        print(f"\nDemographic Stats:")
        print(f"  Age: {s['demographic_stats']['age_mean']:.1f} +/- {s['demographic_stats']['age_std']:.1f}")

        print(f"\nFinancial Stats:")
        print(f"  Income mean: {s['financial_stats']['income_mean']:,.0f} VND")
        print(f"  Income median: {s['financial_stats']['income_median']:,.0f} VND")

        print(f"\nCredit Stats:")
        print(f"  Has credit history: {s['credit_stats']['has_credit_history_pct']:.1f}%")
        if s['credit_stats']['avg_cic_score']:
            print(f"  Avg CIC score: {s['credit_stats']['avg_cic_score']:.0f}")

        print(f"\nTelecom Stats:")
        print(f"  VNPT customers: {s['telecom_stats']['vnpt_customer_pct']:.1f}%")

        print(f"\nLabel Stats:")
        print(f"  Default rate: {s['label_stats']['default_rate']:.2f}%")
        print(f"  Avg PD score: {s['label_stats']['avg_pd_score']:.2f}%")
        print(f"  Risk grade distribution:")
        for grade, pct in sorted(s['label_stats']['risk_grade_distribution'].items()):
            print(f"    {grade}: {pct*100:.1f}%")

        print("\n" + "=" * 60)


# CLI INTERFACE

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vietnamese Credit Scoring Synthetic Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --samples 10000
  python main.py --samples 50000 --output ./data --format both
  python main.py --config production --seed 123
  python main.py --samples 1000 --no-timeseries --no-mnar
        """
    )

    # Core arguments
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10_000,
        help="Number of samples to generate (default: 10000)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["parquet", "csv", "both"],
        default="parquet",
        help="Output format (default: parquet)"
    )

    # Config presets
    parser.add_argument(
        "--config", "-c",
        type=str,
        choices=["default", "small", "production"],
        default=None,
        help="Use preset configuration"
    )

    # Feature flags
    parser.add_argument(
        "--months", "-m",
        type=int,
        default=24,
        help="Time series months (default: 24)"
    )
    parser.add_argument(
        "--default-rate",
        type=float,
        default=0.04,
        help="Target default rate (default: 0.04)"
    )
    parser.add_argument(
        "--no-timeseries",
        action="store_true",
        help="Skip time series generation"
    )
    parser.add_argument(
        "--no-mnar",
        action="store_true",
        help="Skip MNAR missing pattern application"
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip file export (run only)"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (optional)"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)

    # Handle config presets
    if args.config == "small":
        n_samples = 1_000
        months = 12
    elif args.config == "production":
        n_samples = 100_000
        months = 36
    else:
        n_samples = args.samples
        months = args.months

    # Create pipeline config
    config = PipelineConfig(
        n_samples=n_samples,
        seed=args.seed,
        output_dir=args.output,
        output_format=args.format,
        time_series_months=months,
        generate_transaction_series=not args.no_timeseries,
        generate_behavioral_series=not args.no_timeseries,
        apply_mnar_patterns=not args.no_mnar,
        target_default_rate=args.default_rate,
    )

    # Create and run pipeline
    pipeline = SyntheticDataPipeline(config=config, logger=logger)
    result = pipeline.run()

    # Print summary
    pipeline.print_summary(result)

    # Export if not disabled
    if not args.no_export:
        pipeline.export(result)

    logger.info("Done!")


# MODULE EXPORTS

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "SyntheticDataPipeline",
    "setup_logging",
]


if __name__ == "__main__":
    main()
