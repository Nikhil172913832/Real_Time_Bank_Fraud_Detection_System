"""
Data Validation Pipeline
========================

Schema validation and data quality checks using Pandera.
"""

import logging
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class TransactionSchema:
    """Schema definitions for transaction data validation."""
    
    @staticmethod
    def get_raw_transaction_schema() -> DataFrameSchema:
        """
        Get schema for raw transaction data.
        
        Validates:
        - Data types
        - Value ranges
        - Required fields
        - Business rules
        """
        return DataFrameSchema(
            {
                "transaction_id": Column(str, nullable=False, unique=True),
                "sender_id": Column(str, nullable=False),
                "receiver_id": Column(str, nullable=False),
                "amount": Column(
                    float,
                    checks=[
                        Check.greater_than(0),
                        Check.less_than(1_000_000),  # Max transaction amount
                    ],
                    nullable=False
                ),
                "timestamp": Column(pa.DateTime, nullable=False),
                "source": Column(
                    str,
                    checks=Check.isin(["online", "atm", "pos", "mobile"]),
                    nullable=False
                ),
                "merchant_category": Column(str, nullable=False),
                "is_international": Column(bool, nullable=False),
                "country_code": Column(
                    str,
                    checks=[Check.str_length(2, 3)],
                    nullable=False
                ),
            },
            strict=False,  # Allow additional columns
            coerce=True,   # Attempt type coercion
        )
    
    @staticmethod
    def get_feature_schema() -> DataFrameSchema:
        """
        Get schema for engineered features.
        
        Validates feature engineering output.
        """
        return DataFrameSchema(
            {
                "amount": Column(float, checks=Check.greater_than(0)),
                "amount_log": Column(float, checks=Check.greater_than_or_equal_to(0)),
                "amount_rounded": Column(int, checks=Check.isin([0, 1])),
                "hour_of_day": Column(
                    int,
                    checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(23)]
                ),
                "day_of_week": Column(
                    int,
                    checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(6)]
                ),
                "is_weekend": Column(int, checks=Check.isin([0, 1])),
                "month": Column(
                    int,
                    checks=[Check.greater_than_or_equal_to(1), Check.less_than_or_equal_to(12)]
                ),
                "sender_txn_count": Column(int, checks=Check.greater_than_or_equal_to(1)),
                "time_since_last_txn": Column(float, checks=Check.greater_than_or_equal_to(0)),
                "merchant_risk_level": Column(
                    float,
                    checks=[Check.greater_than_or_equal_to(1), Check.less_than_or_equal_to(5)]
                ),
            },
            strict=False,
            coerce=True,
        )


class DataValidator:
    """
    Data validation pipeline for fraud detection.
    
    Performs schema validation, data quality checks, and anomaly detection.
    """
    
    def __init__(self, enable_strict_validation: bool = True):
        """
        Initialize data validator.
        
        Args:
            enable_strict_validation: If True, raise errors on validation failures
        """
        self.enable_strict_validation = enable_strict_validation
        self.raw_schema = TransactionSchema.get_raw_transaction_schema()
        self.feature_schema = TransactionSchema.get_feature_schema()
        self.validation_stats = {
            "total_validated": 0,
            "total_failures": 0,
            "failure_reasons": {}
        }
    
    def validate_raw_transactions(
        self,
        df: pd.DataFrame,
        raise_on_error: Optional[bool] = None
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate raw transaction data.
        
        Args:
            df: Raw transaction DataFrame
            raise_on_error: Override strict validation setting
            
        Returns:
            Tuple of (validated DataFrame, validation report)
        """
        raise_on_error = raise_on_error if raise_on_error is not None else self.enable_strict_validation
        
        logger.info(f"Validating {len(df):,} raw transactions...")
        
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            # Validate schema
            validated_df = self.raw_schema.validate(df, lazy=True)
            validation_report["passed"] = len(validated_df)
            logger.info(f"✅ All {len(validated_df):,} transactions passed validation")
            
        except pa.errors.SchemaErrors as e:
            validation_report["failed"] = len(e.failure_cases)
            validation_report["errors"] = e.failure_cases.to_dict('records')
            
            logger.warning(f"⚠️ {len(e.failure_cases)} validation failures")
            
            # Log sample errors
            for error in e.failure_cases.head(5).to_dict('records'):
                logger.warning(f"  - {error}")
            
            if raise_on_error:
                raise
            else:
                # Return original DataFrame with validation report
                validated_df = df
        
        # Update stats
        self.validation_stats["total_validated"] += validation_report["total_records"]
        self.validation_stats["total_failures"] += validation_report["failed"]
        
        return validated_df, validation_report
    
    def validate_features(
        self,
        df: pd.DataFrame,
        raise_on_error: Optional[bool] = None
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate engineered features.
        
        Args:
            df: Feature DataFrame
            raise_on_error: Override strict validation setting
            
        Returns:
            Tuple of (validated DataFrame, validation report)
        """
        raise_on_error = raise_on_error if raise_on_error is not None else self.enable_strict_validation
        
        logger.info(f"Validating {len(df):,} feature records...")
        
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            validated_df = self.feature_schema.validate(df, lazy=True)
            validation_report["passed"] = len(validated_df)
            logger.info(f"✅ All {len(validated_df):,} features passed validation")
            
        except pa.errors.SchemaErrors as e:
            validation_report["failed"] = len(e.failure_cases)
            validation_report["errors"] = e.failure_cases.to_dict('records')
            
            logger.warning(f"⚠️ {len(e.failure_cases)} feature validation failures")
            
            if raise_on_error:
                raise
            else:
                validated_df = df
        
        return validated_df, validation_report
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform data quality checks.
        
        Checks:
        - Missing values
        - Duplicate records
        - Outliers
        - Data distribution
        
        Args:
            df: DataFrame to check
            
        Returns:
            Data quality report
        """
        logger.info("Performing data quality checks...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "missing_values": {},
            "duplicates": 0,
            "outliers": {},
            "distribution": {}
        }
        
        # Missing values
        missing = df.isnull().sum()
        report["missing_values"] = {
            col: int(count) for col, count in missing.items() if count > 0
        }
        
        # Duplicates
        if "transaction_id" in df.columns:
            report["duplicates"] = df["transaction_id"].duplicated().sum()
        
        # Outliers (for numeric columns)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    report["outliers"][col] = int(outliers)
        
        # Distribution stats
        if "amount" in df.columns:
            report["distribution"]["amount"] = {
                "mean": float(df["amount"].mean()),
                "median": float(df["amount"].median()),
                "std": float(df["amount"].std()),
                "min": float(df["amount"].min()),
                "max": float(df["amount"].max())
            }
        
        # Log summary
        logger.info(f"Data quality report:")
        logger.info(f"  - Total records: {report['total_records']:,}")
        logger.info(f"  - Missing values: {sum(report['missing_values'].values())}")
        logger.info(f"  - Duplicates: {report['duplicates']}")
        logger.info(f"  - Outliers: {sum(report['outliers'].values())}")
        
        return report
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get cumulative validation statistics."""
        return self.validation_stats.copy()
