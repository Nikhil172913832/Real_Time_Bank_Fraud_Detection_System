"""
Feature Engineering Pipeline

Stateful feature engineering with fit/transform pattern to prevent data leakage.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Stateful feature engineering preventing train-test leakage.

    Usage:
        engineer = FeatureEngineer()
        engineer.fit(train_df)
        train_features = engineer.transform(train_df)
        test_features = engineer.transform(test_df)
    """

    def __init__(self):
        self.fitted = False
        self.user_stats = {}
        self.global_stats = {}

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Learn statistics from training data only."""
        logger.info(f"Fitting feature engineer on {len(df):,} transactions")

        if "TransactionAmt" in df.columns:
            self.global_stats["amount_mean"] = df["TransactionAmt"].mean()
            self.global_stats["amount_std"] = df["TransactionAmt"].std()
            self.global_stats["amount_median"] = df["TransactionAmt"].median()

        self.fitted = True
        logger.info("Feature engineer fitted")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature transformations using fitted statistics."""
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")

        logger.info(f"Transforming {len(df):,} transactions")
        df = df.copy()

        if "TransactionAmt" in df.columns:
            df["amount_log"] = np.log1p(df["TransactionAmt"])
            df["amount_decimal"] = df["TransactionAmt"] % 1
            df["amount_vs_mean"] = (
                df["TransactionAmt"] / self.global_stats["amount_mean"]
            )
            df["amount_vs_median"] = (
                df["TransactionAmt"] / self.global_stats["amount_median"]
            )

        if "TransactionDT" in df.columns:
            df["hour"] = (df["TransactionDT"] // 3600) % 24
            df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        logger.info(f"Transformation complete: {len(df.columns)} columns")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def prepare_for_model(self, df: pd.DataFrame, drop_cols=None) -> pd.DataFrame:
        """Prepare features for model training."""
        if drop_cols is None:
            drop_cols = ["TransactionID", "TransactionDT", "isFraud"]

        df = df.copy()
        existing_drops = [col for col in drop_cols if col in df.columns]

        if existing_drops:
            df = df.drop(columns=existing_drops)

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype("category").cat.codes

        df = df.fillna(-999)

        return df
