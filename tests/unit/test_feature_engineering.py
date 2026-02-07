"""
Unit tests for feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.engineering import FeatureEngineer


def test_feature_engineer_fit_transform():
    """Test basic fit/transform pattern."""
    df = pd.DataFrame(
        {
            "TransactionAmt": [100, 200, 300],
            "TransactionDT": [1000, 2000, 3000],
            "isFraud": [0, 1, 0],
        }
    )

    engineer = FeatureEngineer()
    engineer.fit(df)

    assert engineer.fitted
    assert "amount_mean" in engineer.global_stats

    transformed = engineer.transform(df)
    assert "amount_log" in transformed.columns
    assert "hour" in transformed.columns


def test_no_data_leakage():
    """Test that transform uses only training statistics."""
    train_df = pd.DataFrame(
        {
            "TransactionAmt": [100, 200, 300],
            "TransactionDT": [1000, 2000, 3000],
            "isFraud": [0, 1, 0],
        }
    )

    test_df = pd.DataFrame(
        {
            "TransactionAmt": [1000, 2000],
            "TransactionDT": [4000, 5000],
            "isFraud": [1, 0],
        }
    )

    engineer = FeatureEngineer()
    engineer.fit(train_df)

    train_mean = engineer.global_stats["amount_mean"]

    test_transformed = engineer.transform(test_df)

    assert test_transformed["amount_vs_mean"].iloc[0] == pytest.approx(
        1000 / train_mean
    )


def test_transform_before_fit_raises_error():
    """Test that transform without fit raises error."""
    df = pd.DataFrame({"TransactionAmt": [100, 200], "TransactionDT": [1000, 2000]})

    engineer = FeatureEngineer()

    with pytest.raises(ValueError, match="Must call fit"):
        engineer.transform(df)


def test_prepare_for_model():
    """Test model preparation."""
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionAmt": [100, 200],
            "TransactionDT": [1000, 2000],
            "isFraud": [0, 1],
            "feature1": [1.0, 2.0],
        }
    )

    engineer = FeatureEngineer()
    engineer.fit(df)
    transformed = engineer.transform(df)
    prepared = engineer.prepare_for_model(transformed)

    assert "TransactionID" not in prepared.columns
    assert "isFraud" not in prepared.columns
    assert "feature1" in prepared.columns
