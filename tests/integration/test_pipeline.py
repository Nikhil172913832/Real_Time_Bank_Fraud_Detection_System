"""
Integration test for end-to-end pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.loader import FraudDataLoader
from src.features.engineering import FeatureEngineer
from src.models.validation import temporal_train_test_split
from src.models.ensemble import EnsembleFraudDetector


def test_end_to_end_pipeline():
    """Test complete pipeline from data loading to prediction."""

    sample_data = pd.DataFrame(
        {
            "TransactionID": range(1000),
            "TransactionDT": np.arange(1000) * 100,
            "TransactionAmt": np.random.uniform(10, 1000, 1000),
            "isFraud": np.random.randint(0, 2, 1000),
        }
    )

    train_df, test_df = temporal_train_test_split(sample_data, test_size=0.2)

    engineer = FeatureEngineer()
    engineer.fit(train_df)

    train_features = engineer.transform(train_df)
    test_features = engineer.transform(test_df)

    train_X = engineer.prepare_for_model(train_features)
    test_X = engineer.prepare_for_model(test_features)

    train_y = train_df["isFraud"]
    test_y = test_df["isFraud"]

    model = EnsembleFraudDetector(contamination=0.1)
    model.fit(train_X, train_y)

    predictions = model.predict(test_X)

    assert len(predictions) == len(test_y)
    assert all(p in [0, 1] for p in predictions)


def test_temporal_split_no_overlap():
    """Test that temporal split has no time overlap."""
    df = pd.DataFrame(
        {
            "TransactionDT": range(1000),
            "TransactionAmt": np.random.uniform(10, 1000, 1000),
            "isFraud": np.random.randint(0, 2, 1000),
        }
    )

    train, test = temporal_train_test_split(df, test_size=0.2)

    assert train["TransactionDT"].max() <= test["TransactionDT"].min()
