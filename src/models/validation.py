"""
Time-based validation for fraud detection.
Prevents data leakage from future transactions.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def temporal_train_test_split(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    test_size: float = 0.2,
    validation_size: float = 0.0,
) -> Tuple[pd.DataFrame, ...]:
    """
    Split data by time, not randomly.

    This is critical for fraud detection because:
    1. Fraud patterns evolve over time
    2. Random split allows future data to leak into training
    3. Real deployment only has past data to train on

    Args:
        df: DataFrame with time column
        time_col: Name of timestamp column
        test_size: Fraction of data for test set (most recent)
        validation_size: Fraction for validation set (if > 0)

    Returns:
        If validation_size > 0: (train, val, test)
        Otherwise: (train, test)

    Example:
        >>> train, test = temporal_train_test_split(df, test_size=0.2)
        >>> # Train on first 80% of time, test on last 20%
    """
    # Sort by time
    df = df.sort_values(time_col).reset_index(drop=True)

    n = len(df)

    if validation_size > 0:
        # Three-way split: train / val / test
        train_end = int(n * (1 - test_size - validation_size))
        val_end = int(n * (1 - test_size))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        logger.info("Temporal split (train/val/test):")
        logger.info(
            f"  Train: {len(train):,} rows | {df[time_col].iloc[0]} to {df[time_col].iloc[train_end - 1]}"
        )
        logger.info(
            f"  Val:   {len(val):,} rows | {df[time_col].iloc[train_end]} to {df[time_col].iloc[val_end - 1]}"
        )
        logger.info(
            f"  Test:  {len(test):,} rows | {df[time_col].iloc[val_end]} to {df[time_col].iloc[-1]}"
        )

        return train, val, test
    else:
        # Two-way split: train / test
        split_idx = int(n * (1 - test_size))

        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        logger.info("Temporal split (train/test):")
        logger.info(
            f"  Train: {len(train):,} rows | {df[time_col].iloc[0]} to {df[time_col].iloc[split_idx - 1]}"
        )
        logger.info(
            f"  Test:  {len(test):,} rows | {df[time_col].iloc[split_idx]} to {df[time_col].iloc[-1]}"
        )

        # Verify no time overlap
        assert train[time_col].max() <= test[time_col].min(), "Time overlap detected!"

        return train, test


def walk_forward_validation(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    n_splits: int = 5,
    test_size: float = 0.2,
) -> list:
    """
    Walk-forward validation for time series.

    Creates multiple train/test splits where test set is always
    after training set in time.

    Args:
        df: DataFrame with time column
        time_col: Name of timestamp column
        n_splits: Number of train/test splits
        test_size: Fraction for each test set

    Returns:
        List of (train_idx, test_idx) tuples

    Example:
        >>> splits = walk_forward_validation(df, n_splits=5)
        >>> for train_idx, test_idx in splits:
        >>>     train = df.iloc[train_idx]
        >>>     test = df.iloc[test_idx]
        >>>     # Train and evaluate
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    test_n = int(n * test_size)

    splits = []
    for i in range(n_splits):
        # Calculate split points
        test_end = n - (n_splits - i - 1) * (test_n // n_splits)
        test_start = test_end - test_n

        train_idx = list(range(test_start))
        test_idx = list(range(test_start, test_end))

        splits.append((train_idx, test_idx))

        logger.info(
            f"Split {i + 1}/{n_splits}: Train={len(train_idx)}, Test={len(test_idx)}"
        )

    return splits


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    df = pd.DataFrame(
        {
            "TransactionDT": range(1000),
            "amount": np.random.randn(1000),
            "isFraud": np.random.randint(0, 2, 1000),
        }
    )

    # Test temporal split
    train, test = temporal_train_test_split(df, test_size=0.2)
    print(f"\nTrain: {len(train)} rows")
    print(f"Test: {len(test)} rows")

    # Test walk-forward validation
    print("\nWalk-forward validation:")
    splits = walk_forward_validation(df, n_splits=3)
