import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
import warnings

logger = logging.getLogger(__name__)


class FraudDataLoader:
    """Load and validate IEEE-CIS fraud detection dataset."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.train_transaction = None
        self.train_identity = None
        self.test_transaction = None
        self.test_identity = None

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Please download IEEE-CIS dataset first:\n"
                f"  kaggle competitions download -c ieee-fraud-detection -p {self.data_dir}"
            )

    def load(
        self, load_test: bool = False, sample_frac: Optional[float] = None
    ) -> pd.DataFrame:
        """Load and merge transaction and identity data."""
        logger.info("Loading IEEE-CIS fraud detection dataset...")

        if load_test:
            txn_file = self.data_dir / "test_transaction.csv"
            identity_file = self.data_dir / "test_identity.csv"
        else:
            txn_file = self.data_dir / "train_transaction.csv"
            identity_file = self.data_dir / "train_identity.csv"

        if not txn_file.exists():
            raise FileNotFoundError(
                f"Transaction file not found: {txn_file}\n"
                f"Please download IEEE-CIS dataset first."
            )

        logger.info(f"Loading transaction data from {txn_file.name}...")
        self.train_transaction = pd.read_csv(txn_file)
        logger.info(f"Loaded {len(self.train_transaction):,} transactions")

        if identity_file.exists():
            logger.info(f"Loading identity data from {identity_file.name}...")
            self.train_identity = pd.read_csv(identity_file)
            logger.info(f"Loaded {len(self.train_identity):,} identity records")

            logger.info("Merging transaction and identity data...")
            df = self.train_transaction.merge(
                self.train_identity, on="TransactionID", how="left"
            )
            logger.info(f"Merged dataset: {len(df):,} rows, {len(df.columns)} columns")
        else:
            logger.warning(f"Identity file not found: {identity_file.name}")
            logger.warning("Proceeding with transaction data only")
            df = self.train_transaction.copy()

        if sample_frac is not None:
            logger.info(f"Sampling {sample_frac:.1%} of data...")
            df = df.sample(frac=sample_frac, random_state=42)
            logger.info(f"Sampled dataset: {len(df):,} rows")

        if "isFraud" in df.columns:
            fraud_count = df["isFraud"].sum()
            fraud_rate = df["isFraud"].mean()
            logger.info(f"Fraud transactions: {fraud_count:,} ({fraud_rate:.2%})")

        return df

    def validate(self, df: pd.DataFrame) -> None:
        """Validate data quality and log warnings."""
        logger.info("Validating data quality...")

        required = ["TransactionID", "TransactionDT", "TransactionAmt"]
        if "isFraud" in df.columns:
            required.append("isFraud")

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Required columns present")

        if "isFraud" in df.columns:
            fraud_rate = df["isFraud"].mean()
            if fraud_rate < 0.01 or fraud_rate > 0.5:
                logger.warning(
                    f"Unusual fraud rate: {fraud_rate:.2%} "
                    f"(expected 2-5% for IEEE-CIS)"
                )
            else:
                logger.info(f"Fraud rate: {fraud_rate:.2%}")

        dupes = df["TransactionID"].duplicated().sum()
        if dupes > 0:
            logger.warning(f"Found {dupes:,} duplicate transactions")
        else:
            logger.info("No duplicate transactions")

        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        high_missing = missing_pct[missing_pct > 50]

        if len(high_missing) > 0:
            logger.warning(
                f"{len(high_missing)} columns with >50% missing values:\n"
                f"{high_missing.head(10)}"
            )

        logger.info(f"Data types: {df.dtypes.value_counts().to_dict()}")

        logger.info("=" * 60)
        logger.info("Data validation summary")
        logger.info("=" * 60)
        logger.info(f"Total rows: {len(df):,}")
        logger.info(f"Total columns: {len(df.columns)}")
        logger.info(
            f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        )
        logger.info("=" * 60)

    def get_basic_stats(self, df: pd.DataFrame) -> dict:
        """Get basic statistics about the dataset."""
        stats = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }

        if "isFraud" in df.columns:
            stats["fraud_count"] = int(df["isFraud"].sum())
            stats["fraud_rate"] = float(df["isFraud"].mean())

        stats["missing_values"] = {
            "total": int(df.isnull().sum().sum()),
            "columns_with_missing": int((df.isnull().sum() > 0).sum()),
        }

        return stats


def quick_load_and_validate(
    data_dir: str = "data/raw", sample_frac: Optional[float] = None
) -> pd.DataFrame:
    """Load and validate data in one call."""
    loader = FraudDataLoader(data_dir=data_dir)
    df = loader.load(sample_frac=sample_frac)
    loader.validate(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("IEEE-CIS Fraud Detection Data Loader")
    print("=" * 60)

    try:
        loader = FraudDataLoader(data_dir="data/raw")
        df = loader.load(sample_frac=0.1)

        loader.validate(df)

        stats = loader.get_basic_stats(df)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\nData loader finished successfully")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo download the dataset:")
        print("  1. Install Kaggle CLI: pip install kaggle")
        print("  2. Configure API token: https://www.kaggle.com/settings")
        print("  3. Download dataset:")
        print("     kaggle competitions download -c ieee-fraud-detection -p data/raw/")
        print("     cd data/raw && unzip ieee-fraud-detection.zip")
