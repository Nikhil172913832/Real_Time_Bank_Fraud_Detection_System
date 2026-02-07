"""
Training script for advanced deep learning models.
"""

import logging
import pickle
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from src.data.loader import FraudDataLoader
from src.features.engineering import FeatureEngineer
from src.models.validation import temporal_train_test_split
from src.models.gnn_fraud_detector import GNNFraudDetector
from src.models.transformer_fraud_detector import TransformerFraudDetectorWrapper
from src.models.hybrid_detector import HybridFraudDetectorWrapper
from src.models.advanced_metrics import (
    comprehensive_evaluation,
    print_evaluation_report,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_advanced_models(model_type: str = "hybrid"):
    """
    Train advanced deep learning models.

    Args:
        model_type: One of ['gnn', 'transformer', 'hybrid']
    """
    logger.info(f"Training {model_type.upper()} model")

    loader = FraudDataLoader(data_dir="data/raw")
    df = loader.load(sample_frac=0.1)
    loader.validate(df)

    logger.info("Splitting data temporally")
    train_df, test_df = temporal_train_test_split(df, test_size=0.2)

    train_labels = (
        train_df["isFraud"].values
        if "isFraud" in train_df.columns
        else np.zeros(len(train_df))
    )
    test_labels = (
        test_df["isFraud"].values
        if "isFraud" in test_df.columns
        else np.zeros(len(test_df))
    )

    logger.info(f"Training set: {len(train_df):,} samples")
    logger.info(f"Test set: {len(test_df):,} samples")
    logger.info(f"Fraud rate (train): {train_labels.mean():.2%}")
    logger.info(f"Fraud rate (test): {test_labels.mean():.2%}")

    if model_type == "gnn":
        logger.info("Training Graph Neural Network")
        model = GNNFraudDetector(hidden_dim=64, num_layers=2)
        model.fit(train_df, train_labels, epochs=30)

    elif model_type == "transformer":
        logger.info("Training Temporal Transformer")
        model = TransformerFraudDetectorWrapper(
            input_dim=2, d_model=128, nhead=8, num_layers=4
        )
        model.fit(train_df, epochs=15)

    elif model_type == "hybrid":
        logger.info("Training Hybrid GNN + Transformer")
        model = HybridFraudDetectorWrapper(
            gnn_hidden_dim=64, transformer_d_model=128, fusion_dim=64
        )
        model.fit(
            train_df,
            train_labels,
            gnn_epochs=30,
            transformer_epochs=15,
            fusion_epochs=10,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info("Evaluating model")
    test_pred_proba = model.predict_proba(test_df)[:, 1]

    metrics = comprehensive_evaluation(
        test_labels, test_pred_proba, k_values=[10, 50, 100, 500]
    )

    print_evaluation_report(metrics)

    logger.info("Saving model")
    Path("models").mkdir(exist_ok=True)

    model_path = f"models/{model_type}_fraud_detector.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to {model_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=["gnn", "transformer", "hybrid"],
        help="Model type to train",
    )
    args = parser.parse_args()

    metrics = train_advanced_models(model_type=args.model)
