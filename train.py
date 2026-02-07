"""
Training pipeline for IEEE-CIS fraud detection model.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from xgboost import XGBClassifier

from src.data.loader import FraudDataLoader
from src.features.engineering import FeatureEngineer
from src.models.validation import temporal_train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model():
    """Train fraud detection model on IEEE-CIS dataset."""

    logger.info("Starting training pipeline")

    loader = FraudDataLoader(data_dir="data/raw")
    df = loader.load()
    loader.validate(df)

    logger.info("Splitting data temporally")
    train_df, test_df = temporal_train_test_split(df, test_size=0.2)

    logger.info("Engineering features")
    engineer = FeatureEngineer()
    engineer.fit(train_df)

    train_features = engineer.transform(train_df)
    test_features = engineer.transform(test_df)

    train_X = engineer.prepare_for_model(train_features)
    test_X = engineer.prepare_for_model(test_features)

    train_y = train_df["isFraud"]
    test_y = test_df["isFraud"]

    logger.info(f"Training set: {len(train_X):,} samples")
    logger.info(f"Test set: {len(test_X):,} samples")
    logger.info(f"Features: {len(train_X.columns)}")

    logger.info("Training XGBoost model")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train_X, train_y)

    logger.info("Evaluating model")
    train_pred = model.predict_proba(train_X)[:, 1]
    test_pred = model.predict_proba(test_X)[:, 1]

    train_auc = roc_auc_score(train_y, train_pred)
    test_auc = roc_auc_score(test_y, test_pred)

    test_pred_binary = (test_pred > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y, test_pred_binary, average="binary"
    )

    logger.info("=" * 60)
    logger.info("Model Performance")
    logger.info("=" * 60)
    logger.info(f"Train ROC-AUC: {train_auc:.4f}")
    logger.info(f"Test ROC-AUC:  {test_auc:.4f}")
    logger.info(f"Precision:     {precision:.4f}")
    logger.info(f"Recall:        {recall:.4f}")
    logger.info(f"F1-Score:      {f1:.4f}")
    logger.info("=" * 60)

    logger.info("Saving model and feature engineer")
    Path("models").mkdir(exist_ok=True)

    with open("models/fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/feature_engineer.pkl", "wb") as f:
        pickle.dump(engineer, f)

    feature_names = list(train_X.columns)
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    logger.info("Training complete")

    return {
        "train_auc": train_auc,
        "test_auc": test_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    metrics = train_model()
