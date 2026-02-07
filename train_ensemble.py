"""
Training pipeline with ensemble model.
"""

import logging
import pickle
from pathlib import Path

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from src.data.loader import FraudDataLoader
from src.features.engineering import FeatureEngineer
from src.models.validation import temporal_train_test_split
from src.models.ensemble import EnsembleFraudDetector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_ensemble():
    """Train ensemble fraud detection model."""

    logger.info("Starting ensemble training pipeline")

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

    logger.info("Training ensemble model")
    model = EnsembleFraudDetector(contamination=train_y.mean())
    model.fit(train_X, train_y)

    logger.info("Evaluating ensemble")
    train_pred = model.predict_proba(train_X)[:, 1]
    test_pred = model.predict_proba(test_X)[:, 1]

    train_auc = roc_auc_score(train_y, train_pred)
    test_auc = roc_auc_score(test_y, test_pred)

    test_pred_binary = (test_pred > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y, test_pred_binary, average="binary"
    )

    logger.info("=" * 60)
    logger.info("Ensemble Performance")
    logger.info("=" * 60)
    logger.info(f"Train ROC-AUC: {train_auc:.4f}")
    logger.info(f"Test ROC-AUC:  {test_auc:.4f}")
    logger.info(f"Precision:     {precision:.4f}")
    logger.info(f"Recall:        {recall:.4f}")
    logger.info(f"F1-Score:      {f1:.4f}")
    logger.info("=" * 60)

    logger.info("Saving models")
    Path("models").mkdir(exist_ok=True)

    model.save("models/ensemble")

    with open("models/feature_engineer.pkl", "wb") as f:
        pickle.dump(engineer, f)

    logger.info("Training complete")

    return {
        "train_auc": train_auc,
        "test_auc": test_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    metrics = train_ensemble()
