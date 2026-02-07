"""
Evaluation and benchmarking script.
"""

import logging
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.data.loader import FraudDataLoader
from src.features.engineering import FeatureEngineer
from src.models.validation import temporal_train_test_split
from src.monitoring.drift_detector import DriftDetector

try:
    from src.models.advanced_metrics import (
        compute_auprc,
        precision_at_k,
        recall_at_k,
        comprehensive_evaluation,
        print_evaluation_report,
    )

    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str, engineer_path: str, use_advanced_metrics: bool = True
):
    """Comprehensive model evaluation."""

    logger.info("Loading model and data")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(engineer_path, "rb") as f:
        engineer = pickle.load(f)

    loader = FraudDataLoader(data_dir="data/raw")
    df = loader.load()

    train_df, test_df = temporal_train_test_split(df, test_size=0.2)

    test_features = engineer.transform(test_df)
    test_X = engineer.prepare_for_model(test_features)
    test_y = test_df["isFraud"]

    logger.info("Evaluating model performance")

    start_time = time.time()
    test_pred_proba = model.predict_proba(test_X)[:, 1]
    inference_time = time.time() - start_time

    test_pred = (test_pred_proba > 0.5).astype(int)

    if use_advanced_metrics and ADVANCED_METRICS_AVAILABLE:
        logger.info("Using advanced metrics for imbalanced data")
        metrics = comprehensive_evaluation(
            test_y.values, test_pred_proba, k_values=[10, 50, 100, 500]
        )
        print_evaluation_report(metrics)

        metrics["throughput"] = len(test_X) / inference_time
        metrics["latency_ms"] = inference_time / len(test_X) * 1000
    else:
        auc = roc_auc_score(test_y, test_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y, test_pred, average="binary"
        )

        cm = confusion_matrix(test_y, test_pred)
        tn, fp, fn, tp = cm.ravel()

        logger.info("=" * 70)
        logger.info("Model Evaluation Results")
        logger.info("=" * 70)
        logger.info(f"Dataset: IEEE-CIS Fraud Detection")
        logger.info(f"Test samples: {len(test_X):,}")
        logger.info(f"Fraud rate: {test_y.mean():.2%}")
        logger.info("")
        logger.info("Performance Metrics:")
        logger.info(f"  ROC-AUC:    {auc:.4f}")
        logger.info(f"  Precision:  {precision:.4f}")
        logger.info(f"  Recall:     {recall:.4f}")
        logger.info(f"  F1-Score:   {f1:.4f}")
        logger.info("")
        logger.info("Confusion Matrix:")
        logger.info(f"  True Negatives:  {tn:,}")
        logger.info(f"  False Positives: {fp:,}")
        logger.info(f"  False Negatives: {fn:,}")
        logger.info(f"  True Positives:  {tp:,}")
        logger.info("")
        logger.info("Performance:")
        logger.info(f"  Inference time: {inference_time:.2f}s")
        logger.info(f"  Throughput:     {len(test_X) / inference_time:.0f} TPS")
        logger.info(f"  Latency (avg):  {inference_time / len(test_X) * 1000:.2f}ms")
        logger.info("=" * 70)

        metrics = {
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "throughput": len(test_X) / inference_time,
            "latency_ms": inference_time / len(test_X) * 1000,
        }

    logger.info("\nChecking for data drift")
    drift_detector = DriftDetector(engineer.transform(train_df), drift_threshold=0.3)
    drift_results = drift_detector.check_drift(test_features)

    if drift_results["drift_detected"]:
        logger.warning(f"Drift detected: {drift_results['drift_share']:.1%}")
    else:
        logger.info("No significant drift detected")

    metrics["drift_detected"] = drift_results["drift_detected"]

    return metrics


if __name__ == "__main__":
    metrics = evaluate_model("models/fraud_model.pkl", "models/feature_engineer.pkl")
