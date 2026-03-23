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
    """Evaluate a trained model on the held-out split."""

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
        logger.info("Using advanced metrics")
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

        logger.info("Test samples: %s | Fraud rate: %.2f%%", f"{len(test_X):,}", test_y.mean() * 100)
        logger.info(
            "AUC: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
            auc,
            precision,
            recall,
            f1,
        )
        logger.info("TN: %s | FP: %s | FN: %s | TP: %s", f"{tn:,}", f"{fp:,}", f"{fn:,}", f"{tp:,}")
        logger.info(
            "Inference: %.2fs | Throughput: %.0f TPS | Latency: %.2f ms",
            inference_time,
            len(test_X) / inference_time,
            inference_time / len(test_X) * 1000,
        )

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
