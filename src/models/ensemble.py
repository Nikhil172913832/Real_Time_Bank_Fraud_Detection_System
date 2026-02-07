"""
Ensemble fraud detector combining supervised and unsupervised methods.
"""

import logging
import pickle
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import numpy as np

logger = logging.getLogger(__name__)


class EnsembleFraudDetector:
    """
    Two-stage ensemble:
    1. Isolation Forest (unsupervised) - catches novel patterns
    2. XGBoost (supervised) - catches known patterns

    Flag as fraud if EITHER model predicts fraud.
    """

    def __init__(self, contamination=0.035):
        self.isolation_forest = IsolationForest(
            contamination=contamination, random_state=42, n_jobs=-1
        )
        self.xgboost = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
        )
        self.fitted = False

    def fit(self, X_train, y_train):
        """Train both models."""
        logger.info("Training ensemble models")

        X_legit = X_train[y_train == 0]
        logger.info(
            f"Training Isolation Forest on {len(X_legit):,} legitimate transactions"
        )
        self.isolation_forest.fit(X_legit)

        logger.info(f"Training XGBoost on {len(X_train):,} transactions")
        self.xgboost.fit(X_train, y_train)

        self.fitted = True
        logger.info("Ensemble training complete")

    def predict(self, X):
        """Predict using ensemble."""
        if not self.fitted:
            raise ValueError("Must call fit() before predict()")

        if_pred = self.isolation_forest.predict(X)
        if_fraud = (if_pred == -1).astype(int)

        xgb_fraud = self.xgboost.predict(X)

        ensemble_pred = np.maximum(if_fraud, xgb_fraud)

        return ensemble_pred

    def predict_proba(self, X):
        """Return fraud probability from XGBoost."""
        if not self.fitted:
            raise ValueError("Must call fit() before predict_proba()")

        return self.xgboost.predict_proba(X)

    def save(self, path_prefix: str):
        """Save ensemble models."""
        with open(f"{path_prefix}_if.pkl", "wb") as f:
            pickle.dump(self.isolation_forest, f)

        with open(f"{path_prefix}_xgb.pkl", "wb") as f:
            pickle.dump(self.xgboost, f)

        logger.info(f"Ensemble saved to {path_prefix}_*.pkl")

    def load(self, path_prefix: str):
        """Load ensemble models."""
        with open(f"{path_prefix}_if.pkl", "rb") as f:
            self.isolation_forest = pickle.load(f)

        with open(f"{path_prefix}_xgb.pkl", "rb") as f:
            self.xgboost = pickle.load(f)

        self.fitted = True
        logger.info(f"Ensemble loaded from {path_prefix}_*.pkl")
