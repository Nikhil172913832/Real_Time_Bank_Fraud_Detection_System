"""
Model drift detection and monitoring.
"""

import logging
import pandas as pd
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detect data drift using statistical tests.
    Monitors feature distributions and triggers retraining.
    """

    def __init__(self, reference_data: pd.DataFrame, drift_threshold: float = 0.3):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.reference_stats = self._compute_stats(reference_data)

    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """Compute statistical summary of data."""
        stats = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "median": df[col].median(),
            }

        return stats

    def check_drift(self, current_data: pd.DataFrame) -> Dict:
        """Check for drift between reference and current data."""
        current_stats = self._compute_stats(current_data)

        drifted_features = []
        drift_scores = {}

        for col in self.reference_stats:
            if col not in current_stats:
                continue

            ref_mean = self.reference_stats[col]["mean"]
            ref_std = self.reference_stats[col]["std"]
            curr_mean = current_stats[col]["mean"]

            if ref_std > 0:
                drift_score = abs(curr_mean - ref_mean) / ref_std
                drift_scores[col] = drift_score

                if drift_score > 2.0:
                    drifted_features.append(col)

        drift_share = (
            len(drifted_features) / len(self.reference_stats)
            if self.reference_stats
            else 0
        )
        drift_detected = drift_share > self.drift_threshold

        if drift_detected:
            logger.warning(f"Drift detected: {drift_share:.1%} of features drifted")
            logger.warning(f"Drifted features: {drifted_features[:5]}")

        return {
            "drift_detected": drift_detected,
            "drift_share": drift_share,
            "drifted_features": drifted_features,
            "drift_scores": drift_scores,
        }
