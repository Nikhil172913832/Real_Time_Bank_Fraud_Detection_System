"""
Enhanced Model Performance Monitoring
=====================================

Extended monitoring with drift detection, performance tracking, and alerting.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
from scipy import stats
from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger(__name__)


class EnhancedModelMonitor:
    """
    Enhanced model monitoring with drift detection and performance tracking.
    
    Features:
    - Data drift detection (KS test, PSI)
    - Concept drift detection (performance degradation)
    - Feature importance drift
    - Prediction distribution monitoring
    - Prometheus metrics integration
    """
    
    def __init__(
        self,
        model_name: str = "fraud_detector",
        drift_threshold: float = 0.1,
        performance_threshold: float = 0.05
    ):
        """
        Initialize enhanced model monitor.
        
        Args:
            model_name: Name of the model being monitored
            drift_threshold: Threshold for drift detection (0.0-1.0)
            performance_threshold: Threshold for performance degradation (0.0-1.0)
        """
        self.model_name = model_name
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        
        # Reference data for drift detection
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_predictions: Optional[np.ndarray] = None
        self.reference_labels: Optional[np.ndarray] = None
        self.reference_performance: Dict[str, float] = {}
        
        # Monitoring history
        self.drift_history: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for monitoring."""
        self.drift_score_gauge = Gauge(
            f'{self.model_name}_drift_score',
            'Data drift score (PSI)',
            ['feature']
        )
        
        self.performance_gauge = Gauge(
            f'{self.model_name}_performance',
            'Model performance metric',
            ['metric']
        )
        
        self.drift_alert_counter = Counter(
            f'{self.model_name}_drift_alerts_total',
            'Total number of drift alerts'
        )
        
        self.prediction_distribution = Histogram(
            f'{self.model_name}_prediction_distribution',
            'Distribution of fraud probabilities',
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
    
    def set_reference_data(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None
    ):
        """
        Set reference (baseline) data for drift detection.
        
        Args:
            X: Reference feature DataFrame
            y: Reference labels (optional)
            predictions: Reference predictions (optional)
        """
        self.reference_data = X.copy()
        self.reference_labels = y.copy() if y is not None else None
        self.reference_predictions = predictions.copy() if predictions is not None else None
        
        logger.info(f"Reference data set: {len(X):,} samples, {len(X.columns)} features")
        
        # Calculate reference performance if labels and predictions available
        if y is not None and predictions is not None:
            from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
            
            self.reference_performance = {
                'roc_auc': roc_auc_score(y, predictions),
                'precision': precision_score(y, (predictions > 0.5).astype(int)),
                'recall': recall_score(y, (predictions > 0.5).astype(int)),
                'f1_score': f1_score(y, (predictions > 0.5).astype(int))
            }
            
            logger.info(f"Reference performance: {self.reference_performance}")
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        method: str = "psi"
    ) -> Dict[str, Any]:
        """
        Detect data drift using statistical tests.
        
        Args:
            current_data: Current feature DataFrame
            method: Drift detection method ("psi" or "ks")
            
        Returns:
            Drift detection report
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        logger.info(f"Detecting data drift using {method.upper()} method...")
        
        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "features_drifted": [],
            "drift_scores": {},
            "overall_drift": False
        }
        
        # Get common features
        common_features = list(set(self.reference_data.columns) & set(current_data.columns))
        
        for feature in common_features:
            if method == "psi":
                drift_score = self._calculate_psi(
                    self.reference_data[feature],
                    current_data[feature]
                )
            elif method == "ks":
                drift_score = self._kolmogorov_smirnov_test(
                    self.reference_data[feature],
                    current_data[feature]
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            drift_report["drift_scores"][feature] = float(drift_score)
            
            # Update Prometheus metric
            self.drift_score_gauge.labels(feature=feature).set(drift_score)
            
            # Check if drifted
            if drift_score > self.drift_threshold:
                drift_report["features_drifted"].append(feature)
                logger.warning(f"⚠️ Drift detected in feature '{feature}': {drift_score:.4f}")
        
        # Overall drift flag
        drift_report["overall_drift"] = len(drift_report["features_drifted"]) > 0
        
        if drift_report["overall_drift"]:
            self.drift_alert_counter.inc()
            logger.warning(f"🚨 Data drift detected in {len(drift_report['features_drifted'])} features")
        else:
            logger.info("✅ No significant data drift detected")
        
        # Store in history
        self.drift_history.append(drift_report)
        
        return drift_report
    
    def _calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures the shift in distribution between two datasets.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        
        Args:
            reference: Reference data
            current: Current data
            bins: Number of bins for discretization
            
        Returns:
            PSI score
        """
        # Handle missing values
        reference = reference.dropna()
        current = current.dropna()
        
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages
        ref_pct = ref_counts / len(reference)
        curr_pct = curr_counts / len(current)
        
        # Avoid division by zero
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)
        
        # Calculate PSI
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        
        return float(psi)
    
    def _kolmogorov_smirnov_test(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> float:
        """
        Perform Kolmogorov-Smirnov test for distribution comparison.
        
        Args:
            reference: Reference data
            current: Current data
            
        Returns:
            KS statistic (0.0-1.0)
        """
        reference = reference.dropna()
        current = current.dropna()
        
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        
        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(reference, current)
        
        return float(ks_statistic)
    
    def track_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Track model performance and detect degradation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Performance tracking report
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )
        
        logger.info("Tracking model performance...")
        
        # Calculate metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        
        # Update Prometheus metrics
        for metric_name, value in metrics.items():
            if metric_name != "timestamp":
                self.performance_gauge.labels(metric=metric_name).set(value)
        
        # Check for performance degradation
        if self.reference_performance:
            degradation_detected = False
            degraded_metrics = []
            
            for metric_name, current_value in metrics.items():
                if metric_name == "timestamp":
                    continue
                
                if metric_name in self.reference_performance:
                    reference_value = self.reference_performance[metric_name]
                    degradation = reference_value - current_value
                    
                    if degradation > self.performance_threshold:
                        degradation_detected = True
                        degraded_metrics.append({
                            "metric": metric_name,
                            "reference": reference_value,
                            "current": current_value,
                            "degradation": degradation
                        })
            
            metrics["degradation_detected"] = degradation_detected
            metrics["degraded_metrics"] = degraded_metrics
            
            if degradation_detected:
                logger.warning(f"⚠️ Performance degradation detected:")
                for dm in degraded_metrics:
                    logger.warning(
                        f"  - {dm['metric']}: {dm['reference']:.4f} → {dm['current']:.4f} "
                        f"(Δ {dm['degradation']:.4f})"
                    )
            else:
                logger.info("✅ No performance degradation detected")
        
        # Store in history
        self.performance_history.append(metrics)
        
        # Log metrics
        logger.info(f"Current performance:")
        for metric_name, value in metrics.items():
            if metric_name not in ["timestamp", "degradation_detected", "degraded_metrics"]:
                logger.info(f"  - {metric_name}: {value:.4f}")
        
        return metrics
    
    def monitor_prediction_distribution(self, predictions: np.ndarray):
        """
        Monitor distribution of predictions.
        
        Args:
            predictions: Array of prediction probabilities
        """
        for pred in predictions:
            self.prediction_distribution.observe(float(pred))
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring summary.
        
        Returns:
            Summary of all monitoring activities
        """
        return {
            "model_name": self.model_name,
            "reference_data_size": len(self.reference_data) if self.reference_data is not None else 0,
            "total_drift_checks": len(self.drift_history),
            "total_performance_checks": len(self.performance_history),
            "recent_drift": self.drift_history[-1] if self.drift_history else None,
            "recent_performance": self.performance_history[-1] if self.performance_history else None,
        }
