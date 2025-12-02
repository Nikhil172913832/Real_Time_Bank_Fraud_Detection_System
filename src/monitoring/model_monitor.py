"""
Model Monitoring and Data Drift Detection
=========================================

This module provides comprehensive monitoring for ML models in production:
- Data drift detection
- Model performance tracking
- Alert generation
- Metric visualization
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from prometheus_client import Gauge, Counter, Histogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Monitor model performance and detect data drift in production.
    
    Features:
    - Statistical drift detection (KS test, PSI)
    - Model performance degradation alerts
    - Feature distribution monitoring
    - Prometheus metrics export
    """
    
    def __init__(
        self,
        model_name: str = "fraud_detector",
        drift_threshold: float = 0.05,
        performance_threshold: float = 0.10
    ):
        """
        Initialize the model monitor.
        
        Args:
            model_name: Name of the model being monitored
            drift_threshold: P-value threshold for drift detection
            performance_threshold: Acceptable performance degradation (10%)
        """
        self.model_name = model_name
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        
        # Reference data for drift detection
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_metrics: Optional[Dict[str, float]] = None
        
        # Prometheus metrics
        self.model_accuracy = Gauge(
            f'{model_name}_accuracy',
            'Current model accuracy'
        )
        self.data_drift_score = Gauge(
            f'{model_name}_drift_score',
            'Data drift score (0-1)',
            ['feature']
        )
        self.false_positive_rate = Gauge(
            f'{model_name}_false_positive_rate',
            'False positive rate'
        )
        self.false_negative_rate = Gauge(
            f'{model_name}_false_negative_rate',
            'False negative rate'
        )
        self.prediction_distribution = Counter(
            f'{model_name}_predictions',
            'Distribution of predictions',
            ['class']
        )
        
        logger.info(f"Model monitor initialized for {model_name}")
    
    def set_reference_data(
        self,
        X_reference: pd.DataFrame,
        y_reference: Optional[pd.Series] = None,
        y_pred_reference: Optional[np.ndarray] = None
    ) -> None:
        """
        Set reference data for drift detection and performance comparison.
        
        Args:
            X_reference: Reference feature data (training/validation set)
            y_reference: Reference labels (optional)
            y_pred_reference: Reference predictions (optional)
        """
        self.reference_data = X_reference.copy()
        
        # Calculate reference metrics if labels and predictions provided
        if y_reference is not None and y_pred_reference is not None:
            self.reference_metrics = {
                'accuracy': accuracy_score(y_reference, y_pred_reference),
                'precision': precision_score(y_reference, y_pred_reference),
                'recall': recall_score(y_reference, y_pred_reference),
                'f1_score': f1_score(y_reference, y_pred_reference)
            }
            logger.info(f"Reference metrics set: {self.reference_metrics}")
        
        logger.info(f"Reference data set with {len(X_reference)} samples")
    
    def detect_data_drift(
        self,
        X_current: pd.DataFrame,
        method: str = "ks"
    ) -> Dict[str, Any]:
        """
        Detect data drift using statistical tests.
        
        Args:
            X_current: Current production data
            method: Drift detection method ('ks' or 'psi')
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'features': {},
            'drift_detected': False,
            'drifted_features': []
        }
        
        # Check each feature for drift
        for feature in self.reference_data.columns:
            if feature not in X_current.columns:
                continue
            
            ref_values = self.reference_data[feature].dropna()
            curr_values = X_current[feature].dropna()
            
            if method == "ks":
                drift_score, p_value = self._kolmogorov_smirnov_test(
                    ref_values,
                    curr_values
                )
            elif method == "psi":
                drift_score = self._population_stability_index(
                    ref_values,
                    curr_values
                )
                p_value = drift_score  # PSI doesn't have p-value
            else:
                raise ValueError(f"Unknown drift detection method: {method}")
            
            has_drift = (
                p_value < self.drift_threshold if method == "ks"
                else drift_score > 0.2  # PSI threshold
            )
            
            drift_results['features'][feature] = {
                'drift_score': float(drift_score),
                'p_value': float(p_value) if method == "ks" else None,
                'has_drift': has_drift
            }
            
            if has_drift:
                drift_results['drift_detected'] = True
                drift_results['drifted_features'].append(feature)
            
            # Update Prometheus metric
            self.data_drift_score.labels(feature=feature).set(drift_score)
        
        if drift_results['drift_detected']:
            logger.warning(
                f"Data drift detected in features: {drift_results['drifted_features']}"
            )
        else:
            logger.info("No significant data drift detected")
        
        return drift_results
    
    def _kolmogorov_smirnov_test(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test for distribution comparison.
        
        Returns:
            Tuple of (test_statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return statistic, p_value
    
    def _population_stability_index(
        self,
        reference: pd.Series,
        current: pd.Series,
        buckets: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Small change
        PSI >= 0.2: Significant change
        
        Returns:
            PSI score
        """
        # Create bins based on reference data
        try:
            breakpoints = np.percentile(
                reference,
                np.linspace(0, 100, buckets + 1)
            )
            breakpoints = np.unique(breakpoints)
        except Exception:
            return 0.0
        
        # Calculate distributions
        ref_percents = np.histogram(reference, bins=breakpoints)[0] / len(reference)
        curr_percents = np.histogram(current, bins=breakpoints)[0] / len(current)
        
        # Add small constant to avoid division by zero
        ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
        curr_percents = np.where(curr_percents == 0, 0.0001, curr_percents)
        
        # Calculate PSI
        psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
        
        return float(psi)
    
    def track_performance(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Track model performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Calculate FP and FN rates
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Update Prometheus metrics
        self.model_accuracy.set(metrics['accuracy'])
        self.false_positive_rate.set(metrics['false_positive_rate'])
        self.false_negative_rate.set(metrics['false_negative_rate'])
        
        # Update prediction distribution
        for pred in y_pred:
            self.prediction_distribution.labels(class_=str(pred)).inc()
        
        # Check for performance degradation
        if self.reference_metrics:
            degradation = self._check_performance_degradation(metrics)
            if degradation['degraded']:
                logger.warning(
                    f"Performance degradation detected: {degradation['message']}"
                )
        
        logger.info(f"Performance metrics: {metrics}")
        return metrics
    
    def _check_performance_degradation(
        self,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check if model performance has degraded significantly."""
        degradation_info = {
            'degraded': False,
            'message': '',
            'metrics': {}
        }
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric_name not in current_metrics or metric_name not in self.reference_metrics:
                continue
            
            current = current_metrics[metric_name]
            reference = self.reference_metrics[metric_name]
            
            # Calculate relative degradation
            rel_degradation = (reference - current) / reference
            
            degradation_info['metrics'][metric_name] = {
                'current': current,
                'reference': reference,
                'degradation': rel_degradation
            }
            
            if rel_degradation > self.performance_threshold:
                degradation_info['degraded'] = True
                degradation_info['message'] += (
                    f"{metric_name}: {current:.3f} vs {reference:.3f} "
                    f"({rel_degradation*100:.1f}% drop); "
                )
        
        return degradation_info
    
    def generate_report(
        self,
        X_current: pd.DataFrame,
        y_true: Optional[pd.Series] = None,
        y_pred: Optional[np.ndarray] = None,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report.
        
        Args:
            X_current: Current feature data
            y_true: True labels (optional)
            y_pred: Predictions (optional)
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with complete monitoring report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'sample_size': len(X_current)
        }
        
        # Data drift detection
        if self.reference_data is not None:
            report['drift_detection'] = self.detect_data_drift(X_current)
        
        # Performance metrics
        if y_true is not None and y_pred is not None:
            report['performance'] = self.track_performance(
                y_true,
                y_pred,
                y_pred_proba
            )
        
        # Feature statistics
        report['feature_stats'] = {
            'numerical': self._calculate_feature_stats(X_current),
            'missing_values': X_current.isnull().sum().to_dict()
        }
        
        return report
    
    def _calculate_feature_stats(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calculate basic statistics for numerical features."""
        stats_dict = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median())
            }
        
        return stats_dict
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return generate_latest().decode('utf-8')


def example_usage():
    """Example usage of ModelMonitor."""
    # Create sample data
    np.random.seed(42)
    X_ref = pd.DataFrame(np.random.randn(1000, 5), columns=[f'feature_{i}' for i in range(5)])
    y_ref = pd.Series(np.random.randint(0, 2, 1000))
    y_pred_ref = np.random.randint(0, 2, 1000)
    
    # Initialize monitor
    monitor = ModelMonitor(model_name="fraud_detector")
    
    # Set reference data
    monitor.set_reference_data(X_ref, y_ref, y_pred_ref)
    
    # Simulate current data with drift
    X_curr = pd.DataFrame(
        np.random.randn(500, 5) + 0.5,  # Shifted distribution
        columns=[f'feature_{i}' for i in range(5)]
    )
    y_curr = pd.Series(np.random.randint(0, 2, 500))
    y_pred_curr = np.random.randint(0, 2, 500)
    
    # Generate report
    report = monitor.generate_report(X_curr, y_curr, y_pred_curr)
    
    print("=== Monitoring Report ===")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Sample size: {report['sample_size']}")
    
    if 'drift_detection' in report:
        drift = report['drift_detection']
        print(f"\nDrift detected: {drift['drift_detected']}")
        if drift['drift_detected']:
            print(f"Drifted features: {drift['drifted_features']}")
    
    if 'performance' in report:
        perf = report['performance']
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy: {perf['accuracy']:.3f}")
        print(f"  Precision: {perf['precision']:.3f}")
        print(f"  Recall: {perf['recall']:.3f}")
        print(f"  F1-Score: {perf['f1_score']:.3f}")


if __name__ == "__main__":
    example_usage()
