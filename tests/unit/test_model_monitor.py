"""
Unit tests for ModelMonitor
"""

import pytest
import numpy as np
import pandas as pd
from src.monitoring.model_monitor import ModelMonitor


@pytest.fixture
def sample_data():
    """Create sample reference and current data."""
    np.random.seed(42)
    X_ref = pd.DataFrame(
        np.random.randn(100, 5),
        columns=[f'feature_{i}' for i in range(5)]
    )
    y_ref = pd.Series(np.random.randint(0, 2, 100))
    y_pred_ref = np.random.randint(0, 2, 100)
    
    X_curr = pd.DataFrame(
        np.random.randn(50, 5),
        columns=[f'feature_{i}' for i in range(5)]
    )
    y_curr = pd.Series(np.random.randint(0, 2, 50))
    y_pred_curr = np.random.randint(0, 2, 50)
    
    return X_ref, y_ref, y_pred_ref, X_curr, y_curr, y_pred_curr


def test_model_monitor_initialization():
    """Test monitor initialization."""
    monitor = ModelMonitor(model_name="test_model")
    assert monitor.model_name == "test_model"
    assert monitor.drift_threshold == 0.05
    assert monitor.reference_data is None


def test_set_reference_data(sample_data):
    """Test setting reference data."""
    X_ref, y_ref, y_pred_ref, _, _, _ = sample_data
    monitor = ModelMonitor()
    
    monitor.set_reference_data(X_ref, y_ref, y_pred_ref)
    
    assert monitor.reference_data is not None
    assert monitor.reference_metrics is not None
    assert 'accuracy' in monitor.reference_metrics


def test_detect_data_drift_ks(sample_data):
    """Test drift detection using KS test."""
    X_ref, _, _, X_curr, _, _ = sample_data
    monitor = ModelMonitor()
    monitor.set_reference_data(X_ref)
    
    drift_results = monitor.detect_data_drift(X_curr, method="ks")
    
    assert 'timestamp' in drift_results
    assert 'features' in drift_results
    assert 'drift_detected' in drift_results
    assert len(drift_results['features']) == 5


def test_detect_data_drift_psi(sample_data):
    """Test drift detection using PSI."""
    X_ref, _, _, X_curr, _, _ = sample_data
    monitor = ModelMonitor()
    monitor.set_reference_data(X_ref)
    
    drift_results = monitor.detect_data_drift(X_curr, method="psi")
    
    assert 'method' in drift_results
    assert drift_results['method'] == "psi"
    assert len(drift_results['features']) == 5


def test_track_performance(sample_data):
    """Test performance tracking."""
    _, y_ref, y_pred_ref, _, y_curr, y_pred_curr = sample_data
    monitor = ModelMonitor()
    monitor.set_reference_data(None, y_ref, y_pred_ref)
    
    metrics = monitor.track_performance(y_curr, y_pred_curr)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'false_positive_rate' in metrics


def test_generate_report(sample_data):
    """Test report generation."""
    X_ref, y_ref, y_pred_ref, X_curr, y_curr, y_pred_curr = sample_data
    monitor = ModelMonitor()
    monitor.set_reference_data(X_ref, y_ref, y_pred_ref)
    
    report = monitor.generate_report(X_curr, y_curr, y_pred_curr)
    
    assert 'timestamp' in report
    assert 'model_name' in report
    assert 'drift_detection' in report
    assert 'performance' in report
    assert 'feature_stats' in report


def test_drift_detection_without_reference():
    """Test that drift detection fails without reference data."""
    monitor = ModelMonitor()
    X_curr = pd.DataFrame(np.random.randn(10, 3))
    
    with pytest.raises(ValueError):
        monitor.detect_data_drift(X_curr)


def test_population_stability_index():
    """Test PSI calculation."""
    monitor = ModelMonitor()
    
    # Same distribution should have low PSI
    ref = pd.Series(np.random.normal(0, 1, 1000))
    curr = pd.Series(np.random.normal(0, 1, 1000))
    
    psi = monitor._population_stability_index(ref, curr)
    assert psi < 0.1  # No significant change
    
    # Different distribution should have higher PSI
    curr_shifted = pd.Series(np.random.normal(2, 1, 1000))
    psi_shifted = monitor._population_stability_index(ref, curr_shifted)
    assert psi_shifted > 0.1  # Significant change
