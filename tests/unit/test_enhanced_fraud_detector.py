"""
Unit tests for EnhancedFraudDetector
"""

import pytest
import numpy as np
import pandas as pd
from src.models.enhanced_fraud_detector import EnhancedFraudDetector


@pytest.fixture
def sample_data():
    """Create sample training and test data."""
    np.random.seed(42)
    X_train = pd.DataFrame(
        np.random.randn(100, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )
    y_train = pd.Series(np.random.randint(0, 2, 100))
    X_test = pd.DataFrame(
        np.random.randn(20, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )
    return X_train, y_train, X_test


def test_enhanced_fraud_detector_initialization():
    """Test detector initialization."""
    detector = EnhancedFraudDetector(model_type="xgboost", enable_shap=True)
    assert detector.model_type == "xgboost"
    assert detector.fraud_threshold == 0.2
    assert detector.enable_shap is True


def test_train_xgboost_model(sample_data):
    """Test training XGBoost model."""
    X_train, y_train, _ = sample_data
    detector = EnhancedFraudDetector(model_type="xgboost", enable_shap=False)
    
    metrics = detector.train(X_train, y_train)
    
    assert 'train_accuracy' in metrics
    assert detector.model is not None
    assert detector.feature_names is not None


def test_predict_with_explanation(sample_data):
    """Test prediction with explanation."""
    X_train, y_train, X_test = sample_data
    detector = EnhancedFraudDetector(model_type="xgboost", enable_shap=True)
    detector.train(X_train, y_train)
    
    transaction = X_test.iloc[:1]
    fraud_prob, is_fraud, explanation = detector.predict_with_explanation(transaction)
    
    assert isinstance(fraud_prob, float)
    assert isinstance(is_fraud, bool)
    assert 'fraud_probability' in explanation
    assert 'model_type' in explanation


def test_get_feature_importance(sample_data):
    """Test feature importance extraction."""
    X_train, y_train, _ = sample_data
    detector = EnhancedFraudDetector(model_type="xgboost", enable_shap=False)
    detector.train(X_train, y_train)
    
    importance_df = detector.get_feature_importance(top_n=5)
    
    assert len(importance_df) == 5
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns


def test_ensemble_model_training(sample_data):
    """Test ensemble model training."""
    X_train, y_train, _ = sample_data
    detector = EnhancedFraudDetector(model_type="ensemble", enable_shap=False)
    
    metrics = detector.train(X_train, y_train)
    
    assert 'train_accuracy' in metrics
    assert detector.model is not None


def test_save_and_load_model(sample_data, tmp_path):
    """Test model save and load."""
    X_train, y_train, X_test = sample_data
    detector = EnhancedFraudDetector(model_type="xgboost", enable_shap=False)
    detector.train(X_train, y_train)
    
    # Save model
    model_path = tmp_path / "test_model.pkl"
    detector.save(str(model_path))
    
    # Load model
    loaded_detector = EnhancedFraudDetector.load(str(model_path))
    
    # Test prediction
    transaction = X_test.iloc[:1]
    original_pred = detector.model.predict_proba(transaction)[0, 1]
    loaded_pred = loaded_detector.model.predict_proba(transaction)[0, 1]
    
    assert abs(original_pred - loaded_pred) < 1e-5


def test_invalid_model_type():
    """Test invalid model type raises error."""
    detector = EnhancedFraudDetector(model_type="invalid_model")
    
    with pytest.raises(ValueError):
        X = pd.DataFrame(np.random.randn(10, 5))
        y = pd.Series(np.random.randint(0, 2, 10))
        detector.train(X, y)
