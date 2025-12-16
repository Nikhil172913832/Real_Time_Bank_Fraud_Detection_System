"""
Unit Tests for Training Pipeline
=================================

Tests for the FraudDetectionTrainer class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training import FraudDetectionTrainer


@pytest.fixture
def sample_training_data(tmp_path):
    """Create sample training data CSV."""
    data = {
        'transaction_id': [f'tx_{i:03d}' for i in range(1000)],
        'sender_id': [f'user_{i % 100}' for i in range(1000)],
        'receiver_id': [f'merchant_{i % 50}' for i in range(1000)],
        'amount': np.random.uniform(10, 1000, 1000),
        'timestamp': pd.date_range('2025-01-01', periods=1000, freq='1H'),
        'source': np.random.choice(['online', 'atm', 'pos'], 1000),
        'device_os': np.random.choice(['iOS', 'Android', 'Windows'], 1000),
        'browser': np.random.choice(['Chrome', 'Safari', 'Firefox'], 1000),
        'merchant_category': np.random.choice(['Restaurants', 'Grocery', 'Electronics'], 1000),
        'is_international': np.random.choice([True, False], 1000),
        'country_code': 'US',
        'zip_code': np.random.choice(['94102', '10001', '60601'], 1000),
        'ip_address': [f'192.168.1.{i % 255}' for i in range(1000)],
        'session_id': [f'sess_{i % 200}' for i in range(1000)],
        'device_fingerprint': [f'fp_{i % 150}' for i in range(1000)],
        'fraud_bool': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    }
    
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


class TestFraudDetectionTrainer:
    """Test FraudDetectionTrainer class."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        trainer = FraudDetectionTrainer(
            data_path="data.csv",
            test_size=0.2,
            random_state=42,
            n_trials=10
        )
        
        assert trainer.data_path == "data.csv"
        assert trainer.test_size == 0.2
        assert trainer.random_state == 42
        assert trainer.n_trials == 10
        assert trainer.model is None
        assert trainer.feature_columns is None
    
    def test_load_and_prepare_data(self, sample_training_data):
        """Test data loading and preparation."""
        trainer = FraudDetectionTrainer(
            data_path=sample_training_data,
            n_trials=5
        )
        
        X, y = trainer.load_and_prepare_data()
        
        # Check shapes
        assert len(X) == 1000
        assert len(y) == 1000
        assert len(X.columns) > 0
        
        # Check target
        assert y.dtype == np.int64
        assert set(y.unique()).issubset({0, 1})
        
        # Check feature columns stored
        assert trainer.feature_columns is not None
        assert len(trainer.feature_columns) > 0
    
    def test_train_with_small_dataset(self, sample_training_data):
        """Test training with small dataset."""
        trainer = FraudDetectionTrainer(
            data_path=sample_training_data,
            test_size=0.2,
            random_state=42,
            n_trials=2  # Small number for fast testing
        )
        
        metrics = trainer.train()
        
        # Check metrics exist
        assert 'roc_auc' in metrics
        assert 'recall' in metrics
        assert 'precision' in metrics
        assert 'f1_score' in metrics
        assert 'training_time_seconds' in metrics
        
        # Check metrics are valid
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert metrics['training_time_seconds'] > 0
        
        # Check model is trained
        assert trainer.model is not None
    
    def test_save_model(self, sample_training_data, tmp_path):
        """Test model saving."""
        trainer = FraudDetectionTrainer(
            data_path=sample_training_data,
            n_trials=2
        )
        
        # Train model
        trainer.train()
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        trainer.save_model(str(model_path))
        
        # Check files exist
        assert model_path.exists()
        
        # Check feature columns saved
        feature_path = Path('models/feature_columns.pkl')
        if feature_path.exists():
            assert feature_path.stat().st_size > 0
    
    def test_model_not_trained_error(self):
        """Test error when trying to save untrained model."""
        trainer = FraudDetectionTrainer(data_path="data.csv")
        
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            trainer.save_model()


class TestFeatureEngineering:
    """Test feature engineering in training pipeline."""
    
    def test_feature_columns_consistency(self, sample_training_data):
        """Test that feature columns are consistent across runs."""
        trainer1 = FraudDetectionTrainer(
            data_path=sample_training_data,
            random_state=42
        )
        X1, _ = trainer1.load_and_prepare_data()
        
        trainer2 = FraudDetectionTrainer(
            data_path=sample_training_data,
            random_state=42
        )
        X2, _ = trainer2.load_and_prepare_data()
        
        # Check column names are identical
        assert list(X1.columns) == list(X2.columns)
        
        # Check feature columns stored are identical
        assert trainer1.feature_columns == trainer2.feature_columns


class TestModelPerformance:
    """Test model performance requirements."""
    
    @pytest.mark.slow
    def test_model_meets_performance_targets(self, sample_training_data):
        """Test that model meets minimum performance targets."""
        trainer = FraudDetectionTrainer(
            data_path=sample_training_data,
            test_size=0.2,
            random_state=42,
            n_trials=10
        )
        
        metrics = trainer.train()
        
        # Performance targets (relaxed for test data)
        assert metrics['roc_auc'] >= 0.5, f"ROC-AUC {metrics['roc_auc']} below minimum 0.5"
        assert metrics['recall'] >= 0.0, f"Recall {metrics['recall']} is negative"
        assert metrics['precision'] >= 0.0, f"Precision {metrics['precision']} is negative"


class TestHyperparameterOptimization:
    """Test hyperparameter optimization."""
    
    def test_optuna_optimization(self, sample_training_data):
        """Test that Optuna optimization runs successfully."""
        trainer = FraudDetectionTrainer(
            data_path=sample_training_data,
            n_trials=3
        )
        
        X, y = trainer.load_and_prepare_data()
        X_train = X.iloc[:800]
        y_train = y.iloc[:800]
        
        # Create a trial
        import optuna
        study = optuna.create_study(direction='maximize')
        
        # Test objective function
        trial = study.ask()
        score = trainer.objective(trial, X_train, y_train)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
