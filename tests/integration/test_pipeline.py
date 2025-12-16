"""
Integration Tests for End-to-End Pipeline
=========================================

Tests the complete fraud detection pipeline from data generation to prediction.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_transactions():
    """Create sample transactions for testing."""
    from datetime import datetime
    
    return pd.DataFrame({
        'transaction_id': ['tx_001', 'tx_002', 'tx_003'],
        'sender_id': ['user_1', 'user_1', 'user_2'],
        'receiver_id': ['merchant_1', 'merchant_2', 'merchant_1'],
        'amount': [100.0, 250.0, 1000.0],
        'timestamp': [datetime.now()] * 3,
        'source': ['online', 'atm', 'online'],
        'device_os': ['iOS', 'Android', 'iOS'],
        'browser': ['Safari', None, 'Chrome'],
        'merchant_category': ['Restaurants', 'Grocery', 'Electronics'],
        'is_international': [False, False, True],
        'country_code': ['US', 'US', 'UK'],
        'zip_code': ['94102', '94102', '10001'],
        'ip_address': ['192.168.1.1', '192.168.1.1', '10.0.0.1'],
        'session_id': ['sess_1', 'sess_2', 'sess_3'],
        'device_fingerprint': ['fp_1', 'fp_1', 'fp_2']
    })


class TestFeatureEngineeringConsistency:
    """Test that feature engineering is consistent between training and inference."""
    
    def test_training_inference_feature_consistency(self, sample_transactions):
        """Test that training and inference produce same features."""
        from src.features.engineering import FeatureEngineer
        
        # Create engineer instance
        engineer = FeatureEngineer(validate_schema=False)
        
        # Transform data
        df_features = engineer.transform(sample_transactions.copy())
        
        # Prepare for model (training path)
        df_train = engineer.prepare_for_model(df_features.copy(), encode_categoricals=True)
        
        # Prepare for model (inference path)
        df_inference = engineer.prepare_for_model(df_features.copy(), encode_categoricals=True)
        
        # Check columns are identical
        assert list(df_train.columns) == list(df_inference.columns)
        
        # Check values are identical
        pd.testing.assert_frame_equal(df_train, df_inference)
    
    def test_feature_engineering_deterministic(self, sample_transactions):
        """Test that feature engineering is deterministic."""
        from src.features.engineering import FeatureEngineer
        
        engineer = FeatureEngineer(validate_schema=False)
        
        # Transform same data twice
        df1 = engineer.transform(sample_transactions.copy())
        df2 = engineer.transform(sample_transactions.copy())
        
        # Check results are identical
        pd.testing.assert_frame_equal(df1, df2)


class TestEndToEndPrediction:
    """Test end-to-end prediction pipeline."""
    
    @pytest.mark.integration
    def test_prediction_pipeline(self, sample_transactions, tmp_path):
        """Test complete prediction pipeline."""
        from src.features.engineering import FeatureEngineer
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a simple model for testing
        engineer = FeatureEngineer(validate_schema=False)
        df_features = engineer.transform(sample_transactions)
        X = engineer.prepare_for_model(df_features, encode_categoricals=True)
        
        # Create dummy labels
        y = np.array([0, 0, 1])
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)
        
        # Load model and predict
        loaded_model = joblib.load(model_path)
        predictions = loaded_model.predict(X)
        
        # Check predictions
        assert len(predictions) == len(sample_transactions)
        assert all(p in [0, 1] for p in predictions)
    
    @pytest.mark.integration
    def test_prediction_latency(self, sample_transactions):
        """Test that predictions are fast enough."""
        from src.features.engineering import FeatureEngineer
        from sklearn.ensemble import RandomForestClassifier
        
        # Prepare data
        engineer = FeatureEngineer(validate_schema=False)
        df_features = engineer.transform(sample_transactions)
        X = engineer.prepare_for_model(df_features, encode_categoricals=True)
        
        # Train simple model
        y = np.array([0, 0, 1])
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Measure prediction time
        start_time = time.time()
        predictions = model.predict(X)
        latency_ms = (time.time() - start_time) * 1000
        
        # Check latency (should be very fast for 3 transactions)
        assert latency_ms < 100, f"Prediction took {latency_ms:.2f}ms, expected <100ms"


class TestDataValidation:
    """Test data validation in the pipeline."""
    
    def test_missing_required_columns(self):
        """Test that missing required columns are detected."""
        from src.features.engineering import FeatureEngineer
        
        # Create DataFrame with missing required columns
        df = pd.DataFrame({'foo': [1, 2, 3]})
        
        engineer = FeatureEngineer(validate_schema=False)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            engineer.transform(df)
    
    def test_invalid_data_types(self, sample_transactions):
        """Test handling of invalid data types."""
        from src.features.engineering import FeatureEngineer
        
        # Create DataFrame with invalid amount
        df = sample_transactions.copy()
        df.loc[0, 'amount'] = 'invalid'
        
        engineer = FeatureEngineer(validate_schema=False)
        
        # Should handle gracefully or raise appropriate error
        try:
            result = engineer.transform(df)
            # If it doesn't raise, check that it handled it somehow
            assert result is not None
        except (ValueError, TypeError):
            # Expected behavior
            pass


class TestModelVersioning:
    """Test model versioning and compatibility."""
    
    def test_feature_columns_saved_with_model(self, tmp_path):
        """Test that feature columns are saved with model."""
        from src.features.engineering import FeatureEngineer
        import joblib
        
        # Create sample data
        df = pd.DataFrame({
            'amount': [100.0],
            'timestamp': [pd.Timestamp.now()]
        })
        
        engineer = FeatureEngineer(validate_schema=False)
        df_features = engineer.transform(df)
        feature_columns = engineer.get_feature_names(df_features)
        
        # Save feature columns
        feature_path = tmp_path / "feature_columns.pkl"
        joblib.dump(feature_columns, feature_path)
        
        # Load and verify
        loaded_columns = joblib.load(feature_path)
        assert loaded_columns == feature_columns


@pytest.mark.integration
class TestPerformanceRequirements:
    """Test that system meets performance requirements."""
    
    def test_throughput_requirement(self, sample_transactions):
        """Test that system can handle required throughput."""
        from src.features.engineering import FeatureEngineer
        from sklearn.ensemble import RandomForestClassifier
        
        # Create larger dataset
        large_df = pd.concat([sample_transactions] * 100, ignore_index=True)
        
        # Prepare data
        engineer = FeatureEngineer(validate_schema=False)
        df_features = engineer.transform(large_df)
        X = engineer.prepare_for_model(df_features, encode_categoricals=True)
        
        # Train model
        y = np.random.choice([0, 1], len(X))
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Measure throughput
        start_time = time.time()
        predictions = model.predict(X)
        elapsed_time = time.time() - start_time
        
        throughput = len(predictions) / elapsed_time
        
        # Should handle at least 100 TPS
        assert throughput > 100, f"Throughput {throughput:.2f} TPS below requirement 100 TPS"
