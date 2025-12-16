"""
Unit Tests for Feature Engineering
==================================

Tests for the FeatureEngineer class to ensure correct feature transformations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.engineering import FeatureEngineer


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    base_time = datetime(2025, 12, 16, 14, 30, 0)
    
    return pd.DataFrame({
        'transaction_id': ['tx_001', 'tx_002', 'tx_003', 'tx_004'],
        'sender_id': ['user_1', 'user_1', 'user_2', 'user_1'],
        'receiver_id': ['merchant_1', 'merchant_2', 'merchant_1', 'merchant_3'],
        'amount': [100.0, 250.50, 1000.0, 50.0],
        'timestamp': [
            base_time,
            base_time + timedelta(minutes=30),
            base_time + timedelta(hours=1),
            base_time + timedelta(seconds=30)
        ],
        'source': ['online', 'atm', 'online', 'mobile'],
        'device_os': ['iOS', 'Android', 'iOS', 'iOS'],
        'browser': ['Safari', None, 'Chrome', 'Safari'],
        'merchant_category': ['Restaurants', 'Grocery', 'Electronics', 'Gambling'],
        'is_international': [False, False, True, False],
        'country_code': ['US', 'US', 'UK', 'US'],
        'zip_code': ['94102', '94102', '10001', '94105'],
        'ip_address': ['192.168.1.1', '192.168.1.1', '10.0.0.1', '192.168.1.2'],
        'session_id': ['sess_1', 'sess_2', 'sess_3', 'sess_1'],
        'device_fingerprint': ['fp_1', 'fp_1', 'fp_2', 'fp_1']
    })


@pytest.fixture
def feature_engineer():
    """Create FeatureEngineer instance."""
    return FeatureEngineer(validate_schema=False)


class TestTemporalFeatures:
    """Test temporal feature engineering."""
    
    def test_hour_of_day(self, feature_engineer, sample_transactions):
        """Test hour_of_day extraction."""
        df = feature_engineer._temporal_features(sample_transactions.copy())
        assert 'hour_of_day' in df.columns
        assert df['hour_of_day'].iloc[0] == 14
        assert df['hour_of_day'].min() >= 0
        assert df['hour_of_day'].max() <= 23
    
    def test_day_of_week(self, feature_engineer, sample_transactions):
        """Test day_of_week extraction."""
        df = feature_engineer._temporal_features(sample_transactions.copy())
        assert 'day_of_week' in df.columns
        assert df['day_of_week'].min() >= 0
        assert df['day_of_week'].max() <= 6
    
    def test_is_weekend(self, feature_engineer, sample_transactions):
        """Test weekend flag."""
        df = feature_engineer._temporal_features(sample_transactions.copy())
        assert 'is_weekend' in df.columns
        assert df['is_weekend'].isin([0, 1]).all()
    
    def test_is_late_night(self, feature_engineer, sample_transactions):
        """Test late night flag."""
        # Create late night transaction
        late_night_df = sample_transactions.copy()
        late_night_df.loc[0, 'timestamp'] = datetime(2025, 12, 16, 2, 0, 0)
        
        df = feature_engineer._temporal_features(late_night_df)
        assert 'is_late_night' in df.columns
        assert df['is_late_night'].iloc[0] == 1
    
    def test_month_features(self, feature_engineer, sample_transactions):
        """Test month-related features."""
        df = feature_engineer._temporal_features(sample_transactions.copy())
        assert 'month' in df.columns
        assert 'is_month_start' in df.columns
        assert 'is_month_end' in df.columns
        assert df['month'].min() >= 1
        assert df['month'].max() <= 12


class TestAmountFeatures:
    """Test amount feature engineering."""
    
    def test_amount_log(self, feature_engineer, sample_transactions):
        """Test log transform of amount."""
        df = feature_engineer._amount_features(sample_transactions.copy())
        assert 'amount_log' in df.columns
        # log1p(100) ≈ 4.615
        assert np.isclose(df['amount_log'].iloc[0], np.log1p(100.0), rtol=0.01)
    
    def test_amount_rounded(self, feature_engineer, sample_transactions):
        """Test rounded amount flag."""
        df = feature_engineer._amount_features(sample_transactions.copy())
        assert 'amount_rounded' in df.columns
        assert df['amount_rounded'].iloc[0] == 1  # 100.0 is rounded
        assert df['amount_rounded'].iloc[1] == 0  # 250.50 is not rounded


class TestVelocityFeatures:
    """Test velocity feature engineering."""
    
    def test_sender_txn_count(self, feature_engineer, sample_transactions):
        """Test cumulative transaction count."""
        df = feature_engineer._velocity_features(sample_transactions.copy())
        assert 'sender_txn_count' in df.columns
        # user_1 should have counts 1, 2, 3
        user_1_counts = df[df['sender_id'] == 'user_1']['sender_txn_count'].values
        assert user_1_counts[0] == 1
        assert user_1_counts[1] == 2
    
    def test_time_since_last_txn(self, feature_engineer, sample_transactions):
        """Test time since last transaction."""
        df = feature_engineer._velocity_features(sample_transactions.copy())
        assert 'time_since_last_txn' in df.columns
        assert df['time_since_last_txn'].iloc[0] == 0  # First transaction
        assert df['time_since_last_txn'].iloc[1] > 0  # Subsequent transaction
    
    def test_is_rapid_txn(self, feature_engineer, sample_transactions):
        """Test rapid transaction flag."""
        df = feature_engineer._velocity_features(sample_transactions.copy())
        assert 'is_rapid_txn' in df.columns
        # Transaction 4 is 30 seconds after transaction 1 (same user)
        assert df['is_rapid_txn'].iloc[3] == 1


class TestDeviceFeatures:
    """Test device feature engineering."""
    
    def test_user_device_count(self, feature_engineer, sample_transactions):
        """Test unique device count per user."""
        df = feature_engineer._device_features(sample_transactions.copy())
        assert 'user_device_count' in df.columns
        # user_1 uses only fp_1
        assert df[df['sender_id'] == 'user_1']['user_device_count'].iloc[0] == 1
    
    def test_is_new_device(self, feature_engineer, sample_transactions):
        """Test new device flag."""
        df = feature_engineer._device_features(sample_transactions.copy())
        assert 'is_new_device' in df.columns
        assert df['is_new_device'].isin([0, 1]).all()


class TestLocationFeatures:
    """Test location feature engineering."""
    
    def test_user_location_count(self, feature_engineer, sample_transactions):
        """Test unique location count per user."""
        df = feature_engineer._location_features(sample_transactions.copy())
        assert 'user_location_count' in df.columns
        # user_1 uses 2 different zip codes
        assert df[df['sender_id'] == 'user_1']['user_location_count'].iloc[0] == 2
    
    def test_is_location_change(self, feature_engineer, sample_transactions):
        """Test location change flag."""
        df = feature_engineer._location_features(sample_transactions.copy())
        assert 'is_location_change' in df.columns
        # user_1 has location change
        assert df[df['sender_id'] == 'user_1']['is_location_change'].iloc[0] == 1


class TestRiskFeatures:
    """Test risk feature engineering."""
    
    def test_merchant_risk_level(self, feature_engineer, sample_transactions):
        """Test merchant risk level mapping."""
        df = feature_engineer._risk_features(sample_transactions.copy())
        assert 'merchant_risk_level' in df.columns
        # Restaurants = 1, Gambling = 5
        assert df[df['merchant_category'] == 'Restaurants']['merchant_risk_level'].iloc[0] == 1
        assert df[df['merchant_category'] == 'Gambling']['merchant_risk_level'].iloc[0] == 5
    
    def test_is_high_risk_merchant(self, feature_engineer, sample_transactions):
        """Test high-risk merchant flag."""
        df = feature_engineer._risk_features(sample_transactions.copy())
        assert 'is_high_risk_merchant' in df.columns
        # Gambling (risk=5) should be flagged
        assert df[df['merchant_category'] == 'Gambling']['is_high_risk_merchant'].iloc[0] == 1
        # Restaurants (risk=1) should not be flagged
        assert df[df['merchant_category'] == 'Restaurants']['is_high_risk_merchant'].iloc[0] == 0


class TestFullPipeline:
    """Test complete feature engineering pipeline."""
    
    def test_transform(self, feature_engineer, sample_transactions):
        """Test full transformation pipeline."""
        df = feature_engineer.transform(sample_transactions.copy())
        
        # Check all feature groups are present
        assert 'hour_of_day' in df.columns  # Temporal
        assert 'amount_log' in df.columns  # Amount
        assert 'sender_txn_count' in df.columns  # Velocity
        assert 'user_device_count' in df.columns  # Device
        assert 'user_location_count' in df.columns  # Location
        assert 'merchant_risk_level' in df.columns  # Risk
    
    def test_prepare_for_model(self, feature_engineer, sample_transactions):
        """Test model preparation (dropping columns, encoding)."""
        df = feature_engineer.transform(sample_transactions.copy())
        df_prepared = feature_engineer.prepare_for_model(df)
        
        # Check that ID columns are dropped
        assert 'transaction_id' not in df_prepared.columns
        assert 'sender_id' not in df_prepared.columns
        assert 'receiver_id' not in df_prepared.columns
        
        # Check that categorical features are encoded
        assert any('source_' in col for col in df_prepared.columns)
        assert any('merchant_category_' in col for col in df_prepared.columns)
    
    def test_feature_consistency(self, feature_engineer, sample_transactions):
        """Test that same input produces same output."""
        df1 = feature_engineer.transform(sample_transactions.copy())
        df2 = feature_engineer.transform(sample_transactions.copy())
        
        # Check that results are identical
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_missing_columns_handling(self, feature_engineer):
        """Test handling of missing optional columns."""
        minimal_df = pd.DataFrame({
            'amount': [100.0],
            'timestamp': [datetime.now()]
        })
        
        # Should not raise error
        df = feature_engineer.transform(minimal_df)
        assert 'amount_log' in df.columns
        assert 'hour_of_day' in df.columns


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self, feature_engineer):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['amount', 'timestamp'])
        df = feature_engineer.transform(empty_df)
        assert len(df) == 0
    
    def test_missing_required_columns(self, feature_engineer):
        """Test error when required columns are missing."""
        invalid_df = pd.DataFrame({'foo': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            feature_engineer.transform(invalid_df)
    
    def test_null_values(self, feature_engineer, sample_transactions):
        """Test handling of null values."""
        df_with_nulls = sample_transactions.copy()
        df_with_nulls.loc[0, 'device_fingerprint'] = None
        
        # Should not raise error
        df = feature_engineer.transform(df_with_nulls)
        assert df is not None
