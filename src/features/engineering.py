"""
Feature Engineering Pipeline
============================

Centralized feature engineering to ensure consistency between training and inference.
Prevents feature drift by using the same transformations in both pipelines.
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Centralized feature engineering pipeline for fraud detection.
    
    This class ensures that the same feature transformations are applied
    during both training and inference, preventing feature drift.
    
    Features:
    - Temporal features (hour, day, weekend, month)
    - Amount features (log transform, rounding)
    - Velocity features (transaction counts, time gaps)
    - Device features (device counts, new device flags)
    - Location features (location counts, changes)
    - Risk features (merchant risk levels)
    
    Example:
        >>> engineer = FeatureEngineer()
        >>> df_features = engineer.transform(df_transactions)
    """
    
    def __init__(self, validate_schema: bool = True):
        """
        Initialize the feature engineer.
        
        Args:
            validate_schema: Whether to validate input/output schemas with Pydantic
        """
        self.validate_schema = validate_schema
        self.categorical_features = [
            "source", "device_os", "browser", "merchant_category",
            "is_international", "country_code", "merchant_risk_level",
            "device_match", "hour_of_day", "day_of_week",
            "is_weekend", "month"
        ]
        self.merchant_risk_mapping = {
            'Grocery': 1, 'Restaurants': 1, 'Utilities': 1,
            'Clothing': 2, 'Gas': 2, 'Health': 2,
            'Travel': 3, 'Entertainment': 3,
            'Electronics': 4, 'Online Services': 4,
            'Gambling': 5, 'Jewelry': 5, 'Gift Cards': 5, 'Money Transfer': 5
        }
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature transformations to the input DataFrame.
        
        Args:
            df: Raw transaction DataFrame
            
        Returns:
            DataFrame with engineered features
            
        Raises:
            ValueError: If required columns are missing
        """
        logger.info(f"Engineering features for {len(df):,} transactions")
        
        # Validate required columns
        self._validate_input(df)
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Apply transformations
        df = self._temporal_features(df)
        df = self._amount_features(df)
        df = self._velocity_features(df)
        df = self._device_features(df)
        df = self._location_features(df)
        df = self._risk_features(df)
        
        logger.info(f"Feature engineering complete: {len(df.columns)} total columns")
        return df
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate that required columns are present."""
        required_columns = ['amount', 'timestamp']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer temporal features from timestamp.
        
        Features:
        - hour_of_day: Hour of transaction (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - is_weekend: Binary flag for weekend
        - month: Month of transaction (1-12)
        - is_month_start: Binary flag for start of month (1st, 2nd)
        - is_month_end: Binary flag for end of month (29th, 30th, 31st)
        - is_late_night: Binary flag for late night hours (0-4 AM)
        """
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found, skipping temporal features")
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['timestamp'].dt.month
        df['is_month_start'] = df['timestamp'].dt.day.isin([1, 2]).astype(int)
        df['is_month_end'] = df['timestamp'].dt.day.isin([29, 30, 31]).astype(int)
        
        # Derived risk features
        df['is_late_night'] = df['hour_of_day'].isin([0, 1, 2, 3, 4]).astype(int)
        
        logger.debug("Temporal features engineered")
        return df
    
    def _amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer amount-based features.
        
        Features:
        - amount_log: Log transform of amount (log1p for stability)
        - amount_rounded: Binary flag if amount is a round number
        """
        if 'amount' not in df.columns:
            logger.warning("No amount column found, skipping amount features")
            return df
        
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_rounded'] = (df['amount'] % 1 == 0).astype(int)
        
        logger.debug("Amount features engineered")
        return df
    
    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer velocity features (transaction patterns over time).
        
        Features:
        - sender_txn_count: Cumulative transaction count per sender
        - time_since_last_txn: Seconds since last transaction by sender
        - is_rapid_txn: Binary flag for rapid transactions (<60 seconds)
        
        Note: Requires data to be sorted by timestamp for accurate calculation.
        """
        if 'sender_id' not in df.columns or 'timestamp' not in df.columns:
            logger.warning("Missing sender_id or timestamp, skipping velocity features")
            return df
        
        # Sort by timestamp for proper velocity calculation
        df = df.sort_values('timestamp')
        
        # Sender transaction count (cumulative)
        df['sender_txn_count'] = df.groupby('sender_id').cumcount() + 1
        
        # Time since last transaction (in seconds)
        df['time_since_last_txn'] = (
            df.groupby('sender_id')['timestamp']
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )
        
        # Flag for rapid transactions (<60 seconds)
        df['is_rapid_txn'] = (df['time_since_last_txn'] < 60).astype(int)
        
        logger.debug("Velocity features engineered")
        return df
    
    def _device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer device-based features.
        
        Features:
        - user_device_count: Number of unique devices per user
        - is_new_device: Binary flag for new device (only 1 device seen)
        """
        if 'device_fingerprint' not in df.columns or 'sender_id' not in df.columns:
            logger.warning("Missing device_fingerprint or sender_id, skipping device features")
            return df
        
        # Count unique devices per user
        device_counts = df.groupby('sender_id')['device_fingerprint'].transform('nunique')
        df['user_device_count'] = device_counts
        df['is_new_device'] = (device_counts == 1).astype(int)
        
        logger.debug("Device features engineered")
        return df
    
    def _location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer location-based features.
        
        Features:
        - user_location_count: Number of unique locations per user
        - is_location_change: Binary flag for location change (>1 location)
        """
        if 'zip_code' not in df.columns or 'sender_id' not in df.columns:
            logger.warning("Missing zip_code or sender_id, skipping location features")
            return df
        
        # Count unique locations per user
        location_counts = df.groupby('sender_id')['zip_code'].transform('nunique')
        df['user_location_count'] = location_counts
        df['is_location_change'] = (location_counts > 1).astype(int)
        
        logger.debug("Location features engineered")
        return df
    
    def _risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer risk-based features.
        
        Features:
        - merchant_risk_level: Risk level of merchant category (1-5)
        - is_high_risk_merchant: Binary flag for high-risk merchants (>=4)
        """
        if 'merchant_category' not in df.columns:
            logger.warning("Missing merchant_category, skipping risk features")
            return df
        
        # Map merchant categories to risk levels
        df['merchant_risk_level'] = df['merchant_category'].map(
            self.merchant_risk_mapping
        ).fillna(3)  # Default to medium risk
        
        # Binary flag for high-risk merchants
        df['is_high_risk_merchant'] = (df['merchant_risk_level'] >= 4).astype(int)
        
        logger.debug("Risk features engineered")
        return df
    
    def prepare_for_model(
        self,
        df: pd.DataFrame,
        drop_cols: Optional[list] = None,
        encode_categoricals: bool = True
    ) -> pd.DataFrame:
        """
        Prepare features for model input by dropping unnecessary columns
        and encoding categorical variables.
        
        Args:
            df: DataFrame with engineered features
            drop_cols: Additional columns to drop
            encode_categoricals: Whether to one-hot encode categorical features
            
        Returns:
            DataFrame ready for model input
        """
        # Default columns to drop
        default_drop_cols = [
            'fraud_bool', 'is_fraud', 'pattern', 'transaction_id',
            'sender_id', 'receiver_id', 'timestamp', 'zip_code',
            'ip_address', 'session_id', 'device_fingerprint',
            'transaction_date'
        ]
        
        if drop_cols:
            default_drop_cols.extend(drop_cols)
        
        # Drop columns
        df = df.drop(
            columns=[col for col in default_drop_cols if col in df.columns],
            errors='ignore'
        )
        
        # One-hot encode categorical features
        if encode_categoricals:
            categorical_features_present = [
                col for col in self.categorical_features if col in df.columns
            ]
            if categorical_features_present:
                df = pd.get_dummies(
                    df,
                    columns=categorical_features_present,
                    prefix=categorical_features_present
                )
                logger.debug(f"Encoded {len(categorical_features_present)} categorical features")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> list:
        """
        Get list of feature names after transformation.
        
        Args:
            df: Transformed DataFrame
            
        Returns:
            List of feature column names
        """
        # Prepare for model to get final feature names
        df_prepared = self.prepare_for_model(df.copy())
        return df_prepared.columns.tolist()
