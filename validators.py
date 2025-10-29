"""
Data Validation Models for Fraud Detection System
=================================================

This module provides Pydantic models for validating transactions,
API requests, and configuration data to ensure data integrity.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class DeviceOS(str, Enum):
    """Valid device operating systems."""
    ANDROID = "Android"
    IOS = "iOS"
    WINDOWS = "Windows"
    MAC = "Mac"
    LINUX = "Linux"
    UNKNOWN = "Unknown"


class TransactionSource(str, Enum):
    """Valid transaction sources."""
    MOBILE_APP = "MOBILE_APP"
    WEB = "WEB"
    ATM = "ATM"
    POS = "POS"
    UNKNOWN = "UNKNOWN"


class Transaction(BaseModel):
    """
    Transaction data model with validation.
    
    Example:
        >>> transaction = Transaction(
        ...     transaction_id='txn123',
        ...     sender_id='user456',
        ...     receiver_id='user789',
        ...     amount=100.50,
        ...     source='MOBILE_APP',
        ...     device_os='Android'
        ... )
    """
    
    # Required fields
    transaction_id: str = Field(..., min_length=1, max_length=100)
    sender_id: str = Field(..., min_length=1, max_length=100)
    receiver_id: str = Field(..., min_length=1, max_length=100)
    amount: float = Field(..., gt=0, description="Transaction amount must be positive")
    source: TransactionSource = Field(default=TransactionSource.UNKNOWN)
    
    # Optional fields with defaults
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    device_os: Optional[DeviceOS] = Field(default=DeviceOS.UNKNOWN)
    browser: Optional[str] = Field(default=None, max_length=50)
    zip_code: Optional[str] = Field(default=None, max_length=20)
    merchant_category: Optional[str] = Field(default=None, max_length=50)
    ip_address: Optional[str] = Field(default=None, max_length=45)  # IPv6 compatible
    session_id: Optional[str] = Field(default=None, max_length=100)
    account_age_days: Optional[int] = Field(default=0, ge=0)
    is_international: Optional[bool] = Field(default=False)
    country_code: Optional[str] = Field(default=None, max_length=3)
    merchant_risk_level: Optional[int] = Field(default=1, ge=1, le=5)
    device_match: Optional[bool] = Field(default=True)
    
    @validator('amount')
    def validate_amount(cls, v):
        """Ensure amount is reasonable."""
        if v > 1_000_000:
            raise ValueError('Transaction amount exceeds maximum allowed ($1M)')
        if v < 0.01:
            raise ValueError('Transaction amount must be at least $0.01')
        return round(v, 2)
    
    @validator('ip_address')
    def validate_ip(cls, v):
        """Validate IP address format (basic validation)."""
        if v is None:
            return v
        
        # Basic IPv4/IPv6 format check
        import re
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        ipv6_pattern = r'^([0-9a-fA-F]{0,4}:){7}[0-9a-fA-F]{0,4}$'
        
        if not (re.match(ipv4_pattern, v) or re.match(ipv6_pattern, v)):
            raise ValueError(f'Invalid IP address format: {v}')
        
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Ensure timestamp is not in the future."""
        if v and v > datetime.now():
            raise ValueError('Transaction timestamp cannot be in the future')
        return v
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PredictionRequest(BaseModel):
    """
    API prediction request model.
    
    Example:
        >>> request = PredictionRequest(
        ...     transaction_id='txn123',
        ...     features={'amount': 100.0, 'is_international': 1}
        ... )
    """
    
    transaction_id: str = Field(..., min_length=1)
    features: Dict[str, Any] = Field(...)
    
    @validator('features')
    def validate_features(cls, v):
        """Ensure features dict is not empty."""
        if not v:
            raise ValueError('Features dictionary cannot be empty')
        return v


class BatchPredictionRequest(BaseModel):
    """
    API batch prediction request model.
    
    Example:
        >>> request = BatchPredictionRequest(
        ...     transactions=[
        ...         {'transaction_id': 'txn1', 'features': {...}},
        ...         {'transaction_id': 'txn2', 'features': {...}}
        ...     ]
        ... )
    """
    
    transactions: List[PredictionRequest] = Field(..., min_items=1, max_items=1000)
    
    @validator('transactions')
    def validate_batch_size(cls, v):
        """Ensure batch size is reasonable."""
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 transactions')
        return v


class PredictionResponse(BaseModel):
    """
    API prediction response model.
    
    Example:
        >>> response = PredictionResponse(
        ...     transaction_id='txn123',
        ...     is_fraud=True,
        ...     fraud_probability=0.85,
        ...     processing_time_ms=50.5
        ... )
    """
    
    transaction_id: str
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float = Field(..., ge=0.0)
    shap_values: Optional[Dict[str, float]] = None
    model_version: str = Field(default='v1.0')
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FraudAlert(BaseModel):
    """
    Fraud alert model for database storage.
    
    Example:
        >>> alert = FraudAlert(
        ...     transaction_id='txn123',
        ...     user_id='user456',
        ...     amount=500.0,
        ...     fraud_probability=0.85
        ... )
    """
    
    transaction_id: str
    user_id: str
    amount: float = Field(..., gt=0)
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    shap_values: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    detected_at: datetime = Field(default_factory=datetime.now)
    alert_sent: bool = Field(default=False)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelMetrics(BaseModel):
    """
    Model performance metrics model.
    
    Example:
        >>> metrics = ModelMetrics(
        ...     model_version='v1.2.0',
        ...     roc_auc=0.98,
        ...     recall=0.80,
        ...     precision=0.75
        ... )
    """
    
    model_version: str
    roc_auc: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    training_date: datetime = Field(default_factory=datetime.now)
    n_samples: Optional[int] = Field(None, gt=0)
    n_features: Optional[int] = Field(None, gt=0)
    
    @root_validator
    def calculate_f1(cls, values):
        """Calculate F1 score if not provided."""
        if values.get('f1_score') is None:
            precision = values.get('precision', 0)
            recall = values.get('recall', 0)
            
            if precision + recall > 0:
                values['f1_score'] = 2 * (precision * recall) / (precision + recall)
            else:
                values['f1_score'] = 0.0
        
        return values
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """
    Health check response model.
    
    Example:
        >>> health = HealthCheckResponse(
        ...     status='healthy',
        ...     timestamp=datetime.now(),
        ...     components={'database': 'healthy', 'model': 'healthy'}
        ... )
    """
    
    status: str = Field(..., regex='^(healthy|unhealthy|degraded)$')
    timestamp: datetime = Field(default_factory=datetime.now)
    version: Optional[str] = None
    components: Optional[Dict[str, str]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


def validate_transaction_dict(data: Dict[str, Any]) -> Transaction:
    """
    Validate a transaction dictionary and return a Transaction object.
    
    Args:
        data: Dictionary containing transaction data
        
    Returns:
        Validated Transaction object
        
    Raises:
        ValidationError: If validation fails
        
    Example:
        >>> data = {'transaction_id': 'txn123', 'amount': 100.0, ...}
        >>> transaction = validate_transaction_dict(data)
    """
    return Transaction(**data)


def validate_prediction_request(data: Dict[str, Any]) -> PredictionRequest:
    """
    Validate a prediction request dictionary.
    
    Args:
        data: Dictionary containing prediction request data
        
    Returns:
        Validated PredictionRequest object
        
    Raises:
        ValidationError: If validation fails
    """
    return PredictionRequest(**data)
