"""
Feature Schemas for Validation
==============================

Pydantic models for validating transaction inputs and feature outputs.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator


class TransactionInput(BaseModel):
    """Schema for raw transaction input."""
    
    transaction_id: str
    sender_id: str
    receiver_id: str
    amount: float = Field(gt=0, description="Transaction amount must be positive")
    timestamp: datetime
    source: str = Field(..., regex="^(online|atm|pos|mobile)$")
    device_os: Optional[str] = None
    browser: Optional[str] = None
    merchant_category: str
    is_international: bool
    country_code: str = Field(..., min_length=2, max_length=3)
    zip_code: Optional[str] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    device_fingerprint: Optional[str] = None
    
    @validator('amount')
    def amount_must_be_reasonable(cls, v):
        """Validate amount is within reasonable bounds."""
        if v > 1_000_000:
            raise ValueError('Amount exceeds maximum allowed value')
        return v
    
    @validator('timestamp')
    def timestamp_not_future(cls, v):
        """Validate timestamp is not in the future."""
        if v > datetime.now():
            raise ValueError('Timestamp cannot be in the future')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "tx_12345",
                "sender_id": "user_001",
                "receiver_id": "merchant_123",
                "amount": 150.00,
                "timestamp": "2025-12-16T14:30:00",
                "source": "online",
                "device_os": "iOS",
                "browser": "Safari",
                "merchant_category": "Restaurants",
                "is_international": False,
                "country_code": "US",
                "zip_code": "94102",
                "ip_address": "192.168.1.1",
                "session_id": "sess_abc123",
                "device_fingerprint": "fp_xyz789"
            }
        }


class FeatureOutput(BaseModel):
    """Schema for engineered features output."""
    
    # Amount features
    amount: float = Field(gt=0)
    amount_log: float
    amount_rounded: int = Field(ge=0, le=1)
    
    # Temporal features
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: int = Field(ge=0, le=1)
    month: int = Field(ge=1, le=12)
    is_month_start: int = Field(ge=0, le=1)
    is_month_end: int = Field(ge=0, le=1)
    is_late_night: int = Field(ge=0, le=1)
    
    # Velocity features
    sender_txn_count: int = Field(ge=1)
    time_since_last_txn: float = Field(ge=0)
    is_rapid_txn: int = Field(ge=0, le=1)
    
    # Device features
    user_device_count: Optional[int] = Field(ge=1)
    is_new_device: Optional[int] = Field(ge=0, le=1)
    
    # Location features
    user_location_count: Optional[int] = Field(ge=1)
    is_location_change: Optional[int] = Field(ge=0, le=1)
    
    # Risk features
    merchant_risk_level: float = Field(ge=1, le=5)
    is_high_risk_merchant: int = Field(ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "amount": 150.00,
                "amount_log": 5.01,
                "amount_rounded": 1,
                "hour_of_day": 14,
                "day_of_week": 2,
                "is_weekend": 0,
                "month": 12,
                "is_month_start": 0,
                "is_month_end": 0,
                "is_late_night": 0,
                "sender_txn_count": 5,
                "time_since_last_txn": 3600.0,
                "is_rapid_txn": 0,
                "user_device_count": 2,
                "is_new_device": 0,
                "user_location_count": 1,
                "is_location_change": 0,
                "merchant_risk_level": 2.0,
                "is_high_risk_merchant": 0
            }
        }
