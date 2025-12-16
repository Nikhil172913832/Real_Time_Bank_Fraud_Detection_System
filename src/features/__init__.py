"""
Feature Engineering Package
===========================

Centralized feature engineering for fraud detection to ensure consistency
between training and inference pipelines.
"""

from src.features.engineering import FeatureEngineer
from src.features.schemas import TransactionInput, FeatureOutput

__all__ = ["FeatureEngineer", "TransactionInput", "FeatureOutput"]
