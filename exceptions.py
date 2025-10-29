"""
Custom Exception Classes for Fraud Detection System
===================================================

This module defines custom exceptions for better error handling and debugging
throughout the fraud detection system.
"""

from typing import List, Optional, Any


class FraudDetectionError(Exception):
    """Base exception for fraud detection system."""
    pass


class ModelError(FraudDetectionError):
    """Base exception for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when model cannot be loaded."""
    
    def __init__(self, model_path: str, reason: str):
        self.model_path = model_path
        self.reason = reason
        super().__init__(f"Failed to load model from {model_path}: {reason}")


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""
    
    def __init__(self, reason: str, transaction_id: Optional[str] = None):
        self.transaction_id = transaction_id
        self.reason = reason
        msg = f"Model prediction failed: {reason}"
        if transaction_id:
            msg += f" (transaction: {transaction_id})"
        super().__init__(msg)


class DatabaseError(FraudDetectionError):
    """Base exception for database-related errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    def __init__(self, db_url: str, reason: str):
        self.db_url = db_url
        self.reason = reason
        # Don't expose credentials in error message
        safe_url = db_url.split('@')[-1] if '@' in db_url else db_url
        super().__init__(f"Database connection failed to {safe_url}: {reason}")


class DatabaseQueryError(DatabaseError):
    """Raised when database query fails."""
    
    def __init__(self, query: str, reason: str):
        self.query = query
        self.reason = reason
        super().__init__(f"Database query failed: {reason}")


class KafkaError(FraudDetectionError):
    """Base exception for Kafka-related errors."""
    pass


class KafkaConnectionError(KafkaError):
    """Raised when Kafka connection fails."""
    
    def __init__(self, bootstrap_servers: str, reason: str):
        self.bootstrap_servers = bootstrap_servers
        self.reason = reason
        super().__init__(f"Kafka connection failed to {bootstrap_servers}: {reason}")


class KafkaProducerError(KafkaError):
    """Raised when Kafka message production fails."""
    
    def __init__(self, topic: str, reason: str):
        self.topic = topic
        self.reason = reason
        super().__init__(f"Failed to produce message to topic {topic}: {reason}")


class KafkaConsumerError(KafkaError):
    """Raised when Kafka message consumption fails."""
    
    def __init__(self, topic: str, reason: str):
        self.topic = topic
        self.reason = reason
        super().__init__(f"Failed to consume message from topic {topic}: {reason}")


class ConfigurationError(FraudDetectionError):
    """Base exception for configuration-related errors."""
    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, config_key: str):
        self.config_key = config_key
        super().__init__(f"Required configuration '{config_key}' is missing")


class InvalidConfigError(ConfigurationError):
    """Raised when configuration value is invalid."""
    
    def __init__(self, config_key: str, value: Any, reason: str):
        self.config_key = config_key
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid configuration for '{config_key}': {reason}")


class ValidationError(FraudDetectionError):
    """Base exception for validation errors."""
    pass


class FeatureValidationError(ValidationError):
    """Raised when feature validation fails."""
    
    def __init__(self, missing_features: Optional[List[str]] = None, extra_features: Optional[List[str]] = None):
        self.missing_features = missing_features or []
        self.extra_features = extra_features or []
        
        msg_parts = []
        if missing_features:
            msg_parts.append(f"Missing features: {', '.join(missing_features)}")
        if extra_features:
            msg_parts.append(f"Extra features: {', '.join(extra_features)}")
        
        super().__init__("; ".join(msg_parts))


class TransactionValidationError(ValidationError):
    """Raised when transaction data validation fails."""
    
    def __init__(self, field: str, reason: str, transaction_id: Optional[str] = None):
        self.field = field
        self.reason = reason
        self.transaction_id = transaction_id
        
        msg = f"Invalid transaction field '{field}': {reason}"
        if transaction_id:
            msg += f" (transaction: {transaction_id})"
        super().__init__(msg)


class AlertError(FraudDetectionError):
    """Base exception for alerting errors."""
    pass


class EmailAlertError(AlertError):
    """Raised when email alert sending fails."""
    
    def __init__(self, recipient: str, reason: str):
        self.recipient = recipient
        self.reason = reason
        super().__init__(f"Failed to send email alert to {recipient}: {reason}")


class ResourceError(FraudDetectionError):
    """Base exception for resource-related errors."""
    pass


class InsufficientResourcesError(ResourceError):
    """Raised when system resources are insufficient."""
    
    def __init__(self, resource_type: str, required: str, available: str):
        self.resource_type = resource_type
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient {resource_type}: required {required}, available {available}"
        )


class CircuitBreakerError(FraudDetectionError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        super().__init__(f"Circuit breaker open for service: {service_name}")


class DataGenerationError(FraudDetectionError):
    """Raised when synthetic data generation fails."""
    
    def __init__(self, pattern: str, reason: str):
        self.pattern = pattern
        self.reason = reason
        super().__init__(f"Data generation failed for pattern '{pattern}': {reason}")
