"""
Centralized Prometheus Metrics for Fraud Detection System
=========================================================

This module provides centralized Prometheus metrics definitions and
helper functions for consistent monitoring across all components.
"""

from typing import Callable, Any
from functools import wraps
import time

from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from logger import get_logger

logger = get_logger(__name__)


# ========================================
# Prediction Metrics
# ========================================

prediction_total = Counter(
    'fraud_detection_predictions_total',
    'Total number of fraud predictions',
    ['prediction', 'model_version']
)

prediction_latency = Histogram(
    'fraud_detection_prediction_latency_seconds',
    'Fraud prediction latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

prediction_errors = Counter(
    'fraud_detection_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

# ========================================
# Kafka Metrics
# ========================================

kafka_messages_consumed = Counter(
    'fraud_detection_kafka_messages_consumed_total',
    'Total Kafka messages consumed',
    ['topic', 'status']
)

kafka_consumer_lag = Gauge(
    'fraud_detection_kafka_consumer_lag',
    'Current Kafka consumer lag',
    ['topic', 'partition']
)

kafka_batch_size = Histogram(
    'fraud_detection_kafka_batch_size',
    'Size of Kafka message batches',
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

kafka_processing_latency = Histogram(
    'fraud_detection_kafka_processing_latency_seconds',
    'Time to process Kafka message batch',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# ========================================
# Database Metrics
# ========================================

database_queries_total = Counter(
    'fraud_detection_database_queries_total',
    'Total database queries executed',
    ['operation', 'status']
)

database_query_latency = Histogram(
    'fraud_detection_database_query_latency_seconds',
    'Database query latency',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

database_connection_pool_size = Gauge(
    'fraud_detection_database_connection_pool_size',
    'Current database connection pool size'
)

database_connection_pool_available = Gauge(
    'fraud_detection_database_connection_pool_available',
    'Available connections in pool'
)

# ========================================
# API Metrics
# ========================================

api_requests_total = Counter(
    'fraud_detection_api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status_code']
)

api_request_latency = Histogram(
    'fraud_detection_api_request_latency_seconds',
    'API request latency',
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

api_active_requests = Gauge(
    'fraud_detection_api_active_requests',
    'Number of currently active API requests'
)

# ========================================
# Model Performance Metrics
# ========================================

model_accuracy = Gauge(
    'fraud_detection_model_accuracy',
    'Model accuracy score',
    ['model_version']
)

model_roc_auc = Gauge(
    'fraud_detection_model_roc_auc',
    'Model ROC-AUC score',
    ['model_version']
)

model_recall = Gauge(
    'fraud_detection_model_recall',
    'Model recall score',
    ['model_version']
)

model_precision = Gauge(
    'fraud_detection_model_precision',
    'Model precision score',
    ['model_version']
)

# ========================================
# Fraud Detection Metrics
# ========================================

fraud_alerts_total = Counter(
    'fraud_detection_alerts_total',
    'Total fraud alerts generated',
    ['alert_type']
)

fraud_detected_amount = Summary(
    'fraud_detection_detected_amount',
    'Amount of detected fraudulent transactions'
)

fraud_probability_distribution = Histogram(
    'fraud_detection_probability_distribution',
    'Distribution of fraud probabilities',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# ========================================
# System Metrics
# ========================================

system_info = Info(
    'fraud_detection_system',
    'Fraud detection system information'
)

uptime_seconds = Gauge(
    'fraud_detection_uptime_seconds',
    'System uptime in seconds'
)


# ========================================
# Helper Decorators
# ========================================

def track_prediction_time(func: Callable) -> Callable:
    """
    Decorator to track prediction latency.
    
    Example:
        @track_prediction_time
        def predict(data):
            return model.predict(data)
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            latency = time.time() - start_time
            prediction_latency.observe(latency)
            return result
        except Exception as e:
            prediction_errors.labels(error_type=type(e).__name__).inc()
            raise
    return wrapper


def track_api_request(func: Callable) -> Callable:
    """
    Decorator to track API request metrics.
    
    Example:
        @track_api_request
        def predict_endpoint():
            return jsonify({'prediction': 0.8})
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        api_active_requests.inc()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            latency = time.time() - start_time
            api_request_latency.observe(latency)
            return result
        finally:
            api_active_requests.dec()
    
    return wrapper


def track_database_query(operation: str):
    """
    Decorator factory to track database query metrics.
    
    Args:
        operation: Type of database operation (SELECT, INSERT, UPDATE, etc.)
        
    Example:
        @track_database_query('INSERT')
        def insert_fraud_alert(data):
            db.execute("INSERT INTO fraud_alerts ...")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                status = 'error'
                raise
            finally:
                latency = time.time() - start_time
                database_query_latency.observe(latency)
                database_queries_total.labels(
                    operation=operation,
                    status=status
                ).inc()
        
        return wrapper
    return decorator


def record_fraud_detection(is_fraud: bool, probability: float, amount: float, model_version: str = 'v1.0') -> None:
    """
    Record fraud detection metrics.
    
    Args:
        is_fraud: Whether fraud was detected (probability > threshold)
        probability: Fraud probability from model
        amount: Transaction amount
        model_version: Model version identifier
        
    Example:
        >>> record_fraud_detection(
        ...     is_fraud=True,
        ...     probability=0.85,
        ...     amount=500.0,
        ...     model_version='v1.2.0'
        ... )
    """
    # Record prediction
    prediction_label = 'fraud' if is_fraud else 'legitimate'
    prediction_total.labels(
        prediction=prediction_label,
        model_version=model_version
    ).inc()
    
    # Record fraud probability distribution
    fraud_probability_distribution.observe(probability)
    
    # Record fraud alert and amount if fraud detected
    if is_fraud:
        fraud_alerts_total.labels(alert_type='automatic').inc()
        fraud_detected_amount.observe(amount)


def update_model_metrics(
    accuracy: float,
    roc_auc: float,
    recall: float,
    precision: float,
    model_version: str = 'v1.0'
) -> None:
    """
    Update model performance metrics.
    
    Args:
        accuracy: Model accuracy score
        roc_auc: Model ROC-AUC score
        recall: Model recall score
        precision: Model precision score
        model_version: Model version identifier
        
    Example:
        >>> update_model_metrics(
        ...     accuracy=0.97,
        ...     roc_auc=0.98,
        ...     recall=0.80,
        ...     precision=0.75,
        ...     model_version='v1.2.0'
        ... )
    """
    model_accuracy.labels(model_version=model_version).set(accuracy)
    model_roc_auc.labels(model_version=model_version).set(roc_auc)
    model_recall.labels(model_version=model_version).set(recall)
    model_precision.labels(model_version=model_version).set(precision)
    
    logger.info(
        f"Model metrics updated for {model_version}: "
        f"ROC-AUC={roc_auc:.4f}, Recall={recall:.4f}, Precision={precision:.4f}"
    )


def record_kafka_consumption(
    topic: str,
    batch_size: int,
    processing_time: float,
    status: str = 'success'
) -> None:
    """
    Record Kafka consumption metrics.
    
    Args:
        topic: Kafka topic name
        batch_size: Number of messages in batch
        processing_time: Time taken to process batch
        status: Processing status ('success' or 'error')
        
    Example:
        >>> record_kafka_consumption(
        ...     topic='transactions',
        ...     batch_size=100,
        ...     processing_time=0.5,
        ...     status='success'
        ... )
    """
    kafka_messages_consumed.labels(topic=topic, status=status).inc(batch_size)
    kafka_batch_size.observe(batch_size)
    kafka_processing_latency.observe(processing_time)
