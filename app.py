"""
Flask REST API for Real-Time Fraud Detection
============================================

Production-grade API for serving fraud detection predictions.

Endpoints:
    - POST /predict: Single transaction prediction
    - POST /predict/batch: Batch predictions
    - GET /health: Health check
    - GET /metrics: Performance metrics
    - GET /model/info: Model information
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
CORS(app)

# Load model and feature columns
MODEL_PATH = os.getenv('MODEL_PATH', 'models/xgb_final.pkl')
FEATURE_COLUMNS_PATH = os.getenv('FEATURE_COLUMNS_PATH', 'models/feature_columns.pkl')
FRAUD_THRESHOLD = float(os.getenv('FRAUD_THRESHOLD', '0.2'))

try:
    model = joblib.load(MODEL_PATH)
    expected_columns = joblib.load(FEATURE_COLUMNS_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    logger.info(f"Expected features: {len(expected_columns)}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    expected_columns = None

# Prometheus metrics
prediction_counter = Counter(
    'fraud_predictions_total',
    'Total number of fraud predictions',
    ['prediction']
)
prediction_latency = Histogram(
    'fraud_prediction_latency_seconds',
    'Fraud prediction latency'
)
api_request_counter = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

# Categorical features for encoding
CATEGORICAL_FEATURES = [
    "source", "device_os", "browser", "merchant_category",
    "is_international", "country_code", "merchant_risk_level",
    "device_match", "hour_of_day", "day_of_week", "is_weekend", "month"
]

# Model metadata
MODEL_VERSION = "v1.2.0"
MODEL_LOAD_TIME = datetime.now().isoformat()


def require_model(f):
    """Decorator to ensure model is loaded."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'The ML model could not be loaded. Please check server logs.'
            }), 503
        return f(*args, **kwargs)
    return decorated_function


def track_request(endpoint: str):
    """Decorator to track API requests."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                status = result[1] if isinstance(result, tuple) else 200
                api_request_counter.labels(
                    endpoint=endpoint,
                    method=request.method,
                    status=status
                ).inc()
                return result
            except Exception as e:
                api_request_counter.labels(
                    endpoint=endpoint,
                    method=request.method,
                    status=500
                ).inc()
                raise e
        return decorated_function
    return decorator


def preprocess_transaction(tx: Dict) -> pd.DataFrame:
    """
    Preprocess a single transaction for prediction.
    
    Args:
        tx: Transaction dictionary
        
    Returns:
        Preprocessed DataFrame ready for model prediction
    """
    df = pd.DataFrame([tx])
    
    # Drop columns not needed for prediction
    drop_cols = [
        'fraud_bool', 'pattern', 'transaction_id', 'sender_id', 'receiver_id',
        'timestamp', 'zip_code', 'ip_address', 'session_id',
        'device_fingerprint', 'transaction_date'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # One-hot encode categorical features
    categorical_features_present = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=categorical_features_present, prefix=categorical_features_present)
    
    # Align to training column structure
    for col in expected_columns:
        if col not in df_encoded:
            df_encoded[col] = 0
    df_encoded = df_encoded[expected_columns]
    
    return df_encoded


@app.route('/health', methods=['GET'])
@track_request('health')
def health_check():
    """
    Health check endpoint.
    
    Returns:
        200: Service is healthy
        503: Service unavailable
    """
    if model is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'model_version': MODEL_VERSION,
        'model_loaded_at': MODEL_LOAD_TIME,
        'uptime_seconds': (datetime.now() - datetime.fromisoformat(MODEL_LOAD_TIME)).total_seconds()
    }), 200


@app.route('/predict', methods=['POST'])
@require_model
@track_request('predict')
def predict():
    """
    Predict fraud probability for a single transaction.
    
    Request Body:
        JSON object with transaction features
        
    Returns:
        200: Prediction successful
        400: Invalid input
        500: Server error
    """
    start_time = time.time()
    
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess
        X = preprocess_transaction(data)
        
        # Predict
        fraud_prob = float(model.predict_proba(X)[0, 1])
        is_fraud = fraud_prob > FRAUD_THRESHOLD
        
        # Track metrics
        prediction_counter.labels(
            prediction='fraud' if is_fraud else 'legitimate'
        ).inc()
        
        latency_ms = (time.time() - start_time) * 1000
        prediction_latency.observe(time.time() - start_time)
        
        # Response
        response = {
            'transaction_id': data.get('transaction_id', 'unknown'),
            'fraud_probability': round(fraud_prob, 4),
            'is_fraud': is_fraud,
            'threshold': FRAUD_THRESHOLD,
            'prediction_time_ms': round(latency_ms, 2),
            'model_version': MODEL_VERSION,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log
        log_level = logging.WARNING if is_fraud else logging.INFO
        logger.log(
            log_level,
            f"Prediction: {response['transaction_id']} | "
            f"Fraud: {is_fraud} | Prob: {fraud_prob:.4f} | "
            f"Latency: {latency_ms:.2f}ms"
        )
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
@require_model
@track_request('predict_batch')
def predict_batch():
    """
    Predict fraud probability for multiple transactions.
    
    Request Body:
        JSON array of transaction objects
        
    Returns:
        200: Predictions successful
        400: Invalid input
        500: Server error
    """
    start_time = time.time()
    
    try:
        # Parse request
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({'error': 'Expected array of transactions'}), 400
        
        if len(data) == 0:
            return jsonify({'error': 'Empty batch'}), 400
        
        # Process batch
        predictions = []
        for tx in data:
            X = preprocess_transaction(tx)
            fraud_prob = float(model.predict_proba(X)[0, 1])
            is_fraud = fraud_prob > FRAUD_THRESHOLD
            
            predictions.append({
                'transaction_id': tx.get('transaction_id', 'unknown'),
                'fraud_probability': round(fraud_prob, 4),
                'is_fraud': is_fraud
            })
            
            # Track metrics
            prediction_counter.labels(
                prediction='fraud' if is_fraud else 'legitimate'
            ).inc()
        
        latency_ms = (time.time() - start_time) * 1000
        avg_latency_ms = latency_ms / len(data)
        
        logger.info(
            f"Batch prediction: {len(data)} transactions | "
            f"Avg latency: {avg_latency_ms:.2f}ms"
        )
        
        return jsonify({
            'predictions': predictions,
            'batch_size': len(data),
            'total_time_ms': round(latency_ms, 2),
            'avg_time_ms': round(avg_latency_ms, 2),
            'model_version': MODEL_VERSION
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Prometheus-formatted metrics
    """
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/model/info', methods=['GET'])
@require_model
@track_request('model_info')
def model_info():
    """
    Get model information and metadata.
    
    Returns:
        Model information and configuration
    """
    try:
        # Try to load metrics if available
        metrics_data = {}
        try:
            import json
            with open('metrics/training_metrics.json', 'r') as f:
                metrics_data = json.load(f)
        except:
            pass
        
        return jsonify({
            'model_version': MODEL_VERSION,
            'model_type': 'XGBoost Classifier',
            'loaded_at': MODEL_LOAD_TIME,
            'model_path': MODEL_PATH,
            'fraud_threshold': FRAUD_THRESHOLD,
            'num_features': len(expected_columns),
            'performance_metrics': {
                'roc_auc': metrics_data.get('roc_auc'),
                'recall': metrics_data.get('recall'),
                'precision': metrics_data.get('precision'),
                'f1_score': metrics_data.get('f1_score')
            } if metrics_data else None
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving model info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """API root endpoint with documentation."""
    return jsonify({
        'service': 'Fraud Detection API',
        'version': MODEL_VERSION,
        'endpoints': {
            'POST /predict': 'Predict single transaction',
            'POST /predict/batch': 'Predict multiple transactions',
            'GET /health': 'Health check',
            'GET /metrics': 'Prometheus metrics',
            'GET /model/info': 'Model information'
        },
        'documentation': 'https://github.com/yourusername/fraud-detection'
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Get configuration from environment
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '5000'))
    debug = os.getenv('API_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Fraud Detection API on {host}:{port}")
    logger.info(f"Model version: {MODEL_VERSION}")
    logger.info(f"Fraud threshold: {FRAUD_THRESHOLD}")
    
    app.run(host=host, port=port, debug=debug)
