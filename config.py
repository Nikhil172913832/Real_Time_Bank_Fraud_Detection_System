# Configuration Management for Fraud Detection System
# ==================================================
# Centralized configuration with environment variable support

import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration class."""
    
    # Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'fraud_detection')
    DB_USER = os.getenv('DB_USER', 'fraud_user')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'fraud_password_secure_123')
    
    @property
    def DATABASE_URL(self) -> str:
        """Construct PostgreSQL connection string."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(',')
    KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'transactions')
    KAFKA_CONSUMER_GROUP = os.getenv('KAFKA_CONSUMER_GROUP', 'fraud-detector')
    KAFKA_AUTO_OFFSET_RESET = os.getenv('KAFKA_AUTO_OFFSET_RESET', 'latest')
    KAFKA_ENABLE_AUTO_COMMIT = os.getenv('KAFKA_ENABLE_AUTO_COMMIT', 'true').lower() == 'true'
    KAFKA_MAX_POLL_RECORDS = int(os.getenv('KAFKA_MAX_POLL_RECORDS', '500'))
    KAFKA_SESSION_TIMEOUT_MS = int(os.getenv('KAFKA_SESSION_TIMEOUT_MS', '30000'))
    
    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/xgb_final.pkl')
    FEATURE_COLUMNS_PATH = os.getenv('FEATURE_COLUMNS_PATH', 'models/feature_columns.pkl')
    FRAUD_THRESHOLD = float(os.getenv('FRAUD_THRESHOLD', '0.2'))
    
    # Inference Configuration
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))
    BATCH_TIMEOUT = int(os.getenv('BATCH_TIMEOUT', '2'))  # seconds
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '5000'))
    API_DEBUG = os.getenv('API_DEBUG', 'false').lower() == 'true'
    API_WORKERS = int(os.getenv('API_WORKERS', '4'))
    
    # Email Alert Configuration
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '465'))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
    ALERT_EMAIL_FROM = os.getenv('ALERT_EMAIL_FROM', SMTP_USERNAME)
    ALERT_EMAIL_TO = os.getenv('ALERT_EMAIL_TO', '')
    ENABLE_EMAIL_ALERTS = os.getenv('ENABLE_EMAIL_ALERTS', 'false').lower() == 'true'
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')
    
    # Performance Targets (for monitoring)
    TARGET_THROUGHPUT_TPS = int(os.getenv('TARGET_THROUGHPUT_TPS', '1000'))
    TARGET_LATENCY_MS = int(os.getenv('TARGET_LATENCY_MS', '150'))
    TARGET_ROC_AUC = float(os.getenv('TARGET_ROC_AUC', '0.98'))
    TARGET_RECALL = float(os.getenv('TARGET_RECALL', '0.80'))
    
    # Feature Flags
    ENABLE_PROMETHEUS_METRICS = os.getenv('ENABLE_PROMETHEUS_METRICS', 'true').lower() == 'true'
    ENABLE_MODEL_MONITORING = os.getenv('ENABLE_MODEL_MONITORING', 'true').lower() == 'true'
    ENABLE_PERFORMANCE_TRACKING = os.getenv('ENABLE_PERFORMANCE_TRACKING', 'true').lower() == 'true'
    
    @classmethod
    def to_dict(cls) -> Dict[str, any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and key.isupper() and 'PASSWORD' not in key
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration values."""
        required_vars = [
            'DB_HOST', 'DB_NAME', 'DB_USER',
            'KAFKA_BOOTSTRAP_SERVERS', 'KAFKA_TOPIC'
        ]
        
        missing = []
        for var in required_vars:
            if not getattr(cls, var):
                missing.append(var)
        
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        
        return True


# Create global config instance
config = Config()

# Validate on import
try:
    config.validate()
except ValueError as e:
    print(f"Warning: Configuration validation failed: {e}")
