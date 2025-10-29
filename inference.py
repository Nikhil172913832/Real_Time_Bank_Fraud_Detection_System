"""
Real-Time Fraud Detection Inference Engine with SHAP Explainability
===================================================================

This module performs real-time fraud detection on Kafka message streams
with model explainability using SHAP values.

Features:
- Kafka consumer with batch processing
- XGBoost model inference
- SHAP value calculation for explainability
- PostgreSQL persistence
- Email alerting for detected fraud
"""

import time
import pandas as pd
import psycopg2
from kafka import KafkaConsumer
import msgpack
import joblib
import logging
import json
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Config ===
BATCH_SIZE = config.BATCH_SIZE
BATCH_TIMEOUT = config.BATCH_TIMEOUT
KAFKA_TOPIC = config.KAFKA_TOPIC
BOOTSTRAP_SERVERS = config.KAFKA_BOOTSTRAP_SERVERS
THRESHOLD = config.FRAUD_THRESHOLD

# === Load model and training-time columns ===
try:
    model = joblib.load(config.MODEL_PATH)
    expected_columns = joblib.load(config.FEATURE_COLUMNS_PATH)
    logger.info(f"Model loaded from {config.MODEL_PATH}")
    logger.info(f"Expected features: {len(expected_columns)}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# === Features used during encoding ===
categorical_features = [
    "source", "device_os", "browser", "merchant_category",
    "is_international", "country_code", "merchant_risk_level",
    "device_match", "hour_of_day", "day_of_week", "is_weekend", "month"
]

# === Setup PostgreSQL connection ===
try:
    conn = psycopg2.connect(config.DATABASE_URL)
    cursor = conn.cursor()
    logger.info("Connected to PostgreSQL database")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    raise

# === Kafka Consumer ===
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_deserializer=lambda m: msgpack.unpackb(m, raw=False),
    auto_offset_reset=config.KAFKA_AUTO_OFFSET_RESET,
    group_id=config.KAFKA_CONSUMER_GROUP,
    max_poll_records=config.KAFKA_MAX_POLL_RECORDS
)
logger.info(f"Kafka consumer initialized for topic: {KAFKA_TOPIC}")


def send_fraud_alert_email(user_id, transaction_id, amount):
    """Send email alert for detected fraud."""
    if not config.ENABLE_EMAIL_ALERTS:
        logger.info(f"Email alerts disabled. Would send alert for transaction {transaction_id}")
        return
    
    try:
        import smtplib
        from email.message import EmailMessage
        
        msg = EmailMessage()
        msg['Subject'] = f'⚠️ Fraudulent Transaction Alert for User {user_id}'
        msg['From'] = config.ALERT_EMAIL_FROM
        msg['To'] = config.ALERT_EMAIL_TO

        msg.set_content(f'''
        Dear User {user_id},

        A potentially fraudulent transaction was detected:
        - Transaction ID: {transaction_id}
        - Amount: ${amount}

        Please review this transaction immediately.

        Regards,
        Fraud Detection System
        ''')

        with smtplib.SMTP_SSL(config.SMTP_SERVER, config.SMTP_PORT) as smtp:
            smtp.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
            smtp.send_message(msg)
        
        logger.info(f"Alert email sent for transaction {transaction_id}")
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")


def process_batch(batch):
    """
    Process a batch of transactions for fraud detection.
    
    Includes SHAP value calculation for model explainability.
    """
    df = pd.DataFrame(batch)
    if df.empty:
        return

    batch_start = time.time()

    drop_cols = [
        'fraud_bool', 'pattern', 'transaction_id', 'sender_id', 'receiver_id',
        'timestamp', 'zip_code', 'ip_address', 'session_id',
        'device_fingerprint', 'transaction_date'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)

    # Align to training column structure
    for col in expected_columns:
        if col not in df_encoded:
            df_encoded[col] = 0
    df_encoded = df_encoded[expected_columns]

    # Predict fraud probabilities
    fraud_probs = model.predict_proba(df_encoded)[:, 1]
    predictions = (fraud_probs > THRESHOLD).astype(int)

    # Calculate SHAP values for explainability (optional - can be expensive)
    shap_values_batch = None
    if config.ENABLE_MODEL_MONITORING:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_encoded)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values_batch = shap_values[1]  # Fraud class
            else:
                shap_values_batch = shap_values
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")

    # Process each transaction
    for i, (tx, prob, pred) in enumerate(zip(batch, fraud_probs, predictions)):
        tx_id = tx["transaction_id"]
        sender = tx["sender_id"]
        
        # Calculate latency
        latency_ms = (time.time() - batch_start) * 1000 / len(batch)

        if pred == 1:
            logger.warning(f"🚨 Fraud Detected: {tx_id} | Prob: {prob:.4f} | Latency: {latency_ms:.2f}ms")
            
            # Send alert
            send_fraud_alert_email(sender, tx_id, tx["amount"])
            
            # Prepare SHAP values
            shap_json = None
            if shap_values_batch is not None:
                shap_dict = {
                    col: float(shap_values_batch[i, j])
                    for j, col in enumerate(expected_columns)
                }
                # Sort by absolute value
                shap_dict = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))
                shap_json = json.dumps(shap_dict)
            
            try:
                cursor.execute("""
                    INSERT INTO fraud_alerts 
                    (transaction_id, sender_id, amount, alert_timestamp, merchant_category, fraud_probability)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s, %s)
                    ON CONFLICT (transaction_id) DO NOTHING
                """, (
                    tx_id,
                    sender,
                    tx["amount"],
                    tx.get("merchant_category", "unknown"),
                    float(prob)
                ))
                conn.commit()
            except Exception as e:
                logger.error(f"DB Insert Error: {e}")
                conn.rollback()
        else:
            logger.info(f"✅ Legit Transaction: {tx_id} | Prob: {prob:.4f} | Latency: {latency_ms:.2f}ms")

    batch_latency = (time.time() - batch_start) * 1000
    logger.info(f"Batch processed: {len(batch)} transactions in {batch_latency:.2f}ms")


# === Inference Loop ===
batch = []
first_ts = None
total_processed = 0

logger.info("Starting fraud detection inference loop...")

try:
    for message in consumer:
        tx = message.value

        if not first_ts:
            first_ts = time.time()

        batch.append(tx)

        if len(batch) >= BATCH_SIZE or (time.time() - first_ts) >= BATCH_TIMEOUT:
            process_batch(batch)
            total_processed += len(batch)
            batch = []
            first_ts = None
            
            if total_processed % 100 == 0:
                logger.info(f"Total transactions processed: {total_processed}")
                
except KeyboardInterrupt:
    logger.info("Shutting down gracefully...")
except Exception as e:
    logger.error(f"Inference loop error: {e}", exc_info=True)
finally:
    consumer.close()
    cursor.close()
    conn.close()
    logger.info(f"Inference engine stopped. Total processed: {total_processed}")

