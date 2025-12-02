# API Documentation

## Base URL
```
http://localhost:5000/api/v1
```

## Authentication

All API endpoints (except `/health`) require authentication using JWT tokens.

### Obtaining a Token
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using the Token
Include the token in the Authorization header:
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## Endpoints

### 1. Health Check

Check if the service is running and healthy.

**Endpoint:** `GET /health`

**Authentication:** None required

**Response:**
```json
{
  "status": "healthy",
  "model_version": "v1.2.0",
  "model_loaded_at": "2025-12-02T10:30:00",
  "uptime_seconds": 3600,
  "checks": {
    "database": "ok",
    "kafka": "ok",
    "redis": "ok",
    "model": "ok"
  }
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service is unhealthy

---

### 2. Single Transaction Prediction

Predict fraud probability for a single transaction.

**Endpoint:** `POST /predict`

**Authentication:** Required

**Rate Limit:** 100 requests per minute

**Request Body:**
```json
{
  "transaction_id": "tx_12345",
  "user_id": "user_789",
  "amount": 1500.00,
  "currency": "USD",
  "source": "online",
  "device_os": "iOS",
  "browser": "Safari",
  "merchant_category": "retail",
  "merchant_id": "merch_456",
  "is_international": false,
  "country_code": "US",
  "hour_of_day": 14,
  "day_of_week": 2,
  "timestamp": "2025-12-02T14:30:00Z"
}
```

**Response:**
```json
{
  "transaction_id": "tx_12345",
  "fraud_probability": 0.0234,
  "is_fraud": false,
  "risk_level": "low",
  "threshold": 0.2,
  "prediction_time_ms": 45.2,
  "model_version": "v1.2.0",
  "timestamp": "2025-12-02T14:30:01Z",
  "explanation": {
    "top_features": [
      {"feature": "amount", "impact": 0.12},
      {"feature": "velocity_24h", "impact": 0.08},
      {"feature": "merchant_risk_level", "impact": 0.06}
    ],
    "shap_values_url": "/api/v1/explanations/tx_12345"
  }
}
```

**Risk Levels:**
- `low`: fraud_probability < 0.2
- `medium`: 0.2 ≤ fraud_probability < 0.5
- `high`: 0.5 ≤ fraud_probability < 0.8
- `critical`: fraud_probability ≥ 0.8

**Status Codes:**
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid input
- `401 Unauthorized` - Missing/invalid token
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

---

### 3. Batch Predictions

Predict fraud probability for multiple transactions in a single request.

**Endpoint:** `POST /predict/batch`

**Authentication:** Required

**Rate Limit:** 10 requests per minute

**Max Batch Size:** 100 transactions

**Request Body:**
```json
{
  "transactions": [
    {
      "transaction_id": "tx_001",
      "amount": 1500.00,
      "source": "online"
    },
    {
      "transaction_id": "tx_002",
      "amount": 5000.00,
      "source": "atm"
    }
  ]
}
```

**Response:**
```json
{
  "batch_id": "batch_abc123",
  "batch_size": 2,
  "total_time_ms": 89.5,
  "avg_time_ms": 44.75,
  "model_version": "v1.2.0",
  "predictions": [
    {
      "transaction_id": "tx_001",
      "fraud_probability": 0.0234,
      "is_fraud": false,
      "risk_level": "low"
    },
    {
      "transaction_id": "tx_002",
      "fraud_probability": 0.7821,
      "is_fraud": true,
      "risk_level": "high"
    }
  ],
  "summary": {
    "total": 2,
    "fraud": 1,
    "legitimate": 1,
    "fraud_rate": 0.5
  }
}
```

**Status Codes:**
- `200 OK` - Predictions successful
- `400 Bad Request` - Invalid input or batch too large
- `401 Unauthorized` - Missing/invalid token
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

---

### 4. Get Model Information

Retrieve information about the current model and its performance.

**Endpoint:** `GET /model/info`

**Authentication:** Required

**Response:**
```json
{
  "model_version": "v1.2.0",
  "model_type": "XGBoost Classifier",
  "trained_at": "2025-11-25T10:00:00Z",
  "loaded_at": "2025-12-02T09:00:00Z",
  "model_path": "models/xgb_final.pkl",
  "fraud_threshold": 0.2,
  "num_features": 73,
  "performance_metrics": {
    "roc_auc": 0.982,
    "recall": 0.805,
    "precision": 0.923,
    "f1_score": 0.860,
    "false_positive_rate": 0.021,
    "test_samples": 10000
  },
  "feature_importance": [
    {"feature": "amount", "importance": 0.145},
    {"feature": "velocity_24h", "importance": 0.098},
    {"feature": "merchant_risk_level", "importance": 0.087}
  ]
}
```

**Status Codes:**
- `200 OK` - Information retrieved
- `401 Unauthorized` - Missing/invalid token
- `500 Internal Server Error` - Server error

---

### 5. Prometheus Metrics

Export metrics in Prometheus format for monitoring.

**Endpoint:** `GET /metrics`

**Authentication:** None required (typically firewalled)

**Response:**
```
# HELP fraud_predictions_total Total number of fraud predictions
# TYPE fraud_predictions_total counter
fraud_predictions_total{prediction="fraud"} 1234.0
fraud_predictions_total{prediction="legitimate"} 8766.0

# HELP fraud_prediction_latency_seconds Fraud prediction latency
# TYPE fraud_prediction_latency_seconds histogram
fraud_prediction_latency_seconds_bucket{le="0.01"} 1500.0
fraud_prediction_latency_seconds_bucket{le="0.05"} 8900.0
fraud_prediction_latency_seconds_bucket{le="0.1"} 9950.0
fraud_prediction_latency_seconds_bucket{le="+Inf"} 10000.0
fraud_prediction_latency_seconds_sum 450.5
fraud_prediction_latency_seconds_count 10000.0

# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="predict",method="POST",status="200"} 9850.0
api_requests_total{endpoint="predict",method="POST",status="400"} 120.0
api_requests_total{endpoint="predict",method="POST",status="500"} 30.0
```

**Content-Type:** `text/plain; version=0.0.4`

---

### 6. Get Explanation

Get detailed SHAP explanation for a prediction.

**Endpoint:** `GET /explanations/{transaction_id}`

**Authentication:** Required

**Response:**
```json
{
  "transaction_id": "tx_12345",
  "fraud_probability": 0.0234,
  "base_value": 0.15,
  "shap_values": [
    {"feature": "amount", "value": 1500.0, "shap_value": -0.05},
    {"feature": "velocity_24h", "value": 3, "shap_value": -0.02},
    {"feature": "merchant_risk_level", "value": "low", "shap_value": -0.03}
  ],
  "waterfall_plot_url": "/api/v1/plots/waterfall/tx_12345",
  "force_plot_url": "/api/v1/plots/force/tx_12345"
}
```

---

### 7. Model Comparison (A/B Testing)

Compare predictions from multiple model versions.

**Endpoint:** `POST /predict/compare`

**Authentication:** Required (admin only)

**Request Body:**
```json
{
  "transaction": {
    "amount": 1500.00,
    "source": "online"
  },
  "models": ["v1.1.0", "v1.2.0"]
}
```

**Response:**
```json
{
  "transaction_id": "tx_12345",
  "comparisons": [
    {
      "model_version": "v1.1.0",
      "fraud_probability": 0.0289,
      "is_fraud": false,
      "prediction_time_ms": 52.3
    },
    {
      "model_version": "v1.2.0",
      "fraud_probability": 0.0234,
      "is_fraud": false,
      "prediction_time_ms": 45.2
    }
  ],
  "recommendation": "v1.2.0"
}
```

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "details": {
    "field": "amount",
    "constraint": "must be positive"
  },
  "timestamp": "2025-12-02T14:30:00Z",
  "path": "/api/v1/predict",
  "request_id": "req_abc123"
}
```

### Common Error Codes

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| 400 | Bad Request | Invalid input data |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service is down |

---

## Rate Limiting

Rate limits are applied per IP address and per authenticated user.

| Endpoint | Limit |
|----------|-------|
| `/predict` | 100 requests/minute |
| `/predict/batch` | 10 requests/minute |
| `/model/info` | 30 requests/minute |
| `/explanations/*` | 50 requests/minute |

**Rate Limit Headers:**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1638446400
```

---

## Pagination

For endpoints that return lists (future feature):

**Request:**
```http
GET /api/v1/transactions?page=2&per_page=50
```

**Response:**
```json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "per_page": 50,
    "total_pages": 10,
    "total_items": 500
  }
}
```

---

## Webhooks

Subscribe to fraud alerts via webhooks.

**Configuration:**
```json
{
  "url": "https://your-server.com/webhook",
  "events": ["fraud_detected", "high_risk_transaction"],
  "secret": "your_webhook_secret"
}
```

**Webhook Payload:**
```json
{
  "event": "fraud_detected",
  "transaction_id": "tx_12345",
  "fraud_probability": 0.8521,
  "timestamp": "2025-12-02T14:30:00Z",
  "signature": "sha256=..."
}
```

---

## SDKs

### Python
```python
from fraud_detection_client import FraudDetectionAPI

client = FraudDetectionAPI(
    base_url="http://localhost:5000",
    api_key="your_api_key"
)

prediction = client.predict({
    "amount": 1500.00,
    "source": "online"
})

print(f"Fraud probability: {prediction.fraud_probability}")
```

### JavaScript
```javascript
const FraudDetectionAPI = require('fraud-detection-client');

const client = new FraudDetectionAPI({
  baseUrl: 'http://localhost:5000',
  apiKey: 'your_api_key'
});

const prediction = await client.predict({
  amount: 1500.00,
  source: 'online'
});

console.log(`Fraud probability: ${prediction.fraudProbability}`);
```

---

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
```
http://localhost:5000/openapi.json
```

Interactive documentation (Swagger UI):
```
http://localhost:5000/docs
```

ReDoc documentation:
```
http://localhost:5000/redoc
```
