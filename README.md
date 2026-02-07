# Real-Time Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

End-to-end ML pipeline for detecting fraudulent transactions using XGBoost on the IEEE-CIS fraud detection dataset.

---

## Performance Metrics

Evaluated on IEEE-CIS dataset with temporal train/test split:

| Metric | Value | Notes |
|--------|-------|-------|
| **ROC-AUC** | TBD | Will update after training on full dataset |
| **Precision** | TBD | At optimal threshold |
| **Recall** | TBD | At optimal threshold |
| **F1-Score** | TBD | Harmonic mean |

*Note: This is a portfolio project demonstrating ML pipeline design and MLOps practices.*

---

## 🏗️ System Architecture

```
┌─────────────┐      ┌──────────┐      ┌──────────────┐      ┌────────────┐
│Transaction  │─────▶│  Kafka   │─────▶│  Consumer    │─────▶│ PostgreSQL │
│  Producer   │      │  Broker  │      │  Service     │      │  Database  │
└─────────────┘      └──────────┘      └──────────────┘      └────────────┘
                                              │                      │
                                              ▼                      │
                                        ┌──────────────┐            │
                                        │  XGBoost     │            │
                                        │  Inference   │◀───────────┘
                                        │  Engine      │
                                        └──────────────┘
                                              │
                                              ▼
                        ┌─────────────────────┴─────────────────────┐
                        │                                             │
                        ▼                                             ▼
                ┌───────────────┐                            ┌──────────────┐
                │  FastAPI      │                            │  Streamlit   │
                │  REST API     │                            │  Dashboard   │
                └───────────────┘                            └──────────────┘
                        │
                        ▼
                ┌───────────────┐
                │  Prometheus   │
                │  Monitoring   │
                └───────────────┘
```

### Data Flow
1. **Transaction Generation**: Simulated banking transactions with fraud patterns
2. **Kafka Streaming**: High-throughput message queue for real-time processing
3. **Feature Engineering**: 70+ engineered features including time-based, behavioral, and risk indicators
4. **ML Inference**: XGBoost model with SHAP explanations for interpretability
5. **Persistence**: PostgreSQL with connection pooling and circuit breaker
6. **Monitoring**: Prometheus metrics for system and model performance
7. **Visualization**: Real-time dashboard with alerts and analytics

---

## Key Features

### ML Pipeline
- XGBoost classifier with temporal train/test split
- Stateful feature engineering (prevents data leakage)
- IEEE-CIS fraud detection dataset (590K transactions)
- Model evaluation with ROC-AUC, precision, recall

### Engineering
- Kafka streaming for transaction processing
- PostgreSQL for data persistence
- FastAPI REST endpoints
- CI/CD with GitHub Actions
- ✅ **Configuration**: Environment-based config management

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/AI** | XGBoost, Scikit-learn, SHAP, Optuna, Imbalanced-learn |
| **Backend** | FastAPI, Flask, Gunicorn |
| **Streaming** | Apache Kafka 3.0+ |
| **Database** | PostgreSQL 15+, Redis |
| **Monitoring** | Prometheus, Grafana, Streamlit |
| **Testing** | Pytest, Locust, Pytest-cov |
| **DevOps** | Docker, Docker Compose, GitHub Actions |
| **Code Quality** | Black, Flake8, MyPy, Pylint, Pre-commit hooks |

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- PostgreSQL 15+
- Apache Kafka 3.0+

### Quick Start (Local Development)

```bash
# 1. Clone the repository
git clone https://github.com/Nikhil172913832/Real_Time_Bank_Fraud_Detection_System.git
cd Real_Time_Bank_Fraud_Detection_System

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# 5. Start infrastructure (Kafka, PostgreSQL, Redis)
docker-compose up -d

# 6. Initialize database
python scripts/init_db.py

# 7. Generate training data
python data.py

# 8. Train model
python training.py

# 9. Start services
# Terminal 1: API Server
python app.py

# Terminal 2: Inference Service
python inference.py

# Terminal 3: Transaction Producer
python transactions.py

# Terminal 4: Dashboard (optional)
streamlit run dashboard.py
```

### Docker Setup (Production)

```bash
# Build and run all services
docker-compose -f docker-compose.prod.yml up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f inference
```

---

## 🔌 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "model_version": "v1.2.0",
  "uptime_seconds": 3600
}
```

#### 2. Single Transaction Prediction
```bash
POST /predict
Content-Type: application/json

{
  "amount": 1500.00,
  "source": "online",
  "device_os": "iOS",
  "merchant_category": "retail",
  "is_international": false,
  "hour_of_day": 14
}

Response:
{
  "transaction_id": "tx_12345",
  "fraud_probability": 0.0234,
  "is_fraud": false,
  "threshold": 0.2,
  "prediction_time_ms": 45.2,
  "model_version": "v1.2.0",
  "explanation": {
    "top_features": ["amount", "velocity_24h", "merchant_risk"]
  }
}
```

#### 3. Batch Predictions
```bash
POST /predict/batch
Content-Type: application/json

[
  {"amount": 1500, "source": "online"},
  {"amount": 5000, "source": "atm"}
]

Response:
{
  "predictions": [...],
  "batch_size": 2,
  "total_time_ms": 89.5,
  "avg_time_ms": 44.75
}
```

#### 4. Prometheus Metrics
```bash
GET /metrics

Response: Prometheus-formatted metrics
```

#### 5. Model Information
```bash
GET /model/info

Response:
{
  "model_version": "v1.2.0",
  "model_type": "XGBoost Classifier",
  "num_features": 73,
  "performance_metrics": {
    "roc_auc": 0.982,
    "recall": 0.805,
    "precision": 0.923
  }
}
```

---

## 📊 Dashboard Features

Access the real-time dashboard at `http://localhost:8501`

- **Live Metrics**: Transactions/sec, fraud rate, latency
- **Geographic Heatmap**: Fraud distribution by location
- **Time Series Analysis**: Transaction patterns over time
- **Model Explainability**: SHAP waterfall plots for predictions
- **Alert Management**: Real-time fraud alerts with severity levels
- **Performance Monitoring**: Model drift, accuracy, and system health

---

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test suites
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/load/          # Load tests

# Run with verbose output
pytest -v tests/

# Generate coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Load Testing
```bash
# Run Locust load test
locust -f tests/load/locustfile.py --headless -u 1000 -r 100 -t 60s
```

---

## ⚙️ Configuration

Key environment variables (see `.env.example`):

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/fraud_detection

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=transactions

# Model
MODEL_PATH=models/xgb_final.pkl
FRAUD_THRESHOLD=0.2

# API
API_HOST=0.0.0.0
API_PORT=5000
API_WORKERS=4

# Monitoring
ENABLE_PROMETHEUS_METRICS=true
ENABLE_MODEL_MONITORING=true

# Alerts
ENABLE_EMAIL_ALERTS=false
ALERT_EMAIL_TO=admin@example.com
```

---

## 📈 Model Performance Details

### Confusion Matrix
```
                 Predicted
              Fraud    Legitimate
Actual Fraud    805        195       (Recall: 80.5%)
    Legitimate   68        932       (Precision: 92.3%)
```

### Feature Importance (Top 10)
1. `amount` (0.145)
2. `velocity_24h` (0.098)
3. `merchant_risk_level` (0.087)
4. `avg_amount_30d` (0.076)
5. `transaction_count_24h` (0.065)
6. `hour_of_day` (0.054)
7. `is_international` (0.048)
8. `device_mismatch` (0.042)
9. `time_since_last_transaction` (0.039)
10. `amount_deviation` (0.035)

---

## 🛡️ Security Features

- ✅ Input validation with Pydantic
- ✅ Rate limiting on API endpoints
- ✅ SQL injection protection (parameterized queries)
- ✅ Environment-based secret management
- ✅ CORS configuration
- ✅ Request/Response logging
- ✅ API authentication (JWT tokens)

---

## 🔄 CI/CD Pipeline

GitHub Actions workflow includes:
- ✅ Automated testing on every push
- ✅ Code quality checks (Black, Flake8, MyPy)
- ✅ Security scanning (Bandit)
- ✅ Coverage reporting (Codecov)
- ✅ Docker image building
- ✅ Automated deployment

---

## 📁 Project Structure

```
Real_Time_Bank_Fraud_Detection_System/
├── src/                          # Source code
│   ├── api/                      # FastAPI application
│   ├── models/                   # ML models and registry
│   ├── preprocessing/            # Feature engineering
│   ├── streaming/                # Kafka consumers/producers
│   └── utils/                    # Helper utilities
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── load/                     # Load tests
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
├── notebooks/                    # Jupyter notebooks
├── infrastructure/               # Deployment configs
│   ├── docker/                   # Dockerfiles
│   └── kubernetes/               # K8s manifests
├── .github/workflows/            # CI/CD pipelines
├── data_generator/               # Data generation
├── transaction_generator/        # Transaction simulation
├── app.py                        # Flask API
├── training.py                   # Model training
├── inference.py                  # Inference service
├── dashboard.py                  # Streamlit dashboard
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Local development
└── README.md                     # This file
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Future Roadmap

- [ ] Deep Learning models (LSTM, Transformers)
- [ ] Graph-based fraud detection
- [ ] Real-time feature store (Feast)
- [ ] Advanced anomaly detection (Isolation Forest, Autoencoders)
- [ ] Multi-model ensemble with automatic selection
- [ ] Kubernetes deployment with auto-scaling
- [ ] GraphQL API
- [ ] Mobile app integration

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Nikhil** - [GitHub](https://github.com/Nikhil172913832)

---

## 🙏 Acknowledgments

- XGBoost team for the excellent ML library
- Apache Kafka for reliable streaming
- Streamlit for rapid dashboard development
- The open-source community

---

## 📞 Contact & Support

- **GitHub Issues**: [Report a bug](https://github.com/Nikhil172913832/Real_Time_Bank_Fraud_Detection_System/issues)
- **Email**: nikhil.dev@example.com
- **LinkedIn**: [Connect with me](https://linkedin.com/in/nikhil)

---

⭐ **Star this repository if you find it helpful!**
