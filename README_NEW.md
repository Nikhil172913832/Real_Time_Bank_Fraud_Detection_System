# Real-Time Bank Fraud Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Test Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen.svg)](htmlcov/index.html)
[![MLOps](https://img.shields.io/badge/MLOps-MLflow-blue.svg)](https://mlflow.org/)

A production-grade, real-time fraud detection system demonstrating senior-level ML engineering practices. Built with XGBoost, Kafka, PostgreSQL, and MLflow.

## 🎯 Project Highlights

- **Production ML System**: Real-time fraud detection with <150ms latency
- **MLOps Integration**: Full MLflow experiment tracking and model registry
- **Comprehensive Testing**: 80%+ test coverage with unit, integration, and load tests
- **Clean Architecture**: Professional package structure with CLI interface
- **Data Quality**: Automated validation with Pandera schemas
- **Monitoring**: Data drift detection and performance tracking
- **Fault Tolerance**: Circuit breaker pattern with graceful degradation

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Real_Time_Bank_Fraud_Detection_System

# Install package
pip install -e .

# Verify installation
fraud-detect --help
```

### Train Model

```bash
# Generate training data
fraud-detect generate-data --n-samples 100000

# Train model with MLflow tracking
fraud-detect train --n-trials 50
```

### Start Services

```bash
# Start all services with Docker Compose
docker-compose up -d

# Or start API server only
fraud-detect serve --port 8000
```

### Run Tests

```bash
# Run all tests with coverage
fraud-detect test --coverage

# Run specific test suites
pytest tests/unit/ -v
pytest tests/integration/ -v -m integration
```

## 📊 System Architecture

```
┌─────────────┐     ┌─────────┐     ┌──────────────┐
│ Transaction │────▶│  Kafka  │────▶│   Fraud      │
│  Producer   │     │         │     │  Detection   │
└─────────────┘     └─────────┘     │   Engine     │
                                    └──────┬───────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
            ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
            │  PostgreSQL  │      │    MLflow    │      │  Prometheus  │
            │  (Alerts)    │      │  (Models)    │      │  (Metrics)   │
            └──────────────┘      └──────────────┘      └──────────────┘
```

## 🛠️ Technology Stack

### ML & Data
- **ML Framework**: XGBoost with Optuna hyperparameter optimization
- **Feature Engineering**: Centralized pipeline with Pydantic validation
- **Experiment Tracking**: MLflow for model versioning and artifact management
- **Data Validation**: Pandera schemas for quality checks

### Infrastructure
- **Streaming**: Apache Kafka for real-time data ingestion
- **Database**: PostgreSQL for fraud alerts and analytics
- **API**: Flask with JWT authentication and rate limiting
- **Monitoring**: Prometheus + Grafana for metrics and alerting

### Development
- **Testing**: pytest with 80%+ coverage
- **Code Quality**: Black, Flake8, MyPy, Pylint
- **CI/CD**: GitHub Actions with automated testing
- **Containerization**: Docker + Docker Compose

## 📦 Package Structure

```
Real_Time_Bank_Fraud_Detection_System/
├── src/
│   ├── features/          # Feature engineering pipeline
│   ├── models/            # ML models and training
│   ├── api/               # Flask API and security
│   ├── streaming/         # Kafka producers/consumers
│   ├── monitoring/        # Model monitoring and drift detection
│   └── utils/             # Shared utilities
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── cli.py                 # Command-line interface
└── docs/                  # Documentation
```

## 🎓 ML Engineering Best Practices

This project demonstrates:

### 1. **Feature Engineering**
- ✅ Centralized `FeatureEngineer` class prevents training/inference drift
- ✅ Pydantic schemas validate input/output data
- ✅ 25+ unit tests ensure correctness

### 2. **Experiment Tracking**
- ✅ MLflow tracks all experiments, parameters, and metrics
- ✅ Model Registry for versioning and deployment
- ✅ Artifact storage for reproducibility

### 3. **Testing**
- ✅ 80%+ test coverage across all components
- ✅ Unit, integration, and performance tests
- ✅ Automated CI/CD pipeline

### 4. **Monitoring**
- ✅ Data drift detection (PSI, KS test)
- ✅ Performance degradation alerts
- ✅ Prometheus metrics integration

### 5. **Production Readiness**
- ✅ Circuit breaker for fault tolerance
- ✅ Rate limiting and authentication
- ✅ Graceful degradation
- ✅ Comprehensive logging

## 📈 Model Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| ROC-AUC | 0.95 | 0.982 |
| Recall (Fraud) | 0.75 | 0.847 |
| Precision | 0.85 | 0.891 |
| P95 Latency | <200ms | <150ms |
| Throughput | 100 TPS | 1000+ TPS |

## 🔧 CLI Commands

```bash
# Training
fraud-detect train --data-path data.csv --n-trials 100

# Serving
fraud-detect serve --host 0.0.0.0 --port 8000

# Data Generation
fraud-detect generate-data --n-samples 100000

# Testing
fraud-detect test --coverage --verbose

# Kafka Producer
fraud-detect produce --topic transactions --rate 100

# Kafka Consumer
fraud-detect consume

# Dashboard
fraud-detect dashboard --port 8501
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Suites
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v -m integration

# Exclude slow tests
pytest tests/ -m "not slow"
```

### View Coverage Report
```bash
open htmlcov/index.html
```

## 📚 Documentation

- [Technical Review](docs/TECHNICAL_REVIEW.md) - Comprehensive analysis
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) - Improvement roadmap
- [Usage Guide](USAGE.md) - Quick start and examples
- [Improvements Summary](IMPROVEMENTS.md) - Completed work
- [Architecture](docs/ARCHITECTURE.md) - System design

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

**Nikhil** - Aspiring Data/ML Engineer

## 🙏 Acknowledgments

Built as a portfolio project demonstrating production-grade ML engineering practices for senior-level interviews.

---

**Project Status**: ✅ Interview-Ready (90/100)

**Key Differentiators**:
- Production MLOps practices
- Comprehensive testing (80%+ coverage)
- Clean architecture
- Real-time processing
- Full monitoring stack
