# Real-Time Bank Fraud Detection System - Usage Guide

## Quick Start

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

### Basic Usage

#### 1. Generate Training Data
```bash
fraud-detect generate-data --n-samples 100000 --output data.csv
```

#### 2. Train Model
```bash
# Basic training
fraud-detect train --data-path data.csv

# With custom parameters
fraud-detect train \
  --data-path data.csv \
  --n-trials 50 \
  --test-size 0.2 \
  --random-state 42
```

#### 3. Start API Server
```bash
# Default (localhost:5000)
fraud-detect serve

# Custom host/port
fraud-detect serve --host 0.0.0.0 --port 8000
```

#### 4. Run Tests
```bash
# All tests
fraud-detect test

# With coverage
fraud-detect test --coverage

# Verbose output
fraud-detect test --verbose
```

## MLflow Integration

### Start MLflow Server
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5001
```

### Access MLflow UI
Open browser: http://localhost:5001

### View Experiments
- Navigate to "fraud_detection" experiment
- Compare runs by metrics
- View model artifacts
- Download models from registry

## API Usage

### Health Check
```bash
curl http://localhost:5000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_001",
    "amount": 100.0,
    "source": "online",
    "merchant_category": "Restaurants",
    "is_international": false
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"transaction_id": "tx_001", "amount": 100.0, ...},
    {"transaction_id": "tx_002", "amount": 500.0, ...}
  ]'
```

### Metrics (Prometheus)
```bash
curl http://localhost:5000/metrics
```

## Development

### Run Tests
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v -m integration

# With coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Code Quality
```bash
# Format code
black .

# Lint
flake8 .
pylint src/

# Type checking
mypy src/
```

## Docker Deployment

### Start All Services
```bash
docker-compose up -d
```

### Services
- **API**: http://localhost:5000
- **MLflow**: http://localhost:5001
- **Kafka UI**: http://localhost:8080
- **PgAdmin**: http://localhost:5050

## Troubleshooting

### Model Not Found
```bash
# Check if model exists
ls -la models/

# Retrain model
fraud-detect train
```

### MLflow Connection Error
```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5001
```

### Test Failures
```bash
# Install test dependencies
pip install -r requirements.txt

# Run specific test
pytest tests/unit/test_feature_engineering.py::TestTemporalFeatures::test_hour_of_day -v
```

## Project Structure

```
Real_Time_Bank_Fraud_Detection_System/
├── src/
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   ├── api/               # API layer
│   ├── data/              # Data generation
│   ├── streaming/         # Kafka integration
│   ├── monitoring/        # Model monitoring
│   └── utils/             # Shared utilities
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── cli.py                 # Command-line interface
├── training.py            # Training script
├── inference.py           # Inference script
├── app.py                 # Flask API
└── setup.py               # Package configuration
```

## Next Steps

1. **Review technical_review.md** for detailed analysis
2. **Review implementation_plan.md** for improvement roadmap
3. **Review walkthrough.md** for completed work summary
4. **Explore Priority 2 improvements** for additional enhancements
