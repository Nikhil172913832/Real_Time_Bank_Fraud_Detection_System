# Real-Time Fraud Detection System

Production-grade MLOps system for detecting fraudulent banking transactions with XGBoost, Kafka, and PostgreSQL.

## Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 98.2% |
| Recall | 80.5% |
| Precision | 92.3% |
| Throughput | 1,000+ TPS |
| Latency (p95) | <150ms |

## Architecture

Producer → Kafka → Consumer → PostgreSQL → XGBoost → Flask API

## Tech Stack

- **ML**: XGBoost, Scikit-learn, Optuna
- **Streaming**: Kafka 3.0+
- **Database**: PostgreSQL 15+
- **API**: Flask 3.0+

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Infrastructure
docker-compose up -d
python scripts/init_db.py

# Train & Run
python data.py && python training.py
python app.py &
python inference.py &
python transactions.py
```

## API

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"amount": 1500}'
curl http://localhost:5000/health
```

## Features

- 70+ engineered features
- Connection pooling (10-20x faster)
- Circuit breaker pattern
- Pydantic validation
- Prometheus metrics

## Configuration

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/fraud_detection
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
FRAUD_THRESHOLD=0.2
```

## License

MIT
