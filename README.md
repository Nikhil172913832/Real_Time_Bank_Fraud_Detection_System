# Real-Time Fraud Detection System

End-to-end ML pipeline for detecting fraudulent transactions using adversarial training and ensemble methods on the IEEE-CIS fraud detection dataset.

## Features

### ML Pipeline
- XGBoost classifier with temporal train/test split
- Adversarial training with Q-learning agent
- Ensemble model (Isolation Forest + XGBoost)
- Stateful feature engineering (prevents data leakage)
- Drift detection and monitoring

### Dataset
- IEEE-CIS Fraud Detection (590K transactions)
- Real-world fraud patterns
- Temporal validation (no data leakage)

### Engineering
- Kafka streaming for transaction processing
- PostgreSQL for data persistence
- FastAPI REST endpoints
- CI/CD with GitHub Actions
- Comprehensive unit tests

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Download Data
```bash
# Install Kaggle CLI
pip install kaggle

# Download IEEE-CIS dataset
kaggle competitions download -c ieee-fraud-detection -p data/raw/
cd data/raw && unzip ieee-fraud-detection.zip
```

### Train Models
```bash
# Train base XGBoost model
python train.py

# Train ensemble model
python train_ensemble.py

# Train adversarial agent
python train_adversarial.py
```

### Evaluate
```bash
python evaluate.py
```

## Architecture

```
Data Loading → Feature Engineering → Model Training → Adversarial Training
     ↓                  ↓                   ↓                  ↓
IEEE-CIS          Stateful Fit/      XGBoost +         Q-Learning
Dataset           Transform         Isolation         Agent
                                    Forest
```

## Key Differentiators

1. **Adversarial Training**: Q-learning agent learns to evade detection, model adapts
2. **Ensemble Approach**: Supervised (XGBoost) + Unsupervised (Isolation Forest)
3. **No Data Leakage**: Temporal validation and stateful feature engineering
4. **Production Ready**: Drift detection, monitoring, comprehensive tests

## Project Structure

```
.
├── src/
│   ├── data/              # Data loading and validation
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementations
│   ├── adversarial/       # Q-learning agent
│   └── monitoring/        # Drift detection
├── tests/
│   └── unit/              # Unit tests
├── train.py               # Base model training
├── train_ensemble.py      # Ensemble training
├── train_adversarial.py   # Adversarial training
└── evaluate.py            # Model evaluation
```

## Performance

Metrics on IEEE-CIS test set (temporal split):

| Metric | Value |
|--------|-------|
| ROC-AUC | Run `evaluate.py` |
| Precision | Run `evaluate.py` |
| Recall | Run `evaluate.py` |
| F1-Score | Run `evaluate.py` |

## Development

### Run Tests
```bash
pytest tests/
```

### Code Quality
```bash
black src/ tests/
flake8 src/ tests/
```

## License

MIT
