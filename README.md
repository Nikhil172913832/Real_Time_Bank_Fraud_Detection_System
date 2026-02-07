# Real-Time Fraud Detection System

Advanced fraud detection using state-of-the-art deep learning: Graph Neural Networks, Transformers, and adversarial training on IEEE-CIS dataset.

## Features

### Deep Learning Models
- **Graph Neural Network (GNN)**: Heterogeneous graph with attention for transaction networks
- **Temporal Transformer**: Multi-head self-attention for sequential patterns
- **Hybrid Model**: Combines GNN (structural) + Transformer (temporal)
- **Ensemble**: XGBoost + Isolation Forest
- **Adversarial Training**: Q-learning agent that adapts to evade detection

### Advanced Metrics
- **AUPRC**: Area under precision-recall curve (key metric for imbalanced data)
- **Precision@K / Recall@K**: Operational metrics for top-K predictions
- **Cost-Sensitive**: Penalizes false negatives 10x more than false positives
- **Focal Loss**: Addresses class imbalance in training

### Dataset
- IEEE-CIS Fraud Detection (590K transactions)
- Real-world fraud patterns
- Temporal validation (no data leakage)

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

# Train advanced deep learning models
python train_advanced.py --model gnn          # Graph Neural Network
python train_advanced.py --model transformer  # Temporal Transformer
python train_advanced.py --model hybrid       # GNN + Transformer
```

### Evaluate
```bash
python evaluate.py
```

## Architecture

```
Transaction Data
    ├─→ GNN (Graph Structure) ────────┐
    ├─→ Transformer (Temporal) ────────┼─→ Hybrid Fusion ─→ Prediction
    ├─→ XGBoost (Supervised) ──────────┤
    └─→ Isolation Forest (Anomaly) ────┘
         ↓
    Adversarial Agent (Q-Learning)
         ↓
    Adaptive Retraining
```

## Key Differentiators

1. **Graph Neural Networks**: Models transaction network with users, merchants, devices
2. **Temporal Transformers**: Captures long-range sequential dependencies with attention
3. **Hybrid Architecture**: Combines structural (GNN) and temporal (Transformer) patterns
4. **Advanced Metrics**: AUPRC, Precision@K for imbalanced data
5. **Adversarial Training**: Q-learning agent learns to evade detection
6. **No Data Leakage**: Temporal validation and stateful feature engineering

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
