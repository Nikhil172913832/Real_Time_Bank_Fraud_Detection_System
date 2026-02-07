# Model Architecture

## Ensemble Approach

The system uses a two-stage ensemble combining supervised and unsupervised methods:

### 1. Isolation Forest (Unsupervised)
- Trained only on legitimate transactions
- Detects anomalies and novel fraud patterns
- Contamination rate set to observed fraud rate (~3.5%)

### 2. XGBoost (Supervised)
- Trained on labeled data (fraud + legitimate)
- Learns known fraud patterns
- Hyperparameters: 100 estimators, max depth 6, learning rate 0.1

### Ensemble Logic
```python
prediction = max(isolation_forest_pred, xgboost_pred)
```
Flag as fraud if EITHER model predicts fraud.

## Feature Engineering

Stateful feature engineering with fit/transform pattern:

### Fit Phase (Training Data Only)
```python
engineer.fit(train_df)
# Computes statistics: mean, std, median
```

### Transform Phase (Train and Test)
```python
train_features = engineer.transform(train_df)
test_features = engineer.transform(test_df)
# Uses pre-computed statistics, no data leakage
```

### Features Created
- `amount_log`: Log-transformed amount
- `amount_decimal`: Decimal portion of amount
- `amount_vs_mean`: Amount relative to training mean
- `amount_vs_median`: Amount relative to training median
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6)
- `is_weekend`: Weekend flag

## Temporal Validation

Critical for fraud detection:

```python
# Sort by time
df = df.sort_values('TransactionDT')

# Split: first 80% train, last 20% test
split_idx = int(len(df) * 0.8)
train = df[:split_idx]
test = df[split_idx:]
```

This ensures:
- No future data in training
- Realistic evaluation (model only sees past)
- Prevents data leakage

## Drift Detection

Monitors feature distribution changes:

```python
drift_detector = DriftDetector(reference_data=train_df)
drift_results = drift_detector.check_drift(current_data=new_df)

if drift_results['drift_detected']:
    # Trigger retraining
```

Drift threshold: 30% of features with >2 standard deviations shift
