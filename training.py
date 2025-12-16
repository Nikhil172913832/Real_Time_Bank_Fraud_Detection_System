"""
High-Performance Fraud Detection Model Training Pipeline
========================================================

This module implements a production-grade training pipeline for fraud detection:
- XGBoost classifier with hyperparameter optimization
- Advanced feature engineering (70+ features)
- Stratified K-fold cross-validation
- Comprehensive metrics tracking (ROC-AUC, Recall, Precision, F1)
- Model versioning and artifact storage

Performance Target: 98% ROC-AUC, 80% Recall
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib
import optuna
from optuna.samplers import TPESampler

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    recall_score, 
    precision_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('metrics', exist_ok=True)


class FraudDetectionTrainer:
    """
    Production-grade trainer for fraud detection models.
    
    Features:
    - Automated feature engineering
    - Hyperparameter optimization with Optuna
    - Cross-validation with proper stratification
    - Comprehensive metrics tracking
    - Model versioning
    """
    
    def __init__(
        self, 
        data_path: str = "data.csv",
        test_size: float = 0.2,
        random_state: int = 42,
        n_trials: int = 100
    ):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.n_trials = n_trials
        self.model = None
        self.feature_columns = None
        self.metrics = {}
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data and perform feature engineering.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df):,} transactions")
        
        # Import FeatureEngineer
        from src.features.engineering import FeatureEngineer
        
        # Feature engineering using centralized class
        engineer = FeatureEngineer(validate_schema=False)
        df = engineer.transform(df)
        
        # Separate features and target
        target_col = 'fraud_bool' if 'fraud_bool' in df.columns else 'is_fraud'
        y = df[target_col].astype(int)
        
        # Prepare features for model
        X = engineer.prepare_for_model(df, encode_categoricals=True)
        
        # Store feature columns for inference
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Feature engineering complete: {len(self.feature_columns)} features")
        logger.info(f"Fraud ratio: {y.mean():.4f} ({y.sum():,} / {len(y):,})")
        
        return X, y
    
    def objective(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training target
            
        Returns:
            Average ROC-AUC score from cross-validation
        """
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100),
            'random_state': self.random_state,
            'tree_method': 'hist',
            'eval_metric': 'auc'
        }
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            model = XGBClassifier(**params)
            model.fit(X_fold_train, y_fold_train, verbose=False)
            
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            score = roc_auc_score(y_fold_val, y_pred_proba)
            scores.append(score)
        
        return np.mean(scores)
    
    def train(self) -> Dict:
        """
        Execute the complete training pipeline.
        
        Returns:
            Dictionary containing training metrics
        """
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Starting Fraud Detection Model Training Pipeline")
        logger.info("=" * 80)
        
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        logger.info(f"Train set: {len(X_train):,} | Test set: {len(X_test):,}")
        
        # Hyperparameter optimization
        logger.info(f"Starting hyperparameter optimization ({self.n_trials} trials)...")
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        logger.info(f"Best ROC-AUC from CV: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {json.dumps(best_params, indent=2)}")
        
        # Train final model with best parameters
        logger.info("Training final model with best hyperparameters...")
        self.model = XGBClassifier(
            **best_params,
            random_state=self.random_state,
            tree_method='hist',
            eval_metric='auc'
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'recall': recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'average_precision': average_precision_score(y_test, y_pred_proba),
            'training_time_seconds': time.time() - start_time,
            'n_features': len(self.feature_columns),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'best_params': best_params,
            'timestamp': datetime.now().isoformat()
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
        
        # Log results
        logger.info("=" * 80)
        logger.info("FINAL MODEL PERFORMANCE METRICS")
        logger.info("=" * 80)
        logger.info(f"ROC-AUC Score:          {self.metrics['roc_auc']:.4f} (Target: 0.98)")
        logger.info(f"Recall (Fraud):         {self.metrics['recall']:.4f} (Target: 0.80)")
        logger.info(f"Precision:              {self.metrics['precision']:.4f}")
        logger.info(f"F1-Score:               {self.metrics['f1_score']:.4f}")
        logger.info(f"Average Precision:      {self.metrics['average_precision']:.4f}")
        logger.info(f"Training Time:          {self.metrics['training_time_seconds']:.2f}s")
        logger.info("-" * 80)
        logger.info("Confusion Matrix:")
        logger.info(f"  TN: {cm[0, 0]:,} | FP: {cm[0, 1]:,}")
        logger.info(f"  FN: {cm[1, 0]:,} | TP: {cm[1, 1]:,}")
        logger.info("=" * 80)
        
        # Classification report
        logger.info("\nDetailed Classification Report:")
        logger.info("\n" + classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        
        # Feature importance
        self._analyze_feature_importance(X)
        
        return self.metrics
    
    def _analyze_feature_importance(self, X: pd.DataFrame, top_n: int = 20):
        """Analyze and log top feature importances."""
        if self.model is None:
            return
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {top_n} Most Important Features:")
        logger.info("-" * 60)
        for idx, row in feature_importance_df.head(top_n).iterrows():
            logger.info(f"{row['feature']:40s} {row['importance']:.4f}")
        
        # Save to file
        feature_importance_df.to_csv('metrics/feature_importance.csv', index=False)
    
    def save_model(self, model_path: str = 'models/xgb_final.pkl'):
        """
        Save trained model and feature columns.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save feature columns
        feature_path = 'models/feature_columns.pkl'
        joblib.dump(self.feature_columns, feature_path)
        logger.info(f"Feature columns saved to {feature_path}")
        
        # Save metrics
        metrics_path = 'metrics/training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save model card
        self._save_model_card()
    
    def _save_model_card(self):
        """Generate a model card with comprehensive documentation."""
        model_card = f"""
# Fraud Detection Model Card

## Model Information
- **Model Type**: XGBoost Classifier
- **Version**: v1.2.0
- **Training Date**: {self.metrics['timestamp']}
- **Framework**: XGBoost 2.0+

## Performance Metrics
- **ROC-AUC**: {self.metrics['roc_auc']:.4f}
- **Recall**: {self.metrics['recall']:.4f}
- **Precision**: {self.metrics['precision']:.4f}
- **F1-Score**: {self.metrics['f1_score']:.4f}

## Training Data
- **Total Samples**: {self.metrics['train_samples'] + self.metrics['test_samples']:,}
- **Train Samples**: {self.metrics['train_samples']:,}
- **Test Samples**: {self.metrics['test_samples']:,}
- **Number of Features**: {self.metrics['n_features']}
- **Training Time**: {self.metrics['training_time_seconds']:.2f} seconds

## Hyperparameters
```json
{json.dumps(self.metrics['best_params'], indent=2)}
```

## Confusion Matrix
- True Negatives: {self.metrics['confusion_matrix']['tn']:,}
- False Positives: {self.metrics['confusion_matrix']['fp']:,}
- False Negatives: {self.metrics['confusion_matrix']['fn']:,}
- True Positives: {self.metrics['confusion_matrix']['tp']:,}

## Intended Use
This model is designed for real-time fraud detection in banking transactions.
It should be used as part of a comprehensive fraud prevention system.

## Limitations
- Performance may degrade on transaction patterns not seen during training
- Requires periodic retraining to adapt to evolving fraud patterns
- Feature engineering pipeline must match training exactly

## Ethical Considerations
- Model decisions should be reviewed by human analysts
- Regular audits for bias and fairness
- Transparency in fraud detection decisions
"""
        
        with open('metrics/MODEL_CARD.md', 'w') as f:
            f.write(model_card)
        logger.info("Model card saved to metrics/MODEL_CARD.md")


def main():
    """Main training pipeline execution."""
    trainer = FraudDetectionTrainer(
        data_path="data.csv",
        test_size=0.2,
        random_state=42,
        n_trials=100  # Increase for better optimization (time-intensive)
    )
    
    # Train model
    metrics = trainer.train()
    
    # Save model artifacts
    trainer.save_model()
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"✅ Model achieves {metrics['roc_auc']*100:.2f}% ROC-AUC")
    logger.info(f"✅ Fraud recall: {metrics['recall']*100:.2f}%")
    logger.info(f"✅ Model saved to models/xgb_final.pkl")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

