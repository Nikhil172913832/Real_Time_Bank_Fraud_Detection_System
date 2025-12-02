"""
Enhanced Fraud Detection with Ensemble Models and SHAP Explanations
===================================================================

This module provides an advanced fraud detection system with:
- Ensemble learning (XGBoost, LightGBM, Random Forest)
- SHAP explainability for model interpretability
- Model stacking and voting strategies
- Feature importance analysis
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with: pip install lightgbm")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedFraudDetector:
    """
    Advanced fraud detection with ensemble methods and explainability.
    
    Features:
    - Multiple model ensemble (XGBoost, LightGBM, Random Forest)
    - SHAP-based explanations
    - Confidence calibration
    - Feature importance tracking
    """
    
    def __init__(
        self,
        model_type: str = "ensemble",
        fraud_threshold: float = 0.2,
        enable_shap: bool = True
    ):
        """
        Initialize the enhanced fraud detector.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'rf', 'ensemble', 'stacking')
            fraud_threshold: Threshold for fraud classification
            enable_shap: Whether to enable SHAP explanations
        """
        self.model_type = model_type
        self.fraud_threshold = fraud_threshold
        self.enable_shap = enable_shap
        self.model: Optional[Any] = None
        self.explainer: Optional[shap.Explainer] = None
        self.feature_names: Optional[List[str]] = None
        
    def build_ensemble(self) -> VotingClassifier:
        """
        Build a voting ensemble of multiple models.
        
        Returns:
            VotingClassifier with soft voting
        """
        estimators = [
            ('xgb', XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ))
        ]
        
        if LIGHTGBM_AVAILABLE:
            estimators.append(('lgb', lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )))
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        logger.info(f"Built voting ensemble with {len(estimators)} models")
        return ensemble
    
    def build_stacking(self) -> StackingClassifier:
        """
        Build a stacking ensemble with meta-learner.
        
        Returns:
            StackingClassifier with logistic regression meta-learner
        """
        base_estimators = [
            ('xgb', XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ))
        ]
        
        if LIGHTGBM_AVAILABLE:
            base_estimators.append(('lgb', lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42,
                verbose=-1
            )))
        
        stacking = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            n_jobs=-1
        )
        
        logger.info("Built stacking ensemble with logistic regression meta-learner")
        return stacking
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train the fraud detection model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training metrics dictionary
        """
        self.feature_names = list(X_train.columns)
        
        # Build model based on type
        if self.model_type == "ensemble":
            self.model = self.build_ensemble()
        elif self.model_type == "stacking":
            self.model = self.build_stacking()
        elif self.model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        elif self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        elif self.model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
        logger.info(f"Training {self.model_type} model...")
        
        if X_val is not None and y_val is not None:
            # Train with validation set if provided
            if hasattr(self.model, 'fit'):
                eval_set = [(X_val, y_val)]
                if self.model_type in ["xgboost", "lightgbm"]:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        verbose=False
                    )
                else:
                    self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        # Initialize SHAP explainer
        if self.enable_shap:
            self._initialize_shap_explainer(X_train)
        
        # Calculate training metrics
        train_score = self.model.score(X_train, y_train)
        metrics = {"train_accuracy": train_score}
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            metrics["val_accuracy"] = val_score
        
        logger.info(f"Training complete. Metrics: {metrics}")
        return metrics
    
    def _initialize_shap_explainer(self, X_sample: pd.DataFrame) -> None:
        """Initialize SHAP explainer for model interpretability."""
        try:
            if self.model_type in ["xgboost", "lightgbm", "rf"]:
                # Tree explainer for tree-based models
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Use sampling for ensemble models
                sample_data = shap.sample(X_sample, min(100, len(X_sample)))
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    sample_data
                )
            logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None
    
    def predict_with_explanation(
        self,
        transaction: pd.DataFrame,
        top_n_features: int = 10
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Predict fraud probability with SHAP explanation.
        
        Args:
            transaction: Single transaction as DataFrame
            top_n_features: Number of top features to return in explanation
            
        Returns:
            Tuple of (fraud_probability, is_fraud, explanation_dict)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get prediction
        fraud_prob = float(self.model.predict_proba(transaction)[0, 1])
        is_fraud = fraud_prob > self.fraud_threshold
        
        # Generate explanation
        explanation = {
            "fraud_probability": fraud_prob,
            "is_fraud": is_fraud,
            "threshold": self.fraud_threshold,
            "model_type": self.model_type
        }
        
        if self.enable_shap and self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(transaction)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Get positive class
                
                # Get feature contributions
                feature_contributions = []
                for i, feature in enumerate(self.feature_names):
                    contribution = float(shap_values[0][i]) if len(shap_values.shape) > 1 else float(shap_values[i])
                    feature_contributions.append({
                        "feature": feature,
                        "value": float(transaction[feature].values[0]),
                        "shap_value": contribution,
                        "impact": "increases" if contribution > 0 else "decreases"
                    })
                
                # Sort by absolute SHAP value
                feature_contributions.sort(
                    key=lambda x: abs(x["shap_value"]),
                    reverse=True
                )
                
                explanation["top_features"] = feature_contributions[:top_n_features]
                explanation["base_value"] = float(self.explainer.expected_value)
                
            except Exception as e:
                logger.warning(f"Failed to generate SHAP explanation: {e}")
                explanation["explanation_error"] = str(e)
        
        return fraud_prob, is_fraud, explanation
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract feature importance based on model type
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif self.model_type in ["ensemble", "stacking"]:
            # Average importance across base estimators
            importances = np.zeros(len(self.feature_names))
            count = 0
            for name, estimator in self.model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    importances += estimator.feature_importances_
                    count += 1
            if count > 0:
                importances /= count
        else:
            logger.warning("Feature importance not available for this model")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save(self, filepath: str) -> None:
        """Save model and explainer to disk."""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'fraud_threshold': self.fraud_threshold,
            'feature_names': self.feature_names,
            'enable_shap': self.enable_shap
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EnhancedFraudDetector':
        """Load model from disk."""
        model_data = joblib.load(filepath)
        
        detector = cls(
            model_type=model_data['model_type'],
            fraud_threshold=model_data['fraud_threshold'],
            enable_shap=model_data['enable_shap']
        )
        detector.model = model_data['model']
        detector.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {filepath}")
        return detector


def example_usage():
    """Example usage of EnhancedFraudDetector."""
    # Create sample data
    X_train = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    y_train = pd.Series(np.random.randint(0, 2, 1000))
    
    # Initialize and train detector
    detector = EnhancedFraudDetector(model_type="ensemble", enable_shap=True)
    metrics = detector.train(X_train, y_train)
    print(f"Training metrics: {metrics}")
    
    # Make prediction with explanation
    transaction = X_train.iloc[:1]
    fraud_prob, is_fraud, explanation = detector.predict_with_explanation(transaction)
    
    print(f"\nFraud Probability: {fraud_prob:.4f}")
    print(f"Is Fraud: {is_fraud}")
    print(f"\nTop Features:")
    for feat in explanation.get('top_features', [])[:5]:
        print(f"  {feat['feature']}: {feat['shap_value']:.4f} ({feat['impact']} risk)")
    
    # Get feature importance
    importance = detector.get_feature_importance(top_n=10)
    print(f"\nFeature Importance:\n{importance}")


if __name__ == "__main__":
    example_usage()
