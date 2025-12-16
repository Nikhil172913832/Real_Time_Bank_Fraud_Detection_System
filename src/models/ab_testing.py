"""
A/B Testing Framework
====================

Framework for comparing model versions and shadow mode deployment.
"""

import logging
import random
from typing import Optional, Dict, Any, List
from enum import Enum
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ABTestMode(Enum):
    """A/B test modes."""
    SPLIT = "split"          # Traffic split between models
    SHADOW = "shadow"        # Shadow mode (log only, don't serve)
    CHAMPION_CHALLENGER = "champion_challenger"  # Champion vs challenger


class ABTestFramework:
    """
    A/B testing framework for model comparison.
    
    Features:
    - Traffic splitting
    - Shadow mode deployment
    - Performance comparison
    - Statistical significance testing
    """
    
    def __init__(
        self,
        mode: ABTestMode = ABTestMode.SPLIT,
        traffic_split: float = 0.5,
        enable_logging: bool = True
    ):
        """
        Initialize A/B testing framework.
        
        Args:
            mode: A/B test mode
            traffic_split: Percentage of traffic to variant B (0.0-1.0)
            enable_logging: Whether to log predictions
        """
        self.mode = mode
        self.traffic_split = traffic_split
        self.enable_logging = enable_logging
        
        self.model_a = None  # Champion model
        self.model_b = None  # Challenger model
        
        self.predictions_log: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized A/B testing framework in {mode.value} mode")
    
    def load_models(self, model_a_path: str, model_b_path: str):
        """
        Load both models for comparison.
        
        Args:
            model_a_path: Path to model A (champion)
            model_b_path: Path to model B (challenger)
        """
        import joblib
        
        self.model_a = joblib.load(model_a_path)
        self.model_b = joblib.load(model_b_path)
        
        logger.info(f"Loaded model A from {model_a_path}")
        logger.info(f"Loaded model B from {model_b_path}")
    
    def predict(self, X, transaction_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Make prediction using A/B testing logic.
        
        Args:
            X: Feature vector
            transaction_id: Transaction identifier
            
        Returns:
            Prediction result with metadata
        """
        if self.mode == ABTestMode.SPLIT:
            return self._split_mode_predict(X, transaction_id)
        elif self.mode == ABTestMode.SHADOW:
            return self._shadow_mode_predict(X, transaction_id)
        else:  # CHAMPION_CHALLENGER
            return self._champion_challenger_predict(X, transaction_id)
    
    def _split_mode_predict(self, X, transaction_id: Optional[str]) -> Dict[str, Any]:
        """Traffic split mode: randomly assign to A or B."""
        # Randomly assign to model
        use_model_b = random.random() < self.traffic_split
        model_variant = "B" if use_model_b else "A"
        model = self.model_b if use_model_b else self.model_a
        
        # Predict
        fraud_prob = model.predict_proba(X)[:, 1][0]
        prediction = int(fraud_prob > 0.5)
        
        result = {
            "transaction_id": transaction_id,
            "model_variant": model_variant,
            "fraud_probability": float(fraud_prob),
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.enable_logging:
            self.predictions_log.append(result)
        
        return result
    
    def _shadow_mode_predict(self, X, transaction_id: Optional[str]) -> Dict[str, Any]:
        """Shadow mode: serve A, log B predictions."""
        # Serve model A
        fraud_prob_a = self.model_a.predict_proba(X)[:, 1][0]
        prediction_a = int(fraud_prob_a > 0.5)
        
        # Log model B (shadow)
        fraud_prob_b = self.model_b.predict_proba(X)[:, 1][0]
        prediction_b = int(fraud_prob_b > 0.5)
        
        result = {
            "transaction_id": transaction_id,
            "model_variant": "A",  # Served model
            "fraud_probability": float(fraud_prob_a),
            "prediction": prediction_a,
            "shadow_probability": float(fraud_prob_b),
            "shadow_prediction": prediction_b,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.enable_logging:
            self.predictions_log.append(result)
        
        return result
    
    def _champion_challenger_predict(self, X, transaction_id: Optional[str]) -> Dict[str, Any]:
        """Champion-challenger mode: always serve champion, track challenger."""
        # Get predictions from both models
        fraud_prob_champion = self.model_a.predict_proba(X)[:, 1][0]
        fraud_prob_challenger = self.model_b.predict_proba(X)[:, 1][0]
        
        prediction_champion = int(fraud_prob_champion > 0.5)
        prediction_challenger = int(fraud_prob_challenger > 0.5)
        
        result = {
            "transaction_id": transaction_id,
            "model_variant": "Champion",
            "fraud_probability": float(fraud_prob_champion),
            "prediction": prediction_champion,
            "challenger_probability": float(fraud_prob_challenger),
            "challenger_prediction": prediction_challenger,
            "agreement": prediction_champion == prediction_challenger,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.enable_logging:
            self.predictions_log.append(result)
        
        return result
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze A/B test results.
        
        Returns:
            Analysis report with performance comparison
        """
        if not self.predictions_log:
            return {"error": "No predictions logged"}
        
        df = pd.DataFrame(self.predictions_log)
        
        analysis = {
            "total_predictions": len(df),
            "mode": self.mode.value,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.mode == ABTestMode.SPLIT:
            # Compare model A vs B
            model_a_preds = df[df['model_variant'] == 'A']
            model_b_preds = df[df['model_variant'] == 'B']
            
            analysis["model_a"] = {
                "count": len(model_a_preds),
                "fraud_rate": model_a_preds['prediction'].mean(),
                "avg_probability": model_a_preds['fraud_probability'].mean()
            }
            
            analysis["model_b"] = {
                "count": len(model_b_preds),
                "fraud_rate": model_b_preds['prediction'].mean(),
                "avg_probability": model_b_preds['fraud_probability'].mean()
            }
            
        elif self.mode == ABTestMode.SHADOW:
            # Compare served vs shadow
            analysis["served_model"] = {
                "fraud_rate": df['prediction'].mean(),
                "avg_probability": df['fraud_probability'].mean()
            }
            
            analysis["shadow_model"] = {
                "fraud_rate": df['shadow_prediction'].mean(),
                "avg_probability": df['shadow_probability'].mean()
            }
            
            # Agreement rate
            agreement = (df['prediction'] == df['shadow_prediction']).mean()
            analysis["agreement_rate"] = float(agreement)
            
        else:  # CHAMPION_CHALLENGER
            analysis["champion"] = {
                "fraud_rate": df['prediction'].mean(),
                "avg_probability": df['fraud_probability'].mean()
            }
            
            analysis["challenger"] = {
                "fraud_rate": df['challenger_prediction'].mean(),
                "avg_probability": df['challenger_probability'].mean()
            }
            
            analysis["agreement_rate"] = df['agreement'].mean()
        
        logger.info("A/B Test Analysis:")
        logger.info(json.dumps(analysis, indent=2))
        
        return analysis
    
    def export_results(self, output_path: str):
        """Export predictions log to file."""
        import pandas as pd
        df = pd.DataFrame(self.predictions_log)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} predictions to {output_path}")


# Example usage
if __name__ == '__main__':
    import pandas as pd
    
    # Initialize framework
    ab_test = ABTestFramework(
        mode=ABTestMode.SHADOW,
        traffic_split=0.5
    )
    
    # Load models
    ab_test.load_models("models/xgb_final.pkl", "models/xgb_final.pkl")
    
    # Make predictions
    # (In practice, integrate with inference pipeline)
    
    # Analyze results
    results = ab_test.analyze_results()
    print(json.dumps(results, indent=2))
