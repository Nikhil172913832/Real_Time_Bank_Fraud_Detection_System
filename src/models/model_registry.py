"""
Model Registry and Versioning with MLflow
=========================================

This module provides model versioning, A/B testing, and experiment tracking
using MLflow for the fraud detection system.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Install with: pip install mlflow")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manage model versions, experiments, and A/B testing with MLflow.
    
    Features:
    - Model versioning and tracking
    - Experiment management
    - A/B testing framework
    - Model comparison and rollback
    - Performance tracking over time
    """
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "fraud-detection"
    ):
        """
        Initialize the model registry.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required. Install with: pip install mlflow")
        
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set up MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        self.client = MlflowClient(tracking_uri=tracking_uri)
        logger.info(f"Model registry initialized with experiment: {experiment_name}")
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        artifacts: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Log a trained model to MLflow with metrics and parameters.
        
        Args:
            model: Trained model object
            model_name: Name for the model
            metrics: Dictionary of metric names and values
            params: Dictionary of hyperparameters
            artifacts: Dictionary of artifact paths to log
            tags: Dictionary of tags for the model
            
        Returns:
            Run ID of the logged model
        """
        with mlflow.start_run(run_name=model_name) as run:
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            try:
                if hasattr(model, '__class__') and 'XGBoost' in model.__class__.__name__:
                    mlflow.xgboost.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")
            except Exception as e:
                logger.warning(f"Failed to log model with mlflow backend: {e}")
                # Fallback to generic logging
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                    joblib.dump(model, f.name)
                    mlflow.log_artifact(f.name, "model")
                    os.unlink(f.name)
            
            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    if os.path.exists(path):
                        mlflow.log_artifact(path, name)
            
            # Set tags
            if tags:
                mlflow.set_tags(tags)
            
            # Add default tags
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("timestamp", datetime.now().isoformat())
            
            run_id = run.info.run_id
            logger.info(f"Model logged with run_id: {run_id}")
            
        return run_id
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        stage: str = "None"
    ) -> str:
        """
        Register a model version in the model registry.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for registered model
            stage: Stage to transition to ('None', 'Staging', 'Production', 'Archived')
            
        Returns:
            Model version
        """
        model_uri = f"runs:/{run_id}/model"
        
        try:
            model_details = mlflow.register_model(model_uri, model_name)
            version = model_details.version
            
            # Transition to stage
            if stage != "None":
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage
                )
            
            logger.info(f"Model registered: {model_name} v{version} (stage: {stage})")
            return version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Any:
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the registered model
            version: Specific version to load (optional)
            stage: Stage to load from ('Staging', 'Production', etc.)
            
        Returns:
            Loaded model object
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}/Production"
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def compare_models(
        self,
        model_versions: List[Tuple[str, str]],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compare multiple model versions on test data.
        
        Args:
            model_versions: List of (model_name, version) tuples
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with comparison metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        results = []
        
        for model_name, version in model_versions:
            try:
                model = self.load_model(model_name, version=version)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'model_name': model_name,
                    'version': version,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                results.append(metrics)
                logger.info(f"Evaluated {model_name} v{version}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name} v{version}: {e}")
        
        comparison_df = pd.DataFrame(results)
        return comparison_df
    
    def run_ab_test(
        self,
        model_a: Tuple[str, str],
        model_b: Tuple[str, str],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        traffic_split: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run A/B test between two models.
        
        Args:
            model_a: (model_name, version) for model A
            model_b: (model_name, version) for model B
            X_test: Test features
            y_test: Test labels
            traffic_split: Fraction of traffic for model A (0-1)
            
        Returns:
            Dictionary with A/B test results
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Split test data
        n_samples = len(X_test)
        n_a = int(n_samples * traffic_split)
        
        indices = np.random.permutation(n_samples)
        indices_a = indices[:n_a]
        indices_b = indices[n_a:]
        
        X_a, y_a = X_test.iloc[indices_a], y_test.iloc[indices_a]
        X_b, y_b = X_test.iloc[indices_b], y_test.iloc[indices_b]
        
        # Load models
        model_a_obj = self.load_model(model_a[0], version=model_a[1])
        model_b_obj = self.load_model(model_b[0], version=model_b[1])
        
        # Predictions
        y_pred_a = model_a_obj.predict(X_a)
        y_pred_b = model_b_obj.predict(X_b)
        
        # Calculate metrics
        results = {
            'model_a': {
                'name': f"{model_a[0]} v{model_a[1]}",
                'traffic': traffic_split,
                'samples': len(y_a),
                'accuracy': accuracy_score(y_a, y_pred_a),
                'precision': precision_score(y_a, y_pred_a),
                'recall': recall_score(y_a, y_pred_a),
                'f1_score': f1_score(y_a, y_pred_a)
            },
            'model_b': {
                'name': f"{model_b[0]} v{model_b[1]}",
                'traffic': 1 - traffic_split,
                'samples': len(y_b),
                'accuracy': accuracy_score(y_b, y_pred_b),
                'precision': precision_score(y_b, y_pred_b),
                'recall': recall_score(y_b, y_pred_b),
                'f1_score': f1_score(y_b, y_pred_b)
            }
        }
        
        # Determine winner
        if results['model_a']['f1_score'] > results['model_b']['f1_score']:
            results['winner'] = 'model_a'
            results['improvement'] = (
                (results['model_a']['f1_score'] - results['model_b']['f1_score']) 
                / results['model_b']['f1_score'] * 100
            )
        else:
            results['winner'] = 'model_b'
            results['improvement'] = (
                (results['model_b']['f1_score'] - results['model_a']['f1_score']) 
                / results['model_a']['f1_score'] * 100
            )
        
        logger.info(f"A/B test complete. Winner: {results['winner']}")
        return results
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        from_stage: str = "Staging",
        to_stage: str = "Production"
    ) -> None:
        """
        Promote a model from one stage to another.
        
        Args:
            model_name: Name of the registered model
            version: Version to promote
            from_stage: Current stage
            to_stage: Target stage
        """
        try:
            # Archive current production model
            if to_stage == "Production":
                current_production = self.client.get_latest_versions(
                    model_name,
                    stages=["Production"]
                )
                for model_version in current_production:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=model_version.version,
                        stage="Archived"
                    )
            
            # Promote new version
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=to_stage
            )
            
            logger.info(f"Promoted {model_name} v{version} to {to_stage}")
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
    
    def rollback_model(
        self,
        model_name: str,
        target_version: str
    ) -> None:
        """
        Rollback to a previous model version.
        
        Args:
            model_name: Name of the registered model
            target_version: Version to rollback to
        """
        self.promote_model(
            model_name,
            target_version,
            from_stage="Archived",
            to_stage="Production"
        )
        logger.info(f"Rolled back {model_name} to v{target_version}")
    
    def get_model_history(
        self,
        model_name: str
    ) -> pd.DataFrame:
        """
        Get version history for a registered model.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            DataFrame with version history
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            history = []
            for version in versions:
                history.append({
                    'version': version.version,
                    'stage': version.current_stage,
                    'run_id': version.run_id,
                    'created_at': datetime.fromtimestamp(version.creation_timestamp / 1000),
                    'updated_at': datetime.fromtimestamp(version.last_updated_timestamp / 1000)
                })
            
            history_df = pd.DataFrame(history).sort_values('version', ascending=False)
            return history_df
            
        except Exception as e:
            logger.error(f"Failed to get model history: {e}")
            return pd.DataFrame()


def example_usage():
    """Example usage of ModelRegistry."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.randint(0, 2, 1000))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize registry
    registry = ModelRegistry(
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="fraud-detection-demo"
    )
    
    # Train and log model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    metrics = {
        'accuracy': model.score(X_test, y_test),
        'n_estimators': 100
    }
    params = {'n_estimators': 100, 'random_state': 42}
    
    run_id = registry.log_model(
        model=model,
        model_name="fraud_detector_rf",
        metrics=metrics,
        params=params
    )
    
    # Register model
    version = registry.register_model(run_id, "fraud_detector", stage="Staging")
    print(f"Registered model version: {version}")
    
    # Promote to production
    registry.promote_model("fraud_detector", version, to_stage="Production")
    
    # Get history
    history = registry.get_model_history("fraud_detector")
    print(f"\nModel History:\n{history}")


if __name__ == "__main__":
    if MLFLOW_AVAILABLE:
        example_usage()
    else:
        print("MLflow not available. Please install: pip install mlflow")
