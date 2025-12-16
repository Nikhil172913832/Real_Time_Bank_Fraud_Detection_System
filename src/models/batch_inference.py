"""
Batch Inference Pipeline
========================

Offline scoring and historical analysis for fraud detection.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class BatchInferenceEngine:
    """
    Batch inference engine for offline fraud detection.
    
    Features:
    - Process large datasets efficiently
    - Generate fraud scores for historical data
    - Export results to various formats
    - Performance metrics and reporting
    """
    
    def __init__(
        self,
        model_path: str = "models/xgb_final.pkl",
        feature_columns_path: str = "models/feature_columns.pkl",
        batch_size: int = 10000
    ):
        """
        Initialize batch inference engine.
        
        Args:
            model_path: Path to trained model
            feature_columns_path: Path to feature columns
            batch_size: Number of records to process per batch
        """
        self.model_path = model_path
        self.feature_columns_path = feature_columns_path
        self.batch_size = batch_size
        
        # Load model and feature columns
        self.model = joblib.load(model_path)
        self.feature_columns = joblib.load(feature_columns_path)
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Feature columns: {len(self.feature_columns)}")
    
    def process_file(
        self,
        input_path: str,
        output_path: str,
        threshold: float = 0.5,
        include_explanations: bool = False
    ) -> Dict[str, Any]:
        """
        Process transactions from file and save results.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output CSV file
            threshold: Fraud probability threshold
            include_explanations: Whether to include SHAP explanations
            
        Returns:
            Processing statistics
        """
        logger.info(f"Processing file: {input_path}")
        start_time = datetime.now()
        
        # Read input file
        df = pd.read_csv(input_path)
        total_records = len(df)
        logger.info(f"Loaded {total_records:,} records")
        
        # Process in batches
        results = []
        fraud_count = 0
        
        for i in range(0, total_records, self.batch_size):
            batch = df.iloc[i:i + self.batch_size]
            batch_results = self._process_batch(batch, threshold, include_explanations)
            results.append(batch_results)
            fraud_count += batch_results['is_fraud'].sum()
            
            if (i + self.batch_size) % 50000 == 0:
                logger.info(f"Processed {i + self.batch_size:,} / {total_records:,} records")
        
        # Combine results
        results_df = pd.concat(results, ignore_index=True)
        
        # Save to file
        results_df.to_csv(output_path, index=False)
        
        # Calculate statistics
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        stats = {
            "total_records": total_records,
            "fraud_detected": int(fraud_count),
            "fraud_rate": fraud_count / total_records,
            "processing_time_seconds": elapsed_time,
            "records_per_second": total_records / elapsed_time,
            "output_file": output_path
        }
        
        logger.info(f"Processing complete:")
        logger.info(f"  - Total records: {total_records:,}")
        logger.info(f"  - Fraud detected: {fraud_count:,} ({stats['fraud_rate']:.2%})")
        logger.info(f"  - Processing time: {elapsed_time:.2f}s")
        logger.info(f"  - Throughput: {stats['records_per_second']:.0f} records/s")
        
        return stats
    
    def _process_batch(
        self,
        batch: pd.DataFrame,
        threshold: float,
        include_explanations: bool
    ) -> pd.DataFrame:
        """Process a single batch of transactions."""
        from src.features.engineering import FeatureEngineer
        
        # Feature engineering
        engineer = FeatureEngineer(validate_schema=False)
        features = engineer.transform(batch.copy())
        X = engineer.prepare_for_model(features, encode_categoricals=True)
        
        # Align columns
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns]
        
        # Predict
        fraud_probs = self.model.predict_proba(X)[:, 1]
        predictions = (fraud_probs > threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'transaction_id': batch['transaction_id'],
            'fraud_probability': fraud_probs,
            'is_fraud': predictions,
            'threshold': threshold
        })
        
        # Add SHAP explanations if requested
        if include_explanations:
            try:
                import shap
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Get top 3 features for each prediction
                top_features = []
                for i in range(len(shap_values)):
                    feature_impacts = dict(zip(self.feature_columns, shap_values[i]))
                    top_3 = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    top_features.append(str(dict(top_3)))
                
                results['top_features'] = top_features
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}")
        
        return results
    
    def analyze_results(self, results_path: str) -> Dict[str, Any]:
        """
        Analyze batch inference results.
        
        Args:
            results_path: Path to results CSV
            
        Returns:
            Analysis report
        """
        df = pd.read_csv(results_path)
        
        analysis = {
            "total_transactions": len(df),
            "fraud_detected": int(df['is_fraud'].sum()),
            "fraud_rate": df['is_fraud'].mean(),
            "avg_fraud_probability": df['fraud_probability'].mean(),
            "probability_distribution": {
                "min": float(df['fraud_probability'].min()),
                "q25": float(df['fraud_probability'].quantile(0.25)),
                "median": float(df['fraud_probability'].median()),
                "q75": float(df['fraud_probability'].quantile(0.75)),
                "max": float(df['fraud_probability'].max())
            }
        }
        
        logger.info("Analysis Results:")
        logger.info(f"  - Total: {analysis['total_transactions']:,}")
        logger.info(f"  - Fraud: {analysis['fraud_detected']:,} ({analysis['fraud_rate']:.2%})")
        logger.info(f"  - Avg Probability: {analysis['avg_fraud_probability']:.4f}")
        
        return analysis


def main():
    """Example usage of batch inference engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Fraud Detection")
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Fraud threshold')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size')
    parser.add_argument('--explain', action='store_true', help='Include SHAP explanations')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = BatchInferenceEngine(batch_size=args.batch_size)
    
    # Process file
    stats = engine.process_file(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        include_explanations=args.explain
    )
    
    # Analyze results
    analysis = engine.analyze_results(args.output)
    
    print("\n" + "="*80)
    print("BATCH INFERENCE COMPLETE")
    print("="*80)
    print(f"Processed: {stats['total_records']:,} records")
    print(f"Fraud Detected: {stats['fraud_detected']:,} ({analysis['fraud_rate']:.2%})")
    print(f"Throughput: {stats['records_per_second']:.0f} records/second")
    print(f"Output: {stats['output_file']}")


if __name__ == '__main__':
    main()
