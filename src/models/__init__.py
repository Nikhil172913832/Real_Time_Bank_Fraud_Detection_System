"""
Models module for fraud detection.

Provides unified interface for all model types:
- Traditional ML: XGBoost, Ensemble
- Deep Learning: GNN, Transformer, Hybrid
- Adversarial: Q-learning agent
"""

from src.models.ensemble import EnsembleFraudDetector
from src.models.validation import temporal_train_test_split

try:
    from src.models.gnn_fraud_detector import GNNFraudDetector
    from src.models.transformer_fraud_detector import TransformerFraudDetectorWrapper
    from src.models.hybrid_detector import HybridFraudDetectorWrapper
    from src.models.advanced_metrics import (
        compute_auprc,
        precision_at_k,
        recall_at_k,
        comprehensive_evaluation,
        print_evaluation_report,
    )

    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

__all__ = [
    "EnsembleFraudDetector",
    "temporal_train_test_split",
    "DL_AVAILABLE",
]

if DL_AVAILABLE:
    __all__.extend(
        [
            "GNNFraudDetector",
            "TransformerFraudDetectorWrapper",
            "HybridFraudDetectorWrapper",
            "compute_auprc",
            "precision_at_k",
            "recall_at_k",
            "comprehensive_evaluation",
            "print_evaluation_report",
        ]
    )
