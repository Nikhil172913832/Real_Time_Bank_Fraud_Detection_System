"""
Unit tests for advanced metrics.
"""

import pytest
import numpy as np
from src.models.advanced_metrics import (
    compute_auprc,
    precision_at_k,
    recall_at_k,
    cost_sensitive_score,
    comprehensive_evaluation,
)


def test_compute_auprc():
    """Test AUPRC computation."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.9])

    auprc = compute_auprc(y_true, y_pred_proba)

    assert 0 <= auprc <= 1
    assert auprc > 0.5


def test_precision_at_k():
    """Test Precision@K."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.3, 0.85, 0.15, 0.95, 0.75])

    p_at_5 = precision_at_k(y_true, y_pred_proba, k=5)

    assert 0 <= p_at_5 <= 1
    assert p_at_5 == 1.0


def test_recall_at_k():
    """Test Recall@K."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.3, 0.85, 0.15, 0.95, 0.75])

    r_at_5 = recall_at_k(y_true, y_pred_proba, k=5)

    assert 0 <= r_at_5 <= 1

    total_frauds = np.sum(y_true)
    assert r_at_5 <= 5 / total_frauds


def test_cost_sensitive_score():
    """Test cost-sensitive scoring."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])

    cost = cost_sensitive_score(y_true, y_pred, fn_cost=10.0, fp_cost=1.0)

    assert cost >= 0
    assert cost == 10.0


def test_comprehensive_evaluation():
    """Test comprehensive evaluation."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.3, 0.85, 0.15, 0.95, 0.75])

    metrics = comprehensive_evaluation(y_true, y_pred_proba, k_values=[5, 10])

    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "precision@5" in metrics
    assert "recall@5" in metrics
    assert "cost_sensitive_score" in metrics

    assert 0 <= metrics["auroc"] <= 1
    assert 0 <= metrics["auprc"] <= 1
