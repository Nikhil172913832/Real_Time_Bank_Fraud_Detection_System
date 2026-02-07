"""
Unit tests for GNN fraud detector.
"""

import pytest
import torch
import pandas as pd
import numpy as np
from src.models.gnn_fraud_detector import (
    HeterogeneousGNN,
    GraphBuilder,
    GNNFraudDetector,
)


def test_graph_builder():
    """Test graph construction from transactions."""
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionAmt": [100.0, 200.0, 300.0],
            "TransactionDT": [1000, 2000, 3000],
            "card1": ["A", "B", "A"],
            "ProductCD": ["W", "W", "C"],
            "DeviceInfo": ["D1", "D2", "D1"],
            "isFraud": [0, 1, 0],
        }
    )

    builder = GraphBuilder()
    graph = builder.build_graph(df)

    assert "transaction" in graph.node_types
    assert graph["transaction"].x.shape[0] == 3

    assert len(builder.user_mapping) == 2
    assert len(builder.merchant_mapping) == 2


def test_heterogeneous_gnn_forward():
    """Test GNN forward pass."""
    model = HeterogeneousGNN(hidden_dim=64, num_layers=2)

    x_dict = {
        "user": torch.randn(5, 64),
        "transaction": torch.randn(10, 64),
        "merchant": torch.randn(3, 64),
        "device": torch.randn(4, 64),
    }

    edge_index_dict = {
        ("user", "makes", "transaction"): torch.randint(0, 5, (2, 15)),
        ("transaction", "to", "merchant"): torch.randint(0, 3, (2, 10)),
        ("transaction", "uses", "device"): torch.randint(0, 4, (2, 8)),
    }

    output = model(x_dict, edge_index_dict)

    assert output.shape == (10,)


def test_gnn_fraud_detector_fit():
    """Test GNN training."""
    df = pd.DataFrame(
        {
            "TransactionID": range(20),
            "TransactionAmt": np.random.uniform(50, 500, 20),
            "TransactionDT": range(1000, 1020),
            "card1": np.random.choice(["A", "B", "C"], 20),
            "ProductCD": np.random.choice(["W", "C"], 20),
            "DeviceInfo": np.random.choice(["D1", "D2"], 20),
            "isFraud": np.random.randint(0, 2, 20),
        }
    )

    labels = df["isFraud"].values

    detector = GNNFraudDetector(hidden_dim=32, num_layers=1)
    detector.fit(df, labels, epochs=2)

    assert detector.model is not None
