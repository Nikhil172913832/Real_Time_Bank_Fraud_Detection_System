"""
Unit tests for Transformer fraud detector.
"""

import pytest
import torch
import pandas as pd
import numpy as np
from src.models.transformer_fraud_detector import (
    PositionalEncoding,
    TransformerFraudDetector,
    SequenceBuilder,
)


def test_positional_encoding():
    """Test positional encoding."""
    pe = PositionalEncoding(d_model=128, max_len=100)

    x = torch.randn(1, 50, 128)
    output = pe(x)

    assert output.shape == x.shape


def test_transformer_forward():
    """Test transformer forward pass."""
    model = TransformerFraudDetector(input_dim=10, d_model=64, nhead=4, num_layers=2)

    x = torch.randn(8, 20, 10)
    output = model(x)

    assert output.shape == (8,)


def test_sequence_builder():
    """Test sequence building."""
    df = pd.DataFrame(
        {
            "card1": ["A", "A", "B", "B", "B"],
            "TransactionAmt": [100, 200, 150, 250, 300],
            "TransactionDT": [1000, 2000, 1500, 2500, 3000],
            "isFraud": [0, 0, 1, 0, 1],
        }
    )

    builder = SequenceBuilder(max_seq_len=10)
    sequences, masks, labels = builder.build_sequences(df)

    assert sequences.shape[1] == 10
    assert sequences.shape[2] == 2
    assert len(labels) == 2
