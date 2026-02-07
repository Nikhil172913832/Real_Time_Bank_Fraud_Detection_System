"""
Unit tests for adversarial agent.
"""

import pytest
import pandas as pd
import numpy as np
from src.adversarial.agent import FraudsterAgent


def test_agent_initialization():
    """Test agent initializes correctly."""
    agent = FraudsterAgent()

    assert agent.lr == 0.1
    assert agent.gamma == 0.95
    assert agent.epsilon == 0.1
    assert len(agent.actions) > 0
    assert len(agent.q_table) == 0


def test_get_state():
    """Test state discretization."""
    agent = FraudsterAgent()

    txn = pd.Series({"TransactionAmt": 500, "hour": 14})

    state = agent.get_state(txn)
    assert isinstance(state, str)
    assert "amt" in state
    assert "hr" in state


def test_choose_action():
    """Test action selection."""
    agent = FraudsterAgent()

    state = "amt_5_hr_2"
    action = agent.choose_action(state)

    assert action in agent.actions


def test_apply_action():
    """Test action application modifies transaction."""
    agent = FraudsterAgent()

    txn = pd.Series({"TransactionAmt": 100, "hour": 12})

    modified = agent.apply_action(txn, "increase_amount")
    assert modified["TransactionAmt"] > txn["TransactionAmt"]

    modified = agent.apply_action(txn, "decrease_amount")
    assert modified["TransactionAmt"] < txn["TransactionAmt"]


def test_q_value_update():
    """Test Q-learning update."""
    agent = FraudsterAgent()

    state = "state1"
    next_state = "state2"
    action = agent.actions[0]
    reward = 1.0

    agent.update_q_value(state, action, reward, next_state)

    assert state in agent.q_table
    assert action in agent.q_table[state]
