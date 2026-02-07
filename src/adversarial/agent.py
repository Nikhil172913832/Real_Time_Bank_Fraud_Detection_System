"""
Adversarial fraudster agent using Q-learning.
Learns to evade fraud detection by manipulating transaction features.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List
import pickle

logger = logging.getLogger(__name__)


class FraudsterAgent:
    """
    Q-learning agent that learns to generate fraudulent transactions
    that evade the fraud detection model.

    State: Discretized transaction features
    Action: Feature manipulations (amount, timing, etc.)
    Reward: +1 if fraud not detected, -1 if detected
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

        self.actions = [
            "increase_amount",
            "decrease_amount",
            "change_time_early",
            "change_time_late",
            "split_transaction",
        ]

    def get_state(self, transaction: pd.Series) -> str:
        """Convert transaction to discrete state."""
        amount_bin = pd.cut([transaction["TransactionAmt"]], bins=10, labels=False)[0]
        hour_bin = int(transaction.get("hour", 12) // 6)

        state = f"amt_{amount_bin}_hr_{hour_bin}"
        return state

    def choose_action(self, state: str) -> str:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)

        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

        return max(self.q_table[state], key=self.q_table[state].get)

    def apply_action(self, transaction: pd.Series, action: str) -> pd.Series:
        """Apply action to modify transaction."""
        txn = transaction.copy()

        if action == "increase_amount":
            txn["TransactionAmt"] *= np.random.uniform(1.1, 1.5)
        elif action == "decrease_amount":
            txn["TransactionAmt"] *= np.random.uniform(0.5, 0.9)
        elif action == "change_time_early":
            if "hour" in txn:
                txn["hour"] = np.random.randint(6, 12)
        elif action == "change_time_late":
            if "hour" in txn:
                txn["hour"] = np.random.randint(18, 23)
        elif action == "split_transaction":
            txn["TransactionAmt"] /= np.random.randint(2, 4)

        return txn

    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Q-learning update rule."""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}

        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())

        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)

        self.q_table[state][action] = new_q

    def train_episode(
        self, fraud_detector, feature_engineer, n_transactions: int = 100
    ) -> Dict:
        """Train agent for one episode."""
        successes = 0
        total_reward = 0

        for _ in range(n_transactions):
            txn = self._generate_base_fraud()
            state = self.get_state(txn)

            action = self.choose_action(state)
            modified_txn = self.apply_action(txn, action)
            next_state = self.get_state(modified_txn)

            is_detected = self._predict_fraud(
                modified_txn, fraud_detector, feature_engineer
            )
            reward = -1 if is_detected else +1

            self.update_q_value(state, action, reward, next_state)

            if not is_detected:
                successes += 1
            total_reward += reward

        evasion_rate = successes / n_transactions

        return {
            "evasion_rate": evasion_rate,
            "total_reward": total_reward,
            "q_table_size": len(self.q_table),
        }

    def _generate_base_fraud(self) -> pd.Series:
        """Generate a base fraudulent transaction."""
        return pd.Series(
            {
                "TransactionAmt": np.random.uniform(100, 1000),
                "hour": np.random.randint(0, 24),
            }
        )

    def _predict_fraud(self, transaction: pd.Series, model, feature_engineer) -> bool:
        """Predict if transaction is detected as fraud."""
        df = pd.DataFrame([transaction])
        df_features = feature_engineer.transform(df)
        X = feature_engineer.prepare_for_model(df_features)

        pred_proba = model.predict_proba(X)[0, 1]
        return pred_proba > 0.5

    def save(self, path: str):
        """Save agent state."""
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        logger.info(f"Agent saved to {path}")

    def load(self, path: str):
        """Load agent state."""
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        logger.info(f"Agent loaded from {path}")
