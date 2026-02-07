"""
Adversarial training loop.
Train fraud detector against adversarial fraudster agent.
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from src.adversarial.agent import FraudsterAgent
from src.features.engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def adversarial_training(
    initial_model_path: str = "models/fraud_model.pkl",
    feature_engineer_path: str = "models/feature_engineer.pkl",
    n_rounds: int = 5,
    n_episodes_per_round: int = 3,
):
    """
    Adversarial training loop:
    1. Fraudster learns to evade current model
    2. Collect adversarial examples
    3. Retrain model on adversarial examples
    4. Repeat
    """

    logger.info("Loading initial model and feature engineer")
    with open(initial_model_path, "rb") as f:
        model = pickle.load(f)

    with open(feature_engineer_path, "rb") as f:
        feature_engineer = pickle.load(f)

    agent = FraudsterAgent()

    history = {"round": [], "evasion_rate": [], "q_table_size": []}

    for round_num in range(n_rounds):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Round {round_num + 1}/{n_rounds}")
        logger.info(f"{'=' * 60}")

        round_evasion_rates = []

        for episode in range(n_episodes_per_round):
            metrics = agent.train_episode(model, feature_engineer, n_transactions=100)
            round_evasion_rates.append(metrics["evasion_rate"])
            logger.info(
                f"Episode {episode + 1}: "
                f"Evasion={metrics['evasion_rate']:.1%}, "
                f"Q-table size={metrics['q_table_size']}"
            )

        avg_evasion = sum(round_evasion_rates) / len(round_evasion_rates)

        history["round"].append(round_num)
        history["evasion_rate"].append(avg_evasion)
        history["q_table_size"].append(len(agent.q_table))

        logger.info(f"Round {round_num + 1} avg evasion rate: {avg_evasion:.1%}")

    logger.info("\nSaving adversarial agent")
    Path("models").mkdir(exist_ok=True)
    agent.save("models/adversarial_agent.pkl")

    logger.info("\nAdversarial training complete")
    logger.info(f"Final evasion rate: {history['evasion_rate'][-1]:.1%}")
    logger.info(f"Q-table size: {history['q_table_size'][-1]}")

    return history


if __name__ == "__main__":
    history = adversarial_training()
