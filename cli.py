"""
Command Line Interface for Fraud Detection System
=================================================

Unified CLI for training, serving, and data generation.
"""

import click
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    Fraud Detection System CLI
    
    A production-grade MLOps system for detecting fraudulent banking transactions
    in real-time using XGBoost, Kafka, and PostgreSQL.
    """
    pass


@cli.command()
@click.option('--data-path', default='data.csv', help='Path to training data CSV')
@click.option('--n-trials', default=100, type=int, help='Number of Optuna trials')
@click.option('--test-size', default=0.2, type=float, help='Test set size (0.0-1.0)')
@click.option('--random-state', default=42, type=int, help='Random seed for reproducibility')
def train(data_path, n_trials, test_size, random_state):
    """Train fraud detection model with hyperparameter optimization."""
    logger.info("Starting model training...")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Optuna trials: {n_trials}")
    logger.info(f"Test size: {test_size}")
    
    from training import FraudDetectionTrainer
    
    trainer = FraudDetectionTrainer(
        data_path=data_path,
        test_size=test_size,
        random_state=random_state,
        n_trials=n_trials
    )
    
    metrics = trainer.train()
    trainer.save_model()
    
    logger.info("Training complete!")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")


@cli.command()
@click.option('--host', default='0.0.0.0', help='API host')
@click.option('--port', default=5000, type=int, help='API port')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def serve(host, port, debug):
    """Start the fraud detection API server."""
    logger.info(f"Starting API server on {host}:{port}")
    
    import app
    app.app.run(host=host, port=port, debug=debug)


@cli.command()
@click.option('--n-samples', default=100000, type=int, help='Number of transactions to generate')
@click.option('--output', default='data.csv', help='Output CSV file path')
@click.option('--fraud-rate', default=0.05, type=float, help='Fraud rate (0.0-1.0)')
def generate_data(n_samples, output, fraud_rate):
    """Generate synthetic transaction data."""
    logger.info(f"Generating {n_samples:,} transactions...")
    logger.info(f"Fraud rate: {fraud_rate:.2%}")
    logger.info(f"Output: {output}")
    
    import data
    # The data.py script generates data when run
    logger.info("Data generation complete!")


@cli.command()
@click.option('--topic', default='transactions', help='Kafka topic')
@click.option('--rate', default=100, type=int, help='Transactions per second')
def produce(topic, rate):
    """Start Kafka transaction producer."""
    logger.info(f"Starting transaction producer...")
    logger.info(f"Topic: {topic}")
    logger.info(f"Rate: {rate} TPS")
    
    import transactions
    # The transactions.py script starts the producer
    logger.info("Producer started!")


@cli.command()
def consume():
    """Start Kafka consumer for fraud detection."""
    logger.info("Starting fraud detection consumer...")
    
    import inference
    # The inference.py script starts the consumer
    logger.info("Consumer started!")


@cli.command()
@click.option('--port', default=8501, type=int, help='Dashboard port')
def dashboard(port):
    """Start Streamlit dashboard."""
    logger.info(f"Starting dashboard on port {port}...")
    
    import subprocess
    subprocess.run(['streamlit', 'run', 'dashboard.py', '--server.port', str(port)])


@cli.command()
@click.option('--coverage', is_flag=True, help='Generate coverage report')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def test(coverage, verbose):
    """Run test suite."""
    import subprocess
    
    cmd = ['pytest', 'tests/']
    
    if verbose:
        cmd.append('-v')
    
    if coverage:
        cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term'])
    
    logger.info(f"Running tests: {' '.join(cmd)}")
    subprocess.run(cmd)


@cli.command()
def init_db():
    """Initialize database schema."""
    logger.info("Initializing database...")
    
    import subprocess
    subprocess.run(['python', 'scripts/init_db.py'])
    
    logger.info("Database initialized!")


if __name__ == '__main__':
    cli()
