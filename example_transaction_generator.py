"""
Example usage of refactored transaction generator.

This shows how to use the modular transaction_generator package
to simulate real-time fraud transactions.
"""

from datetime import datetime
from transaction_generator import FraudSimulator


def main():
    # Initialize the fraud simulator
    print("Initializing FraudSimulator...")
    simulator = FraudSimulator()
    
    # Generate burst fraud pattern
    print("\n1. Generating burst fraud...")
    burst_txns = simulator.generate_burst_fraud(is_fraud=True)
    print(f"   Generated {len(burst_txns)} burst transactions")
    
    # Send to Kafka
    print("   Sending to Kafka...")
    simulator.send_to_kafka(burst_txns, topic="transactions")
    
    # Generate money laundering pattern
    print("\n2. Generating money laundering...")
    ml_txns = simulator.generate_money_laundering()
    print(f"   Generated {len(ml_txns)} money laundering transactions")
    simulator.send_to_kafka(ml_txns, topic="transactions")
    
    # Generate account takeover
    print("\n3. Generating account takeover...")
    takeover_txns = simulator.generate_account_takeover()
    print(f"   Generated {len(takeover_txns)} account takeover transactions")
    simulator.send_to_kafka(takeover_txns, topic="transactions")
    
    # Generate legitimate burst
    print("\n4. Generating legitimate burst...")
    legit_burst = simulator.generate_burst_fraud(is_fraud=False)
    print(f"   Generated {len(legit_burst)} legitimate transactions")
    simulator.send_to_kafka(legit_burst, topic="transactions")
    
    # Clean up
    print("\nClosing simulator...")
    simulator.close()
    print("✓ Done!")


if __name__ == "__main__":
    main()
