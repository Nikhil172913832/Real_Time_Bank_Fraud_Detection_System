"""Refactored Fraud Simulator using modular components."""

import os
import random
import psycopg2
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any

from .config import TransactionConfig
from .kafka_client import TransactionKafkaClient
from .user_sampler import UserSampler
from .fraud_patterns import (
    BurstFraudPattern,
    MoneyLaunderingPattern,
    AccountTakeoverPattern
)
from logger import get_logger

log = get_logger(__name__)


class FraudSimulator:
    """Orchestrates fraud transaction generation and streaming."""
    
    def __init__(self):
        # Configuration
        self.config = TransactionConfig()
        
        # Database
        self.db_url = os.getenv("DB_URL")
        assert self.db_url, "DB_URL environment variable not set"
        
        # Kafka
        self.kafka_client = TransactionKafkaClient()
        
        # State
        self.current_time = datetime.now()
        self.consumers = self._fetch_all_consumers()
        
        # User sampler
        self.user_sampler = UserSampler(self.consumers)
        
        # Initialize mule accounts
        self.mule_accounts = set(random.sample(
            list(self.consumers.keys()), 
            min(30, len(self.consumers))
        ))
        
        # Fraud patterns
        self.burst_pattern = BurstFraudPattern(self.config)
        self.ml_pattern = MoneyLaunderingPattern(self.config)
        self.takeover_pattern = AccountTakeoverPattern(self.config)
        
        log.info("FraudSimulator initialized successfully")
    
    def _fetch_all_consumers(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all consumer data from database."""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM consumers")
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                    
                    consumers = {}
                    for row in rows:
                        consumer_dict = dict(zip(columns, row))
                        consumers[consumer_dict["user_id"]] = consumer_dict
                    
                    log.info(f"Fetched {len(consumers)} consumers from database")
                    return consumers
        except Exception as e:
            log.error(f"Error fetching consumers: {e}")
            return {}
    
    def generate_burst_fraud(self, start_time: datetime = None, is_fraud: bool = True):
        """Generate burst fraud pattern."""
        start_time = start_time or self.current_time
        sender_id = self.user_sampler.sample_user()
        sender_data = self.user_sampler.get_user_data(sender_id)
        
        transactions = self.burst_pattern.generate(
            start_time=start_time,
            sender_id=sender_id,
            sender_data=sender_data,
            mule_accounts=self.mule_accounts,
            user_sampler=self.user_sampler,
            is_fraud=is_fraud
        )
        
        log.info(f"Generated {len(transactions)} burst transactions")
        return transactions
    
    def generate_money_laundering(self, start_time: datetime = None):
        """Generate money laundering pattern."""
        start_time = start_time or self.current_time
        source_sender = self.user_sampler.sample_user()
        
        transactions = self.ml_pattern.generate(
            start_time=start_time,
            source_sender=source_sender,
            mule_accounts=self.mule_accounts,
            user_sampler=self.user_sampler,
            consumers=self.consumers
        )
        
        log.info(f"Generated {len(transactions)} money laundering transactions")
        return transactions
    
    def generate_account_takeover(self, start_time: datetime = None):
        """Generate account takeover pattern."""
        start_time = start_time or self.current_time
        
        # Select victim with decent balance
        victims = [
            uid for uid, user in self.consumers.items() 
            if user.get("balance", 0) > 500
        ]
        
        if not victims:
            log.warning("No suitable victims for account takeover")
            return []
        
        victim_id = random.choice(victims)
        victim_data = self.user_sampler.get_user_data(victim_id)
        
        transactions = self.takeover_pattern.generate(
            start_time=start_time,
            victim_id=victim_id,
            victim_data=victim_data,
            mule_accounts=self.mule_accounts,
            user_sampler=self.user_sampler
        )
        
        log.info(f"Generated {len(transactions)} account takeover transactions")
        return transactions
    
    def send_to_kafka(self, transactions: list, topic: str = "transactions"):
        """Send transactions to Kafka."""
        for txn in transactions:
            try:
                self.kafka_client.send_transaction(topic, txn)
            except Exception as e:
                log.error(f"Failed to send transaction {txn.get('transaction_id')}: {e}")
    
    def close(self):
        """Clean up resources."""
        self.kafka_client.close()
        log.info("FraudSimulator closed")
