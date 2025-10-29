"""Money laundering pattern generator."""

import random
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any

from .base import FraudPattern
from ..config import TransactionConfig
from faker import Faker

fake = Faker()


class MoneyLaunderingPattern(FraudPattern):
    """Generates money laundering transactions with multiple hops."""
    
    def __init__(self, config: TransactionConfig):
        super().__init__()
        self.config = config
    
    def generate(
        self,
        start_time: datetime,
        source_sender: str,
        mule_accounts: set,
        user_sampler: Any,
        consumers: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate money laundering chain with multiple hops.
        
        Args:
            start_time: Start time for the chain
            source_sender: Initial sender ID
            mule_accounts: Set of mule account IDs
            user_sampler: UserSampler instance
            consumers: Consumer data dictionary
        
        Returns:
            List of transaction dictionaries
        """
        ml_transactions = []
        
        # Initial large amount
        initial_amount = round(random.uniform(1000, 10000), 2)
        
        # First hop to mule
        first_mule = random.choice(list(mule_accounts)) if mule_accounts else user_sampler.sample_user()
        
        # Chain of 3-8 transactions
        chain_length = random.randint(3, 8)
        current_amount = initial_amount
        current_sender = source_sender
        current_receiver = first_mule
        
        for i in range(chain_length):
            # Fee for each hop
            fee_percentage = random.uniform(0.05, 0.15)
            
            if i > 0:
                current_sender = current_receiver
                # Choose next receiver
                if random.random() < 0.6 and mule_accounts:
                    current_receiver = random.choice(list(mule_accounts - {current_sender}))
                else:
                    current_receiver = user_sampler.sample_user()
                    while current_receiver == current_sender or current_receiver in mule_accounts:
                        current_receiver = user_sampler.sample_user()
            
            # Amount after fee
            hop_amount = current_amount * (1 - fee_percentage) if i > 0 else current_amount
            current_amount = hop_amount
            
            # Time delay between hops
            hop_time = start_time + timedelta(minutes=random.randint(i*30, i*180))
            
            # Get sender data
            sender_data = consumers.get(current_sender, {})
            zip_history = sender_data.get("zip_history", [fake.zipcode()])
            ip_history = sender_data.get("ip_history", [fake.ipv4_public()])
            avg_txn_amount = float(sender_data.get("avg_txn_amount") or 100)
            
            # Source and device
            source = random.choices(["MOBILE_APP", "WEB"], weights=[0.5, 0.5])[0]
            device_os = "Unknown" if source == "WEB" else random.choices(
                ["Android", "iOS"], weights=[0.6, 0.4]
            )[0]
            
            # Device fingerprint
            device_fingerprint = None
            if device_os != "Unknown":
                fingerprint_str = f"{device_os}|{random.choice(ip_history)}|{random.choices(self.config.browsers, self.config.browser_weights)[0]}"
                device_fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()
            
            # Metrics
            last_txn_time = sender_data.get("last_txn_time", start_time)
            time_since_last = (hop_time - last_txn_time).total_seconds() / 60.0
            amount_velocity = round(hop_amount / (time_since_last + 1e-6), 2)
            amount_to_avg_ratio = round(hop_amount / (avg_txn_amount + 1e-6), 2)
            device_match = int(device_fingerprint == sender_data.get("device_fingerprint"))
            
            # Merchant (often money transfer, gift cards, gambling)
            merchant_category = random.choice(
                ["Money Transfer", "Gift Cards", "Gambling"] 
                if random.random() < 0.5 else self.config.merchant_categories
            )
            merchant_risk_level = self.config.merchant_risk.get(merchant_category, 1)
            
            transaction = {
                "transaction_id": str(uuid.uuid4()),
                "sender_id": current_sender,
                "receiver_id": current_receiver,
                "timestamp": hop_time,
                "amount": round(hop_amount, 2),
                "source": source,
                "device_os": device_os,
                "browser": random.choices(self.config.browsers, self.config.browser_weights)[0] if source == "WEB" else None,
                "zip_code": random.choice(zip_history) if zip_history else fake.zipcode(),
                "merchant_category": merchant_category,
                "ip_address": random.choice(ip_history) if ip_history else fake.ipv4_public(),
                "session_id": str(uuid.uuid4()),
                "account_age_days": sender_data.get("account_age_days", 0),
                "fraud_bool": 1,
                "is_international": False,
                "pattern": f"money_laundering_hop_{i+1}",
                "device_fingerprint": device_fingerprint,
                "country_code": "US",
                "merchant_risk_level": merchant_risk_level,
                "time_since_last_txn": round(time_since_last, 2),
                "amount_velocity": amount_velocity,
                "amount_to_average_ratio": amount_to_avg_ratio,
                "device_match": device_match,
                "hour_of_day": hop_time.hour,
                "day_of_week": hop_time.weekday(),
                "is_weekend": int(hop_time.weekday() >= 5),
                "month": hop_time.month,
                "transaction_date": hop_time.date()
            }
            ml_transactions.append(transaction)
        
        return ml_transactions
