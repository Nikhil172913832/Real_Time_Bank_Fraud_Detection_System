"""Burst fraud pattern generator."""

import random
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any
from decimal import Decimal

from .base import FraudPattern
from ..config import TransactionConfig
from faker import Faker

fake = Faker()


class BurstFraudPattern(FraudPattern):
    """Generates burst fraud transactions."""
    
    def __init__(self, config: TransactionConfig):
        super().__init__()
        self.config = config
    
    def generate(
        self,
        start_time: datetime,
        sender_id: str,
        sender_data: Dict[str, Any],
        mule_accounts: set,
        user_sampler: Any,
        is_fraud: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate a burst of transactions.
        
        Args:
            start_time: Start time for the burst
            sender_id: ID of the sender
            sender_data: Sender's user data
            mule_accounts: Set of mule account IDs
            user_sampler: UserSampler instance
            is_fraud: Whether this is fraudulent burst
        
        Returns:
            List of transaction dictionaries
        """
        burst = []
        
        # Determine location and device
        if is_fraud:
            zip_code = fake.zipcode()
            ip_address = fake.ipv4_public()
            is_international = random.random() < 0.05
            country_code = random.choice(["US", "GB", "IN", "DE", "AU", "JP", "CN"]) if is_international else "US"
            
            # Risky merchants for fraud
            if random.random() < 0.5:
                selected_merchants = [
                    m for m in self.config.merchant_categories 
                    if self.config.merchant_risk.get(m, 1) >= 4
                ]
            else:
                selected_merchants = self.config.merchant_categories
        else:
            # Use existing location/device
            zip_history = sender_data.get("zip_history", [fake.zipcode()])
            ip_history = sender_data.get("ip_history", [fake.ipv4_public()])
            zip_code = random.choice(zip_history) if zip_history else fake.zipcode()
            ip_address = random.choice(ip_history) if ip_history else fake.ipv4_public()
            is_international = random.random() < 0.00005
            country_code = "US"
            selected_merchants = self.config.merchant_categories
        
        # Burst size
        burst_size = random.randint(3, 15 if is_fraud else 8)
        
        for _ in range(burst_size):
            receiver_id = user_sampler.sample_user()
            
            # For fraud, sometimes send to mule accounts
            if is_fraud and random.random() < 0.2 and mule_accounts:
                receiver_id = random.choice(list(mule_accounts))
            
            # Avoid self-transfers
            while receiver_id == sender_id:
                receiver_id = user_sampler.sample_user()
            
            device_os = random.choices(
                ["Android", "iOS", "Unknown"], 
                weights=[0.6, 0.3, 0.1]
            )[0]
            
            # Calculate amount
            sender_balance = float(sender_data.get("balance", 1000))
            avg_txn_amount = float(sender_data.get("avg_txn_amount") or 100)
            
            if is_fraud:
                # Drain account quickly
                amount = min(
                    sender_balance * random.uniform(0.1, 0.4),
                    random.uniform(50, 500)
                )
                # Sometimes test with micro amounts
                if random.random() < 0.2:
                    amount = round(random.uniform(0.5, 5.0), 2)
            else:
                # Legitimate bursts are smaller
                amount = round(avg_txn_amount * random.uniform(0.5, 1.5), 2)
            
            # Merchant
            merchant_category = random.choice(selected_merchants)
            merchant_risk_level = self.config.merchant_risk.get(merchant_category, 1)
            
            # Device fingerprint
            device_fingerprint = None
            if device_os != "Unknown":
                fingerprint_str = f"{device_os}|{ip_address}|{random.choices(self.config.browsers, self.config.browser_weights)[0]}"
                device_fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()
            
            # Calculate metrics
            last_txn_time = sender_data.get("last_txn_time", start_time)
            time_since_last = (start_time - last_txn_time).total_seconds() / 60.0
            amount_velocity = round(amount / (time_since_last + 1e-6), 2)
            amount_to_avg_ratio = round(amount / (avg_txn_amount + 1e-6), 2)
            device_match = int(device_fingerprint == sender_data.get("device_fingerprint"))
            
            transaction = {
                "transaction_id": str(uuid.uuid4()),
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "timestamp": start_time + timedelta(seconds=random.randint(1, 60)),
                "amount": round(amount, 2),
                "source": "MOBILE_APP" if random.random() < 0.8 else "WEB",
                "device_os": device_os,
                "browser": random.choices(self.config.browsers, self.config.browser_weights)[0] if device_os == "Unknown" else None,
                "zip_code": zip_code,
                "amount_velocity": amount_velocity,
                "amount_to_average_ratio": amount_to_avg_ratio,
                "device_match": device_match,
                "is_international": is_international,
                "device_fingerprint": device_fingerprint,
                "country_code": country_code,
                "merchant_risk_level": merchant_risk_level,
                "merchant_category": merchant_category,
                "ip_address": ip_address,
                "session_id": str(uuid.uuid4()),
                "account_age_days": sender_data.get("account_age_days", 0),
                "fraud_bool": int(self.introduce_ambiguity("burst", is_fraud)),
                "pattern": "burst_" + ("fraud" if is_fraud else "legitimate"),
                "hour_of_day": start_time.hour,
                "day_of_week": start_time.weekday(),
                "is_weekend": int(start_time.weekday() >= 5),
                "month": start_time.month,
                "transaction_date": start_time.date(),
                "time_since_last_txn": round(time_since_last, 2)
            }
            burst.append(transaction)
        
        return burst
