"""Account takeover pattern generator."""

import random
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any

from .base import FraudPattern
from ..config import TransactionConfig
from faker import Faker

fake = Faker()


class AccountTakeoverPattern(FraudPattern):
    """Generates account takeover transactions."""
    
    def __init__(self, config: TransactionConfig):
        super().__init__()
        self.config = config
    
    def generate(
        self,
        start_time: datetime,
        victim_id: str,
        victim_data: Dict[str, Any],
        mule_accounts: set,
        user_sampler: Any
    ) -> List[Dict[str, Any]]:
        """
        Generate account takeover pattern.
        
        Args:
            start_time: Start time for takeover
            victim_id: ID of the victim account
            victim_data: Victim's user data
            mule_accounts: Set of mule account IDs
            user_sampler: UserSampler instance
        
        Returns:
            List of transaction dictionaries
        """
        takeover_transactions = []
        
        # New location/device for attacker
        new_ip = fake.ipv4_public()
        new_zip = fake.zipcode()
        is_international = random.random() < 0.05
        country_code = random.choice(["US", "GB", "IN", "DE", "AU", "JP", "CN"]) if is_international else "US"
        
        # Device characteristics
        if random.random() < 0.5:
            device_os = random.choices(["Android", "iOS"], weights=[0.6, 0.4])[0]
        else:
            device_os = random.choices(["Windows", "macOS", "Linux"], weights=[0.6, 0.35, 0.05])[0]
        
        fingerprint_str = f"{device_os}|{new_ip}|{random.choices(self.config.browsers, self.config.browser_weights)[0]}"
        device_fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()
        
        # Metrics for credential change
        avg_txn_amount = float(victim_data.get("avg_txn_amount") or 100)
        last_txn_time = victim_data.get("last_txn_time", start_time)
        time_since_last = (start_time - last_txn_time).total_seconds() / 60.0
        device_match = int(device_fingerprint == victim_data.get("device_fingerprint"))
        
        # Account credential change event
        account_change = {
            "transaction_id": str(uuid.uuid4()),
            "sender_id": victim_id,
            "receiver_id": victim_id,
            "timestamp": start_time,
            "amount": 0.0,
            "source": random.choices(["WEB", "MOBILE_APP"], weights=[0.8, 0.2])[0],
            "device_os": device_os,
            "browser": random.choices(self.config.browsers, self.config.browser_weights)[0],
            "zip_code": new_zip,
            "merchant_category": "Account Management",
            "ip_address": new_ip,
            "session_id": str(uuid.uuid4()),
            "account_age_days": victim_data.get("account_age_days", 0),
            "fraud_bool": True,
            "pattern": "account_takeover_credential_change",
            "device_fingerprint": device_fingerprint,
            "country_code": country_code,
            "merchant_risk_level": 5,
            "time_since_last_txn": round(time_since_last, 2),
            "amount_velocity": 0,
            "amount_to_average_ratio": 0,
            "device_match": device_match,
            "is_international": is_international,
            "hour_of_day": start_time.hour,
            "day_of_week": start_time.weekday(),
            "is_weekend": int(start_time.weekday() >= 5),
            "month": start_time.month,
            "transaction_date": start_time.date()
        }
        takeover_transactions.append(account_change)
        
        # Drain account in 1-3 transactions
        drain_attempts = random.randint(1, 3)
        total_balance = float(victim_data.get("balance", 1000))
        remaining_balance = total_balance
        
        for i in range(drain_attempts):
            # Calculate drain amount
            if i == drain_attempts - 1:
                # Last transaction drains everything
                drain_amount = remaining_balance
            else:
                drain_amount = remaining_balance * random.uniform(0.3, 0.7)
            
            remaining_balance -= drain_amount
            
            # Target: often mule accounts
            receiver_id = random.choice(list(mule_accounts)) if mule_accounts and random.random() < 0.7 else user_sampler.sample_user()
            
            # Time delay
            drain_time = start_time + timedelta(minutes=random.randint(5 + i*10, 30 + i*20))
            
            # Merchant
            merchant_category = random.choice(
                ["Money Transfer", "Gift Cards", "Cash Advance"] 
                if random.random() < 0.6 else self.config.merchant_categories
            )
            merchant_risk_level = self.config.merchant_risk.get(merchant_category, 1)
            
            # Metrics
            time_since_change = (drain_time - start_time).total_seconds() / 60.0
            amount_velocity = round(drain_amount / (time_since_change + 1e-6), 2)
            amount_to_avg_ratio = round(drain_amount / (avg_txn_amount + 1e-6), 2)
            
            drain_transaction = {
                "transaction_id": str(uuid.uuid4()),
                "sender_id": victim_id,
                "receiver_id": receiver_id,
                "timestamp": drain_time,
                "amount": round(drain_amount, 2),
                "source": random.choices(["MOBILE_APP", "WEB"], weights=[0.6, 0.4])[0],
                "device_os": device_os,
                "browser": random.choices(self.config.browsers, self.config.browser_weights)[0] if random.random() < 0.5 else None,
                "zip_code": new_zip,
                "merchant_category": merchant_category,
                "ip_address": new_ip,
                "session_id": str(uuid.uuid4()),
                "account_age_days": victim_data.get("account_age_days", 0),
                "fraud_bool": int(self.introduce_ambiguity("drain", True)),
                "pattern": f"account_takeover_drain_{i+1}",
                "device_fingerprint": device_fingerprint,
                "country_code": country_code,
                "merchant_risk_level": merchant_risk_level,
                "time_since_last_txn": round(time_since_change, 2),
                "amount_velocity": amount_velocity,
                "amount_to_average_ratio": amount_to_avg_ratio,
                "device_match": device_match,
                "is_international": is_international,
                "hour_of_day": drain_time.hour,
                "day_of_week": drain_time.weekday(),
                "is_weekend": int(drain_time.weekday() >= 5),
                "month": drain_time.month,
                "transaction_date": drain_time.date()
            }
            takeover_transactions.append(drain_transaction)
        
        return takeover_transactions
