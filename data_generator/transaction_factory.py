"""Transaction factory for generating transactions by pattern."""

import random
import uuid
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from faker import Faker

from .config import DataConfig, HIGH_RISK_HOURS, HIGH_RISK_DATES

fake = Faker()


class TransactionFactory:
    """Generates individual transactions based on pattern type."""
    
    def __init__(self, config: DataConfig, user_pool, state_manager):
        self.config = config
        self.user_pool = user_pool
        self.state = state_manager
    
    def introduce_ambiguity(self, pattern: str, is_fraud: bool) -> bool:
        """Add ambiguity to fraud labels."""
        if not is_fraud:
            return False
        
        ambiguity_rules = {
            'account_takeover': 0.92,
            'credential_change': 0.90,
            'drain': 0.88,
            'burst': 0.85,
            'micro': 0.75,
            'laundering': 0.97,
            'late_night': 0.80,
            'location': 0.70,
            'ip': 0.75,
            'international': 0.85,
            'high_risk': 0.82,
            'suspicious': 0.88,
            'mule': 0.78,
            'insufficient': 0.985,
            'new_account': 0.70,
            'weekend': 0.80,
            'payday': 0.75
        }
        
        r = random.random()
        for key, prob in ambiguity_rules.items():
            if key in pattern.lower():
                return r < prob
        
        return is_fraud
    
    def generate(
        self, 
        pattern_type: str = "normal",
        is_fraud: bool = False,
        start_date: datetime = None,
        date_range: int = 365
    ) -> Dict[str, Any]:
        """
        Generate a single transaction.
        
        Args:
            pattern_type: Type of fraud pattern
            is_fraud: Whether transaction is fraudulent
            start_date: Start date for timestamp generation
            date_range: Date range in days
        
        Returns:
            Transaction dictionary
        """
        if pattern_type is None:
            pattern_type = "normal"
            is_fraud = False
        
        start_date = start_date or datetime(2023, 1, 1)
        
        # Generate timestamp based on pattern
        timestamp = self._generate_timestamp(pattern_type, start_date, date_range)
        
        # Select users
        sender_id = self._select_sender(pattern_type)
        receiver_id = self._select_receiver(pattern_type, sender_id)
        
        # Generate amount
        amount = self._generate_amount(pattern_type, is_fraud, sender_id)
        
        # Source and device
        source, device_os, browser = self._generate_device_info(is_fraud)
        
        # Location data
        zip_code, ip_address, is_international, country_code = self._generate_location(
            pattern_type, sender_id, is_fraud
        )
        
        # Merchant
        merchant_category = self._select_merchant(pattern_type, is_fraud)
        merchant_risk_level = self.config.merchant_risk.get(merchant_category, 1)
        
        # Device fingerprint
        device_fingerprint = None
        if source in ["MOBILE_APP", "WEB"]:
            fingerprint_str = f"{device_os}|{ip_address}|{browser or ''}"
            device_fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()
        
        # Check funds
        has_sufficient_funds = True
        if pattern_type == "insufficient_funds":
            amount = self.state.get_balance(sender_id) * (1 + random.uniform(0.1, 1.0))
            has_sufficient_funds = False
        else:
            has_sufficient_funds = self.state.has_sufficient_funds(sender_id, amount)
        
        # Build transaction
        record = {
            "transaction_id": str(uuid.uuid4()),
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "timestamp": timestamp,
            "amount": round(amount, 2),
            "source": source,
            "device_os": device_os,
            "browser": browser,
            "zip_code": zip_code,
            "merchant_category": merchant_category,
            "ip_address": ip_address,
            "session_id": str(uuid.uuid4()),
            "account_age_days": self.state.account_age.get(sender_id, 0),
            "is_international": is_international,
            "country_code": country_code,
            "device_fingerprint": device_fingerprint,
            "merchant_risk_level": merchant_risk_level,
            "initial_avg_txn": self.state.initial_user_avg_txn.get(sender_id, 100),
            "has_sufficient_funds": has_sufficient_funds,
            "pattern": pattern_type,
            "fraud_bool": int(self.introduce_ambiguity(pattern_type, is_fraud))
        }
        
        # Update state
        if has_sufficient_funds or is_fraud:
            self.state.set_zip_code(sender_id, zip_code)
            self.state.set_ip_address(sender_id, ip_address)
            self.state.update_last_txn_time(sender_id, timestamp)
            
            if has_sufficient_funds:
                self.state.update_balance(sender_id, amount)
        
        return record
    
    def _generate_timestamp(self, pattern: str, start_date: datetime, date_range: int) -> datetime:
        """Generate timestamp based on pattern."""
        if pattern == "late_night":
            days_offset = np.random.randint(0, date_range)
            hours_offset = random.choice(HIGH_RISK_HOURS)
            minutes_offset = np.random.randint(0, 60)
            return start_date + timedelta(days=int(days_offset), hours=hours_offset, minutes=minutes_offset)
        
        elif pattern == "payday_fraud":
            days_offset = random.choice([
                d for d in range(date_range) 
                if (start_date + timedelta(days=d)).day in HIGH_RISK_DATES
            ])
            hours_offset = np.random.randint(0, 24)
            minutes_offset = np.random.randint(0, 60)
            return start_date + timedelta(days=int(days_offset), hours=hours_offset, minutes=minutes_offset)
        
        elif pattern == "weekend_fraud":
            weekend_days = [
                d for d in range(date_range) 
                if (start_date + timedelta(days=d)).weekday() in [5, 6]
            ]
            days_offset = random.choice(weekend_days)
            hours_offset = np.random.randint(0, 24)
            minutes_offset = np.random.randint(0, 60)
            return start_date + timedelta(days=int(days_offset), hours=hours_offset, minutes=minutes_offset)
        
        else:
            days_offset = np.random.randint(0, date_range)
            hours_offset = np.random.randint(0, 24)
            minutes_offset = np.random.randint(0, 60)
            return start_date + timedelta(days=int(days_offset), hours=hours_offset, minutes=minutes_offset)
    
    def _select_sender(self, pattern: str) -> str:
        """Select sender based on pattern."""
        if pattern == "new_account_fraud":
            potential_new = [u for u in self.user_pool.all_users if u not in self.state.user_last_txn_time]
            if potential_new:
                return random.choice(potential_new)
            # Fallback to newest accounts
            newest = sorted([(u, age) for u, age in self.state.account_age.items()], key=lambda x: x[1])[:20]
            if newest:
                return random.choice([u for u, _ in newest])
        
        return self.user_pool.sample_user()
    
    def _select_receiver(self, pattern: str, sender_id: str) -> str:
        """Select receiver based on pattern."""
        if pattern == "mule_transfer" and self.user_pool.mule_accounts:
            return random.choice(list(self.user_pool.mule_accounts))
        
        receiver_id = self.user_pool.sample_user()
        while receiver_id == sender_id:
            receiver_id = self.user_pool.sample_user()
        
        return receiver_id
    
    def _generate_amount(self, pattern: str, is_fraud: bool, sender_id: str) -> float:
        """Generate transaction amount."""
        if pattern == "micro_fraud":
            return round(np.random.uniform(0.5, 10.0), 2)
        
        if is_fraud:
            if random.random() < 0.7:
                amount = np.random.lognormal(mean=5.0, sigma=1.0)
                if random.random() < 0.2:
                    amount = float(int(amount))
            else:
                amount = np.random.lognormal(mean=4.0, sigma=1.0)
        else:
            amount = np.random.lognormal(mean=3.5, sigma=1.2)
            if random.random() < 0.3:
                amount = round(amount, 0)
        
        return amount
    
    def _generate_device_info(self, is_fraud: bool):
        """Generate source, device OS, and browser."""
        if is_fraud:
            source = random.choices(
                ["MOBILE_APP", "WEB", "POS", "PHONE", "ATM"],
                weights=[0.4, 0.5, 0.05, 0.025, 0.025]
            )[0]
        else:
            source = random.choices(
                ["MOBILE_APP", "WEB", "POS", "PHONE", "ATM"],
                weights=[0.6, 0.3, 0.05, 0.025, 0.025]
            )[0]
        
        device_os = "Unknown"
        browser = None
        
        if source == "MOBILE_APP":
            device_os = random.choices(["Android", "iOS"], weights=[0.7, 0.3])[0]
        elif source == "WEB":
            device_os = random.choices(["Windows", "macOS", "Linux"], weights=[0.6, 0.35, 0.05])[0]
            browser = random.choices(self.config.browsers, self.config.browser_weights)[0]
        elif source == "POS":
            device_os = "POS_TERMINAL"
        elif source == "PHONE":
            device_os = random.choices(["Android", "iOS", "Unknown"], weights=[0.45, 0.45, 0.1])[0]
        elif source == "ATM":
            device_os = "ATM_TERMINAL"
        
        return source, device_os, browser
    
    def _generate_location(self, pattern: str, sender_id: str, is_fraud: bool):
        """Generate zip code, IP, international flag, country code."""
        zip_code = self.state.get_zip_code(sender_id)
        ip_address = self.state.get_ip_address(sender_id)
        is_international = False
        country_code = "US"
        
        if pattern in ["location_change", "location_and_ip_change"]:
            new_zip = fake.zipcode()
            while new_zip[:1] == zip_code[:1]:
                new_zip = fake.zipcode()
            zip_code = new_zip
        
        if pattern in ["ip_change", "location_and_ip_change"]:
            new_ip = fake.ipv4_public()
            while new_ip.split('.')[0] == ip_address.split('.')[0]:
                new_ip = fake.ipv4_public()
            ip_address = new_ip
        
        if pattern == "international":
            country_code = random.choice([c for c in self.config.countries.keys() if c != "US"])
            is_international = True
        elif random.random() < 0.03:
            country_code = random.choice(list(self.config.countries.keys()))
            if country_code != "US":
                is_international = True
        
        return zip_code, ip_address, is_international, country_code
    
    def _select_merchant(self, pattern: str, is_fraud: bool) -> str:
        """Select merchant category."""
        if pattern == "high_risk_merchant":
            high_risk = [m for m in self.config.merchant_categories if self.config.merchant_risk.get(m, 1) >= 4]
            return random.choice(high_risk) if high_risk else random.choice(self.config.merchant_categories)
        
        if pattern == "suspicious_merchant":
            return random.choice(["Gift Cards", "Money Transfer", "Gambling"])
        
        if is_fraud:
            return random.choices(
                self.config.merchant_categories,
                weights=[self.config.merchant_risk.get(m, 1) for m in self.config.merchant_categories]
            )[0]
        else:
            return random.choices(
                self.config.merchant_categories,
                weights=[1/self.config.merchant_risk.get(m, 1) for m in self.config.merchant_categories]
            )[0]
