"""Main data generator orchestrator."""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

from .config import DataConfig, FRAUD_PATTERNS
from .user_pool import UserPool
from .state_manager import StateManager
from .transaction_factory import TransactionFactory
from helper import weighted_user_sample
from fraud_patterns import simulate_burst, simulate_money_laundering, simulate_account_takeover
from faker import Faker

fake = Faker()


class DataGenerator:
    """Orchestrates batch data generation."""
    
    def __init__(
        self,
        n_records: int = 1_000_000,
        fraud_ratio: float = 0.012,
        start_date: datetime = None,
        end_date: datetime = None,
        config_path: str = "constants.yaml"
    ):
        self.n_records = n_records
        self.fraud_ratio = fraud_ratio
        self.start_date = start_date or datetime(2023, 1, 1)
        self.end_date = end_date or datetime(2023, 12, 31)
        self.date_range = (self.end_date - self.start_date).days
        
        # Components
        self.config = DataConfig(config_path)
        self.user_pool = UserPool()
        self.state = StateManager(self.user_pool)
        self.factory = TransactionFactory(self.config, self.user_pool, self.state)
        
        # Calculate fraud distribution
        self.fraud_counts = self._calculate_fraud_distribution()
        
        print(f"DataGenerator initialized:")
        print(f"  - Total records: {n_records:,}")
        print(f"  - Fraud ratio: {fraud_ratio:.1%}")
        print(f"  - Date range: {self.start_date.date()} to {self.end_date.date()}")
        print(f"  - Users: {len(self.user_pool.all_users):,}")
    
    def _calculate_fraud_distribution(self) -> Dict[str, int]:
        """Calculate number of fraud transactions per pattern."""
        total_fraud = int(self.n_records * self.fraud_ratio)
        fraud_counts = {
            pattern: int(total_fraud * ratio) 
            for pattern, ratio in FRAUD_PATTERNS.items() 
            if ratio > 0
        }
        
        # Adjust to match exact total
        remaining = total_fraud - sum(fraud_counts.values())
        if remaining > 0:
            for pattern in sorted(fraud_counts.keys()):
                fraud_counts[pattern] += 1
                remaining -= 1
                if remaining == 0:
                    break
        
        print(f"\nFraud distribution ({total_fraud:,} fraud transactions):")
        for pattern, count in sorted(fraud_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  - {pattern}: {count:,}")
        
        return fraud_counts
    
    def generate_complex_patterns(self) -> List[Dict[str, Any]]:
        """Generate complex patterns (burst, money laundering, account takeover)."""
        complex_records = []
        
        # Money laundering
        for _ in range(self.fraud_counts.get("money_laundering", 0)):
            start_time = self.start_date + pd.Timedelta(
                days=pd.np.random.randint(0, self.date_range),
                seconds=pd.np.random.randint(0, 86400)
            )
            complex_records.extend(simulate_money_laundering(
                start_time,
                self.user_pool.active_users,
                self.user_pool.moderate_users,
                self.user_pool.low_activity_users,
                self.user_pool.mule_accounts,
                fake,
                self.config.merchant_categories,
                self.state.sender_zip_history,
                self.state.sender_ip_history,
                self.config.browsers,
                self.config.browser_weights,
                self.state.account_age
            ))
        
        # Account takeover
        for _ in range(self.fraud_counts.get("account_takeover", 0)):
            start_time = self.start_date + pd.Timedelta(
                days=pd.np.random.randint(0, self.date_range),
                seconds=pd.np.random.randint(0, 86400)
            )
            complex_records.extend(simulate_account_takeover(
                start_time,
                self.user_pool.all_users,
                self.user_pool.mule_accounts,
                fake,
                self.state.user_balance,
                self.config.browsers,
                self.config.browser_weights,
                self.state.account_age
            ))
        
        # Burst fraud
        for _ in range(self.fraud_counts.get("burst_fraud", 0)):
            start_time = self.start_date + pd.Timedelta(
                days=pd.np.random.randint(0, self.date_range),
                seconds=pd.np.random.randint(0, 86400)
            )
            complex_records.extend(simulate_burst(
                start_time,
                self.user_pool.active_users,
                self.user_pool.moderate_users,
                self.user_pool.low_activity_users,
                fake,
                self.config.merchant_categories,
                self.config.merchant_risk,
                self.state.sender_zip_history,
                self.state.sender_ip_history,
                self.user_pool.mule_accounts,
                self.state.user_balance,
                self.state.user_avg_txn,
                self.config.browsers,
                self.config.browser_weights,
                self.state.account_age,
                is_fraud=True
            ))
        
        return complex_records
    
    def generate(self) -> pd.DataFrame:
        """
        Generate complete dataset.
        
        Returns:
            DataFrame with generated transactions
        """
        records = []
        
        # Complex patterns first
        print("\nGenerating complex patterns...")
        complex_records = self.generate_complex_patterns()
        records.extend(complex_records)
        print(f"Generated {len(complex_records):,} complex pattern records")
        
        # Calculate remaining records
        remaining_count = self.n_records - len(complex_records)
        remaining_fraud = sum([
            count for pattern, count in self.fraud_counts.items()
            if pattern not in ["money_laundering", "account_takeover", "burst_fraud"]
        ])
        remaining_normal = remaining_count - remaining_fraud
        
        print(f"\nGenerating remaining records:")
        print(f"  - Fraud: {remaining_fraud:,}")
        print(f"  - Normal: {remaining_normal:,}")
        
        # Create fraud records list
        fraud_records_to_generate = []
        for pattern, count in self.fraud_counts.items():
            if pattern not in ["normal", "money_laundering", "account_takeover", "burst_fraud"]:
                fraud_records_to_generate.extend([pattern] * count)
        
        # Generate fraud records
        for pattern in tqdm(fraud_records_to_generate, desc="Fraud records"):
            record = self.factory.generate(
                pattern_type=pattern,
                is_fraud=True,
                start_date=self.start_date,
                date_range=self.date_range
            )
            records.append(record)
        
        # Generate normal records
        for _ in tqdm(range(remaining_normal), desc="Normal records"):
            record = self.factory.generate(
                pattern_type="normal",
                is_fraud=False,
                start_date=self.start_date,
                date_range=self.date_range
            )
            records.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        print(f"\n✓ Generated {len(df):,} total records")
        print(f"✓ Fraud rate: {df['fraud_bool'].mean():.2%}")
        
        return df
