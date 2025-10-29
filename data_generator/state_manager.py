"""State manager for tracking user metadata during generation."""

import random
import numpy as np
from typing import Dict, Any
from faker import Faker

fake = Faker()


class StateManager:
    """Manages user state during data generation."""
    
    def __init__(self, user_pool):
        self.user_pool = user_pool
        
        # User metadata
        self.sender_zip_history: Dict[str, str] = {}
        self.sender_ip_history: Dict[str, str] = {}
        self.user_device_history: Dict[str, str] = {}
        self.user_balance: Dict[str, float] = {}
        self.user_avg_txn: Dict[str, float] = {}
        self.initial_user_avg_txn: Dict[str, float] = {}
        self.user_location_history: Dict[str, Any] = {}
        self.user_last_txn_time: Dict[str, Any] = {}
        self.account_age: Dict[str, int] = {}
        
        self._initialize_user_state()
    
    def _initialize_user_state(self) -> None:
        """Initialize state for all users."""
        for user in self.user_pool.all_users:
            self.user_balance[user] = max(
                100, 
                np.random.lognormal(mean=6.5, sigma=1.5)
            )
            self.user_avg_txn[user] = max(
                5, 
                np.random.lognormal(mean=3.5, sigma=1.0)
            )
            self.account_age[user] = random.randint(30, 1500)
            
            # New accounts are higher risk
            if self.account_age[user] < 90 and random.random() < 0.0025:
                self.user_pool.mule_accounts.add(user)
        
        # Store initial avg for reference
        self.initial_user_avg_txn = self.user_avg_txn.copy()
    
    def get_zip_code(self, user_id: str) -> str:
        """Get or initialize user's zip code."""
        if user_id not in self.sender_zip_history:
            self.sender_zip_history[user_id] = fake.zipcode()
        return self.sender_zip_history[user_id]
    
    def set_zip_code(self, user_id: str, zip_code: str) -> None:
        """Update user's zip code."""
        self.sender_zip_history[user_id] = zip_code
    
    def get_ip_address(self, user_id: str) -> str:
        """Get or initialize user's IP address."""
        if user_id not in self.sender_ip_history:
            self.sender_ip_history[user_id] = fake.ipv4_public()
        return self.sender_ip_history[user_id]
    
    def set_ip_address(self, user_id: str, ip_address: str) -> None:
        """Update user's IP address."""
        self.sender_ip_history[user_id] = ip_address
    
    def get_balance(self, user_id: str) -> float:
        """Get user balance."""
        return self.user_balance.get(user_id, 1000.0)
    
    def update_balance(self, user_id: str, amount: float) -> None:
        """Deduct amount from user balance."""
        if user_id in self.user_balance:
            self.user_balance[user_id] = max(0, self.user_balance[user_id] - amount)
    
    def has_sufficient_funds(self, user_id: str, amount: float) -> bool:
        """Check if user has sufficient funds."""
        return amount <= self.get_balance(user_id)
    
    def update_last_txn_time(self, user_id: str, timestamp: Any) -> None:
        """Update last transaction time."""
        self.user_last_txn_time[user_id] = timestamp
