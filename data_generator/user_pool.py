"""User pool generation and management."""

import random
import uuid
import numpy as np
from typing import List, Set


class UserPool:
    """Manages user pool for data generation."""
    
    def __init__(
        self, 
        n_active: int = 500,
        n_moderate: int = 1500,
        n_low: int = 2000
    ):
        self.active_users: List[str] = [str(uuid.uuid4()) for _ in range(n_active)]
        self.moderate_users: List[str] = [str(uuid.uuid4()) for _ in range(n_moderate)]
        self.low_activity_users: List[str] = [str(uuid.uuid4()) for _ in range(n_low)]
        
        self.all_users = self.active_users + self.moderate_users + self.low_activity_users
        
        # Mule accounts (30 random users)
        self.mule_accounts: Set[str] = set(random.sample(self.all_users, 30))
        
        # Frequent receivers
        self.frequent_receivers: List[str] = random.sample(self.all_users, 50)
    
    def sample_user(self, weights: List[float] = None) -> str:
        """
        Sample user with activity-based weights.
        
        Args:
            weights: [active_weight, moderate_weight, low_weight]
        
        Returns:
            User ID
        """
        weights = weights or [0.7, 0.2, 0.1]
        
        category = random.choices(
            population=["active", "moderate", "low"],
            weights=weights
        )[0]
        
        if category == "active":
            return random.choice(self.active_users)
        elif category == "moderate":
            return random.choice(self.moderate_users)
        else:
            return random.choice(self.low_activity_users)
    
    def get_user_category(self, user_id: str) -> str:
        """Get user activity category."""
        if user_id in self.active_users:
            return "active"
        elif user_id in self.moderate_users:
            return "moderate"
        elif user_id in self.low_activity_users:
            return "low"
        return "unknown"
