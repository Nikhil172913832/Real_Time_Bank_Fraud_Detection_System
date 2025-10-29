"""User sampling utilities for transaction generation."""

import random
from typing import Dict, Any, List
from logger import get_logger

log = get_logger(__name__)


class UserSampler:
    """Handles weighted user sampling based on activity profiles."""
    
    def __init__(self, consumers: Dict[str, Dict[str, Any]]):
        self.consumers = consumers
        self._categorize_users()
    
    def _categorize_users(self) -> None:
        """Categorize users by activity profile."""
        self.active_users = [
            uid for uid, user in self.consumers.items()
            if user.get("user_profile") == "active"
        ]
        self.moderate_users = [
            uid for uid, user in self.consumers.items()
            if user.get("user_profile") == "moderate"
        ]
        self.low_activity_users = [
            uid for uid, user in self.consumers.items()
            if user.get("user_profile") == "low"
        ]
        
        log.info(
            f"User categories - Active: {len(self.active_users)}, "
            f"Moderate: {len(self.moderate_users)}, "
            f"Low: {len(self.low_activity_users)}"
        )
    
    def sample_user(self, weights: Dict[str, float] = None) -> str:
        """
        Sample a user based on activity profile weights.
        
        Args:
            weights: Custom weights for each category (active, moderate, low)
        
        Returns:
            User ID
        """
        default_weights = {"active": 0.7, "moderate": 0.2, "low": 0.1}
        weights = weights or default_weights
        
        category = random.choices(
            population=list(weights.keys()),
            weights=list(weights.values())
        )[0]
        
        if category == "active" and self.active_users:
            return random.choice(self.active_users)
        elif category == "moderate" and self.moderate_users:
            return random.choice(self.moderate_users)
        elif self.low_activity_users:
            return random.choice(self.low_activity_users)
        
        # Fallback to any user
        return random.choice(list(self.consumers.keys()))
    
    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get user data by ID."""
        return self.consumers.get(user_id, {})
    
    def update_user_data(self, user_id: str, updates: Dict[str, Any]) -> None:
        """Update user data."""
        if user_id in self.consumers:
            self.consumers[user_id].update(updates)
