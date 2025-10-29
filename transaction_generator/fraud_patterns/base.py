"""Base class for fraud pattern generators."""

import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime


class FraudPattern(ABC):
    """Base class for all fraud patterns."""
    
    def __init__(self):
        self.ambiguity_rules = {
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
    
    def introduce_ambiguity(self, pattern: str, is_fraud: bool) -> bool:
        """
        Add ambiguity to fraud classification.
        Some fraud patterns may appear legitimate.
        """
        if not is_fraud:
            return False
        
        r = random.random()
        for key, prob in self.ambiguity_rules.items():
            if key in pattern.lower():
                return r < prob
        
        return is_fraud
    
    @abstractmethod
    def generate(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Generate fraud pattern transactions."""
        pass
