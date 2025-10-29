"""Configuration for data generation."""

import yaml
from pathlib import Path
from typing import Dict, List

# Default fraud pattern distribution
FRAUD_PATTERNS = {
    "normal": 0.0,
    "late_night": 0.103,
    "payday_fraud": 0.082,
    "weekend_fraud": 0.072,
    "micro_fraud": 0.051,
    "location_change": 0.08,
    "ip_change": 0.081,
    "location_and_ip_change": 0.07,
    "high_risk_merchant": 0.10,
    "suspicious_merchant": 0.08,
    "mule_transfer": 0.07,
    "insufficient_funds": 0.05,
    "new_account_fraud": 0.06,
    "international": 0.06,
    "money_laundering": 0.02,
    "account_takeover": 0.02,
    "burst_fraud": 0.001
}

HIGH_RISK_HOURS = list(range(1, 5))
HIGH_RISK_DATES = [1, 15, 30]


class DataConfig:
    """Configuration for batch data generation."""
    
    def __init__(self, config_path: str = "constants.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML."""
        with open(self.config_path, "r") as file:
            self._config = yaml.safe_load(file)
    
    @property
    def merchant_categories(self) -> List[str]:
        """Get merchant categories."""
        return list(self._config.get("merchant_category", {}).keys())
    
    @property
    def merchant_risk(self) -> Dict[str, int]:
        """Get merchant risk levels."""
        return self._config.get("merchant_category", {})
    
    @property
    def browsers(self) -> List[str]:
        """Get browsers."""
        return self._config.get("browsers", [])
    
    @property
    def browser_weights(self) -> List[float]:
        """Get browser weights."""
        return self._config.get("browser_weights", [])
    
    @property
    def countries(self) -> Dict[str, str]:
        """Get countries."""
        return self._config.get("countries", {})
