"""Configuration management for transaction generation."""

import yaml
from pathlib import Path
from typing import Dict, List, Any
from logger import get_logger

log = get_logger(__name__)


class TransactionConfig:
    """Manages configuration for transaction generation."""
    
    def __init__(self, config_path: str = "constants.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as file:
                self._config = yaml.safe_load(file)
            log.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            log.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    @property
    def merchant_categories(self) -> List[str]:
        """Get list of merchant categories."""
        return list(self._config.get("merchant_categories", {}).keys())
    
    @property
    def merchant_risk(self) -> Dict[str, int]:
        """Get merchant risk levels."""
        return self._config.get("merchant_categories", {})
    
    @property
    def browsers(self) -> List[str]:
        """Get list of browsers."""
        return self._config.get("browsers", [])
    
    @property
    def browser_weights(self) -> List[float]:
        """Get browser weights."""
        return self._config.get("browser_weights", [])
    
    @property
    def countries(self) -> Dict[str, str]:
        """Get country codes."""
        return self._config.get("countries", {})
    
    @property
    def high_risk_hours(self) -> List[int]:
        """Get high-risk hours (1 AM - 5 AM)."""
        return list(range(1, 5))
    
    @property
    def high_risk_dates(self) -> List[int]:
        """Get high-risk dates (payday)."""
        return [1, 15, 30]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self._config.get(key, default)
