"""
Lightweight Feature Store
=========================

Redis-based feature store for caching and serving features.
"""

import logging
import redis
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Lightweight Redis-based feature store.
    
    Features:
    - Feature caching with TTL
    - Batch feature retrieval
    - Feature versioning
    - Online/offline feature serving
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        default_ttl: int = 3600
    ):
        """
        Initialize feature store.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            default_ttl: Default TTL for features (seconds)
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        self.default_ttl = default_ttl
        
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
    
    def _make_key(self, entity_id: str, feature_group: str) -> str:
        """Create Redis key for feature."""
        return f"features:{feature_group}:{entity_id}"
    
    def set_features(
        self,
        entity_id: str,
        feature_group: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """
        Store features in Redis.
        
        Args:
            entity_id: Entity identifier (e.g., user_id, transaction_id)
            feature_group: Feature group name
            features: Dictionary of features
            ttl: Time-to-live in seconds
        """
        key = self._make_key(entity_id, feature_group)
        ttl = ttl or self.default_ttl
        
        # Add metadata
        features_with_meta = {
            **features,
            "_timestamp": datetime.now().isoformat(),
            "_feature_group": feature_group
        }
        
        # Store in Redis
        self.redis_client.setex(
            key,
            ttl,
            json.dumps(features_with_meta)
        )
        
        logger.debug(f"Stored features for {entity_id} in {feature_group}")
    
    def get_features(
        self,
        entity_id: str,
        feature_group: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve features from Redis.
        
        Args:
            entity_id: Entity identifier
            feature_group: Feature group name
            
        Returns:
            Dictionary of features or None if not found
        """
        key = self._make_key(entity_id, feature_group)
        data = self.redis_client.get(key)
        
        if data:
            features = json.loads(data)
            logger.debug(f"Retrieved features for {entity_id} from {feature_group}")
            return features
        else:
            logger.debug(f"Features not found for {entity_id} in {feature_group}")
            return None
    
    def get_batch_features(
        self,
        entity_ids: List[str],
        feature_group: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve features for multiple entities.
        
        Args:
            entity_ids: List of entity identifiers
            feature_group: Feature group name
            
        Returns:
            Dictionary mapping entity_id to features
        """
        results = {}
        
        for entity_id in entity_ids:
            features = self.get_features(entity_id, feature_group)
            if features:
                results[entity_id] = features
        
        logger.info(f"Retrieved features for {len(results)}/{len(entity_ids)} entities")
        return results
    
    def delete_features(self, entity_id: str, feature_group: str):
        """Delete features from Redis."""
        key = self._make_key(entity_id, feature_group)
        self.redis_client.delete(key)
        logger.debug(f"Deleted features for {entity_id} from {feature_group}")
    
    def feature_exists(self, entity_id: str, feature_group: str) -> bool:
        """Check if features exist in Redis."""
        key = self._make_key(entity_id, feature_group)
        return self.redis_client.exists(key) > 0


class UserFeatureStore(FeatureStore):
    """Specialized feature store for user features."""
    
    def set_user_features(
        self,
        user_id: str,
        transaction_count: int,
        avg_transaction_amount: float,
        last_transaction_time: datetime,
        device_count: int,
        location_count: int,
        ttl: int = 86400  # 24 hours
    ):
        """Store user-level features."""
        features = {
            "transaction_count": transaction_count,
            "avg_transaction_amount": avg_transaction_amount,
            "last_transaction_time": last_transaction_time.isoformat(),
            "device_count": device_count,
            "location_count": location_count
        }
        
        self.set_features(user_id, "user_profile", features, ttl)
    
    def get_user_features(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user-level features."""
        return self.get_features(user_id, "user_profile")
    
    def update_transaction_stats(
        self,
        user_id: str,
        new_amount: float
    ):
        """Update user transaction statistics."""
        features = self.get_user_features(user_id)
        
        if features:
            # Update stats
            count = features.get("transaction_count", 0)
            avg = features.get("avg_transaction_amount", 0)
            
            new_count = count + 1
            new_avg = (avg * count + new_amount) / new_count
            
            features["transaction_count"] = new_count
            features["avg_transaction_amount"] = new_avg
            features["last_transaction_time"] = datetime.now().isoformat()
            
            self.set_features(user_id, "user_profile", features)
            logger.debug(f"Updated transaction stats for {user_id}")


# Example usage
if __name__ == '__main__':
    # Initialize feature store
    store = UserFeatureStore()
    
    # Store user features
    store.set_user_features(
        user_id="user_123",
        transaction_count=50,
        avg_transaction_amount=125.50,
        last_transaction_time=datetime.now(),
        device_count=2,
        location_count=1
    )
    
    # Retrieve features
    features = store.get_user_features("user_123")
    print(f"User features: {json.dumps(features, indent=2)}")
    
    # Update stats
    store.update_transaction_stats("user_123", 200.0)
    
    # Retrieve updated features
    updated_features = store.get_user_features("user_123")
    print(f"Updated features: {json.dumps(updated_features, indent=2)}")
