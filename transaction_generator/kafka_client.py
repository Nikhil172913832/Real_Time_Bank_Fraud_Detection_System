"""Kafka producer client for transaction streaming."""

import msgpack
from kafka import KafkaProducer
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any
from logger import get_logger
from exceptions import KafkaConnectionError

log = get_logger(__name__)


class TransactionKafkaClient:
    """Manages Kafka producer for transaction streaming."""
    
    def __init__(self, bootstrap_servers: list = None):
        self.bootstrap_servers = bootstrap_servers or ["localhost:9092"]
        self.producer = self._create_producer()
    
    def _custom_serializer(self, obj: Any) -> Any:
        """Custom serializer for datetime and Decimal objects."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return obj
    
    def _create_producer(self) -> KafkaProducer:
        """Create and configure Kafka producer."""
        try:
            producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: msgpack.packb(
                    v, default=self._custom_serializer
                ),
                retries=5
            )
            log.info(f"Kafka producer connected to {self.bootstrap_servers}")
            return producer
        except Exception as e:
            log.error(f"Failed to create Kafka producer: {e}")
            raise KafkaConnectionError(f"Kafka connection failed: {e}")
    
    def send_transaction(self, topic: str, transaction: Dict[str, Any]) -> None:
        """Send transaction to Kafka topic."""
        try:
            future = self.producer.send(topic, transaction)
            future.get(timeout=10)
        except Exception as e:
            log.error(f"Failed to send transaction to Kafka: {e}")
            raise
    
    def close(self) -> None:
        """Close Kafka producer."""
        if self.producer:
            self.producer.close()
            log.info("Kafka producer closed")
