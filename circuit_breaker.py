"""
Circuit Breaker Pattern for Fraud Detection System
==================================================

This module implements the circuit breaker pattern to handle failures
in external services (database, Kafka, email) gracefully and prevent
cascading failures.
"""

import time
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps
from threading import Lock

from exceptions import CircuitBreakerError
from logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    
    The circuit breaker monitors failures and "opens" when a threshold is reached,
    preventing further calls to the failing service. After a timeout, it enters
    "half-open" state to test if the service has recovered.
    
    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Too many failures, requests are rejected immediately
        HALF_OPEN: Testing recovery, limited requests allowed
    
    Example:
        >>> breaker = CircuitBreaker(
        ...     name='database',
        ...     failure_threshold=5,
        ...     recovery_timeout=60.0
        ... )
        >>> 
        >>> @breaker.call
        ... def query_database():
        ...     return db.execute("SELECT * FROM fraud_alerts")
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[BaseException] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Service name for logging
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type to catch
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self.lock = Lock()
        
        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"threshold={failure_threshold}, timeout={recovery_timeout}s"
        )
    
    def call(self, func: Callable) -> Callable:
        """
        Decorator to wrap function calls with circuit breaker logic.
        
        Args:
            func: Function to protect with circuit breaker
            
        Returns:
            Wrapped function
            
        Example:
            >>> @breaker.call
            ... def risky_operation():
            ...     return external_service.call()
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerError(self.name)
            
            try:
                # Attempt the operation
                result = func(*args, **kwargs)
                
                # Success - reset if needed
                if self.state == CircuitState.HALF_OPEN:
                    self._transition_to_closed()
                
                return result
                
            except self.expected_exception as e:
                # Record failure
                self._record_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return False
        
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.warning(
                f"Circuit breaker '{self.name}': Failure {self.failure_count}/{self.failure_threshold}"
            )
            
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()
    
    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker '{self.name}' OPENED after {self.failure_count} failures"
            )
    
    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        with self.lock:
            self.state = CircuitState.HALF_OPEN
            logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state (testing recovery)")
    
    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state (reset)."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            logger.info(f"Circuit breaker '{self.name}' CLOSED (service recovered)")
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self.lock:
            self._transition_to_closed()
            logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    def get_state(self) -> dict:
        """
        Get current circuit breaker state.
        
        Returns:
            Dictionary with state information
        """
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'recovery_timeout': self.recovery_timeout
        }


# ========================================
# Pre-configured Circuit Breakers
# ========================================

# Database circuit breaker
database_breaker = CircuitBreaker(
    name='postgresql',
    failure_threshold=5,
    recovery_timeout=30.0
)

# Kafka circuit breaker
kafka_breaker = CircuitBreaker(
    name='kafka',
    failure_threshold=3,
    recovery_timeout=15.0
)

# Email service circuit breaker
email_breaker = CircuitBreaker(
    name='email_service',
    failure_threshold=3,
    recovery_timeout=60.0
)


def get_all_circuit_states() -> dict:
    """
    Get state of all circuit breakers.
    
    Returns:
        Dictionary mapping circuit breaker names to their states
        
    Example:
        >>> states = get_all_circuit_states()
        >>> print(states['postgresql']['state'])
        'closed'
    """
    return {
        breaker.name: breaker.get_state()
        for breaker in [database_breaker, kafka_breaker, email_breaker]
    }
