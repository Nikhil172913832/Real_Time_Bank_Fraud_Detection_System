"""
Circuit Breaker Pattern Implementation
======================================

Robust error handling with circuit breaker, retry logic, and graceful degradation.
"""

import logging
import time
from typing import Callable, Any, Optional
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    Prevents cascading failures by failing fast when a service is down.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are rejected immediately
    - HALF_OPEN: Testing if service has recovered
    
    Example:
        >>> breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        >>> 
        >>> @breaker
        >>> def call_external_service():
        >>>     # Make external call
        >>>     pass
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting recovery (HALF_OPEN)
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker HALF_OPEN for {func.__name__}")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN for {func.__name__}"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker recovered, setting to CLOSED")
        
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker OPEN after {self.failure_count} failures"
            )
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        logger.info("Circuit breaker manually reset")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for exponential delay
        exceptions: Tuple of exceptions to catch
        
    Example:
        >>> @retry(max_attempts=3, delay=1.0, backoff=2.0)
        >>> def unstable_function():
        >>>     # May fail occasionally
        >>>     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}), "
                        f"retrying in {current_delay:.1f}s: {e}"
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
            
        return wrapper
    return decorator


class GracefulDegradation:
    """
    Graceful degradation handler.
    
    Provides fallback behavior when primary service fails.
    """
    
    @staticmethod
    def with_fallback(primary: Callable, fallback: Callable) -> Callable:
        """
        Execute primary function, fall back to fallback on failure.
        
        Args:
            primary: Primary function to try
            fallback: Fallback function if primary fails
            
        Returns:
            Decorated function
        """
        @wraps(primary)
        def wrapper(*args, **kwargs):
            try:
                return primary(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Primary function {primary.__name__} failed: {e}. "
                    f"Using fallback {fallback.__name__}"
                )
                return fallback(*args, **kwargs)
        
        return wrapper
    
    @staticmethod
    def with_default(func: Callable, default: Any) -> Callable:
        """
        Execute function, return default value on failure.
        
        Args:
            func: Function to execute
            default: Default value to return on failure
            
        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Function {func.__name__} failed: {e}. "
                    f"Returning default value: {default}"
                )
                return default
        
        return wrapper


# Example usage for database connections
db_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    timeout=60,
    expected_exception=Exception
)

# Example usage for external API calls
api_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    timeout=30,
    expected_exception=Exception
)
