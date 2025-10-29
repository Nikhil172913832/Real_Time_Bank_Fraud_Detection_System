"""
Database Utilities for Fraud Detection System
=============================================

This module provides database connection pooling, context managers,
and retry logic for robust PostgreSQL interactions.
"""

import time
from contextlib import contextmanager
from typing import Generator, Optional, Any, Dict, List
import psycopg2
from psycopg2 import pool, Error as Psycopg2Error
from psycopg2.extras import execute_batch, RealDictCursor

from config import Config
from exceptions import DatabaseConnectionError, DatabaseQueryError
from logger import get_logger

logger = get_logger(__name__)


class DatabasePool:
    """
    PostgreSQL connection pool manager with automatic reconnection.
    
    This class provides a thread-safe connection pool with configurable
    min/max connections and automatic connection recycling.
    
    Example:
        >>> db_pool = DatabasePool()
        >>> with db_pool.get_connection() as conn:
        ...     with conn.cursor() as cur:
        ...         cur.execute("SELECT * FROM fraud_alerts LIMIT 10")
        ...         results = cur.fetchall()
    """
    
    _instance: Optional['DatabasePool'] = None
    _pool: Optional[pool.ThreadedConnectionPool] = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one pool instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        minconn: int = 1,
        maxconn: int = 20,
        database_url: Optional[str] = None
    ):
        """
        Initialize database connection pool.
        
        Args:
            minconn: Minimum number of connections to maintain
            maxconn: Maximum number of connections allowed
            database_url: PostgreSQL connection string (uses Config if not provided)
        """
        if self._pool is not None:
            return  # Already initialized
        
        # Get database URL from config property
        config_instance = Config()
        self.database_url = database_url or config_instance.DATABASE_URL
        self.minconn = minconn
        self.maxconn = maxconn
        
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=self.minconn,
                maxconn=self.maxconn,
                dsn=self.database_url
            )
            logger.info(f"Database connection pool initialized (min={minconn}, max={maxconn})")
        except Psycopg2Error as e:
            raise DatabaseConnectionError(self.database_url, str(e)) from e
    
    @contextmanager
    def get_connection(self, autocommit: bool = False) -> Generator:
        """
        Get a database connection from the pool.
        
        This is a context manager that automatically returns the connection
        to the pool when done.
        
        Args:
            autocommit: Whether to enable autocommit mode
            
        Yields:
            psycopg2 connection object
            
        Raises:
            DatabaseConnectionError: If connection cannot be obtained
            
        Example:
            >>> with db_pool.get_connection() as conn:
            ...     with conn.cursor() as cur:
            ...         cur.execute("SELECT 1")
        """
        if self._pool is None:
            raise DatabaseConnectionError(self.database_url, "Connection pool not initialized")
        
        conn = None
        try:
            conn = self._pool.getconn()
            if conn is None:
                raise DatabaseConnectionError(self.database_url, "No connections available")
            
            conn.autocommit = autocommit
            yield conn
            
            if not autocommit and not conn.closed:
                conn.commit()
                
        except Psycopg2Error as e:
            if conn and not conn.closed:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise DatabaseQueryError("Unknown", str(e)) from e
        finally:
            if conn is not None:
                self._pool.putconn(conn)
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        if self._pool is not None:
            self._pool.closeall()
            logger.info("All database connections closed")
            self._pool = None


# Global database pool instance
db_pool = DatabasePool()


def execute_query(
    query: str,
    params: Optional[tuple] = None,
    fetch: bool = False,
    fetch_one: bool = False,
    dict_cursor: bool = False,
    retry_count: int = 3,
    retry_delay: float = 1.0
) -> Optional[List[Any]]:
    """
    Execute a SQL query with automatic retry and error handling.
    
    Args:
        query: SQL query string
        params: Query parameters (for parameterized queries)
        fetch: Whether to fetch results
        fetch_one: Whether to fetch only one result
        dict_cursor: Use dictionary cursor (returns dict instead of tuple)
        retry_count: Number of retries on failure
        retry_delay: Delay between retries in seconds
        
    Returns:
        Query results if fetch=True, otherwise None
        
    Raises:
        DatabaseQueryError: If query fails after all retries
        
    Example:
        >>> results = execute_query(
        ...     "SELECT * FROM fraud_alerts WHERE user_id = %s",
        ...     params=('user123',),
        ...     fetch=True,
        ...     dict_cursor=True
        ... )
    """
    last_error = None
    
    for attempt in range(retry_count):
        try:
            cursor_factory = RealDictCursor if dict_cursor else None
            
            with db_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=cursor_factory) as cur:
                    cur.execute(query, params)
                    
                    if fetch_one:
                        return cur.fetchone()
                    elif fetch:
                        return cur.fetchall()
                    else:
                        return None
                        
        except Psycopg2Error as e:
            last_error = e
            logger.warning(
                f"Query failed (attempt {attempt + 1}/{retry_count}): {e}"
            )
            
            if attempt < retry_count - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            continue
    
    # All retries exhausted
    raise DatabaseQueryError(query[:100], str(last_error)) from last_error


def execute_batch_query(
    query: str,
    params_list: List[tuple],
    page_size: int = 100,
    retry_count: int = 3
) -> None:
    """
    Execute a batch of SQL queries efficiently.
    
    This uses psycopg2's execute_batch for better performance when
    inserting or updating multiple rows.
    
    Args:
        query: SQL query string with placeholders
        params_list: List of parameter tuples
        page_size: Number of records to insert per batch
        retry_count: Number of retries on failure
        
    Raises:
        DatabaseQueryError: If batch execution fails
        
    Example:
        >>> execute_batch_query(
        ...     "INSERT INTO fraud_alerts (user_id, amount) VALUES (%s, %s)",
        ...     params_list=[('user1', 100.0), ('user2', 200.0)]
        ... )
    """
    last_error = None
    
    for attempt in range(retry_count):
        try:
            with db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    execute_batch(cur, query, params_list, page_size=page_size)
                    logger.info(f"Batch executed successfully ({len(params_list)} records)")
                    return
                    
        except Psycopg2Error as e:
            last_error = e
            logger.warning(
                f"Batch query failed (attempt {attempt + 1}/{retry_count}): {e}"
            )
            
            if attempt < retry_count - 1:
                time.sleep(1.0 * (attempt + 1))
            continue
    
    raise DatabaseQueryError(query[:100], str(last_error)) from last_error


def insert_fraud_alert(
    transaction_id: str,
    user_id: str,
    amount: float,
    fraud_probability: float,
    shap_values: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Insert a fraud alert into the database.
    
    Args:
        transaction_id: Unique transaction identifier
        user_id: User identifier
        amount: Transaction amount
        fraud_probability: Model's fraud probability (0-1)
        shap_values: SHAP explanation values
        metadata: Additional transaction metadata
        
    Example:
        >>> insert_fraud_alert(
        ...     transaction_id='txn123',
        ...     user_id='user456',
        ...     amount=500.0,
        ...     fraud_probability=0.85,
        ...     shap_values={'amount': 0.3, 'is_international': 0.2}
        ... )
    """
    import json
    
    query = """
        INSERT INTO fraud_alerts 
        (transaction_id, user_id, amount, fraud_probability, shap_values, metadata, detected_at)
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
    """
    
    params = (
        transaction_id,
        user_id,
        amount,
        fraud_probability,
        json.dumps(shap_values) if shap_values else None,
        json.dumps(metadata) if metadata else None
    )
    
    try:
        execute_query(query, params)
        logger.debug(f"Fraud alert inserted for transaction {transaction_id}")
    except DatabaseQueryError as e:
        logger.error(f"Failed to insert fraud alert: {e}")
        raise


def check_database_health() -> Dict[str, Any]:
    """
    Check database connection and basic health.
    
    Returns:
        Dictionary with health status and metrics
        
    Example:
        >>> health = check_database_health()
        >>> print(health['status'])
        'healthy'
    """
    try:
        start_time = time.time()
        execute_query("SELECT 1", fetch_one=True)
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get safe database URL (hide credentials)
        config_instance = Config()
        db_url = config_instance.DATABASE_URL
        safe_url = db_url.split('@')[-1] if '@' in db_url else 'unknown'
        
        return {
            'status': 'healthy',
            'latency_ms': round(latency, 2),
            'database_url': safe_url
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }
