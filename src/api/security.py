"""
API Rate Limiting and Authentication
====================================

Rate limiting and JWT authentication for API endpoints.
"""

import logging
import jwt
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from functools import wraps
from flask import request, jsonify
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per hour", "20 per minute"]
)


class JWTAuth:
    """
    JWT authentication handler.
    
    Provides token generation, validation, and user authentication.
    """
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Initialize JWT authentication.
        
        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm (default: HS256)
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def generate_token(
        self,
        user_id: str,
        expiration_hours: int = 24,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate JWT token.
        
        Args:
            user_id: User identifier
            expiration_hours: Token expiration time in hours
            additional_claims: Additional claims to include
            
        Returns:
            JWT token string
        """
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=expiration_hours),
            "iat": datetime.utcnow()
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Generated token for user {user_id}")
        
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            jwt.ExpiredSignatureError: If token is expired
            jwt.InvalidTokenError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            raise
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise
    
    def require_auth(self, func):
        """
        Decorator to require authentication for endpoint.
        
        Usage:
            @app.route('/protected')
            @jwt_auth.require_auth
            def protected_endpoint():
                return jsonify({"message": "Success"})
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get token from Authorization header
            auth_header = request.headers.get('Authorization')
            
            if not auth_header:
                return jsonify({"error": "No authorization header"}), 401
            
            try:
                # Extract token (format: "Bearer <token>")
                token = auth_header.split(" ")[1]
                payload = self.verify_token(token)
                
                # Add user info to request context
                request.user_id = payload.get("user_id")
                request.token_payload = payload
                
                return func(*args, **kwargs)
                
            except IndexError:
                return jsonify({"error": "Invalid authorization header format"}), 401
                
            except jwt.ExpiredSignatureError:
                return jsonify({"error": "Token expired"}), 401
                
            except jwt.InvalidTokenError:
                return jsonify({"error": "Invalid token"}), 401
        
        return wrapper


class APIKeyAuth:
    """
    Simple API key authentication.
    
    Alternative to JWT for simpler use cases.
    """
    
    def __init__(self, valid_keys: set):
        """
        Initialize API key authentication.
        
        Args:
            valid_keys: Set of valid API keys
        """
        self.valid_keys = valid_keys
    
    def require_api_key(self, func):
        """
        Decorator to require API key for endpoint.
        
        Usage:
            @app.route('/protected')
            @api_key_auth.require_api_key
            def protected_endpoint():
                return jsonify({"message": "Success"})
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            
            if not api_key:
                return jsonify({"error": "No API key provided"}), 401
            
            if api_key not in self.valid_keys:
                logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
                return jsonify({"error": "Invalid API key"}), 401
            
            return func(*args, **kwargs)
        
        return wrapper


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


def custom_rate_limit(
    calls: int,
    period: int,
    key_func=None
):
    """
    Custom rate limiting decorator.
    
    Args:
        calls: Number of calls allowed
        period: Time period in seconds
        key_func: Function to extract rate limit key (default: IP address)
        
    Example:
        @custom_rate_limit(calls=10, period=60)
        def endpoint():
            pass
    """
    if key_func is None:
        key_func = get_remote_address
    
    # Simple in-memory store (use Redis in production)
    rate_limit_store: Dict[str, list] = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = key_func(request)
            current_time = time.time()
            
            # Initialize or clean old entries
            if key not in rate_limit_store:
                rate_limit_store[key] = []
            
            # Remove old timestamps
            rate_limit_store[key] = [
                ts for ts in rate_limit_store[key]
                if current_time - ts < period
            ]
            
            # Check rate limit
            if len(rate_limit_store[key]) >= calls:
                logger.warning(f"Rate limit exceeded for {key}")
                return jsonify({
                    "error": "Rate limit exceeded",
                    "retry_after": period
                }), 429
            
            # Add current timestamp
            rate_limit_store[key].append(current_time)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Example usage
def setup_api_security(app, secret_key: str):
    """
    Setup API security (rate limiting and authentication).
    
    Args:
        app: Flask application
        secret_key: Secret key for JWT
    """
    # Initialize rate limiter
    limiter.init_app(app)
    
    # Initialize JWT auth
    jwt_auth = JWTAuth(secret_key)
    
    # Example protected endpoint
    @app.route('/api/token', methods=['POST'])
    @limiter.limit("5 per minute")
    def get_token():
        """Generate JWT token (login endpoint)."""
        data = request.get_json()
        user_id = data.get('user_id')
        password = data.get('password')
        
        # Validate credentials (implement your logic)
        if not user_id or not password:
            return jsonify({"error": "Missing credentials"}), 400
        
        # Generate token
        token = jwt_auth.generate_token(user_id)
        
        return jsonify({
            "token": token,
            "expires_in": 24 * 3600  # 24 hours
        })
    
    return jwt_auth
