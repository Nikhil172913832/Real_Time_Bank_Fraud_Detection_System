"""
Centralized Logging Configuration for Fraud Detection System
============================================================

This module provides centralized logging configuration with proper
formatters, handlers, and log levels for consistent logging across all modules.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from config import Config


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'  # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        result = super().format(record)
        
        # Reset levelname to avoid issues with other handlers
        record.levelname = levelname
        
        return result


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_rotating: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure and return a logger with file and console handlers.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        log_file: Path to log file (optional, uses config if not provided)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        enable_rotating: Whether to use rotating file handler
        max_bytes: Maximum size of log file before rotation (if rotating enabled)
        backup_count: Number of backup files to keep (if rotating enabled)
    
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Starting fraud detection")
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Set log level from config or parameter
    log_level = level or Config.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if enable_file:
        # Use provided log file or fall back to config
        log_file_path = log_file or Config.LOG_FILE
        
        # Create logs directory if it doesn't exist
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        
        if enable_rotating:
            # Rotating file handler (automatically rotates when file gets too large)
            file_handler: logging.Handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            # Simple file handler
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Prevent log messages from being propagated to the root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with default configuration.
    
    This is a convenience function that creates a logger with standard settings.
    For custom configuration, use setup_logger() directly.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
        
    Example:
        >>> from logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing transaction")
    """
    return setup_logger(name)


def configure_root_logger() -> None:
    """
    Configure the root logger to catch all unhandled log messages.
    
    This should be called once at application startup to ensure all
    log messages are properly captured, even from third-party libraries.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Only show warnings and above from third-party libs
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # File handler for root logger
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/root.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(file_handler)


# Configure root logger when module is imported
configure_root_logger()
