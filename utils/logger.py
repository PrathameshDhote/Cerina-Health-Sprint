"""
Centralized logging configuration for the application.

Provides structured logging with JSON formatting for production
and human-readable formatting for development.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    from pythonjsonlogger import jsonlogger
    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False
    print("Warning: pythonjsonlogger not installed. Falling back to standard logging.")

from config import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format the message
        result = super().format(record)
        
        # Reset levelname for next use
        record.levelname = levelname
        
        return result


def setup_logger(name: str = "cerina", level: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name
        level: Logging level (defaults to settings.log_level)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level
    log_level = level or settings.log_level
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Choose formatter based on environment
    if settings.is_production and HAS_JSON_LOGGER:
        # JSON formatter for production (easier to parse by log aggregators)
        json_formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(json_formatter)
    else:
        # Human-readable formatter for development
        if settings.enable_debug_logging:
            # Colored formatter for better readability
            formatter = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        else:
            # Simple formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler (optional, for persistent logs)
    if settings.enable_debug_logging:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create daily log file
        log_filename = f"cerina_foundry_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_dir / log_filename)
        file_handler.setLevel(logging.DEBUG)
        
        # Use detailed formatter for file logs
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logger()


# Convenience functions

def log_exception(exc: Exception, context: Optional[str] = None):
    """
    Log an exception with context.
    
    Args:
        exc: Exception to log
        context: Optional context description
    """
    if context:
        logger.error(f"{context}: {type(exc).__name__}: {str(exc)}", exc_info=True)
    else:
        logger.error(f"{type(exc).__name__}: {str(exc)}", exc_info=True)


def log_api_request(method: str, path: str, status_code: int, duration: float):
    """
    Log API request in structured format.
    
    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration: Request duration in seconds
    """
    logger.info(
        f"API Request",
        extra={
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration * 1000, 2)
        }
    )


def log_agent_activity(agent_name: str, action: str, details: Optional[dict] = None):
    """
    Log agent activity in structured format.
    
    Args:
        agent_name: Name of the agent
        action: Action performed
        details: Optional additional details
    """
    log_data = {
        "agent": agent_name,
        "action": action,
        "timestamp": datetime.now().isoformat()
    }
    
    if details:
        log_data.update(details)
    
    logger.info(f"[{agent_name}] {action}", extra=log_data)


def log_workflow_event(thread_id: str, event_type: str, details: Optional[dict] = None):
    """
    Log workflow event in structured format.
    
    Args:
        thread_id: Thread identifier
        event_type: Type of event
        details: Optional additional details
    """
    log_data = {
        "thread_id": thread_id,
        "event_type": event_type,
        "timestamp": datetime.now().isoformat()
    }
    
    if details:
        log_data.update(details)
    
    logger.info(f"Workflow [{thread_id}]: {event_type}", extra=log_data)


# Context manager for logging blocks

class LogBlock:
    """Context manager for logging a block of operations."""
    
    def __init__(self, operation: str, level: str = "INFO"):
        """
        Initialize log block.
        
        Args:
            operation: Description of the operation
            level: Log level
        """
        self.operation = operation
        self.level = level.upper()
        self.start_time = None
    
    def __enter__(self):
        """Enter the context."""
        self.start_time = datetime.now()
        log_method = getattr(logger, self.level.lower(), logger.info)
        log_method(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            log_method = getattr(logger, self.level.lower(), logger.info)
            log_method(f"Completed: {self.operation} (took {duration:.2f}s)")
        else:
            logger.error(
                f"Failed: {self.operation} (after {duration:.2f}s) - {exc_type.__name__}: {exc_val}",
                exc_info=True
            )
        
        # Don't suppress exceptions
        return False


# Performance logging decorator

def log_performance(operation_name: Optional[str] = None):
    """
    Decorator to log function performance.
    
    Args:
        operation_name: Optional name for the operation (defaults to function name)
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Performance: {op_name} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Performance: {op_name} failed after {duration:.3f}s - {str(e)}")
                raise
        
        return wrapper
    return decorator
