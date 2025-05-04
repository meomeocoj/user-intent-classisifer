"""
Logging configuration and setup.
"""
import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from query_router.core.config import load_config

def setup_logging() -> structlog.BoundLogger:
    """
    Configure structured logging for the application.
    
    Returns:
        A configured structlog logger instance.
    """
    config = load_config()
    log_config = config["logging"]
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=config["app"]["log_level"].upper(),
    )
    
    # Define processors based on config
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Add timestamp if configured
    if log_config.get("include_timestamp", True):
        processors.append(structlog.processors.TimeStamper(fmt="iso"))
    
    # Add JSON formatting for production
    if log_config.get("format", "json") == "json":
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ])
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer()
        ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Create and return logger
    logger = structlog.get_logger()
    
    # Log startup message
    logger.info(
        "logging_initialized",
        log_format=log_config["format"],
        log_level=config["app"]["log_level"],
    )
    
    return logger

def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Optional name for the logger (usually __name__)
        
    Returns:
        A configured structlog logger
    """
    return structlog.get_logger(name) 