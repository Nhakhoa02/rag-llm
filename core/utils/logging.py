"""
Logging configuration and utilities for the RAG LLM system.
"""

import os
import sys
import logging
import structlog
from pathlib import Path
from typing import Optional, Dict, Any
from rich.logging import RichHandler
from rich.console import Console

from config.config import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Setup structured logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json, text)
        log_file: Optional log file path
    """
    # Use settings if not provided
    log_level = log_level or settings.monitoring.log_level
    log_format = log_format or settings.monitoring.log_format
    log_file = log_file or settings.monitoring.log_file
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.getLogger().addHandler(file_handler)
    
    # Add rich handler for development
    if settings.deployment.environment == "development":
        console = Console()
        rich_handler = RichHandler(console=console, rich_tracebacks=True)
        rich_handler.setLevel(getattr(logging, log_level.upper()))
        logging.getLogger().addHandler(rich_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger instance for this class."""
        return get_logger(self.__class__.__name__)
    
    def log_method_call(self, method_name: str, **kwargs) -> None:
        """Log method call with parameters."""
        self.logger.debug(
            "Method called",
            method=method_name,
            parameters=kwargs,
            class_name=self.__class__.__name__
        )
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Log error with context."""
        self.logger.error(
            "Error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            class_name=self.__class__.__name__
        )
    
    def log_metric(self, metric_name: str, value: float, **kwargs) -> None:
        """Log metric with additional context."""
        self.logger.info(
            "Metric recorded",
            metric_name=metric_name,
            value=value,
            **kwargs
        ) 