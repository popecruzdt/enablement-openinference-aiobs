"""
Utility functions for AI Travel Advisor Application
Includes logging configuration and common helper functions
"""

import logging
import logging.config
import json
from pathlib import Path
from typing import Dict, Any

from src.config import get_settings


def setup_logging() -> None:
    """
    Configure application logging with JSON format for observability
    """
    settings = get_settings()
    
    # Check if pythonjsonlogger is available for JSON logging
    try:
        from pythonjsonlogger import jsonlogger
        use_json = True
    except ImportError:
        use_json = False

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Override uvicorn's loggers so they route through our handler (stdout).
    # Uvicorn installs its own stderr StreamHandlers before the app is imported;
    # setting propagate=False and pointing them at our handler fixes that.
    uvicorn_loggers = {
        "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error":  {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
    }

    if use_json:
        # JSON logging configuration
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                    "rename_fields": {
                        "asctime": "timestamp",
                        "levelname": "level"
                    },
                    "datefmt": "%Y-%m-%dT%H:%M:%S%z"
                },
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "default": {
                    "level": log_level,
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": log_level,
                "handlers": ["default"],
            },
            "loggers": uvicorn_loggers,
        }
    else:
        # Standard logging configuration (fallback when pythonjsonlogger unavailable)
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "default": {
                    "level": log_level,
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": log_level,
                "handlers": ["default"],
            },
            "loggers": uvicorn_loggers,
        }

    logging.config.dictConfig(logging_config)


def format_rag_response(response: str, max_length: int | None = None) -> Dict[str, Any]:
    """
    Format RAG response for API consumption
    
    Args:
        response: The response from the RAG chain
        max_length: Maximum length for truncation (optional)
        
    Returns:
        Dict with formatted response
    """
    if max_length and len(response) > max_length:
        response = response[:max_length] + "..."
    
    return {
        "message": response,
        "status": "success",
        "type": "travel_advice"
    }


def format_error_response(error: str, status_code: int = 500) -> Dict[str, Any]:
    """
    Format error response for API consumption
    
    Args:
        error: Error message
        status_code: HTTP status code
        
    Returns:
        Dict with formatted error
    """
    return {
        "message": error,
        "status": "error",
        "status_code": status_code
    }
