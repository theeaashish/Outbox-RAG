import logging
import logging.config

from app.core.config import settings

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": ("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": settings.log_level,
    },
}


def configure_logging() -> None:
    """Configure logging"""
    logging.config.dictConfig(LOGGING_CONFIG)
