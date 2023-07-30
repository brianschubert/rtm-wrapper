"""
Misc utilities.
"""

import logging.config


def setup_debug_root_logging(level: int = logging.NOTSET) -> None:
    """
    Configure the root logger with a basic debugging configuration.

    All records at the given level or above will be written to stdout.

    This function should be called once near the start of an application entry point,
    BEFORE any calls to ``logging.getLogger`` are made.

    Disables any existing loggers.
    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "console": {
                    "format": "[{asctime},{msecs:06.2f}] {levelname:7s} ({threadName}:{name}) {funcName}:{lineno} {message}",
                    "style": "{",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                    "validate": True,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "console",
                    "level": "NOTSET",  # Capture everything.
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"handlers": ["console"], "level": level},
        }
    )
