"""
Misc utilities.
"""
import importlib.metadata
import logging.config
import subprocess
from typing import Callable, Hashable, TypeVar

_T = TypeVar("_T", bound=Hashable)
_V = TypeVar("_V")


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


def partition_dict(
    dictionary: dict[_T, _V], predicate: Callable[[_T], bool]
) -> tuple[dict[_T, _V], dict[_T, _V]]:
    """
    Partition the given dictionary based on the provided predicate.

    >>> d = {i: i**2 for i in range(6)}
    >>> partition_dict(d, lambda x: x % 2 == 0)
    ({0: 0, 2: 4, 4: 16}, {1: 1, 3: 9, 5: 25})
    """
    left_dict = {}
    right_dict = {}
    for key, value in dictionary.items():
        if predicate(key):
            left_dict[key] = value
        else:
            right_dict[key] = value

    return left_dict, right_dict


def build_version() -> str:
    base_version = importlib.metadata.version("rtm-wrapper")
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            check=True,
            capture_output=True,
        )
        build_commit = result.stdout.strip()
        return f"{base_version}+{build_commit}"
    except (FileNotFoundError, subprocess.SubprocessError):
        return base_version
