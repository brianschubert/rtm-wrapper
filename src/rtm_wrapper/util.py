"""
Misc utilities.
"""
from __future__ import annotations

import importlib.metadata
import logging.config
import platform
import subprocess
from typing import Any, Callable, Hashable, Iterable, TypeVar

from typing_extensions import Never

_T = TypeVar("_T")
_H = TypeVar("_H", bound=Hashable)


class TrapCalledError(RuntimeError):
    """Raised when a trap callable is invoked."""

    message: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, message, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.message = message
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return f"Trap called: {self.message}."


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
    dictionary: dict[_H, _T], predicate: Callable[[_H], bool]
) -> tuple[dict[_H, _T], dict[_H, _T]]:
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


def platform_summary() -> str:
    return f"{platform.python_implementation()} {platform.python_version()} ({' '.join(platform.uname())})"


def first_or(iterable: Iterable[_T], default: _T | None = None) -> _T | None:
    try:
        return next(iter(iterable))
    except StopIteration:
        return default


def trap(message: str) -> Callable[..., Never]:
    """Return a trap callable that raises when called."""

    def _raise(*args: Any, **kwargs: Any) -> Never:
        raise TrapCalledError(message, args, kwargs)

    return _raise
