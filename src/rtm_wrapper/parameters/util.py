"""
Misc utilities related to parameter handling.
"""

from __future__ import annotations

from typing import TypedDict

from typing_extensions import NotRequired


class ParameterError(Exception):
    """Raised on invalid parameter access."""


class UnsetParameterError(ParameterError):
    """Raised on attempt to access an unset parameter."""


class MetadataDict(TypedDict):
    """Metadata dictionary containing an optional title and unit."""

    title: NotRequired[str]
    """Optional title."""
    unit: NotRequired[str]
    """Optional unit."""
