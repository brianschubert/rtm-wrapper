from __future__ import annotations

import importlib.resources
import pathlib
from importlib.abc import Traversable
from typing import Final, TYPE_CHECKING

if TYPE_CHECKING:
    from Py6S import SixS

_RESOURCE_ROOT: Final[Traversable] = importlib.resources.files(__package__)

_SIXS_BIN: Final[Traversable] = _RESOURCE_ROOT / f"sixsV1.1"


def sixs_bin() -> pathlib.Path:
    if not isinstance(_SIXS_BIN, pathlib.Path):
        raise RuntimeError(
            f"6S binary package resource represented as non-path resource: {_SIXS_BIN}"
        )

    return _SIXS_BIN


def make_wrapper() -> SixS:
    try:
        from Py6S import SixS
    except ImportError as ex:
        raise RuntimeError("Py6S not installed") from ex
    return SixS(str(sixs_bin()))


def test_wrapper() -> None:
    try:
        from Py6S import SixS
    except ImportError as ex:
        raise RuntimeError("Py6S not installed") from ex
    SixS.test(sixs_bin())
