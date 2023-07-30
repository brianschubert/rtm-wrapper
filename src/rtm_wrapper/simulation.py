from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import xarray as xr


@dataclass
class Inputs:
    """
    Common input specification for RTM simulations.

    Temporary / unstable representation.
    """

    water: float

    ozone: float

    aot: list[tuple[float, float]]

    wavelength: float


@dataclass
class Outputs:
    """
    Common output format for RTM simulations.

    Temporary / unstable representation.
    """

    dataset: xr.Dataset


class SweepSimulation:
    """
    Sweep specification over model inputs.
    """

    base: Inputs

    sweep: xr.Dataset

    def __init__(self, base: Inputs, sweep: dict[str, Any]) -> None:
        self.base = base
        self.sweep = xr.Dataset(coords=sweep)

    @property
    def sweep_size(self) -> int:
        return math.prod(self.sweep_shape)

    @property
    def sweep_shape(self) -> tuple[int, ...]:
        return tuple(self.sweep.coords.dims.values())


class SweepExecutor:
    def run(self, inputs: SweepSimulation):
        ...

    def collect_results(self) -> Any:
        ...


class SerialExecutor(SweepExecutor):
    pass
