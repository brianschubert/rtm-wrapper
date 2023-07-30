from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
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

    Temporary / unstable representation.
    """

    base: Inputs

    sweep_grid: xr.DataArray

    def __init__(self, sweep: dict[str, Any], base: Inputs) -> None:
        self.base = base

        # For now, only support dense cartesian product sweeps of input parameters.
        dims = list(sweep)
        sweep_shape = [len(param) for param in sweep.values()]
        input_grid = np.full(sweep_shape, fill_value=None, dtype=object)

        with np.nditer(
            input_grid, flags=["multi_index", "refs_ok"], op_flags=["writeonly"]
        ) as it:
            for x in it:
                overrides = {}
                for idx, dim in zip(it.multi_index, dims):
                    overrides[dim] = sweep[dim][idx]

                x[...] = dataclasses.replace(base, **overrides)

        self.sweep_grid = xr.DataArray(input_grid, coords=sweep)

    @property
    def sweep_size(self) -> int:
        return self.sweep_grid.data.size

    @property
    def sweep_shape(self) -> tuple[int, ...]:
        return self.sweep_grid.data.shape


class SweepExecutor:
    def run(self, inputs: SweepSimulation):
        ...

    def collect_results(self) -> Any:
        ...


class SerialExecutor(SweepExecutor):
    pass
