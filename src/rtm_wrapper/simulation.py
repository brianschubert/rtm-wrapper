from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import xarray as xr
from typing_extensions import TypeAlias

SweepParameter: TypeAlias = tuple[str, Sequence[Any]]
SweepScript: TypeAlias = list[Union[SweepParameter, list[SweepParameter]]]


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

    apparent_radiance: float


class SweepSimulation:
    """
    Sweep specification over model inputs.

    Temporary / unstable representation.
    """

    sweep_grid: xr.DataArray

    def __init__(self, script: SweepScript, base: Inputs) -> None:
        sweep_coords = _script2coords(script)

        # TODO maybe tidy.
        # We create an intermediate empty dataset to validate the sweep coordinates
        # and to resolve the dimension sizes.
        # Unfortunately, while xr.DataArray supports the creation of empty/default-filled
        # arrays, it breaks when there are dimensions without coordinates.
        resolve_dataset = xr.Dataset(coords=sweep_coords)
        self.sweep_grid = xr.DataArray(
            np.empty(list(resolve_dataset.dims.values()), dtype=object),
            coords=resolve_dataset.coords,
        )

        # Populate sweep grid with input combinations.
        with np.nditer(
            self.sweep_grid,
            flags=["multi_index", "refs_ok"],
            op_flags=["writeonly"],
        ) as it:
            for x in it:
                overrides = {
                    k: v.item()
                    for k, v in self.sweep_grid[it.multi_index].coords.items()
                }
                x[...] = dataclasses.replace(base, **overrides)

    @property
    def sweep_size(self) -> int:
        return self.sweep_grid.data.size

    @property
    def sweep_shape(self) -> tuple[int, ...]:
        return self.sweep_grid.data.shape


def _script2coords(script: SweepScript) -> dict[str, tuple[str, Sequence[Any]]]:
    """
    Convert sweep script format to corresponding xarray coordinates.

    This function does not check the validity of the generated coordinates. We let
    xarray handle that on its own.
    """
    coords = {}

    for stage_idx, raw_stage in enumerate(script):
        stage_name = f"stage{stage_idx}"
        if isinstance(raw_stage, tuple):
            stage = [raw_stage]
        else:
            stage = raw_stage

        for param_name, param_values in stage:
            if param_name in coords:
                raise ValueError(
                    f"parameter '{param_name}' set more than once in sweep script. "
                    f"First appeared in {coords[param_name][0]}. "
                    f"Appeared again in {stage_name}"
                )

            coords[param_name] = (stage_name, param_values)

    return coords


class SweepExecutor:
    def run(self, inputs: SweepSimulation):
        ...

    def collect_results(self) -> Any:
        ...


class SerialExecutor(SweepExecutor):
    pass
