from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Union

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

    alt_sensor: float
    """Altitude of sensor in km."""

    alt_target: float
    """Altitude of target in km."""

    atmosphere: Literal["MidlatitudeSummer"] | tuple[float, float]
    """
    Atmosphere profile. Either standard profile name or tuple of the form
    (
        total water along vertical path in g/cm^2, 
        total ozone in along vertical path in cm-atm
    )
    """

    aerosol_profile: Literal["Maritime", "Continental"]
    """
    Aerosol profile, given as standard name.
    """

    aerosol_aot: list[tuple[float, float]] | None
    """
    Detailed AOT profile, given as list of tuples of the form (layer thickness, layer aot).
    """

    refl_background: float | np.ndarray
    """
    Reflectance of background. 
    
    Either float for spectrally-constant reflectance or an Nx2 array with
    wavelengths in the first column and reflectances in the seconds column.
    """

    refl_target: float | np.ndarray
    """
    Reflectance of target. 
    
    Either float for spectrally-constant reflectance or an Nx2 array with
    wavelengths in the first column and reflectances in the seconds colum
    """

    wavelength: float
    """Simulated wavelength."""


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
            np.empty(tuple(resolve_dataset.dims.values()), dtype=object),
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
