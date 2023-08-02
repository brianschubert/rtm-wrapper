from __future__ import annotations

import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

import numpy as np
import xarray as xr
from typing_extensions import TypeAlias

from rtm_wrapper import util
from rtm_wrapper.parameters import Parameter

InputParameterName: TypeAlias = Literal[
    "altitude_sensor",
    "altitude_target",
    "atmosphere",
    "aerosol_profile",
    "ground",
    "wavelength",
]
ParameterValues: TypeAlias = Sequence[Any]
SweepScript: TypeAlias = dict[str, Union[ParameterValues, dict[str, Any]]]


@dataclass
class Inputs(Parameter):
    """
    Common input specification for RTM simulations.

    Temporary / unstable representation.
    """

    altitude_sensor: Parameter

    altitude_target: Parameter

    atmosphere: Parameter

    aerosol_profile: Parameter

    ground: Parameter

    wavelength: Parameter

    # OLD INTERFACE
    # alt_sensor: Annotated[
    #     Union[float, Literal["sealevel", "satellite"]],
    #     "Sensor Altitude",
    #     "kilometers",
    # ]
    # """Altitude of sensor. Either predefined or km the open interval (0, 100)."""
    #
    # # TODO verify that any target altitude really is allowable.
    # alt_target: Annotated[
    #     Union[float, Literal["sealevel"]], "Target Altitude", "kilometers"
    # ]
    # """
    # Altitude of target. Either predefined or km (any non-negative float).
    # """
    #
    # atmosphere: Annotated[
    #     Union[Literal["MidlatitudeSummer"], tuple[float, float]],
    #     "Atmosphere",
    #     None,
    # ]
    # """
    # Atmosphere profile. Either standard profile name or tuple of the form
    # (
    #     total water along vertical path in g/cm^2,
    #     total ozone in along vertical path in cm-atm
    # )
    # """
    #
    # aerosol_profile: Annotated[
    #     Literal["Maritime", "Continental"], "Aerosol Profile", None
    # ]
    # """
    # Aerosol profile, given as standard name.
    # """
    #
    # aerosol_aot: Annotated[Optional[list[tuple[float, float]]], "AOT Layers", None]
    # """
    # Detailed AOT profile, given as list of tuples of the form (layer thickness, layer aot).
    # """
    #
    # refl_background: Annotated[Union[float, np.ndarray], "Background Reflectance", None]
    # """
    # Reflectance of background.
    #
    # Either float for spectrally-constant reflectance or an Nx2 array with
    # wavelengths in the first column and reflectances in the seconds column.
    # """
    #
    # refl_target: Annotated[Union[float, np.ndarray], "Target Reflectance", None]
    # """
    # Reflectance of target.
    #
    # Either float for spectrally-constant reflectance or an Nx2 array with
    # wavelengths in the first column and reflectances in the seconds colum
    # """
    #
    # wavelength: Annotated[float, "Wavelength", "micrometers"]
    # """Simulated wavelength."""


@dataclass
class Outputs:
    """
    Common output format for RTM simulations.

    Temporary / unstable representation.
    """

    apparent_radiance: Annotated[float, "Apparent Radiance", "W/sr-m^2"]


class SweepSimulation:
    """
    Sweep specification over model inputs.

    Temporary / unstable representation.
    """

    sweep_spec: xr.Dataset

    def __init__(self, script: SweepScript, base: Inputs) -> None:
        sweep_coords = _script2coords(script, base)
        input_names = typing.get_type_hints(Inputs).keys()

        # TODO maybe tidy.
        # TODO update outdated comment below
        # We create an intermediate empty dataset to validate the sweep coordinates
        # and to resolve the dimension sizes.
        # Unfortunately, while xr.DataArray supports the creation of empty/default-filled
        # arrays, it breaks when there are dimensions without coordinates.
        resolve_dataset = xr.Dataset(coords=sweep_coords)
        sweep_dims = resolve_dataset.indexes.dims
        self.sweep_spec = resolve_dataset.assign(
            grid=(
                tuple(sweep_dims.keys()),
                np.empty(tuple(sweep_dims.values()), dtype=object),
            )
        )

        # Populate sweep grid with input combinations.
        with np.nditer(
            self.sweep_spec.grid,
            flags=["multi_index", "refs_ok"],
            op_flags=["writeonly"],
        ) as it:
            for x in it:
                overrides = {
                    k: v.item() if v.size == 1 else v.squeeze()
                    for k, v in self.sweep_spec.isel(
                        {
                            dim: index
                            for dim, index in zip(
                                self.sweep_spec.grid.dims, it.multi_index
                            )
                        }
                    ).coords.items()
                    if k.partition("__")[0] in input_names and "/" not in k
                }
                x[...] = base.replace(**overrides)

    @property
    def sweep_size(self) -> int:
        return self.sweep_spec.grid.data.size

    @property
    def sweep_shape(self) -> tuple[int, ...]:
        return self.sweep_spec.grid.data.shape


def _script2coords(
    script: SweepScript,
    base: Inputs,
) -> dict[str, tuple[str, Sequence[Any], dict[str, Any]]]:
    """
    Convert sweep script format to corresponding xarray coordinates.

    This function does not check the validity of the generated coordinates. We let
    xarray handle that on its own.
    """
    coords = {}
    top_input_names = typing.get_args(InputParameterName)

    for sweep_name, sweep_spec in script.items():
        if isinstance(sweep_spec, dict):
            # This sweep axes is a "compound" dimension that varies multiple parameters
            # and/or includes custom attributes.
            if sweep_name in top_input_names:
                raise ValueError(
                    f"compound sweep axes '{sweep_name}' must not be an input parameter name"
                )

            attribute_parts, sweep_parameters = util.partition_dict(
                sweep_spec, _is_special
            )

            # Assume at least one parameter was specific, and that all parameter values
            # have the same length.
            sweep_len = len(next(iter(sweep_parameters.values())))

            coords[sweep_name] = (
                sweep_name,
                attribute_parts.get("__coords__", np.arange(sweep_len)),
                {
                    key[2:-2]: value
                    for key, value in attribute_parts.items()
                    if key != "__coords__"
                },
            )
        else:
            # This sweep axes is a "simple" axes that varies a single parameter.
            sweep_parameters = {sweep_name: sweep_spec}

        for param_path, param_values in sweep_parameters.items():
            # TODO parameter path validation and uniqueness checking?
            attrs = base.get_metadata(param_path)

            param_coordinates = np.asarray(param_values)
            dims = [sweep_name]
            if param_coordinates.ndim != 1:
                dims += [f"{param_path}/{i}" for i in range(param_coordinates.ndim - 1)]

            coords[param_path] = (dims, param_coordinates, attrs)

    return coords


def _is_special(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")
