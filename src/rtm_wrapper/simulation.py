from __future__ import annotations

import copy
import math
import operator
import typing
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final, Literal, Union

import numpy as np
import xarray as xr
from typing_extensions import TypeAlias

from rtm_wrapper import util
from rtm_wrapper.parameters import (
    MetadataDict,
    Parameter,
    ParameterError,
    ParameterField,
)

InputTopName: TypeAlias = Literal[
    "altitude_sensor",
    "altitude_target",
    "atmosphere",
    "aerosol_profile",
    "ground",
    "geometry",
    "wavelength",
]
INPUT_TOP_NAMES: Final[frozenset[InputTopName]] = frozenset(
    typing.get_args(InputTopName)
)

# OutputName: TypeAlias = Literal[
#     "apparent_radiance",
#     "transmittance_scattering_down",
#     "transmittance_scattering_up",
#     "transmittance_direct_down",
#     "transmittance_direct_up",
#     "transmittance_diffuse_down",
#     "transmittance_diffuse_up",
#     "transmittance_total_gas",
#     "total_transmission",
#     "spherical_albedo",
#     "single_scattering_albedo",
#     "solar_spectrum",
#     "direct_solar_irradiance",
#     "diffuse_solar_irradiance",
# ]
# OUTPUT_NAMES: Final[frozenset[OutputName]] = frozenset(typing.get_args(OutputName))

ParameterValues: TypeAlias = Sequence[Any]
SweepScript: TypeAlias = dict[str, Union[ParameterValues, dict[str, Any]]]

_PARAMETER_AXES_SEP: Final[str] = "/"


class Inputs(Parameter):
    """
    Common input specification for RTM simulations.

    Temporary / unstable representation.
    """

    altitude_sensor = ParameterField(Parameter)

    altitude_target = ParameterField(Parameter)

    atmosphere = ParameterField(Parameter)

    aerosol_profile = ParameterField(Parameter)

    geometry = ParameterField(Parameter)

    ground = ParameterField(Parameter)

    wavelength = ParameterField(Parameter)


# @dataclass
# class Outputs:
#     """
#     Common output format for RTM simulations.
#
#     Temporary / unstable representation.
#     """
#
#     apparent_radiance: Annotated[
#         float, MetadataDict(title="Apparent Radiance", unit="W/sr-m^2")
#     ]
#
#     transmittance_scattering_down: Annotated[
#         float, MetadataDict(title="Downward Scattering", unit="1")
#     ]
#
#     transmittance_scattering_up: Annotated[
#         float, MetadataDict(title="Upward Scattering", unit="1")
#     ]
#
#     transmittance_direct_down: Annotated[
#         float, MetadataDict(title="Direct Downward Transmittance", unit="1")
#     ]
#
#     transmittance_direct_up: Annotated[
#         float, MetadataDict(title="Direct Upward Transmittance", unit="1")
#     ]
#
#     transmittance_diffuse_down: Annotated[
#         float, MetadataDict(title="Diffuse Downward Transmittance", unit="1")
#     ]
#
#     transmittance_diffuse_up: Annotated[
#         float, MetadataDict(title="Diffuse Downward Transmittance", unit="1")
#     ]
#
#     transmittance_total_gas: Annotated[
#         float, MetadataDict(title="Total Gas Transmittance", unit="1")
#     ]
#
#     total_transmission: Annotated[
#         float, MetadataDict(title="Total Transmission", unit="1")
#     ]
#
#     spherical_albedo: Annotated[float, MetadataDict(title="Spherical Albedo")]
#
#     single_scattering_albedo: Annotated[
#         float, MetadataDict(title="Single Scattering Albedo")
#     ]
#
#     solar_spectrum: Annotated[float, MetadataDict(title="Solar Spectrum")]
#
#     direct_solar_irradiance: Annotated[
#         float, MetadataDict(title="Direct Solar irradiance")
#     ]
#
#     diffuse_solar_irradiance: Annotated[
#         float, MetadataDict(title="Diffuse Solar irradiance")
#     ]


class SweepSimulation:
    """
    Sweep specification over model inputs.

    Temporary / unstable representation.
    """

    sweep_spec: xr.Dataset

    base: Inputs

    _input_coords: frozenset[str]

    def __init__(self, script: SweepScript, base: Inputs) -> None:
        sweep_coords = _script2coords(script, base)

        # Create an empty dataset to validate the sweep coordinates
        # and to resolve the dimension sizes.
        self.sweep_spec = xr.Dataset(coords=sweep_coords)
        self.base = base

        # TODO more robust input coordinate detection
        self._input_coords = frozenset(  # type: ignore
            coord
            for coord in self.sweep_spec.coords.keys()
            if any(coord.startswith(top_name) for top_name in INPUT_TOP_NAMES)  # type: ignore
        )

        # Populate sweep grid with input combinations.
        # with np.nditer(
        #     self.sweep_spec.grid,
        #     flags=["multi_index", "refs_ok"],
        #     op_flags=["writeonly"],  # type: ignore
        # ) as it:
        #     for x in it:
        #         overrides = {
        #             k: v.item() if v.size == 1 else v.squeeze()
        #             for k, v in self.sweep_spec.isel(
        #                 {
        #                     dim: index
        #                     for dim, index in zip(
        #                         self.sweep_spec.grid.dims, it.multi_index
        #                     )
        #                 }
        #             ).coords.items()
        #             if k in input_coords
        #         }
        #         x[...] = base.replace(overrides)  # type: ignore

    def __getitem__(self, item: tuple[int, ...]) -> Inputs | np.ndarray:
        overrides = {
            k: v.item() if v.size == 1 else v.squeeze()
            for k, v in self.sweep_spec.isel(
                {dim: index for dim, index in zip(self.dims.keys(), item)}
            ).coords.items()
            if k in self._input_coords
        }
        return self.base.replace(overrides)

    @property
    def dims(self) -> Mapping[str, int]:
        return self.sweep_spec.indexes.dims  # type: ignore

    @property
    def sweep_size(self) -> int:
        return math.prod(self.sweep_shape)

    @property
    def sweep_shape(self) -> tuple[int, ...]:
        return tuple(self.dims.values())

    def split(
        self, sections: int | Sequence[int], dim: str
    ) -> Iterable[SweepSimulation]:
        # TODO: decide on default value, if any.
        # if dim is None:
        # Pick first.
        # dim = next(iter(self.dims.keys()))
        # Pick largest.
        # dim = max(self.dims.items(), key=operator.itemgetter(1))[0]

        try:
            indices = np.arange(self.dims[dim])
        except KeyError:
            raise ValueError(
                f"invalid dim '{dim}' - must be one of {list(self.dims.keys())}"
            )

        section_indices = np.array_split(indices, sections)

        for sec_idx in section_indices:
            # Make shallow copy.
            sweep_section = copy.copy(self)
            sweep_section.sweep_spec = self.sweep_spec.isel({dim: sec_idx})
            yield sweep_section


_CoordsDict: TypeAlias = dict[str, tuple[Sequence[str], Sequence[Any], MetadataDict]]


def _script2coords(script: SweepScript, base: Inputs) -> _CoordsDict:
    """
    Convert sweep script format to corresponding xarray coordinates.

    This function does not check the validity of the generated coordinates. We let
    xarray handle that on its own.
    """
    coords: _CoordsDict = {}

    for sweep_name, sweep_spec in script.items():
        if isinstance(sweep_spec, dict):
            # This sweep axes is a "compound" dimension that varies multiple parameters
            # and/or includes custom attributes.
            if sweep_name in INPUT_TOP_NAMES:
                raise ValueError(
                    f"compound sweep axes '{sweep_name}' must not be an input parameter name"
                )

            attribute_parts, sweep_parameters = util.partition_dict(
                sweep_spec, _is_special
            )

            # Assume at least one parameter was not specific, and that all parameter values
            # have the same length.
            sweep_len = len(next(iter(sweep_parameters.values())))

            coords[sweep_name] = (
                (sweep_name,),
                attribute_parts.get("__coords__", np.arange(sweep_len)),
                {  # type: ignore
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

            try:
                attrs = base.get_metadata(param_path)
            except ParameterError as ex:
                raise ParameterError(f"invalid parameter '{param_path}': {ex}")

            # TODO get dtype from field
            param_coordinates = np.asarray(param_values)
            dims = [sweep_name]
            if param_coordinates.ndim != 1:
                dims += [
                    f"{param_path}{_PARAMETER_AXES_SEP}{i}"
                    for i in range(param_coordinates.ndim - 1)
                ]

            coords[param_path] = (dims, param_coordinates, attrs)  # type: ignore

    return coords


def _is_special(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _is_grid_coord(coord_name: str) -> bool:
    first_is_input = coord_name.partition("__")[0] in INPUT_TOP_NAMES
    not_parameter_expansion = _PARAMETER_AXES_SEP not in coord_name
    return first_is_input and not_parameter_expansion


# Verify that the available name globals match their corresponding dataclass fields.
# This is assumed in several places.
# if not {f.name for f in dataclasses.fields(Inputs)} == set(INPUT_TOP_NAMES):
#     raise ImportError(
#         "Detected misconfiguration in the available input names. This is a bug."
#     )
# if not {f.name for f in dataclasses.fields(Outputs)} == set(OUTPUT_NAMES):
#     raise ImportError(
#         "Detected misconfiguration in the available output names. This is a bug."
#     )
