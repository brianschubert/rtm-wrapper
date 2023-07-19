from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import Py6S.outputs as sixs_outputs
from nptyping import Float, Int, NDArray, Object
from Py6S import SixS

_TRANSMITTANCE_NAMES: TypeAlias = Literal[
    "aerosol_scattering",
    "ch4",
    "co",
    "co2",
    "global_gas",
    "no2",
    "oxygen",
    "ozone",
    "rayleigh_scattering",
    "total_scattering",
    "water",
]

_RAT_NAMES: TypeAlias = Literal[
    "direction_of_plane_polarization",
    "optical_depth_plane",
    "optical_depth_total",
    "phase_function_I",
    "phase_function_Q",
    "phase_function_U",
    "polarized_reflectance",
    "primary_degree_of_polarization",
    "reflectance_I",
    "reflectance_Q",
    "reflectance_U",
    "single_scattering_albedo",
    "spherical_albedo",
]

_TransmittanceDict: TypeAlias = dict[_TRANSMITTANCE_NAMES, sixs_outputs.Transmittance]
_RatDict: TypeAlias = dict[_RAT_NAMES, sixs_outputs.RayleighAerosolTotal]

_FloatArray: TypeAlias = NDArray[Literal["*"], Float]
_IntArray: TypeAlias = NDArray[Literal["*"], Int]

_TransmittanceArray: TypeAlias = NDArray[
    Literal["*"], Literal["[downward, upward, total]: Float"]
]
_TransmittanceArrayDType = np.dtype(
    [("downward", np.float_), ("upward", np.float_), ("total", np.float_)]
)

_RatArray: TypeAlias = NDArray[
    Literal["*"], Literal["[rayleigh, aerosol, total]: Float"]
]
_RatArrayDType = np.dtype(
    [("rayleigh", np.float_), ("aerosol", np.float_), ("total", np.float_)]
)


class _OutputDict(TypedDict):
    version: str
    month: int
    day: int
    solar_z: int
    solar_a: int
    view_z: int
    view_a: int
    scattering_angle: float
    azimuthal_angle_difference: float
    visibility: float
    aot550: float
    ground_pressure: float
    ground_altitude: float
    apparent_reflectance: float
    apparent_radiance: float
    total_gaseous_transmittance: float
    wv_above_aerosol: float
    wv_mixed_with_aerosol: float
    wv_under_aerosol: float
    apparent_polarized_reflectance: float
    apparent_polarized_radiance: float
    direction_of_plane_of_polarization: float
    total_polarization_ratio: float
    percent_direct_solar_irradiance: float
    percent_diffuse_solar_irradiance: float
    percent_environmental_irradiance: float
    atmospheric_intrinsic_reflectance: float
    background_reflectance: float
    pixel_reflectance: float
    direct_solar_irradiance: float
    diffuse_solar_irradiance: float
    environmental_irradiance: float
    atmospheric_intrinsic_radiance: float
    background_radiance: float
    pixel_radiance: float
    solar_spectrum: float


@dataclass(frozen=True)
class Py6SDenseOutput:
    """
    Dense array representation of Py6S sweep outputs.

    Exposes derived quantities as properties.
    """

    # Values
    version: str
    month: _IntArray
    day: _IntArray
    solar_z: _IntArray
    solar_a: _IntArray
    view_z: _IntArray
    view_a: _IntArray
    scattering_angle: _FloatArray
    azimuthal_angle_difference: _FloatArray
    visibility: _FloatArray
    aot550: _FloatArray
    ground_pressure: _FloatArray
    ground_altitude: _FloatArray
    apparent_reflectance: _FloatArray
    apparent_radiance: _FloatArray
    total_gaseous_transmittance: _FloatArray
    wv_above_aerosol: _FloatArray
    wv_mixed_with_aerosol: _FloatArray
    wv_under_aerosol: _FloatArray
    apparent_polarized_reflectance: _FloatArray
    apparent_polarized_radiance: _FloatArray
    direction_of_plane_of_polarization: _FloatArray
    total_polarization_ratio: _FloatArray
    percent_direct_solar_irradiance: _FloatArray
    percent_diffuse_solar_irradiance: _FloatArray
    percent_environmental_irradiance: _FloatArray
    atmospheric_intrinsic_reflectance: _FloatArray
    background_reflectance: _FloatArray
    pixel_reflectance: _FloatArray
    direct_solar_irradiance: _FloatArray
    diffuse_solar_irradiance: _FloatArray
    environmental_irradiance: _FloatArray
    atmospheric_intrinsic_radiance: _FloatArray
    background_radiance: _FloatArray
    pixel_radiance: _FloatArray
    solar_spectrum: _FloatArray

    # Transmittances
    transmittance_aerosol_scattering: _TransmittanceArray
    transmittance_ch4: _TransmittanceArray
    transmittance_co: _TransmittanceArray
    transmittance_co2: _TransmittanceArray
    transmittance_global_gas: _TransmittanceArray
    transmittance_no2: _TransmittanceArray
    transmittance_oxygen: _TransmittanceArray
    transmittance_ozone: _TransmittanceArray
    transmittance_rayleigh_scattering: _TransmittanceArray
    transmittance_total_scattering: _TransmittanceArray
    transmittance_water: _TransmittanceArray

    # RATs
    direction_of_plane_polarization: _RatArray
    optical_depth_plane: _RatArray
    optical_depth_total: _RatArray
    phase_function_I: _RatArray
    phase_function_Q: _RatArray
    phase_function_U: _RatArray
    polarized_reflectance: _RatArray
    primary_degree_of_polarization: _RatArray
    reflectance_I: _RatArray
    reflectance_Q: _RatArray
    reflectance_U: _RatArray
    single_scattering_albedo: _RatArray
    spherical_albedo: _RatArray

    @classmethod
    def from_py6s(cls, outputs: NDArray[Literal["*"], Object]) -> Py6SDenseOutput:
        if len(outputs) == 0:
            # Given empty output array
            version = "unknown"
        else:
            # Assume same for all entries.
            version = outputs[0].values["version"]

        attrs = {}

        # Extract value arrays.
        for value_key, value_type in typing.get_type_hints(_OutputDict).items():
            if value_key == "version":
                continue
            attrs[value_key] = np.array(
                [out.values[value_key] for out in outputs], dtype=value_type
            )

        # Extract transmittance arrays.
        for trans_key in _TRANSMITTANCE_NAMES.__args__:
            attrs[f"transmittance_{trans_key}"] = np.array(
                [
                    (
                        out.trans[trans_key].downward,
                        out.trans[trans_key].upward,
                        out.trans[trans_key].total,
                    )
                    for out in outputs
                ],
                dtype=_TransmittanceArrayDType,
            )

        # Extract RAT arrays.
        for rat_key in _RAT_NAMES.__args__:
            attrs[rat_key] = np.array(
                [
                    (
                        out.rat[rat_key].rayleigh,
                        out.rat[rat_key].aerosol,
                        out.rat[rat_key].total,
                    )
                    for out in outputs
                ],
                dtype=_RatArrayDType,
            )

        return cls(version=version, **attrs)

    @property
    def cos_zenith_solar(self) -> _FloatArray:
        return np.cos(np.deg2rad(self.solar_z))

    @property
    def cos_zenith_view(self) -> _FloatArray:
        return np.cos(np.deg2rad(self.view_z))

    @property
    def transmittance_direct_down(self) -> _FloatArray:
        return np.exp(-self.optical_depth_total["total"] / self.cos_zenith_solar)

    @property
    def transmittance_direct_up(self) -> _FloatArray:
        return np.exp(-self.optical_depth_total["total"] / self.cos_zenith_view)

    @property
    def transmittance_diffuse_down(self) -> _FloatArray:
        return (
            self.transmittance_total_scattering["downward"]
            - self.transmittance_direct_down
        )

    @property
    def transmittance_diffuse_up(self) -> _FloatArray:
        return (
            self.transmittance_total_scattering["upward"] - self.transmittance_direct_up
        )

    @property
    def radiance_solar_reflected(self) -> _FloatArray:
        return (
            self.cos_zenith_solar
            * self.solar_spectrum
            * self.transmittance_global_gas["total"]
            * self.transmittance_total_scattering["total"]
        ) / np.pi


def make_sixs_wrapper() -> SixS:
    try:
        import sixs_bin

        return sixs_bin.make_wrapper()
    except ImportError:
        # [6s] extra not installed.
        pass

    # Let Py6s try finding 6S on PATH.
    s = SixS()

    if s.sixs_path is None:
        raise RuntimeError(
            f"No 6S binary could be found. Make sure 6S is installed and on the path. "
            f"Alternatively, install {__package__.split('.')[0]} with the [6s] "
            f"feature enabled."
        )
    return s
