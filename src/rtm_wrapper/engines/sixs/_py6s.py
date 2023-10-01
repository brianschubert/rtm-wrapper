"""
Utilities for interfacing with Py6S.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

import Py6S
from Py6S import outputs as sixs_outputs
from typing_extensions import TypeAlias

from rtm_wrapper.parameters import MetadataDict


def make_sixs_wrapper() -> Py6S.SixS:
    try:
        import sixs_bin

        return sixs_bin.make_wrapper()
    except ImportError:
        # [6s] extra not installed.
        pass

    # Let Py6s try finding 6S on PATH.
    s = Py6S.SixS()

    if s.sixs_path is None:
        raise RuntimeError(
            f"No 6S binary could be found. Make sure 6S is installed and on your PATH. "
            f"Tip: Install {__package__.split('.')[0]} with the [6s] "
            f"feature enabled to compile and install a local 6S binary."
        )
    return s


_TransmittanceName: TypeAlias = Literal[
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
_RatName: TypeAlias = Literal[
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
_TransmittanceDict: TypeAlias = dict[_TransmittanceName, sixs_outputs.Transmittance]
_RatDict: TypeAlias = dict[_RatName, sixs_outputs.RayleighAerosolTotal]


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
    apparent_radiance: Annotated[
        float, MetadataDict(title="Apparent Radiance", unit="W/sr-m^2")
    ]
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
