from __future__ import annotations

import copy
import math
import typing
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
import Py6S
import Py6S.outputs as sixs_outputs
from nptyping import Float, Int, NDArray, Object, Structure
from typing_extensions import TypeAlias

import rtm_wrapper.parameters as rtm_param
from rtm_wrapper.engines.base import RTMEngine
from rtm_wrapper.simulation import Inputs, Outputs


class PySixSEngine(RTMEngine):
    _base_wrapper: Py6S.SixS

    def __init__(self, wrapper: Py6S.SixS | None = None) -> None:
        if wrapper is None:
            wrapper = make_sixs_wrapper()
        if wrapper.sixs_path is None:
            raise ValueError("misconfigured wrapper - sixs_path not defined")
        self._base_wrapper = wrapper

    def run_simulation(self, inputs: Inputs) -> Outputs:
        wrapper = copy.deepcopy(self._base_wrapper)
        self.load_inputs(inputs, wrapper)

        wrapper.run()
        outputs = wrapper.outputs

        # Extract select outputs.
        cos_zenith_solar = math.cos(math.radians(outputs.values["solar_z"]))
        cos_zenith_view = math.cos(math.radians(outputs.values["view_z"]))
        m_optical_depth_total = -outputs.rat["optical_depth_total"].total
        total_scattering = outputs.trans["total_scattering"]
        t_scat_d, t_scat_u = total_scattering.downward, total_scattering.upward

        # Derive transmittances
        t_dir_d = math.exp(m_optical_depth_total / cos_zenith_solar)
        t_dir_u = math.exp(m_optical_depth_total / cos_zenith_view)
        t_diff_d = t_scat_d - t_dir_d
        t_diff_u = t_scat_u - t_dir_u

        return Outputs(
            apparent_radiance=outputs.values["apparent_radiance"],
            transmittance_scattering_down=t_scat_d,
            transmittance_scattering_up=t_scat_u,
            transmittance_direct_down=t_dir_d,
            transmittance_direct_up=t_dir_u,
            transmittance_diffuse_down=t_diff_d,
            transmittance_diffuse_up=t_diff_u,
        )


def pysixs_default_inputs() -> Inputs:
    """
    Return input parameters that replicate Py6S's defaults.
    """
    return Inputs(
        altitude_sensor=rtm_param.AltitudePredefined(name="sealevel"),
        altitude_target=rtm_param.AltitudePredefined(name="sealevel"),
        atmosphere=rtm_param.AtmospherePredefined(name="MidlatitudeSummer"),
        aerosol_profile=rtm_param.AerosolProfilePredefined(name="Maritime"),
        ground=rtm_param.GroundReflectanceHomogenousUniformLambertian(reflectance=0.3),
        wavelength=rtm_param.WavelengthFixed(value=0.5),
    )


# TODO validation and error checking.


@PySixSEngine.params.register("wavelength")
def _handle(inputs: rtm_param.WavelengthFixed, wrapper: Py6S.SixS) -> None:
    wrapper.wavelength = Py6S.Wavelength(inputs.value)


@PySixSEngine.params.register("altitude_sensor")
def _handle(inputs: rtm_param.AltitudePredefined, wrapper: Py6S.SixS) -> None:
    if inputs.name == "sealevel":
        wrapper.altitudes.set_sensor_sea_level()
    elif inputs.name == "satellite":
        wrapper.altitudes.set_sensor_satellite_level()
    else:
        raise RuntimeError(f"bad parameter {inputs=}")


@PySixSEngine.params.register("altitude_target")
def _handle(inputs: rtm_param.AltitudePredefined, wrapper: Py6S.SixS) -> None:
    if inputs.name == "sealevel":
        wrapper.altitudes.set_sensor_sea_level()
    else:
        raise RuntimeError(f"bad parameter {inputs=}")


@PySixSEngine.params.register("atmosphere")
def _handle(inputs: rtm_param.AtmospherePredefined, wrapper: Py6S.SixS) -> None:
    atmos_profile = getattr(Py6S.AtmosProfile, inputs.name)
    wrapper.atmos_profile = Py6S.AtmosProfile.PredefinedType(atmos_profile)


@PySixSEngine.params.register("atmosphere", rtm_param.AtmosphereWaterOzone)
def _handle(inputs: rtm_param.AtmosphereWaterOzone, wrapper: Py6S.SixS) -> None:
    wrapper.atmos_profile = Py6S.AtmosProfile.UserWaterAndOzone(
        inputs.water, inputs.ozone
    )


@PySixSEngine.params.register("aerosol_profile")
def _handle(inputs: rtm_param.AerosolProfilePredefined, wrapper: Py6S.SixS) -> None:
    aero_profile = getattr(Py6S.AeroProfile, inputs.name)
    wrapper.aero_profile = Py6S.AeroProfile.PredefinedType(aero_profile)


@PySixSEngine.params.register("ground")
def _handle(
    inputs: rtm_param.GroundReflectanceHomogenousUniformLambertian, wrapper: Py6S.SixS
) -> None:
    wrapper.ground_reflectance = Py6S.GroundReflectance.HomogeneousLambertian(
        inputs.reflectance
    )


# @PySixSEngine.params.register("ground")
# def _handle(
#     inputs: rtm_param.GroundReflectanceHeterogeneousLambertian, wrapper: Py6S.SixS
# ) -> None:
#     wrapper.ground_reflectance = Py6S.GroundReflectance.HeterogeneousLambertian(
#         radius=0.5,
#         ro_target=np.stack(
#             (inputs.target.wavelengths, inputs.target.spectrum), axis=-1
#         ),
#         ro_env=np.stack(
#             (inputs.background.wavelengths, inputs.background.spectrum), axis=-1
#         ),
#     )


# Original helpers below.

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
_FloatArray: TypeAlias = NDArray[Literal["*"], Float]

_IntArray: TypeAlias = NDArray[Literal["*"], Int]

_TransmittanceArray: TypeAlias = NDArray[
    Literal["*"], Structure["[downward, upward, total]: Float"]
]
_TransmittanceArrayDType = np.dtype(
    [("downward", np.float_), ("upward", np.float_), ("total", np.float_)]
)

_RatArray: TypeAlias = NDArray[
    Literal["*"], Structure["[rayleigh, aerosol, total]: Float"]
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
        for trans_key in typing.get_args(_TransmittanceName):
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
        for rat_key in typing.get_args(_RatName):
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
