from __future__ import annotations

import math
import operator
import typing
from typing import Annotated, Any, Callable

import numpy as np
import Py6S

from rtm_wrapper import parameters as rtm_param
from rtm_wrapper.engines.sixs import PySixSEngine
from rtm_wrapper.engines.sixs._py6s import _OutputDict, _RatName, _TransmittanceName


def _item_attr_getter(item: Any, attr: str) -> Callable[[Any], Any]:
    """Return a callable that fetches ``obj[item].attr`` from a given object ``obj``."""
    get_item = operator.itemgetter(item)
    get_attr = operator.attrgetter(attr)

    def getter(obj: Any) -> Any:
        return get_attr(get_item(obj))

    return getter


def _register_py6s_outputs() -> None:
    """
    Register outputs for all available Py6S outputs with a ``py6s_`` prefix.
    """
    for value_name, output_hints in typing.get_type_hints(
        _OutputDict, include_extras=True
    ).items():
        if typing.get_origin(output_hints) is Annotated:
            dtype, metadata = typing.get_args(output_hints)
        else:
            dtype = output_hints
            metadata = {}

        metadata.setdefault("title", value_name.replace("_", " ").title())

        PySixSEngine.outputs.register(
            operator.itemgetter(value_name),
            name=f"py6s_{value_name}",
            depends=("py6s_values",),
            dtype=dtype,
            **metadata,
        )

    for trans_name in typing.get_args(_TransmittanceName):
        for component in ("upward", "downward", "total"):
            title = (
                f"{component.capitalize()}"
                f" {trans_name.replace('_', ' ').title()} Transmittance"
            )
            PySixSEngine.outputs.register(
                _item_attr_getter(trans_name, component),
                name=f"py6s_transmittance_{trans_name}_{component}",
                depends=("py6s_trans",),
                dtype=np.dtype(float),
                title=title,
                unit="1",
            )
    for rat_name in typing.get_args(_RatName):
        for component in ("rayleigh", "aerosol", "total"):
            title = f"{component.capitalize()}" f" {rat_name.replace('_', ' ').title()}"
            PySixSEngine.outputs.register(
                _item_attr_getter(rat_name, component),
                name=f"py6s_{rat_name}_{component}",
                depends=("py6s_rat",),
                dtype=np.dtype(float),
                title=title,
            )


# Immediately register Py6S outputs so that they can be used in the general output
# registrations below.
_register_py6s_outputs()

#
# Output extractors.
#


@PySixSEngine.outputs.register(title="Apparent Radiance", unit="W/sr-m^2")
def apparent_radiance(py6s_values: _OutputDict) -> float:
    return py6s_values["apparent_radiance"]


@PySixSEngine.outputs.register(title="Cosine Zenith Solar", unit="1")
def cos_zenith_solar(py6s_solar_z: float) -> float:
    m = math
    return m.cos(m.radians(py6s_solar_z))


@PySixSEngine.outputs.register(title="Cosine Zenith View", unit="1")
def cos_zenith_view(py6s_view_z: float) -> float:
    m = math
    return m.cos(m.radians(py6s_view_z))


@PySixSEngine.outputs.register(title="Direct Downward Transmittance", unit="1")
def transmittance_direct_down(
    py6s_optical_depth_total_total: float, cos_zenith_solar: float
) -> float:
    return math.exp(-py6s_optical_depth_total_total / cos_zenith_solar)


@PySixSEngine.outputs.register(title="Direct Upward Transmittance", unit="1")
def transmittance_direct_up(
    py6s_optical_depth_total_total: float, cos_zenith_view: float
) -> float:
    return math.exp(-py6s_optical_depth_total_total / cos_zenith_view)


@PySixSEngine.outputs.register(title="Diffuse Downward Transmittance", unit="1")
def transmittance_diffuse_down(
    py6s_transmittance_total_scattering_downward: float,
    transmittance_direct_down: float,
) -> float:
    return py6s_transmittance_total_scattering_downward - transmittance_direct_down


@PySixSEngine.outputs.register(title="Diffuse Upward Transmittance", unit="1")
def transmittance_diffuse_up(
    py6s_transmittance_total_scattering_upward: float, transmittance_direct_up: float
) -> float:
    return py6s_transmittance_total_scattering_upward - transmittance_direct_up


@PySixSEngine.outputs.register(title="Total Transmission", unit="1")
def total_transmission(
    transmittance_direct_down: float,
    transmittance_direct_up: float,
    transmittance_diffuse_down: float,
    transmittance_diffuse_up: float,
    py6s_transmittance_global_gas_total: float,
) -> float:
    return (
        transmittance_direct_down * transmittance_direct_up
        + transmittance_diffuse_down * transmittance_direct_up
        + transmittance_direct_down * transmittance_diffuse_up
        + transmittance_diffuse_down * transmittance_diffuse_up
    ) * py6s_transmittance_global_gas_total


@PySixSEngine.outputs.register(title="Downward Scattering", unit="1")
def transmittance_scattering_down(
    py6s_transmittance_total_scattering_downward: float,
) -> float:
    return py6s_transmittance_total_scattering_downward


@PySixSEngine.outputs.register(title="Upward Scattering", unit="1")
def transmittance_scattering_up(
    py6s_transmittance_total_scattering_upward: float,
) -> float:
    return py6s_transmittance_total_scattering_upward


@PySixSEngine.outputs.register(title="Spherical Albedo", unit="1")
def spherical_albedo(py6s_spherical_albedo_total: float) -> float:
    return py6s_spherical_albedo_total


#
# Parameter handlers.
#

# TODO validation and error checking.


@PySixSEngine.params.register("wavelength")
def _handle0(inputs: rtm_param.WavelengthFixed, wrapper: Py6S.SixS) -> None:
    wrapper.wavelength = Py6S.Wavelength(inputs.value)


@PySixSEngine.params.register("altitude_sensor")
def _handle1(inputs: rtm_param.AltitudePredefined, wrapper: Py6S.SixS) -> None:
    if inputs.name == "sealevel":
        wrapper.altitudes.set_sensor_sea_level()
    elif inputs.name == "satellite":
        wrapper.altitudes.set_sensor_satellite_level()
    else:
        raise RuntimeError(f"bad parameter {inputs=}")


@PySixSEngine.params.register("altitude_target")
def _handle2(inputs: rtm_param.AltitudePredefined, wrapper: Py6S.SixS) -> None:
    if inputs.name == "sealevel":
        wrapper.altitudes.set_sensor_sea_level()
    else:
        raise RuntimeError(f"bad parameter {inputs=}")


@PySixSEngine.params.register("altitude_target")
def _handle3(inputs: rtm_param.AltitudeKilometers, wrapper: Py6S.SixS) -> None:
    wrapper.altitudes.set_target_custom_altitude(inputs.value)


@PySixSEngine.params.register("atmosphere")
def _handle4(inputs: rtm_param.AtmospherePredefined, wrapper: Py6S.SixS) -> None:
    atmos_profile = getattr(Py6S.AtmosProfile, inputs.name)
    wrapper.atmos_profile = Py6S.AtmosProfile.PredefinedType(atmos_profile)


@PySixSEngine.params.register("atmosphere", rtm_param.AtmosphereWaterOzone)
def _handle5(inputs: rtm_param.AtmosphereWaterOzone, wrapper: Py6S.SixS) -> None:
    wrapper.atmos_profile = Py6S.AtmosProfile.UserWaterAndOzone(
        inputs.water, inputs.ozone
    )


@PySixSEngine.params.register("aerosol_profile")
def _handle6(inputs: rtm_param.AerosolProfilePredefined, wrapper: Py6S.SixS) -> None:
    aero_profile = getattr(Py6S.AeroProfile, inputs.profile)
    wrapper.aero_profile = Py6S.AeroProfile.PredefinedType(aero_profile)


@PySixSEngine.params.register("ground")
def _handle7(
    inputs: rtm_param.GroundReflectanceHomogenousUniformLambertian, wrapper: Py6S.SixS
) -> None:
    wrapper.ground_reflectance = Py6S.GroundReflectance.HomogeneousLambertian(
        inputs.reflectance
    )


@PySixSEngine.params.register("ground")
def _handle8(
    inputs: rtm_param.GroundReflectanceHeterogeneousLambertian, wrapper: Py6S.SixS
) -> None:
    wrapper.ground_reflectance = Py6S.GroundReflectance.HeterogeneousLambertian(
        radius=0.5,
        ro_target=np.stack(
            (inputs.target.wavelengths, inputs.target.spectrum), axis=-1
        ),
        ro_env=np.stack(
            (inputs.background.wavelengths, inputs.background.spectrum), axis=-1
        ),
    )


@PySixSEngine.params.register("aerosol_profile")
def _handle9(inputs: rtm_param.AerosolAOTSingleLayer, wrapper: Py6S.SixS) -> None:
    wrapper.aero_profile = Py6S.AeroProfile.UserProfile(
        getattr(Py6S.AeroProfile, inputs.profile)
    )
    wrapper.aero_profile.add_layer(inputs.height, inputs.aot)


@PySixSEngine.params.register("aerosol_profile")
def _handle10(inputs: rtm_param.AerosolAOTLayers, wrapper: Py6S.SixS) -> None:
    if not inputs.layers.ndim == 2 and inputs.layers.shape[-1]:
        raise ValueError(
            f"bad shape for AOT layers: expected (*, 2) or (2,), got ({inputs.layers.shape}"
        )
    wrapper.aero_profile = Py6S.AeroProfile.UserProfile(
        getattr(Py6S.AeroProfile, inputs.profile)
    )
    for layer in inputs.layers:
        wrapper.aero_profile.add_layer(*layer)


@PySixSEngine.params.register("geometry")
def _handle11(inputs: rtm_param.GeometryAngleDate, wrapper: Py6S.SixS) -> None:
    geometry = Py6S.Geometry.User()
    geometry.solar_z = inputs.solar_zenith.as_degrees()
    geometry.solar_a = inputs.solar_azimuth.as_degrees()
    geometry.view_z = inputs.view_zenith.as_degrees()
    geometry.view_a = inputs.solar_azimuth.as_degrees()
    geometry.day = inputs.day
    geometry.month = inputs.month
    wrapper.geometry = geometry
