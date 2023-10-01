"""
Engine for the 6S radiative transfer model.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import ClassVar

import Py6S

import rtm_wrapper.parameters as rtm_param
from rtm_wrapper.engines.base import (
    EngineOutputs,
    OutputName,
    ParameterRegistry,
    RTMEngine,
)
from rtm_wrapper.simulation import Inputs

from ._py6s import make_sixs_wrapper


class PySixSEngine(RTMEngine):
    """
    Engine for running 6S using `Py6S`_.

    Use with :func:`pysixs_default_inputs` as the :class:`~rtm_wrapper.simulation.Inputs`
    to replicate the out-of-box behavior of ``Py6S``.

    .. _Py6S: https://pypi.org/project/Py6S/
    """

    params: ClassVar[ParameterRegistry[[Py6S.SixS]]]

    virtual_outputs: ClassVar = ("py6s_values", "py6s_trans", "py6s_rat")

    default_outputs: ClassVar = (
        "apparent_radiance",
        "transmittance_direct_down",
        "transmittance_direct_up",
        "transmittance_diffuse_down",
        "transmittance_diffuse_up",
        "transmittance_scattering_down",
        "transmittance_scattering_up",
        "total_transmission",
        "spherical_albedo",
    )

    _base_wrapper: Py6S.SixS

    def __init__(
        self,
        wrapper: Py6S.SixS | None = None,
        *,
        outputs: Iterable[OutputName] | None = None,
    ) -> None:
        if wrapper is None:
            wrapper = make_sixs_wrapper()
        if wrapper.sixs_path is None:
            raise ValueError("misconfigured wrapper - sixs_path not defined")
        self._base_wrapper = wrapper

        super().__init__(outputs=outputs)

    def run_simulation(self, inputs: Inputs) -> EngineOutputs:
        # Note: this implementation avoids mutating any instance state so that it can
        # be safely run from multiple threads on the same object.

        wrapper = copy.deepcopy(self._base_wrapper)
        self.params.process(inputs, wrapper)

        wrapper.run()
        sixs_outputs = wrapper.outputs

        outputs: EngineOutputs = {
            "py6s_values": sixs_outputs.values,
            "py6s_trans": sixs_outputs.trans,
            "py6s_rat": sixs_outputs.rat,
        }
        self._extract_outputs(outputs)
        return outputs

        # # Extract select outputs.
        # cos_zenith_solar = math.cos(math.radians(outputs.values["solar_z"]))
        # cos_zenith_view = math.cos(math.radians(outputs.values["view_z"]))
        # m_optical_depth_total = -outputs.rat["optical_depth_total"].total
        # total_scattering = outputs.trans["total_scattering"]
        # t_scat_d, t_scat_u = total_scattering.downward, total_scattering.upward
        #
        # # Derive transmittances
        # t_dir_d = math.exp(m_optical_depth_total / cos_zenith_solar)
        # t_dir_u = math.exp(m_optical_depth_total / cos_zenith_view)
        # t_diff_d = t_scat_d - t_dir_d
        # t_diff_u = t_scat_u - t_dir_u
        # t_gas = outputs.trans["global_gas"].total
        #
        # total_transmission = (
        #     t_dir_d * t_dir_u
        #     + t_diff_d * t_dir_u
        #     + t_dir_d * t_diff_u
        #     + t_diff_d * t_diff_u
        # ) * t_gas
        #
        # return EngineOutputs(
        #     apparent_radiance=outputs.values["apparent_radiance"],
        #     transmittance_scattering_down=t_scat_d,
        #     transmittance_scattering_up=t_scat_u,
        #     transmittance_direct_down=t_dir_d,
        #     transmittance_direct_up=t_dir_u,
        #     transmittance_diffuse_down=t_diff_d,
        #     transmittance_diffuse_up=t_diff_u,
        #     transmittance_total_gas=t_gas,
        #     total_transmission=total_transmission,
        #     spherical_albedo=outputs.rat["spherical_albedo"].total,
        #     single_scattering_albedo=outputs.rat["single_scattering_albedo"].total,
        #     solar_spectrum=outputs.values["solar_spectrum"],
        #     direct_solar_irradiance=outputs.values["direct_solar_irradiance"],
        #     diffuse_solar_irradiance=outputs.values["diffuse_solar_irradiance"],
        # )


def pysixs_default_inputs() -> Inputs:
    """
    Return input parameters that replicate Py6S's defaults.
    """
    return Inputs(
        altitude_sensor=rtm_param.AltitudePredefined(name="sealevel"),
        altitude_target=rtm_param.AltitudePredefined(name="sealevel"),
        atmosphere=rtm_param.AtmospherePredefined(name="MidlatitudeSummer"),
        aerosol_profile=rtm_param.AerosolProfilePredefined(profile="Maritime"),
        ground=rtm_param.GroundReflectanceHomogenousUniformLambertian(reflectance=0.3),
        geometry=rtm_param.GeometryAngleDate(
            solar_zenith=rtm_param.AngleDegreesParameter(degrees=32),
            solar_azimuth=rtm_param.AngleDegreesParameter(degrees=264),
            view_zenith=rtm_param.AngleDegreesParameter(degrees=23),
            view_azimuth=rtm_param.AngleDegreesParameter(degrees=190),
            day=14,
            month=7,
        ),
        wavelength=rtm_param.WavelengthFixed(value=0.5),
    )


def pysixs_standard_inputs() -> Inputs:
    """
    Return modified version of Py6S' default input parameters.

    Sets the default sensor altitude to satellite level.
    """
    return pysixs_default_inputs().replace(altitude_sensor__name="satellite")


# Register parameters and outputs.
from . import _impl
