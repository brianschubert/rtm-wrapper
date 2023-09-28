import warnings

# Temporarily silence deprecated alias warnings from nptyping 2.5.0 for numpy>=1.24.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="nptyping")

import dataclasses
from typing import Final, Literal

import numpy as np
import numpy.testing
import Py6S
import pytest
from nptyping import Float, NDArray, Object

import rtm_wrapper

# Skip entire module for now, until tests are updated.
pytest.skip(allow_module_level=True)

_PY6S_TEST_OUTPUT: Final = """\
6S wrapper script by Robin Wilson
Using 6S located at {}
Running 6S using a set of test parameters
6sV version: 1.1
The results are:
Expected result: 619.158000
Actual result: 619.158000
#### Results agree, Py6S is working correctly
"""


@pytest.fixture
def sample_experiment() -> (
    tuple[
        Py6S.SixS,
        NDArray[Literal["*"], Float],
        NDArray[Literal["*"], Object],
    ]
):
    sixs = rtm_wrapper.make_sixs_wrapper()
    sixs.ground_reflectance = Py6S.GroundReflectance().HomogeneousLambertian(0.7)
    sixs.geometry = Py6S.Geometry.User()
    sixs.geometry.solar_z = 32
    sixs.geometry.solar_a = 264
    sixs.geometry.view_z = 23
    sixs.geometry.view_a = 190
    sixs.geometry.day = 14
    sixs.geometry.month = 7
    sixs.aero_profile = Py6S.AeroProfile.PredefinedType(Py6S.AeroProfile.Maritime)
    sixs.altitudes.set_target_sea_level()
    sixs.altitudes.set_sensor_satellite_level()

    wavelengths, raw_outputs = Py6S.SixSHelpers.Wavelengths.run_vnir(sixs)

    return sixs, wavelengths, raw_outputs


def test_py6s_available(capsys) -> None:
    sixs = rtm_wrapper.make_sixs_wrapper()

    Py6S.SixS.test(sixs.sixs_path)

    captured = capsys.readouterr()

    assert captured.out == _PY6S_TEST_OUTPUT.format(sixs.sixs_path)
    assert captured.err == ""


def test_dense_outputs() -> None:
    sixs = rtm_wrapper.make_sixs_wrapper()

    _wavelengths, results = Py6S.SixSHelpers.Wavelengths.run_vnir(sixs)
    dense_outputs = rtm_wrapper.Py6SDenseOutput.from_py6s(results)

    # Ensure that all fields contained in the Py6S outputs are represented
    # in the extract dense outputs
    output_names = set(dir(results[0]))
    extracted_names = {f.name for f in dataclasses.fields(dense_outputs)}
    assert output_names == extracted_names


def test_dense_outputs_empty() -> None:
    dense_outputs = rtm_wrapper.Py6SDenseOutput.from_py6s(np.array([]))

    # Version not set.
    assert dense_outputs.version == "unknown"

    # All fields are empty arrays.
    for field in dataclasses.fields(dense_outputs):
        if field.name == "version":
            continue
        assert getattr(dense_outputs, field.name).size == 0


def test_dense_outputs_derived() -> None:
    sixs = rtm_wrapper.make_sixs_wrapper()

    _wavelengths, results = Py6S.SixSHelpers.Wavelengths.run_vnir(sixs)

    dense_outputs = rtm_wrapper.Py6SDenseOutput.from_py6s(results)

    # Ensure that all properties can be computed without error.
    for p in dir(dense_outputs):
        _ = getattr(dense_outputs, p)


def test_dense_outputs_relations(sample_experiment) -> None:
    """
    Verify that various Py6S outputs agree with their derived values.
    """

    sixs, wavelengths, raw_outputs = sample_experiment
    out = rtm_wrapper.Py6SDenseOutput.from_py6s(raw_outputs)

    rho_t = rho_b = sixs.ground_reflectance[-1] * np.ones(wavelengths.shape)
    s_star = 1 / (1 - rho_b * out.spherical_albedo["total"])

    tolerances = {
        "atol": 0.001,
        "rtol": 0.001,
    }

    # Compare atmospheric radiance with derived value
    l_d_up = (
        out.cos_zenith_solar
        * out.solar_spectrum
        * out.atmospheric_intrinsic_reflectance
    ) / np.pi
    np.testing.assert_allclose(
        l_d_up, out.atmospheric_intrinsic_radiance, atol=0.05, rtol=0.05
    )

    # Compare reflected radiance with derived value.
    l_r = out.radiance_solar_reflected * rho_t * s_star
    np.testing.assert_allclose(
        l_r, out.pixel_radiance + out.background_radiance, **tolerances
    )

    # Compare direct solar irradiance with derived value.
    e_dir_comp = (
        out.cos_zenith_solar
        * out.solar_spectrum
        * out.transmittance_direct_down
        * out.transmittance_global_gas["downward"]
    )
    np.testing.assert_allclose(e_dir_comp, out.direct_solar_irradiance, **tolerances)

    # Compare diffuse solar irradiance with derived value.
    e_diff_comp = (
        out.cos_zenith_solar
        * out.solar_spectrum
        * out.transmittance_diffuse_down
        * out.transmittance_global_gas["downward"]
    )
    np.testing.assert_allclose(e_diff_comp, out.diffuse_solar_irradiance, **tolerances)

    # Compare target radiance with derived value.
    l_t = (
        out.cos_zenith_solar
        * out.solar_spectrum
        * out.transmittance_global_gas["total"]
        * (out.transmittance_direct_down + out.transmittance_diffuse_down)
        * out.transmittance_direct_up
        * rho_t
        * s_star
    ) / np.pi
    np.testing.assert_allclose(l_t, out.pixel_radiance, **tolerances)

    # Compare background radiance with derived value.
    l_bd = (
        out.cos_zenith_solar
        * out.solar_spectrum
        * out.transmittance_global_gas["total"]
        * (out.transmittance_direct_down + out.transmittance_diffuse_down)
        * out.transmittance_diffuse_up
        * rho_t
        * s_star
    ) / np.pi
    np.testing.assert_allclose(l_bd, out.background_radiance, **tolerances)

    # Compare apparent radiance with derived value.
    l_m = l_t + l_bd + l_d_up
    np.testing.assert_allclose(l_m, out.apparent_radiance, atol=0.05, rtol=0.05)
