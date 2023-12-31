import numpy as np

import rtm_wrapper.parameters as rtm_param
from rtm_wrapper.engines.sixs import PySixSEngine, pysixs_default_inputs
from rtm_wrapper.execution import SerialExecutor
from rtm_wrapper.simulation import SweepSimulation


def test_sweep_layout() -> None:
    wl = np.arange(0.2, 2.5, 0.1)
    atmo = ["a", "b", "c"]

    sweep = SweepSimulation(
        {
            "wavelength.value": wl,
            "atmosphere.name": atmo,
        },
        base=pysixs_default_inputs(),
    )

    assert list(sweep.dims.items()) == [
        ("wavelength.value", wl.size),
        ("atmosphere.name", len(atmo)),
    ]
    assert sweep.sweep_size == len(wl) * len(atmo)
    assert sweep.sweep_shape == (len(wl), len(atmo))


def test_sweep_basic_single() -> None:
    wl = np.arange(0.2, 2.5, 0.1)

    script = {
        "wavelength.value": wl,
    }
    sweep = SweepSimulation(script, base=pysixs_default_inputs())

    engine = PySixSEngine()
    runner = SerialExecutor()

    runner.run(sweep, engine)
    results = runner.collect_results()

    assert dict(results.dims.items()) == {k: v.size for k, v in script.items()}
    assert {v.name: v.data for v in results.coords.values()} == script


def test_sweep_basic_product() -> None:
    wl = np.arange(0.2, 2.5, 0.2)
    ozone = np.arange(0.25, 0.46, 0.05)
    water = np.arange(1, 5, 1)

    script = {
        "wavelength.value": wl,
        "atmosphere.ozone": ozone,
        "atmosphere.water": water,
    }

    sweep = SweepSimulation(
        script,
        base=pysixs_default_inputs().replace(
            atmosphere=rtm_param.AtmosphereWaterOzone()
        ),
    )

    engine = PySixSEngine()
    runner = SerialExecutor()

    runner.run(sweep, engine)
    results = runner.collect_results()

    assert dict(results.dims.items()) == {k: v.size for k, v in script.items()}


def test_sweep_compound_attributes() -> None:
    wavelengths = np.arange(0.2, 2.5, 0.1)
    sweep = SweepSimulation(
        {
            "wl": {
                "wavelength.value": wavelengths,
                # Represent wavelengths in nanometers installed of micrometers.
                "__coords__": wavelengths * 1e3,
                "__title__": "Wavelength",
                "__unit__": "nanometers",
            },
        },
        base=pysixs_default_inputs(),
    )

    engine = PySixSEngine()
    runner = SerialExecutor()

    runner.run(sweep, engine)
    results = runner.collect_results()

    assert dict(results.dims.items()) == {"wl": wavelengths.size}
    assert {v.name: list(v.data) for v in results.coords.values()} == {
        "wl": list(wavelengths * 1e3),
        "wavelength.value": list(wavelengths),
    }


def test_sweep_basic_array_valued() -> None:
    wavelengths = np.arange(0.2, 2.5, 0.1)
    mock_spectrum_target = np.linspace(0, 1, wavelengths.size)
    mock_spectrum_background = 0.5 * mock_spectrum_target

    script = {
        "target": {
            # Vary measured wavelengths and values.
            "ground.target.wavelengths": [wavelengths, wavelengths * 0.5],
            "ground.target.spectrum": [
                mock_spectrum_target,
                mock_spectrum_target * 0.2,
            ],
            # Same wavelengths, vary spectrum.
            "ground.background.spectrum": [
                mock_spectrum_background,
                mock_spectrum_target * 0.3,
            ],
        },
        "wavelength.value": wavelengths,
    }

    sweep = SweepSimulation(
        script,
        base=pysixs_default_inputs().replace(
            ground=rtm_param.GroundReflectanceHeterogeneousLambertian(
                target=rtm_param.GroundReflectanceHomogenousLambertian(),
                background=rtm_param.GroundReflectanceHomogenousLambertian(
                    wavelengths=wavelengths
                ),
            )
        ),
    )

    engine = PySixSEngine()
    runner = SerialExecutor()

    runner.run(sweep, engine)
    results = runner.collect_results()

    assert dict(results.dims.items()) == {
        "wavelength.value": wavelengths.size,
        "target": 2,
        "ground.target.wavelengths/0": mock_spectrum_target.size,
        "ground.target.spectrum/0": mock_spectrum_target.size,
        "ground.background.spectrum/0": mock_spectrum_target.size,
    }


def test_sweep_split() -> None:
    wl = np.arange(0.27, 2.5, 0.01)
    ozone = np.arange(0.25, 0.46, 0.05)
    water = np.arange(1, 5, 1)

    assert wl.size == 223

    script = {
        "wavelength.value": wl,
        "atmosphere.ozone": ozone,
        "atmosphere.water": water,
    }

    sweep = SweepSimulation(
        script,
        base=pysixs_default_inputs().replace(
            atmosphere=rtm_param.AtmosphereWaterOzone()
        ),
    )

    split_dim = "wavelength.value"
    num_splits = 5
    split_sweeps = list(sweep.split(num_splits, split_dim))
    assert len(split_sweeps) == num_splits

    # Verify that splits contain the expected number of coordinate elements.
    residue = 223 % 5
    num_big = num_splits - residue
    for full_sweep in split_sweeps[:num_big]:
        assert dict(full_sweep.dims.items()) == {
            "wavelength.value": (223 // 5) + 1,
            "atmosphere.ozone": ozone.size,
            "atmosphere.water": water.size,
        }
    for full_sweep in split_sweeps[num_splits:]:
        assert dict(full_sweep.dims.items()) == {
            "wavelength.value": (223 // 5),
            "atmosphere.ozone": ozone.size,
            "atmosphere.water": water.size,
        }

    # Verify that the split coordinates stack back into the original coordinates.
    np.testing.assert_equal(
        wl, np.hstack([s.sweep_spec.coords[split_dim].values for s in split_sweeps])
    )
