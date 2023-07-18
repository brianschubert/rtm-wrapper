import dataclasses
import warnings
from typing import Final

import numpy as np

# Temporary silence deprecated alias warnings with nptyping 2.5.0 for numpy>=1.24.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="nptyping")

import Py6S

import rtm_wrapper

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

    dense_outputs = rtm_wrapper.Py6SDenseOutput.from_py6s(results).compute_derived()
