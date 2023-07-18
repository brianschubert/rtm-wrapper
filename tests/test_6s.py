import dataclasses
import warnings

# Temporary silence deprecated alias warnings with nptyping 2.5.0 for numpy>=1.24.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="nptyping")

import Py6S

import rtm_wrapper


def test_dense_outputs():
    sixs = rtm_wrapper.make_sixs_wrapper()

    _wavelengths, results = Py6S.SixSHelpers.Wavelengths.run_vnir(sixs)

    dense_outputs = rtm_wrapper.Py6SDenseOutput.from_py6s(results)

    # Ensure that all fields contained in the Py6S outputs are represented
    # in the extract dense outputs
    output_names = set(dir(results[0]))
    extracted_names = {f.name for f in dataclasses.fields(dense_outputs)}
    assert output_names == extracted_names
