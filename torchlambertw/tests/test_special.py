"""Module for testing special module."""

import timeit

import numpy as np
import pytest
import scipy.special
import torch

from .. import special

# https://en.wikipedia.org/wiki/Lambert_W_function#Special_values
_LAMBERTW_0_SPECIAL_VALUES = [
    (0.0, 0.0),
    (-0.1, -0.11183255915896297),  # compared to scipy.special.lambertw
    (1, 0.5671433),  # omega constant
    (special._M_EXP_INV, -1.0),
    (special._M_EXP_INV - 0.00001, np.nan),
    (-10, np.nan),
    (2 * np.log(2), np.log(2)),
    (np.exp(1), 1),
    (1e5, 9.284571428622108),  # compared to scipy.special.lambertw
    (np.inf, np.inf),
    (-np.inf, np.nan),
]
_LAMBERTW_M1_SPECIAL_VALUES = [
    (special._M_EXP_INV, -1.0),
    (-0.1, -3.577152063957297),  # compared to scipy.special.lambertw
    (-0.0009, -9.236251966692597),  # see https://github.com/gmgeorg/torchlambertw/issues/4
    (np.inf, np.nan),
    (0.0, -np.inf),
]


def test_specific_values_for_principal_branch():
    for z, expected in _LAMBERTW_0_SPECIAL_VALUES:
        torch_z = torch.tensor(z)
        w = float(special.lambertw(torch_z, k=0).numpy())
        print(z, expected, w)
        if np.isnan(expected):
            assert np.isnan(w)
        else:
            assert w == pytest.approx(expected, 1e-6)


def test_specific_values_for_nonprincipal_branch():
    for z, expected in _LAMBERTW_M1_SPECIAL_VALUES:
        torch_z = torch.tensor(z)
        w = float(special.lambertw(torch_z, k=-1).numpy())
        print(z, expected)
        if np.isnan(expected):
            assert np.isnan(w)
        else:
            assert w == pytest.approx(expected, 1e-6)


def _test_x():
    return np.array([[10.0, 1, 2.0], [-0.1, 0.0, 4.0]])


def test_w_inverse_of_xexp():
    x_data = torch.tensor(_test_x())
    xexp_result = special.xexp(x_data)
    w_xexp_result = special.lambertw(xexp_result)
    np.testing.assert_allclose(x_data.numpy(), w_xexp_result.numpy())


def _test_z():
    return np.array([[0.0, 1, 2.0], [-1.0, 3, 4.0]])


@pytest.mark.skip(reason="skip for now")
def test_torch_equals_scipy():
    data = _test_z()
    torch_data = torch.tensor(data)
    scipy_results = scipy.special.lambertw(data)
    torch_results = special.lambertw(torch_data)
    np.testing.assert_allclose(torch_results.numpy(), scipy_results.real)


@pytest.mark.skip(reason="skip; only used for local testing on local machine")
def test_speed():
    x_data = torch.Tensor(_test_z())

    # Define a function to run the Lambert W function and measure the time
    def _run_lambertw():
        _ = special.lambertw(x_data)

    # Measure the execution time using timeit
    execution_time = timeit.timeit(_run_lambertw, number=100)  # Adjust the number of runs as needed
    print(execution_time)
    assert execution_time < 0.05, f"Execution time exceeded 100ms: {execution_time:.4f} seconds"
