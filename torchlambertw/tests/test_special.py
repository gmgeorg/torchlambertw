"""Module for testing special module."""

import torch
import numpy as np
import scipy.special
import pytest
import timeit

from .. import special

# https://en.wikipedia.org/wiki/Lambert_W_function#Special_values
_LAMBERTW_SPECIAL_VALUES = [
    (0.0, 0.0),
    (1, 0.5671433),  # omega constant
    (special._M_EXP_INV, -1.0),
    (special._M_EXP_INV - 0.00001, np.nan),
    (-10, np.nan),
    (2 * np.log(2), np.log(2)),
    (np.exp(1), 1),
    (np.inf, np.inf),
    (-np.inf, np.nan),
]


def _test_z():
    return np.array([[0.0, 1, 2.0], [-1.0, 3, 4.0]])


def test_specific_values():
    for (z, expected) in _LAMBERTW_SPECIAL_VALUES:
        torch_z = torch.tensor(z)
        w = float(special.lambertw(torch_z).numpy())
        print(z, expected)
        if np.isnan(expected):
            assert np.isnan(w)
        else:
            assert w == pytest.approx(expected, 1e-6)


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
    execution_time = timeit.timeit(
        _run_lambertw, number=100
    )  # Adjust the number of runs as needed
    print(execution_time)
    assert (
        execution_time < 0.05
    ), f"Execution time exceeded 100ms: {execution_time:.4f} seconds"
