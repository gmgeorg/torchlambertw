"""Module for testing special module."""

import torch
import numpy as np
import scipy.special
import pytest
import parameterized

from .. import special


_W_IDENTITIES = [
    (0.0, 0.0),
    (1, 0.0),
    (np.exp(-1), 0),
]


def test_specific_values():
    for (z, w_z) in _W_IDENTITIES:
        torch_z = torch.tensor(z)
        result = special.lambertw(torch_z).numpy()[0][0]
        assert result == pytest.approx(w_z, 1e-6)


@pytest.mark.skip(reason="skip for now")
def test_torch_equals_scipy():
    data = np.array([[1, 2.0], [3, 4.0]])
    torch_data = torch.tensor(data)
    scipy_results = scipy.special.lambertw(data)
    torch_results = special.lambertw(torch_data)
    np.testing.assert_allclose(torch_results.numpy(), scipy_results.real)
