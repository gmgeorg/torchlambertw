"""Module for testing torch transforms module."""

import numpy as np
import pytest
import torch

from torchlambertw import transforms


def _test_data():
    rng = np.random.RandomState(seed=42)
    x = rng.normal(size=10)
    return torch.tensor(x)


@pytest.mark.parametrize("gamma", [(-0.1), (0.0), (0.1), (0.2)])
def test_w_gamma(gamma):
    u = _test_data()
    u_gamma = transforms.H_gamma(u, gamma=gamma)
    w_u_gamma = transforms.W_gamma(u_gamma, gamma=gamma, k=0)
    np.testing.assert_allclose(u.numpy(), w_u_gamma.numpy(), atol=1e-6)


@pytest.mark.parametrize("delta", [(-0.1), (0.0), (0.1), (0.2)])
def test_w_delta(delta):
    u = _test_data()
    u_delta = transforms.G_delta(u, delta=delta)
    w_u_delta = transforms.W_delta(u_delta, delta=delta)
    np.testing.assert_allclose(u.numpy(), w_u_delta.numpy(), rtol=1e-6)

    if delta > 0:
        assert all(torch.abs(u_delta) > torch.abs(u))


@pytest.mark.parametrize(
    "loc,scale,delta", [(0.0, 1.0, 0.5), (0.4, 2.0, 0.1), (0.4, 2.0, 0.001)]
)
def test_torch_transform_inverse_equality(loc, scale, delta):
    x = _test_data()
    torch_trafo = transforms.TailLambertWTransform(
        shift=torch.tensor(loc),
        scale=torch.tensor(scale),
        tailweight=torch.tensor(delta),
    )

    y = torch_trafo(torch.tensor(x))
    y_reverse = torch_trafo._inverse(y)

    np.testing.assert_allclose(x, y_reverse.numpy(), atol=1e-6)
