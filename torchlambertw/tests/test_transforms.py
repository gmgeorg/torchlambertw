"""Module for testing torch transforms module."""

import torch
import numpy as np
import scipy.special
import pytest
import timeit

from torchlambertw import transforms


def _test_data():
    rng = np.random.RandomState(seed=42)
    x = rng.normal(size=10)
    return torch.tensor(x)


@pytest.mark.parametrize(
    "gamma",
    [(-0.1), (0.0), (0.1), (0.2)],
)
def test_w_gamma(gamma):
    u = _test_data()
    u_gamma = transforms.H_gamma(u, gamma=gamma)
    w_u_gamma = transforms.W_gamma(u_gamma, gamma=gamma, k=0)
    np.testing.assert_allclose(u.numpy(), w_u_gamma.numpy())


@pytest.mark.parametrize(
    "delta",
    [(-0.1), (0.0), (0.1), (0.2)],
)
def test_w_delta(delta):
    u = _test_data()
    u_delta = transforms.G_delta(u, delta=delta)
    w_u_delta = transforms.W_delta(u_delta, delta=delta)
    np.testing.assert_allclose(u.numpy(), w_u_delta.numpy())
