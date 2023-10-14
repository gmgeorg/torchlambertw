"""Module for testing preprocessing module."""

import torch
import numpy as np
import scipy.special
import pytest
import timeit


from ..preprocessing import np_transforms
from ..preprocessing import base
from torchlambertw import transforms


def _test_data():
    rng = np.random.RandomState(seed=42)
    x = rng.normal(size=10)
    return x


@pytest.mark.parametrize(
    "gamma",
    [(-0.1), (0.0), (0.1), (0.2)],
)
def test_w_gamma(gamma):
    u = _test_data()
    u_gamma = np_transforms.H_gamma(u, gamma=gamma)
    w_u_gamma = np_transforms.W_gamma(u_gamma, gamma=gamma, k=0)
    np.testing.assert_allclose(u, w_u_gamma)


@pytest.mark.parametrize(
    "delta",
    [(-0.1), (0.0), (0.1), (0.2)],
)
def test_w_delta(delta):
    u = _test_data()
    u_delta = np_transforms.G_delta(u, delta=delta)
    w_u_delta = np_transforms.W_delta(u_delta, delta=delta)
    np.testing.assert_allclose(u, w_u_delta)


@pytest.mark.parametrize(
    "loc,scale,delta,eps",
    [
        (0.0, 1.0, 0.0, 1e-6),
        (0.0, 1.0, 0.0001, 1e-2),
        (0.4, 2.0, 0.0, 1e-6),
        (
            0.4,
            2.0,
            0.0001,
            1e-2,
        ),  # small deviation from delta = 0 results in small deviation only
    ],
)
def test_identity_transform(loc, scale, delta, eps):
    x = _test_data()
    torch_trafo = transforms.LambertWTailTransform(
        shift=torch.tensor(loc),
        scale=torch.tensor(scale),
        tailweight=torch.tensor(delta),
    )

    torch_result = torch_trafo(torch.tensor(x)).numpy()
    np_result = np_transforms.W_tau(
        y=x,
        tau=base.Tau(
            loc=loc,
            scale=scale,
            lambertw_params=base.LambertWParams(delta=delta),
        ),
    )
    np.testing.assert_allclose(np_result, x, atol=eps)
    np.testing.assert_allclose(torch_result, x, atol=eps)


@pytest.mark.parametrize(
    "loc,scale,delta",
    [(0.0, 1.0, 0.5), (0.4, 2.0, 0.1), (0.4, 2.0, 0.001)],
)
def test_np_torch_transform_equality(loc, scale, delta):
    x = _test_data()
    torch_trafo = transforms.LambertWTailTransform(
        shift=torch.tensor(loc),
        scale=torch.tensor(scale),
        tailweight=torch.tensor(delta),
    )

    torch_result = torch_trafo(torch.tensor(x)).numpy()
    np_result = np_transforms.W_tau(
        y=x,
        tau=base.Tau(
            loc=loc,
            scale=scale,
            lambertw_params=base.LambertWParams(delta=delta),
        ),
    )
    np.testing.assert_allclose(np_result, torch_result)
