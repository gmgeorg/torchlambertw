"""Module for testing preprocessing module."""

import torch
import numpy as np
import scipy.special
import pytest
import timeit

from torchlambertw import transforms


def _test_data():
    rng = np.random.RandomState(seed=42)
    x = rng.normal(size=10)
    return x


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
    torch_trafo = transforms.TailLambertWTransform(
        shift=torch.tensor(loc),
        scale=torch.tensor(scale),
        tailweight=torch.tensor(delta),
    )

    torch_result = torch_trafo._inverse(torch.tensor(x)).numpy()
    np.testing.assert_allclose(torch_result, x, atol=eps)


@pytest.mark.parametrize(
    "loc,scale,delta", [(0.0, 1.0, 0.5), (0.4, 2.0, 0.1), (0.4, 2.0, 0.001)]
)
def test_np_transform_inverse_equality(loc, scale, delta):
    x = _test_data()
    torch_trafo = transforms.TailLambertWTransform(
        shift=torch.tensor(loc),
        scale=torch.tensor(scale),
        tailweight=torch.tensor(delta),
    )
    y = torch_trafo._inverse(torch.tensor(x))

    y_reverse = torch_trafo(y)
    np.testing.assert_allclose(x, y_reverse.numpy().ravel(), atol=1e-5)
