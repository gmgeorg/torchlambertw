"""Module for testing preprocessing module."""

import torch
import numpy as np
import scipy.special
import pytest
import timeit

from torchlambertw import distributions


def _test_data():
    rng = np.random.RandomState(seed=42)
    x = rng.normal(size=10)
    return x


@pytest.mark.parametrize(
    "loc,scale,delta,eps",
    [
        (0.0, 1.0, 0.0, 1e-6),
        (0.4, 2.0, 0.0, 1e-6),
    ],
)
def test_identity(loc, scale, delta, eps):
    distr = torch.distributions.Normal(loc=loc, scale=scale)
    lw_distr = distributions.TailLambertWDistribution(
        base_distribution=torch.distributions.Normal,
        base_dist_args={"loc": loc, "scale": scale},
        shift=loc,
        scale=scale,
        tailweight=delta,
    )

    x = torch.tensor(np.linspace(-3, 3, 100))
    lw_log_probs = lw_distr.log_prob(x)
    log_probs = distr.log_prob(x)
    np.testing.assert_allclose(log_probs.numpy(), lw_log_probs.numpy(), atol=eps)
