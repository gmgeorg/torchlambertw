"""Module for testing preprocessing module."""

import torch
import numpy as np
import scipy.special
import pytest
import timeit

from torchlambertw import distributions


@pytest.mark.parametrize(
    "dist_name,expected",
    [
        ("Normal", True),
        ("Exponential", False),
    ],
)
def test_is_location_family(dist_name, expected):
    assert distributions.is_location_family(dist_name) == expected


@pytest.mark.parametrize(
    "dist_name,base_dist_args,shift,scale,use_mean_variance,expected",
    [
        ("Normal", {"loc": 0.1, "scale": 1.0}, 0.1, 1.0, True, (0.1, 1.0)),
        ("Normal", {"loc": 0.1, "scale": 1.0}, 0.1, 1.0, False, (0.1, 1.0)),
        ("Normal", {"loc": 0.1, "scale": 1.0}, None, None, True, (0.1, 1.0)),
        ("Exponential", {"rate": 2.0}, -20.0, 10.0, True, (-20.0, 10.0)),
        ("Exponential", {"rate": 2.0}, None, None, True, (0.0, 1.0 / 2.0)),
    ],
)
def test_update_shift_scale(
    dist_name, base_dist_args, shift, scale, use_mean_variance, expected
):
    distr = distributions.get_distribution_constructor(dist_name)(
        **base_dist_args,
    )
    result = distributions._update_shift_scale(
        shift,
        scale,
        distr,
        use_mean_variance=use_mean_variance,
    )
    exptected_tensor = tuple([distributions._to_tensor(v) for v in expected])
    assert result == exptected_tensor


@pytest.mark.parametrize(
    "dist_name, params, expected_mean, expected_stddev",
    [
        ("Normal", {"loc": 3.0, "scale": 2.0}, 3.0, 2.0),
        ("Exponential", {"rate": 3.0}, 1 / 3.0, 1 / 3.0),
    ],
)
def test_get_distribution_constructor(
    dist_name, params, expected_mean, expected_stddev
):
    distr_constr = distributions.get_distribution_constructor(dist_name)
    distr = distr_constr(**params)
    assert distr.mean.numpy() == pytest.approx(expected_mean, 1e-5)
    assert distr.stddev.numpy() == pytest.approx(expected_stddev, 1e-5)


@pytest.mark.parametrize(
    "loc,scale,delta,eps",
    [
        (0.0, 1.0, 0.0, 1e-6),
        (0.4, 2.0, 0.0, 1e-6),
    ],
)
def test_identity_h(loc, scale, delta, eps):
    distr = torch.distributions.Normal(loc=loc, scale=scale)
    lw_tail_distr = distributions.TailLambertWDistribution(
        base_distribution=torch.distributions.Normal,
        base_dist_args={"loc": loc, "scale": scale},
        shift=loc,
        scale=scale,
        tailweight=delta,
    )

    x = torch.tensor(np.linspace(-3, 3, 100))
    lw_log_probs = lw_tail_distr.log_prob(x)
    log_probs = distr.log_prob(x)
    np.testing.assert_allclose(log_probs.numpy(), lw_log_probs.numpy(), atol=eps)


@pytest.mark.parametrize(
    "loc,scale,gamma,eps",
    [
        (0.0, 1.0, 0.0, 1e-6),
        (0.4, 2.0, 0.0, 1e-6),
    ],
)
def test_identity_s(loc, scale, gamma, eps):
    distr = torch.distributions.Normal(loc=loc, scale=scale)
    lw_skew_distr = distributions.SkewLambertWDistribution(
        base_distribution=torch.distributions.Normal,
        base_dist_args={"loc": loc, "scale": scale},
        shift=loc,
        scale=scale,
        skewweight=gamma,
    )

    x = torch.tensor(np.linspace(-3, 3, 100))
    lw_log_probs = lw_skew_distr.log_prob(x)
    log_probs = distr.log_prob(x)
    np.testing.assert_allclose(log_probs.numpy(), lw_log_probs.numpy(), atol=eps)
