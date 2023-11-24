"""Module for testing preprocessing module."""


import numpy as np
import pytest
import torch

# import torchlambertw as tlw

# from torchlambertw.distributions import base as td_base

from torchlambertw import distributions as td


@pytest.mark.parametrize(
    "loc,scale,delta,eps", [(0.0, 1.0, 0.0, 1e-6), (0.4, 2.0, 0.0, 1e-6)]
)
def test_identity_h(loc, scale, delta, eps):
    distr = torch.distributions.Normal(loc=loc, scale=scale)
    lw_tail_distr = td.base.TailLambertWDistribution(
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
    "loc,scale,gamma,eps", [(0.0, 1.0, 0.0, 1e-6), (0.4, 2.0, 0.0, 1e-6)]
)
def test_identity_s(loc, scale, gamma, eps):
    distr = torch.distributions.Normal(loc=loc, scale=scale)
    lw_skew_distr = td.base.SkewLambertWDistribution(
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


@pytest.mark.parametrize(
    "distr_constr,lambertw_distr_constr,params,gamma,eps",
    [
        (
            torch.distributions.Normal,
            td.SkewLambertWNormal,
            {"loc": 0.0, "scale": 1.0},
            0.1,
            1e-6,
        ),
        (
            torch.distributions.LogNormal,
            td.SkewLambertWLogNormal,
            {"loc": 0.0, "scale": 1.0},
            0.1,
            1e-6,
        ),
        (
            torch.distributions.Exponential,
            td.SkewLambertWExponential,
            {"rate": 2.0},
            0.1,
            1e-6,
        ),
        (
            torch.distributions.Weibull,
            td.SkewLambertWWeibull,
            {"concentration": 2.0, "scale": 1.0},
            0.1,
            1e-6,
        ),
        (
            torch.distributions.Gamma,
            td.SkewLambertWGamma,
            {"concentration": 2.0, "rate": 1.0},
            0.1,
            1e-6,
        ),
    ],
)
def test_skew_lw_distr(distr_constr, lambertw_distr_constr, params, gamma, eps):
    _ = distr_constr(**params)
    lw_skew_distr = td.base.SkewLambertWDistribution(
        base_distribution=distr_constr,
        base_dist_args=params,
        skewweight=gamma,
        use_mean_variance=True,
    )
    lw_skew_distr2 = lambertw_distr_constr(skewweight=gamma, **params)

    x = torch.tensor(np.linspace(1, 3, 100))
    lw_log_probs = lw_skew_distr.log_prob(x)
    lw_log_probs2 = lw_skew_distr2.log_prob(x)
    np.testing.assert_allclose(lw_log_probs2.numpy(), lw_log_probs.numpy(), atol=eps)


@pytest.mark.parametrize(
    "distr_constr,lambertw_distr_constr,params,delta,eps",
    [
        (
            torch.distributions.Normal,
            td.TailLambertWNormal,
            {"loc": 0.0, "scale": 1.0},
            0.1,
            1e-6,
        ),
        (
            torch.distributions.LogNormal,
            td.TailLambertWLogNormal,
            {"loc": 0.0, "scale": 1.0},
            0.1,
            1e-6,
        ),
        (
            torch.distributions.Exponential,
            td.TailLambertWExponential,
            {"rate": 2.0},
            0.1,
            1e-6,
        ),
        (
            torch.distributions.Weibull,
            td.TailLambertWWeibull,
            {"concentration": 2.0, "scale": 1.0},
            0.1,
            1e-6,
        ),
        (
            torch.distributions.Gamma,
            td.TailLambertWGamma,
            {"concentration": 2.0, "rate": 1.0},
            0.1,
            1e-6,
        ),
    ],
)
def test_tail_lw_distr(distr_constr, lambertw_distr_constr, params, delta, eps):
    _ = distr_constr(**params)
    lw_skew_distr = td.base.TailLambertWDistribution(
        base_distribution=distr_constr,
        base_dist_args=params,
        tailweight=delta,
        use_mean_variance=True,
    )
    lw_skew_distr2 = lambertw_distr_constr(tailweight=delta, **params)

    x = torch.tensor(np.linspace(1, 3, 100))
    lw_log_probs = lw_skew_distr.log_prob(x)
    lw_log_probs2 = lw_skew_distr2.log_prob(x)
    np.testing.assert_allclose(lw_log_probs2.numpy(), lw_log_probs.numpy(), atol=eps)
