"""Module for testing distribution/utils.py module."""

import pytest

from torchlambertw.distributions import utils


@pytest.mark.parametrize(
    "dist_name,expected",
    [("Normal", True), ("Exponential", False), ("LogNormal", False)],
)
def test_is_location_family(dist_name, expected):
    assert utils.is_location_family(dist_name) == expected


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
def test_update_shift_scale(dist_name, base_dist_args, shift, scale, use_mean_variance, expected):
    distr = utils.get_distribution_constructor(dist_name)(**base_dist_args)
    result = utils.update_shift_scale(shift, scale, distr, use_mean_variance=use_mean_variance)
    exptected_tensor = tuple([utils.to_tensor(v) for v in expected])
    assert result == exptected_tensor


@pytest.mark.parametrize(
    "dist_name, params, expected_mean, expected_stddev",
    [
        ("Normal", {"loc": 3.0, "scale": 2.0}, 3.0, 2.0),
        ("Exponential", {"rate": 3.0}, 1 / 3.0, 1 / 3.0),
    ],
)
def test_get_distribution_constructor(dist_name, params, expected_mean, expected_stddev):
    distr_constr = utils.get_distribution_constructor(dist_name)
    distr = distr_constr(**params)
    assert distr.mean.numpy() == pytest.approx(expected_mean, 1e-5)
    assert distr.stddev.numpy() == pytest.approx(expected_stddev, 1e-5)


@pytest.mark.parametrize(
    "dist_name,expected",
    [
        ("Normal", ["loc", "scale"]),
        ("Exponential", ["rate"]),
        ("StudentT", ["loc", "scale", "df"]),
    ],
)
def test_get_distribution_args(dist_name, expected):
    distr_constr = utils.get_distribution_constructor(dist_name)
    args = utils.get_distribution_args(distr_constr)
    assert set(args) == set(expected)
