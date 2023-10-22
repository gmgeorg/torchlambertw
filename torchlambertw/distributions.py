"""Module for Lambert W x F distributions."""

import torch

from typing import Union, Dict

from . import special
from . import transforms
import torch.distributions as td

_PARAM_DTYPE = Union[float, torch.tensor]


def _to_tensor(x: _PARAM_DTYPE) -> torch.Tensor:
    """Converts to tensor if a float/int."""
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x)

    return x


class TailLambertWDistribution(td.transformed_distribution.TransformedDistribution):
    r"""
    Creates a heavy-tail Lambert W x F distribution parameterized by
    `shift`, `scale`, and `tailweight` where::

        X ~ F(shift, scale)
        U = (X - shift) / scale
        Z = U * exp(tailweight / 2. * U^2)
        Y = Z * scale + shift
        Y ~ Lambert W x F(shift, scale, tailweight)

    Args:
        shift (float or Tensor): location/shift parameter of the latent F distribution.
          For (non-negative) scale-family distributions (e.g., exponential or gamma)
          this should be set to 0.
        scale (float or Tensor): scale parameter of latent F distribution
        tailweight (float or Tensor): tailweight ("delta") of the Lambert W x F distribution.
          If 0., then it reduces to a F(loc, scale) distribution.
    """
    arg_constraints = {
        "shift": td.constraints.real,
        "scale": td.constraints.positive,
        "tailweight": td.constraints.greater_than_eq(0.0),
    }
    support = td.constraints.real
    has_rsample = True

    def __init__(
        self,
        base_distribution: torch.distributions.Distribution,
        base_dist_args: Dict[str, _PARAM_DTYPE],
        shift: _PARAM_DTYPE,
        scale: _PARAM_DTYPE,
        tailweight: _PARAM_DTYPE,
        use_mean_variance: bool = True,
        validate_args=None,
    ):
        """Initialize the distribution."""
        self.shift = _to_tensor(shift)
        self.tailweight = _to_tensor(tailweight)
        scale = _to_tensor(scale)

        base_distr = base_distribution(**base_dist_args, validate_args=validate_args)
        super().__init__(
            base_distribution=base_distr,
            transforms=transforms.TailLambertWTransform(
                shift=shift, scale=scale, tailweight=tailweight
            ),
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LambertWDistribution, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.transforms[0].shift

    @property
    def scale(self):
        return self.transforms[0].scale

    @property
    def mean(self):
        pass

    @property
    def mode(self):
        pass

    @property
    def variance(self):
        pass


class TailLambertWNormal(td.transformed_distribution.TransformedDistribution):
    r"""
    Creates a heavy-tail Lambert W x Normal distribution parameterized by
    `loc`, `scale`, and `tailweight` where::

        X ~ Normal(loc, scale)
        U = (X - loc) / scale
        Z = U * exp(tailweight / 2. * U^2)
        Y = Z * scale + loc
        Y ~ Lambert W x Normal(loc, scale, tailweight)

    Args:
        loc (float or Tensor): mean of latent Normal distribution
        scale (float or Tensor): standard deviation of latent Normal distribution
        tailweight (float or Tensor): tailweight ("delta") of the Lambert W x Normal distribution.
          If 0., then it reduces to a Normal(loc, scale) distribution.
    """
    arg_constraints = {
        "loc": td.constraints.real,
        "scale": td.constraints.positive,
        "tailweight": td.constraints.greater_than_eq(0.0),
    }
    support = td.constraints.real
    has_rsample = True

    def __init__(
        self,
        loc: _PARAM_DTYPE,
        scale: _PARAM_DTYPE,
        tailweight: _PARAM_DTYPE,
        validate_args=None,
    ):
        """Initializes the class."""
        loc = _to_tensor(loc)
        scale = _to_tensor(scale)
        tailweight = _to_tensor(tailweight)

        self.tailweight = tailweight
        super().__init__(
            base_distribution=td.Normal(
                loc=loc, scale=scale, validate_args=validate_args
            ),
            transforms=transforms.TailLambertWTransform(
                shift=loc, scale=scale, tailweight=tailweight
            ),
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LambertWNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        loc_abs_times_tailweight = (self.base_dist.loc.abs() + 1) * (
            1.0 - self.tailweight
        )
        return torch.where(loc_abs_times_tailweight > 0, self.base_dist.loc, torch.nan)

    @property
    def mode(self):
        return self.base_dist.loc

    @property
    def variance(self):
        z_variance = 1.0 / torch.pow(1 - 2 * self.tailweight, 1.5)
        y_variance = z_variance * self.base_dist.scale.pow(2)
        return torch.where(y_variance > 0, y_variance, torch.inf)


class SkewLambertWDistribution(td.transformed_distribution.TransformedDistribution):
    r"""
    Creates a skewed Lambert W x Normal distribution parameterized by
    `shift`, `scale`, and `skewweight` where::

        X ~ F(shift, scale)
        U = (X - shift) / scale
        Z = U * exp(skewweight * U)
        Y = Z * scale + shift
        Y ~ Lambert W x F(shift, scale, tailweight)

    Args:
        shift (float or Tensor): location/shift parameter of the latent F distribution.
          For (non-negative) scale-family distributions (e.g., exponential or gamma)
          this should be set to 0.
        scale (float or Tensor): scale parameter of latent F distribution
        tailweight (float or Tensor): tailweight ("delta") of the Lambert W x F distribution.
          If 0., then it reduces to a F(shift, scale) distribution.
    """
    arg_constraints = {
        "shift": td.constraints.real,
        "scale": td.constraints.positive,
        "skewweight": td.constraints.real,
    }
    support = td.constraints.real
    has_rsample = True

    def __init__(
        self,
        base_distribution: torch.distributions.Distribution,
        base_dist_args: Dict[str, _PARAM_DTYPE],
        shift: _PARAM_DTYPE,
        scale: _PARAM_DTYPE,
        skewweight: _PARAM_DTYPE,
        use_mean_variance: bool = True,
        validate_args=None,
    ):
        self.skewweight = _to_tensor(skewweight)
        self.shift = _to_tensor(shift)
        # Not self.scale since a .scale is a propery of Distribution object.
        scale = _to_tensor(scale)
        base_distr = base_distribution(**base_dist_args, validate_args=validate_args)

        super().__init__(
            base_distribution=base_distr,
            transforms=transforms.SkewLambertWTransform(
                shift=shift, scale=scale, skewweight=skewweight
            ),
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LambertWDistribution, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.transforms[0].shift

    @property
    def scale(self):
        return self.transforms[0].scale

    @property
    def mean(self):
        pass

    @property
    def mode(self):
        pass

    @property
    def variance(self):
        pass


class SkewLambertWNormal(td.transformed_distribution.TransformedDistribution):
    r"""
    Creates a skewed Lambert W x Normal distribution parameterized by
    `loc`, `scale`, and `skewweight` where::

        X ~ Normal(loc, scale)
        U = (X - loc) / scale
        Z = U * exp(skewweight * U)
        Y = Z * scale + loc
        Y ~ Lambert W x Normal(loc, scale, skewweight)

    Args:
        loc (float or Tensor): mean of latent Normal distribution
        scale (float or Tensor): standard deviation of latent Normal distribution
        skewweight (float or Tensor): skewweight ("gamma") of the Lambert W x Normal distribution.
          If 0., then it reduces to a Normal(loc, scale) distribution.
    """
    arg_constraints = {
        "loc": td.constraints.real,
        "scale": td.constraints.positive,
        "skewweight": td.constraints.real,
    }
    support = td.constraints.real
    has_rsample = True

    def __init__(
        self,
        loc: _PARAM_DTYPE,
        scale: _PARAM_DTYPE,
        skewweight: _PARAM_DTYPE,
        validate_args=None,
    ):
        self.skewweight = _to_tensor(skewweight)
        super().__init__(
            base_distribution=td.Normal(
                loc=loc, scale=scale, validate_args=validate_args
            ),
            transforms=transforms.SkewLambertWTransform(
                shift=loc, scale=scale, skewweight=skewweight
            ),
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LambertWNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        # See eq (4.1) in Goerg 2011.
        mu_y = self.base_dist.loc + self.base_dist.scale * self.skewweight * torch.exp(
            0.5 * self.skewweight ** 2
        )
        return mu_y

    @property
    def mode(self):
        pass

    @property
    def variance(self):
        # See eq (4.2) in Goerg 2011.
        gamma_sq = self.skewweight ** 2
        exp_gamma_sq = torch.exp(gamma_sq)
        sigma2_y = (
            self.base_dist.scale.pow(2)
            * exp_gamma_sq
            * ((4 * gamma_sq + 1) * exp_gamma_sq - gamma_sq)
        )
        return sigma2_y


class SkewLambertWExponential(td.transformed_distribution.TransformedDistribution):
    r"""
    Creates a skewed Lambert W x Exponential distribution parameterized by
    `rate` (= 1 / scale), and `skewweight` where::

        X ~ Exponential(rate)
        U = X * rate ( = X / scale)
        Z = U * exp(skewweight * U)
        Y = Z / rate
        Y ~ Lambert W x Exponential(rate, skewweight)

    See Goerg 2011 and also Kaarik et al (2023) - https://arxiv.org/pdf/2307.05644.pdf

    Args:
        rate (float or Tensor): rate of log of the distribution
        skewweight (float or Tensor): skewweight ("gamma") of the Lambert W x Normal distribution.
          If 0., then it reduces to a Normal(loc, scale) distribution.
    """
    arg_constraints = {
        "rate": td.constraints.nonnegative,
        "skewweight": td.constraints.real,
    }
    support = td.constraints.nonnegative
    has_rsample = True

    def __init__(
        self, rate: _PARAM_DTYPE, skewweight: _PARAM_DTYPE, validate_args=None
    ):
        self.skewweight = skewweight
        self.rate = rate
        super().__init__(
            base_distribution=td.Exponential(rate=rate, validate_args=validate_args),
            transforms=transforms.SkewLambertWTransform(
                shift=0.0, scale=1.0 / rate, skewweight=skewweight
            ),
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LambertWNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def mean(self):
        # See p13 in https://arxiv.org/pdf/2307.05644.pdf
        mean_y = 1.0 / (self.rate * (1.0 - self.skewweight) ** 2)
        return torch.where(mean_y > 0, mean_y, torch.nan)

    @property
    def mode(self):
        return torch.zeros_like(self.rate)

    @property
    def variance(self):
        # See p13 in https://arxiv.org/pdf/2307.05644.pdf
        # variance = EY^2 - (mean)**2
        denom = 1.0 - 2 * self.skewweight
        second_moment = 2.0 / (self.rate.pow(2.0) * torch.pow(denom, 2.0 + 1.0))
        variance = second_moment - self.mean ** 2
        return variance
