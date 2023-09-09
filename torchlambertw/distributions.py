"""Module for Lambert W x F distributions."""

import torch

# This is WIP.
from . import special
from . import transforms
import torch.distributions as td


class LambertWNormal(td.transformed_distribution.TransformedDistribution):
    r"""
    Creates a Lambert W x Normal distribution parameterized by
    `loc`, `scale`, and `tailweight` where::

        X ~ Normal(loc, scale)
        U = (X - loc) / scale
        Z = U * exp(tailweight / 2. * U^2)
        Y = Z * scale + loc
        Y ~ Lambert W x Normal(loc, scale, tailweight)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log of the distribution
    """
    arg_constraints = {
        "loc": td.constraints.real,
        "scale": td.constraints.positive,
        "tailweight": td.constraints.greater_than_eq(0.0),
    }
    support = td.constraints.real
    has_rsample = True

    def __init__(self, loc, scale, tailweight, validate_args=None):
        self.tailweight = tailweight
        super().__init__(
            base_distribution=td.Normal(
                loc=loc, scale=scale, validate_args=validate_args
            ),
            transforms=transforms.LambertWTailTransform(
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
