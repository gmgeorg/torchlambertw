"""Module for Lambert W x Normal distributions."""

import torch

from . import base

from . import utils
from .. import constants as con


class TailLambertWNormal(base.TailLambertWDistribution):
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

    def __init__(
        self,
        loc: con.PARAM_DTYPE,
        scale: con.PARAM_DTYPE,
        tailweight: con.PARAM_DTYPE,
        validate_args=None,
    ):
        """Initializes the class."""
        loc = utils.to_tensor(loc)
        scale = utils.to_tensor(scale)
        tailweight = utils.to_tensor(tailweight)

        self.tailweight = tailweight
        super().__init__(
            base_distribution=torch.distributions.Normal,
            base_dist_args={"loc": loc, "scale": scale},
            tailweight=tailweight,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TailLambertWNormal, _instance)
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


class SkewLambertWNormal(base.SkewLambertWDistribution):
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

    def __init__(
        self,
        loc: con.PARAM_DTYPE,
        scale: con.PARAM_DTYPE,
        skewweight: con.PARAM_DTYPE,
        validate_args=None,
    ):
        loc = utils.to_tensor(loc)
        scale = utils.to_tensor(scale)
        self.skewweight = utils.to_tensor(skewweight)

        super().__init__(
            base_distribution=torch.distributions.Normal,
            base_dist_args={"loc": loc, "scale": scale},
            skewweight=self.skewweight,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SkewLambertWNormal, _instance)
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
            0.5 * self.skewweight**2
        )
        return mu_y

    @property
    def mode(self):
        pass

    @property
    def variance(self):
        # See eq (4.2) in Goerg 2011.
        gamma_sq = self.skewweight**2
        exp_gamma_sq = torch.exp(gamma_sq)
        sigma2_y = (
            self.base_dist.scale.pow(2)
            * exp_gamma_sq
            * ((4 * gamma_sq + 1) * exp_gamma_sq - gamma_sq)
        )
        return sigma2_y
