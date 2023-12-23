"""Base module for Lambert W x F distributions."""

from typing import Optional, Dict

import torch
import torch.distributions as td

from . import utils
from .. import constants as con
from .. import transforms


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
    has_rsample = True

    def __init__(
        self,
        base_distribution: torch.distributions.Distribution,
        base_dist_args: Dict[str, con.PARAM_DTYPE],
        tailweight: con.PARAM_DTYPE,
        use_mean_variance: bool = True,
        shift: Optional[con.PARAM_DTYPE] = None,
        scale: Optional[con.PARAM_DTYPE] = None,
        validate_args=None,
    ):
        """Initialize the distribution."""

        assert isinstance(use_mean_variance, bool)
        self.use_mean_variance = use_mean_variance

        base_distr = base_distribution(**base_dist_args, validate_args=validate_args)
        shift, scale = utils.update_shift_scale(
            shift,
            scale,
            distribution=base_distr,
            use_mean_variance=self.use_mean_variance,
        )
        self.shift = utils.to_tensor(shift)
        self.tailweight = utils.to_tensor(tailweight)

        super().__init__(
            base_distribution=base_distr,
            transforms=transforms.TailLambertWTransform(
                shift=shift, scale=scale, tailweight=tailweight
            ),
            validate_args=validate_args,
        )
        self.arg_constraints = base_distr.arg_constraints.copy()
        self.arg_constraints["tailweight"] = td.constraints.greater_than_eq(0.0)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TailLambertWDistribution, _instance)
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
    has_rsample = True

    def __init__(
        self,
        base_distribution: torch.distributions.Distribution,
        base_dist_args: Dict[str, con.PARAM_DTYPE],
        skewweight: con.PARAM_DTYPE,
        use_mean_variance: bool = True,
        shift: Optional[con.PARAM_DTYPE] = None,
        scale: Optional[con.PARAM_DTYPE] = None,
        validate_args=None,
    ):
        assert isinstance(use_mean_variance, bool)
        self.use_mean_variance = use_mean_variance

        self.skewweight = utils.to_tensor(skewweight)
        base_distr = base_distribution(**base_dist_args, validate_args=validate_args)

        shift, scale = utils.update_shift_scale(
            shift,
            scale,
            distribution=base_distr,
            use_mean_variance=self.use_mean_variance,
        )

        self.shift = utils.to_tensor(shift)

        super().__init__(
            base_distribution=base_distr,
            transforms=transforms.SkewLambertWTransform(
                shift=shift, scale=scale, skewweight=skewweight
            ),
            validate_args=validate_args,
        )
        self.arg_constraints = base_distr.arg_constraints.copy()
        self.arg_constraints["skewweight"] = td.constraints.real

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SkewLambertWDistribution, _instance)
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
