"""Module for Lambert W x Exponential distributions."""

import torch

from . import base
from . import utils


class TailLambertWExponential(base.TailLambertWDistribution):
    """Tail Lambert W x Exponential distribution to use as args."""

    def __init__(
        self,
        rate: torch.Tensor,
        tailweight: torch.Tensor,
        use_mean_variance: bool = True,
        **kwargs
    ):
        rate = utils.to_tensor(rate)
        tailweight = utils.to_tensor(tailweight)
        super().__init__(
            base_distribution=torch.distributions.Exponential,
            base_dist_args={"rate": rate},
            use_mean_variance=use_mean_variance,
            tailweight=tailweight,
        )


class SkewLambertWExponential(base.SkewLambertWDistribution):
    """Skew Lambert W x Exponential distribution to use as args.

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
         skewweight (float or Tensor): skewweight ("gamma") of the Lambert W x Exponential distribution.
           If 0., then it reduces to a Exponential(rate) distribution.
    """

    def __init__(
        self,
        rate: torch.Tensor,
        skewweight: torch.Tensor,
        use_mean_variance: bool = True,
        **kwargs
    ):
        rate = utils.to_tensor(rate)
        skewweight = utils.to_tensor(skewweight)
        super().__init__(
            base_distribution=torch.distributions.Exponential,
            base_dist_args={"rate": rate},
            use_mean_variance=use_mean_variance,
            skewweight=skewweight,
        )

    @property
    def mean(self):
        # See p13 in https://arxiv.org/pdf/2307.05644.pdf
        mean_y = 1.0 / (self.base_dist.rate * (1.0 - self.skewweight) ** 2)
        return torch.where(mean_y > 0, mean_y, torch.nan)

    @property
    def mode(self):
        return torch.zeros_like(self.base_dist.rate)

    @property
    def variance(self):
        # See p13 in https://arxiv.org/pdf/2307.05644.pdf
        # variance = EY^2 - (mean)**2
        denom = 1.0 - 2 * self.skewweight
        second_moment = 2.0 / (
            self.base_dist.rate.pow(2.0) * torch.pow(denom, 2.0 + 1.0)
        )
        variance = second_moment - self.mean**2
        return variance
