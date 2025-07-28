"""Module for Lambert W x LogNormal distribution."""

import torch
import torch.distributions

from . import base
from . import utils


class TailLambertWLogNormal(base.TailLambertWDistribution):
    """Tail Lambert W x LogNormal distribution to use as args."""

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        tailweight: torch.Tensor,
        use_mean_variance: bool = True,
        **kwargs,
    ):
        loc = utils.to_tensor(loc)
        scale = utils.to_tensor(scale)
        tailweight = utils.to_tensor(tailweight)

        self.tailweight = tailweight
        super().__init__(
            base_distribution=torch.distributions.LogNormal,
            base_dist_args={"loc": loc, "scale": scale},
            use_mean_variance=use_mean_variance,
            tailweight=tailweight,
            **kwargs,
        )


class SkewLambertWLogNormal(base.SkewLambertWDistribution):
    """Skew Lambert W x LogNormal distribution to use as args."""

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        skewweight: torch.Tensor,
        use_mean_variance: bool = True,
        **kwargs,
    ):
        loc = utils.to_tensor(loc)
        scale = utils.to_tensor(scale)
        skewweight = utils.to_tensor(skewweight)

        self.skewweight = skewweight
        super().__init__(
            base_distribution=torch.distributions.LogNormal,
            base_dist_args={"loc": loc, "scale": scale},
            use_mean_variance=use_mean_variance,
            skewweight=skewweight,
            **kwargs,
        )
