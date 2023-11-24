"""Module for Lambert W x LogNormal distribution."""

import torch
import torch.distributions

from . import base


class TailLambertWLogNormal(base.TailLambertWDistribution):
    """Tail Lambert W x LogNormal distribution to use as args."""

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        tailweight: torch.Tensor,
        use_mean_variance: bool = True,
        **kwargs
    ):
        super().__init__(
            base_distribution=torch.distributions.LogNormal,
            base_dist_args={"loc": loc, "scale": scale},
            use_mean_variance=use_mean_variance,
            tailweight=tailweight,
            **kwargs
        )


class SkewLambertWLogNormal(base.SkewLambertWDistribution):
    """Skew Lambert W x LogNormal distribution to use as args."""

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        skewweight: torch.Tensor,
        use_mean_variance: bool = True,
        **kwargs
    ):
        super().__init__(
            base_distribution=torch.distributions.LogNormal,
            base_dist_args={"loc": loc, "scale": scale},
            use_mean_variance=use_mean_variance,
            skewweight=skewweight,
            **kwargs
        )
