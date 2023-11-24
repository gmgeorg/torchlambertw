"""Module for Lambert W x Weibull distribution."""

import torch
import torch.distributions

from . import base


class TailLambertWWeibull(base.TailLambertWDistribution):
    """Tail Lambert W x Weibull distribution to use as args."""

    def __init__(
        self,
        concentration: torch.Tensor,
        scale: torch.Tensor,
        tailweight: torch.Tensor,
        use_mean_variance: bool = True,
        **kwargs
    ):
        super().__init__(
            base_distribution=torch.distributions.Weibull,
            base_dist_args={"concentration": concentration, "scale": scale},
            use_mean_variance=use_mean_variance,
            tailweight=tailweight,
        )


class SkewLambertWWeibull(base.SkewLambertWDistribution):
    """Skew Lambert W x Weibull distribution to use as args."""

    def __init__(
        self,
        concentration: torch.Tensor,
        scale: torch.Tensor,
        skewweight: torch.Tensor,
        use_mean_variance: bool = True,
        **kwargs
    ):
        super().__init__(
            base_distribution=torch.distributions.Weibull,
            base_dist_args={"concentration": concentration, "scale": scale},
            use_mean_variance=use_mean_variance,
            skewweight=skewweight,
        )
