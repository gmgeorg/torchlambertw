"""Module for Lambert W x Gamma distributions."""

import torch

from . import base


class TailLambertWGamma(base.TailLambertWDistribution):
    """Tail Lambert W x Gamma distribution to use as args."""

    def __init__(
        self,
        concentration: torch.Tensor,
        rate: torch.Tensor,
        tailweight: torch.Tensor,
        use_mean_variance: bool = True,
        **kwargs
    ):
        super().__init__(
            base_distribution=torch.distributions.Gamma,
            base_dist_args={"concentration": concentration, "rate": rate},
            use_mean_variance=use_mean_variance,
            tailweight=tailweight,
        )


class SkewLambertWGamma(base.SkewLambertWDistribution):
    """Skew Lambert W x Gamma distribution to use as args."""

    def __init__(
        self,
        concentration: torch.Tensor,
        rate: torch.Tensor,
        skewweight: torch.Tensor,
        use_mean_variance: bool = True,
        **kwargs
    ):
        super().__init__(
            base_distribution=torch.distributions.Gamma,
            base_dist_args={"concentration": concentration, "rate": rate},
            use_mean_variance=use_mean_variance,
            skewweight=skewweight,
        )
