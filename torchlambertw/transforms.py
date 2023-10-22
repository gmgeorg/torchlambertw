"""Transforms for Lambert W distributions and normalization layers.

Lambert W transforms and Lambert W x F distributions come in 2 flavors:

* skewed Lambert W transforms / distributions ('gamma') -- see Goerg (2011) (https://doi.org/10.1214/11-AOAS457)

* heavy-tailed (potentially skewed) Lambert W transforms / distributions ('delta' & 'alpha') -- see Goerg (2015).

Transform names follow TensorFlow probability naming convention.
"""


import torch
import torch.distributions as td
from . import special

_EPS = torch.finfo(torch.float32).eps


def H_gamma(u: torch.tensor, gamma: torch.tensor) -> torch.tensor:
    """Computes base transform for skewed Lambert W x F distributions (Goerg 2011)."""
    return u * torch.exp(gamma * u)


def W_gamma(z: torch.tensor, gamma: torch.tensor, k: int) -> torch.tensor:
    """Computes W_gamma(z), the inverse of H_gamma(u)."""
    return torch.where(
        torch.abs(torch.tensor(gamma)) < _EPS,
        z,
        special.lambertw(gamma * z, k=k) / gamma,
    )


def G_delta(u: torch.tensor, delta: torch.tensor) -> torch.tensor:
    """Computes base transform for heavy-tailed Lambert W x F distributions (Goerg 2015)."""
    return u * torch.exp(delta / 2.0 * torch.pow(u, 2.0))


def G_delta_alpha(
    u: torch.tensor, delta: torch.tensor, alpha: torch.tensor
) -> torch.tensor:
    """Computes base transform for heavy-tailed Lambert W x F distributions (Goerg 2015)."""
    return u * torch.exp(delta / 2.0 * ((u ** 2.0) ** alpha))


def W_delta(z: torch.tensor, delta: torch.tensor) -> torch.tensor:
    """Computes W_delta(z), the inverse of G_delta(u)."""
    delta_z2 = delta * z * z
    return torch.where(
        torch.abs(delta_z2) < _EPS,
        z,
        torch.sqrt(special.lambertw(delta_z2, k=0) / delta) * torch.sign(z),
    )


# Distribution transforms


class TailLambertWTransform(td.transforms.Transform):
    r"""
    Transform via the mapping :math:`y = x * \exp(delta / 2 * x**2)`.
    """
    domain = td.constraints.real
    codomain = td.constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        shift: torch.tensor,
        scale: torch.tensor,
        tailweight: torch.tensor,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.shift = shift
        self.scale = scale
        self.tailweight = tailweight

    def __eq__(self, other):
        return isinstance(other, LambertWTailTransform)

    def _normalize(self, input):
        return (input - self.shift) / self.scale

    def _inverse_normalize(self, output):
        return output * self.scale + self.shift

    def _call(self, x):
        u = self._normalize(x)
        z = G_delta(u, delta=self.tailweight)
        return self._inverse_normalize(z)

    def _inverse(self, y):
        z = self._normalize(y)
        u = W_delta(z, delta=self.tailweight)
        return self._inverse_normalize(u)

    def log_abs_det_jacobian(self, x, y):
        u_sq = self._normalize(x).pow(2.0)
        # absolute value not needed as all terms here are >= 0.
        return torch.log(
            (1 + self.tailweight * u_sq) * torch.exp(0.5 * self.tailweight * u_sq)
        )


class SkewLambertWTransform(td.transforms.Transform):
    r"""
    Transform via the mapping :math:`y = x * \exp(gamma * x)`.
    """
    domain = td.constraints.real
    codomain = td.constraints.real
    bijective = False
    sign = +1

    def __init__(
        self,
        shift: torch.tensor,
        scale: torch.tensor,
        skewweight: torch.tensor,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.shift = shift
        self.scale = scale
        self.skewweight = skewweight

    def __eq__(self, other):
        return isinstance(other, LambertWSkewTransform)

    def _normalize(self, input):
        return (input - self.shift) / self.scale

    def _inverse_normalize(self, output):
        return output * self.scale + self.shift

    def _call(self, x):
        u = self._normalize(x)
        z = H_gamma(u, gamma=self.skewweight)
        return self._inverse_normalize(z)

    def _inverse(self, y):
        # Needs to use principal branch for inverse transformation; it's
        # not entirely bijective but very low probability event of non-bijectivity
        # for small gamma (approaches -> 0 for gamma -> 0).
        # TODO: implement with non-principal branch probability as well [needs to be
        # implemented as standalone distribution, not relying on TransformedDistribution
        # as not bijective when accounting for non-prinipal branch. See R package LambertW
        # for details on implementing this exactly]
        z = self._normalize(y)
        u = W_gamma(z, gamma=self.skewweight, k=0)
        return self._inverse_normalize(u)

    def log_abs_det_jacobian(self, x, y):
        u = self._normalize(x)
        return torch.log(
            torch.abs((self.skewweight * u + 1.0) * torch.exp(-self.skewweight * u))
        )
