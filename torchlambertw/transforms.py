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
    return u * torch.tensor(gamma * u)


def W_gamma(z: torch.tensor, gamma: torch.tensor, k: int) -> torch.tensor:
    """Computes W_gamma(z), the inverse of H_gamma(u)."""
    return special.lambertw(gamma * z, k=k) / gamma


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


# distribution transforms


class LambertWTailTransform(td.transforms.Transform):
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

    def _call(self, x):
        u = (x - self.shift) / self.scale
        return G_delta(u, delta=self.tailweight) * self.scale + self.shift

    def _inverse(self, y):
        z = (y - self.shift) / self.scale
        return W_delta(z, delta=self.tailweight) * self.scale + self.shift

    # TODO:
    def log_abs_det_jacobian(self, x, y):
        raise NotImplementedError(
            "Jacobian has not been implemented yet for LambertWTailTransform()."
        )