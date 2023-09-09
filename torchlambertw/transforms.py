"""Transforms for Lambert W distributions and normalization layers.

Lambert W transforms and Lambert W x F distributions come in 2 flavors:

* skewed Lambert W transforms / distributions ('gamma') -- see Goerg (2011) (https://doi.org/10.1214/11-AOAS457)

* heavy-tailed (potentially skewed) Lambert W transforms / distributions ('delta' & 'alpha') -- see Goerg (2015).

Transform names follow original convention in `LambertW` R package (https://github.com/gmgeorg/LambertW).
"""


import torch
from . import special

_EPS = torch.finfo(torch.float32).eps


def H_gamma(u: torch.tensor, gamma: torch.tensor) -> torch.tensor:
    """Computes base transform for skewed Lambert W x F distributions (Goerg 2011)."""
    return u * torch.tensor(gamma * u)


def G_delta(u: torch.tensor, delta: torch.tensor) -> torch.tensor:
    """Computes base transform for heavy-tailed Lambert W x F distributions (Goerg 2015)."""
    return u * torch.exp(delta / 2.0 * torch.pow(u, 2.0))


def G_delta_alpha(
    u: torch.tensor, delta: torch.tensor, alpha: torch.tensor
) -> torch.tensor:
    """Computes base transform for heavy-tailed Lambert W x F distributions (Goerg 2015)."""
    return u * torch.exp(delta / 2.0 * ((u ** 2.0) ** alpha))


def W_gamma(z: torch.tensor, gamma: torch.tensor, k: int) -> torch.tensor:
    """Computes W_gamma(z), the inverse of H_gamma(u)."""
    return special.lambertw(gamma * z, k=k) / gamma


def W_delta(z: torch.tensor, delta: torch.tensor) -> torch.tensor:
    """Computes W_delta(z), the inverse of G_delta(u)."""
    delta_z2 = delta * z * z
    return torch.where(
        torch.abs(delta_z2) < _EPS,
        z,
        torch.sqrt(special.lambertw(delta_z2, k=0) / delta) * torch.sign(z),
    )
