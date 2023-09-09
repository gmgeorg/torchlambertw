"""Module for implementing the special function, Lambert W.

In special module to follow the style of scipy.special.* and tfp.special.*
"""

import torch


def _init_lambertw(x: torch.Tensor) -> torch.Tensor:
    z0 = torch.zeros_like(x)
    z0[x > 1.0] = x[x > 1.0]
    z0[x < -2.0] = x[x < -2.0].exp() * (1.0 - x[x < -2.0].exp())
    pade = lambda x: x * (3.0 + 6.0 * x + x ** 2.0) / (3.0 + 9.0 * x + 5.0 * x ** 2)
    z0[(x <= 1.0) & (x >= -2.0)] = pade(x[(x <= 1.0) & (x >= -2.0)].exp())
    z0[z0 == 0.0] = 1e-6
    return z0


def lambertw(x: torch.Tensor, branch: int = 0) -> torch.Tensor:
    """
    Computes the Lambert function via Halley algorithm which converges cubically.
    The initialization is performed with a local approximation.

    Args:
      x: input Tensor.
      branch: 0 or -1; 0 uses the principal branch (default); -1 the non-principal branch.
    """
    z = _init_lambertw(x)
    eps = torch.finfo(x.dtype).eps
    a = lambda w: (w * ((w + eps).log() + w - x)) / (1 + w)
    b = lambda w: -1 / (w * (1 + w))
    for i in range(4):
        c = a(z)
        z = torch.max(
            z - c / (1 - 0.5 * c * b(z)), torch.tensor([eps], dtype=x.dtype)[:, None]
        )
    return z
