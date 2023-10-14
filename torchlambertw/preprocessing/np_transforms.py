"""Module for W-related transformations in scipy/numpy."""

import numpy as np
import scipy.special

from . import base

_EPS = np.finfo(np.float32).eps


def H_gamma(u: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Computes base transform for skewed Lambert W x F distributions (Goerg 2011)."""
    return u * np.exp(gamma * u)


def W_gamma(z: np.ndarray, gamma: np.ndarray, k: int) -> np.ndarray:
    """Computes W_gamma(z), the inverse of H_gamma(u)."""
    if np.abs(gamma) < _EPS:
        return z
    return np.real(scipy.special.lambertw(gamma * z, k=k)) / gamma


def G_delta(u: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Computes base transform for heavy-tailed Lambert W x F distributions (Goerg 2015)."""
    return u * np.exp(delta / 2.0 * np.power(u, 2.0))


def G_delta_alpha(u: np.ndarray, delta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Computes base transform for heavy-tailed Lambert W x F distributions (Goerg 2015)."""
    return u * np.exp(delta / 2.0 * ((u ** 2.0) ** alpha))


def W_delta(z: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Computes W_delta(z), the inverse of G_delta(u)."""
    if np.abs(delta) < _EPS:
        return z
    delta_z2 = delta * z * z
    return np.where(
        np.abs(delta_z2) < _EPS,
        z,
        np.sqrt(np.real(scipy.special.lambertw(delta_z2, k=0)) / delta) * np.sign(z),
    )


def normalize_by_tau(y: np.ndarray, tau: base.Tau) -> np.ndarray:
    """Computes the backtransform for an observed skewed, heavy-tailed dataset."""
    z = (y - tau.loc) / tau.scale
    lambertw_type = tau.lambertw_params.lambertw_type
    if lambertw_type == base.LambertWType.S:
        u = W_gamma(z, gamma=tau.lambertw_params.gamma)
    elif lambertw_type == base.LambertWType.H:
        u = W_delta(z, delta=tau.lambertw_params.delta)
    elif lambertw_type == base.LambertWType.HH:
        raise NotImplementedError(f"lambertw_type={lambertw_type} not implemented yet.")

    x = u * tau.scale + tau.loc
    return x


def inverse_normalize_by_tau(x: np.ndarray, tau: base.Tau) -> np.ndarray:
    """Computes the backtransform for an observed skewed, heavy-tailed dataset."""
    u = (x - tau.loc) / tau.scale
    lambertw_type = tau.lambertw_params.lambertw_type
    if lambertw_type == base.LambertWType.S:
        z = H_gamma(u, gamma=tau.lambertw_params.gamma)
    elif lambertw_type == base.LambertWType.H:
        z = G_delta(u, delta=tau.lambertw_params.delta)
    elif lambertw_type == base.LambertWType.HH:
        raise NotImplementedError(f"lambertw_type={lambertw_type} not implemented yet.")

    y = z * tau.scale + tau.loc
    return y
