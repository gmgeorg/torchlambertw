"""Module for W-related transformations in scipy/numpy."""

import numpy as np
import scipy.special

_EPS = np.finfo(np.float32).eps


def H_gamma(u: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Computes base transform for skewed Lambert W x F distributions (Goerg 2011)."""
    return u * np.exp(gamma * u)


def W_gamma(z: np.ndarray, gamma: np.ndarray, k: int) -> np.ndarray:
    """Computes W_gamma(z), the inverse of H_gamma(u)."""
    return np.real(special.lambertw(gamma * z, k=k)) / gamma


def G_delta(u: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Computes base transform for heavy-tailed Lambert W x F distributions (Goerg 2015)."""
    return u * np.exp(delta / 2.0 * np.pow(u, 2.0))


def G_delta_alpha(u: np.ndarray, delta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Computes base transform for heavy-tailed Lambert W x F distributions (Goerg 2015)."""
    return u * np.exp(delta / 2.0 * ((u ** 2.0) ** alpha))


def W_delta(z: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Computes W_delta(z), the inverse of G_delta(u)."""
    delta_z2 = delta * z * z
    return np.where(
        np.abs(delta_z2) < _EPS,
        z,
        np.real(np.sqrt(scipy.special.lambertw(delta_z2, k=0)) / delta) * np.sign(z),
    )


def W_tau(y: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Compute 