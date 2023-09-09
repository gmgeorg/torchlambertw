"""A torch implementation of the Lambert W function.

In special module to follow the style of scipy.special.* and tfp.special.*

This implementation is a direct translation from TensorFlow Probability
https://www.tensorflow.org/probability/api_docs/python/tfp/math/lambertw

Vectorized, except for while loop. TODO to make this work in vectorized/GPU version
(equivalent of tf.while_loop() in TensorFlow).
"""

from typing import Tuple
import torch
import numpy as np

_EPS = torch.finfo(torch.float32).eps
# Constant for - 1 / e.  This is the lowest 'z' for which principal / non-principal W
# is real valued (W(-1/e) = -1).  For any z < -1 / exp(1), W(z) = NA.
_EXP_INV = np.exp(-1)
_M_EXP_INV = -1 * _EXP_INV


def _lambertw_winitzki_approx(z: torch.Tensor) -> torch.Tensor:
    """
    Computes Winitzki approximation to Lambert W function at z >= -1/exp(1).

    Args:
        z: Value for which W(z) should be computed. Expected z >= -1/exp(1).

    Returns:
        lambertw_winitzki_approx: Approximation for W(z) for z >= -1/exp(1).
    """
    log1pz = torch.log1p(z)
    return log1pz * (1.0 - torch.log1p(log1pz) / (2.0 + log1pz))


def _halley_iteration(
    w: torch.Tensor, z: torch.Tensor, tol: float, iteration_count: int
) -> Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Halley's method on root finding of w for the equation w * exp(w) = z.

    Args:
        w (torch.Tensor): Current value of w.
        z (torch.Tensor): Value for which the root is being found.
        tol (float): Tolerance for convergence.
        iteration_count (int): Current iteration count.

    Returns:
        Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, int]:
            - bool: Whether the iteration should stop.
            - torch.Tensor: Updated value of w.
            - torch.Tensor: Input value z.
            - torch.Tensor: Delta value.
            - int: Updated iteration count.
    """
    f = w - z * torch.exp(-w)
    delta = f / (w + 1.0 - 0.5 * (w + 2.0) * f / (w + 1.0))
    w_next = w - delta
    converged = torch.abs(delta) <= tol * torch.abs(w_next)
    should_stop_next = torch.all(converged) or (iteration_count >= 100)
    return should_stop_next, w_next, z, delta, iteration_count + 1


def _fritsch_iteration(
    w: torch.Tensor, z: torch.Tensor, tol: float, iteration_count: int
) -> Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Root finding iteration for W(z) using Fritsch iteration.

    Args:
        w (torch.Tensor): Current value of w.
        z (torch.Tensor): Value for which the root is being found.
        tol (float): Tolerance for convergence.
        iteration_count (int): Current iteration count.

    Returns:
        Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, int]:
            - bool: Whether the iteration should stop.
            - torch.Tensor: Updated value of w.
            - torch.Tensor: Input value z.
            - torch.Tensor: Delta value.
            - int: Updated iteration count.
    """
    zn = torch.log(torch.abs(z)) - torch.log(torch.abs(w)) - w
    wp1 = w + 1.0
    q = 2.0 * wp1 * (wp1 + 2.0 / 3.0 * zn)
    q_minus_2zn = q - 2.0 * zn
    error = zn / wp1 * (1.0 + zn / q_minus_2zn)
    delta = torch.abs(error * w)
    converged = delta <= tol
    should_stop_next = torch.all(converged) or (iteration_count >= 100)
    return should_stop_next, w * (1.0 + error), z, delta, iteration_count + 1


def _lambertw_principal_branch_nonna(z):
    """Computes principal branch of Lambert W function for input with nonna output."""
    # check if z > -1 (vectorized)
    w = torch.where(z >= _M_EXP_INV, _lambertw_winitzki_approx(z), z)
    stop_condition = False
    counter = 0
    while not stop_condition:
        counter += 1
        stop_condition, w, z, _, _ = _halley_iteration(w, z, _EPS, counter)

    # if z = _M_EXP_INV, return exactly -1. If z = Inf, return Inf
    return torch.where(torch.abs(z - _M_EXP_INV) < _EPS, -1 * torch.ones_like(z), w)


def _lambertw_principal_branch(z: torch.Tensor) -> torch.Tensor:
    """Computes principal branch for z.

    For z < -1/exp(1), it returns nan; for z = inf, it returns inf.
    """
    w = torch.where(z >= _M_EXP_INV, _lambertw_principal_branch_nonna(z), torch.nan)
    return torch.where(torch.isposinf(z), torch.inf, w)


def _lambertw_nonprincipal_branch_nonna(z: torch.Tensor) -> torch.Tensor:
    """Computes the non-principal branch of z; only defined for z in [-1/exp(1), 0)."""
    # See eq (4.19) of Corless et al. (1996).
    L1 = torch.log(-z)
    L1_sq = L1 * L1

    L2 = torch.log(-L1)
    L2_sq = L2 * L2

    L3 = L2 / L1
    w = (
        L1
        - L2
        + L3
        + L3 * (-2.0 + L2) / (2.0 * L1)
        + (L3 * ((6.0 - 9.0 * L2 + 2 * L2_sq) / 6.0 * L1_sq))
    )

    stop_condition = False
    counter = 0
    while not stop_condition:
        counter += 1
        stop_condition, w, z, _, _ = _fritsch_iteration(w, z, _EPS, counter)
    # if z = _M_EXP_INV, return exactly -1.
    return torch.where(torch.abs(z - _M_EXP_INV) < _EPS, -1 * torch.ones_like(z), w)


def _lambertw_nonprincipal_branch(z: torch.Tensor) -> torch.Tensor:
    """Computes non-principal branch (-1) for z, denoted as Wm1(z).

    For z < -1/exp(1), it returns nan; for z = inf, it returns inf.
    """
    # non-principal branch is only defined for [-1/exp(1), 0). For z->0 Wm1(z) = -Inf.
    mask = (z >= _M_EXP_INV) & (z < 0.0)
    w = torch.where(mask, _lambertw_nonprincipal_branch_nonna(z), torch.nan)
    return torch.where(torch.abs(z) < _EPS, -1 * torch.inf, w)


def lambertw(z: torch.Tensor, branch: int = 0) -> torch.Tensor:
    """Computes the Lambert W function of 'z' for principal (0) and non-principal (-1) branch.

    Args:
        z: Value for which W(z) should be computed. Expected z >= -1/exp(1).

    Returns:
        W(z), a tensor of same shape and float dtype as input; with W(z, branch) values.
        Potentially contains NA and +/- Inf values.
    """
    if np.abs(branch) < _EPS:
        return _lambertw_principal_branch(z)
    elif np.abs(branch + 1) < _EPS:
        return _lambertw_nonprincipal_branch(z)
    else:
        raise NotImplementedError(f"branch={branch} is not implemented. Only 0 or -1.")
