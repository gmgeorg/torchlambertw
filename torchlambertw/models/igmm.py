"""Module for iterative generalized methods of moments (IGMM) estimation."""

from typing import Optional

import numpy as np
import scipy.stats
import numpy as np
import warnings

from ..preprocessing import np_transforms
from . import base
import scipy.optimize


def kurtosis(x):
    """Computes kurtosis of data.  For normal distribution this will be 3 (ie not excess kurtosis)."""
    return scipy.stats.kurtosis(x) + 3


def delta_taylor(
    y: np.ndarray, kurtosis_y: Optional[float] = None, dist_name: str = "normal"
):
    """Computes the taylor approximation of the 'delta' parameter given univariate data."""
    if kurtosis_y is None:
        kurtosis_y = kurtosis(y)

    if not isinstance(kurtosis_y, (int, float)) or kurtosis_y <= 0:
        raise ValueError("kurtosis_y must be a positive numeric value")

    if dist_name == "normal":
        if 66 * kurtosis_y - 162 > 0:
            delta_hat = max(0, 1 / 66 * (np.sqrt(66 * kurtosis_y - 162) - 6))
            delta_hat = min(delta_hat, 2)
        else:
            delta_hat = 0.0
    else:
        raise NotImplementedError(
            "Other distribution than 'normal' is not supported for the Taylor approximation."
        )

    return float(delta_hat)


def delta_gmm(
    z: np.ndarray,
    type: str = "h",
    kurtosis_x: float = 3.0,
    skewness_x: float = 0.0,
    delta_init: Optional[float] = None,
    tol: float = np.finfo(float).eps ** 0.25,
    not_negative: bool = False,
    lower: float = -1.0,
    upper: float = 3.0,
):
    """Computes an estimate of delta (tail parameter) per Taylor approximation of the kurtosis."""
    assert isinstance(kurtosis_x, (int, float))
    assert isinstance(skewness_x, (int, float))
    assert len(delta_init) <= 2 if delta_init is not None else True
    assert tol > 0
    assert lower < upper

    delta_init = delta_init or delta_taylor(z)

    def _obj_fct(delta: float):
        if not_negative:
            # convert delta to > 0
            delta = np.exp(delta)
        u_g = np_transforms.W_delta(z, delta=delta)
        if np.any(np.isinf(u_g)):
            return kurtosis_x ** 2

        empirical_kurtosis = kurtosis(u_g)
        # for delta -> Inf, u.g can become (numerically) a constant vector
        # thus kurtosis(u.g) = NA.  In this case set empirical.kurtosis
        # to a very large value and continue.
        if np.isnan(empirical_kurtosis):
            empirical_kurtosis = 1e6
            warnings.warning(
                "Kurtosis estimate was NA. ",
                "Set to large value (",
                empirical_kurtosis,
                ") for optimization to continue.\n",
                "Please double-check results (in particular the 'delta' ",
                "estimate).",
            )
        return (empirical_kurtosis - kurtosis_x) ** 2

    if not_negative:
        delta_init = np.log(delta_init + 0.001)

    delta_estimate: base.DeltaEstimate = None
    if not_negative:
        res = scipy.optimize.minimize(
            _obj_fct, delta_init, method="BFGS", tol=tol, options={"disp": False}
        )
        delta_estimate = base.DeltaEstimate(
            delta=res.x[0],
            iterations=res.nit,
            method="taylor",
            converged=res.success,
            optimizer_result=res,
        )
    else:
        res = scipy.optimize.minimize_scalar(
            _obj_fct, bounds=(lower, upper), method="bounded", options={"xatol": tol}
        )
        delta_estimate = base.DeltaEstimate(
            delta=res.x,
            iterations=res.nfev,
            method="taylor",
            converged=res.success,
            optimizer_result=res,
        )

    delta_hat = delta_estimate.delta
    if not_negative:
        delta_hat = np.exp(delta_hat)
        if np.abs(delta_hat - 1) < 1e-7:
            delta_hat = np.round(delta_hat, 6)

    delta_hat = np.minimum(np.maximum(delta_hat, lower), upper)
    delta_estimate.delta = delta_hat
    return delta_estimate
