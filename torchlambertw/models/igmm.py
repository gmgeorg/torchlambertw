"""Module for iterative generalized methods of moments (IGMM) estimation."""

from typing import Optional

import numpy as np
import scipy.stats
import numpy as np
import warnings
import sklearn
import scipy.optimize

from ..preprocessing import np_transforms
from ..preprocessing import base as p_base
from . import base
from . import moments


def delta_taylor(
    y: np.ndarray, kurtosis_y: Optional[float] = None, dist_name: str = "normal"
):
    """Computes the taylor approximation of the 'delta' parameter given univariate data."""
    if kurtosis_y is None:
        kurtosis_y = moments.kurtosis(y)

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

        empirical_kurtosis = moments.kurtosis(u_g)
        # for delta -> Inf, u.g can become (numerically) a constant vector
        # thus kurtosis(u.g) = NA.  In this case set empirical.kurtosis
        # to a very large value and continue.
        if np.isnan(empirical_kurtosis):
            empirical_kurtosis = 1e6

            error_msg = f"""
            Kurtosis estimate was NA. Setting to large value ({empirical_kurtosis})
            for optimization to continue.\n Double-check results (in particular the 'delta'
            estimate)        
            """

            warnings.warn(error_msg)
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
            n_iterations=res.nit,
            method="gmm",
            converged=res.success,
            optimizer_result=res,
        )
    else:
        res = scipy.optimize.minimize_scalar(
            _obj_fct, bounds=(lower, upper), method="bounded", options={"xatol": tol}
        )
        delta_estimate = base.DeltaEstimate(
            delta=res.x,
            n_iterations=res.nfev,
            method="gmm",
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


class IGMM(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Computes the IGMM for multivariate (column-wise) Lambert W x F distributions."""

    def __init__(
        self,
        lambertw_type: str = "h",
        skewness_x: float = 0.0,
        kurtosis_x: float = 3.0,
        max_iter: int = 100,
        lr: float = 0.01,
        not_negative: bool = True,
        location_family: bool = True,
        lower: float = 0.0,
        upper: float = 3.0,
        tolerance: float = 1e-6,
        verbose: int = 0,
    ):
        assert max_iter > 0
        assert verbose >= 0

        self.lambertw_type = p_base.LambertWType(lambertw_type)
        self.skewness_x = skewness_x
        self.kurtosis_x = kurtosis_x
        self.max_iter = max_iter
        self.lr = lr
        self.verbose = verbose
        self.tolerance = tolerance

        self.location_family = location_family
        self.not_negative = not_negative
        self.lower = lower
        self.upper = upper
        self.total_iter = 0
        # estimated parameters
        self.params_ = {}
        self.init_params = {}

    def _initialize_params(self, data):
        lambertw_params_init = p_base.LambertWParams(
            delta=delta_gmm(data, not_negative=True).delta,
        )
        loc_est = np.median(data)
        u_init = np_transforms.W_delta(data - loc_est, delta=lambertw_params_init.delta)

        tau_init = p_base.Tau(
            loc=np.mean(u_init),
            scale=u_init.std(),
            lambertw_params=lambertw_params_init,
        )
        self.init_params = tau_init
        self.trace_params = None

    def fit(self, data: np.ndarray):
        """Trains the IGMM of a Lambert W x F distribution based on methods of moments."""
        self._initialize_params(data)

        tau_trace = np.zeros(shape=(self.max_iter + 1, 3))
        tau_trace[0,] = (
            self.init_params.loc,
            self.init_params.scale,
            self.init_params.lambertw_params.delta,
        )

        for kk in range(self.max_iter):
            current = tau_trace[kk, :]
            if self.verbose:
                if (kk) % self.verbose == 0:
                    print(f"Epoch [{kk}/{self.max_iter}], Params: {current}")

            tau_tmp = p_base.Tau(
                loc=current[0],
                scale=current[1],
                lambertw_params=p_base.LambertWParams(delta=current[2]),
            )
            zz = (data - tau_tmp.loc) / tau_tmp.scale

            delta_estimate = delta_gmm(
                zz,
                delta_init=tau_tmp.lambertw_params.delta,
                kurtosis_x=self.kurtosis_x,
                tol=self.tolerance,
                not_negative=self.not_negative,
                lower=self.lower,
                upper=self.upper,
            )
            delta_hat = delta_estimate.delta

            uu = np_transforms.W_delta(zz, delta=delta_hat)
            xx = uu * tau_tmp.scale + tau_tmp.loc
            tau_trace[
                kk + 1,
            ] = (np.mean(xx), np.std(xx), delta_hat)
            if not self.location_family:
                tau_trace[kk + 1, 0] = 0.0

            self.total_iter += delta_estimate.n_iterations
            tau_diff = tau_trace[kk + 1] - tau_trace[kk]
            if np.linalg.norm(tau_diff) < self.tolerance:
                break

        self.trace_params = tau_trace[: (kk + 1)]
        se = np.array([1, np.sqrt(1 / 2), 1]) / np.sqrt(len(data))

        self.params_ = p_base.Tau(
            lambertw_params=p_base.LambertWParams(
                delta=tau_trace[kk, 2],
            ),
            loc=tau_trace[kk, 0],
            scale=tau_trace[kk, 1],
        )
        if self.verbose:
            print("IGMM: ", self.params_)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transforms data y to the data based on IGMM estimate tau."""
        return np_transforms.normalize_by_tau(data, tau=self.params_)
