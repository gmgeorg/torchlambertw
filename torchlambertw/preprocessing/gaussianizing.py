"""Module for gaussianizing data using Lambert W x F transforms."""
import sklearn
import numpy as np
import torch

from typing import Dict, Any, Optional, Union
from torchlambertw.models import mle
from torchlambertw.models import igmm
from ..preprocessing import np_transforms
from ..preprocessing import base
from torchlambertw import transforms


class Gaussianizer(sklearn.base.TransformerMixin):
    """Module for column-wise Gaussianization using Lambert W x Normal transforms."""

    def __init__(
        self,
        lambertw_type: Union[str, base.LambertWType],
        method: str = "igmm",
        method_kwargs: Optional[dict] = None,
    ):
        """Initializes the class."""
        if isinstance(lambertw_type, str):
            lambertw_type = base.LambertWType(lambertw_type)
        self.lambertw_type = lambertw_type
        self.method = method

        if self.method not in ["mle", "igmm"]:
            raise NotImplementedError(f"method={method} is not implemented.")

        self.method_kwargs = method_kwargs or {}
        self.estimators_per_col: Dict[int, Any] = {}

    def fit(self, data: np.ndarray):
        """Trains a Gaussianizer for every column of 'data'."""
        if len(data.shape) == 1:
            data = data[:, np.newaxis]

        n_cols = data.shape[1]
        for i in range(n_cols):
            if self.method == "mle":
                estimate_clf = mle.MLE(dist_name="normal", **self.method_kwargs)
            elif self.method == "igmm":
                estimate_clf = igmm.IGMM(lambertw_type=self.lambertw_type)
            else:
                raise NotImplementedError(f"method={self.method} is not implemented")

            estimate_clf.fit(data[:, i])
            self.estimators_per_col[i] = estimate_clf
            del estimate_clf
        return self

    def transform(self, data) -> np.ndarray:
        """Transforms data to a gaussiani version."""
        is_univariate_input = False
        if len(data.shape) == 1:
            is_univariate_input = True
            data = data[:, np.newaxis]

        result = np.zeros_like(data)
        n_cols = len(self.estimators_per_col)
        for i in range(n_cols):
            if self.method == "igmm":
                tau_tmp = self.estimators_per_col[i].params_
            elif self.method == "mle":
                tau_tmp = p_base.Tau(
                    loc=self.estimators_per_col[i].params_.beta["loc"],
                    scale=self.estimators_per_col[i].params_.beta["scale"],
                    lambertw_params=self.estimators_per_col[i].params_.lambertw_params,
                )
            else:
                raise NotImplementedError(f"method={method} is not implemented.")
            result[:, i] = np_transforms.normalize_by_tau(y=data[:, i], tau=tau_tmp)
        if is_univariate_input:
            result = result.ravel()
        return result
