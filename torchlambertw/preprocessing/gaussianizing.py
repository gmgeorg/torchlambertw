"""Module for gaussianizing data using Lambert W x F transforms."""
import sklearn
import numpy as np
import torch

from typing import Dict, Any, Optional, Union
from torchlambertw.models import mle
from ..preprocessing import np_transforms
from ..preprocessing import base
from torchlambertw import transforms


class Gaussianizer(sklearn.base.TransformerMixin):
    """Module for column-wise Gaussianization using Lambert W x Normal transforms."""

    def __init__(
        self,
        lambertw_type: Union[str, base.LambertWType],
        mle_kwargs: Optional[dict] = None,
    ):
        if isinstance(lambertw_type, str):
            lambertw_type = base.LambertWType(lambertw_type)
        self.lambertw_type = lambertw_type
        self.lambertw_result = None
        self.mle_kwargs = mle_kwargs
        self.mle_per_col: Dict[int, Any] = {}

    def fit(self, data: np.ndarray):
        if len(data.shape) == 1:
            data = data[:, np.newaxis]

        n_cols = data.shape[1]
        for i in range(n_cols):
            tmp = mle.MLE(dist_name="normal", **self.mle_kwargs)
            tmp.fit(data[:, i])
            self.mle_per_col[i] = tmp
            del tmp
        return self

    def transform(self, data) -> np.ndarray:
        """Transforms data to a gaussiani version."""
        is_univariate_input = False
        if len(data.shape) == 1:
            is_univariate_input = True
            data = data[:, np.newaxis]

        result = np.zeros_like(data)
        n_cols = len(self.mle_per_col)
        for i in range(n_cols):
            tmp_trafo = transforms.LambertWTailTransform(
                shift=torch.tensor(self.mle_per_col[i].params_.beta["loc"]),
                scale=torch.tensor(self.mle_per_col[i].params_.beta["scale"]),
                tailweight=torch.tensor(
                    self.mle_per_col[i].params_.lambertw_params.delta
                ),
            )
            result[:, i] = np_transforms.normalize_by_tau(
                y=data[:, i],
                tau=base.Tau(
                    loc=self.mle_per_col[i].params_.beta["loc"],
                    scale=self.mle_per_col[i].params_.beta["scale"],
                    lambertw_params=self.mle_per_col[i].params_.lambertw_params,
                ),
            )
            # result[:, i] = tmp_trafo(torch.tensor(data[:, i])).numpy()
        if is_univariate_input:
            result = result.ravel()
        return result
