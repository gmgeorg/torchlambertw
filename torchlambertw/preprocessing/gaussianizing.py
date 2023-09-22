"""Module for gaussianizing data using Lambert W x F transforms."""
import sklearn
import numpy as np
import torch

from typing import Dict, Any, Optional
from torchlambertw.models import mle
import torchlambertw.models.w_transforms
from torchlambertw import transforms


class Gaussianizer(sklearn.base.TransformerMixin):
    """Module for column-wise Gaussianization using Lambert W x Normal transforms."""

    def __init__(self, type: str, mle_kwargs: Optional[dict] = None):
        self.type = type
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
                tailweight=torch.tensor(self.mle_per_col[i].params_.delta),
            )
            result[:, i] = tmp_trafo(torch.tensor(data[:, i])).numpy()
        if is_univariate_input:
            result = result.ravel()
        return result
