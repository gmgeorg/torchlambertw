"""Module for maximum likelihood estimation of univariate data for Lambert W x F distributions.


In particular Lambert W x Normal distribution with ability to gaussianize data.
"""
import sklearn
import torch
import numpy as np
from typing import Optional
import torchlambertw.distributions
from torchlambertw.models import igmm


class MLE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Computes the MLE for univariate Lambert W x F distributions."""

    def __init__(
        self,
        dist_name: str,
        distribution: Optional[torch.distributions.Distribution] = None,
        n_init: int = 10,
        lr: float = 0.001,
        verbose: int = 0,
    ):
        self.distribution = distribution
        self.dist_name = dist_name
        self.n_init = n_init
        self.lr = lr
        self.verbose = verbose
        # estimated parameters
        self.params_ = {}
        self.optim_params = {}

    def _initialize_params(self, data):
        _eps = 1e-4
        delta_init = igmm.delta_taylor(data)
        self.optim_params["mu"] = torch.tensor(data.mean(), requires_grad=True)
        self.optim_params["log_sigma"] = torch.tensor(
            np.log(data.std() + _eps), requires_grad=True
        )
        self.optim_params["log_delta"] = torch.tensor(
            np.log(igmm.delta_taylor(data) + _eps), requires_grad=True
        )

    def fit(self, data: np.ndarray):
        self._initialize_params(data)
        init_params = list(self.optim_params.values())
        optimizer = torch.optim.Adam(init_params, lr=self.lr)
        tr_data = torch.tensor(data)
        for epoch in range(self.n_init):

            optimizer.zero_grad()  # Clear gradients
            loglik = (
                torchlambertw.distributions.LambertWNormal(
                    self.optim_params["mu"],
                    torch.exp(self.optim_params["log_sigma"]),
                    torch.exp(self.optim_params["log_delta"]),
                )
                .log_prob(tr_data)
                .sum()
            )  # Calculate log likelihood

            loss = -loglik  # Negative log likelihood as the loss
            loss.backward()  # Backpropagate gradients
            optimizer.step()  # Update parameters
            if self.verbose:
                if (epoch + 1) % self.verbose == 0:
                    print(f"Epoch [{epoch+1}/{self.n_init}], Loss: {loss.item()}")

        self.params_["mu"] = float(self.optim_params["mu"].detach().numpy())
        self.params_["sigma"] = np.exp(self.optim_params["log_sigma"].detach().numpy())
        self.params_["delta"] = np.exp(self.optim_params["log_delta"].detach().numpy())
        if self.verbose:
            print("MLE: ", self.params_)
        return self

    def transform(self, data):
        return self

    def inverse_transform(self, data):
        return self
