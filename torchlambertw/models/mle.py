"""Module for maximum likelihood estimation of univariate data for Lambert W x F distributions.


In particular Lambert W x Normal distribution with ability to gaussianize data.
"""
import sklearn
import torch
import numpy as np
from typing import Optional
from torch.optim import lr_scheduler

import torchlambertw.distributions
from torchlambertw.models import igmm
from . import base


class MLE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Computes the MLE for univariate Lambert W x F distributions."""

    def __init__(
        self,
        dist_name: str,
        distribution: Optional[torch.distributions.Distribution] = None,
        n_init: int = 100,
        lr: float = 0.05,
        verbose: int = 0,
    ):
        self.distribution = distribution
        self.dist_name = dist_name
        self.n_init = n_init
        self.lr = lr
        self.verbose = verbose
        # estimated parameters
        self.params_ = {}
        self.init_params = {}
        self.optim_params = {}
        self.losses = []

    def _initialize_params(self, data):
        _eps = 1e-4

        theta_init = base.Theta(
            delta=igmm.delta_gmm(data, not_negative=True).delta,
            beta={
                "loc": np.median(data),
                "scale": data.std(),  # adjust based on delta prior estimate
            },
            gamma=0.0,
        )
        self.init_params = theta_init

        self.optim_params["loc"] = torch.tensor(
            theta_init.beta["loc"], requires_grad=True
        )
        self.optim_params["log_sigma"] = torch.tensor(
            np.log(theta_init.beta["scale"] + _eps), requires_grad=True
        )
        self.optim_params["log_delta"] = torch.tensor(
            np.log(theta_init.delta + _eps), requires_grad=True
        )

    def fit(self, data: np.ndarray):
        """Trains the MLE of a Lambert W distribution based on torch likelihood optimization."""
        self._initialize_params(data)
        init_params = list(self.optim_params.values())
        optimizer = torch.optim.NAdam(init_params, lr=self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        tr_data = torch.tensor(data)
        for epoch in range(self.n_init):
            optimizer.zero_grad()  # Clear gradients
            loglik = (
                torchlambertw.distributions.TailLambertWNormal(
                    self.optim_params["loc"],
                    torch.exp(self.optim_params["log_sigma"]),
                    torch.exp(self.optim_params["log_delta"]),
                )
                .log_prob(tr_data)
                .sum()
            )  # Calculate log likelihood

            loss = -loglik  # Negative log likelihood as the loss
            loss.backward()  # Backpropagate gradients
            optimizer.step()  # Update parameters
            scheduler.step()
            self.losses.append(loss.item())

            if self.verbose:
                if (epoch + 1) % self.verbose == 0:
                    print(f"Epoch [{epoch+1}/{self.n_init}], Loss: {loss.item()}")

        self.params_ = base.Theta(
            delta=np.exp(self.optim_params["log_delta"].detach().numpy()),
            beta={
                "loc": float(self.optim_params["loc"].detach().numpy()),
                "scale": np.exp(self.optim_params["log_sigma"].detach().numpy()),
            },
        )
        if self.verbose:
            print("MLE: ", self.params_)
        return self
