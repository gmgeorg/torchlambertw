"""Base class holding core class and result definitions."""


from typing import Any, Dict
import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class DeltaEstimate:
    """Class for keeping Lambert W x F parameters."""

    delta: float
    method: str
    n_iterations: int
    converged: bool
    optimizer_result: Any


@dataclasses.dataclass
class LambertWEstimate:
    """Class for keeping Lambert W x F parameters."""

    dist_name: str
    theta_init: dict
    theta: dict
    beta: dict
    delta: float
    gamma: float
    tau: np.ndarray
    distribution: torch.distributions.Distribution
