"""Module for constants used in module."""

from typing import Union
import torch

PARAM_DTYPE = Union[float, torch.Tensor]
EPS = torch.finfo(torch.float32).eps


LOCATION_FAMILY_DISTRIBUTIONS = [
    "Normal",
    "Uniform",
    "Cauchy",
    "Laplace",
    "Pareto",
    "StudentT",
]
