"""Module for holding transformation paramters."""

from typing import Any, Dict, Union
import dataclasses

import enum
import numpy as np

_EPS = np.finfo(np.float32).eps

_FLOAT_OR_ARRAY = Union[float, np.ndarray]


class LambertWType(enum.Enum):
    """Which type of Lambert W x F transformations."""

    # Skewed Lambert W x F
    S = "s"
    # Heavy-tailed Lambert W x F
    H = "h"
    # double heavy-tailed Lambert W x F
    HH = "hh"


def _is_eq_value(x: _FLOAT_OR_ARRAY, value: float) -> bool:
    """Returns True if 'x' is equal to value; False otherwise."""
    l1_norm = np.sum(np.abs(x - value))
    return l1_norm < _EPS


def _to_two_dim_array(x: _FLOAT_OR_ARRAY) -> np.ndarray:
    if isinstance(x, float):
        return np.array([[x]])
    if isinstance(x, np.ndarray):
        if len(x.shape) == 0:
            return np.array([[x]])
        if len(x.shape) == 1:
            return x[:, np.newaxis]
        if len(x.shape) == 2:
            return x

    raise ValueError("Could not convert successfully to 2-dim array.")


def _check_params(
    gamma: _FLOAT_OR_ARRAY, delta: _FLOAT_OR_ARRAY, alpha: _FLOAT_OR_ARRAY
) -> None:
    """checks that parameters are correctly set."""
    if not _is_eq_value(gamma, 0.0):
        assert _is_eq_value(delta, 0.0)
        assert _is_eq_value(alpha, 1.0)

    if not _is_eq_value(delta, 0.0):
        assert _is_eq_value(gamma, 0.0)

    assert np.all(alpha > 0.0)

    assert 1 <= delta.shape[1] <= 2


@dataclasses.dataclass
class LambertWParams:
    """Class for keeping Lambert W x F parameters."""

    gamma: np.ndarray = 0.0
    delta: np.ndarray = 0.0
    alpha: np.ndarray = 1.0

    def __post_init__(self):
        self.gamma = _to_two_dim_array(self.gamma)
        self.delta = _to_two_dim_array(self.delta)
        self.alpha = _to_two_dim_array(self.alpha)
        _check_params(self.gamma, self.delta, self.alpha)

    @property
    def lambertw_type(self):
        if not _is_eq_value(self.gamma, 0.0):
            return LambertWType.S

        if self.delta.shape[1] == 1:
            return LambertWType.H

        if self.delta.shape[1] == 2:
            return LambertWType.HH

        raise ValueError(
            "Lambert W Parameters gamma, delta, alpha do not uniquely identify the type."
        )


@dataclasses.dataclass
class Theta:
    """Class for keeping Lambert W x F parameters."""

    beta: Dict[
        str,
        _FLOAT_OR_ARRAY,
    ]
    lambertw_params: LambertWParams


@dataclasses.dataclass
class Tau:
    """Class for keeping Lambert W x F parameters for transforming data.

    Uses loc/scale for both mean_variance and location_scale families.
    """

    loc: np.ndarray
    scale: np.ndarray
    lambertw_params: LambertWParams
