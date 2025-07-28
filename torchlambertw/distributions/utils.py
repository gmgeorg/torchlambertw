"""Utilities for the distributions module."""

from typing import Tuple, List, Optional

import inspect
import torch

from .. import constants as con

_DROP_ARGS_NAMES = ["self", "validate_args", "logits", "args", "kwargs"]


def is_location_family(distribution_name: str) -> bool:
    """Is the distribution a location family."""
    assert isinstance(distribution_name, str)
    return distribution_name in con.LOCATION_FAMILY_DISTRIBUTIONS


def get_distribution_name(distribution: torch.distributions.Distribution) -> str:
    """Get the name of the torch.distribution.Distribution"""
    dist_type = type(distribution)
    dist_name = dist_type.__name__
    return dist_name


def get_distribution_constructor(
    distribution_name: str,
) -> torch.distributions.Distribution:
    """Gets the distribution constructor given the name."""
    assert isinstance(distribution_name, str)
    # Use getattr to get the distribution class constructor by name
    return getattr(torch.distributions, distribution_name)


def get_distribution_args(
    distribution_constructur: torch.distributions.Distribution,
) -> List[str]:
    """Gets the parameter names for a distribution constructor."""
    constructor_signature = inspect.signature(distribution_constructur.__init__)
    args = [
        param for param in constructor_signature.parameters.keys() if param not in _DROP_ARGS_NAMES
    ]

    return args


def to_tensor(x: con.PARAM_DTYPE) -> torch.Tensor:
    """Converts to tensor if a float/int."""
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x)

    return x


def update_shift_scale(
    shift: Optional[con.PARAM_DTYPE],
    scale: Optional[con.PARAM_DTYPE],
    distribution: torch.distributions.Distribution,
    use_mean_variance: bool,
) -> Tuple[con.PARAM_DTYPE, con.PARAM_DTYPE]:
    """Updates shift/scale parameters depending on distribution."""
    is_loc_fam = is_location_family(get_distribution_name(distribution))
    if shift is None:
        shift = distribution.mean
        if not use_mean_variance and is_loc_fam:
            shift = distribution.loc

        if not is_loc_fam:
            shift = 0.0

    if scale is None:
        scale = distribution.stddev

        if not use_mean_variance:
            scale = distribution.scale

    return (to_tensor(shift), to_tensor(scale))
