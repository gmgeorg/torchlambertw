"""Base class holding core class and result definitions."""


import dataclasses


@dataclasses.dataclass
class LambertWResult:
    """Class for keeping Lambert W x F parameters."""

    dist_name: str
    theta: dict
    beta: dict
    delta: float
    gamma: float
    tau: np.ndarray
    distribution: torch.distribution.Distribution
