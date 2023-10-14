"""Computes centralized moments of data."""

import scipy


def kurtosis(x):
    """Computes kurtosis of data.  For normal distribution this will be 3 (ie not excess kurtosis)."""
    return scipy.stats.kurtosis(x) + 3


def skewness(x):
    """Computes skewness of data.  For normal distribution this will be 0."""
    return scipy.stats.skewness(x)
