"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from project3_netflix.common import GaussianMixture


def normal(x: np.ndarray, mean: np.ndarray, dispersion: float) -> float:
    """
    computes probability density for given value
    in normal distribution
    with given mean and dispersion parameters

    :param x: (1,d) array holding data point
    :param mean: (1,d) array holding mean
    :param dispersion: float dispersion of distribution

    :return: float probability density
    """
    return np.exp(
        -1 * np.square(np.linalg.norm(x - mean)) / (2 * dispersion)
    ) / (
        np.sqrt(2 * np.pi * dispersion)
    )


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    counts = np.zeros((
        X.shape[0],
        mixture.p.shape[0]
    ))
    counts = mixture.p * normal()
    for prob in mixture.p:
        counts[:, ] = 0
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    raise NotImplementedError
