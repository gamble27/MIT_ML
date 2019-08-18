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
        np.power(2 * np.pi * dispersion, x.shape[0]/2)
    )


def prior_probability(x: np.ndarray, mixture: GaussianMixture) -> float:
    """
    computes prior probability of given data point
    in the given mixture model

    :param x: given data point
    :param mixture: current mixture
    :return: prior probability of x
    """
    K = mixture.mu.shape[0]  # number of clusters
    prob = 0
    for j in range(K):
        prob += mixture.p[j] * normal(
            x, mixture.mu[j], mixture.var[j]
        )
    return prob


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
    n_clusters = mixture.mu.shape[0]
    n_cases = X.shape[0]

    priors = np.array([
        prior_probability(x, mixture)
        for x in X
    ])

    posteriors = np.zeros((n_cases, n_clusters))

    for i in range(n_cases):
        for j in range(n_clusters):
            posteriors[i, j] = mixture.p[j] * normal(
                X[i], mixture.mu[j], mixture.var[j]
            ) / priors[i]

    return posteriors, np.sum(np.log(priors))


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
    n_cases = post.shape[0]
    n_clusters = post.shape[1]

    n_cases_post = np.sum(post, axis=0)

    p = np.zeros((n_clusters,))  # here can be problems with shape in the future
    var = np.zeros((n_clusters,))
    mu = np.zeros((n_clusters, X.shape[1]))

    for j in range(n_clusters):
        p[j] = n_cases_post[j] / n_cases

        for i in range(n_cases):
            mu[j] += post[i, j] * X[i]
        mu[j] = mu[j] / n_cases_post[j]

        for i in range(n_cases):
            var[j] += post[i, j] * np.square(np.linalg.norm(X[i] - mu[j]))
        var[j] = var[j] / (n_cases_post[j] * X.shape[1])

    return GaussianMixture(mu, var, p)


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

    priors = np.array([
        prior_probability(x, mixture)
        for x in X
    ])
    old_likelihood = np.sum(np.log(priors))

    new_mixture = mixture

    while True:
        posteriors, new_likelihood = estep(X, new_mixture)
        new_mixture = mstep(X, posteriors)

        if new_likelihood - old_likelihood <= pow(10, -6) * np.abs(new_likelihood):
            return new_mixture, posteriors, new_likelihood
        else:
            old_likelihood = new_likelihood
