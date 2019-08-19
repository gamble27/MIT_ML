"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from project3_netflix.common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n_cases = X.shape[0]
    n_clusters = mixture.p.shape[0]

    Cu = np.abs(np.sign(X))

    f = np.zeros((n_cases, n_clusters))
    for j in range(n_clusters):
        for i in range(n_cases):
            d = np.sum(np.abs(np.sign(X[i])))
            f[i, j] = (np.log(mixture.p[j] + 1e-16) -
                       (np.linalg.norm((X[i] - mixture.mu[j])*Cu[i]) ** 2) / (2 * mixture.var[j]) -
                       np.log(2*np.pi*mixture.var[j]) * d / 2)

    log_post = np.zeros((n_cases, n_clusters))
    for i in range(n_cases):
        log_post[i] = f[i] - logsumexp(f[i])

    log_likelihood = np.sum(logsumexp(f, axis=1))
    return np.exp(log_post), log_likelihood


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    d = X.shape[1]
    n_cases = X.shape[0]
    n_clusters = mixture.var.shape[0]

    Cu = np.abs(np.sign(X))

    p = np.sum(post, axis=0) / n_cases
    var = np.zeros((n_clusters,))
    mu = np.zeros((n_clusters, d))

    for j in range(n_clusters):
        for l in range(d):
            denom = np.sum(post[:, j] * Cu[:, l])
            if denom >= 1:
                mu[j, l] = np.sum(post[:, j] * Cu[:, l] * X[:, l]) / denom
            else:
                mu[j, l] = mixture.mu[j, l]
        for i in range(n_cases):
            var[j] += post[i, j] * np.linalg.norm((X[i] - mu[j])*Cu[i])**2
        var[j] /= np.sum(post[:, j] * Cu.sum(axis=1))
        var[j] = max(var[j], min_variance)

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
    posteriors, old_likelihood = estep(X, mixture)
    new_mixture = mstep(X, posteriors, mixture)

    while True:
        posteriors, new_likelihood = estep(X, new_mixture)
        new_mixture = mstep(X, posteriors, new_mixture)

        if new_likelihood - old_likelihood <= pow(10, -6) * np.abs(new_likelihood):
            return new_mixture, posteriors, new_likelihood
        else:
            old_likelihood = new_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n_cases = X.shape[0]
    n_clusters = mixture.p.shape[0]

    Cu = np.abs(np.sign(X))

    f = np.zeros((n_cases, n_clusters))
    for j in range(n_clusters):
        for i in range(n_cases):
            d = np.sum(np.abs(np.sign(X[i])))
            f[i, j] = (np.log(mixture.p[j] + 1e-16) -
                       (np.linalg.norm((X[i] - mixture.mu[j]) * Cu[i]) ** 2) / (2 * mixture.var[j]) -
                       np.log(2 * np.pi * mixture.var[j]) * d / 2)

    log_post = np.zeros((n_cases, n_clusters))
    for i in range(n_cases):
        log_post[i] = f[i] - logsumexp(f[i])
    post = np.exp(log_post)

    new_X = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] > 0:
                new_X[i, j] = X[i, j]
            else:
                new_X[i, j] = np.sum(post[i, :] * mixture.mu[:, j]) / np.sum(post[i, :])

    return new_X
