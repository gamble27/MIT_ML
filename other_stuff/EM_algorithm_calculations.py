import numpy as np
import unittest


def squared_norm(x):
    return np.square(np.abs(x))


def normal(x, mean, dispersion):
    """
    computes probability density for given value
    in normal distribution
    with given mean and dispersion parameters

    :param x: float
    :param mean: float mean of distribution
    :param dispersion: float dispersion of distribution
    :return: float probability density
    """
    return np.exp(
        -1 * squared_norm(x - mean) / (2 * dispersion)
    ) / (
        np.sqrt(2 * np.pi * dispersion)
    )


def prior_probability(x, means, dispersions, cluster_probabilities):
    """
    computes prior probability of given data point
    in the given mixture model

    :param x: given data point
    :param means: means of given K clusters
    :param dispersions: dispersions of given K clusters
    :param cluster_probabilities: probabilities of K clusters
    :return: prior probability of x
    """
    K = means.shape[0]  # number of clusters
    prob = 0
    for j in range(K):
        prob += cluster_probabilities[j] * normal(
            x, means[j], dispersions[j]
        )
    return prob


def EM_step(X, means, dispersions, cluster_probabilities):
    """
    performs a step of EM algorithm

    :param X: array of given data
    :param means: means of given K clusters
    :param dispersions: dispersions of given K clusters
    :param cluster_probabilities: probabilities of K clusters
    :return:
    """
    n_clusters = means.shape[0]
    n_cases = X.shape[0]

    probabilities = np.zeros((n_clusters, n_cases))
    updated_means = np.zeros((n_clusters,))
    updated_cluster_probabilities = np.zeros((n_clusters,))
    updated_dispersions = np.zeros((n_clusters,))

    for j in range(n_clusters):
        # E step
        for i in range(n_cases):
            probabilities[j, i] = cluster_probabilities[j] * normal(
                X[i], means[j], dispersions[j]
            ) / prior_probability(
                X[i], means, dispersions, cluster_probabilities
            )

        # M step
        n_cases_j = np.sum(probabilities, axis=1)[j]
        updated_cluster_probabilities[j] = n_cases_j / n_cases
        updated_means[j] = np.dot(
            probabilities[j, :], X
        ) / n_cases_j

        squared_norm_vectorized = np.vectorize(squared_norm)
        updated_mean_vectorized = np.ones(X.shape) * updated_means[j]

        updated_dispersions[j] = np.dot(
            probabilities[j, :], squared_norm_vectorized(X - updated_mean_vectorized)
        ) / n_cases_j

    return updated_means, updated_dispersions, updated_cluster_probabilities


class EMTestingModules(unittest.TestCase):

    def setUp(self) -> None:
        self.X = np.array([0.2, -0.9, -1, 1.2, 1.8])
        self.means = np.array([-3, 2])
        self.dispersions = np.array([4, 4])
        self.cluster_probabilities = np.array([0.5, 0.5])

    def test_1_scalar(self):
        """
        lecture 14 exercise
        """
        self.assertIsNotNone(
            EM_step(self.X, self.means, self.dispersions, self.cluster_probabilities)
        )

    def test_2_normal(self):
        """
        normal distribution
        for all data points
        in the 1st cluster
        """
        print("test 2: normal distributions")

        mean = self.means[0]
        dispersion = self.dispersions[0]

        for i, x in enumerate(self.X):
            print(i+1, normal(x, mean, dispersion), sep=' : ')

    def test_3_prior(self):
        """
        prior probabilities
        for all data points
        in the tested model
        """
        print("test 3: prior probabilities")

        for i, x in enumerate(self.X):
            print(i+1, prior_probability(
                x, self.means, self.dispersions, self.cluster_probabilities
            ), sep=' : ')


if __name__ == "__main__":
    unittest.main()
