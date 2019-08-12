import numpy as np
import unittest


def l1_norm(x):
    """
    l1 norm (Manhattan distance)

    :param x: ndarray vector
    :return: l1 norm of x
    """
    norm = 0
    for x_i in x:
        norm += np.abs(x_i)
    return norm


def l2_norm(x):
    """
    l2 norm (Euclidean distance)

    :param x: ndarray vector
    :return: l2 norm of x
    """
    return np.linalg.norm(x)


def distance_matrix(cluster, norm=np.linalg.norm):
    """
    forms a distance matrix
    between cluster members
    :param norm: norm function
    :param cluster: ndarray n_members*d
    :return: ndarray n_members*n_members
    """
    n_members = cluster.shape[0]
    distances = np.zeros((n_members, n_members))
    for i in range(n_members):
        for j in range(n_members):
            distances[i, j] = distances[j, i] = norm(
                cluster[i] - cluster[j]
            )
    return distances


def k_medoids(data_points, medoids, norm=np.linalg.norm):
    """
    performs k-medoids algorithm
    until convergence

    k - number of clusters

    :param norm: norm function
    :param data_points: data for clustering, n*d ndarray
    :param medoids: k points from data_points, k*d ndarray
    :return: clusters (len=k dictionary,
    keys = cluster centers as tuples,
    values = ndarray ?*d)
    """
    # here goes do-while loop made by me. sorry.
    while True:  # ouch!
        # forward propagation
        labels = {tuple(m): [] for m in medoids}
        for x in data_points:
            shortest_distance = norm(x - medoids[0])
            curr_label = medoids[0]
            for m in medoids[1:]:
                if norm(x - m) < shortest_distance:
                    curr_label = m
            labels[tuple(curr_label)].append(x)

        # back propagation
        new_labels = {}
        for cluster in labels.values():
            distances = distance_matrix(np.array(cluster), norm)
            dist_sums = np.sum(distances, axis=1)
            idx = 0
            for i in range(1,dist_sums.shape[0]):
                if dist_sums[i] < dist_sums[idx]:
                    idx = i
            new_labels[tuple(cluster[idx])] = cluster

        # do-while loop ending
        l1 = np.array(list(labels.keys()))
        medoids = np.array(list(new_labels.keys()))
        if np.equal(l1, medoids).all():
            break

    return new_labels


def k_means(data_points, centers, norm=np.linalg.norm):
    """
    performs k-means algorithm
    until convergence

    :param norm: norm function
    :param data_points: data for clustering, n*d ndarray
    :param centers: k points - cluster centers, k*d ndarray
    :return: clusters (len=k dictionary,
             keys = cluster centers as tuples,
             values = ndarray ?*d)
    """
    # here goes do-while loop made by me. sorry.
    while True:  # ouch!
        # forward propagation
        labels = {tuple(m): [] for m in centers}
        for x in data_points:
            shortest_distance = norm(x - centers[0])
            curr_label = centers[0]
            for m in centers[1:]:
                if norm(x - m) < shortest_distance:
                    curr_label = m
            labels[tuple(curr_label)].append(x)

        # back propagation
        new_labels = {}
        for cluster in labels.values():
            new_center = np.array(cluster).mean(axis=0)
            new_labels[tuple(new_center)] = cluster

        # do-while loop ending
        l1 = np.array(list(labels.keys()))
        centers = np.array(list(new_labels.keys()))
        if np.equal(l1, centers).all():
            break

    return new_labels


class ClusteringTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.data_points = np.array([
            [0, -6],
            [4, 4],
            [0, 0],
            [-5, 2]
        ])
        # self.n_clusters = 2
        self.initialized_means = np.array([
            self.data_points[0],
            self.data_points[-1]
        ])

    def test_k_means(self):
        clusters = k_means(
            self.data_points,
            self.initialized_means,
             norm=l1_norm
        )
        self.assertIsInstance(clusters, dict)

        print("test 1: k-means, Manhattan norm")
        for key in clusters:
            print(key, ': ', end='')
            print(*clusters[key], sep='\n')

    def test_k_medoids_l1(self):
        clusters = k_medoids(
            self.data_points,
            self.initialized_means,
            norm=l1_norm
        )
        self.assertIsInstance(clusters, dict)

        print("test 2: k-medoids, Manhattan norm")
        for key in clusters:
            print(key, ': ', end='')
            print(*clusters[key], sep='\n')

    def test_k_medoids_l2(self):
        clusters = k_medoids(
            self.data_points,
            self.initialized_means,
            norm=l2_norm
        )
        self.assertIsInstance(clusters, dict)

        print("test 3: k-medoids, Euclidean norm")
        for key in clusters:
            print(key, ': ', end='')
            print(*clusters[key], sep='\n')


if __name__ == '__main__':
    unittest.main()
