import numpy as np


def hinge_loss(z):
    """
    Hinge loss function
    :param z: agreement, usually T(i)x(i)+T0
    :return: float: hinge loss for given z value
    """
    if z >= 1:
        return 0
    else:
        return 1 - z


def squared_error_loss(z):
    """
    Squared error loss function
    :param z: agreement, usually T(i)x(i)+T0
    :return: float: squared error loss for given z value
    """
    return z**2 / 2


def empirical_risk(loss_function, feature_matrix, labels, theta, theta_0=0):
    """
    Empirical risk function, computes average loss
    :param loss_function: loss function used to compute av. loss,
                          takes agreement as a parameter
    :param feature_matrix: actually, array of N feature_vectors (dim=d) 
    :param labels: array of N labels for these vectors
    :param theta: vector (dim=d)
    :param theta_0: offset
    :return: float: empirical risk
    """
    if feature_matrix.shape[0] != labels.shape[0] or \
       feature_matrix.shape[1] != theta.shape[0]:
        raise TypeError("wrong shape")

    risk = 0
    for i in range(labels.shape[0]):
        risk += loss_function(
            labels[i] - (np.dot(theta, feature_matrix[i]) + theta_0)
        )
    return risk / labels.shape[0]


if __name__ == "__main__":
    x1 = np.array([1, 0, 1], np.float)
    x2 = np.array([1, 1, 1], np.float)
    x3 = np.array([1, 1, -1], np.float)
    x4 = np.array([-1, 1, 1], np.float)
    feature_vectors = np.array([x1, x2, x3, x4])

    feature_labels = np.array([2, 2.7, -0.7, 2], np.float)

    theta = np.array([0, 1, 2], np.float)

    risk1 = empirical_risk(
        hinge_loss,
        feature_vectors, feature_labels,
        theta, 0
    )
    print(risk1)

    risk2 = empirical_risk(
        squared_error_loss,
        feature_vectors, feature_labels,
        theta, 0
    )
    print(risk2)
