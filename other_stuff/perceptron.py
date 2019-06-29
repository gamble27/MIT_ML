import unittest
import numpy as np


# passive-aggressive perceptron algorithm
def perceptron_through_origin(x, y):
    """
    Depedencies: numpy

    Takes training set (x,y) and performs a
    classifier through origin search:
    it finds theta multiplier so that
    (theta*x) = 0

    This one iterates until we find a classifier
    which correctly classify ALL the vectors
    from the training set

    :param x: training set vectors (ndarray)
    :param y: training set labels (ndarray)

    :return:  theta
    """
    theta = np.array([0]*x.shape[1], np.float)
    iterate_flag = True  # if theta changes through the training set, we run
    counter = 0
    while iterate_flag:
        iterate_flag = False
        for i in range(x.shape[0]):
            if (y[i]*np.dot(x[i], theta)) <= 0:
                theta += y[i]*x[i]
                iterate_flag = True
                counter += 1
                print(list(theta), ',')

    print(counter)
    return theta


def perceptron_with_offset(x, y):
    """
    Depedencies: numpy

    Takes training set (x,y) and performs a
    classifier through origin search:
    it finds theta multiplier so that
    (theta*x) = 0

    This one iterates until we find a classifier
    which correctly classify ALL the vectors
    from the training set

    :param x: training set vectors (ndarray)
    :param y: training set labels (ndarray)

    :return:  theta, theta_0 (offset)
    """
    theta = np.array([0]*x.shape[1], np.float)
    theta_0 = 0
    iterate_flag = True  # if theta changes through the training set, we run
    # counter = 0
    while iterate_flag:
        iterate_flag = False
        for i in range(x.shape[0]):
            if (y[i]*(np.dot(x[i], theta) + theta_0)) <= 0:
                theta += y[i]*x[i]
                theta_0 += y[i]
                iterate_flag = True
                # counter += 1
                print(list(theta), ',', theta_0)

    # print(counter)
    return theta, theta_0


class PerceptronTestingModules(unittest.TestCase):
    def set_1_through_origin(self):
        # set 1
        x1 = np.array([-1, -1])
        x2 = np.array([1, 0])
        x3 = np.array([-1, 10])

        y1 = 1
        y2 = -1
        y3 = 1

        training_set_vectors = np.array([x2, x3, x1], np.float)
        training_set_labels = np.array([y2, y3, y1], np.float)

        theta = perceptron_through_origin(training_set_vectors,
                                          training_set_labels)

        self.assertIsNotNone(theta, "{}".format(theta))

    def set_2_with_offset(self):
        # set 2
        training_set_vectors = np.array(
            [
                [-4, 2],
                [-2, 1],
                [-1, -1],
                [2, 2],
                [1, -2]
            ],
            np.float)
        training_set_labels = np.array(
            [1, 1, -1, -1, -1],
            np.float)
        theta, theta_0 = perceptron_with_offset(training_set_vectors,
                                                training_set_labels)

        print(theta, theta_0)

        self.assertIsNotNone(theta, "{}".format(theta))
        self.assertIsNotNone(theta_0, "{}".format(theta_0))

    def set_2_through_origin(self):
        training_set_vectors = np.array(
            [
                [-4, 2],
                [-2, 1],
                [-1, -1],
                [2, 2],
                [1, -2]
            ],
            np.float)
        training_set_labels = np.array(
            [1, 1, -1, -1, -1],
            np.float)
        theta = perceptron_through_origin(training_set_vectors,
                                          training_set_labels)
        print(theta)
        self.assertIsNotNone(theta, "ok")

    def set_3_through_origin(self):
        training_set_vectors = np.array(
            [
                [1, 0],
                [0, 1]
            ],
            np.float)
        training_set_labels = np.array(
            [1, 1],
            np.float)
        theta = perceptron_through_origin(training_set_vectors,
                                          training_set_labels)
        print(theta)
        self.assertIsNotNone(theta, "ok")


if __name__ == "__main__":
    # unittest.main()
    PerceptronTestingModules().set_3_through_origin()
