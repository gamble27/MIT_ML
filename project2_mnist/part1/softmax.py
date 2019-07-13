import sys
sys.path.append("..")
# import utils
from project2_mnist.utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from random import randint

def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))


def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    # H = []
    # for i in range(X.shape[0]):
    #     arr = np.array([np.dot(theta[j], X[i]) for j in range(theta.shape[0])])
    #     # a, b = theta[0].shape, X[0].shape
    #     c = np.max(arr) / temp_parameter
    #     buff = np.array([np.exp(np.dot(theta[j], X[i])/temp_parameter-c) for j in range(theta.shape[0])])
    #     h = buff / sum(buff)
    #     H.append(h)
    # return np.array(H).transpose()

    # k, n = theta.shape[0], X.shape[0]
    # H = np.zeros((k, n))
    # dots = np.matmul(theta, np.transpose(X))
    # # transdots = dots.transpose()  # c[i] = max(transdots[i]) for i in 1,n
    # # id_vect = np.ones((k, 1))
    # for i in range(n):
    #     buff = np.exp(dots[i]/temp_parameter - np.max(dots[i]))
    #     H[i] = buff / np.sum(buff)
    #
    # return H

    # correct
    k, n = theta.shape[0], X.shape[0]
    H = np.zeros((n, k))
    for i in range(n):
        arr = np.zeros((k,))
        for j in range(k):
            arr[j] = np.dot(theta[j], X[i])
        c = np.max(arr) / temp_parameter
        buff = np.zeros((k, ))
        for j in range(k):
            buff[j] = np.exp(np.dot(theta[j], X[i])/temp_parameter-c)
        h = buff / np.sum(buff)
        H[i] = h
    return H.transpose()

    # k, n = theta.shape[0], X.shape[0]
    # dots = np.matmul(theta, np.transpose(X))
    # C = np.max(dots.transpose(), axis=1)
    # C = C.reshape(C.shape[0], 1)
    # mlt = np.ones((1, k))
    # C = np.matmul(C, mlt).transpose()
    # A = np.exp(dots-C)
    # buff = np.array(list(np.sum(A, axis=0))*k).reshape((k, n))
    # # buff = np.sum(A, axis=0)
    # H = A / buff
    # return H


def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    # this works correctly

    prob = np.log(compute_probabilities(
        X, theta, temp_parameter
    ))  # shape = (k, n)
    C = lambda_factor * np.power(np.linalg.norm(theta), 2) / 2
    for i in range(Y.shape[0]):  # n
        for j in range(theta.shape[0]):  # k
            if Y[i] == j:
                C -= prob[j][i] / Y.shape[0]
    return C

    # prob_ln = np.log(compute_probabilities(
    #     X, theta, temp_parameter
    # ))  # shape = (k, n)
    # C = lambda_factor * np.power(np.linalg.norm(theta), 2) / 2
    # n, k = X.shape[0], theta.shape[0]
    # M = sparse.coo_matrix(([1] * n, (Y, range(n))), shape=(k, n)).toarray()
    # C -= np.sum(M*prob_ln) / n
    # return C


def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    # this works 4x times slowly

    # gradient = lambda_factor * theta
    # prob = compute_probabilities(
    #     X, theta, temp_parameter
    # )
    # for j in range(theta.shape[0]):
    #     for i in range(Y.shape[0]):
    #         gradient[j] += X[i] * prob[j][i] / (Y.shape[0]*temp_parameter)
    #         if Y[i] == j:
    #             gradient[j] -= X[i] / (Y.shape[0]*temp_parameter)
    # return theta - alpha * gradient

    # this works much better!
    prob = compute_probabilities(
        X, theta, temp_parameter
    )
    n, k = X.shape[0], theta.shape[0]
    M = sparse.coo_matrix(([1] * n, (Y, range(n))), shape=(k, n)).toarray()
    gradient = lambda_factor * theta
    gradient -= (np.matmul(M, X) - np.matmul(prob, X)) / (n*temp_parameter)
    return theta - alpha * gradient


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    return train_y % 3, test_y % 3


def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    assigned_labels = get_classification(X, theta, temp_parameter)
    train_3 = assigned_labels % 3
    test_3 = Y % 3
    return 1 - np.mean(train_3 == test_3)


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression


def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)


def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()


def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
