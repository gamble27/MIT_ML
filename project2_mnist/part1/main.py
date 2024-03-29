import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from project2_mnist.utils import *
from project2_mnist.part1.linear_regression import *
from project2_mnist.part1.svm import *
from project2_mnist.part1.softmax import *
from project2_mnist.part1.features import *
from project2_mnist.part1.kernel import *

#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
# plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################

# TODO: first fill out functions in linear_regression.py, or the below functions will not work

def run_linear_regression_on_MNIST(lambda_factor=1.0):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x, test_y, theta)
    return test_error


# Don't run this until the relevant functions in linear_regression.py have been fully implemented.
# l = 1
# print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=l), "lambda =", l)
# l = 0.1
# print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=l), "lambda =", l)
# l = 0.01
# print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=l), "lambda =", l)

"""
the results are:

Linear Regression test_error = 0.744    lambda = 1
Linear Regression test_error = 0.7442   lambda = 0.1
Linear Regression test_error = 0.744    lambda = 0.01
"""


#######################################################################
# 3. Support Vector Machine
#######################################################################

# TODO: first fill out functions in svm.py, or the below functions will not work

def run_svm_one_vs_rest_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


# print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())
"""
the result is:

0.007499999999999951 (C=0.1), the very first run
0.007299999999999973 (C=0.01)
0.008499999999999952 (C=1)
"""

def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


# print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())
"""
the result is:

0.08189999999999997 - on 'ovr' strategy
0.06969999999999998 - on 'crammer_singer' strategy
"""

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################


def run_softmax_on_MNIST(temp_parameter=1.):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha= 0.3, lambda_factor = 1.0e-4, k = 10, num_iterations = 150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")

    # TODO: add your code here for the "Using the Current Model" question in tab 4.
    #      and print the test_error_mod3
    return test_error


# print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))
"""
the results are:
temperature     error
0.5             0.08399999999999996
1               0.10050000000000003
2               0.1261
"""

# TODO: Find the error rate for temp_parameter = [.5, 1.0, 2.0]
#      Remember to return the tempParameter to 1, and re-run run_softmax_on_MNIST

def run_softmax_on_MNIST_err_mod3(temp_parameter=1):
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta = read_pickle_data("./theta.pkl.gz")
    test_error = compute_test_error_mod3(test_x, test_y, theta, temp_parameter)
    return test_error


# print('softmax test_error_mod3=', run_softmax_on_MNIST_err_mod3(temp_parameter=1))
"""
the result is:

0.07679999999999998

softmax test_error= 0.10050000000000003
softmax test_error_mod3= 0.07679999999999998
"""

#######################################################################
# 6. Changing Labels
#######################################################################


def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y %= 3
    test_y %= 3

    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha= 0.3, lambda_factor = 1.0e-4, k = 10, num_iterations = 150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    return test_error

# TODO: Run run_softmax_on_MNIST_mod3(), report the error rate

# print('softmax (mod 3) test_error=', run_softmax_on_MNIST_mod3(temp_parameter=1))
"""
the result is:

softmax (mod 3) test_error= 0.18720000000000003
"""
#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##

# # TODO: First fill out the PCA functions in features.py as the below code depends on them.
# n_components = 18
# pcs = principal_components(train_x)
# train_pca = project_onto_PC(train_x, pcs, n_components)
# test_pca = project_onto_PC(test_x, pcs, n_components)
# # train_pca (and test_pca) is a representation of our training (and test) data
# # after projecting each example onto the first 18 principal components.
#
# temp_parameter = 1.0
# theta, cost_function_history = softmax_regression(train_pca, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
# plot_cost_function_over_time(cost_function_history)
# test_error = compute_test_error(test_pca, test_y, theta, temp_parameter)
# # TODO: Train your softmax regression model using (train_pca, train_y)
# #       and evaluate its accuracy on (test_pca, test_y).
# print("test error pca=", test_error)
"""
the result is:

test error pca= 0.1483
"""
#
# # TODO: Use the plot_PC function in features.py to produce scatterplot
# #       of the first 100 MNIST images, as represented in the space spanned by the
# #       first 2 principal components found above.
# plot_PC(train_x[range(100),], pcs, train_y[range(100)])
#
#
# # TODO: Use the reconstruct_PC function in features.py to show
# #       the first and second MNIST images as reconstructed solely from
# #       their 18-dimensional principal component representation.
# #       Compare the reconstructed images with the originals.
# firstimage_reconstructed = reconstruct_PC(train_pca[0, ], pcs, n_components, train_x)
# plot_images(firstimage_reconstructed)
# plot_images(train_x[0,])
#
# secondimage_reconstructed = reconstruct_PC(train_pca[1, ], pcs, n_components, train_x)
# plot_images(secondimage_reconstructed)
# plot_images(train_x[1,])



## Cubic Kernel ##
# TODO: Find the 10-dimensional PCA representation of the training and test set
train_x, train_y, test_x, test_y = get_MNIST_data()
# n_components = 10
# pcs = principal_components(train_x)
# # TODO: First fill out cubicFeatures() function in features.py as the below code requires it.
# train_cube = cubic_features(project_onto_PC(train_x, pcs, n_components))
# test_cube = cubic_features(project_onto_PC(test_x, pcs, n_components))
# # train_cube (and test_cube) is a representation of our training (and test) data
# # after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.
#
# # TODO: Train your softmax regression model using (train_cube, train_y)
# #       and evaluate its accuracy on (test_cube, test_y).
# temp_parameter = 1.0
# theta, cost_function_history = softmax_regression(train_cube, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
# plot_cost_function_over_time(cost_function_history)
# test_error = compute_test_error(test_cube, test_y, theta, temp_parameter)

n_components = 10
pcs = principal_components(train_x)
train_pca10 = cubic_features(project_onto_PC(train_x, pcs, n_components))
test_pca10 = cubic_features(project_onto_PC(test_x, pcs, n_components))
temp_parameter = 1.
theta, cost_function_history = softmax_regression(train_pca10, train_y, temp_parameter, alpha= 0.3, lambda_factor = 1.0e-4, k = 10, num_iterations = 150)
test_error = compute_test_error(test_pca10, test_y, theta, temp_parameter)

print("test error cubic=", test_error)

"""
the result is:

test error cubic= 0.7226. fuck.
                = 0.08650000000000002
"""
