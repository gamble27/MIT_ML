import numpy as np


"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of scalar x, or the maximum between 0 and x."""
    return max(0, x)


def rectified_linear_unit_derivative(x):
    """ Returns the derivative of scalar ReLU."""
    return 1 if x > 0 else 0


def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x


def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1


class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS
        self.input_to_hidden_weights = np.array([[1, 1], [1, 1], [1, 1]])
        self.hidden_to_output_weights = np.array([[1, 1, 1]])
        self.biases = np.array([[0], [0], [0]])
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2, 1), 10), ((3, 3), 21), ((4, 5), 32), ((6, 6), 42)]
        self.testing_points = [(1, 1), (2, 2), (3, 3), (5, 5), (10, 10)]

    def train(self, x1, x2, y):

        # Vectorized functions
        ReLU   =        np.vectorize(rectified_linear_unit)
        d_ReLU =        np.vectorize(rectified_linear_unit_derivative)
        # d_active_out =  np.vectorize(output_layer_activation_derivative)

        ### Forward propagation ###
        input_values = np.array([[x1], [x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = self.input_to_hidden_weights @ input_values + self.biases
        # hidden_layer_weighted_input = np.matmul(
        #     input_values.T, self.input_to_hidden_weights.T
        # ).T + self.biases
        hidden_layer_activation = ReLU(hidden_layer_weighted_input)

        output = (self.hidden_to_output_weights @ hidden_layer_activation).item(0) # should be a scalar
        # output =  np.matmul(
        #     self.hidden_to_output_weights,
        #     hidden_layer_activation
        # ).item(0) # should be a scalar
        activated_output = output_layer_activation(output)

        ### Back propagation ###

        # Compute gradients
        # C = np.power(y - activated_output, 2)/2
        output_layer_error = activated_output - y  # (3 by 1 matrix) d L/d output ?? # C'(outp - y)^2
        hidden_layer_error = output_layer_error * output_layer_activation_derivative(output)

        hidden_to_output_weight_gradients = hidden_layer_error * hidden_layer_activation
        bias_gradients                    = (self.hidden_to_output_weights.T * d_ReLU(hidden_layer_weighted_input)) * hidden_layer_error
        input_to_hidden_weight_gradients  = bias_gradients @ input_values.T

        # Use gradients to adjust weights and biases using gradient descent
        self.hidden_to_output_weights = self.hidden_to_output_weights - self.learning_rate * hidden_to_output_weight_gradients.T
        self.biases = self.biases - self.learning_rate * bias_gradients
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate * input_to_hidden_weight_gradients

    def predict(self, x1, x2):

        input_values = np.array([[x1], [x2]])

        ReLU = np.vectorize(rectified_linear_unit)

        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input = self.input_to_hidden_weights @ input_values + self.biases
        hidden_layer_activation = ReLU(hidden_layer_weighted_input)
        output = (self.hidden_to_output_weights @ hidden_layer_activation).item(0)
        activated_output = output_layer_activation(output)

        return activated_output

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x[0], x[1], y)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return


if __name__ == "__main__":
    x = NeuralNetwork()

    x.train_neural_network()

    # UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
    x.test_neural_network()
