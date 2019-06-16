import numpy as np


def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    return np.random.rand(n, 1)


def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    A = np.random.rand(h, w)
    B = np.random.rand(h, w)
    return A, B, A+B


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    # raise NotImplementedError
    if A.shape != B.shape:
        raise ValueError("arrays must be same shape")
    return np.linalg.norm(A + B)


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a randomized weight matrix and one output.

     Arg:
     weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    if inputs.shape != (2, 1):
        raise TypeError("wrong input shape")
    if weights.shape != (2, 1):
        raise TypeError("wrong weight shape")
    return np.tanh(np.matmul(weights.transpose(), inputs))


if __name__ == "__main__":

    # randomization check
    print(randomization(10))

    # operations check
    a, b, c = operations(2, 2)
    print(*a)
    print(*b)
    print(*c)

    # norm check
    print(norm(randomization(2), randomization(2)))

    # neural network check:
    print(neural_network(randomization(2), randomization(2)))
