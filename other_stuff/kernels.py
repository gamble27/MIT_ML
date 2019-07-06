import numpy as np


def radiate_basis_kernel(x1, x2):
    """
    Radiate basis kernel
    K(x1, x2) = exp(-1/2 * norm(x1-x2)^2)
    :param x1: numpy vector
    :param x2: numpy vector
    :return: float computed by kernel
    """
    if x1.shape != x2.shape:
        raise TypeError("x1, x2 should have the same dimension")

    return np.exp(-0.5 * np.linalg.norm(x1-x2)**2)


if __name__ == "__main__":
    pass
