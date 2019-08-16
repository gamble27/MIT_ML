import numpy as np


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


def prior_x(x, probs, means, vars):
    return sum([
        probs[i]*normal(x, means[i], vars[i])
        for i in range(len(probs))
    ])


def posterior_y_x(y, x, probs, means, vars):
    return probs[y]*normal(x, means[y], vars[y]) / sum([
        probs[i] * normal(x, means[i], vars[i])
        for i in range(len(probs))
    ])


if __name__ == "__main__":
    probs = [0.5, 0.5]
    means = [6, 7]
    vars = [1, 4]
    data_points = [-1, 0, 4, 5, 6]

    # print(sum([
    #     np.log(prior_x(x, probs, means, vars)) for x in data_points
    # ]))  ## -24.512532330086678

    for x in data_points:
        print(posterior_y_x(1,x,probs,means,vars) > posterior_y_x(0,x,probs,means,vars))
