import numpy as np
from project3_netflix.common import init, plot
from project3_netflix.em import run as run_em


X = np.loadtxt("test_incomplete.txt")

mixture, post = init(X, 2, 0)
mixture, post, likelihood = run_em(X, mixture, post)

plot(X, mixture, post)
