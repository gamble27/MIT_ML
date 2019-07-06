import numpy as np
import sympy as sp

v = np.array([4, 2, 1])

y = np.array([[5, 0, 7],
              [0, 2, 0],
              [4, 0, 0],
              [0, 3, 6]])

_lambda = sp.symbols('_lambda')
u = sp.symbols('u')

# calculate u
for i in range(y.shape[0]):
    phrase = ""
    for j in range(y.shape[1]):
        if y[i][j] > 0:
            phrase += "-({}-{}*u)".format(
                y[i][j]*v[j],
                v[j]**2
            )
    phrase += "+u*_lambda"

    print(sp.solveset(eval(phrase), u))