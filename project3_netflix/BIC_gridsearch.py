import numpy as np
from project3_netflix.naive_em import run as run_naive_em
from project3_netflix.common import init, bic, plot

# f = []
# for K in range(1, 5):
#     print(f"   K={K}")
#     for seed in range(5):
#         X = np.loadtxt("toy_data.txt")
#         mixture, post = init(X, K, seed)
#         mix, pst, log_likelihood = run_naive_em(X, mixture, post)
#         X = np.loadtxt("toy_data.txt")
#         f.append(bic(X, mix, log_likelihood))
#         print(f"seed = {seed}:   bic = {bic(X, mix, log_likelihood)}")
#
# print(max(f))

# X = np.loadtxt("toy_data.txt")
# mixture, post = init(X, 3, 4)
# mix, pst, like = run_naive_em(X, mixture, post)
# X = np.loadtxt("toy_data.txt")
# print(bic(X, mixture, like))
#
# plot(X, mix, pst, "fck")

K = 3
f = []
for seed in range(10):
    X = np.loadtxt("toy_data.txt")
    mixture, post = init(X, K, seed)
    mix, pst, log_likelihood = run_naive_em(X, mixture, post)
    X = np.loadtxt("toy_data.txt")
    f.append(bic(X, mix, log_likelihood))
    print(f"seed = {seed}:   bic = {bic(X, mix, log_likelihood)}")

print(max(f))
