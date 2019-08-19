import numpy as np
from project3_netflix.naive_em import run as run_naive_em
from project3_netflix.kmeans import run as run_k_means
from project3_netflix.common import init, plot

X = np.loadtxt("toy_data.txt")

for K in range(1, 5):
    print(f"   K={K}")
    f = []
    for seed in range(5):
        X = np.loadtxt("toy_data.txt")
        mixture, post = init(X, K, seed)
        # em
        X = np.loadtxt("toy_data.txt")
        mix_em, pst_em, log_likelihood = run_naive_em(X, mixture, post)
        f.append(log_likelihood)
        # kmeans
        # mix_km, pst_km, cost_km = run_k_means(X, mixture, post)
        # print(f"seed = {seed}: LL = {log_likelihood}")
    print(max(f))

# seed = 0
# for K in range(1, 5):
#     X = np.loadtxt("toy_data.txt")
#     mixture, post = init(X, K, seed)
#     mix_km, pst_km, cost_km = run_k_means(X, mixture, post)
#     X = np.loadtxt("toy_data.txt")
#     plot(X,mix_km, pst_km, f"km-{K}")

# seed = 0
# K = 2
# X = np.loadtxt("toy_data.txt")
# mixture, post = init(X, K, seed)
# mix,pst,ln = run_naive_em(X,mixture,post)
# plot(X,mix,pst,f"em-{K}, seed={seed}")
