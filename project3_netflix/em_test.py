import numpy as np
from project3_netflix.common import init, plot, rmse
from project3_netflix.em import run as run_em, fill_matrix


########### TASK 1 ###########

# X = np.loadtxt("netflix_incomplete.txt")
#
# Ks = (1, 12)
# for K in Ks:
#     print(f"   K={K}")
#     for seed in range(5):
#         mixture, post = init(X, K, seed)
#         mix, pst, ll = run_em(X, mixture, post)
#         print(f"seed={seed}:  LL={ll}")

"""
   K=1
seed=0:  LL=-1521060.9539852478
seed=1:  LL=-1521060.9539852478
seed=2:  LL=-1521060.9539852478
seed=3:  LL=-1521060.9539852478
seed=4:  LL=-1521060.9539852478
   K=12
seed=0:  LL=-1399803.0466569131
seed=1:  LL=-1390234.4223469393
seed=2:  LL=-1416862.4011512797
seed=3:  LL=-1393521.3929897752
seed=4:  LL=-1416733.8083763556
"""


########### TASK 1 ###########

X_incomplete = np.loadtxt('netflix_incomplete.txt')
X_gold = np.loadtxt('netflix_complete.txt')

mixture, post = init(X_incomplete, K=12, seed=1)
mixture, _, __ = run_em(X_incomplete, mixture, post)

X_pred = fill_matrix(X_incomplete, mixture)

print(rmse(X_gold, X_pred))
# 0.4804908505400694
