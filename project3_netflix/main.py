import numpy as np
import project3_netflix.kmeans as kmeans
import project3_netflix.common as common
import project3_netflix.naive_em as naive_em
import project3_netflix.em as em


X = np.loadtxt("toy_data.txt")

"""    K=1    
0:    5462.297452340002
1:    5462.297452340002
2:    5462.297452340002
3:    5462.297452340002
4:    5462.297452340002
    K=2    
0:    1684.9079502962372
1:    1684.9079502962372
2:    1684.9079502962372
3:    1684.9079502962372
4:    1684.9079502962372
    K=3    
0:    1336.8265256618938
1:    1369.5619965862434
2:    1396.6396874064844
3:    1329.59486715443
4:    1329.59486715443
    K=4    
0:    1069.3964259219192
1:    1134.8086773780892
2:    1075.3348748905864
3:    1075.3348748905864
4:    1035.499826539466
"""

for K in range(1, 5):
    print(f"    K={K}")
    for seed in range(5):
        mixture, post = common.init(X, K, seed)

        mixture, post, cost = kmeans.run(X, mixture, post)

        print(seed, cost, sep=':    ')
