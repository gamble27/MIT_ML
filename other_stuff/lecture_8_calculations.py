import numpy as np
import matplotlib
import matplotlib.pyplot as plt


##############################################
#                   TASK 1                   #
##############################################

# data
features = np.array([
    [1, 1],
    [-1, 1],
    [1, -1],
    [-1, -1],
])

labels = np.array([
    1, -1, -1, 1
])

fs = lambda z: 2 * z - 3

w1 = np.array([
    [0, 0],
    [0, 0],
    [0, 0]
])
w2 = np.array([
    [1, 1],
    [2, 2],
    [-2, -2]
])
w3 = np.array([
    [1, 1],
    [-2, -2],
    [2, 2]
])

ws = np.array([w1, w2, w3])

# calculations
res = np.zeros((ws.shape[0], features.shape[0], features.shape[1]))
for w in range(ws.shape[0]):
    for x in range(features.shape[0]):
        for i in range(features.shape[1]):
            res[w, x, i] = fs(ws[w, 0, i] + np.dot(ws[w, i + 1, :], features[x]))

# plot
colors = ['red', 'green', 'blue']
for w in range(ws.shape[0]):
    fig = plt.figure(figsize=(8, 8))
    x = res[w, :, 0]
    y = res[w, :, 1]
    plt.scatter(x, y, c=labels+1, cmap=matplotlib.colors.ListedColormap(colors))
    plt.title("w[{}]".format(w))
    plt.show()


##############################################
#                   TASK 2                   #
##############################################

# data
# feature vectors & labels stay the same
w = np.array([
    [1, 1],
    [1, -1],
    [-1, 1]
])

f1 = lambda z: 5*z - 2
f2 = lambda z: z if z > 0 else 0
f3 = np.tanh
f4 = lambda z: z

fs = [f1, f2, f3, f4]

# calculations
res = np.zeros((len(fs), features.shape[0], w.shape[1]))
for j, f in enumerate(fs):
    for x in range(features.shape[0]):
        for i in range(w.shape[1]):
            res[j, x, i] = f(w[0, i] + np.dot(w[i+1, :], features[x, :]))

# plot
# colors stay the same
titles = [
    "f(z) = 5z - 2",
    "f(z) = ReLU(z)",
    "f(z) = tanh(z)",
    "f(z) = z"
]
for i in range(len(fs)):
    fig = plt.figure(figsize=(8, 8))
    x = res[i, :, 0]
    y = res[i, :, 1]
    plt.scatter(x, y, c=labels + 1, cmap=matplotlib.colors.ListedColormap(colors))
    plt.title(titles[i])
    plt.show()
