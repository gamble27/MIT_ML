import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# initialize some data
w = np.array([
    [1, 0, -1],
    [0, 1, -1],
    [-1, 0, -1],
    [0, -1, -1]
]) # weights for hidden layer

v = np.array([
    [1, 1, 1, 1, 0],
    [-1, -1, -1, -1, 2]
]) # weights for output layer

x = np.array([[3], [14]]) # input layer

def f(x): # ReLU activation
    x[x<0] = 0
    return x

softmax = lambda u: np.exp(u) / np.sum(np.exp(u))


# task 1
# final net output
z = np.matmul(x.transpose(), w[:, :-1].transpose()) + w[:, -1] # shape = (4, 1)
u = np.dot(f(z), v[:, :-1].transpose()) + v[:, -1] # shape = (2,1)
output = softmax(f(u)) # shape = (2,1)
print(output)
print(f(u))


# task 2
# decision boundaries visualisation for hidden layer
colours = ['red', 'green', 'blue']
plt.set_cmap(matplotlib.colors.ListedColormap(colours))

x = np.arange(-3, 3)
for i in range(w.shape[0]):
    y = -1*(w[i][-1] + x*w[i][0]) / w[i][1]
    plt.plot(x, y)
    plt.plot(y, x)

# plt.show()

# task 3
# specific net outputs :)
u_sp = np.array([[3], [-3]])
u_sp += v[:, -1].reshape((2,1))
print(f(u_sp))
