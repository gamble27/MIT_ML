import numpy as np
import matplotlib.pyplot as plt


rewards = np.fromfile("dqn_reward.dat")

n_samples = 150

x = np.arange(n_samples)
y = rewards[rewards.shape[0]-n_samples:]
fig, ax = plt.subplots()
ax.plot(x, y, color='b')

ax.set_title(f"avg reward {y.mean()}")

ax.minorticks_on()
ax.grid(which='major', color='k', linestyle=':')
ax.grid(which='minor', color='k', linestyle=':')

plt.show()
