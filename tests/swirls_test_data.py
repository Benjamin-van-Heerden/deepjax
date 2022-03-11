#%%
import jax.numpy as np
from jax import random
import matplotlib.pyplot as plt

random_state = 42
key = random.PRNGKey(random_state)

def make_swirl_data(num_classes=3, N=500):
    X = np.zeros((N * num_classes, 2)) # data matrix (each row = single example)
    y = np.zeros(N * num_classes, dtype='uint8') # class labels
    for j in range(num_classes):
        ix1, ix2 = N * j, N * (j + 1)
        r = np.linspace(0.0, 1, N) # radius
        t = np.linspace(j * (2 * np.pi / num_classes), (j + 1) * (2 * np.pi / num_classes), N) + random.normal(key, shape=(N, )) * (0.6 / num_classes) # theta
        X = X.at[ix1: ix2].set(np.c_[r * np.sin(t), r * np.cos(t)])
        y = y.at[ix1: ix2].set(j)
    return X, y

def plot_swirl_data(X, y):
    num_classes = len(np.unique(y))
    plt.figure(figsize=(10,8))
    for j in range(num_classes):
        j_points = X[y == j]
        plt.scatter(j_points[:, 0], j_points[:, 1], label=f"x_{j}")
    plt.legend()
    plt.show()

# X, y = make_swirl_data(num_classes=3)
# plot_swirl_data(X, y)
# %%
