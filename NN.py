#%%
import jax.numpy as np
import jax as jnp
from jax import random
from jax import grad
from jax.lax import complex
import matplotlib.pyplot as plt

random_state = 42
key = random.PRNGKey(random_state)
#%%
a = complex(1., 2.)
b = complex(2., 3.)
a * b
# %%
def f(z):
    x, y = z.real, z.imag
    return 3 * x + 1j * 4 * y

df = grad(f, holomorphic=True)
z = 1. + 4j
df(z)
# %%
import matplotlib.pyplot as plt
#%%
def make_class_data(num_classes=3, N=500):
    D = 2 # dimension
    X = np.zeros((N * num_classes, D)) # data matrix (each row = single example)
    y = np.zeros(N * num_classes, dtype='uint8') # class labels
    for j in range(num_classes):
        ix1, ix2 = N * j, N * (j + 1)
        r = np.linspace(0.0, 1, N) # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + random.normal(key, shape=(N, )) * 0.2 # theta
        X = X.at[ix1: ix2].set(np.c_[r * np.sin(t), r * np.cos(t)])
        y = y.at[ix1: ix2].set(j)
    return X, y

def plot_class_data(X, y):
    # dimension == 2
    if len(X[0]) != 2:
        return
    else:
        num_classes = len(np.unique(y))
        for j in range(num_classes):
            j_points = X[y == j]
            plt.scatter(j_points[:, 0], j_points[:, 1], label=f"x_{j}")
    plt.legend()
    plt.show()


X, y = make_class_data()
plot_class_data(X, y)

# %%

class Data(Dataset):
    
    #  modified from: http://cs231n.github.io/neural-networks-case-study/
    # Constructor
    def __init__(self, K=3, N=500):
        D = 2
        X = np.zeros((N * K, D)) # data matrix (each row = single example)
        y = np.zeros(N * K, dtype='uint8') # class labels
        for j in range(K):
          ix = range(N * j, N * (j + 1))
          r = np.linspace(0.0, 1, N) # radius
          t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2 # theta
          X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
          y[ix] = j
    
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.x = torch.from_numpy(X).type(torch.FloatTensor)
        self.len = y.shape[0]
            
    # Getter
    def __getitem__(self, index):    
        return self.x[index], self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len
    
    # Plot the diagram
    def plot_data(self):
        plt.plot(self.x[self.y[:] == 0, 0].numpy(), self.x[self.y[:] == 0, 1].numpy(), 'o', label="y=0")
        plt.plot(self.x[self.y[:] == 1, 0].numpy(), self.x[self.y[:] == 1, 1].numpy(), 'ro', label="y=1")
        plt.plot(self.x[self.y[:] == 2, 0].numpy(),self.x[self.y[:] == 2, 1].numpy(), 'go',label="y=2")
        plt.legend()