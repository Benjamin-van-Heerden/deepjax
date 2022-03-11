#%%
import jax.numpy as jnp
from jax import random
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp


#%%
random_state = 42
key = random.PRNGKey(random_state)






# %%
import tensorflow
# %%
