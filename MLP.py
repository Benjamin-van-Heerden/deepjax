#%%
import jax.numpy as jnp
from jax import random
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp
import tensorflow as tf
import tensorflow_datasets as tfds
import time
#%%
data_dir = '/tmp/tfds'

mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
mnist_data = tfds.as_numpy(mnist_data)

train_data, test_data = mnist_data['train'], mnist_data['test']

num_labels = info.features['label'].num_classes
h, w, c = info.features['image'].shape
num_pixels = h * w * c

train_images, train_labels = train_data['image'],train_data['label']
test_images, test_labels = test_data['image'], test_data['label']
#%%
def one_hot(x, k, dtype=jnp.float32):
    """
    Create a one-hot encoding of x of size k.
    
    x: array
        The array to be one hot encoded
    k: interger
        The number of classes
    dtype: jnp.dtype, optional(default=float32)
        The dtype to be used on the encoding
    
    """
    return jnp.array(x[:, None] == jnp.arange(k), dtype)
#%%
random_state = 42
key = random.PRNGKey(random_state)
# %%
train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
train_labels = one_hot(train_labels, num_labels)
test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
test_labels = one_hot(test_labels, num_labels)
# %%
def get_train_batches(batch_size):
    """
    This function loads the MNIST and returns a batch of images given the batch size
    
    batch_size: integer
        The batch size, i.e, the number of images to be retrieved at each step
    
    """
    ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=data_dir)
    ds = ds.batch(batch_size).prefetch(1)
    return tfds.as_numpy(ds)
# %%
def random_layer_params(m, n, key, scale=1e-2):
    """
    This function returns two matrices, a W matrix with shape (n, m) and a b matrix with shape (n,)
    
    m: integer
        The first dimension of the W matrix
    n: integer
        The second dimension of the b matrix
    key: PRNGKey
        A Jax PRNGKey
    scale: float, optional(default=1e-2)
        The scale of the random numbers on the matrices
    """
    # Split our key into two new keys, one for each matrix
    w_key, b_key = random.split(key, num=2)
    return scale * random.normal(w_key, (m,n)), scale * random.normal(b_key, (n,))
# %%
def init_network_params(layers_sizes, key):
    """
    Given a list of weights for a neural network, initializes the weights of the network
    
    layers_sizes: list of integers
        The number of neurons on each layer of the network
    key: PRNGKey
        A Jax PRNGKey
    """
    # Generate one subkey for layer in the network
    keys = random.split(key, len(layers_sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(layers_sizes[:-1], layers_sizes[1:], keys)]
# %%
def relu(x):
    return jnp.maximum(0, x)
# %%
def predict(params, x):
    """
    Function to generate a prediction given weights and the activation
    
    params: list of matrices
        The weights for every layer of the network, including the bias
    x: matrix
        The activation, or the features, to be predicted
    """
    activations = x
    
    for w, b in params[:-1]:
        output = jnp.dot(w.T, activations) + b
        activations = relu(output)
        
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w.T, activations) + final_b
    
    return logits - logsumexp(logits)
# %%
batched_predict = vmap(predict, in_axes=(None, 0))
# %%
def accuracy(params, images, targets):
    """
    Calculates the accuracy of the neural network on a set of images

    params: list of matrices
        The weights for every layer of the network, including the bias
    images: list of matrices
        The images to be used on the calculation
    targets: list of labels
        The true labels for each of the targets

    """
    target_class = jnp.argmax(targets, axis=1)
    
    # Predicts the probabilities for each class and get the maximum
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)

    return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)
# %%
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]
# %%
layer_sizes = [784, 512, 512, 10]

# Training parameters
step_size = 0.01
num_epochs = 10
batch_size = 128

# Number of labels
n_targets = 10
params = init_network_params(layer_sizes, random.PRNGKey(0))
for epoch in range(num_epochs):
    epoch_start = time.time()
    for x, y in get_train_batches(batch_size):
        x = jnp.reshape(x, (len(x), num_pixels))
        y = one_hot(y, num_labels)
        params = update(params, x, y)
    epoch_time = time.time() - epoch_start

    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)
    print(f"Epoch {epoch + 1} in {epoch_time:0.2f} sec")
    print(f"Training set accuracy {train_acc}")
    print(f"Test set accuracy {test_acc}\n")
# %%
