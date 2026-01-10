#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'


# 1. SIMPLE DATA AUGMENTATION
def augment(key, x):
    """
    Creates a view by adding noise and random shifting.
    Input x shape: (1, 28, 28)
    """
    k1, k2 = jax.random.split(key)
    
    # 1. Intensity jitter (noise)
    noise = jax.random.normal(k1, x.shape) * 0.5
    x = x + noise
    
    # 2. Random shift (translation invariance)
    # jnp.roll works with dynamic (traced) shifts in JAX
    shift = jax.random.randint(k2, (2,), -3, 3) 
    return jnp.roll(x, shift, axis=(1, 2))

# 2. ENCODER MODEL
class SimCLR(eqx.Module):
    layers: list
    
    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        # Use eqx.nn.Sequential for cleaner layer management,
        # or a list. If using a list, ensuring activation functions are 
        # handled correctly is key. Here we define the trainable layers explicitly.
        self.layers = [
            eqx.nn.Linear(784, 256, key=k1),
            eqx.nn.Linear(256, 128, key=k2),
            eqx.nn.Linear(128, 64, key=k3)
        ]

    def __call__(self, x):
        # Flatten input: (1, 28, 28) -> (784,)
        x = x.flatten()
        
        # Layer 1
        x = jax.nn.relu(self.layers[0](x))
        # Layer 2 (Feature Vector)
        x = jax.nn.relu(self.layers[1](x))
        # Layer 3 (Projector Head) - Linear output usually for the last step
        x = self.layers[2](x)
        
        # Normalize to unit hypersphere for Cosine Similarity
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)

# 3. UNSUPERVISED NT-XENT LOSS
def nt_xent_loss(model, x, key, temp=0.1):
    batch_size = x.shape[0]
    
    # FIX: Split keys for every image in the batch to ensure unique augmentations
    keys = jax.random.split(key, batch_size * 2)
    k1 = keys[:batch_size]
    k2 = keys[batch_size:]

    # Generate two views: vmap over (keys, images)
    # vmap signature: (k, x) -> z
    encode = jax.vmap(model)
    augment_batch = jax.vmap(augment)
    
    view1 = augment_batch(k1, x)
    view2 = augment_batch(k2, x)
    
    z1 = encode(view1)
    z2 = encode(view2)
    
    # Concatenate: z shape (2N, D)
    z = jnp.concatenate([z1, z2], axis=0)
    
    # Cosine similarity matrix (2N x 2N)
    # Since z is already normalized, dot product == cosine similarity
    sim = jnp.matmul(z, z.T) / temp
    
    # FIX: Mask out self-similarity (diagonal)
    # The model shouldn't be rewarded for matching an image with itself
    mask = jnp.eye(2 * batch_size, dtype=bool)
    sim = jnp.where(mask, -1e9, sim) # Set diagonal to -inf so Softmax ignores it
    
    # Labels:
    # Row 0 (View1_img0) should match Row N (View2_img0)
    # Row N (View2_img0) should match Row 0 (View1_img0)
    labels = jnp.arange(batch_size)
    labels = jnp.concatenate([labels + batch_size, labels])
    
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(sim, labels))

# 4. TRAINING
key = jax.random.PRNGKey(2026)
x_unlabeled = jax.random.normal(key, (200, 1, 28, 28)) # Unlabeled input
y_true = jnp.tile(jnp.arange(10), 20) # For viz only

model = SimCLR(key)
optim = optax.adam(1e-3)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def train_step(model, opt_state, x, k):
    loss, grads = eqx.filter_value_and_grad(nt_xent_loss)(model, x, k)
    updates, opt_state = optim.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, opt_state, loss

print("Training...")
losses = []
# Increased steps slightly to see better convergence
for i in range(1000):
    key, subkey = jax.random.split(key)
    model, opt_state, l = train_step(model, opt_state, x_unlabeled, subkey)
    losses.append(l)
    if i % 50 == 0:
        print(f"Step {i}, Loss: {l:.4f}")

# 5. VISUALIZATION
print("Visualizing...")
# Inference mode
embeddings = jax.vmap(model)(x_unlabeled)

# Move to NumPy for sklearn/matplotlib
embeddings_np = np.array(embeddings)
y_true_np = np.array(y_true)

pca_res = PCA(n_components=2).fit_transform(embeddings_np)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot A: Loss Curve
axes[0].plot(losses)
axes[0].set_title("Training Loss (NT-Xent)")
axes[0].set_xlabel("Steps")

# Plot B: Distance Matrix
# We compute dot product on normalized embeddings (Cosine Sim)
dist_matrix = np.dot(embeddings_np, embeddings_np.T)
axes[1].imshow(dist_matrix, cmap='magma')
axes[1].set_title("Pairwise Similarity Matrix")

# Plot C: PCA
scatter = axes[2].scatter(pca_res[:, 0], pca_res[:, 1], c=y_true_np, cmap='tab10', alpha=0.8)
axes[2].set_title("Cluster Organization (PCA)")
fig.colorbar(scatter, ax=axes[2])

plt.tight_layout()
plt.show()