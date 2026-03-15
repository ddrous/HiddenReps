#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt


import seaborn as sns
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

# ==========================================
# 1. Clear Ground Truth (1D Sine Wave)
# ==========================================
X = jnp.linspace(-3, 3, 100).reshape(-1, 1)
Y = jnp.sin(X) # The target mapping

# ==========================================
# 2. Model with STE Bottleneck
# ==========================================
class VQModel(eqx.Module):
    encoder: eqx.nn.MLP
    embedding: eqx.nn.Embedding
    decoder: eqx.nn.Linear
    num_codes: int = eqx.field(static=True)

    def __init__(self, num_codes, embed_dim, init_zero, key):
        self.num_codes = num_codes
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Encoder maps 1D input -> Logits
        self.encoder = eqx.nn.MLP(1, num_codes, width_size=128, depth=3, key=k1)
        
        # Embedding Initialization (The core test)
        if init_zero:
            init_weights = jnp.zeros((num_codes, embed_dim))
            self.embedding = eqx.nn.Embedding(num_codes, embed_dim, weight=init_weights, key=k2)
        else:
            init_weights = jax.random.normal(k2, (num_codes, embed_dim)) * 1e0
            
            # self.embedding = eqx.nn.Embedding(num_codes, embed_dim, weight=init_weights, key=k2)
            self.embedding = eqx.nn.Embedding(num_codes, embed_dim, key=k2)

            ##Scale up the random initialization to make it more distinct from zero
            self.embedding = jax.tree_map(lambda w: w * 2, self.embedding)

        # Decoder maps continuous embedding -> 1D prediction
        self.decoder = eqx.nn.Linear(embed_dim, 1, key=k3)

    def __call__(self, x):
        logits = self.encoder(x)
        
        # STE Trick
        soft_probs = jax.nn.softmax(logits, axis=-1)
        hard_idx = jnp.argmax(logits, axis=-1)
        hard_probs = jax.nn.one_hot(hard_idx, num_classes=self.num_codes)
        ste_probs = soft_probs + jax.lax.stop_gradient(hard_probs - soft_probs)
        
        # Continuous output via dot product
        z = jnp.dot(ste_probs, self.embedding.weight)
        
        return self.decoder(z), hard_idx

# ==========================================
# 3. Training Loop
# ==========================================
def train_model(init_zero, epochs=10000):
    model = VQModel(num_codes=20, embed_dim=16, init_zero=init_zero, key=jax.random.PRNGKey(42))
    optimizer = optax.adam(0.0001)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(m, x, y):
        # vmap across the batch
        preds, indices = jax.vmap(m)(x)
        mse = jnp.mean((preds - y)**2)
        return mse, indices

    @eqx.filter_jit
    def step(m, state, x, y):
        (loss, indices), grads = loss_fn(m, x, y)
        updates, state = optimizer.update(grads, state, m)
        m = eqx.apply_updates(m, updates)
        return m, state, loss, indices

    losses = []
    unique_codes_used = []
    
    for _ in range(epochs):
        model, opt_state, loss, indices = step(model, opt_state, X, Y)
        losses.append(loss)
        unique_codes_used.append(len(jnp.unique(indices)))
        
    return losses, unique_codes_used, model

# Run both experiments
print("Training Zero-Initialized Model...")
losses_zero, codes_zero, model_zero = train_model(init_zero=True)

print("Training Random-Initialized Model...")
losses_rand, codes_rand, model_rand = train_model(init_zero=False)

# ==========================================
# 4. Evaluation & Plotting
# ==========================================
# Get final predictions for the entire X domain
preds_zero, _ = jax.vmap(model_zero)(X)
preds_rand, _ = jax.vmap(model_rand)(X)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(26, 4))

# Plot 1: The Loss
ax1.plot(losses_zero, label="Zero Init", color="red")
ax1.plot(losses_rand, label="Random Init", color="green")
ax1.set_title("Training Loss")
ax1.set_xlabel("Steps")
ax1.set_yscale("log")
ax1.set_ylabel("MSE Loss")
ax1.legend()

# Plot 2: Code Utilization
ax2.plot(codes_zero, label="Zero Init", color="red")
ax2.plot(codes_rand, label="Random Init", color="green")
ax2.set_title("Unique Discrete Codes Used")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Codes (Max 10)")
ax2.legend()

# Plot 3: Function Approximation
ax3.plot(X, Y, label="Ground Truth (Sine)", color="black", linestyle="--", linewidth=2)
ax3.plot(X, preds_zero, label="Zero Init Pred", color="red", alpha=0.8, linewidth=2)
ax3.plot(X, preds_rand, label="Random Init Pred", color="green", alpha=0.8, linewidth=2)
ax3.set_title("Final Function Approximation")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.legend()

plt.tight_layout()
plt.show()













#%%

#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

# ==========================================
# 1. Clear Ground Truth (1D Sine Wave)
# ==========================================
X = jnp.linspace(-3, 3, 100).reshape(-1, 1)
Y = jnp.sin(X) # The target mapping

# ==========================================
# 2. Model with Standard VQ Bottleneck
# ==========================================
class VQModel(eqx.Module):
    encoder: eqx.nn.MLP
    embedding: eqx.nn.Embedding
    decoder: eqx.nn.Linear
    num_codes: int = eqx.field(static=True)

    def __init__(self, num_codes, embed_dim, init_zero, key):
        self.num_codes = num_codes
        k1, k2, k3 = jax.random.split(key, 3)
        
        # CHANGE: Encoder now maps 1D input -> continuous latent vector (embed_dim)
        self.encoder = eqx.nn.MLP(1, embed_dim, width_size=128, depth=3, key=k1)
        
        # Embedding Initialization 
        if init_zero:
            init_weights = jnp.zeros((num_codes, embed_dim))
            self.embedding = eqx.nn.Embedding(num_codes, embed_dim, weight=init_weights, key=k2)
        else:
            self.embedding = eqx.nn.Embedding(num_codes, embed_dim, key=k3)
            # Scale up the random initialization to make it more distinct from zero
            # self.embedding = jax.tree_map(lambda w: w * 2, self.embedding)

        # Decoder maps continuous embedding -> 1D prediction
        self.decoder = eqx.nn.Linear(embed_dim, 1, key=k3)

    def __call__(self, x):
        # 1. Get continuous latent vector
        z_e = self.encoder(x)
        
        # 2. Calculate Euclidean distance to all codebook vectors
        # z_e shape: (embed_dim,), weight shape: (num_codes, embed_dim)
        # Broadcasting automatically handles the pairwise subtraction
        dists = jnp.sum((z_e - self.embedding.weight)**2, axis=-1)
        
        # 3. Find the nearest codebook vector
        hard_idx = jnp.argmin(dists)
        z_q = self.embedding.weight[hard_idx]
        
        # 4. Straight-Through Estimator for the VQ forward pass
        # Forward pass uses z_q (discrete), backward pass routes gradients straight to z_e (continuous)
        z_q_ste = z_e + jax.lax.stop_gradient(z_q - z_e)
        
        # CHANGE: Return z_e and z_q so the loss function can pull them together
        return self.decoder(z_q_ste), hard_idx, z_e, z_q

# ==========================================
# 3. Training Loop (with VQ Losses)
# ==========================================
def train_model(init_zero, epochs=5000):
    model = VQModel(num_codes=20, embed_dim=32, init_zero=init_zero, key=jax.random.PRNGKey(42))
    optimizer = optax.adam(0.001)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(m, x, y):
        # Unpack the new VQ vectors
        preds, indices, z_e_batch, z_q_batch = jax.vmap(m)(x)
        
        # 1. Reconstruction Loss
        mse_loss = jnp.mean((preds - y)**2)
        
        # 2. Codebook Loss: Pulls embeddings toward encoder outputs
        codebook_loss = jnp.mean((jax.lax.stop_gradient(z_e_batch) - z_q_batch)**2)
        
        # 3. Commitment Loss: Stops encoder outputs from fluctuating wildly
        commitment_loss = jnp.mean((z_e_batch - jax.lax.stop_gradient(z_q_batch))**2)
        
        # Standard VQ weighting (beta = 0.25 is common)
        # total_loss = mse_loss + codebook_loss + 0.25 * commitment_loss
        total_loss = mse_loss + codebook_loss + 1.0 * commitment_loss
        # total_loss = mse_loss + 0.25 * commitment_loss

        return total_loss, indices

    @eqx.filter_jit
    def step(m, state, x, y):
        (loss, indices), grads = loss_fn(m, x, y)
        updates, state = optimizer.update(grads, state, m)
        m = eqx.apply_updates(m, updates)
        return m, state, loss, indices

    losses = []
    unique_codes_used = []
    
    for _ in range(epochs):
        model, opt_state, loss, indices = step(model, opt_state, X, Y)
        losses.append(loss)
        unique_codes_used.append(len(jnp.unique(indices)))
        
    return losses, unique_codes_used, model

# Run both experiments
print("Training Zero-Initialized Model...")
losses_zero, codes_zero, model_zero = train_model(init_zero=True)

print("Training Random-Initialized Model...")
losses_rand, codes_rand, model_rand = train_model(init_zero=False)

# ==========================================
# 4. Evaluation & Plotting
# ==========================================
# Unpack the extra return values with underscores
preds_zero, _, _, _ = jax.vmap(model_zero)(X)
preds_rand, _, _, _ = jax.vmap(model_rand)(X)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(26, 4))

# Plot 1: The Loss
ax1.plot(losses_zero, label="Zero Init", color="red")
ax1.plot(losses_rand, label="Random Init", color="green")
ax1.set_title("Training Loss")
ax1.set_xlabel("Steps")
ax1.set_yscale("log")
ax1.set_ylabel("Total Loss")
ax1.legend()

# Plot 2: Code Utilization
ax2.plot(codes_zero, label="Zero Init", color="red")
ax2.plot(codes_rand, label="Random Init", color="green")
ax2.set_title("Unique Discrete Codes Used")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Codes (Max 50)") # Updated label to match num_codes
ax2.legend()

# Plot 3: Function Approximation
ax3.plot(X, Y, label="Ground Truth (Sine)", color="black", linestyle="--", linewidth=2)
ax3.plot(X, preds_zero, label="Zero Init Pred", color="red", alpha=0.8, linewidth=2)
ax3.plot(X, preds_rand, label="Random Init Pred", color="green", alpha=0.8, linewidth=2)
ax3.set_title("Final Function Approximation")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.legend()

plt.tight_layout()
plt.show()
# %%
