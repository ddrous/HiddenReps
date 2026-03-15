#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt

# ==========================================
# 1. Clear Ground Truth (1D Sine Wave)
# ==========================================
X = jnp.linspace(-3, 3, 300).reshape(-1, 1)
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
        self.encoder = eqx.nn.MLP(1, num_codes, width_size=128, depth=1, key=k1)
        
        # Embedding Initialization (The core test)
        if init_zero:
            init_weights = jnp.zeros((num_codes, embed_dim))
        else:
            init_weights = jax.random.normal(k2, (num_codes, embed_dim)) * 0.1
            
        self.embedding = eqx.nn.Embedding(num_codes, embed_dim, weight=init_weights, key=k2)
        
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
def train_model(init_zero, epochs=1000):
    model = VQModel(num_codes=100, embed_dim=1, init_zero=init_zero, key=jax.random.PRNGKey(42))
    optimizer = optax.adam(0.001)
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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

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