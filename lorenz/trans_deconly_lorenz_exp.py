#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import torch
from jax.tree_util import tree_map
from torch.utils import data
import math
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
import diffrax

# --- Configuration ---
CONFIG = {
    "seed": 2026,
    "n_epochs": 800,
    "print_every": 100,
    "batch_size": 512,
    "train_seq_len": 20,
    "train_sub_steps": 5,  # Number of future steps to sample during training loss computation
    "history_len": 15,
    "ar_train_mode": True,
    "n_refine_steps": 10,
    "d_model": 128,
    "lr": 3e-4
}

# Set seeds
key = jax.random.PRNGKey(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

sns.set(style="white", context="talk")

# --- 1. Data Loading ---
train_data = np.load('lorenz/train.npy')
test_data = np.load('lorenz/test.npy')

# Normalization
train_mean = np.mean(train_data, axis=(0,1), keepdims=True)
train_std = np.std(train_data, axis=(0,1), keepdims=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(train_data[0], label='Sample 0')
ax1.set_title("Unnormalised Time series in Lorenz dataset")
ax1.set_xlabel("Time step")
ax1.set_ylabel("Value")

train_data = (train_data - train_mean) / train_std
test_data = (test_data - train_mean) / train_std

print(f"Train data shape: {train_data.shape}")  # (num_trajectories, num_timepoints, state_dimension)
print(f"Test data shape: {test_data.shape}")

## Plot normalisied and unnormalised data on two axies
ax2.plot(train_data[0], label='Sample 0')
ax2.set_title("Normalised Time series in Lorenz dataset")
ax2.set_xlabel("Time step")
ax2.set_ylabel("Value")

#%%
# Dataset
class DynamicsDataset(data.Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float()
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        # Random start index for augmentation
        max_start = self.data.shape[1] - CONFIG["train_seq_len"]
        if max_start > 0:
            rdm_start = np.random.randint(0, max_start)
        else:
            rdm_start = 0
        return self.data[idx, rdm_start : rdm_start + CONFIG["train_seq_len"], :]

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

train_dataset = DynamicsDataset(train_data)
train_loader = data.DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=numpy_collate, num_workers=24)

#%%
# --- 2. Transformer Models (Decoder-Only) ---

class PositionalEncoding(eqx.Module):
    embedding: jax.Array
    def __init__(self, d_model: int, max_len: int = 2000):
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.embedding = pe
    def __call__(self, x, start_idx=0):
        return x + self.embedding[start_idx : start_idx + x.shape[0], :]

class TransformerBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    norm2: eqx.nn.LayerNorm
    
    def __init__(self, d_model, n_heads, d_ff, dropout, key):
        k1, k2 = jax.random.split(key)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=n_heads, query_size=d_model, dropout_p=dropout, key=k1
        )
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.mlp = eqx.nn.MLP(
            in_size=d_model, out_size=d_model, width_size=d_ff, depth=1, activation=jax.nn.gelu, key=k2
        )
        self.norm2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, mask=None, key=None):
        attn_out = self.attention(x, x, x, mask=mask, key=key)
        x = jax.vmap(self.norm1)(x + attn_out)
        mlp_out = jax.vmap(self.mlp)(x)
        x = jax.vmap(self.norm2)(x + mlp_out)
        return x

class DecoderOnlyTransformer(eqx.Module):
    embedding: eqx.nn.Linear
    pos_encoder: PositionalEncoding
    blocks: list
    output_projection: eqx.nn.Linear
    refinement_mlp: eqx.nn.MLP
    
    d_model: int = eqx.field(static=True)
    n_substeps: int = eqx.field(static=True)
    use_refinement: bool = eqx.field(static=True)

    def __init__(self, input_dim, d_model, n_heads, n_layers, d_ff, n_substeps, max_len, use_refinement, key):
        self.d_model = d_model
        self.n_substeps = n_substeps
        self.use_refinement = use_refinement
        
        keys = jax.random.split(key, 5)
        self.embedding = eqx.nn.Linear(input_dim, d_model, key=keys[0])
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout=0.0, key=k)
            for k in jax.random.split(keys[1], n_layers)
        ]
        
        self.output_projection = eqx.nn.Linear(d_model, input_dim, key=keys[2])
        # Zero Init Output
        self.output_projection = eqx.tree_at(
            lambda l: (l.weight, l.bias), 
            self.output_projection, 
            (jnp.zeros_like(self.output_projection.weight), jnp.zeros_like(self.output_projection.bias))
        )

        self.refinement_mlp = eqx.nn.MLP(
            in_size=d_model * 2, out_size=input_dim, width_size=d_ff, depth=1, activation=jax.nn.gelu, key=keys[3]
        )
        # Zero Init Refinement
        self.refinement_mlp = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias),
            self.refinement_mlp,
            (jnp.zeros_like(self.refinement_mlp.layers[-1].weight), jnp.zeros_like(self.refinement_mlp.layers[-1].bias))
        )

    def make_causal_mask(self, seq_len):
        idx = jnp.arange(seq_len)
        return idx[:, None] < idx[None, :]

    def refine_step(self, start_val, context_h):
        def loop_fn(i, curr_val):
            w_emb = self.embedding(curr_val) * jnp.sqrt(self.d_model)
            combined = jnp.concatenate([w_emb, context_h])
            delta = self.refinement_mlp(combined)
            return curr_val + delta
        return jax.lax.fori_loop(0, self.n_substeps, loop_fn, start_val)

    def __call__(self, history, future=None, steps=None, ar_mode=False, key=None):
        # Teacher Forcing Mode (Training)
        if not ar_mode:
            assert future is not None
            full_seq = jnp.concatenate([history, future], axis=0)
            input_seq = full_seq[:-1]
            
            x = jax.vmap(self.embedding)(input_seq) * jnp.sqrt(self.d_model)
            x = self.pos_encoder(x)
            mask = self.make_causal_mask(x.shape[0])
            for block in self.blocks:
                x = block(x, mask=mask, key=key)
            
            if not self.use_refinement:
                deltas = jax.vmap(self.output_projection)(x)
                predictions_next = input_seq + deltas
            else:
                predictions_next = jax.vmap(self.refine_step)(input_seq, x)
            
            # Return only future part
            return predictions_next[history.shape[0]-1:]

        # AR Mode (Inference)
        else:
            if steps is None: steps = future.shape[0] if future is not None else 10
            total_len = history.shape[0] + steps
            
            # Setup buffer
            buffer = jnp.zeros((total_len, history.shape[1]))
            buffer = buffer.at[:history.shape[0]].set(history)
            start_gen_idx = history.shape[0]
            
            def scan_fn(current_buffer, step_idx):
                x = jax.vmap(self.embedding)(current_buffer) * jnp.sqrt(self.d_model)
                x = self.pos_encoder(x)
                causal_mask = self.make_causal_mask(total_len)
                
                for block in self.blocks:
                    x = block(x, mask=causal_mask, key=key)
                
                curr_ptr = start_gen_idx + step_idx
                context_h = x[curr_ptr - 1]
                input_val = current_buffer[curr_ptr - 1]
                
                if not self.use_refinement:
                    delta = self.output_projection(context_h)
                    next_val = input_val + delta
                else:
                    next_val = self.refine_step(input_val, context_h)
                
                new_buffer = current_buffer.at[curr_ptr].set(next_val)
                return new_buffer, next_val

            indices = jnp.arange(steps)
            _, predictions = jax.lax.scan(scan_fn, buffer, indices)
            return predictions

#%%
# --- 3. Neural ODE Model ---

class NeuralODE(eqx.Module):
    func: eqx.nn.MLP
    
    def __init__(self, data_dim, hidden_dim, key):
        self.func = eqx.nn.MLP(
            in_size=data_dim, out_size=data_dim, width_size=hidden_dim, depth=2,
            activation=jax.nn.softplus, key=key
        )
    
    def __call__(self, history, future=None, steps=None, ar_mode=False, key=None):
        # Neural ODE logic differs: it doesn't take 'future' for teacher forcing usually.
        # It integrates from the last history point.
        
        # Initial condition: last point of history
        y0 = history[-1]
        
        if steps is None: steps = future.shape[0] if future is not None else 10
        
        # Integration times
        # Assuming data step size is 1.0 (arbitrary units matching index)
        ts = jnp.arange(steps + 1)
        
        def ode_func(t, y, args):
            return self.func(y)
        
        term = diffrax.ODETerm(ode_func)
        solver = diffrax.Tsit5()
        # Step size controller
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        
        sol = diffrax.diffeqsolve(
            term, solver, t0=ts[0], t1=ts[-1], dt0=0.1, y0=y0,
            saveat=diffrax.SaveAt(ts=ts[1:]), # Save at 1, 2, ... steps
            stepsize_controller=stepsize_controller
        )
        
        return sol.ys # Shape (steps, dim)

#%%
# --- 4. Training Setup & Comparison ---

# Initialize Models
key, k1, k2, k3 = jax.random.split(key, 4)

max_seq_len = max(train_data.shape[1], test_data.shape[1])
models = {
    "Neural ODE": NeuralODE(data_dim=3, hidden_dim=128, key=k1),
    "Transformer (NoRef)": DecoderOnlyTransformer(
        input_dim=3, d_model=CONFIG["d_model"], n_heads=4, n_layers=3, d_ff=128,
        n_substeps=CONFIG["n_refine_steps"], max_len=max_seq_len, use_refinement=False, key=k1
    ),
    "Transformer (Ref)": DecoderOnlyTransformer(
        input_dim=3, d_model=CONFIG["d_model"], n_heads=4, n_layers=3, d_ff=128,
        n_substeps=CONFIG["n_refine_steps"], max_len=max_seq_len, use_refinement=True, key=k1
    )
}

# Optimizers (One per model)
optimizers = {name: optax.adamw(CONFIG["lr"]) for name in models}
opt_states = {name: opt.init(eqx.filter(model, eqx.is_inexact_array)) 
              for name, (model, opt) in zip(models.keys(), zip(models.values(), optimizers.values()))}

# Train Step
@eqx.filter_jit
def train_step(model, optimizer, opt_state, batch_traj, key):
    history_len = CONFIG["history_len"]
    # Slice batch
    future_len = batch_traj.shape[1] - history_len
    batch_hist = batch_traj[:, :history_len]
    batch_fut = batch_traj[:, history_len:]
    
    def loss_fn(m, h, f, k):
        # Neural ODE vs Transformer call signature check
        if isinstance(m, NeuralODE):
            # ODE integrates from h[-1], compares to f
            # ar_mode is irrelevant for ODE in this context, it always integrates
            preds = m(h, steps=f.shape[0])
            # return jnp.mean((preds - f) ** 2)

            indices = jax.random.choice(k, future_len, shape=(CONFIG["train_sub_steps"],), replace=False)
            selected_preds = preds[indices]
            selected_futs = f[indices]
            return jnp.mean((selected_preds - selected_futs) ** 2)

        else:
            # Transformer
            preds = m(h, future=f, ar_mode=CONFIG["ar_train_mode"], key=k)
            # Subsample for stability if needed, but MSE on full sequence is standard
            # return jnp.mean((preds - f) ** 2)

            indices = jax.random.choice(k, future_len, shape=(CONFIG["train_sub_steps"],), replace=False)
            selected_preds = preds[indices]
            selected_futs = f[indices]
            return jnp.mean((selected_preds - selected_futs) ** 2)

    keys = jax.random.split(key, batch_traj.shape[0])
    loss_val, grads = eqx.filter_value_and_grad(
        lambda m: jnp.mean(jax.vmap(loss_fn, in_axes=(None, 0, 0, 0))(m, batch_hist, batch_fut, keys))
    )(model)

    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

# Training Loop
loss_history = {name: [] for name in models}

print(f"Training 3 models: {list(models.keys())}")

for epoch in range(CONFIG["n_epochs"]):
    batch_losses = {name: [] for name in models}
    
    for batch in train_loader:
        batch = jnp.array(batch)
        key, subkey = jax.random.split(key)
        
        for name in models:
            models[name], opt_states[name], loss = train_step(
                models[name], optimizers[name], opt_states[name], batch, subkey
            )
            batch_losses[name].append(loss)
            
    for name in models:
        avg = np.mean(batch_losses[name])
        loss_history[name].append(avg)
        
    if (epoch+1) % CONFIG["print_every"] == 0:
        print(f"Epoch {epoch+1}: " + ", ".join([f"{n}: {l:.5f}" for n, l in zip(loss_history.keys(), [h[-1] for h in loss_history.values()])]))

#%%
# Plot Training Curves
colors = ['r', 'b', 'g']
plt.figure(figsize=(10, 4))
for name, losses in loss_history.items():
    plt.plot(losses, label=name, color=colors.pop(0))
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.yscale('log')
# plt.title("Training Loss Comparison")
plt.legend()
plt.show()

#%%
# --- 5. Metrics & Comparison ---

def compute_metrics(model, test_data, history_len, key):
    # Standard evaluation params
    batch_size = test_data.shape[0]
    total_len = test_data.shape[1]
    future_len = total_len - history_len
    
    # Data
    hist = jnp.array(test_data[:, :history_len])
    true_fut = jnp.array(test_data[:, history_len:])
    
    # Inference
    def predict_single(h, k):
        # Ensure we use AR mode for transformers
        return model(h, steps=future_len, ar_mode=True, key=k)

    keys = jax.random.split(key, batch_size)
    pred_fut = eqx.filter_vmap(predict_single)(hist, keys)
    
    # JAX Metrics
    mse = jnp.mean((pred_fut - true_fut) ** 2)
    mae = jnp.mean(jnp.abs(pred_fut - true_fut))
    
    # CPU Metrics
    pred_np = np.array(pred_fut)
    true_np = np.array(true_fut)
    
    # Wasserstein
    w_dist = []
    for i in range(batch_size):
        # Average over dimensions
        d = np.mean([wasserstein_distance(pred_np[i, :, d], true_np[i, :, d]) for d in range(3)])
        w_dist.append(d)
        
    # Fractal Dimension
    # Note: Short trajectories (len ~50) give poor estimates. 
    # Proper FD requires ~1000 points. We compute on what we have.
    def fractal_dimension(X, max_dim=5): # Reduced max_dim for short seq
        if X.shape[0] < 10: return 0.0
        try:
            nbrs = NearestNeighbors(n_neighbors=min(len(X)-1, max_dim+1)).fit(X)
            distances, _ = nbrs.kneighbors(X)
            radii = np.logspace(-2, 0.5, 10)
            counts = []
            for r in radii:
                count = np.mean(np.sum(distances[:, 1:] < r, axis=1))
                counts.append(count)
            counts = np.array(counts)
            valid = (counts > 0) & (counts < 1.0)
            if np.sum(valid) < 3: return 0.0
            coeffs = np.polyfit(np.log(radii[valid]), np.log(counts[valid]), 1)
            return coeffs[0]
        except:
            return 0.0

    fds = [fractal_dimension(pred_np[i]) for i in range(batch_size)]
    
    return float(mse), float(mae), np.mean(w_dist), np.mean(fds)

print(f"{'Model':<25} | {'MSE':<10} | {'MAE':<10} | {'Wass.':<10} | {'FD':<10}")
print("-" * 75)

metrics_results = {}
for name, model in models.items():
    mse, mae, wass, fd = compute_metrics(model, test_data[:100], CONFIG["history_len"], key) # Test on subset for speed
    metrics_results[name] = (mse, mae, wass, fd)
    print(f"{name:<25} | {mse:.5f}    | {mae:.5f}    | {wass:.4f}     | {fd:.4f}")

#%%
# --- 6. Visual Comparison (3 Examples) ---

# Select 3 random indices without replacement
indices = np.random.choice(test_data.shape[0], 5, replace=False)
colors = ['r', 'b', 'g']

for idx in indices:
    print(f"--- Visualizing Test Sample Index: {idx} ---")
    sample = test_data[idx]
    hist = sample[:CONFIG["history_len"]]
    true_fut = sample[CONFIG["history_len"]:]
    future_len = true_fut.shape[0]

    preds = {}
    for name, model in models.items():
        # Single prediction
        # Ensure input is a JAX array
        preds[name] = model(jnp.array(hist), steps=future_len, ar_mode=True, key=key)

    # Plot 1D (First Dimension)
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(hist)), hist[:, 0], 'k-', lw=2, label="History")
    plt.plot(range(len(hist), len(sample)), true_fut[:, 0], 'k--', lw=2, alpha=0.5, label="True Future")

    for i, (name, pred) in enumerate(preds.items()):
        plt.plot(range(len(hist), len(sample)), pred[:, 0], 
                 ls='-', color=colors[i], label=name, alpha=0.8)

    plt.axvline(len(hist), color='gray', ls=':')
    plt.title(f"1D Trajectory Comparison (Dim 0) - Sample {idx}")
    plt.legend()
    plt.show()

    # Plot 3D
    fig = plt.figure(figsize=(18, 6))

    # Subplot 1: True
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    # Plot history part
    ax.plot(hist[:, 0], hist[:, 1], hist[:, 2], 'k', alpha=0.3, lw=1)
    # Plot future part
    ax.plot(true_fut[:, 0], true_fut[:, 1], true_fut[:, 2], 'k', lw=2, label="True")
    ax.set_title(f"True Trajectory {idx}")

    # Subplots for models
    for i, (name, pred) in enumerate(preds.items()):
        ax = fig.add_subplot(1, 4, i+2, projection='3d')
        # Plot history ghost
        ax.plot(hist[:, 0], hist[:, 1], hist[:, 2], 'k', alpha=0.15, lw=1)
        # Plot prediction
        ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color=colors[i], lw=2)
        # Mark start point
        ax.scatter(pred[0,0], pred[0,1], pred[0,2], color=colors[i], s=20)
        ax.set_title(name)

    plt.tight_layout()
    plt.show()
