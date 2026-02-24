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

sns.set(style="white", context="talk")

# --- 1. Data Loading ---

train_data = np.load('lorenz/train.npy')
test_data = np.load('lorenz/test.npy')

# Normalization
train_mean = np.mean(train_data, axis=(0,1), keepdims=True)
train_std = np.std(train_data, axis=(0,1), keepdims=True)

train_data = (train_data - train_mean) / train_std
test_data = (test_data - train_mean) / train_std

# Dataset
class LorenzDataset(data.Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float()
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

# Hyperparameters
batch_size = 32
history_len = 50   # Length of history fed to encoder
future_len = 50    # Length of future to predict

train_dataset = LorenzDataset(train_data)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate, num_workers=0)

#%%
# --- 2. Transformer Components ---

class PositionalEncoding(eqx.Module):
    embedding: jax.Array

    def __init__(self, d_model: int, max_len: int = 500):
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.embedding = pe

    def __call__(self, x):
        return x + self.embedding[:x.shape[0], :]

class EncoderBlock(eqx.Module):
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
        # Full Self-Attention
        attn_out = self.attention(x, x, x, mask=mask, key=key)
        x = jax.vmap(self.norm1)(x + attn_out)
        
        mlp_out = jax.vmap(self.mlp)(x)
        x = jax.vmap(self.norm2)(x + mlp_out)
        return x

class DecoderBlock(eqx.Module):
    self_attn: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    cross_attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    norm3: eqx.nn.LayerNorm
    
    def __init__(self, d_model, n_heads, d_ff, dropout, key):
        k1, k2, k3 = jax.random.split(key, 3)
        # Masked Self-Attention
        self.self_attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads, query_size=d_model, dropout_p=dropout, key=k1
        )
        self.norm1 = eqx.nn.LayerNorm(d_model)
        
        # Cross-Attention (Query from Decoder, Key/Value from Encoder)
        self.cross_attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads, query_size=d_model, dropout_p=dropout, key=k2
        )
        self.norm2 = eqx.nn.LayerNorm(d_model)
        
        self.mlp = eqx.nn.MLP(
            in_size=d_model, out_size=d_model, width_size=d_ff, depth=1, activation=jax.nn.gelu, key=k3
        )
        self.norm3 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, encoder_memory, mask=None, key=None):
        # 1. Masked Self-Attention
        attn_out = self.self_attn(x, x, x, mask=mask, key=key)
        x = jax.vmap(self.norm1)(x + attn_out)
        
        # 2. Cross-Attention
        # Query = x (Decoder), Key/Value = encoder_memory (Encoder)
        cross_out = self.cross_attn(query=x, key_=encoder_memory, value=encoder_memory, key=key)
        x = jax.vmap(self.norm2)(x + cross_out)
        
        # 3. MLP
        mlp_out = jax.vmap(self.mlp)(x)
        x = jax.vmap(self.norm3)(x + mlp_out)
        return x

class EncoderDecoderTransformer(eqx.Module):
    embedding: eqx.nn.Linear
    pos_encoder: PositionalEncoding
    encoder_blocks: list
    decoder_blocks: list
    
    # Standard Projection
    output_projection: eqx.nn.Linear
    # Refinement MLP
    refinement_mlp: eqx.nn.MLP
    
    d_model: int = eqx.field(static=True)
    n_substeps: int = eqx.field(static=True)

    def __init__(self, input_dim, d_model, n_heads, n_layers, d_ff, n_substeps, max_len, key):
        self.d_model = d_model
        self.n_substeps = n_substeps
        
        keys = jax.random.split(key, 6)
        
        self.embedding = eqx.nn.Linear(input_dim, d_model, key=keys[0])
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        self.encoder_blocks = [
            EncoderBlock(d_model, n_heads, d_ff, dropout=0.0, key=k)
            for k in jax.random.split(keys[1], n_layers)
        ]
        
        self.decoder_blocks = [
            DecoderBlock(d_model, n_heads, d_ff, dropout=0.0, key=k)
            for k in jax.random.split(keys[2], n_layers)
        ]
        
        # Zero-initialized output projection (Standard Mode)
        self.output_projection = eqx.nn.Linear(d_model, input_dim, key=keys[3])
        zeros_w = jnp.zeros_like(self.output_projection.weight)
        zeros_b = jnp.zeros_like(self.output_projection.bias)
        self.output_projection = eqx.tree_at(
            lambda l: (l.weight, l.bias), self.output_projection, (zeros_w, zeros_b)
        )

        # Refinement MLP (NODE Mode)
        # Input: [State_Embedding (d_model), Context_Vector (d_model)]
        self.refinement_mlp = eqx.nn.MLP(
            in_size=d_model * 2,
            out_size=input_dim, # Delta
            width_size=d_ff,
            depth=1,
            activation=jax.nn.gelu,
            key=keys[4]
        )
        # Zero Init Refinement
        zeros_w_ref = jnp.zeros_like(self.refinement_mlp.layers[-1].weight)
        zeros_b_ref = jnp.zeros_like(self.refinement_mlp.layers[-1].bias)
        self.refinement_mlp = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias),
            self.refinement_mlp,
            (zeros_w_ref, zeros_b_ref)
        )

    def make_causal_mask(self, seq_len):
        idx = jnp.arange(seq_len)
        mask = idx[:, None] < idx[None, :] # True means masked out (future)
        return mask

    def encode(self, x, key=None):
        x = jax.vmap(self.embedding)(x) * jnp.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for block in self.encoder_blocks:
            x = block(x, key=key)
        return x

    def decode(self, x, memory, mask, key=None):
        x = jax.vmap(self.embedding)(x) * jnp.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for block in self.decoder_blocks:
            x = block(x, encoder_memory=memory, mask=mask, key=key)
        return x
    
    def refine_step(self, start_val, context_h):
        """
        Runs the refinement loop: x_t -> x_{t+1} using context h.
        start_val: (input_dim,)
        context_h: (d_model,)
        """
        def loop_fn(i, curr_val):
            # Embed current estimate
            w_emb = self.embedding(curr_val) * jnp.sqrt(self.d_model)
            # Concatenate with context
            combined = jnp.concatenate([w_emb, context_h])
            # Predict delta
            delta = self.refinement_mlp(combined)
            return curr_val + delta

        final_val = jax.lax.fori_loop(0, self.n_substeps, loop_fn, start_val)
        return final_val

    def __call__(self, history, future=None, steps=None, ar_mode=False, refinement=False, key=None):
        """
        history: (history_len, input_dim)
        future: (future_len, input_dim) - Optional (Ground Truth for Teacher Forcing)
        steps: int - Optional (Length to predict if future not provided)
        """
        
        # Infer lengths
        if future is not None:
            pred_len = future.shape[0]
        elif steps is not None:
            pred_len = steps
        else:
            raise ValueError("Must provide either 'future' or 'steps'")
            
        # 1. Run Encoder (One-Time)
        memory = self.encode(history, key=key)
        causal_mask = self.make_causal_mask(pred_len)

        # =========================================================
        # MODE A: TEACHER FORCING (Non-AR)
        # =========================================================
        if not ar_mode:
            assert future is not None, "Need ground truth 'future' for Teacher Forcing"
            
            # Prepare Inputs (Shift Right)
            # Input[0] = history[-1]
            # Input[1] = future[0] ...
            start_token = history[-1][None, :]
            decoder_input = jnp.concatenate([start_token, future[:-1]], axis=0)
            
            # Run Decoder (Parallel)
            dec_out = self.decode(decoder_input, memory, mask=causal_mask, key=key)
            
            if not refinement:
                # Standard Projection
                # We predict the delta from the input state to the next state
                deltas = jax.vmap(self.output_projection)(dec_out)
                predictions = decoder_input + deltas
                return predictions
            else:
                # Parallel Refinement
                # dec_out contains 'context_h' for every time step t
                # decoder_input contains 'start_val' for every time step t
                # We map the refinement loop over the time dimension
                
                # refine_step takes (start_val, context_h) -> next_val
                predictions = jax.vmap(self.refine_step)(decoder_input, dec_out)
                return predictions

        # =========================================================
        # MODE B: AUTOREGRESSIVE (AR)
        # =========================================================
        else:
            # Buffer initialization
            start_token = history[-1]
            
            # We maintain a buffer of the *inputs* to the decoder.
            # At step 0, only buffer[0] is valid.
            decoder_input_buffer = jnp.zeros((pred_len, history.shape[1]))
            decoder_input_buffer = decoder_input_buffer.at[0].set(start_token)
            
            def scan_fn(carry, step_idx):
                current_buffer = carry
                
                # 1. Run full decoder on current buffer (Mask handles visibility)
                # Note: For efficiency, one could use a KV-cache, but for ODE/Lorenz
                # sequence lengths (e.g. 100), re-computing is fast enough in JAX.
                dec_out_seq = self.decode(current_buffer, memory, mask=causal_mask, key=key)
                
                # Extract context for the *current* step
                current_context = dec_out_seq[step_idx]
                current_val = current_buffer[step_idx]
                
                # 2. Predict Next Value
                if not refinement:
                    delta = self.output_projection(current_context)
                    next_val = current_val + delta
                else:
                    next_val = self.refine_step(current_val, current_context)
                
                # 3. Update Buffer for *next* step (step_idx + 1)
                # We write next_val into position step_idx + 1
                write_idx = jnp.minimum(step_idx + 1, pred_len - 1)
                new_buffer = current_buffer.at[write_idx].set(next_val)
                
                # We return next_val as the output of this scan step
                # (which collects into the final prediction array)
                return new_buffer, next_val

            indices = jnp.arange(pred_len)
            _, predictions = jax.lax.scan(scan_fn, decoder_input_buffer, indices)
            
            return predictions

#%%
# --- 3. Training Setup ---

# Initialize
key = jax.random.PRNGKey(42)
model = EncoderDecoderTransformer(
    input_dim=3,
    d_model=64,
    n_heads=4,
    n_layers=3,
    d_ff=128,
    n_substeps=5, # 5 micro-steps per time-step
    max_len=200,
    key=key
)

optimizer = optax.adamw(learning_rate=3e-4)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

@eqx.filter_jit
def train_step_batch(model, batch_traj, opt_state, key):
    # Prepare batch
    batch_hist = batch_traj[:, :history_len]
    batch_fut = batch_traj[:, history_len : history_len+future_len]
    
    def loss_fn(model, h, f, k):
        # We train with Refinement ON + Teacher Forcing (Fast Parallel)
        preds = model(h, future=f, ar_mode=False, refinement=True, key=k)
        return jnp.mean((preds - f) ** 2)
    
    # Vmap the loss over the batch
    keys = jax.random.split(key, batch_traj.shape[0])
    loss_val, grads = eqx.filter_value_and_grad(
        lambda m: jnp.mean(jax.vmap(loss_fn, in_axes=(None, 0, 0, 0))(m, batch_hist, batch_fut, keys))
    )(model)
    
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

#%%
# --- 4. Training Loop ---
losses = []
print("Starting training...")

for epoch in range(50):
    batch_losses = []
    key, subkey = jax.random.split(key)
    
    for batch in train_loader:
        batch = jnp.array(batch)
        model, opt_state, loss = train_step_batch(model, batch, opt_state, subkey)
        batch_losses.append(loss)
    
    avg_loss = np.mean(batch_losses)
    losses.append(avg_loss)
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.yscale('log')
plt.title("Training Loss (Refinement Mode)")
plt.show()

#%%
# --- 5. Visualization (Inference with Refinement) ---

idx = 0
sample_traj = jnp.array(test_data[idx])
hist = sample_traj[:history_len]
true_fut = sample_traj[history_len : history_len+future_len]

# Predict using AR Mode + Refinement
# This forces the model to solve the "NODE" step-by-step
pred_fut = model(hist, steps=future_len, ar_mode=True, refinement=True, key=key)

# Plotting
fig = plt.figure(figsize=(14, 6))

# Time Series Plot
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(range(history_len), hist[:, 0], 'k-', label='History', alpha=0.5)
ax1.plot(range(history_len, history_len+future_len), true_fut[:, 0], 'g-', label='True Future')
ax1.plot(range(history_len, history_len+future_len), pred_fut[:, 0], 'r--', label='Predicted (AR+Refine)', linewidth=2)
ax1.axvline(history_len, color='k', linestyle=':', alpha=0.3)
ax1.set_title("Time Series (Dim 0)")
ax1.legend()

# 3D Phase Space Plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot(sample_traj[:, 0], sample_traj[:, 1], sample_traj[:, 2], 'g', alpha=0.3, label="True Traj")
ax2.plot(pred_fut[:, 0], pred_fut[:, 1], pred_fut[:, 2], 'r--', linewidth=2, label="Predicted")
ax2.set_title("3D Phase Space")
ax2.legend()

plt.tight_layout()
plt.show()