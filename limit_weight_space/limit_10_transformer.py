#%%
import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import datetime
import shutil
import sys

# Set plotting style
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'


# --- 1. CONFIGURATION ---
TRAIN = True  
RUN_DIR = "." 

CONFIG = {
    # "seed": time.time_ns() % (2**32 - 1),
    "seed": 2026,

    # Data & MLP Hyperparameters
    "data_samples": 2000,
    "noise_std": 0.005,
    "segments": 11,
    "x_range": [-1.5, 1.5],
    "train_seg_ids": [2, 3, 4, 5, 6, 7, 8], 
    "width_size": 48,

    # Expansion Hyperparameters
    "n_circles": 50,           
    
    # --- TRANSFORMER HYPERPARAMETERS ---
    "lr": 5e-5,      
    "transformer_epochs": 2000,
    "transformer_batch_size": 1,      
    
    # New Params
    "transformer_d_model": 128,    # Embedding Dimension
    "transformer_n_heads": 4,      # Number of Heads
    "transformer_n_layers": 2,     # Number of Transformer Blocks
    "transformer_d_ff": 256,       # Feedforward dimension inside block
    
    "transformer_target_step": 100,    # Total steps to unroll
    "scheduled_loss_weight": False,

    ## Consistency Loss Config
    "n_synthetic_points": 1000,
    "consistency_loss_weight": 0.0, 

    # Regularization Config
    "regularization_step": 75,     
    "regularization_weight": 0.0,  
    
    # Data Selection Mode
    "data_selection": "full_disk",        ## "annulus" or "full_disk"
    "final_step_mode": "none",     
}

print("Config seed is:", CONFIG["seed"])

#%%
# --- 2. UTILITY FUNCTIONS ---

def setup_run_dir(base_dir="experiments"):
    if TRAIN:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_path = Path(base_dir) / timestamp
        run_path.mkdir(parents=True, exist_ok=True)
        
        (run_path / "artefacts").mkdir(exist_ok=True)
        (run_path / "plots").mkdir(exist_ok=True)
        
        try:
            current_file = Path(__file__)
        except NameError:
            current_file = Path(sys.argv[0]) if sys.argv[0] else None
            
        if current_file and current_file.exists():
            shutil.copy(current_file, run_path / "main.py")
            
        return run_path
    else:
        return Path(RUN_DIR)

def flatten_pytree(pytree):
    leaves, tree_def = jtree.tree_flatten(pytree)
    is_array_mask = [x is not None for x in leaves]
    valid_leaves = [x for x in leaves if x is not None]
    
    if len(valid_leaves) == 0:
        return jnp.array([]), [], tree_def, is_array_mask

    flat = jnp.concatenate([x.flatten() for x in valid_leaves])
    shapes = [x.shape for x in valid_leaves]
    return flat, shapes, tree_def, is_array_mask

def unflatten_pytree(flat, shapes, tree_def, is_array_mask):
    if len(shapes) > 0:
        leaves_prod = [np.prod(x) for x in shapes]
        splits = np.cumsum(leaves_prod)[:-1]
        arrays = jnp.split(flat, splits)
        arrays = [a.reshape(s) for a, s in zip(arrays, shapes)]
    else:
        arrays = []
        
    full_leaves = []
    array_idx = 0
    for is_array in is_array_mask:
        if is_array:
            full_leaves.append(arrays[array_idx])
            array_idx += 1
        else:
            full_leaves.append(None)
            
    return jtree.tree_unflatten(tree_def, full_leaves)

#%%
# --- 3. DATA GENERATION ---

def gen_data(seed, n_samples, n_segments=3, local_structure="random", 
             x_range=[-1, 1], slope=2.0, base_intercept=0.0, 
             step_size=2.0, custom_func=None, noise_std=0.5):
    np.random.seed(seed)
    x_min, x_max = x_range
    segment_boundaries = np.linspace(x_min, x_max, n_segments + 1)
    samples_per_seg = [n_samples // n_segments + (1 if i < n_samples % n_segments else 0) 
                       for i in range(n_segments)]
    all_x, all_y, segment_ids = [], [], []
    
    for i in range(n_segments):
        seg_x_min, seg_x_max = segment_boundaries[i], segment_boundaries[i+1]
        n_seg_samples = samples_per_seg[i]
        x_seg = np.random.uniform(seg_x_min, seg_x_max, n_seg_samples)
        
        b = 0
        if local_structure == "constant": b = base_intercept
        elif local_structure == "random": b = np.random.uniform(-5, 5) 
        elif local_structure == "gradual_increase": b = base_intercept + (i * step_size)
        elif local_structure == "gradual_decrease": b = base_intercept - (i * step_size)
        
        noise = np.random.normal(0, noise_std, n_seg_samples)
        y_seg = (slope * x_seg) + b + noise
        all_x.append(x_seg)
        all_y.append(y_seg)
        segment_ids.append(np.full(n_seg_samples, i))
        
    data = np.column_stack((np.concatenate(all_x), np.concatenate(all_y)))
    return data, np.concatenate(segment_ids)

run_path = setup_run_dir()
artefacts_path = run_path / "artefacts"
plots_path = run_path / "plots"

if TRAIN:
    SEED = CONFIG["seed"]
    data, segs = gen_data(SEED, CONFIG["data_samples"], n_segments=CONFIG["segments"], 
                          local_structure="gradual_increase", x_range=CONFIG["x_range"], 
                          slope=0.5, base_intercept=-0.4, step_size=0.2, noise_std=CONFIG["noise_std"])

    train_mask = np.isin(segs, CONFIG["train_seg_ids"])
    test_mask = ~train_mask

    X_train_full = jnp.array(data[train_mask, 0])[:, None]
    Y_train_full = jnp.array(data[train_mask, 1])[:, None]
    X_test = jnp.array(data[test_mask, 0])[:, None]
    Y_test = jnp.array(data[test_mask, 1])[:, None]
    
    np.save(artefacts_path / "X_train_full.npy", X_train_full)
    np.save(artefacts_path / "Y_train_full.npy", Y_train_full)
    np.save(artefacts_path / "X_test.npy", X_test)
    np.save(artefacts_path / "Y_test.npy", Y_test)
    
else:
    print(f"Loading data from {artefacts_path}...")
    try:
        X_train_full = jnp.array(np.load(artefacts_path / "X_train_full.npy"))
        Y_train_full = jnp.array(np.load(artefacts_path / "Y_train_full.npy"))
        X_test = jnp.array(np.load(artefacts_path / "X_test.npy"))
        Y_test = jnp.array(np.load(artefacts_path / "Y_test.npy"))
    except FileNotFoundError:
        raise FileNotFoundError("Could not find data files in artefacts folder. Ensure TRAIN was run at least once.")

x_mean = jnp.mean(X_train_full)
print(f"Data Center (Mean): {x_mean:.4f}")

# Precompute masks
dists = jnp.abs(X_train_full - x_mean).flatten()
radii = jnp.linspace(0.05, jnp.max(dists) + 0.01, CONFIG["n_circles"])
circle_masks = jnp.stack([dists <= r for r in radii]) 

delta_radius = radii[1] - radii[0]
fake_radii = jnp.arange(radii[-1], radii[-1] + (CONFIG["transformer_target_step"]-CONFIG["n_circles"])*delta_radius + 0.01, delta_radius)
all_radii = jnp.concatenate([radii, fake_radii])

#%%
# --- 4. MODEL DEFINITIONS ---

width_size = CONFIG["width_size"]

class MLPModel(eqx.Module):
    layers: list
    def __init__(self, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.layers = [eqx.nn.Linear(1, width_size, key=k1), jax.nn.relu,
                       eqx.nn.Linear(width_size, width_size, key=k2), jax.nn.relu,
                       eqx.nn.Linear(width_size, width_size, key=k4), jax.nn.relu,
                       eqx.nn.Linear(width_size, 1, key=k3)]
    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x
    def predict(self, x):
        return jax.vmap(self)(x)

class PositionalEncoding(eqx.Module):
    pe: jax.Array

    def __init__(self, d_model: int, max_len: int = 500):
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / d_model))
        
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe

    def __call__(self, x):
        return x + self.pe[:x.shape[0], :]

class TransformerBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    norm2: eqx.nn.LayerNorm
    
    def __init__(self, d_model, n_heads, d_ff, dropout, key):
        k1, k2 = jax.random.split(key)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=n_heads, 
            query_size=d_model, 
            use_query_bias=True, 
            use_key_bias=True, 
            use_value_bias=True, 
            use_output_bias=True, 
            dropout_p=dropout, 
            key=k1
        )
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.mlp = eqx.nn.MLP(
            in_size=d_model, 
            out_size=d_model, 
            width_size=d_ff, 
            depth=1, 
            activation=jax.nn.gelu, 
            key=k2
        )
        self.norm2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, mask=None, key=None):
        attn_out = self.attention(x, x, x, mask=mask, key=key)
        x = eqx.filter_vmap(self.norm1)(x + attn_out)
        mlp_out = jax.vmap(self.mlp)(x)
        x = eqx.filter_vmap(self.norm2)(x + mlp_out)
        return x

# class WeightTransformer(eqx.Module):
#     embedding: eqx.nn.Linear
#     pos_encoder: PositionalEncoding
#     blocks: list
#     output_projection: eqx.nn.Linear
    
#     d_model: int = eqx.field(static=True)
#     n_layers: int = eqx.field(static=True)
    
#     def __init__(self, input_dim, d_model, n_heads, n_layers, d_ff, max_len, key):
#         self.d_model = d_model
#         self.n_layers = n_layers
        
#         k_emb, k_out = jax.random.split(key)
#         k_layers = jax.random.split(key, n_layers)
        
#         self.embedding = eqx.nn.Linear(input_dim, d_model, key=k_emb)
#         self.pos_encoder = PositionalEncoding(d_model, max_len)
        
#         self.blocks = [
#             TransformerBlock(d_model, n_heads, d_ff, dropout=0.0, key=k)
#             for k in k_layers
#         ]
        
#         self.output_projection = eqx.nn.Linear(d_model, input_dim, key=k_out)

#     def make_causal_mask(self, seq_len):
#         idx = jnp.arange(seq_len)
#         # causal mask: row i can see col j if i >= j
#         mask = idx[:, None] >= idx[None, :]
#         return mask

#     def __call__(self, x0, steps, key=None):
#         # Shape: (Steps, Input_Dim)
#         traj_buffer = jnp.zeros((steps, x0.shape[0]))
#         traj_buffer = traj_buffer.at[0].set(x0)
        
#         # Pre-compute static mask for the full sequence
#         # We always process the full buffer shape (steps, D)
#         # but the causal mask ensures we don't cheat.
#         full_mask = self.make_causal_mask(steps)
        
#         def scan_step(carry, step_idx):
#             # carry is the current full trajectory buffer of fixed shape (Steps, D)
#             current_traj = carry
            
#             # --- FIX: Do not slice the input dynamically ---
#             # Instead, embed the *full* trajectory (including the future zeros)
#             x = jax.vmap(self.embedding)(current_traj) 
#             x = self.pos_encoder(x)
            
#             # Use the static causal mask
#             # This ensures that position `step_idx` only attends to `0...step_idx`
#             # It will conceptually ignore the zeros at `step_idx+1...end`
#             for block in self.blocks:
#                 x = block(x, mask=full_mask)
                
#             # We want the output at the current step `t` to predict `t+1`
#             # JAX arrays allow dynamic indexing of single elements
#             last_hidden = x[step_idx]
            
#             # Project to get delta
#             delta = self.output_projection(last_hidden)
            
#             # Update prediction
#             current_weight = current_traj[step_idx]
#             next_weight = current_weight + delta
            
#             # Write to buffer
#             # We use minimum to prevent index out of bounds on the very last step
#             write_idx = jnp.minimum(step_idx + 1, steps - 1)
#             new_traj = current_traj.at[write_idx].set(next_weight)
            
#             return new_traj, next_weight

#         step_indices = jnp.arange(steps)
#         final_traj, _ = jax.lax.scan(scan_step, traj_buffer, step_indices)
        
#         return final_traj


class WeightTransformer(eqx.Module):
    embedding: eqx.nn.Linear
    pos_encoder: PositionalEncoding
    blocks: list
    norm_final: eqx.nn.LayerNorm  # Added final normalization
    output_projection: eqx.nn.Linear
    
    d_model: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    
    def __init__(self, input_dim, d_model, n_heads, n_layers, d_ff, max_len, key):
        self.d_model = d_model
        self.n_layers = n_layers
        
        k_emb, k_out = jax.random.split(key)
        k_layers = jax.random.split(key, n_layers)
        
        self.embedding = eqx.nn.Linear(input_dim, d_model, key=k_emb)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout=0.0, key=k)
            for k in k_layers
        ]
        
        self.norm_final = eqx.nn.LayerNorm(d_model)
        
        # --- CRITICAL FIX: Zero Initialization for Output Projection ---
        # This ensures the model initially predicts delta approx 0.
        self.output_projection = eqx.nn.Linear(d_model, input_dim, key=k_out)
        
        # Manually set weights and bias to zero
        zeros_w = jnp.zeros_like(self.output_projection.weight)
        zeros_b = jnp.zeros_like(self.output_projection.bias)
        self.output_projection = eqx.tree_at(
            lambda l: (l.weight, l.bias), 
            self.output_projection, 
            (zeros_w, zeros_b)
        )

    def make_causal_mask(self, seq_len):
        idx = jnp.arange(seq_len)
        mask = idx[:, None] >= idx[None, :]
        return mask

    def __call__(self, x0, steps, key=None):
        traj_buffer = jnp.zeros((steps, x0.shape[0]))
        traj_buffer = traj_buffer.at[0].set(x0)
        full_mask = self.make_causal_mask(steps)
        
        def scan_step(carry, step_idx):
            current_traj = carry
            
            # 1. Embedding scaled by sqrt(d_model) for stability
            x = jax.vmap(self.embedding)(current_traj) * jnp.sqrt(self.d_model)
            x = self.pos_encoder(x)
            
            # 2. Transformer Blocks
            for block in self.blocks:
                x = block(x, mask=full_mask)
            
            # 3. Final Norm (Standard in Pre-Norm architectures)
            x = jax.vmap(self.norm_final)(x)
                
            # 4. Extract last valid hidden state
            last_hidden = x[step_idx]
            
            # 5. Predict Delta
            delta = self.output_projection(last_hidden)
            
            # 6. Update
            next_weight = current_traj[step_idx] + delta
            
            write_idx = jnp.minimum(step_idx + 1, steps - 1)
            new_traj = current_traj.at[write_idx].set(next_weight)
            return new_traj, next_weight

        step_indices = jnp.arange(steps)
        final_traj, _ = jax.lax.scan(scan_step, traj_buffer, step_indices)
        return final_traj



#%%
# --- 5. INITIALIZATION & BATCH GENERATION ---

key = jax.random.PRNGKey(CONFIG["seed"])
k_init, k_tf, key = jax.random.split(key, 3)

# 1. Setup Model Structure (Static)
model_template = MLPModel(k_init)
params_template, static = eqx.partition(model_template, eqx.is_array)
flat_template, shapes, treedef, mask = flatten_pytree(params_template)
input_dim = flat_template.shape[0]

# 2. Generate Batch of Initial States
print(f"Generating {CONFIG['transformer_batch_size']} initial states...")
x0_batch_list = []
gen_key = jax.random.PRNGKey(CONFIG["seed"] + 100)

for _ in range(CONFIG["transformer_batch_size"]):
    gen_key, sk = jax.random.split(gen_key)
    m = MLPModel(sk)
    p, _ = eqx.partition(m, eqx.is_array)
    f, _, _, _ = flatten_pytree(p)
    x0_batch_list.append(f)

x0_batch = jnp.stack(x0_batch_list) 

# 3. Init Transformer
tf_model = WeightTransformer(
    input_dim=input_dim,
    d_model=CONFIG["transformer_d_model"],
    n_heads=CONFIG["transformer_n_heads"],
    n_layers=CONFIG["transformer_n_layers"],
    d_ff=CONFIG["transformer_d_ff"],
    max_len=CONFIG["transformer_target_step"],
    key=k_tf
)

opt = optax.adam(CONFIG["lr"]) 
opt_state = opt.init(eqx.filter(tf_model, eqx.is_array))

#%%
# --- 6. END-TO-END TRAINING LOOP ---

def get_functional_loss(flat_w, step_idx):
    # Unflatten MLP
    params = unflatten_pytree(flat_w, shapes, treedef, mask)
    model = eqx.combine(params, static)
    
    y_pred = model.predict(X_train_full)
    residuals = (y_pred - Y_train_full) ** 2
    
    # --- Masking Logic ---
    is_circle_phase = step_idx < CONFIG["n_circles"]
    safe_circle_idx = jnp.minimum(step_idx, CONFIG["n_circles"] - 1)
    current_circle_mask = circle_masks[safe_circle_idx]
    
    if CONFIG["data_selection"] == "annulus":
        safe_prev_idx = jnp.maximum(0, safe_circle_idx - 1)
        prev_circle_mask = circle_masks[safe_prev_idx]
        annulus_mask = jnp.logical_and(current_circle_mask, ~prev_circle_mask)
        is_step_zero = (step_idx == 0)
        phase_mask = jax.lax.select(is_step_zero, current_circle_mask, annulus_mask)
    else:
        phase_mask = current_circle_mask
        
    # Regularization
    is_reg_step = step_idx == CONFIG["regularization_step"]
    active_mask = jnp.zeros_like(current_circle_mask, dtype=bool)
    active_mask = jax.lax.select(is_circle_phase, phase_mask, active_mask)
    
    if CONFIG["final_step_mode"] == "full":
        full_mask = jnp.ones_like(current_circle_mask, dtype=bool)
        active_mask = jax.lax.select(is_reg_step, full_mask, active_mask)
        
    mask_sum = jnp.sum(active_mask)
    base_loss = jnp.sum(residuals * active_mask[:, None]) / (mask_sum + 1e-6)
    
    eff_weight = jax.lax.select(is_reg_step, CONFIG["regularization_weight"], 1.0)
    
    final_loss = base_loss * eff_weight

    if not CONFIG["scheduled_loss_weight"]:
        return final_loss
    else:
        return (base_loss * eff_weight) / (step_idx**2 + 1)

def get_disconsistency_loss(flat_w_in, flat_w_out, step_idx_in, key):
    residual = jnp.mean((flat_w_in - flat_w_out) ** 2)
    return jnp.maximum(0.0, 1.0 - residual)   ## Hinge loss style

@eqx.filter_value_and_grad
def train_step_fn(model, x0_batch, key):
    total_steps = CONFIG["transformer_target_step"]
    
    # VMAP over batch
    preds_batch = jax.vmap(model, in_axes=(0, None, None))(x0_batch, total_steps, None) # (Batch, Steps, D)

    # step_indices = jnp.arange(total_steps)
    # preds_batch_data = preds_batch

    step_indices = jnp.array([0, CONFIG["n_circles"]//2, CONFIG["n_circles"]-1])       ## TODO
    # step_indices = jax.random.choice(key, CONFIG["n_circles"], shape=(8,), replace=False)
    preds_batch_data = preds_batch[:, step_indices, :]

    def loss_per_seq(seq):
        return jax.vmap(get_functional_loss)(seq, step_indices)
    losses_batch = jax.vmap(loss_per_seq)(preds_batch_data) # (Batch, Steps)
    total_data_loss = jnp.mean(jnp.sum(losses_batch, axis=1))

    # Consistency Loss
    keys = jax.random.split(key, len(step_indices)-1)
    preds_batch_cons = preds_batch[:, step_indices, :]
    def cons_loss_per_seq(seq):
        return jax.vmap(get_disconsistency_loss)(seq[:-1], seq[1:], step_indices[:-1], keys)
    cons_losses_batch = jax.vmap(cons_loss_per_seq)(preds_batch_cons) 
    total_cons_loss = jnp.mean(jnp.sum(cons_losses_batch, axis=1))

    total_loss = total_data_loss + CONFIG["consistency_loss_weight"]*total_cons_loss
    return total_loss

@eqx.filter_jit
def make_step(model, opt_state, x0_batch, key):
    loss, grads = train_step_fn(model, x0_batch, key)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

if TRAIN:
    print(f"🚀 Starting Batch Transformer Training.")
    
    loss_history = []
    train_key = jax.random.PRNGKey(CONFIG["seed"] + 99)
    best_model = tf_model      
    lowest_loss = float('inf')

    for ep in range(CONFIG["transformer_epochs"]):
        train_key, step_key = jax.random.split(train_key)
        tf_model, opt_state, loss = make_step(tf_model, opt_state, x0_batch, step_key)
        loss_history.append(loss)

        if loss < lowest_loss:
            lowest_loss = loss
            best_model = tf_model
        
        if (ep+1) % 200 == 0:
            print(f"Epoch {ep+1} | Loss: {loss:.6f}")

    tf_model = best_model

    eval_key = jax.random.PRNGKey(42)
    final_batch_traj = jax.vmap(tf_model, in_axes=(0, None, None))(x0_batch, CONFIG["transformer_target_step"], None)
    
    np.save(artefacts_path / "final_batch_traj.npy", final_batch_traj)
    np.save(artefacts_path / "loss_history.npy", np.array(loss_history))
    eqx.tree_serialise_leaves(artefacts_path / "tf_model.eqx", tf_model)

else:
    print("Loading results...")
    final_batch_traj = np.load(artefacts_path / "final_batch_traj.npy")
    loss_history = np.load(artefacts_path / "loss_history.npy")
    tf_model = eqx.tree_deserialise_leaves(artefacts_path / "tf_model.eqx", tf_model)

final_traj = final_batch_traj[0]

#%%
# --- 7. VISUALIZATION ---
print("\n=== Generating Dashboards ===")
x_grid = jnp.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], 300)[:, None]

# --- DASHBOARD 2: FUNCTIONAL EVOLUTION ---
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(loss_history, color='teal', linewidth=2)
ax1.set_yscale('log')
ax1.set_title("Training Loss")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
traj_train_losses = []
traj_test_losses = []
for i in range(len(final_traj)):
    w = final_traj[i]
    p = unflatten_pytree(w, shapes, treedef, mask)
    m = eqx.combine(p, static)
    traj_train_losses.append(jnp.mean((m.predict(X_train_full) - Y_train_full)**2))
    traj_test_losses.append(jnp.mean((m.predict(X_test) - Y_test)**2))

ax2.plot(traj_train_losses, label="Train MSE", color='blue', alpha=0.7)
ax2.plot(traj_test_losses, label="Test MSE", color='orange', linewidth=2)
ax2.axvline(CONFIG["n_circles"], color='k', linestyle='--', label="End of Data")
ax2.axvline(CONFIG["regularization_step"], color='red', linestyle=':', label="Reg Step")
ax2.set_yscale('log')
ax2.set_title("Performance Evolution (Single Seed)")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, :])
ax3.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1)
cmap = plt.cm.coolwarm
n_steps = len(final_traj)
for i in range(0, n_steps, 1):
    w = final_traj[i]
    p = unflatten_pytree(w, shapes, treedef, mask)
    m = eqx.combine(p, static)
    label = "Limit" if i==n_steps-1 else None
    alpha = 1.0 if i==n_steps-1 else 0.1
    ax3.plot(x_grid, m.predict(x_grid), color=cmap(i/n_steps), alpha=alpha, linewidth=1.5, label=label)

ax3.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1)
w_reg = final_traj[CONFIG["regularization_step"]]
p_reg = unflatten_pytree(w_reg, shapes, treedef, mask)
ax3.plot(x_grid, eqx.combine(p_reg, static).predict(x_grid), "--", color='red', linewidth=2, label="Reg Step")
ax3.set_title("Function Evolution")
ax3.legend()

plt.tight_layout()
plt.savefig(plots_path / "dashboard_functional.png")
plt.show()

# --- DASHBOARD 1: BATCH LIMITS ---
print("Generating Batch Limits Dashboard...")
fig_batch = plt.figure(figsize=(20, 8))
gs_batch = fig_batch.add_gridspec(1, 3)

steps_to_plot = [CONFIG["n_circles"], CONFIG["regularization_step"], CONFIG["transformer_target_step"] - 1]
titles = ["End of Circles", "Regularization Step", "Final Limit"]

for i, step_idx in enumerate(steps_to_plot):
    ax = fig_batch.add_subplot(gs_batch[0, i])
    ax.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.05)
    ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1)
    
    for b in range(CONFIG["transformer_batch_size"]):
        w = final_batch_traj[b, step_idx]
        p = unflatten_pytree(w, shapes, treedef, mask)
        m = eqx.combine(p, static)
        pred = m.predict(x_grid)
        color = plt.cm.tab20(b % 20)
        ax.plot(x_grid, pred, color=color, alpha=0.6, linewidth=1.5)
        
    ax.set_title(f"{titles[i]} (Step {step_idx})")
    ax.set_ylim(jnp.min(Y_train_full)-1, jnp.max(Y_train_full)+1)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "dashboard_batch_limits.png")
plt.show()

#%% Special plot paramter trajectories
print("Generating Extended Parameter Trajectories Plot...")
fig, ax = plt.subplots(figsize=(7, 10), nrows=1, ncols=1) 
traj_seed_0 = final_batch_traj[0]
plot_ids = np.arange(100)  # First 100 parameters

# plot_up_to = CONFIG["n_circles"]
plot_up_to = CONFIG["transformer_target_step"]

for idx in plot_ids:
    ## Plot the difference x_t - x_(t-1)
    traj = traj_seed_0[:plot_up_to, idx]
    # traj_diff = jnp.concatenate([jnp.array([0.0]), traj[1:] - traj[:-1]])
    # ax.plot(np.arange(plot_up_to), traj_diff, linewidth=1.5, label=f"Param {idx}")

    ax.plot(np.arange(plot_up_to), traj, linewidth=1.5, label=f"Param {idx}")



## Plot vertical lines (One for the n_circles step, one for the regularization step
ax.axvline(CONFIG["n_circles"], color='k', linestyle='--', label="End of Data")
ax.axvline(CONFIG["regularization_step"], color='red', linestyle=':', label="Reg Step")

ax.set_title("Parameter Trajectories")
ax.set_xlabel("Training Step")
ax.set_ylabel("Parameter Value (Delta)")
plt.tight_layout()
plt.savefig(plots_path / "extended_parameter_trajectories.png")
plt.show()

#%% Model Predictions Corresponding to Circles plot
print("Generating n_circles Model Prediction Plot (Circle-Specific Models)...")
# step_idx = 35 
dists = jnp.abs(X_train_full - x_mean).flatten()

def predict_circle_specific_loss(final_batch_traj, X_data):
    circle_losses = {}
    n_points = X_data.shape[0]

    # for circle_idx in range(CONFIG["transformer_target_step"]):
    # for circle_idx in [CONFIG["n_circles"]-1, CONFIG["transformer_target_step"]-1]:
    for circle_idx in [CONFIG["transformer_target_step"]-1]:
        w = final_batch_traj[0, circle_idx]  
        p = unflatten_pytree(w, shapes, treedef, mask)
        m = eqx.combine(p, static)

        circle_masks = []
        for r in all_radii:
            new_mask = jnp.abs(X_data - x_mean) <= r
            circle_masks.append(new_mask.flatten())

        circle_mask = circle_masks[circle_idx]
        X_circle = X_data[circle_mask]

        if X_circle.shape[0] == 0:
            continue  
        y_pred = m.predict(X_circle)
        circle_losses[circle_idx] = (X_circle, y_pred)
    return circle_losses

train_preds_cc = predict_circle_specific_loss(final_batch_traj, X_train_full)
test_preds_cc = predict_circle_specific_loss(final_batch_traj, X_test)

fig, ax = plt.subplots(figsize=(10, 6))     
ax.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1, label="Train Data")
ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1, label="Test Data")

# X_circles of shape (N_circle, 1), y_pred of shape (N_circle, 1)

for circle_idx, (X_circle, y_pred) in train_preds_cc.items():
    ax.scatter(X_circle, y_pred, c='green', s=1, alpha=0.3)
for circle_idx, (X_circle, y_pred) in test_preds_cc.items():
    ax.scatter(X_circle, y_pred, c='red', s=1, alpha=0.3)

# ax.set_title(f"Model Predictions Corresponding to Circles (Circle-Specific Models)")
ax.set_title(f"Model Predictions At Final Time Step")
ax.set_ylim(jnp.min(Y_train_full)-1, jnp.max(Y_train_full)+1)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(plots_path / "model_predictions_n_circles_circle_specific.png")
plt.show()