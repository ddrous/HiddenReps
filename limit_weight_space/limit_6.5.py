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
import json
import datetime
import shutil
import sys
import os
from typing import List, Tuple, Optional, Any

# Set plotting style
sns.set(style="white", context="talk")

# --- 1. CONFIGURATION ---
TRAIN = True  
RUN_DIR = "." 

CONFIG = {
    "seed": time.time_ns() % (2**32 - 1),

    # Data & MLP Hyperparameters
    "data_samples": 2000,
    "noise_std": 0.005,
    "segments": 11,
    "x_range": [-1.5, 1.5],
    "train_seg_ids": [2, 3, 4, 5, 6, 7, 8], 
    "width_size": 48,

    # Expansion Hyperparameters
    "n_circles": 10,           
    
    # Training Config
    "lr": 5e-4,      
    "batch_size": 16,          # Number of parallel GRU initializations
    "gru_hidden_size": 128,    
    "gru_epochs": 500,  
    
    "gru_target_step": 100,    # Total steps to unroll
    
    # Regularization Config
    "regularization_step": 30,     
    "regularization_weight": 0.0,  
    
    # Data Selection Mode
    "data_selection": "annulus",   
    "final_step_mode": "full",     
    
    # Probabilistic Output
    "predict_std": False,           # Enable Mean + Std prediction
}

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

# --- DATA LOADING / GENERATION LOGIC ---
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

# Precompute masks for circles
dists = jnp.abs(X_train_full - x_mean).flatten()
radii = jnp.linspace(0.05, jnp.max(dists) + 0.01, CONFIG["n_circles"])
circle_masks = jnp.stack([dists <= r for r in radii]) 

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

class WeightGRU(eqx.Module):
    cell: eqx.nn.GRUCell
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear
    predict_std: bool
    
    def __init__(self, key, input_dim, hidden_dim, predict_std=False):
        k1, k2, k3 = jax.random.split(key, 3)
        self.predict_std = predict_std
        
        self.encoder = eqx.nn.Linear(input_dim, hidden_dim, key=k1)
        self.cell = eqx.nn.GRUCell(hidden_dim, hidden_dim, key=k2)
        
        # If predict_std, output dimension is 2x (Mean + LogStd)
        out_dim = input_dim * 2 if predict_std else input_dim
        self.decoder = eqx.nn.Linear(hidden_dim, out_dim, key=k3)

    def __call__(self, x0, steps, key=None):
        h0 = jnp.zeros((self.cell.hidden_size,))
        
        # We need random keys for sampling if predict_std is True
        if self.predict_std:
            if key is None:
                raise ValueError("Must provide key when predict_std=True")
            xs = jax.random.split(key, steps)
        else:
            xs = None
        
        def scan_step(carry, inputs):
            h, x_prev = carry
            key_step = inputs # Will be None if xs is None
            
            # Autoregressive input
            current_input = x_prev

            embedded = self.encoder(current_input)
            h_new = self.cell(embedded, h)
            out = self.decoder(h_new)
            
            if self.predict_std:
                # Split output into mean and log_std
                d = out.shape[0] // 2
                mu, log_std = out[:d], out[d:]
                sigma = jnp.exp(log_std)
                
                # Reparameterization Trick: z = mu + sigma * eps
                eps = jax.random.normal(key_step, mu.shape)
                x_sample = mu + sigma * eps
                
                # We return the sample as the next input, but can output tuple (sample, mu, sigma)
                # For functional loss, we usually care about the sample (what the model "chose")
                # or we could return just mu if we want to evaluate the mean path.
                # Let's return the sample to propagate uncertainty.
                
                next_val = x_sample
                output = (x_sample, mu, sigma)
            else:
                x_next = out
                next_val = x_next
                output = (x_next, x_next, jnp.zeros_like(x_next)) # Consistent structure

            return (h_new, next_val), output

        init_carry = (h0, x0)
        _, (samples, means, sigmas) = jax.lax.scan(scan_step, init_carry, xs, length=steps)
        
        return samples, means, sigmas

#%%
# --- 5. INITIALIZATION & BATCH GENERATION ---

key = jax.random.PRNGKey(CONFIG["seed"])
k_init, k_gru, key = jax.random.split(key, 3)

# 1. Setup Model Structure (Static)
model_template = MLPModel(k_init)
params_template, static = eqx.partition(model_template, eqx.is_array)
flat_template, shapes, treedef, mask = flatten_pytree(params_template)
input_dim = flat_template.shape[0]

# 2. Generate Batch of Initial States (Different Seeds)
print(f"Generating {CONFIG['batch_size']} initial states...")
x0_batch_list = []
gen_key = jax.random.PRNGKey(CONFIG["seed"] + 100)

for _ in range(CONFIG["batch_size"]):
    gen_key, sk = jax.random.split(gen_key)
    m = MLPModel(sk)
    p, _ = eqx.partition(m, eqx.is_array)
    f, _, _, _ = flatten_pytree(p)
    x0_batch_list.append(f)

x0_batch = jnp.stack(x0_batch_list) 

# 3. Init GRU
gru_model = WeightGRU(
    k_gru, input_dim, CONFIG["gru_hidden_size"], 
    predict_std=CONFIG["predict_std"]
)
opt = optax.adam(CONFIG["lr"]) 
opt_state = opt.init(eqx.filter(gru_model, eqx.is_array))

#%%
# --- 6. END-TO-END TRAINING LOOP (BATCHED) ---

def get_functional_loss(flat_w, step_idx):
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
    
    return base_loss * eff_weight

@eqx.filter_value_and_grad
def train_step_fn(gru, x0_batch, key):
    total_steps = CONFIG["gru_target_step"]
    
    # VMAP GRU over batch
    # We need a key per batch element if predict_std is True
    keys_batch = jax.random.split(key, x0_batch.shape[0]) if CONFIG["predict_std"] else None
    
    # Run GRU (Output is tuple: (samples, means, sigmas))
    # We use samples for functional loss to train robustly against the noise
    samples_batch, means_batch, sigmas_batch = jax.vmap(gru, in_axes=(0, None, 0))(x0_batch, total_steps, keys_batch)
    
    # predictions_batch = samples_batch 
    
    step_indices = jnp.arange(total_steps)
    
    def loss_per_seq(seq):
        return jax.vmap(get_functional_loss)(seq, step_indices)
        
    losses_batch = jax.vmap(loss_per_seq)(samples_batch) 
    total_loss = jnp.mean(jnp.sum(losses_batch, axis=1))
    
    return total_loss

@eqx.filter_jit
def make_step(gru, opt_state, x0_batch, key):
    loss, grads = train_step_fn(gru, x0_batch, key)
    updates, opt_state = opt.update(grads, opt_state, gru)
    gru = eqx.apply_updates(gru, updates)
    return gru, opt_state, loss

if TRAIN:
    print(f"🚀 Starting Batch GRU Training. Batch Size: {CONFIG['batch_size']}")
    print(f"Probabilistic: {CONFIG['predict_std']}")
    print(f"Reg Weight: {CONFIG['regularization_weight']} @ Step {CONFIG['regularization_step']}")
    
    loss_history = []
    train_key = jax.random.PRNGKey(CONFIG["seed"] + 99)

    for ep in range(CONFIG["gru_epochs"]):
        train_key, step_key = jax.random.split(train_key)
        gru_model, opt_state, loss = make_step(gru_model, opt_state, x0_batch, step_key)
        loss_history.append(loss)
        
        if (ep+1) % 100 == 0:
            print(f"Epoch {ep+1} | Loss: {loss:.6f}")

    # Generate final trajectories (Use Means for deterministic visualization if desired)
    # But for "Plausible Futures" we might want samples.
    # Let's save MEANS for the main analysis to see the "center" of the learned path.
    eval_key = jax.random.PRNGKey(42)
    # Just use None keys for vmap since we want means and we can get them deterministically?
    # Actually no, the call structure requires keys if predict_std is True even if we just want means output.
    keys_eval = jax.random.split(eval_key, x0_batch.shape[0])
    _, final_means, _ = jax.vmap(gru_model, in_axes=(0, None, 0))(x0_batch, CONFIG["gru_target_step"], keys_eval)
    
    final_batch_traj = final_means  # Shape: (batch_size, steps, param_dim)
    np.save(artefacts_path / "final_batch_traj.npy", final_means)
    np.save(artefacts_path / "loss_history.npy", np.array(loss_history))

else:
    print("Loading results...")
    final_batch_traj = np.load(artefacts_path / "final_batch_traj.npy")
    loss_history = np.load(artefacts_path / "loss_history.npy")

# Use the FIRST trajectory mean for standard single-model dashboards
final_traj = final_batch_traj[0]

#%%
# --- 7. VISUALIZATION ---
print("\n=== Generating Dashboards ===")

# --- DASHBOARD 1: BATCH LIMITS (NEW) ---
print("Generating Batch Limits Dashboard...")
fig_batch = plt.figure(figsize=(20, 8))
gs_batch = fig_batch.add_gridspec(1, 3)

steps_to_plot = [CONFIG["n_circles"], CONFIG["regularization_step"], CONFIG["gru_target_step"] - 1]
titles = ["End of Circles", "Regularization Step", "Final Limit"]
x_grid = jnp.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], 300)[:, None]

for i, step_idx in enumerate(steps_to_plot):
    ax = fig_batch.add_subplot(gs_batch[0, i])
    ax.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.05)
    ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1)
    
    for b in range(CONFIG["batch_size"]):
        w = final_batch_traj[b, step_idx] # Using Mean Trajectory
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

# --- DASHBOARD 2: FUNCTIONAL EVOLUTION (Single Representative) ---
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
ax2.set_title("Performance Evolution (Single Seed Mean)")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, :])
ax3.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1)
cmap = plt.cm.viridis
n_steps = len(final_traj)
for i in range(0, n_steps, 2):
    w = final_traj[i]
    p = unflatten_pytree(w, shapes, treedef, mask)
    m = eqx.combine(p, static)
    ax3.plot(x_grid, m.predict(x_grid), color=cmap(i/n_steps), alpha=0.2)

w_reg = final_traj[CONFIG["regularization_step"]]
p_reg = unflatten_pytree(w_reg, shapes, treedef, mask)
ax3.plot(x_grid, eqx.combine(p_reg, static).predict(x_grid), color='red', linewidth=2, label="Reg Step")
ax3.set_title("Function Evolution")
ax3.legend()

plt.tight_layout()
plt.savefig(plots_path / "dashboard_functional.png")
plt.show()

# --- 8. ADVANCED DIAGNOSTICS ---
print("\n=== Generating Advanced Diagnostics ===")
fig_diag = plt.figure(figsize=(20, 16))
gs_diag = fig_diag.add_gridspec(2, 2)

# PLOT A: BATCH PARAMETER ENVELOPES (Top Left)
ax_t1 = fig_diag.add_subplot(gs_diag[0, 0])
np.random.seed(CONFIG["seed"])
param_indices = np.random.choice(input_dim, 5, replace=False)
distinct_colors = ['crimson', 'dodgerblue', 'forestgreen', 'darkorange', 'purple']

# Calculate stats across the batch dimension
batch_mean = jnp.mean(final_batch_traj, axis=0)
batch_min = jnp.min(final_batch_traj, axis=0)
batch_max = jnp.max(final_batch_traj, axis=0)

for i, idx in enumerate(param_indices):
    color = distinct_colors[i % len(distinct_colors)]
    ax_t1.plot(batch_mean[:, idx], color=color, linewidth=2, label=f"Param {idx}")
    ax_t1.fill_between(
        range(CONFIG["gru_target_step"]), 
        batch_min[:, idx], 
        batch_max[:, idx], 
        color=color, 
        alpha=0.15
    )

ax_t1.axvline(CONFIG["n_circles"], color='k', linestyle='--')
ax_t1.axvline(CONFIG["regularization_step"], color='red', linestyle=':')
ax_t1.set_title(f"Parameter Envelopes (Batch Size {CONFIG['batch_size']})")
ax_t1.legend(loc='upper left', fontsize='small')

# PLOT B: STABILITY (Jacobian at Reg Step of First Seed Mean)
ax_e = fig_diag.add_subplot(gs_diag[0, 1])
def get_gru_step_fn(gru):
    def step_fn(joint_state):
        w, h = joint_state
        embedded = gru.encoder(w)
        h_new = gru.cell(embedded, h)
        out = gru.decoder(h_new)
        # Handle split if probabilistic
        if CONFIG["predict_std"]:
            d = out.shape[0] // 2
            mu = out[:d]
            return (mu, h_new) # Deterministic mean path
        return (out, h_new)
    return step_fn

print("Recovering hidden state for stability analysis...")
h0 = jnp.zeros((gru_model.cell.hidden_size,))
x_start = x0_batch[0] 

def diag_scan_step(carry, _):
    h, x_prev = carry
    embedded = gru_model.encoder(x_prev)
    h_new = gru_model.cell(embedded, h)
    out = gru_model.decoder(h_new)
    if CONFIG["predict_std"]:
        d = out.shape[0] // 2
        x_next = out[:d] # Mean
    else:
        x_next = out
    return (h_new, x_next), None

(h_target, w_target), _ = jax.lax.scan(
    diag_scan_step, (h0, x_start), None, length=CONFIG["regularization_step"]
)

step_fn = get_gru_step_fn(gru_model)
try:
    J_tuple = jax.jacfwd(step_fn)((w_target, h_target))
    J_ww = J_tuple[0][0]
    J_wh = J_tuple[0][1]
    J_hw = J_tuple[1][0]
    J_hh = J_tuple[1][1]
    top = jnp.concatenate([J_ww, J_wh], axis=1)
    bot = jnp.concatenate([J_hw, J_hh], axis=1)
    Full_J = jnp.concatenate([top, bot], axis=0)
    eigenvals = jnp.linalg.eigvals(Full_J)
    
    unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
    ax_e.add_patch(unit_circle)
    ax_e.scatter(jnp.real(eigenvals), jnp.imag(eigenvals), alpha=0.6, s=30, color='purple')
    ax_e.set_title("Stability Analysis")
    ax_e.set_aspect('equal')
    lim = max(1.1, float(jnp.max(jnp.abs(eigenvals))) + 0.1)
    ax_e.set_xlim(-lim, lim); ax_e.set_ylim(-lim, lim)
except Exception as e:
    print(f"Jacobian failed: {e}")
    ax_e.text(0.5, 0.5, "Jacobian Failed", ha='center')

# PLOT C: 3D PCA (First Seed Mean Trajectory)
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3)
traj_3d = pca.fit_transform(np.array(final_traj))
ax_3d = fig_diag.add_subplot(gs_diag[1, 0], projection='3d')
ax_3d.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], color='gray', alpha=0.5)
ax_3d.scatter(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], c=range(len(traj_3d)), cmap='viridis', s=40)
ax_3d.set_title("3D PCA Trajectory")

# PLOT D: DATA EXPANSION
ax_circles = fig_diag.add_subplot(gs_diag[1, 1])
point_indices = jnp.sum(~circle_masks, axis=0)
cmap_hard = plt.cm.get_cmap('tab20', CONFIG["n_circles"] + 1)
scatter = ax_circles.scatter(X_train_full, Y_train_full, c=point_indices, cmap=cmap_hard, s=15, alpha=0.9)
for i in range(0, CONFIG["n_circles"], 5):
    r = radii[i]
    ax_circles.axvline(x_mean - r, color='k', linestyle='-', alpha=0.2)
    ax_circles.axvline(x_mean + r, color='k', linestyle='-', alpha=0.2)
plt.colorbar(scatter, ax=ax_circles, ticks=np.arange(0, CONFIG["n_circles"]+1, 5))
ax_circles.set_title("Data Expansion")

plt.tight_layout()
plt.savefig(plots_path / "dashboard_advanced_diagnostics.png")
plt.show()
# %%
