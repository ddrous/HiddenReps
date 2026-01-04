#%%
import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", context="talk")
from pathlib import Path
import time
import json
import datetime
import shutil
import sys
import os
from typing import List, Tuple, Optional, Any

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
    "n_circles": 30,           
    
    # Learning Rates
    "lr": 5e-4,      
    "batch_size": 600,
    
    # GRU Training Config
    "gru_hidden_size": 128,    
    "gru_epochs": 2000,  
    "gru_target_step": 100,        # Total steps to unroll
    
    "loss_type": "mse",            # 'mse' or 'nll'
    "teacher_forcing": False,
    "final_step_mode": "none",     # 'full' (regularize limit) or 'none' (free drift)
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
    loss_type: str
    
    def __init__(self, key, input_dim, hidden_dim, loss_type="mse"):
        k1, k2, k3 = jax.random.split(key, 3)
        self.loss_type = loss_type
        
        self.encoder = eqx.nn.Linear(input_dim, hidden_dim, key=k1)
        self.cell = eqx.nn.GRUCell(hidden_dim, hidden_dim, key=k2)
        
        out_dim = input_dim * 2 if loss_type == "nll" else input_dim
        self.decoder = eqx.nn.Linear(hidden_dim, out_dim, key=k3)

    def __call__(self, x0, steps, key=None):
        h0 = jnp.zeros((self.cell.hidden_size,))
        
        if self.loss_type == "nll" and key is None:
            raise ValueError("Must provide key for NLL sampling")
            
        keys = jax.random.split(key, steps) if key is not None else [None] * steps
        
        def scan_step(carry, inputs):
            h, x_prev = carry
            key_step = inputs
            
            # Autoregressive input (always previous prediction)
            current_input = x_prev

            embedded = self.encoder(current_input)
            h_new = self.cell(embedded, h)
            out = self.decoder(h_new)
            
            if self.loss_type == "nll":
                d = out.shape[0] // 2
                mu, log_std = out[:d], out[d:]
                sigma = jnp.exp(log_std)
                
                eps = jax.random.normal(key_step, mu.shape)
                x_next = mu + sigma * eps
                output = (x_next, mu, sigma)
                next_val = x_next
            else:
                x_next = out
                output = (x_next, x_next, jnp.zeros_like(x_next))
                next_val = x_next
                
            return (h_new, next_val), output

        init_carry = (h0, x0)
        _, (samples, means, sigmas) = jax.lax.scan(scan_step, init_carry, jnp.array(keys))
        
        if self.loss_type == "nll":
            return samples, means, sigmas
        else:
            return samples

#%%
# --- 5. INITIALIZATION ---

key = jax.random.PRNGKey(CONFIG["seed"])
k_init, k_gru, key = jax.random.split(key, 3)

# Initialize MLP to get shapes and initial state (Circle 0)
model_init = MLPModel(k_init)
params_init, static = eqx.partition(model_init, eqx.is_array)
flat_init, shapes, treedef, mask = flatten_pytree(params_init)

input_dim = flat_init.shape[0]

gru_model = WeightGRU(
    k_gru, input_dim, CONFIG["gru_hidden_size"], 
    loss_type=CONFIG["loss_type"]
)

opt = optax.adam(CONFIG["lr"]) 
opt_state = opt.init(eqx.filter(gru_model, eqx.is_array))

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
    
    # 1. Circle Phase: Use radius masks
    safe_circle_idx = jnp.minimum(step_idx, CONFIG["n_circles"] - 1)
    circle_mask = circle_masks[safe_circle_idx]
    
    # 2. Final Step: Check if we are at target_step - 1
    is_final_step = step_idx == (CONFIG["gru_target_step"] - 1)
    
    # 3. Construct Active Mask
    # Start with all Zeros (Default for intermediate extrapolation steps)
    active_mask = jnp.zeros_like(circle_mask, dtype=bool)
    
    # Apply circle mask if in phase
    active_mask = jax.lax.select(is_circle_phase, circle_mask, active_mask)
    
    # Apply final mask if at final step AND mode is 'full'
    if CONFIG["final_step_mode"] == "full":
        full_mask = jnp.ones_like(circle_mask, dtype=bool)
        active_mask = jax.lax.select(is_final_step, full_mask, active_mask)
        
    # Apply Mask
    mask_sum = jnp.sum(active_mask)
    # If mask is empty, loss is 0.0 (masked by 0s in numerator)
    masked_loss = jnp.sum(residuals * active_mask[:, None]) / (mask_sum + 1e-6)
    
    return masked_loss

@eqx.filter_value_and_grad
def train_step_fn(gru, x0, key):
    total_steps = CONFIG["gru_target_step"]
    
    if CONFIG["loss_type"] == "nll":
        samples, means, sigmas = gru(x0, total_steps, key=key)
        predictions = samples 
    else:
        predictions = gru(x0, total_steps, key=key)
        
    step_indices = jnp.arange(total_steps)
    losses = jax.vmap(get_functional_loss)(predictions, step_indices)
    total_loss = jnp.mean(losses)
    
    return total_loss

@eqx.filter_jit
def make_step(gru, opt_state, x0, key):
    loss, grads = train_step_fn(gru, x0, key)
    updates, opt_state = opt.update(grads, opt_state, gru)
    gru = eqx.apply_updates(gru, updates)
    return gru, opt_state, loss

if TRAIN:
    print(f"🚀 Starting End-to-End GRU Training. Run ID: {run_path.name}")
    print(f"Loss: {CONFIG['loss_type']}, Final Reg: {CONFIG['final_step_mode']}")
    
    loss_history = []
    x0_static = flat_init 
    
    train_key = jax.random.PRNGKey(CONFIG["seed"] + 99)

    for ep in range(CONFIG["gru_epochs"]):
        train_key, step_key = jax.random.split(train_key)
        gru_model, opt_state, loss = make_step(gru_model, opt_state, x0_static, step_key)
        loss_history.append(loss)
        
        if (ep+1) % 100 == 0:
            print(f"Epoch {ep+1} | Loss: {loss:.6f}")

    eval_key = jax.random.PRNGKey(42)
    if CONFIG["loss_type"] == "nll":
        _, means, _ = gru_model(x0_static, CONFIG["gru_target_step"], key=eval_key)
        final_traj = means
    else:
        final_traj = gru_model(x0_static, CONFIG["gru_target_step"], key=eval_key)
        
    np.save(artefacts_path / "final_trajectory.npy", final_traj)
    np.save(artefacts_path / "loss_history.npy", np.array(loss_history))

else:
    print("Loading results...")
    final_traj = np.load(artefacts_path / "final_trajectory.npy")
    loss_history = np.load(artefacts_path / "loss_history.npy")
    x0_static = flat_init

#%%
# --- 7. VISUALIZATION ---
print("\n=== Generating Dashboards ===")

# --- DASHBOARD 1: FUNCTIONAL EVOLUTION ---
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 2)

# 1. Training Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(loss_history, color='teal', linewidth=2)
ax1.set_yscale('log')
ax1.set_title("End-to-End Functional Loss")
ax1.set_ylabel("Weighted MSE")
ax1.grid(True, alpha=0.3)

# 2. MLP Limit Model Performance (Train vs Test Loss over Steps)
ax2 = fig.add_subplot(gs[0, 1])

traj_train_losses = []
traj_test_losses = []

for i in range(len(final_traj)):
    w = final_traj[i]
    p = unflatten_pytree(w, shapes, treedef, mask)
    m = eqx.combine(p, static)
    
    y_tr = m.predict(X_train_full)
    l_tr = jnp.mean((y_tr - Y_train_full)**2)
    traj_train_losses.append(l_tr)
    
    y_te = m.predict(X_test)
    l_te = jnp.mean((y_te - Y_test)**2)
    traj_test_losses.append(l_te)

ax2.plot(traj_train_losses, label="Train MSE (Full)", color='blue', alpha=0.7)
ax2.plot(traj_test_losses, label="Test MSE", color='orange', linewidth=2)
ax2.axvline(CONFIG["n_circles"], color='k', linestyle='--', label="Expansion End")
ax2.set_yscale('log')
ax2.set_title("MLP Performance Evolution")
ax2.set_xlabel("Step")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Function Evolution
ax3 = fig.add_subplot(gs[1, :])
ax3.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1, label="Train")
ax3.scatter(X_test, Y_test, c='orange', s=10, alpha=0.2, label="Test")

x_grid = jnp.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], 300)[:, None]
cmap = plt.cm.viridis
n_steps = len(final_traj)

for i in range(0, n_steps, 2):
    w = final_traj[i]
    p = unflatten_pytree(w, shapes, treedef, mask)
    m = eqx.combine(p, static)
    pred = m.predict(x_grid)
    
    color = cmap(i / n_steps)
    alpha = 0.2 if i < CONFIG["n_circles"] else 0.6
    ax3.plot(x_grid, pred, color=color, alpha=alpha)

w_lim = final_traj[-1]
p_lim = unflatten_pytree(w_lim, shapes, treedef, mask)
m_lim = eqx.combine(p_lim, static)
pred_lim = m_lim.predict(x_grid)
ax3.plot(x_grid, pred_lim, color='red', linestyle='--', linewidth=3, label="Limit")

ax3.set_title("Function Evolution (Colored by Step)")
ax3.legend()

plt.tight_layout()
plt.savefig(plots_path / "dashboard_functional.png")
plt.show()

# --- DASHBOARD 2: PLAUSIBLE FUTURES (NLL ONLY) ---
if CONFIG["loss_type"] == "nll":
    print("Generating Futures Dashboard...")
    fig2 = plt.figure(figsize=(16, 8))
    gs2 = fig2.add_gridspec(1, 2)
    
    # A. Param Trajectories
    ax_pf = fig2.add_subplot(gs2[0, 0])
    
    n_samples = 30
    viz_key = jax.random.PRNGKey(999)
    future_stack = []
    
    for _ in range(n_samples):
        viz_key, sk = jax.random.split(viz_key)
        s, _, _ = gru_model(x0_static, CONFIG["gru_target_step"], key=sk)
        future_stack.append(s)
    
    future_stack = jnp.stack(future_stack) 
    
    # Pick random params
    np.random.seed(42)
    p_indices = np.random.choice(input_dim, 3, replace=False)
    colors = ['r', 'g', 'b']
    

    for i, pid in enumerate(p_indices):
        p_mean = jnp.mean(future_stack[:, :, pid], axis=0)
        p_min = jnp.min(future_stack[:, :, pid], axis=0)
        p_max = jnp.max(future_stack[:, :, pid], axis=0)
        
        ax_pf.plot(p_mean, color=colors[i], label=f"Param {pid}")
        ax_pf.fill_between(range(CONFIG["gru_target_step"]), p_min, p_max, color=colors[i], alpha=0.15)
    
    ax_pf.axvline(CONFIG["n_circles"], color='k', linestyle='--')
    ax_pf.set_title("Parameter Futures (Confidence Intervals)")
    ax_pf.legend()
    
    # B. Limit Ensemble
    ax_pl = fig2.add_subplot(gs2[0, 1])
    ax_pl.scatter(X_train_full, Y_train_full, c='gray', s=10, alpha=0.1)
    
    for i in range(20):
        w = future_stack[i, -1]
        p = unflatten_pytree(w, shapes, treedef, mask)
        m = eqx.combine(p, static)
        pred = m.predict(x_grid)
        ax_pl.plot(x_grid, pred, color='crimson', alpha=0.1)
        
    ax_pl.plot(x_grid, pred_lim, color='black', linestyle='--', label="Mean Limit")
    ax_pl.set_title("Plausible Limits")
    
    plt.tight_layout()
    plt.savefig(plots_path / "dashboard_futures.png")
    plt.show()


#%%
# --- 8. ADVANCED DIAGNOSTICS & EXTRA DASHBOARDS ---
# --- 8. ADVANCED DIAGNOSTICS DASHBOARD ---
print("\n=== Generating Advanced Diagnostics Dashboard ===")

# Create Unified Figure
fig_diag = plt.figure(figsize=(20, 16))
gs_diag = fig_diag.add_gridspec(2, 2)

# --- PLOT A: PARAMETER TRAJECTORIES (Top Left) ---
ax_t1 = fig_diag.add_subplot(gs_diag[0, 0])
np.random.seed(CONFIG["seed"])
param_indices = np.random.choice(input_dim, 10, replace=False)

# Get Mean Trajectory
eval_key = jax.random.PRNGKey(42)
if CONFIG["loss_type"] == "nll":
    _, mean_traj, _ = gru_model(x0_static, CONFIG["gru_target_step"], key=eval_key)
else:
    mean_traj = gru_model(x0_static, CONFIG["gru_target_step"], key=eval_key)

# Define distinct colormaps for each parameter to distinguish them
# We use sequential colormaps so time evolution is visible (Light -> Dark)
distinct_cmaps = [
    plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges,
    plt.cm.Greys, plt.cm.YlOrBr, plt.cm.PuRd, plt.cm.GnBu, plt.cm.RdPu
]

for i, idx in enumerate(param_indices):
    traj = mean_traj[:, idx]
    cmap = distinct_cmaps[i % len(distinct_cmaps)]
    
    # Plot segments colored by time using the specific colormap for this param
    for t in range(len(traj) - 1):
        # Time progress 0.0 -> 1.0
        time_prog = t / len(traj)
        # We start color at 0.3 intensity so it's visible, up to 1.0
        color = cmap(0.3 + 0.7 * time_prog)
        ax_t1.plot([t, t+1], [traj[t], traj[t+1]], color=color, linewidth=1.5)
        
    # Add a legend entry for the final point
    ax_t1.scatter([], [], color=cmap(0.9), label=f"Param {idx}")

ax_t1.axvline(CONFIG["n_circles"], color='k', linestyle='--', label="End of Data")
ax_t1.set_title("Parameter Trajectories (Unique Color Maps per Param)")
ax_t1.set_xlabel("Step")
ax_t1.set_ylabel("Weight Value")
ax_t1.legend(loc='upper right', fontsize='small', ncol=2)
ax_t1.grid(True, alpha=0.3)

# --- PLOT B: STABILITY ANALYSIS / EIGENVALUES (Top Right) ---
ax_e = fig_diag.add_subplot(gs_diag[0, 1])
print("Computing Stability Analysis...")

def get_gru_step_fn(gru, loss_type):
    def step_fn(joint_state):
        w, h = joint_state
        embedded = gru.encoder(w)
        h_new = gru.cell(embedded, h)
        w_new = gru.decoder(h_new)
        if loss_type == "nll":
            d = w_new.shape[0] // 2
            w_new = w_new[:d]
        return (w_new, h_new)
    return step_fn

# 1. Get Hidden State at Limit
limit_w = mean_traj[-1]
embedded = gru_model.encoder(limit_w)
h_limit = jnp.zeros((gru_model.cell.hidden_size,))
for _ in range(20):
    h_limit = gru_model.cell(embedded, h_limit)

# 2. Compute Jacobian
step_fn = get_gru_step_fn(gru_model, CONFIG["loss_type"])
jacobian_fn = jax.jacfwd(step_fn)

try:
    J_tuple = jacobian_fn((limit_w, h_limit))
    J_ww = J_tuple[0][0]
    J_wh = J_tuple[0][1]
    J_hw = J_tuple[1][0]
    J_hh = J_tuple[1][1]
    top = jnp.concatenate([J_ww, J_wh], axis=1)
    bot = jnp.concatenate([J_hw, J_hh], axis=1)
    Full_J = jnp.concatenate([top, bot], axis=0)
    
    eigenvals = jnp.linalg.eigvals(Full_J)
    
    unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=2, label='Stability Boundary')
    ax_e.add_patch(unit_circle)
    
    re = jnp.real(eigenvals)
    im = jnp.imag(eigenvals)
    ax_e.scatter(re, im, alpha=0.6, s=30, color='purple', label='Eigenvalues')
    
    max_rad = jnp.max(jnp.abs(eigenvals))
    ax_e.set_title(f"Jacobian Eigenvalues at Limit (Max |λ| = {max_rad:.4f})")
    ax_e.set_xlabel("Real Part")
    ax_e.set_ylabel("Imaginary Part")
    ax_e.grid(True, alpha=0.3)
    ax_e.axhline(0, color='k', alpha=0.3)
    ax_e.axvline(0, color='k', alpha=0.3)
    lim = max(1.1, float(max_rad) + 0.1)
    ax_e.set_xlim(-lim, lim)
    ax_e.set_ylim(-lim, lim)
    ax_e.set_aspect('equal')
    ax_e.legend()
    
except Exception as e:
    print(f"Skipping Eigenvalue Plot: {e}")
    ax_e.text(0.5, 0.5, "Jacobian Computation Failed", ha='center')

# --- PLOT C: 3D TRAJECTORY PCA (Bottom Left) ---
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 

print("Computing 3D PCA Trajectory...")
pca = PCA(n_components=3)
traj_3d = pca.fit_transform(np.array(mean_traj))

# Change subplot to (1, 0)
ax_3d = fig_diag.add_subplot(gs_diag[1, 0], projection='3d')

# Plot Line
ax_3d.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], color='gray', alpha=0.5)
sc = ax_3d.scatter(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], c=range(len(traj_3d)), cmap='viridis', s=40)

# Mark Key Events
ax_3d.scatter(traj_3d[0, 0], traj_3d[0, 1], traj_3d[0, 2], c='green', s=150, marker='o', label='Start')
ax_3d.scatter(traj_3d[CONFIG["n_circles"], 0], traj_3d[CONFIG["n_circles"], 1], traj_3d[CONFIG["n_circles"], 2], c='orange', s=150, marker='D', label='End of Data')
ax_3d.scatter(traj_3d[-1, 0], traj_3d[-1, 1], traj_3d[-1, 2], c='red', s=200, marker='*', label='Limit')

ax_3d.set_title("Weight Trajectory in PCA Space")
ax_3d.set_xlabel("PC 1")
ax_3d.set_ylabel("PC 2")
ax_3d.set_zlabel("PC 3")
ax_3d.legend()

# --- PLOT D: DATA EXPANSION CIRCLES (Bottom Right - New) ---
ax_circles = fig_diag.add_subplot(gs_diag[1, 1])

# We want to plot the data points colored by which circle they belong to.
# radii is the threshold for each step.
# circle_masks[i] is boolean mask for radius[i].
# We need to find the *first* index i where mask is True for each point.

# Convert boolean masks to integer index of first appearance
# Summing ~masks gives count of how many circles EXCLUDE the point.
# If a point is in Circle 5 (and thus 6, 7...), it is excluded by 0,1,2,3,4.
# So sum(~masks) = 5. This works perfectly as an index.
point_circle_indices = jnp.sum(~circle_masks, axis=0)

# Points outside all circles will have index = n_circles
# Points inside first circle will have index = 0

cmap_circles = plt.cm.turbo  # Good distinct colors
scatter = ax_circles.scatter(X_train_full, Y_train_full, c=point_circle_indices, cmap=cmap_circles, s=15, alpha=0.6)

# Draw the boundary of the last circle to show extent
final_r = radii[-1]
ax_circles.axvline(x_mean - final_r, color='k', linestyle=':', label='Max Radius')
ax_circles.axvline(x_mean + final_r, color='k', linestyle=':')

cbar = plt.colorbar(scatter, ax=ax_circles)
cbar.set_label("Circle Index (Expansion Step)")

ax_circles.set_title(f"Data Expansion (Colored by Circle Index)")
ax_circles.set_xlabel("X")
ax_circles.set_ylabel("Y")
ax_circles.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plots_path / "dashboard_advanced_diagnostics.png")
plt.show()