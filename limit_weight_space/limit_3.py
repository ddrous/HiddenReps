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
TRAIN = True  # Set to False to load from RUN_DIR
RUN_DIR = "." # Set this when TRAIN = False

CONFIG = {
    "seed": time.time_ns() % (2**32 - 1),
    "batch_size": 32,
    
    # Data Splitting Hyperparameters
    "train_seg_ids": [2, 3, 4, 5, 6], 
    
    # Separate Learning Rates
    "lr_mlp": 0.001,      
    "lr_gru": 1e-3,      
    
    # Expansion Project Hyperparameters
    "n_circles": 50,           
    "epochs_per_circle": 750,  
    "gru_hidden_size": 128,    
    "gru_epochs": 500,  
    
    # New GRU Constraints
    "gru_target_step": 70,      # "As if we had 100 steps"
    "functional_reg_weight": 0.1, # Weight for the functional loss at step 100
    
    # Model Hyperparameters
    "width_size": 16,
    "noise_std": 0.015,
    "data_samples": 1000,
    "segments": 9,
    "x_range": [-1.5, 1.5],
}

#%%
# --- 2. UTILITY FUNCTIONS ---

def setup_run_dir(base_dir="experiments"):
    if TRAIN:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_path = Path(base_dir) / timestamp
        run_path.mkdir(parents=True, exist_ok=True)
        
        # Create Subfolders
        (run_path / "artefacts").mkdir(exist_ok=True)
        (run_path / "plots").mkdir(exist_ok=True)
        
        # Save the current script
        try:
            # Try to get the file path if running as a script
            current_file = Path(__file__)
        except NameError:
            # Fallback for notebooks/interactive (try to find the file or skip)
            current_file = Path(sys.argv[0]) if sys.argv[0] else None
            
        if current_file and current_file.exists():
            shutil.copy(current_file, run_path / "main.py")
            
        return run_path
    else:
        # If not training, we use the provided RUN_DIR
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

class SimpleDataHandler:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(x))
    
    def get_iterator(self):
        if len(self.x) == 0: return 
        np.random.shuffle(self.indices)
        n_batches = max(1, len(self.x) // self.batch_size)
        
        for start_idx in range(0, len(self.x), self.batch_size):
            batch_idx = self.indices[start_idx:start_idx + self.batch_size]
            if len(batch_idx) > 0:
                yield self.x[batch_idx], self.y[batch_idx], batch_idx 

# --- DATA LOADING / GENERATION LOGIC ---
run_path = setup_run_dir()
artefacts_path = run_path / "artefacts"
plots_path = run_path / "plots"

if TRAIN:
    # Generate Data Fresh
    SEED = CONFIG["seed"]
    data, segs = gen_data(SEED, CONFIG["data_samples"], n_segments=CONFIG["segments"], 
                          local_structure="gradual_increase", x_range=CONFIG["x_range"], 
                          slope=0.5, base_intercept=-0.4, step_size=0.1, noise_std=CONFIG["noise_std"])

    train_mask = np.isin(segs, CONFIG["train_seg_ids"])
    test_mask = ~train_mask

    X_train_full = jnp.array(data[train_mask, 0])[:, None]
    Y_train_full = jnp.array(data[train_mask, 1])[:, None]
    X_test = jnp.array(data[test_mask, 0])[:, None]
    Y_test = jnp.array(data[test_mask, 1])[:, None]
    
    # Save Data for Reproducibility
    np.save(artefacts_path / "X_train_full.npy", X_train_full)
    np.save(artefacts_path / "Y_train_full.npy", Y_train_full)
    np.save(artefacts_path / "X_test.npy", X_test)
    np.save(artefacts_path / "Y_test.npy", Y_test)
    
else:
    # Load Data from Artefacts
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

#%%
# --- 4. MODEL DEFINITIONS ---

width_size = CONFIG["width_size"]
class MLPModel(eqx.Module):
    layers: list
    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(1, width_size, key=k1), jax.nn.relu,
                       eqx.nn.Linear(width_size, width_size, key=k2), jax.nn.relu,
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
    
    def __init__(self, key, input_dim, hidden_dim):
        k1, k2, k3 = jax.random.split(key, 3)
        self.encoder = eqx.nn.Linear(input_dim, hidden_dim, key=k1)
        self.cell = eqx.nn.GRUCell(hidden_dim, hidden_dim, key=k2)
        self.decoder = eqx.nn.Linear(hidden_dim, input_dim, key=k3)

    def __call__(self, inputs, h0=None):
        # inputs: (Seq, Dim)
        # Returns all states
        if h0 is None:
            h0 = jnp.zeros((self.cell.hidden_size,))
            
        def scan_fn(h, x):
            embedded = self.encoder(x)
            h_new = self.cell(embedded, h)
            out = self.decoder(h_new)
            return h_new, out
            
        _, preds = jax.lax.scan(scan_fn, h0, inputs)
        return preds

    def rollout(self, current_weight, steps=50, h_init=None):
        if h_init is None:
            # Cold start from current weight
            embedded = self.encoder(current_weight)
            h = jnp.zeros((self.cell.hidden_size,)) 
            h = self.cell(embedded, h) 
        else:
            h = h_init

        trajectory = []
        curr = current_weight
        
        for _ in range(steps):
            embedded = self.encoder(curr)
            h = self.cell(embedded, h)
            pred_next = self.decoder(h)
            trajectory.append(pred_next)
            curr = pred_next
            
        return jnp.stack(trajectory)

#%%
# --- 5. EXPANDING TRAINING LOOP (OR LOADING) ---

# Need to setup model structure to get static parts/shapes even if not training
key = jax.random.PRNGKey(CONFIG["seed"])
k_init, k_train, k_gru = jax.random.split(key, 3)
model_init = MLPModel(k_init)
params_init, static = eqx.partition(model_init, eqx.is_array)
flat_init, shapes, treedef, mask = flatten_pytree(params_init)

if TRAIN:
    print(f"🚀 Starting Expansion Project. Run ID: {run_path.name}")
    
    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred = model.predict(x)
        return jnp.mean((pred - y) ** 2)

    @eqx.filter_jit
    def make_step(model, opt_state, x, y, optimizer): 
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    dists = jnp.abs(X_train_full - x_mean).flatten()
    max_dist = jnp.max(dists)
    radii = jnp.linspace(0.05, max_dist + 0.01, CONFIG["n_circles"])
    
    weight_history = []
    loss_history_by_circle = []
    test_error_history = []
    
    print("\n=== Beginning Expansion Training ===")
    
    for i, r in enumerate(radii):
        mask_idx = dists <= r
        X_sub = X_train_full[mask_idx]
        Y_sub = Y_train_full[mask_idx]
        
        if len(X_sub) < 5: 
            weight_history.append(flat_init)
            test_error_history.append(1.0) 
            loss_history_by_circle.append([])
            continue
            
        dm = SimpleDataHandler(X_sub, Y_sub, CONFIG["batch_size"])
        model_current = model_init 
        optimizer = optax.adam(CONFIG["lr_mlp"])
        opt_state = optimizer.init(eqx.filter(model_current, eqx.is_array))
        
        circle_losses = []
        for epoch in range(CONFIG["epochs_per_circle"]):
            batch_losses = []
            for bx, by, _ in dm.get_iterator():
                model_current, opt_state, loss = make_step(model_current, opt_state, bx, by, optimizer)
                batch_losses.append(loss)
            
            if len(batch_losses) > 0:
                circle_losses.append(np.mean(batch_losses))
        
        loss_history_by_circle.append(circle_losses)
        
        params_now, _ = eqx.partition(model_current, eqx.is_array)
        flat_params, _, _, _ = flatten_pytree(params_now)
        weight_history.append(flat_params)
        
        y_test_pred = model_current.predict(X_test)
        test_mse = jnp.mean((y_test_pred - Y_test) ** 2)
        test_error_history.append(test_mse)
        
        if (i+1) % 5 == 0:
            print(f"Circle {i+1}/{CONFIG['n_circles']} (r={r:.2f}) | Data Points: {len(X_sub)} | Test MSE: {test_mse:.4f}")

    weight_timeseries = jnp.stack(weight_history)

    # Save Artefacts
    np.save(artefacts_path / "weight_timeseries.npy", np.array(weight_timeseries))
    # Save test error history
    np.save(artefacts_path / "test_error_history.npy", np.array(test_error_history))
    # Save loss history (list of lists)
    with open(artefacts_path / "loss_history.json", "w") as f:
        # Convert numpy floats to native python floats for JSON serialization
        clean_loss = [[float(x) for x in sublist] for sublist in loss_history_by_circle]
        json.dump(clean_loss, f)

else:
    print(f"Loading weights and histories from {artefacts_path}...")
    try:
        weight_timeseries = jnp.array(np.load(artefacts_path / "weight_timeseries.npy"))
        test_error_history = list(np.load(artefacts_path / "test_error_history.npy"))
        with open(artefacts_path / "loss_history.json", "r") as f:
            loss_history_by_circle = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load artefacts: {e}. Run with TRAIN=True first.")

#%%
# --- 6. LIMIT FINDING (GRU) WITH FUNCTIONAL REGULARIZATION ---
print("\n=== Finding the Limit Cycle (Functional Reg) ===")

input_dim = weight_timeseries.shape[1]
gru_model = WeightGRU(k_gru, input_dim, CONFIG["gru_hidden_size"])

opt_gru = optax.adam(CONFIG["lr_gru"]) 
opt_state_gru = opt_gru.init(eqx.filter(gru_model, eqx.is_array))

# Helper to compute functional loss given a flattened weight vector
def functional_loss(flat_w):
    # Unflatten
    params = unflatten_pytree(flat_w, shapes, treedef, mask)
    model = eqx.combine(params, static)
    
    # Predict on Full Train Set
    y_pred = model.predict(X_train_full)
    
    # MSE vs Ground Truth Train Labels
    return jnp.mean((y_pred - Y_train_full) ** 2)

@eqx.filter_value_and_grad
def gru_loss_fn(gru, seq):
    # seq: (N_circles, D) -> e.g. 50 steps
    # We want to match the trajectory for the known steps
    # And regularize the step at `target_step`
    
    N_known = seq.shape[0]
    target_step = CONFIG["gru_target_step"]
    
    # 1. Run GRU on the known sequence to match trajectory
    # Input: seq[0...N-2], Target: seq[1...N-1]
    # Standard next-step prediction
    inputs_known = seq[:-1]
    targets_known = seq[1:]
    
    # We need to run the scan to get hidden states if we want to continue efficiently,
    # or we can just use the generic call.
    preds_known = gru(inputs_known) # (N-1, D)
    
    loss_traj = jnp.mean((preds_known - targets_known) ** 2)
    
    # 2. Rollout to Target Step
    # We need to start the rollout from the last known state
    # To do this correctly, we need the hidden state after processing the whole sequence
    # Let's run scan keeping the final hidden state
    
    h0 = jnp.zeros((gru.cell.hidden_size,))
    def scan_fn(h, x):
        embedded = gru.encoder(x)
        h_new = gru.cell(embedded, h)
        return h_new, h_new
    
    # Run over all known inputs to get final h
    _, hs = jax.lax.scan(scan_fn, h0, seq[:-1])
    h_final_known = hs[-1] # Hidden state after seeing index 48 (predicting 49)
    
    # The last prediction we have is preds_known[-1] which approximates seq[-1]
    # We start autoregressive rollout from preds_known[-1]
    curr_input = preds_known[-1]
    h_curr = h_final_known
    
    # Number of steps to unroll: from N to target_step
    # Current index is N (50). We want to reach target_step (100).
    steps_to_roll = target_step - N_known
    
    def rollout_step(carry, _):
        h, x = carry
        embedded = gru.encoder(x)
        h_new = gru.cell(embedded, h)
        pred = gru.decoder(h_new)
        return (h_new, pred), pred

    if steps_to_roll > 0:
        (h_end, final_pred), _ = jax.lax.scan(rollout_step, (h_curr, curr_input), None, length=steps_to_roll)
    else:
        # If target step is within training range (unlikely given config)
        final_pred = curr_input # Fallback
    
    # 3. Functional Regularization on the FINAL predicted weight
    loss_func = functional_loss(final_pred)
    
    return loss_traj + CONFIG["functional_reg_weight"] * loss_func

@eqx.filter_jit
def train_gru(gru, opt_state, seq):
    loss, grads = gru_loss_fn(gru, seq)
    updates, opt_state = opt_gru.update(grads, opt_state, gru)
    gru = eqx.apply_updates(gru, updates)
    return gru, opt_state, loss

gru_losses = []
print(f"Training GRU with functional regularization at step {CONFIG['gru_target_step']}...")
for ep in range(CONFIG["gru_epochs"]):
    gru_model, opt_state_gru, loss = train_gru(gru_model, opt_state_gru, weight_timeseries)
    gru_losses.append(loss)
    if (ep+1) % 200 == 0:
        print(f"GRU Train Step {ep+1} | Loss: {loss:.6f}")

# Extrapolate for Visualization
# We just use the standard rollout function here for the plot
last_weight = weight_timeseries[-1]
future_weights = gru_model.rollout(last_weight, steps=50) # Just 50 steps for viz

limit_weight = future_weights[-1] # This is approximate limit for viz
limit_params = unflatten_pytree(limit_weight, shapes, treedef, mask)
limit_model = eqx.combine(limit_params, static)

#%%
# --- 7. VISUALIZATION DASHBOARD ---

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2)

# Plot 1: Loss History
ax1 = fig.add_subplot(gs[0, 0])
cmap_loss = plt.cm.viridis
total_epochs = 0
for idx, losses in enumerate(loss_history_by_circle):
    if len(losses) == 0: continue
    epochs = np.arange(total_epochs, total_epochs + len(losses))
    color = cmap_loss(idx / len(loss_history_by_circle))
    ax1.plot(epochs, losses, color=color, linewidth=1.5)
    total_epochs += len(losses)
ax1.set_yscale('log')
ax1.set_title("Training Loss (Each Circle Colored)")
ax1.set_xlabel("Cumulative Epochs")
ax1.set_ylabel("MSE")

# Plot 2: Weight Evolution
ax2 = fig.add_subplot(gs[1, 0])
np.random.seed(42)
param_indices = np.random.choice(weight_timeseries.shape[1], 10, replace=False)
for p_idx in param_indices:
    traj = weight_timeseries[:, p_idx]
    ax2.plot(traj, alpha=0.8)
for p_idx in param_indices:
    future_traj = future_weights[:, p_idx]
    x_future = np.arange(len(weight_timeseries), len(weight_timeseries) + len(future_traj))
    ax2.plot(x_future, future_traj, linestyle=':', alpha=0.6, color='black')
ax2.set_title("Weight Trajectories (10 Random Params)")
ax2.set_xlabel("Circle Index -> Extrapolation")

# Plot 3: Predictions Evolution
ax4 = fig.add_subplot(gs[0, 1])
ax4.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.2, label="Train Segments")
ax4.scatter(X_test, Y_test, c='orange', s=10, alpha=0.2, label="Test Segments")

x_grid = jnp.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], 300)[:, None]
cmap_evol = plt.cm.Greens
n_states = weight_timeseries.shape[0]
for i in range(0, n_states, 2): 
    # Must unflatten using the shapes from init
    w = weight_timeseries[i]
    p_tmp = unflatten_pytree(w, shapes, treedef, mask)
    tmp_model = eqx.combine(p_tmp, static)
    pred = tmp_model.predict(x_grid)
    color = cmap_evol(0.2 + 0.8 * (i / n_states))
    alpha = 0.3 + 0.5 * (i / n_states)
    ax4.plot(x_grid, pred, color=color, alpha=alpha, linewidth=1.5)

limit_pred = limit_model.predict(x_grid)
ax4.plot(x_grid, limit_pred, color='red', linestyle='--', linewidth=3, label="Limit (GRU)")
ax4.set_title(f"Model Evolution & Limit")
ax4.legend()

# Plot 4: Test Error
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(test_error_history, marker='o', color='crimson')
ax3.set_title("Test Error Evolution")
ax3.set_xlabel("Circle Radius Index")
ax3.set_ylabel("MSE")
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')
y_test_limit = limit_model.predict(X_test)
limit_mse = jnp.mean((y_test_limit - Y_test) ** 2)
ax3.axhline(limit_mse, color='green', linestyle='--', label="Limit MSE")
ax3.legend()

plt.tight_layout()
plt.savefig(plots_path / "expansion_dashboard.png")
plt.show()

# --- 8. ADVANCED GRU DIAGNOSTICS ---
print("\n=== Running Advanced Diagnostics ===")

def get_limit_state(gru, weight):
    embedded = gru.encoder(weight)
    h0 = jnp.zeros((gru.cell.hidden_size,))
    h = h0
    for _ in range(10):
        h = gru.cell(embedded, h)
    return h

limit_h = get_limit_state(gru_model, limit_weight)

def coupled_step(joint_state):
    w, h = joint_state
    embedded = gru_model.encoder(w)
    h_new = gru_model.cell(embedded, h)
    w_new = gru_model.decoder(h_new)
    return (w_new, h_new)

print("Computing Jacobian of the limit cycle...")
jacobian_fn = jax.jacfwd(coupled_step)
J_tuple = jacobian_fn((limit_weight, limit_h))

J_ww = J_tuple[0][0]
J_wh = J_tuple[0][1]
J_hw = J_tuple[1][0]
J_hh = J_tuple[1][1]

top = jnp.concatenate([J_ww, J_wh], axis=1)
bot = jnp.concatenate([J_hw, J_hh], axis=1)
Full_J = jnp.concatenate([top, bot], axis=0)

eigenvals = jnp.linalg.eigvals(Full_J)
print(f"Computed {len(eigenvals)} eigenvalues.")

fig_diag = plt.figure(figsize=(14, 6))
gs_diag = fig_diag.add_gridspec(1, 2)

# Plot A: GRU Training Loss
ax_loss = fig_diag.add_subplot(gs_diag[0, 0])
ax_loss.plot(gru_losses, color='purple', linewidth=1.5)
ax_loss.set_yscale('log')
ax_loss.set_title("GRU Limit Finder: Training Convergence")
ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("Loss (Log Scale)")
ax_loss.grid(True, alpha=0.3)

# Plot B: Eigenvalues Spectrum
ax_eig = fig_diag.add_subplot(gs_diag[0, 1])
unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', label='Unit Circle')
ax_eig.add_patch(unit_circle)

re = jnp.real(eigenvals)
im = jnp.imag(eigenvals)
ax_eig.scatter(re, im, alpha=0.6, s=20, color='teal', label='Eigenvalues')

max_radius = jnp.max(jnp.abs(eigenvals))
ax_eig.text(0.05, 0.95, f"Max |λ|: {max_radius:.4f}", transform=ax_eig.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax_eig.axvline(0, color='k', linewidth=0.5, alpha=0.5)
ax_eig.axhline(0, color='k', linewidth=0.5, alpha=0.5)
limit_view = max(1.5, float(max_radius) + 0.5)
ax_eig.set_xlim(-limit_view, limit_view)
ax_eig.set_ylim(-limit_view, limit_view)
ax_eig.set_aspect('equal')
ax_eig.set_title("Stability Analysis (Jacobian Eigenvalues)")
ax_eig.set_xlabel("Real Part")
ax_eig.set_ylabel("Imaginary Part")
ax_eig.legend(loc='lower right')
ax_eig.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plots_path / "gru_diagnostics.png")
plt.show()
