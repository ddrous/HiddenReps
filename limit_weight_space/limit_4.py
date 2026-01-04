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

    # Data & MLP Hyperparameters
    "data_samples": 1000,
    "noise_std": 0.015,
    "segments": 9,
    "x_range": [-1.5, 1.5],
    "train_seg_ids": [2, 3, 4, 5, 6], 
    "width_size": 16,

    # Other Hyperparameters
    "n_circles": 30,           
    "epochs_per_circle": 500,  

    # Separate Learning Rates
    "lr_mlp": 0.001,      
    "lr_gru": 1e-3,      
    "batch_size": 32,
    
    # GRU Training Config
    "gru_hidden_size": 500,    
    "gru_epochs": 2000,  
    "gru_target_step": 100,        # Total steps to unroll (50 known + 20 future)
    "functional_reg_weight": 0.1, # Weight for the functional loss at the final step
    "loss_type": "nll",           # 'mse' or 'nll'
    "teacher_forcing": False,      # If True, uses GT inputs for the first 50 steps

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
    loss_type: str
    teacher_forcing: bool
    
    def __init__(self, key, input_dim, hidden_dim, loss_type="mse", teacher_forcing=False):
        k1, k2, k3 = jax.random.split(key, 3)
        self.loss_type = loss_type
        self.teacher_forcing = teacher_forcing
        
        self.encoder = eqx.nn.Linear(input_dim, hidden_dim, key=k1)
        self.cell = eqx.nn.GRUCell(hidden_dim, hidden_dim, key=k2)
        
        # If NLL, output mean and log_std (2 * input_dim)
        out_dim = input_dim * 2 if loss_type == "nll" else input_dim
        self.decoder = eqx.nn.Linear(hidden_dim, out_dim, key=k3)

    def __call__(self, x0, steps, gt_seq=None, key=None):
        """
        x0: Initial state (Input Dim)
        steps: Total steps to unroll
        gt_seq: (Optional) Ground Truth sequence for Teacher Forcing [Length N, Dim]
                If provided, we use it for inputs where available.
        """
        h0 = jnp.zeros((self.cell.hidden_size,))
        
        if self.loss_type == "nll" and key is None:
            raise ValueError("Must provide key for NLL sampling")
            
        keys = jax.random.split(key, steps) if key is not None else [None] * steps
        
        # Prepare inputs for Scan
        # We need to know at each step whether to use GT or previous prediction
        # We'll handle this logic inside the step function
        
        # Helper to safely get GT at index t
        def get_gt(t, dummy):
            # If t < len(gt_seq), return gt_seq[t], else return dummy
            if gt_seq is None:
                return dummy
            
            # Use dynamic slicing or select
            # Note: JAX control flow requires fixed shapes. 
            # We assume gt_seq is passed with sufficient length or we handle OOB.
            is_available = t < gt_seq.shape[0]
            # safe_idx = jnp.minimum(t, gt_seq.shape[0] - 1)
            # return jax.lax.select(is_available, gt_seq[safe_idx], dummy)
            
            # Simpler for JAX: we just pass the step index and decide logic
            return jax.lax.cond(
                is_available,
                lambda: gt_seq[t],
                lambda: dummy
            )

        def scan_step(carry, inputs):
            h, x_prev, t = carry
            key_step = inputs
            
            # Determine Input for this step
            # If teacher forcing is ON and we have GT for this step, use GT.
            # GT for step 't' is used to predict 't+1'. 
            # Note: x0 is t=0. We predict t=1. 
            # Input to predict t=1 is usually x0 (or gt_seq[0]).
            # Input to predict t+1 is x_t.
            
            # If t=0, input is x0 (passed in init_carry). 
            # But inside loop, x_prev is the *previous output*.
            
            # Let's clarify: 
            # Iteration 0: input should be x0. x_prev is initialized to x0.
            # Iteration 1: input should be x1. 
            #   If TF: input is gt_seq[1]. 
            #   If AR: input is pred_1 (which is x_prev from iter 0).
            
            # Logic:
            # If (teacher_forcing AND t < len(gt_seq)): input = gt_seq[t]
            # Else: input = x_prev
            
            use_gt = jnp.logical_and(self.teacher_forcing, gt_seq is not None)
            
            # We need to access gt_seq[t]. 
            # Since gt_seq is closed-over (or passed in), we need to be careful with JAX tracing.
            # Ideally scan iterates over something.
            
            # Actually, standard way is to construct the input sequence beforehand if purely TF.
            # But we switch between TF and AR.
            
            # Let's use the x_prev logic, and overwrite it if TF is on.
            current_input = x_prev
            
            if gt_seq is not None:
                # We need to fetch gt_seq[t]. 
                # Since 't' changes, dynamic_slice is needed.
                gt_val = jax.lax.dynamic_slice(gt_seq, (t, 0), (1, x_prev.shape[0]))[0]
                
                # Only use it if t < len(gt_seq) (handled by caller logic usually, but here be safe)
                # Actually, teacher forcing usually means we feed the *known* history.
                # If we are past known history, we MUST autoregress.
                has_gt = t < gt_seq.shape[0]
                should_force = jnp.logical_and(self.teacher_forcing, has_gt)
                
                current_input = jax.lax.select(should_force, gt_val, x_prev)

            # GRU Core
            embedded = self.encoder(current_input)
            h_new = self.cell(embedded, h)
            out = self.decoder(h_new)
            
            if self.loss_type == "nll":
                d = out.shape[0] // 2
                mu, log_std = out[:d], out[d:]
                sigma = jnp.exp(log_std)
                
                eps = jax.random.normal(key_step, mu.shape)
                x_next = mu + sigma * eps
                
                # Output tuple: (sampled, mu, sigma)
                output = (x_next, mu, sigma)
                next_val = x_next
            else:
                x_next = out
                output = (x_next, x_next, jnp.zeros_like(x_next)) # Dummy sigma
                next_val = x_next
                
            return (h_new, next_val, t + 1), output

        # Init
        # x0 is t=0. We start scan to produce t=1, 2, ...
        init_carry = (h0, x0, 0)
        
        _, (samples, means, sigmas) = jax.lax.scan(scan_step, init_carry, jnp.array(keys))
        
        if self.loss_type == "nll":
            return samples, means, sigmas
        else:
            return samples

#%%
# --- 5. EXPANDING TRAINING LOOP (OR LOADING) ---

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
    np.save(artefacts_path / "test_error_history.npy", np.array(test_error_history))
    with open(artefacts_path / "loss_history.json", "w") as f:
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
# --- 6. LIMIT FINDING (GRU) WITH FUNCTIONAL REGULARIZATION & NLL ---
print(f"\n=== Finding the Limit Cycle (Loss: {CONFIG['loss_type']}, TF: {CONFIG['teacher_forcing']}) ===")

input_dim = weight_timeseries.shape[1]
gru_model = WeightGRU(
    k_gru, input_dim, CONFIG["gru_hidden_size"], 
    loss_type=CONFIG["loss_type"],
    teacher_forcing=CONFIG["teacher_forcing"]
)

opt_gru = optax.adam(CONFIG["lr_gru"]) 
opt_state_gru = opt_gru.init(eqx.filter(gru_model, eqx.is_array))

def functional_loss(flat_w):
    params = unflatten_pytree(flat_w, shapes, treedef, mask)
    model = eqx.combine(params, static)
    y_pred = model.predict(X_train_full)
    return jnp.mean((y_pred - Y_train_full) ** 2)

@eqx.filter_value_and_grad
def gru_loss_fn(gru, gt_seq, key):
    # gt_seq: (N, D) -> e.g. 50 steps
    # We unroll for CONFIG["gru_target_step"] (e.g. 70)
    
    x0 = gt_seq[0]
    total_steps = CONFIG["gru_target_step"]
    
    # Pass gt_seq for Teacher Forcing logic inside call
    # Note: gt_seq includes t=0, so index t in gt_seq corresponds to step t.
    if CONFIG["loss_type"] == "nll":
        samples, means, sigmas = gru(x0, total_steps, gt_seq=gt_seq, key=key)
        predictions = samples
    else:
        predictions = gru(x0, total_steps, gt_seq=gt_seq, key=key)
        means = predictions
    
    # Loss on Overlap (Indices 1 to N-1 of GT matched against Preds 0 to N-2)
    # predictions[i] matches time i+1
    
    n_gt = gt_seq.shape[0] - 1
    preds_overlap = predictions[:n_gt] 
    gt_overlap = gt_seq[1:] 
    
    if CONFIG["loss_type"] == "nll":
        mu_overlap = means[:n_gt]
        sigma_overlap = sigmas[:n_gt]
        var = sigma_overlap ** 2
        # NLL: 0.5 * (log(2pi) + 2log(sigma) + (y-mu)^2/sigma^2)
        nll = 0.5 * (jnp.log(2 * jnp.pi) + jnp.log(var) + (gt_overlap - mu_overlap)**2 / var)
        loss_recon = jnp.mean(nll)
    else:
        loss_recon = jnp.mean((preds_overlap - gt_overlap) ** 2)
        
    final_pred = predictions[-1] # Step 70
    loss_func = functional_loss(final_pred)
    
    return loss_recon + CONFIG["functional_reg_weight"] * loss_func

@eqx.filter_jit
def train_gru(gru, opt_state, seq, key):
    loss, grads = gru_loss_fn(gru, seq, key)
    updates, opt_state = opt_gru.update(grads, opt_state, gru)
    gru = eqx.apply_updates(gru, updates)
    return gru, opt_state, loss

gru_losses = []
gru_key = jax.random.PRNGKey(CONFIG["seed"] + 1)

print(f"Training GRU for {CONFIG['gru_target_step']} steps...")
for ep in range(CONFIG["gru_epochs"]):
    gru_key, step_key = jax.random.split(gru_key)
    gru_model, opt_state_gru, loss = train_gru(gru_model, opt_state_gru, weight_timeseries, step_key)
    gru_losses.append(loss)
    if (ep+1) % 100 == 0:
        print(f"GRU Train Step {ep+1} | Loss: {loss:.6f}")

#%%
# --- 7. VISUALIZATION DASHBOARDS ---
# --- 7. VISUALIZATION DASHBOARDS ---
print("\n=== Generating Visualization Dashboards ===")

# --- DASHBOARD 1: STANDARD (ALWAYS GENERATED) ---
# Changed layout to (2, 3) to fit GRU loss on the left
fig = plt.figure(figsize=(22, 12))
gs = fig.add_gridspec(2, 3)

# 0. GRU Training Loss (New Leftmost Axis)

ax0 = fig.add_subplot(gs[0, 1])
## Add the minimul so that the NLL is always positive
if CONFIG["loss_type"] == "nll":
    min_loss = min(gru_losses)
    gru_losses = [l - min_loss + 1e-4 for l in gru_losses]
ax0.plot(gru_losses, color='purple', linewidth=2)
ax0.set_yscale('log')
ax0.set_title("GRU Limit Finder Loss")
ax0.set_xlabel("GRU Epochs")
ax0.set_ylabel("Loss" + (f" (Shifted by {min_loss:.1e})" if CONFIG["loss_type"] == "nll" else ""))
ax0.grid(True, alpha=0.3)

# 1. MLP Training Loss History
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
ax1.set_title("MLP Training Loss (By Circle)")
ax1.set_xlabel("Cumulative MLP Epochs")
ax1.set_ylabel("MSE")

# 2. Test Error Evolution (With Extrapolation)
ax2 = fig.add_subplot(gs[0, 2])
# Plot observed test error (0-50)
ax2.plot(range(len(test_error_history)), test_error_history, marker='o', color='crimson', label="Observed (0-50)")

# Calculate projected test error for extrapolated states (51-70)
# We use the Mean Trajectory for this
eval_key = jax.random.PRNGKey(999)
x0 = weight_timeseries[0]
if CONFIG["loss_type"] == "nll":
    _, means, _ = gru_model(x0, CONFIG["gru_target_step"], gt_seq=None, key=eval_key)
    mean_traj = means 
else:
    mean_traj = gru_model(x0, CONFIG["gru_target_step"], gt_seq=None, key=eval_key)

# We only care about the extrapolated part (index 49 onwards, corresponding to steps 50+)
extra_test_errors = []
start_idx = len(test_error_history) - 1 # approx step 49/50
for i in range(start_idx, mean_traj.shape[0]):
    w = mean_traj[i]
    p_tmp = unflatten_pytree(w, shapes, treedef, mask)
    tmp_model = eqx.combine(p_tmp, static)
    y_tp = tmp_model.predict(X_test)
    mse = jnp.mean((y_tp - Y_test) ** 2)
    extra_test_errors.append(mse)

# Plot extrapolation
x_extra = range(len(test_error_history) - 1, len(test_error_history) - 1 + len(extra_test_errors))
ax2.plot(x_extra, extra_test_errors, 'o-', color='darkred', linewidth=2, label="Projected (50-70)")

ax2.set_title("Test Error Evolution")
ax2.set_xlabel("Step Index")
ax2.set_ylabel("MSE")
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Mean Trajectory (Weight Space)
ax3 = fig.add_subplot(gs[1, 0])
np.random.seed(42)
param_indices = np.random.choice(weight_timeseries.shape[1], 10, replace=False)
for p_idx in param_indices:
    # Plot GT
    ax3.plot(range(len(weight_timeseries)), weight_timeseries[:, p_idx], color='gray', alpha=0.2)
        # , weight_timeseries[:, p_idx], color='black', alpha=0.3)
    # Plot Mean Pred
    ax3.plot(range(1, 1 + len(mean_traj)), mean_traj[:, p_idx], alpha=0.8)
ax3.axvline(50, color='k', linestyle='--', label="Train Cutoff")
ax3.set_title("Mean Weight Trajectories (10 Random Params)")
ax3.set_xlabel("Step")

# 4. Mean Function Evolution
ax4 = fig.add_subplot(gs[1, 1:]) # Span last 2 cols
ax4.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1, label="Train")
ax4.scatter(X_test, Y_test, c='orange', s=10, alpha=0.2, label="Test")
x_grid = jnp.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], 300)[:, None]
cmap_evol = plt.cm.Greens

# Use mean_traj for plotting
n_states = mean_traj.shape[0]
for i in range(0, n_states, 2): 
    w = mean_traj[i]
    p_tmp = unflatten_pytree(w, shapes, treedef, mask)
    tmp_model = eqx.combine(p_tmp, static)
    pred = tmp_model.predict(x_grid)
    color = cmap_evol(0.2 + 0.8 * (i / n_states))
    ax4.plot(x_grid, pred, color=color, alpha=0.3)

# Limit
limit_w = mean_traj[-1]
p_lim = unflatten_pytree(limit_w, shapes, treedef, mask)
m_lim = eqx.combine(p_lim, static)
limit_pred = m_lim.predict(x_grid)
ax4.plot(x_grid, limit_pred, color='red', linestyle='--', linewidth=2, label="Mean Limit")

ax4.set_title("Mean Function Evolution")
ax4.legend()

plt.tight_layout()
plt.savefig(plots_path / "dashboard_standard.png")
plt.show()

# --- DASHBOARD 2: PLAUSIBLE FUTURES (NLL ONLY) ---
if CONFIG["loss_type"] == "nll":
    print("\n=== Generating Plausible Futures Dashboard ===")
    fig2 = plt.figure(figsize=(16, 8))
    gs2 = fig2.add_gridspec(1, 2)
    
    # A. Plausible Futures (Weight Space - Shaded Zones)
    ax_pf = fig2.add_subplot(gs2[0, 0])
    
    n_samples = 50 # Increase samples for better bounds
    viz_key = jax.random.PRNGKey(42)
    future_trajs = []
    
    # Collect trajectories
    for _ in range(n_samples):
        viz_key, sk = jax.random.split(viz_key)
        samples, _, _ = gru_model(x0, CONFIG["gru_target_step"], gt_seq=None, key=sk)
        full_traj = jnp.concatenate([x0[None, :], samples], axis=0) # (71, D)
        future_trajs.append(full_traj)
    
    future_trajs_stack = jnp.stack(future_trajs) # (Samples, Steps, D)
    
    # Visualize 3 params with Confidence Intervals
    colors = ['red', 'blue', 'green']
    viz_params = param_indices[:3] 

# [Image of confidence interval plot]

    for i, p_idx in enumerate(viz_params):
        # GT (0-50)
        ax_pf.plot(range(len(weight_timeseries)), weight_timeseries[:, p_idx], color=colors[i], linewidth=2, linestyle=':')
        
        # Stats over samples
        p_mean = jnp.mean(future_trajs_stack[:, :, p_idx], axis=0)
        p_min = jnp.min(future_trajs_stack[:, :, p_idx], axis=0)
        p_max = jnp.max(future_trajs_stack[:, :, p_idx], axis=0)
        
        # Plot Mean
        ax_pf.plot(range(CONFIG["gru_target_step"] + 1), p_mean, color=colors[i], linewidth=2, label=f"Param {p_idx}") 
        # Plot Shaded Zone
        ax_pf.fill_between(range(CONFIG["gru_target_step"] + 1), p_min, p_max, color=colors[i], alpha=0.15)
            
    ax_pf.axvline(50, color='k', linestyle='--', label="Cutoff")
    ax_pf.set_title("Plausible Parameter Futures (Min/Max Envelope)")
    ax_pf.set_xlabel("Step")
    ax_pf.legend()
    
    # B. Plausible Limits (Function Space)
    ax_pl = fig2.add_subplot(gs2[0, 1])
    ax_pl.scatter(X_train_full, Y_train_full, c='gray', s=10, alpha=0.1, label="Train")
    # Add Test Points
    ax_pl.scatter(X_test, Y_test, c='orange', s=15, alpha=0.3, label="Test")
    
    # Plot individual limits lightly
    for traj in future_trajs[:20]: # Only plot first 20 to avoid lag
        w_lim = traj[-1]
        p_l = unflatten_pytree(w_lim, shapes, treedef, mask)
        m_l = eqx.combine(p_l, static)
        pred_l = m_l.predict(x_grid)
        ax_pl.plot(x_grid, pred_l, color='crimson', alpha=0.05)
        
    ax_pl.plot(x_grid, limit_pred, color='black', linestyle='--', linewidth=2, label="Mean Limit")
    ax_pl.set_title("Plausible Limits (Ensemble)")
    ax_pl.legend()
    
    plt.tight_layout()
    plt.savefig(plots_path / "dashboard_futures.png")
    plt.show()

# --- 8. ADVANCED DIAGNOSTICS ---
print("\n=== Running Advanced Diagnostics ===")

# (Same diagnostic code as before, using mean limit)
def get_limit_state(gru, weight):
    embedded = gru.encoder(weight)
    h0 = jnp.zeros((gru.cell.hidden_size,))
    h = h0
    for _ in range(10):
        h = gru.cell(embedded, h)
    return h

limit_h = get_limit_state(gru_model, limit_w) # Use mean limit weight

def coupled_step(joint_state):
    w, h = joint_state
    embedded = gru_model.encoder(w)
    h_new = gru_model.cell(embedded, h)
    w_new = gru_model.decoder(h_new)
    # Note: For Jacobian we analyze the deterministic path (mean)
    if CONFIG["loss_type"] == "nll":
        d = w_new.shape[0] // 2
        w_new = w_new[:d] # Just mean
    return (w_new, h_new)

print("Computing Jacobian...")
# Jacobian calculation might fail if shapes don't align perfectly due to NLL output size
# We need to wrap coupled_step to ensure w_new matches w input size
jacobian_fn = jax.jacfwd(coupled_step)
try:
    J_tuple = jacobian_fn((limit_w, limit_h))
    J_ww = J_tuple[0][0]
    J_wh = J_tuple[0][1]
    J_hw = J_tuple[1][0]
    J_hh = J_tuple[1][1]
    top = jnp.concatenate([J_ww, J_wh], axis=1)
    bot = jnp.concatenate([J_hw, J_hh], axis=1)
    Full_J = jnp.concatenate([top, bot], axis=0)
    eigenvals = jnp.linalg.eigvals(Full_J)
    
    # Plot Eigenvalues
    fig_diag = plt.figure(figsize=(6, 6))
    ax_eig = fig_diag.add_subplot(1, 1, 1)
    unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    ax_eig.add_patch(unit_circle)
    re = jnp.real(eigenvals)
    im = jnp.imag(eigenvals)
    ax_eig.scatter(re, im, alpha=0.6, s=20, color='teal')
    max_radius = jnp.max(jnp.abs(eigenvals))
    ax_eig.text(0.05, 0.95, f"Max |λ|: {max_radius:.4f}", transform=ax_eig.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    ax_eig.set_title("Stability Analysis")
    plt.savefig(plots_path / "diagnostics.png")
    plt.show()
except Exception as e:
    print(f"Skipping Jacobian analysis: {e}")
# %%
