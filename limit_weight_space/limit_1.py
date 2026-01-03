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
from typing import List, Tuple, Optional, Any

# --- 1. CONFIGURATION ---
TRAIN = True
RUN_DIR = "" 
CONFIG = {
    "seed": time.time_ns() % (2**32 - 1),
    "lr_nn": 0.005,    
    "batch_size": 32,
    
    # Expansion Project Hyperparameters
    "n_circles": 50,           # Number of expanding circles
    "epochs_per_circle": 100,  # How long to train on each subset
    "gru_hidden_size": 128,    # Size of the Limit Finder GRU
    "gru_epochs": 5000,        # Training steps for the Limit Finder
    
    # Model Hyperparameters
    "width_size": 32,
    "noise_std": 0.05,
    "data_samples": 1000,
    "segments": 9,
    "x_range": [-1.5, 1.5],
}

#%%
# --- 2. UTILITY FUNCTIONS (FIXED) ---

def get_run_path(base_dir="experiments_expansion"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_path = Path(base_dir) / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path

def flatten_pytree(pytree):
    """ 
    Flatten the leaves of a pytree into a single array. 
    Handles 'None' leaves (from eqx.partition) by skipping them.
    Returns: flat_array, shapes, tree_def, is_array_mask
    """
    leaves, tree_def = jtree.tree_flatten(pytree)
    
    # Create a mask to remember which leaves are arrays and which are None
    is_array_mask = [x is not None for x in leaves]
    valid_leaves = [x for x in leaves if x is not None]
    
    if len(valid_leaves) == 0:
        return jnp.array([]), [], tree_def, is_array_mask

    flat = jnp.concatenate([x.flatten() for x in valid_leaves])
    shapes = [x.shape for x in valid_leaves]
    return flat, shapes, tree_def, is_array_mask

def unflatten_pytree(flat, shapes, tree_def, is_array_mask):
    """ 
    Reconstructs a pytree given flattened array and metadata.
    Uses is_array_mask to insert 'None' back into the correct places.
    """
    if len(shapes) > 0:
        leaves_prod = [np.prod(x) for x in shapes]
        splits = np.cumsum(leaves_prod)[:-1]
        arrays = jnp.split(flat, splits)
        arrays = [a.reshape(s) for a, s in zip(arrays, shapes)]
    else:
        arrays = []
        
    # Reconstruct the full list of leaves (including Nones)
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
        if len(self.x) == 0: return # Handle empty data case
        np.random.shuffle(self.indices)
        # Ensure at least one batch
        n_batches = max(1, len(self.x) // self.batch_size)
        
        for start_idx in range(0, len(self.x), self.batch_size):
            batch_idx = self.indices[start_idx:start_idx + self.batch_size]
            if len(batch_idx) > 0:
                yield self.x[batch_idx], self.y[batch_idx], batch_idx 

# Generate Data
SEED = CONFIG["seed"]
data, segs = gen_data(SEED, CONFIG["data_samples"], n_segments=CONFIG["segments"], 
                      local_structure="gradual_increase", x_range=CONFIG["x_range"], 
                      slope=0.5, base_intercept=-0.4, step_size=0.1, noise_std=CONFIG["noise_std"])

# Random split for test set (20%)
indices = np.arange(len(data))
np.random.shuffle(indices)
split = int(0.8 * len(data))
train_idx, test_idx = indices[:split], indices[split:]

X_train_full = jnp.array(data[train_idx, 0])[:, None]
Y_train_full = jnp.array(data[train_idx, 1])[:, None]
X_test = jnp.array(data[test_idx, 0])[:, None]
Y_test = jnp.array(data[test_idx, 1])[:, None]

# Compute Center for Expansion
x_mean = jnp.mean(X_train_full)
print(f"Data Center (Mean): {x_mean:.4f}")

#%%
# --- 4. MODEL DEFINITIONS ---

class MLPModel(eqx.Module):
    layers: list
    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        # Note: Mixing functions (jax.nn.relu) in the list requires eqx.partition before flattening
        self.layers = [eqx.nn.Linear(1, 64, key=k1), jax.nn.relu,
                       eqx.nn.Linear(64, 64, key=k2), jax.nn.relu,
                       eqx.nn.Linear(64, 1, key=k3)]
    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x
    def predict(self, x):
        return jax.vmap(self)(x)

class WeightGRU(eqx.Module):
    """ GRU that processes a sequence of flattened weight vectors to predict the limit. """
    cell: eqx.nn.GRUCell
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear
    
    def __init__(self, key, input_dim, hidden_dim):
        k1, k2, k3 = jax.random.split(key, 3)
        self.encoder = eqx.nn.Linear(input_dim, hidden_dim, key=k1)
        self.cell = eqx.nn.GRUCell(hidden_dim, hidden_dim, key=k2)
        self.decoder = eqx.nn.Linear(hidden_dim, input_dim, key=k3)

    def __call__(self, inputs):
        # inputs: (Seq_Len, Input_Dim)
        # Returns predicted next states
        hidden = jnp.zeros((self.cell.hidden_size,))
        def scan_fn(h, x):
            embedded = self.encoder(x)
            h_new = self.cell(embedded, h)
            out = self.decoder(h_new)
            return h_new, out
        _, preds = jax.lax.scan(scan_fn, hidden, inputs)
        return preds

    def rollout(self, current_weight, steps=50):
        embedded = self.encoder(current_weight)
        h = jnp.zeros((self.cell.hidden_size,)) 
        h = self.cell(embedded, h) 

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
# --- 5. EXPANDING TRAINING LOOP ---

if TRAIN:
    run_dir = get_run_path()
    print(f"🚀 Starting Expansion Project. Run ID: {run_dir.name}")
    
    key = jax.random.PRNGKey(CONFIG["seed"])
    k_init, k_train, k_gru = jax.random.split(key, 3)
    
    # 1. Initialize Model
    model = MLPModel(k_init)
    
    # === FIX: Partition Model (Params vs Static) ===
    # We only want to flatten the ARRAYS (weights/biases), not the functions (relu)
    params, static = eqx.partition(model, eqx.is_array)
    
    # Get Flattened Shape info using the PARAMS partition
    flat_init, shapes, treedef, mask = flatten_pytree(params)
    print(f"Model Parameter Count: {len(flat_init)}")
    
    optimizer = optax.adam(CONFIG["lr_nn"])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred = model.predict(x)
        return jnp.mean((pred - y) ** 2)

    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # 2. Define Radii
    dists = jnp.abs(X_train_full - x_mean).flatten()
    max_dist = jnp.max(dists)
    radii = jnp.linspace(0.05, max_dist + 0.01, CONFIG["n_circles"])
    
    weight_history = []
    loss_history_concat = []
    test_error_history = []
    
    # 3. Main Expansion Loop
    print("\n=== Beginning Expansion Training ===")
    
    for i, r in enumerate(radii):
        # Mask Data
        mask_idx = dists <= r
        X_sub = X_train_full[mask_idx]
        Y_sub = Y_train_full[mask_idx]
        
        if len(X_sub) < 5: 
            # Skip if circle is too small to have data
            # Flatten current params to keep history consistent
            params_now, _ = eqx.partition(model, eqx.is_array)
            weight_history.append(flatten_pytree(params_now)[0])
            test_error_history.append(test_error_history[-1] if len(test_error_history)>0 else 1.0)
            continue
            
        dm = SimpleDataHandler(X_sub, Y_sub, CONFIG["batch_size"])
        
        # Train for this circle
        circle_losses = []
        for epoch in range(CONFIG["epochs_per_circle"]):
            batch_losses = []
            for bx, by, _ in dm.get_iterator():
                model, opt_state, loss = make_step(model, opt_state, bx, by)
                batch_losses.append(loss)
            
            if len(batch_losses) > 0:
                circle_losses.append(np.mean(batch_losses))
        
        loss_history_concat.extend(circle_losses)
        
        # === FIX: Flatten only the params ===
        params_now, _ = eqx.partition(model, eqx.is_array)
        flat_params, _, _, _ = flatten_pytree(params_now)
        weight_history.append(flat_params)
        
        # Test Error
        y_test_pred = model.predict(X_test)
        test_mse = jnp.mean((y_test_pred - Y_test) ** 2)
        test_error_history.append(test_mse)
        
        if (i+1) % 5 == 0:
            print(f"Circle {i+1}/{CONFIG['n_circles']} (r={r:.2f}) | Data Points: {len(X_sub)} | Test MSE: {test_mse:.4f}")

    # Stack weights: (Time, Param_Dim)
    weight_timeseries = jnp.stack(weight_history)
    
    #%%
    # --- 6. LIMIT FINDING (GRU) ---
    print("\n=== Finding the Limit Cycle ===")
    
    input_dim = weight_timeseries.shape[1]
    gru_model = WeightGRU(k_gru, input_dim, CONFIG["gru_hidden_size"])
    
    opt_gru = optax.adam(1e-4) # Low learning rate for stability
    opt_state_gru = opt_gru.init(eqx.filter(gru_model, eqx.is_array))
    
    @eqx.filter_value_and_grad
    def gru_loss_fn(gru, seq):
        inputs = seq[:-1]
        targets = seq[1:]
        preds = gru(inputs)
        return jnp.mean((preds - targets) ** 2)

    @eqx.filter_jit
    def train_gru(gru, opt_state, seq):
        loss, grads = gru_loss_fn(gru, seq)
        updates, opt_state = opt_gru.update(grads, opt_state, gru)
        gru = eqx.apply_updates(gru, updates)
        return gru, opt_state, loss

    gru_losses = []
    for ep in range(CONFIG["gru_epochs"]):
        gru_model, opt_state_gru, loss = train_gru(gru_model, opt_state_gru, weight_timeseries)
        gru_losses.append(loss)
        if (ep+1) % 200 == 0:
            print(f"GRU Train Step {ep+1} | Loss: {loss:.6f}")

    # --- Extrapolate to Limit ---
    last_weight = weight_timeseries[-1]
    future_weights = gru_model.rollout(last_weight, steps=50)
    
    diffs = jnp.linalg.norm(future_weights[1:] - future_weights[:-1], axis=1)
    if diffs[-1] > diffs[0] * 2:
        print("⚠️ Warning: Limit search diverged. Returning final trained state.")
        limit_weight = last_weight
    else:
        print(f"✅ Limit search converged. Delta: {diffs[-1]:.6f}")
        limit_weight = future_weights[-1]

    # === FIX: Reconstruct Limit Model ===
    # 1. Unflatten the weights back into 'params' structure
    limit_params = unflatten_pytree(limit_weight, shapes, treedef, mask)
    # 2. Combine the learnt params with the saved static parts (activations)
    limit_model = eqx.combine(limit_params, static)

    #%%
    # --- 7. VISUALIZATION DASHBOARD ---
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)
    
    # Plot 1: Loss History
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(loss_history_concat, color='k', linewidth=1)
    curr_step = 0
    for _ in range(CONFIG['n_circles']):
        curr_step += CONFIG['epochs_per_circle']
        ax1.axvline(curr_step, color='gray', alpha=0.1)
    ax1.set_yscale('log')
    ax1.set_title("Training Loss (Concatenated)")
    ax1.set_xlabel("Total Epochs")
    ax1.set_ylabel("MSE")
    
    # Plot 2: Test Error
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(test_error_history, marker='o', color='crimson')
    ax2.set_title("Test Error Evolution")
    ax2.set_xlabel("Circle Radius Index")
    ax2.set_ylabel("MSE on Holdout Set")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: GRU Loss
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(gru_losses)
    ax3.set_yscale('log')
    ax3.set_title("GRU Limit Finder Loss")
    ax3.set_xlabel("GRU Epochs")
    
    # Plot 4: Predictions Evolution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(X_train_full, Y_train_full, c='gray', s=10, alpha=0.2, label="Train Data")
    
    x_grid = jnp.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], 300)[:, None]
    
    cmap = plt.cm.Blues
    n_states = len(weight_history)
    
    # Plot evolution of models
    for i in range(0, n_states, 2): # Plot every 2nd for clarity
        w = weight_history[i]
        # Reconstruct temp model for plotting
        p_tmp = unflatten_pytree(w, shapes, treedef, mask)
        tmp_model = eqx.combine(p_tmp, static)
        
        pred = tmp_model.predict(x_grid)
        color = cmap(0.2 + 0.8 * (i / n_states))
        alpha = 0.3 + 0.5 * (i / n_states)
        label = "Evolution" if i == 0 else None
        ax4.plot(x_grid, pred, color=color, alpha=alpha, linewidth=1.5, label=label)

    # Plot Limit Model
    limit_pred = limit_model.predict(x_grid)
    ax4.plot(x_grid, limit_pred, color='red', linestyle='--', linewidth=3, label="Theoretical Limit")
    
    ax4.set_title(f"Model Evolution & Limit")
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(run_dir / "expansion_project_dashboard.png")
    plt.show()

    # Save Metadata
    save_data = {
        "config": CONFIG,
        "final_test_mse": float(test_error_history[-1]),
        "weights_shape": weight_timeseries.shape
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(save_data, f, indent=4)