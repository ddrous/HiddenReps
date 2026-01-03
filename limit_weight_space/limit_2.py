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
    "batch_size": 32,
    
    # Data Splitting Hyperparameters
    "train_seg_ids": [2, 3, 4, 5, 6], # User-defined training segments
    
    # Separate Learning Rates
    "lr_mlp": 0.001,      # MLP Learning Rate
    "lr_gru": 1e-3,      # GRU Limit Finder Learning Rate
    
    # Expansion Project Hyperparameters
    "n_circles": 50,           # Number of expanding circles
    "epochs_per_circle": 750,  # How long to train on each subset
    "gru_hidden_size": 1024,    # Size of the Limit Finder GRU
    "gru_epochs": 1500,        # Training steps for the Limit Finder
    
    # Model Hyperparameters
    "width_size": 32,
    "noise_std": 0.015,
    "data_samples": 1000,
    "segments": 9,
    "x_range": [-1.5, 1.5],
}

#%%
# --- 2. UTILITY FUNCTIONS ---

def get_run_path(base_dir="experiments"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_path = Path(base_dir) / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path

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

# Generate Data
SEED = CONFIG["seed"]
data, segs = gen_data(SEED, CONFIG["data_samples"], n_segments=CONFIG["segments"], 
                      local_structure="gradual_increase", x_range=CONFIG["x_range"], 
                      slope=0.5, base_intercept=-0.4, step_size=0.1, noise_std=CONFIG["noise_std"])

# === NEW SPLITTING LOGIC ===
# "Indices of the training segments are chosen by the user as a hyperparameter"
train_mask = np.isin(segs, CONFIG["train_seg_ids"])
test_mask = ~train_mask

X_train_full = jnp.array(data[train_mask, 0])[:, None]
Y_train_full = jnp.array(data[train_mask, 1])[:, None]
X_test = jnp.array(data[test_mask, 0])[:, None]
Y_test = jnp.array(data[test_mask, 1])[:, None]

# Compute Center for Expansion (Based ONLY on Training Data)
x_mean = jnp.mean(X_train_full)
print(f"Data Center (Mean): {x_mean:.4f}")
print(f"Training Samples: {len(X_train_full)} (Segments {CONFIG['train_seg_ids']})")
print(f"Test Samples: {len(X_test)} (Segments held out)")

#%%
# --- 4. MODEL DEFINITIONS ---

class MLPModel(eqx.Module):
    layers: list
    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(1, 64, key=k1), jax.nn.relu,
                       eqx.nn.Linear(64, 64, key=k2), jax.nn.relu,
                       eqx.nn.Linear(64, 1, key=k3)]
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

    def __call__(self, inputs):
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
    
    # 1. Initialize Base Model 
    model_init = MLPModel(k_init)
    params_init, static = eqx.partition(model_init, eqx.is_array)
    flat_init, shapes, treedef, mask = flatten_pytree(params_init)
    
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

    # 2. Define Radii (Only within Training Data)
    dists = jnp.abs(X_train_full - x_mean).flatten()
    max_dist = jnp.max(dists)
    radii = jnp.linspace(0.05, max_dist + 0.01, CONFIG["n_circles"])
    
    weight_history = []
    loss_history_by_circle = []
    test_error_history = []
    
    # 3. Main Expansion Loop
    print("\n=== Beginning Expansion Training ===")
    
    for i, r in enumerate(radii):
        # Mask Data based on radius from mean
        mask_idx = dists <= r
        X_sub = X_train_full[mask_idx]
        Y_sub = Y_train_full[mask_idx]
        
        if len(X_sub) < 5: 
            weight_history.append(flat_init)
            test_error_history.append(1.0) 
            loss_history_by_circle.append([])
            continue
            
        dm = SimpleDataHandler(X_sub, Y_sub, CONFIG["batch_size"])
        
        # Reset Model: "Each new circle model must be trained from scratch"
        model_current = model_init 
        
        # New Optimizer
        optimizer = optax.adam(CONFIG["lr_mlp"])
        opt_state = optimizer.init(eqx.filter(model_current, eqx.is_array))
        
        # Train
        circle_losses = []
        for epoch in range(CONFIG["epochs_per_circle"]):
            batch_losses = []
            for bx, by, _ in dm.get_iterator():
                model_current, opt_state, loss = make_step(model_current, opt_state, bx, by, optimizer)
                batch_losses.append(loss)
            
            if len(batch_losses) > 0:
                circle_losses.append(np.mean(batch_losses))
        
        loss_history_by_circle.append(circle_losses)
        
        # Flatten and store
        params_now, _ = eqx.partition(model_current, eqx.is_array)
        flat_params, _, _, _ = flatten_pytree(params_now)
        weight_history.append(flat_params)
        
        # Test Error (On strict held-out segments)
        y_test_pred = model_current.predict(X_test)
        test_mse = jnp.mean((y_test_pred - Y_test) ** 2)
        test_error_history.append(test_mse)
        
        if (i+1) % 5 == 0:
            print(f"Circle {i+1}/{CONFIG['n_circles']} (r={r:.2f}) | Data Points: {len(X_sub)} | Test MSE: {test_mse:.4f}")

    weight_timeseries = jnp.stack(weight_history)

    ## We need to save the timeseries as a numpy array
    np.save(run_dir / "weight_timeseries.npy", np.array(weight_timeseries))
    
    #%%
    # --- 6. LIMIT FINDING (GRU) ---
    print("\n=== Finding the Limit Cycle ===")
    
    input_dim = weight_timeseries.shape[1]
    gru_model = WeightGRU(k_gru, input_dim, CONFIG["gru_hidden_size"])
    
    opt_gru = optax.adam(CONFIG["lr_gru"]) 
    opt_state_gru = opt_gru.init(eqx.filter(gru_model, eqx.is_array))
    
    @eqx.filter_value_and_grad
    def gru_loss_fn(gru, seq):
        inputs = seq[:-1]
        targets = seq[1:]
        preds = gru(inputs)
        return jnp.mean((preds - targets) ** 2)

        # ## Replace with a functional loss (unflatted weights) applied to entire traiing set
        # def single_step_loss(pred, target):
        #     pred_params = unflatten_pytree(pred, shapes, treedef, mask)
        #     target_params = unflatten_pytree(target, shapes, treedef, mask)
        #     pred_model = eqx.combine(pred_params, static)
        #     target_model = eqx.combine(target_params, static)
        #     # Use a fixed grid for loss computation
        #     x_grid = jnp.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], 200)[:, None]
        #     y_pred = pred_model.predict(x_grid)
        #     y_target = target_model.predict(x_grid)
        #     return jnp.mean((y_pred - y_target) ** 2)
        # losses = jax.vmap(single_step_loss)(preds, targets)
        # return jnp.mean(losses)

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

    # Extrapolate
    last_weight = weight_timeseries[-1]
    future_weights = gru_model.rollout(last_weight, steps=50)
    
    diffs = jnp.linalg.norm(future_weights[1:] - future_weights[:-1], axis=1)
    if diffs[-1] > diffs[0] * 2:
        print("⚠️ Warning: Limit search diverged. Returning final trained state.")
        limit_weight = last_weight
    else:
        print(f"✅ Limit search converged. Delta: {diffs[-1]:.6f}")
        limit_weight = future_weights[-1]

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
    
    # === NEW: DIFFERENT COLORS FOR TRAIN AND TEST POINTS ===
    ax4.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.2, label="Train Segments")
    ax4.scatter(X_test, Y_test, c='orange', s=10, alpha=0.2, label="Test Segments")
    
    x_grid = jnp.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], 300)[:, None]
    cmap_evol = plt.cm.Greens
    n_states = len(weight_history)
    for i in range(0, n_states, 2): 
        w = weight_history[i]
        p_tmp = unflatten_pytree(w, shapes, treedef, mask)
        tmp_model = eqx.combine(p_tmp, static)
        pred = tmp_model.predict(x_grid)
        color = cmap_evol(0.2 + 0.8 * (i / n_states))
        alpha = 0.3 + 0.5 * (i / n_states)
        ax4.plot(x_grid, pred, color=color, alpha=alpha, linewidth=1.5)
    
    limit_pred = limit_model.predict(x_grid)
    ax4.plot(x_grid, limit_pred, color='red', linestyle='--', linewidth=3, label="Limit")
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
    ## PLot the limit test error as a horizontal line
    y_test_limit = limit_model.predict(X_test)
    limit_mse = jnp.mean((y_test_limit - Y_test) ** 2)
    ax3.axhline(limit_mse, color='green', linestyle='--', label="Limit MSE")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(run_dir / "expansion_project_dashboard_v3.png")
    plt.show()

    save_data = {
        "config": CONFIG,
        "final_test_mse": float(test_error_history[-1]),
        "weights_shape": weight_timeseries.shape
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(save_data, f, indent=4)




#%%
# --- 8. ADVANCED GRU DIAGNOSTICS ---
print("\n=== Running Advanced Diagnostics ===")

# 1. Recover the hidden state at the limit
# We need the full state (weight + hidden) to compute the Jacobian accurately
def get_limit_state(gru, weight):
    # Re-run one step of rollout to get the hidden state corresponding to this weight
    # (Assuming the system has settled, h should be consistent)
    embedded = gru.encoder(weight)
    h0 = jnp.zeros((gru.cell.hidden_size,))
    # We burn in for a few steps to ensure h is consistent with w
    # In a real recurrent limit, w and h evolve together. 
    # Here we approximate h* by running the cell on w* multiple times
    h = h0
    for _ in range(10):
        h = gru.cell(embedded, h)
    return h

limit_h = get_limit_state(gru_model, limit_weight)

# 2. Define the Coupled Dynamics Function (w, h) -> (w_next, h_next)
def coupled_step(joint_state):
    w, h = joint_state
    # GRU Step
    embedded = gru_model.encoder(w)
    h_new = gru_model.cell(embedded, h)
    w_new = gru_model.decoder(h_new)
    return (w_new, h_new)

# 3. Compute Jacobian at the Limit
print("Computing Jacobian of the limit cycle (this may take a moment)...")
# We treat the weight and hidden state as a single vector for stability analysis
# Jacobian J will be of shape (N_params + N_hidden) x (N_params + N_hidden)
jacobian_fn = jax.jacfwd(coupled_step)
J_tuple = jacobian_fn((limit_weight, limit_h))

# Unpack Jacobian blocks: d(new)/d(old)
# J_ww: dw_new/dw  J_wh: dw_new/dh
# J_hw: dh_new/dw  J_hh: dh_new/dh
J_ww = J_tuple[0][0]
J_wh = J_tuple[0][1]
J_hw = J_tuple[1][0]
J_hh = J_tuple[1][1]

# Combine into one full matrix
top = jnp.concatenate([J_ww, J_wh], axis=1)
bot = jnp.concatenate([J_hw, J_hh], axis=1)
Full_J = jnp.concatenate([top, bot], axis=0)

# 4. Compute Eigenvalues
eigenvals = jnp.linalg.eigvals(Full_J)
print(f"Computed {len(eigenvals)} eigenvalues.")

# --- VISUALIZATION ---
fig_diag = plt.figure(figsize=(14, 6))
gs_diag = fig_diag.add_gridspec(1, 2)

# Plot A: GRU Training Loss (Detailed)
ax_loss = fig_diag.add_subplot(gs_diag[0, 0])
ax_loss.plot(gru_losses, color='purple', linewidth=1.5)
ax_loss.set_yscale('log')
ax_loss.set_title("GRU Limit Finder: Training Convergence")
ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("Loss (Log Scale)")
ax_loss.grid(True, alpha=0.3)

# Plot B: Eigenvalues Spectrum
ax_eig = fig_diag.add_subplot(gs_diag[0, 1])
# Unit Circle (Stability Boundary for Discrete Systems)
unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', label='Unit Circle (Stability)')
ax_eig.add_patch(unit_circle)

# Plot Eigenvalues
re = jnp.real(eigenvals)
im = jnp.imag(eigenvals)
ax_eig.scatter(re, im, alpha=0.6, s=20, color='teal', label='Eigenvalues')

# Highlight Max Eigenvalue
max_radius = jnp.max(jnp.abs(eigenvals))
ax_eig.text(0.05, 0.95, f"Max |λ|: {max_radius:.4f}", transform=ax_eig.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Formatting
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
plt.savefig(run_dir / "gru_diagnostics.png")
plt.show()

# Interpretation Text
if max_radius < 1.0:
    print("\n✅ STABILITY CONFIRMED: All eigenvalues are inside the unit circle.")
    print("The system has found a mathematically stable fixed point (attractor).")
else:
    print("\n⚠️ STABILITY WARNING: Some eigenvalues are outside the unit circle.")
    print("The finding may be an unstable fixed point or the GRU dynamic is chaotic.")
# %%


#%%
# --- 9. LINEAR WEIGHT-SPACE RNN (DIFFERENCE DRIVEN) ---

print("\n=== Training Full Matrix Linear RNN ===")

class LinearFullRNN(eqx.Module):
    # Model: theta_t = A @ theta_{t-1} + B @ (theta_{t-1} - theta_{t-2})
    # Uses FULL Matrices (Not Diagonal)
    
    A: jnp.ndarray
    B: jnp.ndarray
    
    def __init__(self, key, param_dim):
        # A initialized to Identity
        # B initialized to Zero
        # We add tiny noise to break symmetry if needed, but clean init is usually better for linear
        self.A = jnp.eye(param_dim)
        self.B = jnp.zeros((param_dim, param_dim))
        
    def __call__(self, theta_prev, theta_prev2):
        # theta_prev: (D,)
        
        diff = theta_prev - theta_prev2
        
        # Full Matrix-Vector Multiplication
        term1 = self.A @ theta_prev
        term2 = self.B @ diff
        
        return term1 + term2

    def rollout(self, init_seq, steps=50):
        trajectory = []
        t_prev = init_seq[-1]
        t_prev2 = init_seq[-2]
        
        for _ in range(steps):
            next_step = self(t_prev, t_prev2)
            trajectory.append(next_step)
            t_prev2 = t_prev
            t_prev = next_step
            
        return jnp.stack(trajectory)

# 1. Initialize
param_dim = weight_timeseries.shape[1]
print(f"Initializing Full Matrices: A and B are both {param_dim}x{param_dim}")

full_rnn = LinearFullRNN(jax.random.PRNGKey(0), param_dim)

# Optimizer
opt_full = optax.adam(0.005) 
opt_state_full = opt_full.init(eqx.filter(full_rnn, eqx.is_array))

@eqx.filter_value_and_grad
def full_loss_fn(model, seq):
    targets = seq[2:]
    t_minus_1 = seq[1:-1]
    t_minus_2 = seq[0:-2]
    
    # Vectorized prediction
    preds = jax.vmap(model)(t_minus_1, t_minus_2)
    
    mse = jnp.mean((preds - targets) ** 2)
    
    # L2 Regularization (Crucial for full matrices with few data points)
    # We penalize deviation from Identity for A, and magnitude for B
    reg_a = jnp.mean((model.A - jnp.eye(param_dim))**2)
    reg_b = jnp.mean(model.B**2)
    
    return mse + 0.1 * (reg_a + reg_b)

@eqx.filter_jit
def train_full(model, opt_state, seq):
    loss, grads = full_loss_fn(model, seq)
    updates, opt_state = opt_full.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

full_losses = []
print("Training Full Matrix RNN (This may be slower)...")
start_t = time.time()

for ep in range(600): 
    full_rnn, opt_state_full, loss = train_full(full_rnn, opt_state_full, weight_timeseries)
    full_losses.append(loss)
    if (ep+1) % 200 == 0:
        print(f"Epoch {ep+1} | Loss: {loss:.6f}")

print(f"Training finished in {time.time() - start_t:.2f}s")

# 2. Extrapolate
full_future = full_rnn.rollout(weight_timeseries[-2:], steps=50)
full_limit_weight = full_future[-1]

# Reconstruct Model
full_limit_params = unflatten_pytree(full_limit_weight, shapes, treedef, mask)
full_limit_model = eqx.combine(full_limit_params, static)

#%%
# --- 10. VISUALIZATION OF FULL MATRIX RNN ---

fig_full = plt.figure(figsize=(20, 10))
gs_full = fig_full.add_gridspec(2, 3)

# Plot A: Loss Curve
ax_f1 = fig_full.add_subplot(gs_full[0, 0])
ax_f1.plot(full_losses, color='darkgreen', linewidth=2)
ax_f1.set_yscale('log')
ax_f1.set_title("Full Matrix RNN Training Loss")
ax_f1.set_ylabel("MSE + L2 Reg")
ax_f1.grid(True, alpha=0.3)

# Plot B: Matrix Heatmaps (Zoomed In)
# We can't plot 4000x4000 easily, so we show the top-left 50x50 block
ax_f2 = fig_full.add_subplot(gs_full[0, 1])
sub_A = full_rnn.A[:50, :50]
im_a = ax_f2.imshow(sub_A, cmap='seismic', vmin=0.8, vmax=1.2) # Centered around Identity (1.0)
plt.colorbar(im_a, ax=ax_f2)
ax_f2.set_title("Matrix A (Top-Left 50x50 Block)\nShould be close to Identity")

ax_f3 = fig_full.add_subplot(gs_full[0, 2])
sub_B = full_rnn.B[:50, :50]
im_b = ax_f3.imshow(sub_B, cmap='seismic', vmin=-0.2, vmax=0.2) # Centered around 0
plt.colorbar(im_b, ax=ax_f3)
ax_f3.set_title("Matrix B (Top-Left 50x50 Block)\nMomentum Terms")

# Plot C: Eigenspectrum of A
# Stability check for Linear System: Eigenvalues of A should be inside unit circle?
# Actually for second order system, it's the Companion Matrix eigenvalues.
# But checking A gives a good heuristic of the primary driver.
ax_f4 = fig_full.add_subplot(gs_full[1, 0])
print("Computing Eigenvalues of A (may take a moment)...")
eigs_A = jnp.linalg.eigvals(full_rnn.A)
re_A = jnp.real(eigs_A)
im_A = jnp.imag(eigs_A)

ax_f4.scatter(re_A, im_A, s=5, alpha=0.5, color='purple')
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
ax_f4.add_patch(circle)
ax_f4.set_title("Eigenvalues of Matrix A")
ax_f4.set_xlabel("Real")
ax_f4.set_ylabel("Imag")
ax_f4.set_aspect('equal')
ax_f4.grid(True, alpha=0.3)

# Plot D: Extrapolation Comparison
ax_f5 = fig_full.add_subplot(gs_full[1, 1:]) # Spans 2 cols

# Background Data
ax_f5.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1, label="Train Data")
ax_f5.scatter(X_test, Y_test, c='red', s=10, alpha=0.1, label="Test Data")

x_grid = jnp.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], 300)[:, None]

# 1. GRU Limit (Crimson)
gru_limit_pred = limit_model.predict(x_grid)
ax_f5.plot(x_grid, gru_limit_pred, color='crimson', linewidth=2, label='GRU Limit')

# 2. Full Linear Limit (Green)
full_limit_pred = full_limit_model.predict(x_grid)
ax_f5.plot(x_grid, full_limit_pred, color='darkgreen', linewidth=2, linestyle='--', label='Full Matrix RNN Limit')

# 3. Last Observed (Black)
# Re-using last from loop
w_last = weight_timeseries[-1]
p_last = unflatten_pytree(w_last, shapes, treedef, mask)
last_model = eqx.combine(p_last, static)
last_pred = last_model.predict(x_grid)
ax_f5.plot(x_grid, last_pred, color='black', linestyle=':', label='Last Observed (r50)')

ax_f5.set_title("Limit Comparison: GRU vs Full Linear RNN")
ax_f5.legend()

plt.tight_layout()
plt.savefig(run_dir / "full_matrix_rnn_analysis.png")
plt.show()

# Stats
diff_norm = jnp.linalg.norm(full_limit_weight - limit_weight)
print(f"\nDistance between GRU Limit and Linear Limit: {diff_norm:.4f}")
# %%
