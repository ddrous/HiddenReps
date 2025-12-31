#%% [markdown]
# - Key takeways. Increasign the radius of TaylorMLP turns it into a linear model. (Overparametrised, but behaves like a linear model).
# - This benefit works during OOD generalization as well, the model remains linear.
# - Turn this into a Energy-based model, by adding a context vector for each data point. We do Taylor expansion around the context. During inference, we can optimize the context to minimize energy.
# - y*,c* = argmin E_{\theta}(x,y,c) during training and infence.

#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.set(style="whitegrid")
from pathlib import Path
import time
import shutil
import json
import datetime
from typing import List, Tuple

# --- UTILS (Recreating necessary logic from your snippets) ---

def get_run_path(base_dir="experiments"):
    """Creates a unique run directory based on timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_path = Path(base_dir) / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path

def save_run(run_dir, model, model_name, config, history):
    """Saves model weights, config, and history."""
    # Save Model
    model_path = run_dir / f"{model_name}.eqx"
    eqx.tree_serialise_leaves(model_path, model)
    
    # Save Config & History
    with open(run_dir / f"{model_name}_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Save History as Numpy
    np.savez(run_dir / f"{model_name}_history.npz", **history)

def load_run(run_dir, model_skeleton, model_name):
    """Loads model weights and history."""
    model_path = run_dir / f"{model_name}.eqx"
    model = eqx.tree_deserialise_leaves(model_path, model_skeleton)
    
    history_path = run_dir / f"{model_name}_history.npz"
    history = np.load(history_path)
    return model, {k: history[k] for k in history.files}

# --- DATA GENERATION ---

def gen_data(seed, n_samples, n_segments=3, local_structure="random", 
             x_range=[-1, 1], slope=2.0, base_intercept=0.0, 
             step_size=2.0, custom_func=None, noise_std=0.5):
    """
    Generates 1D locally linear data (n_samples, 2).
    
    Parameters:
    - local_structure: 'random', 'constant', 'gradual_increase', 'gradual_decrease', 'custom'
    - x_range: list or tuple [x_min, x_max]
    - slope: The constant slope 'a' for all segments
    - base_intercept: Starting intercept for non-random structures
    - step_size: Amount to change intercept by for gradual structures
    - custom_func: Lambda taking (x_center) -> intercept offset (used if structure is 'custom')
    """
    np.random.seed(seed)
    
    # Divide total range into segment boundaries
    x_min, x_max = x_range
    segment_boundaries = np.linspace(x_min, x_max, n_segments + 1)
    
    # Samples per segment (distribute as evenly as possible)
    samples_per_seg = [n_samples // n_segments + (1 if i < n_samples % n_segments else 0) 
                       for i in range(n_segments)]
    
    all_x = []
    all_y = []
    segment_ids = []
    
    current_intercept = base_intercept
    
    for i in range(n_segments):
        # Define x range for this segment
        seg_x_min = segment_boundaries[i]
        seg_x_max = segment_boundaries[i+1]
        
        # Generate X uniformly in this segment
        n_seg_samples = samples_per_seg[i]
        x_seg = np.random.uniform(seg_x_min, seg_x_max, n_seg_samples)
        
        # Determine Intercept (b) based on structure
        b = 0
        if local_structure == "constant":
            b = base_intercept
        elif local_structure == "random":
            # Random b between -5 and 5 relative to base
            b = np.random.uniform(-5, 5) 
        elif local_structure == "gradual_increase":
            b = base_intercept + (i * step_size)
        elif local_structure == "gradual_decrease":
            b = base_intercept - (i * step_size)
        elif local_structure == "custom":
            if custom_func is None:
                raise ValueError("custom_func must be provided for 'custom' structure")
            # Calculate segment center to determine b via lambda
            x_center = (seg_x_min + seg_x_max) / 2
            b = custom_func(x_center)
            
        # Calculate Y = ax + b + noise
        noise = np.random.normal(0, noise_std, n_seg_samples)
        y_seg = (slope * x_seg) + b + noise
        
        all_x.append(x_seg)
        all_y.append(y_seg)
        segment_ids.append(np.full(n_seg_samples, i))
        
    # Concatenate all segments
    X = np.concatenate(all_x)
    Y = np.concatenate(all_y)
    Segments = np.concatenate(segment_ids)
    
    # Shuffle within the arrays to simulate real data collection (optional, keeping ordered here for viz logic)
    # Combining into (n_samples, 2)
    data = np.column_stack((X, Y))
    
    return data, Segments

def plot_datasets(train_data, train_segs, test_data, test_segs):
    plt.figure(figsize=(10, 6))
    
    # Helper to plot segments with shades
    def plot_segments(data, segs, cmap_name, label_prefix):
        unique_segs = np.unique(segs)
        cmap = cm.get_cmap(cmap_name)
        
        # We normalize shades so they are distinct
        norm_range = np.linspace(0.4, 0.9, len(unique_segs))
        
        for idx, seg_id in enumerate(unique_segs):
            mask = segs == seg_id
            color = cmap(norm_range[idx])
            
            plt.scatter(data[mask, 0], data[mask, 1], 
                        color=color, 
                        label=f"{label_prefix} Seg {int(seg_id)}", 
                        alpha=0.8, edgecolors='w', s=60)

    # Plot Training Data (Blues)
    plot_segments(train_data, train_segs, 'Blues', 'Train')
    
    # Plot Testing Data (Oranges) - Different X Range
    plot_segments(test_data, test_segs, 'Oranges', 'Test')
    
    plt.title("Locally Linear Data: Training vs Testing")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

class SimpleDataHandler:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(x))
    
    def get_iterator(self):
        np.random.shuffle(self.indices)
        for start_idx in range(0, len(self.x), self.batch_size):
            batch_idx = self.indices[start_idx:start_idx + self.batch_size]
            yield self.x[batch_idx], self.y[batch_idx]

    def get_full_data(self):
        return self.x, self.y

# --- MODELS ---

class LinearModel(eqx.Module):
    layer: eqx.nn.Linear

    def __init__(self, key):
        self.layer = eqx.nn.Linear(1, 1, key=key)

    def __call__(self, x):
        return self.layer(x)

class MLPModel(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(1, 64, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(64, 64, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(64, 1, key=key3)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class TaylorMLP(eqx.Module):
    layers: list
    radius: float

    def __init__(self, key, input_dim=1, hidden_dim=64, output_dim=1, radius=0.1):
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(input_dim, hidden_dim, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(hidden_dim, hidden_dim, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(hidden_dim, output_dim, key=key3)
        ]
        self.radius = radius

    def _base_forward(self, x):
        """Standard MLP forward pass for a single point."""
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x, key=None):
        """
        Forward pass with Taylor Expansion.
        
        Args:
            x: Input batch of shape (Batch, Dim)
            key: JAX PRNGKey. If provided, applies random Taylor expansion.
                 If None, falls back to standard MLP behavior (x0 = x).
        """
        # If no key is provided (e.g., during simple inference), standard forward
        if key is None:
            return jax.vmap(self._base_forward)(x)

        # 1. Sample x0 (Random neighbor within radius)
        # x0 = x + epsilon, where epsilon ~ Uniform(-radius, radius)
        epsilon = jax.random.uniform(key, x.shape, minval=-self.radius, maxval=self.radius)
        x0 = x + epsilon

        # 2. Define the Taylor Expansion logic for a single point
        def taylor_approx(xi, x0i):
            # We want: f(x) ≈ f(x0) + J(x0) @ (x - x0)
            # eqx.filter_jvp computes f(primal) and J(primal) @ tangent
            # Here: primal = x0i, tangent = (xi - x0i)
            
            f_x0, df_x0_dx_diff = eqx.filter_jvp(self._base_forward, (x0i,), (xi - x0i,))
            return f_x0 + df_x0_dx_diff

        # 3. Apply over the batch
        return jax.vmap(taylor_approx)(x, x0)

# --- TRAINING CONFIGURATION ---
TRAIN = True
RUN_DIR = ""  # Leave empty for new run
CONFIG = {
    "lr": 0.01,
    "batch_size": 32,
    "epochs": 1000,
    "seed": time.time_ns() % 2**32,
    "noise_std": 0.5
}

# 1. Data config
print("--- Generating Data ---")
SEED = CONFIG["seed"]
# Training: Gradual Increase
# --- Configuration ---
TOTAL_SAMPLES = 200
SLOPE = 0.5
NOISE_STD = 0.015
TRAIN_SEG_IDS = [2, 3, 4, 5, 6]  # Segments used for training

# 1. Data Generation
# Structure: "gradual_increase" -> b increases every segment
data, segs = gen_data(
    seed=SEED,
    n_samples=TOTAL_SAMPLES,
    n_segments=9,
    local_structure="gradual_increase",
    x_range=[-1.5, 1.5],
    slope=SLOPE,
    base_intercept=-0.4,
    step_size=0.1,
    noise_std=NOISE_STD
)

## Test segments are those not in TRAIN_SEG_IDS
test_data = data[~np.isin(segs, TRAIN_SEG_IDS)]
test_segs = segs[~np.isin(segs, TRAIN_SEG_IDS)]
train_data = data[np.isin(segs, TRAIN_SEG_IDS)]
train_segs = segs[np.isin(segs, TRAIN_SEG_IDS)]


# --- Plotting ---

print(train_segs, test_segs.shape, train_data.shape, test_data.shape)

plot_datasets(train_data, train_segs, test_data, test_segs)

# Prepare JAX Arrays (Features X need to be column vectors (N, 1))
X_train = jnp.array(train_data[:, 0])[:, None]
Y_train = jnp.array(train_data[:, 1])[:, None]
X_test = jnp.array(test_data[:, 0])[:, None]
Y_test = jnp.array(test_data[:, 1])[:, None]

dm = SimpleDataHandler(X_train, Y_train, CONFIG["batch_size"])


#%% [Running Training & Analysis]

# --- MAIN EXECUTION ---

# Define Model List skeleton
key = jax.random.PRNGKey(SEED)
models_config = [
    ("Linear", LinearModel(key)),
    ("MLP", MLPModel(key)),
    ("TaylorMLP", TaylorMLP(key, radius=1.05))
]
# 2. Train or Load
if TRAIN:
    run_dir = get_run_path()
    print(f"🚀 Starting Training. Run ID: {run_dir.name}")
    
    trained_models = [] # Store (name, model, history)
    
    # Main training key
    train_key = jax.random.PRNGKey(CONFIG["seed"])

    for name, model in models_config:
        print(f"\nTraining {name}...")
        optimizer = optax.adam(CONFIG["lr"])
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_value_and_grad
        def compute_loss(model, x, y, key):
            if isinstance(model, TaylorMLP):
                pred = model(x, key=key)
            else:
                # Fallback for models that don't accept/need a key
                pred = jax.vmap(model)(x)
                
            return jnp.mean((pred - y) ** 2)

        @eqx.filter_jit
        def make_step(model, opt_state, x, y, key):
            loss, grads = compute_loss(model, x, y, key)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        loss_history = []
        
        start_t = time.time()
        for epoch in range(CONFIG["epochs"]):
            batch_losses = []
            for bx, by in dm.get_iterator():
                # Split key for every step to ensure randomness
                train_key, step_key = jax.random.split(train_key)
                
                model, opt_state, loss = make_step(model, opt_state, bx, by, step_key)
                batch_losses.append(loss)
            loss_history.append(np.mean(batch_losses))
        
        print(f"✅ {name} finished in {time.time()-start_t:.2f}s | Final Loss: {loss_history[-1]:.4f}")
        
        # Save
        history_dict = {"train_loss": loss_history}
        save_run(run_dir, model, name, CONFIG, history_dict)
        trained_models.append((name, model, history_dict))

else:
    # LOAD MODE
    if RUN_DIR == "": raise ValueError("Provide a RUN_DIR to load from.")
    run_dir = Path(RUN_DIR)
    print(f"📂 Loading from: {run_dir}")
    
    trained_models = []
    for name, skeleton in models_config:
        try:
            m, h = load_run(run_dir, skeleton, name)
            trained_models.append((name, m, h))
            print(f"Loaded {name}")
        except FileNotFoundError:
            print(f"Could not find model file for {name}")

#%%
# --- ANALYSIS & PLOTTING ---
colors = ['darkred', 'darkgreen', 'magenta'] # Distinct colors for models

# 1. Loss Curves
plt.figure(figsize=(10, 5))
for i, (name, _, history) in enumerate(trained_models):
    plt.plot(history['train_loss'], label=f"{name} Train Loss", linewidth=2, color=colors[i])
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.yscale('log')
plt.title("Training Loss Curves")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(run_dir / "loss_curves.png")
plt.show()

# 2. Predictions vs Ground Truth
plt.figure(figsize=(12, 8))


# Calculate Test Metrics and Plot Fits
x_grid = jnp.linspace(-1.5, 1.5, 500)[:, None] # Covers both train and test range

for i, (name, model, _) in enumerate(trained_models):
    # TaylorMLP is batch-native and accepts 'key'.
    # Standard models (Linear, MLP) are single-point native and do not accept 'key'.
    if isinstance(model, TaylorMLP):
        # Try batch call with key (TaylorMLP path)
        y_grid = model(x_grid, key=None)
        y_test_pred = model(X_test, key=None)
    else:
        # Fallback: vmap over single-point models (Linear/MLP path)
        y_grid = jax.vmap(model)(x_grid)
        y_test_pred = jax.vmap(model)(X_test)
    
    test_mse = jnp.mean((y_test_pred - Y_test)**2)
        
    plt.plot(x_grid, y_grid, color=colors[i], linewidth=4, 
            label=f"{name} (Test MSE: {test_mse:.6f})")

# Plot Data
# Train Data (Blue shades)
for seg in np.unique(train_segs):
    mask = train_segs == seg
    plt.scatter(train_data[mask, 0], train_data[mask, 1], 
                color=plt.cm.Blues(0.5 + seg*0.1), alpha=0.4, label=f"Train Seg {int(seg)}", s=10)

# Test Data (Orange shades)
for seg in np.unique(test_segs):
    mask = test_segs == seg
    plt.scatter(test_data[mask, 0], test_data[mask, 1], 
                color=plt.cm.Oranges(0.5 + seg*0.1), alpha=0.4, label=f"Test Seg {int(seg)}", s=10)

plt.title("Model Predictions vs Locally Linear Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(run_dir / "predictions.png")
plt.show()

print(f"\nPlots saved to {run_dir}")
