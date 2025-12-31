#%%
#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.set(style="white")
from pathlib import Path
import time
import json
import datetime
from typing import List, Tuple, Optional

# --- 1. CONFIGURATION ---
TRAIN = True
RUN_DIR = "" 
CONFIG = {
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "inner_gd_steps": 50,
    "width_size": 64,
    "seed": 42,
    # "seed": time.time_ns() % (2**32 - 1),
    "noise_std": 0.015,
    "context_dim": 1,  # New hyperparameter
    "data_samples": 200,
    "segments": 9,
    "x_range": [-1.5, 1.5]
}

#%%
# --- 2. UTILITY FUNCTIONS ---

def get_run_path(base_dir="experiments"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_path = Path(base_dir) / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path

def save_run(run_dir, model, model_name, config, history):
    model_path = run_dir / f"{model_name}.eqx"
    eqx.tree_serialise_leaves(model_path, model)
    with open(run_dir / f"{model_name}_config.json", "w") as f:
        json.dump(config, f, indent=4)
    np.savez(run_dir / f"{model_name}_history.npz", **history)

def load_run(run_dir, model_skeleton, model_name):
    model_path = run_dir / f"{model_name}.eqx"
    model = eqx.tree_deserialise_leaves(model_path, model_skeleton)
    history_path = run_dir / f"{model_name}_history.npz"
    history = np.load(history_path)
    return model, {k: history[k] for k in history.files}

#%%
# --- 3. DATA GENERATION & PLOTTING ---

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
        elif local_structure == "custom":
            if custom_func is None: raise ValueError("Need custom_func")
            b = custom_func((seg_x_min + seg_x_max) / 2)
            
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
        np.random.shuffle(self.indices)
        for start_idx in range(0, len(self.x), self.batch_size):
            batch_idx = self.indices[start_idx:start_idx + self.batch_size]
            yield self.x[batch_idx], self.y[batch_idx], batch_idx 

# Generate Data
SEED = CONFIG["seed"]
TRAIN_SEG_IDS = [2, 3, 4, 5, 6]

data, segs = gen_data(SEED, CONFIG["data_samples"], n_segments=CONFIG["segments"], 
                      local_structure="gradual_increase", x_range=CONFIG["x_range"], 
                      slope=0.5, base_intercept=-0.4, step_size=0.1, noise_std=CONFIG["noise_std"])

test_mask = ~np.isin(segs, TRAIN_SEG_IDS)
train_mask = np.isin(segs, TRAIN_SEG_IDS)

train_data, train_segs = data[train_mask], segs[train_mask]
test_data, test_segs = data[test_mask], segs[test_mask]

X_train = jnp.array(train_data[:, 0])[:, None]
Y_train = jnp.array(train_data[:, 1])[:, None]
X_test = jnp.array(test_data[:, 0])[:, None]
Y_test = jnp.array(test_data[:, 1])[:, None]

dm = SimpleDataHandler(X_train, Y_train, CONFIG["batch_size"])

# Plot Data
plt.figure(figsize=(10, 6))
plt.scatter(train_data[:, 0], train_data[:, 1], c='blue', alpha=0.5, label='Train')
plt.scatter(test_data[:, 0], test_data[:, 1], c='orange', alpha=0.5, label='Test')
plt.title("Data Distribution")
plt.legend()
plt.show()

#%%
# --- 4. MODEL DEFINITIONS ---
inner_gd_steps = CONFIG["inner_gd_steps"]
width_size = CONFIG["width_size"]

class LinearModel(eqx.Module):
    layer: eqx.nn.Linear
    def __init__(self, key, context_dim=None): 
        self.layer = eqx.nn.Linear(1, 1, key=key)
    
    def __call__(self, x, c=None, key=None):
        return jax.vmap(self.layer)(x)
    
    def predict(self, x):
        return self(x, None, None)

class MLPModel(eqx.Module):
    layers: list
    def __init__(self, key, context_dim=None):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(1, 64, key=k1), jax.nn.relu,
                       eqx.nn.Linear(64, 64, key=k2), jax.nn.relu,
                       eqx.nn.Linear(64, 1, key=k3)]
    
    def _forward(self, x):
        for l in self.layers: x = l(x)
        return x

    def __call__(self, x, c=None, key=None):
        return jax.vmap(self._forward)(x)

    def predict(self, x):
        return self(x, None, None)

class TaylorMLP(eqx.Module):
    layers: list
    radius: float
    def __init__(self, key, radius=0.2, context_dim=None):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(1, 64, key=k1), jax.nn.relu,
                       eqx.nn.Linear(64, 64, key=k2), jax.nn.relu,
                       eqx.nn.Linear(64, 1, key=k3)]
        self.radius = radius
    
    def _forward(self, x):
        for l in self.layers: x = l(x)
        return x
    
    def __call__(self, x, c=None, key=None):
        if key is None: return self.predict(x)
        
        epsilon = jax.random.uniform(key, x.shape, minval=-self.radius, maxval=self.radius)
        x0 = x + epsilon
        
        def approx(xi, x0i):
            f, df = eqx.filter_jvp(self._forward, (x0i,), (xi - x0i,))
            return f + df
        return jax.vmap(approx)(x, x0)

    def predict(self, x):
        return jax.vmap(self._forward)(x)

# --- EBM UTILS & MODELS ---
class EBMUtils:
    @staticmethod
    def get_c0_batch(c_batch):
        """Finds closest neighbor context c0 for each c in batch (excluding self)."""
        # dists: (B, B)
        dists = jnp.linalg.norm(c_batch[:, None, :] - c_batch[None, :, :], axis=-1)
        dists = dists + jnp.eye(dists.shape[0]) * 1e9
        nearest_idx = jnp.argmin(dists, axis=1)
        return c_batch[nearest_idx]

class BaseEBM(eqx.Module):
    neuralnet: eqx.nn.MLP
    y_mean: float

    def __init__(self, key, y_mean=0.0, context_dim=None):
        # Input: x(1) + y(1) = 2
        self.neuralnet = eqx.nn.MLP(in_size=2, out_size=1, width_size=width_size, depth=3, key=key)
        self.y_mean = y_mean

    def energy(self, x, y):
        inp = jnp.concatenate([x, y], axis=0)
        return jnp.squeeze(self.neuralnet(inp))

    def _solve_y(self, x):
        def step(y_curr, _):
            grads = jax.grad(self.energy, argnums=1)(x, y_curr)
            y_next = y_curr - 0.1 * grads
            return y_next, None
        
        y_init = jnp.array([self.y_mean])
        y_final, _ = jax.lax.scan(step, y_init, None, length=inner_gd_steps)
        return y_final

    def __call__(self, x, c=None, key=None):
        return jax.vmap(self._solve_y)(x)

    def predict(self, x):
        return self(x)

class Embedding(eqx.Module):
    weights: eqx.nn.MLP
    def __init__(self, num_embeddings, embedding_dim, key):
        self.weights = jax.random.uniform(key, (num_embeddings, embedding_dim), minval=-0.1, maxval=0.1)
    def __call__(self, indices):
        return self.weights[indices]

class ContextEBM(eqx.Module):
    neuralnet: eqx.nn.MLP
    contexts: eqx.nn.Embedding
    y_mean: float
    c_dim: int

    def __init__(self, key, num_train_samples, y_mean=0.0, context_dim=1):
        k1, k2 = jax.random.split(key)
        self.c_dim = context_dim
        # Input: x(1) + y(1) + c(c_dim)
        self.neuralnet = eqx.nn.MLP(in_size=2 + context_dim, out_size=1, width_size=width_size, depth=3, key=k1)
        self.contexts = Embedding(num_train_samples, context_dim, key=k2)
        # Init contexts at 0
        # self.contexts = eqx.tree_at(lambda t: t.weight, self.contexts, jnp.zeros((num_train_samples, context_dim)))
        self.y_mean = y_mean

    def energy(self, x, y, c):
        inp = jnp.concatenate([x, y, c], axis=0)
        return jnp.squeeze(self.neuralnet(inp))

    def _solve(self, x, c_fixed=None):
        optimize_c = (c_fixed is None)
        c_init = c_fixed if not optimize_c else jnp.zeros((self.c_dim,))
        y_init = jnp.array([self.y_mean])

        def step(state, _):
            y_curr, c_curr = state
            gy, gc = jax.grad(self.energy, argnums=(1, 2))(x, y_curr, c_curr)
            y_next = y_curr - 0.1 * gy
            c_next = (c_curr - 0.1 * gc) if optimize_c else c_curr
            return (y_next, c_next), None

        (y_final, _), _ = jax.lax.scan(step, (y_init, c_init), None, length=inner_gd_steps)
        return y_final

    def __call__(self, x, c=None, key=None):
        # c is batch of contexts or None
        if c is not None:
            return jax.vmap(self._solve)(x, c)
        else:
            # Inference mode
            return jax.vmap(lambda xi: self._solve(xi, None))(x)

    def predict(self, x):
        return self(x, c=None)

class TaylorEBM(eqx.Module):
    neuralnet: eqx.nn.MLP
    contexts: eqx.nn.Embedding
    y_mean: float
    c_dim: int

    def __init__(self, key, num_train_samples, y_mean=0.0, context_dim=1):
        k1, k2 = jax.random.split(key)
        self.c_dim = context_dim
        self.neuralnet = eqx.nn.MLP(in_size=2 + context_dim, out_size=1, width_size=width_size, depth=3, key=k1)
        self.contexts = Embedding(num_train_samples, context_dim, key=k2)
        # self.contexts = eqx.tree_at(lambda t: t.weight, self.contexts, jnp.zeros((num_train_samples, context_dim)))
        self.y_mean = y_mean

    def _base_energy(self, c, x, y):
        inp = jnp.concatenate([x, y, c], axis=0)
        return jnp.squeeze(self.neuralnet(inp))

    def energy_taylor(self, x, y, c, c0):
        # Expansion wrt c, around c0
        E_c0, dE_dc0 = eqx.filter_jvp(lambda c_arg: self._base_energy(c_arg, x, y), (c0,), (c - c0,))
        return E_c0 + dE_dc0

    def _solve(self, x, c_fixed, c0_fixed):
        # Training Mode: Optimize Y using Taylor Energy. Fixed c.
        y_init = jnp.array([self.y_mean])
        
        def step(y_curr, _):
            # Gradient of Taylor Energy wrt Y
            def loss(y_arg): return self.energy_taylor(x, y_arg, c_fixed, c0_fixed)
            gy = jax.grad(loss)(y_curr)
            return y_curr - 0.1 * gy, None

        y_final, _ = jax.lax.scan(step, y_init, None, length=inner_gd_steps)
        return y_final

    def _solve_inference(self, x):
        # Inference Mode: Taylor Disabled. Optimize Y and C (init at 0) on Base Energy.
        c_init = jnp.zeros((self.c_dim,))
        y_init = jnp.array([self.y_mean])

        def step(state, _):
            y_curr, c_curr = state
            def loss(y_arg, c_arg): return self._base_energy(c_arg, x, y_arg)
            gy, gc = jax.grad(loss, argnums=(0, 1))(y_curr, c_curr)
            return (y_curr - 0.1 * gy, c_curr - 0.1 * gc), None

        (y_final, _), _ = jax.lax.scan(step, (y_init, c_init), None, length=inner_gd_steps)
        return y_final

    def __call__(self, x, c=None, key=None):
        if c is not None:
            # Training Mode
            c0_batch = EBMUtils.get_c0_batch(c)
            return jax.vmap(self._solve)(x, c, c0_batch)
        else:
            return self.predict(x)

    def predict(self, x):
        return jax.vmap(self._solve_inference)(x)

#%%
# --- 5. TRAINING LOOP ---

if TRAIN:
    run_dir = get_run_path()
    print(f"🚀 Starting Training. Run ID: {run_dir.name}")
    trained_models = [] 
    
    # Initialize Models
    key = jax.random.PRNGKey(CONFIG["seed"])
    y_mean = jnp.mean(Y_train)
    num_train = len(X_train)
    c_dim = CONFIG["context_dim"]

    models_config = [
        ("Linear", LinearModel(key)),
        ("MLP", MLPModel(key)),
        ("TaylorMLP", TaylorMLP(key, radius=0.2)),
        ("BaseEBM", BaseEBM(key, y_mean)),
        ("ContextEBM", ContextEBM(key, num_train, y_mean, c_dim)),
        ("TaylorEBM", TaylorEBM(key, num_train, y_mean, c_dim))
    ]
    
    train_key = jax.random.PRNGKey(CONFIG["seed"])

    for name, model in models_config:
        print(f"\nTraining {name}...")
        optimizer = optax.adam(CONFIG["lr"])
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        # --- CORRECTED LOSS FUNCTION ---
        @eqx.filter_value_and_grad
        def compute_loss(model, x, y, indices, key):
            # 1. Lookup contexts INSIDE the loss function
            # This ensures gradients flow back to model.contexts
            if hasattr(model, 'contexts'):
                c = jax.vmap(model.contexts)(indices)
            else:
                c = None
            
            # 2. Forward pass
            pred = model(x, c=c, key=key)
            return jnp.mean((pred - y) ** 2)

        @eqx.filter_jit
        def make_step(model, opt_state, x, y, indices, key):
            # We pass indices to compute_loss, not c
            loss, grads = compute_loss(model, x, y, indices, key)
            
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        loss_history = []
        start_t = time.time()
        
        for epoch in range(CONFIG["epochs"]):
            batch_losses = []
            for bx, by, b_idx in dm.get_iterator():
                train_key, step_key = jax.random.split(train_key)
                model, opt_state, loss = make_step(model, opt_state, bx, by, b_idx, step_key)
                batch_losses.append(loss)
            loss_history.append(np.mean(batch_losses))
            
            if epoch % 250 == 0:
                print(f"Ep {epoch} | Loss: {loss_history[-1]:.4f}")
        
        print(f"✅ {name} finished | Final Loss: {loss_history[-1]:.4f}")
        
        history_dict = {"train_loss": loss_history}
        save_run(run_dir, model, name, CONFIG, history_dict)
        trained_models.append((name, model, history_dict))

else:
    if RUN_DIR == "": raise ValueError("Provide RUN_DIR")
    run_dir = Path(RUN_DIR)
    trained_models = []
    # Re-init skeleton with correct config logic if needed
    # ... (Load logic assumes skeleton matches saved)

#%%
# --- 6. ANALYSIS & PLOTTING ---
colors = ['black', 'green', 'blue', 'orange', 'red', 'purple']

# 1. Loss Curves
plt.figure(figsize=(10, 5))
for i, (name, _, history) in enumerate(trained_models):
    plt.plot(history['train_loss'], label=name, linewidth=1, color=colors[i])
plt.yscale('log')
plt.title("Training Loss")
plt.legend()
plt.savefig(run_dir / "loss.png")
plt.show()

# 2. Predictions
plt.figure(figsize=(12, 8))

# Data Background
for seg in np.unique(train_segs):
    mask = train_segs == seg
    plt.scatter(train_data[mask, 0], train_data[mask, 1], c='lightblue', s=20, alpha=0.5)
for seg in np.unique(test_segs):
    mask = test_segs == seg
    plt.scatter(test_data[mask, 0], test_data[mask, 1], c='wheat', s=20, alpha=0.5)

x_grid = jnp.linspace(-1.5, 1.5, 300)[:, None]

for i, (name, model, _) in enumerate(trained_models):
    y_grid = model.predict(x_grid)
    y_test = model.predict(X_test)
        
    mse = jnp.mean((y_test - Y_test)**2)
    plt.plot(x_grid, y_grid, color=colors[i], linewidth=2.5, label=f"{name} (MSE: {mse:.4f})")

plt.legend()
plt.title("Model Predictions (Test MSE)")
plt.savefig(run_dir / "predictions.png")
plt.show()


#%%# 3. Segmented Plot For the Enbeddings
## For the context-based models only
# context_models = [m for m in trained_models if "EBM" in m[0]]   
## only context EBM and TaylorEBM
context_models = [m for m in trained_models if m[0] in ["ContextEBM", "TaylorEBM"]]
plt.figure(figsize=(12, 8))
for i, (name, model, _) in enumerate(context_models):
    # Get contexts for train data
    c_embeddings = jax.vmap(model.contexts)(jnp.arange(len(X_train)))
    plt.subplot(1, len(context_models), i + 1)
    for seg in np.unique(train_segs):
        mask = train_segs == seg
        plt.scatter(X_train[mask, 0], c_embeddings[mask, 0], 
                    color=plt.cm.Blues(0.5 + seg*0.1), alpha=0.6, label=f"Train Seg {int(seg)}", s=10)
    plt.title(f"{name} Context Embeddings")
    plt.xlabel("C1")
    plt.ylabel("C2")
    plt.legend()
plt.savefig(run_dir / "context_embeddings.png")
plt.show()

# print(c_embeddings)
