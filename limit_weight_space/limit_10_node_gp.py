#%%
import jax
import jax.numpy as jnp
import jax.scipy as jsp
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
from typing import Optional, Tuple

# Set plotting style
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

# --- 1. CONFIGURATION ---
TRAIN = True  
RUN_DIR = "." 

CONFIG = {
    "seed": 2026,
    
    # Data Hyperparameters (Must match previous setup)
    "data_samples": 2000,
    "noise_std": 0.005,
    "segments": 11,
    "x_range": [-1.5, 1.5],
    "train_seg_ids": [2, 3, 4, 5, 6, 7, 8], 
    
    # GP Hyperparameters
    "gp_lr": 0.05,            # Learning rate for hyperparameters
    "gp_epochs": 1000,        # Number of optimization steps
    "print_every": 100,
    
    # Initial guesses (Log space)
    "init_lengthscale": 0.5,
    "init_variance": 1.0,
    "init_noise": 0.05
}

print("Config seed is:", CONFIG["seed"])

#%%
# --- 2. UTILITY FUNCTIONS ---

def setup_run_dir(base_dir="experiments"):
    if TRAIN:
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        run_path = Path(base_dir) / timestamp
        run_path.mkdir(parents=True, exist_ok=True)
        
        (run_path / "artefacts").mkdir(exist_ok=True)
        (run_path / "plots").mkdir(exist_ok=True)
        return run_path
    else:
        return Path(RUN_DIR)

#%%
# --- 3. DATA GENERATION (UNCHANGED) ---

def gen_data(seed, n_samples, n_segments=3, x_range=[-3.0, 3.0], noise_std=0.1):
    """
    Generates classical 1D-1D regression dataset: y = sin(10x) + 0.5x
    """
    np.random.seed(seed)
    
    x_min, x_max = x_range
    X = np.random.uniform(x_min, x_max, n_samples)
    
    # y = sin(10x) + 0.5x
    Y = np.sin(10 * X) + 0.5 * X
    
    noise = np.random.normal(0, noise_std, n_samples)
    Y += noise
    
    bins = np.linspace(x_min, x_max, n_segments + 1)
    segs = np.digitize(X, bins) - 1
    segs = np.clip(segs, 0, n_segments - 1)
    
    data = np.column_stack((X, Y))
    return data, segs

run_path = setup_run_dir()
artefacts_path = run_path / "artefacts"
plots_path = run_path / "plots"

if TRAIN:
    SEED = CONFIG["seed"]
    data, segs = gen_data(SEED, CONFIG["data_samples"], CONFIG["segments"], CONFIG["x_range"], CONFIG["noise_std"])

    train_mask = np.isin(segs, CONFIG["train_seg_ids"])
    test_mask = ~train_mask

    # GP usually expects shape (N, D), here D=1
    X_train = jnp.array(data[train_mask, 0])[:, None]
    Y_train = jnp.array(data[train_mask, 1])[:, None]
    X_test = jnp.array(data[test_mask, 0])[:, None]
    Y_test = jnp.array(data[test_mask, 1])[:, None]
    
    # Sort for cleaner plotting later
    sort_idx = jnp.argsort(X_train.flatten())
    X_train = X_train[sort_idx]
    Y_train = Y_train[sort_idx]
    
else:
    # Minimal load logic for inference only
    pass

print(f"Training Samples: {X_train.shape[0]}")
print(f"Test Samples: {X_test.shape[0]}")

#%%
# --- 4. GAUSSIAN PROCESS MODEL ---

class ExactGP(eqx.Module):
    """
    An Exact Gaussian Process model using an RBF Kernel.
    Parameters are stored in log-space to ensure positivity.
    """
    log_lengthscale: jax.Array
    log_signal_var: jax.Array
    log_noise_var: jax.Array

    def __init__(self):
        # Initialize with reasonable defaults or from config
        self.log_lengthscale = jnp.log(jnp.array(CONFIG["init_lengthscale"]))
        self.log_signal_var = jnp.log(jnp.array(CONFIG["init_variance"]))
        self.log_noise_var = jnp.log(jnp.array(CONFIG["init_noise"]))

    @property
    def lengthscale(self): return jnp.exp(self.log_lengthscale)
    
    @property
    def signal_var(self): return jnp.exp(self.log_signal_var)
    
    @property
    def noise_var(self): return jnp.exp(self.log_noise_var)

    def kernel_matrix(self, X1, X2):
        # RBF Kernel: k(x, x') = σ² * exp( - ||x - x'||² / (2ℓ²) )
        # Squared Euclidean distance
        diffs = X1[:, None, :] - X2[None, :, :]
        dist_sq = jnp.sum(diffs**2, axis=-1)
        
        K = self.signal_var * jnp.exp(-0.5 * dist_sq / (self.lengthscale**2))
        return K

    def negative_log_marginal_likelihood(self, X, y):
        N = X.shape[0]
        K = self.kernel_matrix(X, X)
        # Add noise to diagonal
        K = K + (self.noise_var + 1e-6) * jnp.eye(N)
        
        # Cholesky Decomposition: K = L L^T
        L = jsp.linalg.cholesky(K, lower=True)
        
        # Alpha = K⁻¹ y (solved via L)
        # Solve L y_temp = y
        alpha_temp = jsp.linalg.solve_triangular(L, y, lower=True)
        # Solve L^T alpha = alpha_temp
        alpha = jsp.linalg.solve_triangular(L.T, alpha_temp, lower=False)
        
        # NLML = 0.5 * y^T * K⁻¹ * y + 0.5 * log|K| + const
        data_fit = 0.5 * jnp.dot(y.T, alpha)[0, 0]
        complexity_penalty = jnp.sum(jnp.log(jnp.diag(L)))
        constant = 0.5 * N * jnp.log(2 * jnp.pi)
        
        return data_fit + complexity_penalty + constant

    def predict(self, X_train, Y_train, X_test):
        # 1. Compute Kernel Matrices
        K_xx = self.kernel_matrix(X_train, X_train) + (self.noise_var + 1e-6) * jnp.eye(X_train.shape[0])
        K_ss = self.kernel_matrix(X_test, X_test)
        K_sx = self.kernel_matrix(X_test, X_train)

        # 2. Compute Cholesky of Training Kernel
        L = jsp.linalg.cholesky(K_xx, lower=True)
        
        # 3. Compute Mean: μ = K_sx K_xx⁻¹ Y
        alpha = jsp.linalg.cho_solve((L, True), Y_train)
        mu = K_sx @ alpha
        
        # 4. Compute Covariance: Σ = K_ss - K_sx K_xx⁻¹ K_xs
        # V = L⁻¹ K_xs  (so V^T V = K_sx K_xx⁻¹ K_xs)
        v = jsp.linalg.solve_triangular(L, K_sx.T, lower=True)
        cov = K_ss - v.T @ v
        
        # Extract variance (diagonal)
        var = jnp.diag(cov)
        return mu, var

#%%
# --- 5. INITIALIZATION & TRAINING ---

model = ExactGP()
print(f"Initial Params | l: {model.lengthscale:.3f}, var: {model.signal_var:.3f}, noise: {model.noise_var:.3f}")

# Using Adam to optimize hyperparameters
optimizer = optax.adam(learning_rate=CONFIG["gp_lr"])
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def train_step(model, opt_state, X, Y):
    loss, grads = eqx.filter_value_and_grad(lambda m: m.negative_log_marginal_likelihood(X, Y))(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

print("\n=== Starting GP Hyperparameter Optimization ===")
loss_history = []

start_time = time.time()
for epoch in range(CONFIG["gp_epochs"]):
    model, opt_state, loss = train_step(model, opt_state, X_train, Y_train)
    loss_history.append(loss)
    
    if (epoch + 1) % CONFIG["print_every"] == 0:
        print(f"Epoch {epoch+1:4d} | NLML: {loss.item():.4f} | "
              f"L_scale: {model.lengthscale.item():.3f} | "
              f"Sig_Var: {model.signal_var.item():.3f} | "
              f"Noise: {model.noise_var.item():.4f}")

print(f"Optimization finished in {time.time() - start_time:.2f}s")

#%%
# --- 6. VISUALIZATION ---

print("\n=== Generating Dashboards ===")

# Grid for plotting smooth predictions
x_grid = jnp.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], 400)[:, None]

# Get Predictions
mu_grid, var_grid = model.predict(X_train, Y_train, x_grid)
std_grid = jnp.sqrt(var_grid)

# Also predict on Test set for metrics (if needed)
mu_test, var_test = model.predict(X_train, Y_train, X_test)

# --- PLOT 1: LOSS DASHBOARD ---
fig_loss = plt.figure(figsize=(10, 5))
plt.plot(loss_history, linewidth=2, color='teal')
plt.title("Negative Log Marginal Likelihood (Training)")
plt.xlabel("Optimization Steps")
plt.ylabel("NLML Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "gp_loss.png")
plt.show()


# --- PLOT 2: GP PREDICTION WITH UNCERTAINTY ---
fig, ax = plt.subplots(figsize=(12, 7))

# 1. Plot Training Data
ax.scatter(X_train, Y_train, c='black', s=10, alpha=0.5, label="Train Data")

# 2. Plot Test Data (Ground Truth)
ax.scatter(X_test, Y_test, c='red', s=10, marker='x', alpha=0.5, label="Test Data")

# 3. Plot GP Mean
ax.plot(x_grid, mu_grid, color='blue', linewidth=2, label="GP Mean")

# 4. Plot Uncertainty (Shaded Area - 2 Sigma / 95% Confidence)
uncertainty_upper = mu_grid.flatten() + 1.96 * std_grid
uncertainty_lower = mu_grid.flatten() - 1.96 * std_grid

ax.fill_between(
    x_grid.flatten(), 
    uncertainty_lower, 
    uncertainty_upper, 
    color='blue', 
    alpha=0.2, 
    label="95% Confidence ($2\sigma$)"
)

ax.set_title(f"Gaussian Process Fit\nLengthscale: {model.lengthscale:.3f}, Signal Var: {model.signal_var:.3f}, Noise: {model.noise_var:.4f}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Highlight training region bounds (Segments 2-8 imply spatial bounds)
train_min = jnp.min(X_train)
train_max = jnp.max(X_train)
ax.axvline(train_min, color='green', linestyle='--', alpha=0.5)
ax.axvline(train_max, color='green', linestyle='--', alpha=0.5)
ax.text(train_min, ax.get_ylim()[1]*0.9, " Train Start ", ha='right', color='green')
ax.text(train_max, ax.get_ylim()[1]*0.9, " Train End ", ha='left', color='green')

plt.tight_layout()
plt.savefig(plots_path / "gp_prediction_dashboard.png")
plt.show()

print(f"Final trained artifacts saved to {plots_path}")
