#%%
import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import equinox as eqx
import diffrax
import optax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import datetime
import shutil
import sys
import typing
from typing import Optional, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Set plotting style
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['lines.markersize'] = 10

# --- 1. CONFIGURATION ---
TRAIN = True  
RUN_DIR = "."

CONFIG = {
    "seed": 2028,
    "width_size": 24,
    "mlp_batch_size": 64,
    "n_circles": 40,           
    "warmup_steps": 0,
    "lr": 1e-4,      
    "transformer_epochs": 10000,
    "print_every": 1000,
    "transformer_batch_size": 1,      
    "transformer_d_model": 128,    
    "transformer_n_heads": 1,      
    "transformer_n_layers": 1,     
    "transformer_d_ff": 1,       
    "transformer_substeps": 50,     
    "kl_weight": 1e-2,          
    "transformer_target_step": 80,    
    "scheduled_loss_weight": False,
    "n_synthetic_points": 512,
    "consistency_loss_weight": 0.0,
    "regularization_step": 40,     
    "regularization_weight": 0.0,  
    "data_selection": "annulus",        
    "final_step_mode": "none",          
}

print("Config seed is:", CONFIG["seed"])

# --- 2. UTILITY FUNCTIONS ---

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
    else: arrays = []
    full_leaves = []
    idx = 0
    for is_array in is_array_mask:
        if is_array: full_leaves.append(arrays[idx]); idx += 1
        else: full_leaves.append(None)
    return jtree.tree_unflatten(tree_def, full_leaves)

# --- 3. DATA GENERATION ---

def gen_data_air():
    data_path = "air_quality.csv"
    if not os.path.exists(data_path):
        # Fallback synthetic if file missing (for robustness)
        print("Warning: 'air_quality.csv' not found. Using synthetic proxy.")
        np.random.seed(42); X = np.random.uniform(-2, 3, 1000); Y = np.sin(X) + np.random.normal(0, 0.1, 1000)
        segs = (X > 1.0).astype(int)
        return np.column_stack((X, Y)), segs

    df = pd.read_csv(data_path)
    try:
        if 'PT08.S3.NOx.' not in df.columns: df = pd.read_csv(data_path, sep=';', decimal=',')
    except: pass
    
    rename_map = {'PT08.S3(NOx)': 'PT08.S3.NOx.', 'PT08.S5(O3)': 'PT08.S5.O3.'}
    df.rename(columns=rename_map, inplace=True)
    df = df.dropna(subset=['PT08.S3.NOx.', 'PT08.S5.O3.'])

    scaler = StandardScaler()
    df[['PT08.S3.NOx.', 'PT08.S5.O3.']] = scaler.fit_transform(df[['PT08.S3.NOx.', 'PT08.S5.O3.']])

    X = df['PT08.S5.O3.'].values
    Y = df['PT08.S3.NOx.'].values
    segs = (X > 1.0).astype(int)
    data = np.column_stack((X, Y))
    return data, segs

if TRAIN:
    data, segs = gen_data_air()
    train_mask = (segs == 0)
    test_mask = (segs == 1)

    X_train_full = jnp.array(data[train_mask, 0])[:, None]
    Y_train_full = jnp.array(data[train_mask, 1])[:, None]
    X_test = jnp.array(data[test_mask, 0])[:, None]
    Y_test = jnp.array(data[test_mask, 1])[:, None]
    
    # Auto-configure radial expansion based on data extent
    x_mean = jnp.mean(X_train_full)
    dists_train = jnp.abs(X_train_full - x_mean).flatten()
    max_train_dist = jnp.max(dists_train)
    
    dists_test = jnp.abs(X_test - x_mean).flatten()
    max_test_dist = jnp.max(dists_test) if len(dists_test) > 0 else max_train_dist * 1.5
    
    radii = jnp.linspace(0.0, max_train_dist + 0.01, CONFIG["n_circles"])
    delta_radius = radii[1] - radii[0]
    
    total_dist_needed = max(max_test_dist, max_train_dist * 1.5)
    total_steps = int(jnp.ceil(total_dist_needed / delta_radius)) + 5
    
    fake_radii = jnp.arange(radii[-1], radii[-1] + (total_steps - CONFIG["n_circles"])*delta_radius + 0.01, delta_radius)
    all_radii = jnp.concatenate([radii, fake_radii])
    CONFIG["transformer_target_step"] = len(all_radii)
    
    circle_masks = jnp.stack([dists_train <= r for r in radii])

print(f"Data Center (Mean): {x_mean:.4f}")

# --- 4. MODEL DEFINITIONS ---

class MLPModel(eqx.Module):
    layers: list
    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layers = [eqx.nn.Linear(1, 1, use_bias=True, key=k1)]

    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x
    def predict(self, x):
        return jax.vmap(self)(x)

class LinearRNN(eqx.Module):
    A: jax.Array
    y_init: jax.Array 
    def __init__(self, data_dim, hidden_dim, key):
        problem_dim = data_dim*2
        self.A = jnp.eye(problem_dim)
        self.y_init = jax.random.normal(key, shape=(2, problem_dim)) * 1e-4

    def __call__(self, y0, steps, key):
        init_state = (self.y_init[0], self.y_init[1])
        def scan_fn(state, _):
            y_prev1, _ = state
            y_next = self.A @ y_prev1
            return (y_next, y_prev1), y_next
        _, ys = jax.lax.scan(scan_fn, init_state, None, length=steps)
        return ys

# --- 5. INITIALIZATION ---

key = jax.random.PRNGKey(CONFIG["seed"])
k_init, k_tf, key = jax.random.split(key, 3)

model_template = MLPModel(k_init)
params_template, static = eqx.partition(model_template, eqx.is_array)
flat_template, shapes, treedef, mask = flatten_pytree(params_template)
input_dim = flat_template.shape[0]

tf_model = LinearRNN(data_dim=input_dim, hidden_dim=CONFIG["transformer_d_model"], key=k_tf)
opt = optax.adabelief(CONFIG["lr"])
opt_state = opt.init(eqx.filter(tf_model, eqx.is_array))

# Init Batch
x0_batch = jax.random.normal(k_init, (CONFIG["transformer_batch_size"], input_dim)) * 1e-2

# --- 6. LOSS FUNCTIONS ---

def negative_log_likelihood(theta_mu, theta_sigma, X, Y, noise_sigma=1e-3):
    def model_fn(theta, x):
        params = unflatten_pytree(theta, shapes, treedef, mask)
        return eqx.combine(params, static).predict(x)

    mean_y = model_fn(theta_mu, X)
    jacobian = jax.jacfwd(model_fn)(theta_mu, X)
    
    scaled_J = jacobian * theta_sigma[None, None, :]
    cov_y = jnp.einsum('npd,nqd->npq', scaled_J, scaled_J)
    cov_y = cov_y + jnp.eye(mean_y.shape[-1]) * (noise_sigma ** 2)

    if Y is None: return None, mean_y, cov_y

    diff = Y - mean_y
    L = jnp.linalg.cholesky(cov_y)
    z = jax.vmap(lambda Li, di: jax.lax.linalg.triangular_solve(Li, di[:,None], lower=True, left_side=True))(L, diff)
    quad_form = 0.5 * jnp.sum(jnp.squeeze(z)**2, axis=-1)
    log_det = jnp.sum(jnp.log(jnp.diagonal(L, axis1=1, axis2=2)), axis=-1)
    
    return quad_form + log_det, mean_y, cov_y

def compute_output_kl(pred_mean, pred_cov, prior_mean=0.0, prior_sigma=3.0):
    N, D = pred_mean.shape
    prior_var = prior_sigma ** 2
    tr_term = jnp.trace(pred_cov, axis1=-2, axis2=-1) / prior_var
    diff = pred_mean - prior_mean
    quad_term = jnp.sum(diff**2, axis=-1) / prior_var
    log_det_prior = D * jnp.log(prior_var)
    _, log_det_pred = jnp.linalg.slogdet(pred_cov)
    log_det_term = log_det_prior - log_det_pred
    kl = 0.5 * (tr_term + quad_term + log_det_term - D)
    return jnp.mean(kl)

def get_functional_loss(flat_w, step_idx, key=None):
    means_theta, sigmas_theta = flat_w[:input_dim], flat_w[input_dim:2*input_dim]
    
    theta = jax.random.normal(key, shape=sigmas_theta.shape) * sigmas_theta + means_theta 
    params = unflatten_pytree(theta, shapes, treedef, mask)
    model = eqx.combine(params, static)
    residuals = (model.predict(X_train_full) - Y_train_full) ** 2

    _, mean_y, cov_y = negative_log_likelihood(means_theta, sigmas_theta, X_train_full, Y=None)
    kl_div = compute_output_kl(mean_y, cov_y, prior_mean=jnp.mean(Y_train_full), prior_sigma=3.0)

    # Masking Logic
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

    active_mask = jax.lax.select(is_circle_phase, phase_mask, jnp.zeros_like(phase_mask))
    n_active = jnp.sum(active_mask)

    def disable_to_32(active_mask, key):
        noise = jax.random.uniform(key, shape=active_mask.shape)
        scores = jnp.where(active_mask, noise, -jnp.inf)
        _, keep_indices = jax.lax.top_k(scores, CONFIG["mlp_batch_size"])
        new_mask = jnp.zeros_like(active_mask, dtype=jnp.bool_).at[keep_indices].set(True)
        return new_mask

    active_mask = jax.lax.cond(n_active > CONFIG["mlp_batch_size"], disable_to_32, lambda m, k: m, active_mask, key)
    mask_sum = jnp.sum(active_mask)
    base_loss = jnp.sum(residuals * active_mask[:, None]) / (mask_sum + 1e-6)
    return base_loss + CONFIG["kl_weight"] * kl_div

@eqx.filter_value_and_grad
def train_step_fn(model, x0_batch, key):
    total_steps = CONFIG["transformer_target_step"]
    keys = jax.random.split(key, x0_batch.shape[0])
    preds_batch = jax.vmap(model, in_axes=(0, None, 0))(x0_batch, total_steps, keys)

    step_indices = jnp.arange(CONFIG["n_circles"])
    preds_batch_data = preds_batch[:, step_indices, :]

    keys = jax.random.split(key, len(step_indices))
    def loss_per_seq(seq):
        return jax.vmap(get_functional_loss)(seq, step_indices, keys)
    losses_batch = jax.vmap(loss_per_seq)(preds_batch_data)
    total_loss = jnp.mean(jnp.mean(losses_batch, axis=1))
    return total_loss

@eqx.filter_jit
def make_step(model, opt_state, x0_batch, key):
    loss, grads = train_step_fn(model, x0_batch, key)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

# --- 7. TRAINING ---

if TRAIN:
    print(f"🚀 Starting Batch Transformer Training.")
    train_key = jax.random.PRNGKey(CONFIG["seed"] + 99)
    loss_history = []
    
    for ep in range(CONFIG["transformer_epochs"]):
        train_key, step_key = jax.random.split(train_key)
        
        # Perturb initial condition slightly for robustness
        x0_noisy = x0_batch + jax.random.normal(step_key, x0_batch.shape) * 1e-3
        
        tf_model, opt_state, loss = make_step(tf_model, opt_state, x0_noisy, step_key)
        loss_history.append(loss)

        if (ep+1) % CONFIG["print_every"] == 0:
            print(f"Epoch {ep+1} | Loss: {loss:.6f}")

    eval_key = jax.random.PRNGKey(42)
    final_batch_traj = jax.vmap(tf_model, in_axes=(0, None, None))(x0_batch, CONFIG["transformer_target_step"], eval_key)

# --- 8. INFERENCE & PLOTTING ---

print("Generating Prediction Plots...")

def predict_circle_uncertainty(final_batch_traj, X_data, Y_data=None):
    circle_stats = {}
    radii_arr = jnp.array(all_radii)
    dists = jnp.abs(X_data - x_mean).flatten()

    for circle_idx in range(CONFIG["transformer_target_step"]):
        r_outer = radii_arr[circle_idx]
        r_inner = radii_arr[circle_idx-1] if circle_idx > 0 else -1.0
        
        mask = (dists <= r_outer) & (dists > r_inner)
        X_ring = X_data[mask]
        
        if len(X_ring) == 0: continue

        w = final_batch_traj[0, circle_idx, :] 
        theta_mean, theta_std = w[:input_dim], w[input_dim:2*input_dim]
        _, y_mean, cov_y_pred = negative_log_likelihood(theta_mean, theta_std, X_ring, Y=None)
        y_std = jnp.sqrt(jnp.diagonal(cov_y_pred, axis1=1, axis2=2))[:, 0]
        
        Y_ring = Y_data[mask] if Y_data is not None else None
        circle_stats[circle_idx] = (X_ring, Y_ring, y_mean, y_std)

    return circle_stats

train_stats = predict_circle_uncertainty(final_batch_traj, X_train_full, Y_train_full)
test_stats = predict_circle_uncertainty(final_batch_traj, X_test, Y_test)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot Truth
ax.scatter(X_train_full, Y_train_full, c='k', s=10, alpha=0.1, label="True Data")
ax.scatter(X_test, Y_test, c='k', s=10, alpha=0.1)

def plot_bands(stats_dict, color_mean, color_band, label_prefix):
    added_label = False
    total_mse = 0.
    for _, (X_seg, Y_seg, y_mu, y_sigma) in stats_dict.items():
        sort_indices = jnp.argsort(X_seg.flatten())
        X_sorted = X_seg[sort_indices].flatten()
        mu_sorted = y_mu[sort_indices].flatten()
        sigma_sorted = y_sigma[sort_indices].flatten()

        if Y_seg is not None:
             total_mse += jnp.mean((mu_sorted - Y_seg[sort_indices].flatten())**2)

        lbl = f"Mean {label_prefix}" if not added_label else None
        ax.scatter(X_sorted, mu_sorted, c=color_mean, s=50, alpha=0.6, label=lbl, linewidth=2, marker='+')
        
        multiplier = 2
        for x_pt, mu_pt, sigma_pt in zip(X_sorted, mu_sorted, sigma_sorted):
            ax.vlines(x_pt, mu_pt - multiplier * sigma_pt, mu_pt + multiplier * sigma_pt, 
                      color=color_band, alpha=0.1, label=f"{label_prefix} Uncertainty" if not added_label else None)
            added_label = True
            
    if len(stats_dict) > 0:
        print(f"Mean MSE for {label_prefix}: {total_mse / len(stats_dict):.4f}")

plot_bands(train_stats, 'green', 'green', "Train Pred.")
plot_bands(test_stats, 'red', 'red', "Test Pred.")

ax.set_title("Ours", fontsize=26)
ax.set_ylim(-3.5, 3.5)
ax.grid(True, alpha=0.1)
plt.xlabel("Standardized O3")
plt.ylabel("Standardized NOx")
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(plots_path / "oosseq_predictions.pdf")
plt.show()