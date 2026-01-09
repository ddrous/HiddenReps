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
import datetime
import shutil
import sys

from equinox_utils import EideticGRUCell

# Set plotting style
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'


# --- 1. CONFIGURATION ---
TRAIN = True  
RUN_DIR = "." 

CONFIG = {
    "seed": time.time_ns() % (2**32 - 1),
    # "seed": 2026,

    # Data & MLP Hyperparameters
    "data_samples": 2000,
    "noise_std": 0.005,
    "segments": 11,
    "x_range": [-1.5, 1.5],
    "train_seg_ids": [2, 3, 4, 5, 6, 7, 8], 
    "width_size": 48,

    # Expansion Hyperparameters
    "n_circles": 50,           
    
    # GRU & Training Config
    "lr": 5e-5,      
    "gru_hidden_size": 128,    
    "gru_epochs": 5000,  
    "gru_batch_size": 1,      # Number of parallel initializations (Seeds)
    "eidetic_gru": True,        # Whether to use Eidetic GRU Cell
    
    "gru_target_step": 100,    # Total steps to unroll (Horizon)
    "scheduled_loss_weight": False,  # Whether to use scheduled weight for each loss step

    ## Consistency Loss Config
    "n_synthetic_points": 1000,  # Number of synthetic points to sample for consistency loss
    # "consistency_loss_weight": 1.25,  # Weight for the consistency loss term
    "consistency_loss_weight": 0.0,  # Weight for the (de)consistency loss term

    # Regularization Config
    "regularization_step": 75,     # Step at which to apply the 'final' constraint
    "regularization_weight": 0.0,  # Coefficient for the reg loss (0 = drift)
    
    # Data Selection Mode
    "data_selection": "annulus",   # "annulus" or "full_disk"
    "final_step_mode": "none",     # 'full' (regularize at reg_step) or 'none'
}

print("Config seed is:", CONFIG["seed"])

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

## We need a fake set of radii for synthetic data sampling in consistency loss. 
## We simply extend the "radii" with same spacing, uncil we have gru_steps
delta_radius = radii[1] - radii[0]
fake_radii = jnp.arange(radii[-1], radii[-1] + (CONFIG["gru_target_step"]-CONFIG["n_circles"])*delta_radius + 0.01, delta_radius)
all_radii = jnp.concatenate([radii, fake_radii])

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
    
    def __init__(self, key, input_dim, hidden_dim):
        k1, k2, k3 = jax.random.split(key, 3)
        
        self.encoder = eqx.nn.Linear(input_dim, hidden_dim, key=k1)
        if not CONFIG["eidetic_gru"]:
            self.cell = eqx.nn.GRUCell(hidden_dim, hidden_dim, key=k2)
        else:
            self.cell = EideticGRUCell(hidden_dim, hidden_dim, key=k2)      ##TODO
        self.decoder = eqx.nn.Linear(hidden_dim, input_dim, key=k3)

    def __call__(self, x0, steps, key=None):
        # x0 shape: (Input_Dim)
        # We handle batching via vmap in training loop, so here assumes single item logic
        
        h0 = jnp.zeros((self.cell.hidden_size,))
        keys = jax.random.split(key, steps) if key is not None else [None] * steps
        
        def scan_step(carry, inputs):
            h, x_prev = carry
            
            # Autoregressive input (Always previous prediction)
            current_input = x_prev

            embedded = self.encoder(current_input)
            h_new = self.cell(embedded, h)
            out = self.decoder(h_new)
            
            x_next = out
            return (h_new, x_next), x_next

        init_carry = (h0, x0)
        _, predictions = jax.lax.scan(scan_step, init_carry, jnp.array(keys))
        
        return predictions

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
print(f"Generating {CONFIG['gru_batch_size']} initial states...")
x0_batch_list = []
gen_key = jax.random.PRNGKey(CONFIG["seed"] + 100)

# for _ in range(CONFIG["gru_batch_size"]):
#     gen_key, sk = jax.random.split(gen_key)
#     m = MLPModel(sk)
#     p, _ = eqx.partition(m, eqx.is_array)
#     f, _, _, _ = flatten_pytree(p)
#     x0_batch_list.append(f)

# x0_batch = jnp.stack(x0_batch_list) # (Batch, Input_Dim)


def gen_x0_batch(batch_size, key):
    x0_batch_list = []
    gen_key = key

    for _ in range(batch_size):
        gen_key, sk = jax.random.split(gen_key)
        m = MLPModel(sk)
        p, _ = eqx.partition(m, eqx.is_array)
        f, _, _, _ = flatten_pytree(p)

        # ## Perturb slightly around the fixed init model
        # eps = jax.random.uniform(gen_key, shape=f.shape, minval=-1e-5, maxval=1e-5)
        # f = f + eps

        # ## Let's pick 10 paramters at random, and perturb them only
        # eps = jax.random.uniform(gen_key, shape=(10,), minval=-1e-4, maxval=1e-4)
        # param_indices = jax.random.choice(gen_key, f.shape[0], shape=(10,), replace=False)
        # f = f.at[param_indices].add(eps)
    
        x0_batch_list.append(f)

    x0_batch = jnp.stack(x0_batch_list) 
    return x0_batch

x0_batch = gen_x0_batch(CONFIG["gru_batch_size"], gen_key)


# 3. Init GRU
gru_model = WeightGRU(k_gru, input_dim, CONFIG["gru_hidden_size"])
opt = optax.adam(CONFIG["lr"]) 
opt_state = opt.init(eqx.filter(gru_model, eqx.is_array))

#%%
# --- 6. END-TO-END TRAINING LOOP (BATCHED) ---

def get_functional_loss(flat_w, step_idx):
    # Unflatten MLP
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
    
    # Apply Weight Coefficient if at regularization step
    # For normal steps, weight is implicitly 1.0 (or we consider it structural)
    # If is_reg_step, we multiply by regularization_weight
    # Note: If reg_weight is 0, the loss at reg_step becomes 0.
    
    eff_weight = jax.lax.select(is_reg_step, CONFIG["regularization_weight"], 1.0)
    
    final_loss = base_loss * eff_weight

    if not CONFIG["scheduled_loss_weight"]:
        return final_loss
    else:
        return (base_loss * eff_weight) / (step_idx**2 + 1)         ## TODO scale down over time?


def get_consistency_loss(flat_w_in, flat_w_out, step_idx_in, key):
    ## Consistency loss between two consecutive steps. 
    ## The models corresponds to the inner and outer circles must match their predictions on the inner circle data. 

    # Unflatten MLPs
    params_in = unflatten_pytree(flat_w_in, shapes, treedef, mask)
    model_in = eqx.combine(params_in, static)

    params_out = unflatten_pytree(flat_w_out, shapes, treedef, mask)
    model_out = eqx.combine(params_out, static)

    ## Sample the synthetic data, it should all fall within the inner circle;
    ## We know the center of the data, and we know the radius of the inner circle at this step
    ## We want to sample with higer probability close to the perimeter of the inner circle. Gradual probablity increase.
    ## This is synthetic data, so we can sample as much as we want, even outside n_circles in the original data
    circle_idx = jnp.minimum(step_idx_in, CONFIG["gru_target_step"] - 1)
    radius = all_radii[circle_idx]
    n_synthetic = CONFIG["n_synthetic_points"]
    angles = jax.random.uniform(key, shape=(n_synthetic,)) * 2 * jnp.pi
    
    ## Uniform sampling
    # radii_sampled = jax.random.uniform(key, shape=(n_synthetic,)) * radius

    ## Sampling with higher density near the perimeter (closer to radius). Use the beta distribution with alpha>1
    radii_sampled = jax.random.beta(key, a=5.0, b=1.0, shape=(n_synthetic,)) * radius

    X_synthetic = x_mean + radii_sampled * jnp.cos(angles)      ## TODO: add a dimention along axis 1 if x is multi-dim?
    
    y_pred_in = model_in.predict(X_synthetic[:, None])
    y_pred_out = model_out.predict(X_synthetic[:, None])
    residuals = (y_pred_in - y_pred_out) ** 2

    final_loss =  jnp.mean(residuals)
    if CONFIG["scheduled_loss_weight"]:
        return final_loss / (step_idx_in**2 + 1)         ## TODO scale down over time?
    else:
        return final_loss



# def get_disconsistency_loss(flat_w_in, flat_w_out, step_idx_in, key):
#     ## Consistency loss between two consecutive steps. 
#     ## The models corresponds to the inner and outer circles must NOT match their predictions on the anulus data. 

#     # Unflatten MLPs
#     params_in = unflatten_pytree(flat_w_in, shapes, treedef, mask)
#     model_in = eqx.combine(params_in, static)

#     params_out = unflatten_pytree(flat_w_out, shapes, treedef, mask)
#     model_out = eqx.combine(params_out, static)

#     ## Sample the synthetic data, it should all fall within the inner circle;
#     ## We know the center of the data, and we know the radius of the inner circle at this step
#     ## We want to sample with higer probability close to the perimeter of the inner circle. Gradual probablity increase.
#     ## This is synthetic data, so we can sample as much as we want, even outside n_circles in the original data
#     circle_idx = jnp.minimum(step_idx_in, CONFIG["gru_target_step"] - 2)
#     radius_in = all_radii[circle_idx]
#     radius_out = all_radii[circle_idx + 1]
#     n_synthetic = CONFIG["n_synthetic_points"]
#     angles = jax.random.uniform(key, shape=(n_synthetic,)) * 2 * jnp.pi

#     ## Sampling with higher density near the perimeter (closer to radius). Use the beta distribution with alpha>1
#     radii_sampled = jax.random.beta(key, a=5.0, b=1.0, shape=(n_synthetic,)) * (radius_out - radius_in) + radius_in

#     X_synthetic = x_mean + radii_sampled * jnp.cos(angles)      ## TODO: add a dimention along axis 1 if x is multi-dim?
    
#     y_pred_in = model_in.predict(X_synthetic[:, None])
#     y_pred_out = model_out.predict(X_synthetic[:, None])
#     residuals = (y_pred_in - y_pred_out) ** 2

#     # return -jnp.mean(residuals)
#     return jnp.maximum(0.0, 1.0 - jnp.mean(residuals))   ## Hinge loss style


def get_disconsistency_loss(flat_w_in, flat_w_out, step_idx_in, key):
    ## Maybe the consistency loss is just maximiisng the difference between the two models directly?
    residual = jnp.mean((flat_w_in - flat_w_out) ** 2)
    return jnp.maximum(0.0, 1.0 - residual)   ## Hinge loss style

@eqx.filter_value_and_grad
def train_step_fn(gru, x0_batch, key):
    # x0_batch: (Batch, D)
    total_steps = CONFIG["gru_target_step"]
    
    # VMAP GRU over batch
    # keys needs to be (Batch,) for randomness if used, but here GRU is deterministic
    # We broadcast key or split it if needed. 
    # Current GRU doesn't use key for sampling anymore (MSE mode)

    preds_batch = jax.vmap(gru, in_axes=(0, None, None))(x0_batch, total_steps, None) # preds_batch: (Batch, Steps, D)

    ## DO a cumsum along the steps to get the full trajectory
    # preds_batch = jnp.cumsum(preds_batch, axis=1)
    
    step_indices = jnp.arange(total_steps)
    preds_batch_data = preds_batch

    # ## TODO: randomly select two steps per batch member for efficiency
    # step_indices = jax.random.choice(key, CONFIG["n_circles"], shape=(12,), replace=False)
    # # step_indices = jax.random.choice(key, 10, shape=(2,), replace=False)
    # # step_indices = jnp.array([CONFIG["n_circles"]-1])
    # step_indices = jnp.sort(step_indices)
    # preds_batch_data = preds_batch[:, step_indices, :]

    # Calculate loss for each batch member, for each step
    # double vmap: outer over batch, inner over steps
    def loss_per_seq(seq):
        return jax.vmap(get_functional_loss)(seq, step_indices)
    losses_batch = jax.vmap(loss_per_seq)(preds_batch_data) # (Batch, Steps)
    # Mean over batch, Sum over steps
    total_data_loss = jnp.mean(jnp.sum(losses_batch, axis=1))


    ## TODO: Add any additional regularization losses here if needed
    ## This is a consistency loss along the sequence.
    step_indices = jnp.arange(total_steps)
    # step_indices = jnp.arange(CONFIG["n_circles"], total_steps)
    # step_indices = jax.random.choice(key, total_steps, shape=(50,), replace=False)
    # step_indices = jnp.sort(step_indices)
    keys = jax.random.split(key, len(step_indices)-1)
    preds_batch_cons = preds_batch[:, step_indices, :]
    def cons_loss_per_seq(seq):
        # return jax.vmap(get_consistency_loss)(seq[:-1], seq[1:], step_indices[:-1], keys)
        return jax.vmap(get_disconsistency_loss)(seq[:-1], seq[1:], step_indices[:-1], keys)
    cons_losses_batch = jax.vmap(cons_loss_per_seq)(preds_batch_cons) # (Batch, Steps-1)
    total_cons_loss = jnp.mean(jnp.sum(cons_losses_batch, axis=1))

    # total_loss = total_data_loss
    # total_loss = total_cons_loss
    total_loss = total_data_loss + CONFIG["consistency_loss_weight"]*total_cons_loss

    return total_loss

@eqx.filter_jit
def make_step(gru, opt_state, x0_batch, key):
    loss, grads = train_step_fn(gru, x0_batch, key)
    updates, opt_state = opt.update(grads, opt_state, gru)
    gru = eqx.apply_updates(gru, updates)
    return gru, opt_state, loss

if TRAIN:
    print(f"🚀 Starting Batch GRU Training. Batch Size: {CONFIG['gru_batch_size']}")
    print(f"Reg Weight: {CONFIG['regularization_weight']} @ Step {CONFIG['regularization_step']}")
    
    loss_history = []
    train_key = jax.random.PRNGKey(CONFIG["seed"] + 99)
    best_model = gru_model      ## Best model on train set
    lowest_loss = float('inf')

    for ep in range(CONFIG["gru_epochs"]):
        train_key, step_key = jax.random.split(train_key)

        x0_batch = gen_x0_batch(CONFIG["gru_batch_size"], step_key)     ## TODO: regenerate each step?

        gru_model, opt_state, loss = make_step(gru_model, opt_state, x0_batch, step_key)
        loss_history.append(loss)

        if loss < lowest_loss:
            lowest_loss = loss
            best_model = gru_model
        
        if (ep+1) % 500 == 0:
            print(f"Epoch {ep+1} | Loss: {loss:.6f}")

    ## Make sure we use the best model at the end
    gru_model = best_model

    # Generate final trajectories for ALL batch members
    eval_key = jax.random.PRNGKey(42)
    final_batch_traj = jax.vmap(gru_model, in_axes=(0, None, None))(x0_batch, CONFIG["gru_target_step"], None)
    
    np.save(artefacts_path / "final_batch_traj.npy", final_batch_traj)
    np.save(artefacts_path / "loss_history.npy", np.array(loss_history))

    ## Save the gru model as well with Equinox serialization
    eqx.tree_serialise_leaves(artefacts_path / "gru_model.eqx", gru_model)

else:
    print("Loading results...")
    final_batch_traj = np.load(artefacts_path / "final_batch_traj.npy")
    loss_history = np.load(artefacts_path / "loss_history.npy")
    gru_model = eqx.tree_deserialise_leaves(artefacts_path / "gru_model.eqx", gru_model)

# Use the FIRST trajectory for standard single-model dashboards to keep them working
final_traj = final_batch_traj[0]

#%%
# --- 7. VISUALIZATION ---
print("\n=== Generating Dashboards ===")

x_grid = jnp.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], 300)[:, None]

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
ax2.set_title("Performance Evolution (Single Seed)")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, :])
ax3.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1)
cmap = plt.cm.coolwarm
n_steps = len(final_traj)
for i in range(0, n_steps, 1):
    w = final_traj[i]
    p = unflatten_pytree(w, shapes, treedef, mask)
    m = eqx.combine(p, static)
    label = "Limit" if i==n_steps-1 else None
    # alpha = 0.3 + 0.7 * (i / (n_steps - 1))
    alpha = 1.0 if i==n_steps-1 else 0.1
    ax3.plot(x_grid, m.predict(x_grid), color=cmap(i/n_steps), alpha=alpha, linewidth=1.5, label=label)

## PLot the test set as well
ax3.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1)

# Highlight Reg Step
w_reg = final_traj[CONFIG["regularization_step"]]
p_reg = unflatten_pytree(w_reg, shapes, treedef, mask)
ax3.plot(x_grid, eqx.combine(p_reg, static).predict(x_grid), "--", color='red', linewidth=2, label="Reg Step")
ax3.set_title("Function Evolution")
ax3.legend()

plt.tight_layout()
plt.savefig(plots_path / "dashboard_functional.png")
plt.show()



# --- DASHBOARD 1: BATCH LIMITS (NEW) ---
print("Generating Batch Limits Dashboard...")
fig_batch = plt.figure(figsize=(20, 8))
gs_batch = fig_batch.add_gridspec(1, 3)

# Define Key Steps
steps_to_plot = [CONFIG["n_circles"], CONFIG["regularization_step"], CONFIG["gru_target_step"] - 1]
titles = ["End of Circles", "Regularization Step", "Final Limit"]


for i, step_idx in enumerate(steps_to_plot):
    ax = fig_batch.add_subplot(gs_batch[0, i])
    
    # Background Data
    ax.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.05)
    ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1)
    
    # Plot all batch members
    for b in range(CONFIG["gru_batch_size"]):
        w = final_batch_traj[b, step_idx]
        p = unflatten_pytree(w, shapes, treedef, mask)
        m = eqx.combine(p, static)
        pred = m.predict(x_grid)
        
        # Color based on batch index to track consistency
        color = plt.cm.tab20(b % 20)
        ax.plot(x_grid, pred, color=color, alpha=0.6, linewidth=1.5)
        
    ax.set_title(f"{titles[i]} (Step {step_idx})")
    ax.set_ylim(jnp.min(Y_train_full)-1, jnp.max(Y_train_full)+1)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plots_path / "dashboard_batch_limits.png")
plt.show()


#%%
# --- 8. ADVANCED DIAGNOSTICS ---
# Re-using the latest diagnostic code
print("\n=== Generating Advanced Diagnostics ===")
fig_diag = plt.figure(figsize=(20, 16))
gs_diag = fig_diag.add_gridspec(2, 2)

# PLOT A: PARAMETER TRAJECTORIES (First Batch Member)
ax_t1 = fig_diag.add_subplot(gs_diag[0, 0])
np.random.seed(CONFIG["seed"])
param_indices = np.random.choice(input_dim, 10, replace=False)
distinct_cmaps = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]

# Use the first trajectory from the batch for detailed analysis
traj_seed_0 = final_batch_traj[0]

for i, idx in enumerate(param_indices):
    traj = traj_seed_0[:, idx]
    cmap = distinct_cmaps[i % len(distinct_cmaps)]
    for t in range(len(traj) - 1):
        ax_t1.plot([t, t+1], [traj[t], traj[t+1]], color=cmap(0.3 + 0.7 * (t/len(traj))), linewidth=1.5)
    ax_t1.scatter([], [], color=cmap(0.9), label=f"Param {idx}")

ax_t1.axvline(CONFIG["n_circles"], color='k', linestyle='--')
ax_t1.axvline(CONFIG["regularization_step"], color='red', linestyle=':')
ax_t1.set_title("Parameter Trajectories (Single Seed)")
ax_t1.legend(ncol=2, fontsize='small')

# PLOT B: STABILITY (Jacobian at Reg Step of First Seed)
ax_e = fig_diag.add_subplot(gs_diag[0, 1])

def get_gru_step_fn(gru):
    def step_fn(joint_state):
        # One step of GRU dynamics: (w_t, h_t) -> (w_t+1, h_t+1)
        w, h = joint_state
        embedded = gru.encoder(w)
        h_new = gru.cell(embedded, h)
        w_new = gru.decoder(h_new)
        return (w_new, h_new)
    return step_fn

# --- FIX: Proper Autoregressive Scan to Recover Hidden State ---
print("Recovering hidden state for stability analysis...")
h0 = jnp.zeros((gru_model.cell.hidden_size,))
x_start = x0_batch[0] # Initial weight for the first seed

def diag_scan_step(carry, _):
    # Ignore the second arg (which is None)
    h, x_prev = carry
    
    # Autoregressive: Input is previous output
    embedded = gru_model.encoder(x_prev)
    h_new = gru_model.cell(embedded, h)
    x_next = gru_model.decoder(h_new)
    
    return (h_new, x_next), None

# Run forward exactly as training did to get consistent (h, w) at reg_step
(h_target, w_target), _ = jax.lax.scan(
    diag_scan_step,
    (h0, x_start), 
    None, 
    length=CONFIG["regularization_step"]
)

step_fn = get_gru_step_fn(gru_model)
try:
    # Compute Jacobian of the step function at the regularization point
    print(f"Computing Jacobian at step {CONFIG['regularization_step']}...")
    J_tuple = jax.jacfwd(step_fn)((w_target, h_target))
    
    # J_tuple is ((dw'/dw, dw'/dh), (dh'/dw, dh'/dh))
    J_ww = J_tuple[0][0]
    J_wh = J_tuple[0][1]
    J_hw = J_tuple[1][0]
    J_hh = J_tuple[1][1]
    
    top = jnp.concatenate([J_ww, J_wh], axis=1)
    bot = jnp.concatenate([J_hw, J_hh], axis=1)
    Full_J = jnp.concatenate([top, bot], axis=0)
    
    eigenvals = jnp.linalg.eigvals(Full_J)
    
    # Visualization
    unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=2)
    ax_e.add_patch(unit_circle)
    
    re = jnp.real(eigenvals)
    im = jnp.imag(eigenvals)
    ax_e.scatter(re, im, alpha=0.6, s=30, color='purple')
    
    max_rad = jnp.max(jnp.abs(eigenvals))
    ax_e.text(0.05, 0.95, f"Max |λ|: {max_rad:.4f}", transform=ax_e.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    ax_e.set_title("Stability Analysis (Eigenvalues)")
    ax_e.set_aspect('equal')
    ax_e.grid(True, alpha=0.3)
    
    # Zoom to fit unit circle + outliers
    lim = max(1.1, float(max_rad) + 0.1)
    ax_e.set_xlim(-lim, lim)
    ax_e.set_ylim(-lim, lim)
    
except Exception as e:
    print(f"Jacobian computation failed: {e}")
    ax_e.text(0.5, 0.5, "Jacobian Failed\n(See Console)", ha='center')

# PLOT C: 3D PCA
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


pca = PCA(n_components=3)
traj_3d = pca.fit_transform(np.array(traj_seed_0))

ax_3d = fig_diag.add_subplot(gs_diag[1, 0], projection='3d')
ax_3d.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], color='gray', alpha=0.5)
sc = ax_3d.scatter(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], c=range(len(traj_3d)), cmap='viridis', s=40)

# Mark Key Events
ax_3d.scatter(traj_3d[0, 0], traj_3d[0, 1], traj_3d[0, 2], c='green', s=150, marker='o', label='Start')
ax_3d.scatter(traj_3d[CONFIG["n_circles"], 0], traj_3d[CONFIG["n_circles"], 1], traj_3d[CONFIG["n_circles"], 2], c='orange', s=150, marker='D', label='End Data')
ax_3d.scatter(traj_3d[CONFIG["regularization_step"], 0], traj_3d[CONFIG["regularization_step"], 1], traj_3d[CONFIG["regularization_step"], 2], c='red', s=200, marker='*', label='Reg Step')

ax_3d.set_title("Weight Trajectory (PCA)")
ax_3d.legend()

# PLOT D: DATA EXPANSION
ax_circles = fig_diag.add_subplot(gs_diag[1, 1])
point_indices = jnp.sum(~circle_masks, axis=0)

# Use a more distinct colormap
scatter = ax_circles.scatter(X_train_full, Y_train_full, c=point_indices, 
                            cmap='Spectral', s=15, alpha=0.9)

for i in range(0, CONFIG["n_circles"], 5):
    r = radii[i]
    ax_circles.axvline(x_mean - r, color='k', linestyle='-', alpha=0.2)
    ax_circles.axvline(x_mean + r, color='k', linestyle='-', alpha=0.2)
    
    # Add radius labels at the top of each vertical line
    y_top = ax_circles.get_ylim()[1]
    ax_circles.text(x_mean - r, y_top, f'R{i}', 
                   ha='center', va='bottom', fontsize=8, rotation=90)
    ax_circles.text(x_mean + r, y_top, f'R{i}', 
                   ha='center', va='bottom', fontsize=8, rotation=90)

cbar = plt.colorbar(scatter, ax=ax_circles, ticks=np.arange(0, CONFIG["n_circles"]+1, 5))
cbar.set_label("Circle Index")
ax_circles.set_title("Training Data Circles", y=1.02)

plt.tight_layout()
plt.savefig(plots_path / "dashboard_advanced_diagnostics.png")
plt.show()

#%% Special plot paramter trajectories
## We are plotting as many parameters as possible in a tall plot. We only plot the first n_circles steps to focus on the interesting part.
print("Generating Extended Parameter Trajectories Plot...")
fig, ax = plt.subplots(figsize=(7, 10), nrows=1, ncols=1) ## everything in one axis
plot_ids = np.arange(100)  # First 64 parameters

for idx in plot_ids:
    # traj = traj_seed_0[:CONFIG["n_circles"], idx]
    # ax.plot(np.arange(CONFIG["n_circles"]), traj, linewidth=1.5, label=f"Param {idx}")

    ## Plot the difference from initial value to highlight changes
    # traj = traj_seed_0[:CONFIG["n_circles"], idx]
    # traj_diff = traj - traj[0]
    # ax.plot(np.arange(CONFIG["n_circles"]), traj_diff, linewidth=1.5, label=f"Param {idx}")

    ## Plot the difference x_t - x_(t-1) to highlight changes
    traj = traj_seed_0[:CONFIG["n_circles"], idx]
    traj_diff = jnp.concatenate([jnp.array([0.0]), traj[1:] - traj[:-1]])
    ax.plot(np.arange(CONFIG["n_circles"]), traj_diff, linewidth=1.5, label=f"Param {idx}")

ax.set_title("Parameter Trajectories")
ax.set_xlabel("Training Step")
ax.set_ylabel("Parameter Value")
# ax.legend(ncol=2, fontsize='small')
plt.tight_layout()
plt.savefig(plots_path / "extended_parameter_trajectories.png")
plt.show()



#%% Plot the n_circles model predictions on the train and test sets
# This shows how well the model fits the data at the end of the data expansion phase
print("Generating n_circles Model Prediction Plot...")
# step_idx = CONFIG["n_circles"]
step_idx = 35
fig, ax = plt.subplots(figsize=(10, 6))     
# Background Data
# ax.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1, label="Train Data")
ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1, label="Test Data")

## PLot the train data in colors correspond to the radius for this steo_idx
dists = jnp.abs(X_train_full - x_mean).flatten()
r_current = radii[jnp.minimum(step_idx, CONFIG["n_circles"] - 1)]
train_colors = np.where(dists <= r_current, 'green', 'blue')
ax.scatter(X_train_full, Y_train_full, c=train_colors, s=10, alpha=0.3, label="Train Data")

## Create a new final batch traj from a random seed to ensure we have diversity
print("Generating new batch for n_circles prediction plot...")
x0_batch_list = []
gen_key = jax.random.PRNGKey(time.time_ns() % (2**32 - 1))
# gen_key = jax.random.PRNGKey(CONFIG["seed"] + 100)
for _ in range(CONFIG["gru_batch_size"]):
    gen_key, sk = jax.random.split(gen_key)
    m = MLPModel(sk)
    p, _ = eqx.partition(m, eqx.is_array)
    f, _, _, _ = flatten_pytree(p)
    x0_batch_list.append(f)
x0_batch = jnp.stack(x0_batch_list) # (Batch, Input_Dim)
new_final_batch_traj = jax.vmap(gru_model, in_axes=(0, None, None))(x0_batch, CONFIG["gru_target_step"], None)

# Plot all batch members
for b in range(CONFIG["gru_batch_size"]):
    # w = final_batch_traj[b, step_idx]
    w = new_final_batch_traj[b, step_idx]
    p = unflatten_pytree(w, shapes, treedef, mask)
    m = eqx.combine(p, static)
    pred = m.predict(x_grid)
    
    # Color based on batch index to track consistency
    color = plt.cm.tab20(b % 20)
    ax.plot(x_grid, pred, color=color, alpha=0.6, linewidth=1.5)
ax.set_title(f"Model Predictions Corresponding to Circles (Step {step_idx})")
ax.set_ylim(jnp.min(Y_train_full)-1, jnp.max(Y_train_full)+1)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(plots_path / "model_predictions_n_circles.png")
plt.show()


#%% Special evaluation of the loss
## Our annulus training strategy gives us gru_step>n_circles models, each specialised on its own radius.
## So our evalulation is simple. For each point, we find which circle it belongs to, and use the corresponding model to evaluate its loss.
print("Evaluating Final Model Loss with Circle-Specific Models...")
# def evaluate_circle_specific_loss(final_batch_traj, X_data, Y_data):
#     total_loss = 0.0
#     n_points = X_data.shape[0]
    
#     for i in range(n_points):
#         x_point = X_data[i:i+1, :]  # Shape (1, D)
#         y_point = Y_data[i:i+1, :]  # Shape (1, D)
        
#         dist = jnp.abs(x_point - x_mean).item()
        
#         # Determine which circle this point belongs to
#         circle_idx = 0
#         for j, r in enumerate(radii):
#             if dist <= r:
#                 circle_idx = j
#                 break
#         else:
#             circle_idx = CONFIG["n_circles"] - 1  # Assign to the outermost circle if beyond all
        
#         # Use the corresponding model from the final batch trajectory
#         w = final_batch_traj[0, circle_idx]  # Using the first batch member for simplicity
#         p = unflatten_pytree(w, shapes, treedef, mask)
#         m = eqx.combine(p, static)
        
#         y_pred = m.predict(x_point)
#         point_loss = jnp.mean((y_pred - y_point) ** 2)
#         total_loss += point_loss
    
#     avg_loss = total_loss / n_points
#     return avg_loss

def evaluate_circle_specific_loss(final_batch_traj, X_data, Y_data):
    ## THis function should return the average loss, for each circle. only on the corresponding data points.
    ## All this in a dictionnary with key the circle index.
    circle_losses = {}
    n_points = X_data.shape[0]

    for circle_idx in range(CONFIG["gru_target_step"]):
        # Get the model for this circle from the first batch member
        w = final_batch_traj[0, circle_idx]  # Using the first batch member for simplicity
        p = unflatten_pytree(w, shapes, treedef, mask)
        m = eqx.combine(p, static)

        # Get the mask for this circle
        ## Calculate all circle masks
        circle_masks = []
        for r in all_radii:
            new_mask = jnp.abs(X_data - x_mean) <= r
            circle_masks.append(new_mask.flatten())

        circle_mask = circle_masks[circle_idx]

        # Select data points belonging to this circle
        X_circle = X_data[circle_mask]
        Y_circle = Y_data[circle_mask]

        if X_circle.shape[0] == 0:
            continue  # Skip if no points in this circle

        y_pred = m.predict(X_circle)
        circle_loss = jnp.mean((y_pred - Y_circle) ** 2)
        circle_losses[circle_idx] = circle_loss

    return circle_losses

final_test_loss_cc = evaluate_circle_specific_loss(final_batch_traj, X_test, Y_test)
test_loss_cc = jnp.mean(jnp.array(list(final_test_loss_cc.values())))
print(f"Final Test Loss (Circle-Specific Models): {test_loss_cc:.6f}")
final_train_loss_cc = evaluate_circle_specific_loss(final_batch_traj, X_train_full, Y_train_full)

## Replot the Performance Evolution with Circle-Specific Loss as a horizontal line
fig, ax = plt.subplots(figsize=(10, 6))
## Plot final_train_loss_circle_specific and final_test_loss_circle_specific as horizontal lines
ax.plot(traj_train_losses, label="Train MSE", color='blue', alpha=0.7)
# ax.plot(final_train_loss_cc.keys(), final_train_loss_cc.values(), label="Train MSE (CC)", color='blue', linewidth=2, linestyle='--')
ax.plot(traj_test_losses, label="Test MSE", color='orange', linewidth=2)
# ax.plot(final_test_loss_cc.keys(), final_test_loss_cc.values(), label="Test MSE (CC)", color='orange', linewidth=2, linestyle='--')
## PLot a horizontal line for the average circle-specific test loss
ax.axhline(test_loss_cc, color='orange', linestyle='--', label="Avg Test MSE (CC)")
ax.axhline(jnp.mean(jnp.array(list(final_train_loss_cc.values()))), color='blue', linestyle='--', label="Avg Train MSE (CC)")
ax.axvline(CONFIG["n_circles"], color='k', linestyle='--')
ax.axvline(CONFIG["regularization_step"], color='red', linestyle=':')
ax.set_yscale('log')
ax.set_title("Performance Evolution with Circle-Specific (CC) Models")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "performance_evolution_circle_specific.png")
plt.show()

#%% Let's redo the Model Predictions Corresponding to Circles plot with circle-specific models
print("Generating n_circles Model Prediction Plot (Circle-Specific Models)...")
## For each data point (train+test), find which circle it belongs to, and do the prediction with the corresponding model
dists = jnp.abs(X_train_full - x_mean).flatten()
r_current = all_radii[jnp.minimum(step_idx, CONFIG["gru_target_step"] - 1)]

def predict_circle_specific_loss(final_batch_traj, X_data):
    ## THis function should return the average loss, for each circle. only on the corresponding data points.
    ## All this in a dictionnary with key the circle index.
    circle_losses = {}
    n_points = X_data.shape[0]

    # for circle_idx in range(CONFIG["gru_target_step"]):
    for circle_idx in range(CONFIG["gru_target_step"]):
        # Get the model for this circle from the first batch member
        w = final_batch_traj[0, circle_idx]  # Using the first batch member for simplicity
        p = unflatten_pytree(w, shapes, treedef, mask)
        m = eqx.combine(p, static)

        # Get the mask for this circle
        ## Calculate all circle masks
        circle_masks = []
        for r in all_radii:
            new_mask = jnp.abs(X_data - x_mean) <= r
            circle_masks.append(new_mask.flatten())

        circle_mask = circle_masks[circle_idx]

        # Select data points belonging to this circle
        X_circle = X_data[circle_mask]

        if X_circle.shape[0] == 0:
            continue  # Skip if no points in this circle
        y_pred = m.predict(X_circle)
        circle_losses[circle_idx] = (X_circle, y_pred)
    return circle_losses


train_preds_cc = predict_circle_specific_loss(final_batch_traj, X_train_full)
test_preds_cc = predict_circle_specific_loss(final_batch_traj, X_test)

fig, ax = plt.subplots(figsize=(10, 6))     

# Background Data
ax.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1, label="Train Data")
ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1, label="Test Data")

## PLot the train data predictions in colors correspond to the radius for this steo_idx
for circle_idx, (X_circle, y_pred) in train_preds_cc.items():
    ax.scatter(X_circle, y_pred, c='green', s=1, alpha=0.3)
## PLot the test data predictions in colors correspond to the radius for this steo_idx
for circle_idx, (X_circle, y_pred) in test_preds_cc.items():
    ax.scatter(X_circle, y_pred, c='red', s=1, alpha=0.3)

ax.set_title(f"Model Predictions Corresponding to Circles (Circle-Specific Models)")
ax.set_ylim(jnp.min(Y_train_full)-1, jnp.max(Y_train_full)+1)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(plots_path / "model_predictions_n_circles_circle_specific.png")