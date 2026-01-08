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
from scipy import linalg as sp_linalg

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
    
    # S4 & Training Config (Renamed from GRU/NODE)
    "lr": 1e-4,                      # Slightly higher LR often helps S4
    "s4_state_dim": 64,              # Dimension of the SSM state (N)
    "s4_epochs": 500,  
    "s4_batch_size": 1,              
    "use_encoder_decoder": False,     # NEW: If False, runs S4 directly on weights
    "s4_latent_dim": 128,            # Used if encoder/decoder is True
    
    "s4_target_step": 100,           # Total steps to unroll
    "scheduled_loss_weight": False, 

    ## Consistency Loss Config
    "n_synthetic_points": 1000, 
    "consistency_loss_weight": 0.0, 

    # Regularization Config
    "regularization_step": 75,     
    "regularization_weight": 0.0,  
    
    # Data Selection Mode
    "data_selection": "full_disk",  # "annulus" or "full_disk"   
    "final_step_mode": "none",     
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
## We simply extend the "radii" with same spacing, uncil we have s4_steps
delta_radius = radii[1] - radii[0]
fake_radii = jnp.arange(radii[-1], radii[-1] + (CONFIG["s4_target_step"]-CONFIG["n_circles"])*delta_radius + 0.01, delta_radius)
all_radii = jnp.concatenate([radii, fake_radii])

#%%
# --- 4. MODEL DEFINITIONS (S4 STATE SPACE MODEL) ---

def make_hippo(N):
    """
    Constructs the HiPPO-LegS matrix A and vector B.
    Ref: Gu et al. 2021, equation 2.
    A_nk = -(2n+1)^(1/2) * (2k+1)^(1/2) if n > k
    A_nn = -(n+1)
    A_nk = 0 if n < k
    B_n = (2n+1)^(1/2)
    """
    def compute_A_elt(n, k):
        if n > k:
            return -np.sqrt(2*n+1) * np.sqrt(2*k+1)
        elif n == k:
            return -(n+1)
        else:
            return 0.0
            
    A = np.fromfunction(np.vectorize(compute_A_elt), (N, N))
    B = np.sqrt(2 * np.arange(N) + 1).reshape(N, 1)
    return A, B

class S4Layer(eqx.Module):
    """
    A single S4 Layer running in RECURRENT mode.
    Maps input u_t (dim H) -> state x_t (dim H x N) -> output y_t (dim H).
    Uses Bilinear discretization on HiPPO-initialized matrices.
    """
    A: jax.Array  # Continuous dynamics (frozen or learnable)
    B: jax.Array  # Input projection (frozen or learnable)
    C: jax.Array  # Output projection (learnable)
    log_delta: jax.Array # Log step size (learnable)
    
    state_dim: int = eqx.field(static=True)
    feature_dim: int = eqx.field(static=True)

    def __init__(self, feature_dim, state_dim, key):
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        
        # HiPPO Initialization
        A_init, B_init = make_hippo(state_dim)
        
        # In standard S4, A and B are often frozen or slowly learned.
        # We store them as JAX arrays. 
        self.A = jnp.array(A_init)
        self.B = jnp.array(B_init)
        
        # C is initialized normally, shape (feature_dim, state_dim)
        # We treat each feature as having its own readout from the shared dynamics A, B
        # Or more commonly in Deep S4: A and B are broadcast, C is specific.
        k_c, k_d = jax.random.split(key)
        self.C = jax.random.normal(k_c, (feature_dim, state_dim)) * (state_dim ** -0.5)
        
        # Delta (step size) initialized in log space
        # Range usually chosen around 0.001 to 0.1
        self.log_delta = jnp.log(jnp.ones(feature_dim) * 0.01)

    def get_discretized_matrices(self):
        """
        Discretize A, B using Bilinear (Tustin) transform.
        A_bar = (I - delta/2 A)^-1 (I + delta/2 A)
        B_bar = (I - delta/2 A)^-1 (delta B)
        """
        delta = jnp.exp(self.log_delta) # (H,)
        I = jnp.eye(self.state_dim)
        
        # Since we are running in recurrent mode for training (not convolution),
        # we compute this explicitly. 
        # For efficiency with many features, we assume A is shared (broadcast).
        # But Delta is per-channel.
        
        # Helper to discretize one delta
        def discretize_single(d):
            # BLAS inversion is O(N^3). With N=64, this is very fast.
            # (I - d/2 A)
            denom = I - (d / 2.0) * self.A
            numer = I + (d / 2.0) * self.A
            
            # A_bar = denom^-1 * numer
            A_bar = jnp.linalg.solve(denom, numer)
            
            # B_bar = denom^-1 * (d * B)
            B_bar = jnp.linalg.solve(denom, d * self.B)
            return A_bar, B_bar.flatten()

        # vmap over the delta vector (channels)
        A_bars, B_bars = jax.vmap(discretize_single)(delta)
        return A_bars, B_bars

    def __call__(self, x_state, u_input):
        """
        Step function.
        x_state: (H, N) - Previous SSM state
        u_input: (H,)   - Current scalar input per channel
        """
        A_bars, B_bars = self.get_discretized_matrices()
        
        # Update: x_t = A_bar * x_{t-1} + B_bar * u_t
        # Element-wise over H (channels):
        # A_bar[i] is (N, N), x_state[i] is (N,), B_bar[i] is (N,), u_input[i] is scalar
        
        new_state = jax.vmap(lambda A, x, B, u: A @ x + B * u)(
            A_bars, x_state, B_bars, u_input
        )
        
        # Output: y_t = C * x_t
        # C is (H, N). y_t is (H,)
        y_out = jax.vmap(lambda c, x: jnp.dot(c, x))(self.C, new_state)
        
        return new_state, y_out

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

class WeightS4(eqx.Module):
    # If use_encoder_decoder is True:
    encoder: eqx.nn.Linear = None
    decoder: eqx.nn.Linear = None
    s4_layer: S4Layer
    # If use_encoder_decoder is False, we might want a mixing layer after S4
    mixing: eqx.nn.Linear = None 
    
    use_enc_dec: bool = eqx.field(static=True)

    def __init__(self, key, input_dim, hidden_dim, s4_state_dim):
        self.use_enc_dec = CONFIG["use_encoder_decoder"]
        k_s4, k_misc = jax.random.split(key)
        
        if self.use_enc_dec:
            k_enc, k_dec = jax.random.split(k_misc)
            self.encoder = eqx.nn.Linear(input_dim, hidden_dim, key=k_enc)
            self.decoder = eqx.nn.Linear(hidden_dim, input_dim, key=k_dec)
            s4_dim = hidden_dim
        else:
            s4_dim = input_dim
            # If working directly in weight space, we add a mixing layer to allow interactions
            # between weight parameters, as S4 channels are independent.
            self.mixing = eqx.nn.Linear(input_dim, input_dim, key=k_misc)

        self.s4_layer = S4Layer(feature_dim=s4_dim, state_dim=s4_state_dim, key=k_s4)

    def __call__(self, x0, steps, key=None):
        # x0: Initial weight vector (D,)
        
        # 1. Encode if necessary
        if self.use_enc_dec:
            curr_input = self.encoder(x0) # (Latent,)
        else:
            curr_input = x0
            
        # Initial SSM state: Zeros (H, N)
        h_state = jnp.zeros((self.s4_layer.feature_dim, self.s4_layer.state_dim))
        
        # Scan function for autoregressive generation
        def scan_step(carry, _):
            h_prev, u_prev = carry
            
            # S4 Step
            h_next, y_pred = self.s4_layer(h_prev, u_prev)
            
            # Post-processing
            # S4 usually needs a direct feedthrough D*u, but standard relu/mixing often handles it.
            # We add a residual connection or mixing here if needed.
            
            if not self.use_enc_dec:
                # Mixing layer to allow info exchange between channels (weights)
                # S4 channels are independent, so we need this dense layer.
                # y_pred = y_pred + u_prev # Residual connection
                out = self.mixing(y_pred)
                out = jax.nn.gelu(out) + u_prev # Residual dynamics
            else:
                out = jax.nn.gelu(y_pred) + u_prev # Simple residual in latent space

            return (h_next, out), out

        # Run scan
        # We start with u_0 = curr_input
        # We generate 'steps' outputs.
        # Note: The output of step k is u_{k+1}
        
        init_carry = (h_state, curr_input)
        _, raw_preds = jax.lax.scan(scan_step, init_carry, None, length=steps)
        
        # Decode if necessary
        if self.use_enc_dec:
            predictions = jax.vmap(self.decoder)(raw_preds)
        else:
            predictions = raw_preds
            
        return predictions

#%%
# --- 5. INITIALIZATION & BATCH GENERATION ---

key = jax.random.PRNGKey(CONFIG["seed"])
k_init, k_s4, key = jax.random.split(key, 3)

# 1. Setup Model Structure (Static)
model_template = MLPModel(k_init)
params_template, static = eqx.partition(model_template, eqx.is_array)
flat_template, shapes, treedef, mask = flatten_pytree(params_template)
input_dim = flat_template.shape[0]

# 2. Generate Batch of Initial States (Different Seeds)
print(f"Generating {CONFIG['s4_batch_size']} initial states...")
x0_batch_list = []
gen_key = jax.random.PRNGKey(CONFIG["seed"] + 100)

for _ in range(CONFIG["s4_batch_size"]):
    gen_key, sk = jax.random.split(gen_key)
    m = MLPModel(sk)
    p, _ = eqx.partition(m, eqx.is_array)
    f, _, _, _ = flatten_pytree(p)
    x0_batch_list.append(f)

x0_batch = jnp.stack(x0_batch_list) # (Batch, Input_Dim)

# 3. Init S4
s4_model = WeightS4(k_s4, input_dim, CONFIG["s4_latent_dim"], CONFIG["s4_state_dim"])
opt = optax.adam(CONFIG["lr"]) 
opt_state = opt.init(eqx.filter(s4_model, eqx.is_array))

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
    
    eff_weight = jax.lax.select(is_reg_step, CONFIG["regularization_weight"], 1.0)
    final_loss = base_loss * eff_weight

    if not CONFIG["scheduled_loss_weight"]:
        return final_loss
    else:
        return (base_loss * eff_weight) / (step_idx**2 + 1)         


def get_consistency_loss(flat_w_in, flat_w_out, step_idx_in, key):
    # Unflatten MLPs
    params_in = unflatten_pytree(flat_w_in, shapes, treedef, mask)
    model_in = eqx.combine(params_in, static)

    params_out = unflatten_pytree(flat_w_out, shapes, treedef, mask)
    model_out = eqx.combine(params_out, static)

    circle_idx = jnp.minimum(step_idx_in, CONFIG["s4_target_step"] - 1)
    radius = all_radii[circle_idx]
    n_synthetic = CONFIG["n_synthetic_points"]
    angles = jax.random.uniform(key, shape=(n_synthetic,)) * 2 * jnp.pi
    
    ## Sampling with higher density near the perimeter (closer to radius).
    radii_sampled = jax.random.beta(key, a=5.0, b=1.0, shape=(n_synthetic,)) * radius

    X_synthetic = x_mean + radii_sampled * jnp.cos(angles)      
    
    y_pred_in = model_in.predict(X_synthetic[:, None])
    y_pred_out = model_out.predict(X_synthetic[:, None])
    residuals = (y_pred_in - y_pred_out) ** 2

    final_loss =  jnp.mean(residuals)
    if CONFIG["scheduled_loss_weight"]:
        return final_loss / (step_idx_in**2 + 1)       
    else:
        return final_loss

def get_disconsistency_loss(flat_w_in, flat_w_out, step_idx_in, key):
    residual = jnp.mean((flat_w_in - flat_w_out) ** 2)
    return jnp.maximum(0.0, 1.0 - residual)   ## Hinge loss style

@eqx.filter_value_and_grad
def train_step_fn(model, x0_batch, key):
    # x0_batch: (Batch, D)
    total_steps = CONFIG["s4_target_step"]
    
    preds_batch = jax.vmap(model, in_axes=(0, None, None))(x0_batch, total_steps, None) # preds_batch: (Batch, Steps, D)
    
    # step_indices = jnp.arange(total_steps)
    # preds_batch_data = preds_batch

    step_indices = jnp.array([CONFIG["n_circles"]-1])
    preds_batch_data = preds_batch[:, step_indices, :]

    # double vmap: outer over batch, inner over steps
    def loss_per_seq(seq):
        return jax.vmap(get_functional_loss)(seq, step_indices)
    losses_batch = jax.vmap(loss_per_seq)(preds_batch_data) # (Batch, Steps)
    total_data_loss = jnp.mean(jnp.sum(losses_batch, axis=1))

    keys = jax.random.split(key, len(step_indices)-1)
    preds_batch_cons = preds_batch[:, step_indices, :]
    def cons_loss_per_seq(seq):
        return jax.vmap(get_disconsistency_loss)(seq[:-1], seq[1:], step_indices[:-1], keys)
    cons_losses_batch = jax.vmap(cons_loss_per_seq)(preds_batch_cons) 
    total_cons_loss = jnp.mean(jnp.sum(cons_losses_batch, axis=1))

    total_loss = total_data_loss + CONFIG["consistency_loss_weight"]*total_cons_loss

    return total_loss

@eqx.filter_jit
def make_step(model, opt_state, x0_batch, key):
    loss, grads = train_step_fn(model, x0_batch, key)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

if TRAIN:
    print(f"🚀 Starting Batch S4 Training. Batch Size: {CONFIG['s4_batch_size']}")
    print(f"Mode: {'Encoder/Decoder' if CONFIG['use_encoder_decoder'] else 'Direct Weight Space'}")
    
    loss_history = []
    train_key = jax.random.PRNGKey(CONFIG["seed"] + 99)
    best_model = s4_model      ## Best model on train set
    lowest_loss = float('inf')

    for ep in range(CONFIG["s4_epochs"]):
        train_key, step_key = jax.random.split(train_key)
        s4_model, opt_state, loss = make_step(s4_model, opt_state, x0_batch, step_key)
        loss_history.append(loss)

        if loss < lowest_loss:
            lowest_loss = loss
            best_model = s4_model
        
        if (ep+1) % 100 == 0:
            print(f"Epoch {ep+1} | Loss: {loss:.6f}")

    ## Make sure we use the best model at the end
    s4_model = best_model

    # Generate final trajectories for ALL batch members
    eval_key = jax.random.PRNGKey(42)
    final_batch_traj = jax.vmap(s4_model, in_axes=(0, None, None))(x0_batch, CONFIG["s4_target_step"], None)
    
    np.save(artefacts_path / "final_batch_traj.npy", final_batch_traj)
    np.save(artefacts_path / "loss_history.npy", np.array(loss_history))

    ## Save the model
    eqx.tree_serialise_leaves(artefacts_path / "s4_model.eqx", s4_model)

else:
    print("Loading results...")
    final_batch_traj = np.load(artefacts_path / "final_batch_traj.npy")
    loss_history = np.load(artefacts_path / "loss_history.npy")
    s4_model = eqx.tree_deserialise_leaves(artefacts_path / "s4_model.eqx", s4_model)

# Use the FIRST trajectory for standard single-model dashboards
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
    alpha = 1.0 if i==n_steps-1 else 0.1
    ax3.plot(x_grid, m.predict(x_grid), color=cmap(i/n_steps), alpha=alpha, linewidth=1.5, label=label)

ax3.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1)
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
steps_to_plot = [CONFIG["n_circles"], CONFIG["regularization_step"], CONFIG["s4_target_step"] - 1]
titles = ["End of Circles", "Regularization Step", "Final Limit"]


for i, step_idx in enumerate(steps_to_plot):
    ax = fig_batch.add_subplot(gs_batch[0, i])
    
    # Background Data
    ax.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.05)
    ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1)
    
    # Plot all batch members
    for b in range(CONFIG["s4_batch_size"]):
        w = final_batch_traj[b, step_idx]
        p = unflatten_pytree(w, shapes, treedef, mask)
        m = eqx.combine(p, static)
        pred = m.predict(x_grid)
        
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
print("\n=== Generating Advanced Diagnostics ===")
fig_diag = plt.figure(figsize=(20, 16))
gs_diag = fig_diag.add_gridspec(2, 2)

# PLOT A: PARAMETER TRAJECTORIES 
ax_t1 = fig_diag.add_subplot(gs_diag[0, 0])
np.random.seed(CONFIG["seed"])
param_indices = np.random.choice(input_dim, 10, replace=False)
distinct_cmaps = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]

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

# PLOT B: STABILITY (Eigenvalues of Discretized A) TODO
# ax_e = fig_diag.add_subplot(gs_diag[0, 1])

# # For S4, stability is determined by the eigenvalues of the discretized matrix A_bar.
# # A_bar maps x_{t-1} to x_t. If max|eig(A_bar)| < 1, the SSM is stable.
# # Since we have A_bar per channel (due to different Delta), we plot them all.

# print("Computing S4 Eigenvalues...")
# A_bars, _ = s4_model.s4_layer.get_discretized_matrices()
# # A_bars is shape (H, N, N)
# # We sample a few channels to avoid overcrowding the plot
# sample_channels = np.linspace(0, A_bars.shape[0]-1, 10).astype(int)
# all_eigs = []

# for ch in sample_channels:
#     A_curr = A_bars[ch]
#     eigs = jnp.linalg.eigvals(A_curr)
#     all_eigs.append(eigs)

# all_eigs = jnp.concatenate(all_eigs)

# unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=2)
# ax_e.add_patch(unit_circle)

# re = jnp.real(all_eigs)
# im = jnp.imag(all_eigs)
# ax_e.scatter(re, im, alpha=0.5, s=20, color='purple')

# max_rad = jnp.max(jnp.abs(all_eigs))
# ax_e.text(0.05, 0.95, f"Max |λ|: {max_rad:.4f}", transform=ax_e.transAxes, bbox=dict(facecolor='white', alpha=0.8))
# ax_e.set_title("S4 Stability (Eigenvalues of Discretized A)")
# ax_e.set_aspect('equal')
# ax_e.grid(True, alpha=0.3)

# lim = max(1.1, float(max_rad) + 0.1)
# ax_e.set_xlim(-lim, lim)
# ax_e.set_ylim(-lim, lim)

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

scatter = ax_circles.scatter(X_train_full, Y_train_full, c=point_indices, 
                            cmap='Spectral', s=15, alpha=0.9)

for i in range(0, CONFIG["n_circles"], 5):
    r = radii[i]
    ax_circles.axvline(x_mean - r, color='k', linestyle='-', alpha=0.2)
    ax_circles.axvline(x_mean + r, color='k', linestyle='-', alpha=0.2)
    
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
print("Generating Extended Parameter Trajectories Plot...")
fig, ax = plt.subplots(figsize=(7, 10), nrows=1, ncols=1) 
plot_ids = np.arange(100)  # First 100 parameters

for idx in plot_ids:
    ## Plot the difference x_t - x_(t-1) to highlight changes
    traj = traj_seed_0[:CONFIG["n_circles"], idx]
    traj_diff = jnp.concatenate([jnp.array([0.0]), traj[1:] - traj[:-1]])
    ax.plot(np.arange(CONFIG["n_circles"]), traj_diff, linewidth=1.5, label=f"Param {idx}")

ax.set_title("Parameter Trajectories")
ax.set_xlabel("Training Step")
ax.set_ylabel("Parameter Value")
plt.tight_layout()
plt.savefig(plots_path / "extended_parameter_trajectories.png")
plt.show()



#%% Plot the n_circles model predictions on the train and test sets
print("Generating n_circles Model Prediction Plot...")
step_idx = 35
fig, ax = plt.subplots(figsize=(10, 6))     
ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1, label="Test Data")

dists = jnp.abs(X_train_full - x_mean).flatten()
r_current = radii[jnp.minimum(step_idx, CONFIG["n_circles"] - 1)]
train_colors = np.where(dists <= r_current, 'green', 'blue')
ax.scatter(X_train_full, Y_train_full, c=train_colors, s=10, alpha=0.3, label="Train Data")

print("Generating new batch for n_circles prediction plot...")
x0_batch_list = []
gen_key = jax.random.PRNGKey(time.time_ns() % (2**32 - 1))
for _ in range(CONFIG["s4_batch_size"]):
    gen_key, sk = jax.random.split(gen_key)
    m = MLPModel(sk)
    p, _ = eqx.partition(m, eqx.is_array)
    f, _, _, _ = flatten_pytree(p)
    x0_batch_list.append(f)
x0_batch = jnp.stack(x0_batch_list) 
new_final_batch_traj = jax.vmap(s4_model, in_axes=(0, None, None))(x0_batch, CONFIG["s4_target_step"], None)

for b in range(CONFIG["s4_batch_size"]):
    w = new_final_batch_traj[b, step_idx]
    p = unflatten_pytree(w, shapes, treedef, mask)
    m = eqx.combine(p, static)
    pred = m.predict(x_grid)
    
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
print("Evaluating Final Model Loss with Circle-Specific Models...")

def evaluate_circle_specific_loss(final_batch_traj, X_data, Y_data):
    circle_losses = {}
    n_points = X_data.shape[0]

    for circle_idx in range(CONFIG["s4_target_step"]):
        w = final_batch_traj[0, circle_idx] 
        p = unflatten_pytree(w, shapes, treedef, mask)
        m = eqx.combine(p, static)

        circle_masks = []
        for r in all_radii:
            new_mask = jnp.abs(X_data - x_mean) <= r
            circle_masks.append(new_mask.flatten())

        circle_mask = circle_masks[circle_idx]
        X_circle = X_data[circle_mask]
        Y_circle = Y_data[circle_mask]

        if X_circle.shape[0] == 0:
            continue  
        y_pred = m.predict(X_circle)
        circle_loss = jnp.mean((y_pred - Y_circle) ** 2)
        circle_losses[circle_idx] = circle_loss

    return circle_losses

final_test_loss_cc = evaluate_circle_specific_loss(final_batch_traj, X_test, Y_test)
test_loss_cc = jnp.mean(jnp.array(list(final_test_loss_cc.values())))
print(f"Final Test Loss (Circle-Specific Models): {test_loss_cc:.6f}")
final_train_loss_cc = evaluate_circle_specific_loss(final_batch_traj, X_train_full, Y_train_full)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(traj_train_losses, label="Train MSE", color='blue', alpha=0.7)
ax.plot(traj_test_losses, label="Test MSE", color='orange', linewidth=2)
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
dists = jnp.abs(X_train_full - x_mean).flatten()
r_current = all_radii[jnp.minimum(step_idx, CONFIG["s4_target_step"] - 1)]

def predict_circle_specific_loss(final_batch_traj, X_data):
    circle_losses = {}
    n_points = X_data.shape[0]

    for circle_idx in range(CONFIG["s4_target_step"]):
        w = final_batch_traj[0, circle_idx]  
        p = unflatten_pytree(w, shapes, treedef, mask)
        m = eqx.combine(p, static)

        circle_masks = []
        for r in all_radii:
            new_mask = jnp.abs(X_data - x_mean) <= r
            circle_masks.append(new_mask.flatten())

        circle_mask = circle_masks[circle_idx]
        X_circle = X_data[circle_mask]

        if X_circle.shape[0] == 0:
            continue  
        y_pred = m.predict(X_circle)
        circle_losses[circle_idx] = (X_circle, y_pred)
    return circle_losses


train_preds_cc = predict_circle_specific_loss(final_batch_traj, X_train_full)
test_preds_cc = predict_circle_specific_loss(final_batch_traj, X_test)

fig, ax = plt.subplots(figsize=(10, 6))     

ax.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1, label="Train Data")
ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1, label="Test Data")

for circle_idx, (X_circle, y_pred) in train_preds_cc.items():
    ax.scatter(X_circle, y_pred, c='green', s=1, alpha=0.3)
for circle_idx, (X_circle, y_pred) in test_preds_cc.items():
    ax.scatter(X_circle, y_pred, c='red', s=1, alpha=0.3)

ax.set_title(f"Model Predictions Corresponding to Circles (Circle-Specific Models)")
ax.set_ylim(jnp.min(Y_train_full)-1, jnp.max(Y_train_full)+1)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(plots_path / "model_predictions_n_circles_circle_specific.png")
# %%
