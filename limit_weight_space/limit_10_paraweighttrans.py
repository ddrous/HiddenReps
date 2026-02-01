#%%
import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import equinox as eqx
import math
import optax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import datetime
import shutil
import sys

# Set plotting style
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'


# --- 1. CONFIGURATION ---
TRAIN = True  
RUN_DIR = "." 

CONFIG = {
    # "seed": time.time_ns() % (2**32 - 1),
    "seed": 2026,

    "x_range": [-4.0, 4.0],  # Wider range to see the sine wave repeat
    "segments": 5,           # Split into 5 distinct vertical strips
    "train_seg_ids": [1, 2, 3], # Train on the middle

    # Data & MLP Hyperparameters
    "data_samples": 2000,
    "noise_std": 0.005,
    "segments": 11,
    "x_range": [-1.5, 1.5],
    "train_seg_ids": [2, 3, 4, 5, 6, 7, 8], 
    "width_size": 24,
    "mlp_batch_size": 64,

    # Expansion Hyperparameters
    "n_circles": 30*2,           
    
    # --- TRANSFORMER HYPERPARAMETERS ---
    "lr": 5e-6,      
    "transformer_epochs": 20000,
    "print_every": 2000,
    "transformer_batch_size": 4,      
    
    # New Params
    "transformer_d_model": 128*2,    # Embedding Dimension
    "transformer_n_heads": 4,      # Number of Heads
    "transformer_n_layers": 8,     # Number of Transformer Blocks
    "transformer_d_ff": 256//1,       # Feedforward dimension inside block: TODO: not needed atm, see forward pass.
    "transformer_substeps": 50,     # Number of micro-steps per macro step
    
    "transformer_target_step": 60*2,    # Total steps to unroll
    "scheduled_loss_weight": False,

    ## Consistency Loss Config
    "n_synthetic_points": 512,
    "consistency_loss_weight": 0.0,

    # Regularization Config
    "regularization_step": 40*2,     
    "regularization_weight": 0.0,  

    # Data Selection Mode
    "data_selection": "annulus",        ## "annulus" or "full_disk"
    "final_step_mode": "none",          ## "full" or "circle_only"
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

def gen_data(seed, n_samples, n_segments=3, x_range=[-3.0, 3.0], noise_std=0.1):
    """
    Generates a classical 1D-1D regression dataset: y = sin(3x) + 0.5x
    Segments are assigned based on spatial position (x-value), allowing
    for easy OOD splitting (e.g., train on segments [0,1], test on [2]).
    """
    np.random.seed(seed)
    
    # 1. Generate X uniformly across the full range
    x_min, x_max = x_range
    X = np.random.uniform(x_min, x_max, n_samples)
    
    # 2. Define the unchanging relation P(Y|X) (Concept)
    # y = sin(3x) + 0.5x is classic because it has both trend and periodicity
    Y = np.sin(10 * X) + 0.5 * X
    
    # Add noise
    noise = np.random.normal(0, noise_std, n_samples)
    Y += noise
    
    # 3. Create Segments spatially
    # We divide the x_range into n_segments equal distinct regions
    bins = np.linspace(x_min, x_max, n_segments + 1)
    
    # np.digitize returns indices 1..N, we want 0..N-1
    segs = np.digitize(X, bins) - 1
    
    # Clip to ensure bounds (in case of float precision issues at max edge)
    segs = np.clip(segs, 0, n_segments - 1)
    
    # 4. Format Output
    # data shape: (N, 2) -> [x, y]
    # segs shape: (N,)
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

# Precompute masks
dists = jnp.abs(X_train_full - x_mean).flatten()
radii = jnp.linspace(0.0, jnp.max(dists) + 0.01, CONFIG["n_circles"])
# circle_masks = jnp.stack([dists <= r for r in radii]) 
circle_masks = jnp.stack([dists < r for r in radii])        ## TODO: stricktly less than, because radius[0]=0.0 and we don't want any data in there

delta_radius = radii[1] - radii[0]
fake_radii = jnp.arange(radii[-1], radii[-1] + (CONFIG["transformer_target_step"]-CONFIG["n_circles"])*delta_radius + 0.01, delta_radius)
all_radii = jnp.concatenate([radii, fake_radii])

#%%
# --- 4. MODEL DEFINITIONS ---

width_size = CONFIG["width_size"]

class MLPModel(eqx.Module):
    layers: list
    def __init__(self, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        ## Overwite k1 to k3 with fixed seeds for reproducibility TODO
        # k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)
        self.layers = [eqx.nn.Linear(1, width_size, key=k1), jax.nn.relu,
                       eqx.nn.Linear(width_size, width_size, key=k2), jax.nn.relu,
                       eqx.nn.Linear(width_size, width_size, key=k4), jax.nn.relu,
                       eqx.nn.Linear(width_size, 1, key=k3)]
    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x
    def predict(self, x):
        return jax.vmap(self)(x)
# --- 2. Transformer Models (Decoder-Only) ---

class PositionalEncoding(eqx.Module):
    embedding: jax.Array
    def __init__(self, d_model: int, max_len: int = 2000):
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.embedding = pe
    def __call__(self, x, start_idx=0):
        return x + self.embedding[start_idx : start_idx + x.shape[0], :]

class TransformerBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    norm2: eqx.nn.LayerNorm
    
    def __init__(self, d_model, n_heads, d_ff, dropout, key):
        k1, k2 = jax.random.split(key)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=n_heads, query_size=d_model, dropout_p=dropout, key=k1
        )
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.mlp = eqx.nn.MLP(
            in_size=d_model, out_size=d_model, width_size=d_ff, depth=1, activation=jax.nn.gelu, key=k2
        )
        self.norm2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, mask=None, key=None):
        attn_out = self.attention(x, x, x, mask=mask, key=key)
        x = jax.vmap(self.norm1)(x + attn_out)
        mlp_out = jax.vmap(self.mlp)(x)
        x = jax.vmap(self.norm2)(x + mlp_out)
        return x

class DecoderOnlyTransformer(eqx.Module):
    embedding: eqx.nn.Linear
    pos_encoder: PositionalEncoding
    blocks: list
    output_projection: eqx.nn.Linear
    refinement_mlp: eqx.nn.MLP
    
    d_model: int = eqx.field(static=True)
    n_substeps: int = eqx.field(static=True)
    use_refinement: bool = eqx.field(static=True)

    def __init__(self, input_dim, d_model, n_heads, n_layers, d_ff, n_substeps, max_len, use_refinement, key):
        self.d_model = d_model
        self.n_substeps = n_substeps
        self.use_refinement = use_refinement
        
        keys = jax.random.split(key, 5)
        self.embedding = eqx.nn.Linear(input_dim, d_model, key=keys[0])
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout=0.0, key=k)
            for k in jax.random.split(keys[1], n_layers)
        ]
        
        self.output_projection = eqx.nn.Linear(d_model, input_dim, key=keys[2])
        # Zero Init Output
        self.output_projection = eqx.tree_at(
            lambda l: (l.weight, l.bias), 
            self.output_projection, 
            (jnp.zeros_like(self.output_projection.weight), jnp.zeros_like(self.output_projection.bias))
        )

        self.refinement_mlp = eqx.nn.MLP(
            in_size=d_model * 2, out_size=input_dim, width_size=d_ff, depth=1, activation=jax.nn.gelu, key=keys[3]
        )
        # Zero Init Refinement
        self.refinement_mlp = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias),
            self.refinement_mlp,
            (jnp.zeros_like(self.refinement_mlp.layers[-1].weight), jnp.zeros_like(self.refinement_mlp.layers[-1].bias))
        )

    def make_causal_mask(self, seq_len):
        idx = jnp.arange(seq_len)
        return idx[:, None] < idx[None, :]

    def refine_step(self, start_val, context_h):
        def loop_fn(i, curr_val):
            w_emb = self.embedding(curr_val) * jnp.sqrt(self.d_model)
            combined = jnp.concatenate([w_emb, context_h])
            delta = self.refinement_mlp(combined)
            return curr_val + delta
        return jax.lax.fori_loop(0, self.n_substeps, loop_fn, start_val)

    def __call__(self, x0, steps, ar_mode=True, key=None):
        history = x0[None, :]
        ## Future is a vector of zeros
        future = jnp.zeros((steps, x0.shape[0])) if steps is not None else None

        # Teacher Forcing Mode (Training)
        if not ar_mode:
            # assert future is not None
            full_seq = jnp.concatenate([history, future], axis=0)
            input_seq = full_seq[:-1]
            
            x = jax.vmap(self.embedding)(input_seq) * jnp.sqrt(self.d_model)
            x = self.pos_encoder(x)
            mask = self.make_causal_mask(x.shape[0])
            for block in self.blocks:
                x = block(x, mask=mask, key=key)
            
            if not self.use_refinement:
                deltas = jax.vmap(self.output_projection)(x)
                predictions_next = input_seq + deltas
            else:
                predictions_next = jax.vmap(self.refine_step)(input_seq, x)
            
            # Return only future part
            # return predictions_next[history.shape[0]-1:]

            ## Return everything
            return predictions_next

        # AR Mode (Inference)
        else:
            if steps is None: steps = future.shape[0] if future is not None else 10
            total_len = history.shape[0] + steps
            
            # Setup buffer
            buffer = jnp.zeros((total_len, history.shape[1]))
            buffer = buffer.at[:history.shape[0]].set(history)
            start_gen_idx = history.shape[0]
            
            def scan_fn(current_buffer, step_idx):
                x = jax.vmap(self.embedding)(current_buffer) * jnp.sqrt(self.d_model)
                x = self.pos_encoder(x)
                causal_mask = self.make_causal_mask(total_len)
                
                for block in self.blocks:
                    x = block(x, mask=causal_mask, key=key)
                
                curr_ptr = start_gen_idx + step_idx
                context_h = x[curr_ptr - 1]
                input_val = current_buffer[curr_ptr - 1]
                
                if not self.use_refinement:
                    delta = self.output_projection(context_h)
                    next_val = input_val + delta
                else:
                    next_val = self.refine_step(input_val, context_h)
                
                new_buffer = current_buffer.at[curr_ptr].set(next_val)
                return new_buffer, next_val

            indices = jnp.arange(steps)
            _, predictions = jax.lax.scan(scan_fn, buffer, indices)
            return predictions


#%%
# --- 5. INITIALIZATION & BATCH GENERATION ---

key = jax.random.PRNGKey(CONFIG["seed"])
k_init, k_tf, key = jax.random.split(key, 3)

# 1. Setup Model Structure (Static)
model_template = MLPModel(k_init)
params_template, static = eqx.partition(model_template, eqx.is_array)
flat_template, shapes, treedef, mask = flatten_pytree(params_template)
input_dim = flat_template.shape[0]
print(f"MLP Model Parameter Count: {input_dim}")

# 2. Generate Batch of Initial States
print(f"Generating {CONFIG['transformer_batch_size']} initial states...")
# x0_batch_list = []
gen_key = jax.random.PRNGKey(CONFIG["seed"] + 100)

# for _ in range(CONFIG["transformer_batch_size"]):
#     gen_key, sk = jax.random.split(gen_key)
#     m = MLPModel(sk)
#     p, _ = eqx.partition(m, eqx.is_array)
#     f, _, _, _ = flatten_pytree(p)
#     x0_batch_list.append(f)

# x0_batch = jnp.stack(x0_batch_list)

# def gen_x0_batch(batch_size, key):
#     x0_batch_list = []
#     gen_key = key

#     for _ in range(batch_size):
#         gen_key, sk = jax.random.split(gen_key)
#         m = MLPModel(sk)
#         p, _ = eqx.partition(m, eqx.is_array)
#         f, _, _, _ = flatten_pytree(p)
#         x0_batch_list.append(f)
#         ## Devide by 1000 t0 get the params in a smaller range  TODO
#         # x0_batch_list.append(f / 100.0)
#         # x0_batch_list.append(f * 0.0)

#     x0_batch = jnp.stack(x0_batch_list) 
#     return x0_batch


def gen_x0_batch(batch_size, key):
    x0_batch_list = []
    # main_key = jax.random.PRNGKey(42)
    gen_key = key

    for _ in range(batch_size):
        gen_key, sk = jax.random.split(gen_key)
        m = MLPModel(sk)
        p, _ = eqx.partition(m, eqx.is_array)
        f, _, _, _ = flatten_pytree(p)

        # ## Perturb slightly around the fixed init model
        # eps = jax.random.uniform(gen_key, shape=f.shape, minval=-1e-1, maxval=1e-1)
        # f = f + eps

        # ## Let's pick 10 paramters at random, and perturb them only
        # eps = jax.random.uniform(gen_key, shape=(10,), minval=-1e-4, maxval=1e-4)
        # param_indices = jax.random.choice(gen_key, f.shape[0], shape=(10,), replace=False)
        # f = f.at[param_indices].add(eps)
    

        # eps = jax.random.uniform(gen_key, shape=f.shape[0], minval=-1, maxval=1)

        ## Small gaussian noise
        # eps = jax.random.normal(gen_key, shape=f.shape) * 1e-2
        # x0_batch_list.append(f + eps)

        # x0_batch_list.append(f*0.0)
        x0_batch_list.append(f/100.0)
        # x0_batch_list.append(f/1.0)

    x0_batch = jnp.stack(x0_batch_list) 
    return x0_batch

x0_batch = gen_x0_batch(CONFIG["transformer_batch_size"], gen_key)

# # 3. Init Transformer
# tf_model = WeightTransformer(
#     input_dim=input_dim,
#     d_model=CONFIG["transformer_d_model"],
#     n_heads=CONFIG["transformer_n_heads"],
#     n_layers=CONFIG["transformer_n_layers"],
#     d_ff=CONFIG["transformer_d_ff"],
#     max_len=CONFIG["transformer_target_step"],
#     n_substeps=CONFIG["transformer_substeps"],
#     key=k_tf
# )

# # 3. Init Decoder-Only Transformer
tf_model = DecoderOnlyTransformer(
    input_dim=input_dim,
    d_model=CONFIG["transformer_d_model"],
    n_heads=CONFIG["transformer_n_heads"],
    n_layers=CONFIG["transformer_n_layers"],
    d_ff=CONFIG["transformer_d_ff"],
    n_substeps=CONFIG["transformer_substeps"],
    max_len=CONFIG["transformer_target_step"],
    use_refinement=True,
    key=k_tf
)

# opt = optax.adam(CONFIG["lr"]) 
opt = optax.adabelief(CONFIG["lr"]) 
# opt = optax.chain(
#     optax.clip(1e-5),
#     optax.adabelief(CONFIG["lr"]),
# )
opt_state = opt.init(eqx.filter(tf_model, eqx.is_array))

#%%
# --- 6. END-TO-END TRAINING LOOP ---

def get_functional_loss(flat_w, step_idx, key=None):
    # Unflatten MLP
    params = unflatten_pytree(flat_w, shapes, treedef, mask)
    model = eqx.combine(params, static)
    
    y_pred = model.predict(X_train_full)
    residuals = (y_pred - Y_train_full) ** 2
    # ## We don't want to use all of X_train_full, only a randmly selected subset, like a batch
    # n_data_points = X_train_full.shape[0]
    # if key is None:
    #     selected_indices = jnp.arange(n_data_points)
    # else:
    #     key, subkey = jax.random.split(key)
    #     selected_indices = jax.random.choice(subkey, n_data_points, shape=(min(32, n_data_points),), replace=False)
    # X_batch = X_train_full[selected_indices]
    # Y_batch = Y_train_full[selected_indices]
    # y_pred = model.predict(X_batch)
    # residuals = (y_pred - Y_batch) ** 2
    
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
        

    ## Randmly disable n_active-32 datapoitns in active_masks, until exactly 32 points are left to use TODO
    n_active = jnp.sum(active_mask)

    def disable_to_32(active_mask, key):
        # 1. Generate random noise for every point
        noise = jax.random.uniform(key, shape=active_mask.shape)
        
        # 2. Mask out currently inactive points by setting their score to -infinity.
        #    This ensures we only select from the currently active points.
        scores = jnp.where(active_mask, noise, -jnp.inf)
        
        # 3. Find the indices of the top 32 scores.
        #    jax.lax.top_k requires a static integer (32), which works perfectly here.
        _, keep_indices = jax.lax.top_k(scores, CONFIG["mlp_batch_size"])
        
        # 4. Create the new mask: Start with all False, then set the winners to True.
        new_mask = jnp.zeros_like(active_mask, dtype=jnp.bool_)
        new_mask = new_mask.at[keep_indices].set(True)
        
        return new_mask

    # Run the condition
    # If n > 32: randomly subsample down to 32.
    # If n <= 32: keep the mask as is.
    active_mask = jax.lax.cond(
        n_active > CONFIG["mlp_batch_size"], 
        disable_to_32, 
        lambda m, k: m, 
        active_mask, 
        key
    )

    mask_sum = jnp.sum(active_mask)
    base_loss = jnp.sum(residuals * active_mask[:, None]) / (mask_sum + 1e-6)
    
    eff_weight = jax.lax.select(is_reg_step, CONFIG["regularization_weight"], 1.0)
    
    final_loss = base_loss * eff_weight

    if not CONFIG["scheduled_loss_weight"]:
        return final_loss
    else:
        return (base_loss * eff_weight) / (step_idx**2 + 1)


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
    circle_idx = jnp.minimum(step_idx_in, CONFIG["transformer_target_step"] - 1)
    radius = all_radii[circle_idx]
    n_synthetic = CONFIG["n_synthetic_points"]
    angles = jax.random.uniform(key, shape=(n_synthetic,)) * 2 * jnp.pi
    
    ## Uniform sampling
    # radii_sampled = jax.random.uniform(key, shape=(n_synthetic,)) * radius

    ## Sampling with higher density near the perimeter (closer to radius). Use the beta distribution with alpha>1
    radii_sampled = jax.random.beta(key, a=5.0, b=1.0, shape=(n_synthetic,)) * radius     ## TODO: put this back !
    # radii_sampled = jax.random.uniform(key, shape=(n_synthetic,), minval=0.9, maxval=1.1) * radius

    X_synthetic = x_mean + radii_sampled * jnp.cos(angles)      ## TODO: add a dimention along axis 1 if x is multi-dim?
    
    y_pred_in = model_in.predict(X_synthetic[:, None])
    y_pred_out = model_out.predict(X_synthetic[:, None])
    residuals = (y_pred_in - y_pred_out) ** 2

    return jnp.mean(residuals)


def get_disconsistency_loss(flat_w_in, flat_w_out, step_idx_in, key):
    residual = jnp.mean((flat_w_in - flat_w_out) ** 2)
    return jnp.maximum(0.0, 1.0 - residual)   ## Hinge loss style

@eqx.filter_value_and_grad
def train_step_fn(model, x0_batch, key):
    total_steps = CONFIG["transformer_target_step"]
    
    # VMAP over batch
    preds_batch = jax.vmap(model, in_axes=(0, None, None))(x0_batch, total_steps, None) # (Batch, Steps, D)

    # step_indices = jnp.arange(total_steps)
    # preds_batch_data = preds_batch

    # step_indices = jnp.array([0, CONFIG["n_circles"]//2, CONFIG["n_circles"]-1])       ## TODO
    # step_indices = jnp.array([CONFIG["n_circles"]//2, CONFIG["n_circles"]-1])             ## TODO
    # chose_from = jnp.arange(1, CONFIG["n_circles"])
    # step_indices = jax.random.choice(key, CONFIG["n_circles"], shape=(25,), replace=False)
    # ## Always add 0
    # step_indices = jnp.concatenate([jnp.array([0]), step_indices])

    # step_indices = jnp.arange(CONFIG["n_circles"])
    step_indices = jax.random.choice(key, CONFIG["n_circles"], shape=(2,), replace=False)
    preds_batch_data = preds_batch[:, step_indices, :]

    keys = jax.random.split(key, len(step_indices))
    def loss_per_seq(seq):
        return jax.vmap(get_functional_loss)(seq, step_indices, keys)
    losses_batch = jax.vmap(loss_per_seq)(preds_batch_data) # (Batch, Steps)
    total_data_loss = jnp.mean(jnp.sum(losses_batch, axis=1))

    # Consistency Loss
    step_indices = jnp.arange(1, CONFIG["transformer_target_step"])
    keys = jax.random.split(key, len(step_indices)-2)
    preds_batch_cons = preds_batch[:, step_indices, :]
    def cons_loss_per_seq(seq):
        # return jax.vmap(get_disconsistency_loss)(seq[:-1], seq[1:], step_indices[:-1], keys)
        return jax.vmap(get_consistency_loss)(seq[1:-1], seq[2:], step_indices[1:-1], keys)
    cons_losses_batch = jax.vmap(cons_loss_per_seq)(preds_batch_cons) 
    total_cons_loss = jnp.mean(jnp.sum(cons_losses_batch, axis=1))

    total_loss = total_data_loss + CONFIG["consistency_loss_weight"]*total_cons_loss

    # ## Let's penalise large prediction trajectories up to n_circles only
    # inital_preds = preds_batch[:, :CONFIG["n_circles"], :]
    # norm_loss = jnp.mean(jnp.sum(inital_preds**2, axis=(1,2)))
    # total_loss += 1e-3 * norm_loss

    # ## Let's penalise large differences between consecutive steps (for the entire sequence)
    # diffs = preds_batch[:, 1:, :] - preds_batch[:, :-1, :]
    # smoothness_penalty = jnp.mean(jnp.sum(diffs**2, axis=(1,2)))
    # total_loss += 1e-3 * smoothness_penalty

    # ## Let's penalise large values in the sequence (for the n_cirlles-1 step only)
    # abs_penalty = jnp.mean(jnp.sum(jnp.abs(preds_batch), axis=(1,2)))
    # # abs_penalty = jnp.mean(jnp.sum(jnp.abs(preds_batch[:, CONFIG["n_circles"]-1, :]), axis=1))
    # total_loss += 1e-3 * abs_penalty

    ## Let's make sure no prediction is above 1 in absolute value
    max_val = jnp.max(jnp.abs(preds_batch))
    total_loss += 1e-1 * jax.nn.relu(max_val - 20.0)

    return total_loss

@eqx.filter_jit
def make_step(model, opt_state, x0_batch, key):
    loss, grads = train_step_fn(model, x0_batch, key)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

if TRAIN:
    print(f"🚀 Starting Batch Transformer Training.")
    
    loss_history = []
    train_key = jax.random.PRNGKey(CONFIG["seed"] + 99)
    best_model = tf_model      
    lowest_loss = float('inf')

    for ep in range(CONFIG["transformer_epochs"]):
        train_key, step_key = jax.random.split(train_key)

        # if ep % 10 == 0:
        x0_batch = gen_x0_batch(CONFIG["transformer_batch_size"], step_key)     ## TODO: remmeber to remove this, as we previsouly had this fixed

        tf_model, opt_state, loss = make_step(tf_model, opt_state, x0_batch, step_key)
        loss_history.append(loss)

        if loss < lowest_loss:
            lowest_loss = loss
            best_model = tf_model
        
        if (ep+1) % CONFIG["print_every"] == 0:
            # print(f"Epoch {ep+1} | Loss: {loss:.6f}", flush=True)
            ## Log current time as well
            # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] Epoch {ep+1} | Loss: {loss:.6f}", flush=True)

        ## Save five checkpoints during training
        if (ep+1==CONFIG["transformer_epochs"]) or ((ep+1) % (CONFIG["transformer_epochs"] // 5)) == 0:
            eqx.tree_serialise_leaves(artefacts_path / f"tf_model_ep{ep+1}.eqx", tf_model)
            np.save(artefacts_path / f"loss_history_ep{ep+1}.npy", np.array(loss_history))

    tf_model = best_model

    eval_key = jax.random.PRNGKey(42)
    final_batch_traj = jax.vmap(tf_model, in_axes=(0, None, None))(x0_batch, CONFIG["transformer_target_step"], None)
    
    np.save(artefacts_path / "final_batch_traj.npy", final_batch_traj)
    np.save(artefacts_path / "loss_history.npy", np.array(loss_history))
    eqx.tree_serialise_leaves(artefacts_path / "tf_model.eqx", tf_model)

else:
    print("Loading results...")
    final_batch_traj = np.load(artefacts_path / "final_batch_traj.npy")
    loss_history = np.load(artefacts_path / "loss_history.npy")
    tf_model = eqx.tree_deserialise_leaves(artefacts_path / "tf_model.eqx", tf_model)

final_traj = final_batch_traj[0]

#%%
# --- 7. VISUALIZATION ---
print("\n=== Generating Dashboards ===")
x_grid = jnp.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], 300)[:, None]

# --- DASHBOARD 2: FUNCTIONAL EVOLUTION ---
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

# --- DASHBOARD 1: BATCH LIMITS ---
print("Generating Batch Limits Dashboard...")
fig_batch = plt.figure(figsize=(20, 8))
gs_batch = fig_batch.add_gridspec(1, 3)

steps_to_plot = [CONFIG["n_circles"], CONFIG["regularization_step"], CONFIG["transformer_target_step"] - 1]
# steps_to_plot = [CONFIG["n_circles"], 1, CONFIG["transformer_target_step"] - 1]
titles = ["End of Circles", "Regularization Step", "Final Limit"]

for i, step_idx in enumerate(steps_to_plot):
    ax = fig_batch.add_subplot(gs_batch[0, i])
    ax.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.05)
    ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1)
    
    for b in range(CONFIG["transformer_batch_size"]):
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

#%% Special plot paramter trajectories
print("Generating Extended Parameter Trajectories Plot...")
fig, ax = plt.subplots(figsize=(7, 10), nrows=1, ncols=1) 
traj_seed_0 = final_batch_traj[0]
# plot_ids = np.arange(100)  # First 100 parameters
plot_ids = jax.random.choice(jax.random.PRNGKey(42), traj_seed_0.shape[1], shape=(100,), replace=False)

# plot_up_to = CONFIG["n_circles"]
plot_up_to = CONFIG["transformer_target_step"]

for idx in plot_ids:
    ## Plot the difference x_t - x_(t-1)
    traj = traj_seed_0[:plot_up_to, idx]
    # traj_diff = jnp.concatenate([jnp.array([0.0]), traj[1:] - traj[:-1]])
    # ax.plot(np.arange(plot_up_to), traj_diff, linewidth=1.5, label=f"Param {idx}")

    ax.plot(np.arange(plot_up_to), traj, linewidth=1.5, label=f"Param {idx}")



## Plot vertical lines (One for the n_circles step, one for the regularization step
ax.axvline(CONFIG["n_circles"], color='k', linestyle='--', label="End of Data")
ax.axvline(CONFIG["regularization_step"], color='red', linestyle=':', label="Reg Step")

ax.set_title("Parameter Trajectories")
ax.set_xlabel("Step")
ax.set_ylabel("Parameter Value")
plt.tight_layout()
plt.savefig(plots_path / "extended_parameter_trajectories.png")
plt.show()

#%% Model Predictions Corresponding to Circles plot
print("Generating n_circles Model Prediction Plot (Circle-Specific Models)...")
# step_idx = 35 
dists = jnp.abs(X_train_full - x_mean).flatten()

def predict_circle_specific_loss(final_batch_traj, X_data):
    circle_losses = {}
    n_points = X_data.shape[0]

    for circle_idx in range(CONFIG["transformer_target_step"]):
    # for circle_idx in [CONFIG["n_circles"]-1]:
    # for circle_idx in [CONFIG["transformer_target_step"]-1]:
    # for circle_idx in [-1]:
    # for circle_idx in [175]:
        w = final_batch_traj[0, circle_idx]  
        p = unflatten_pytree(w, shapes, treedef, mask)
        m = eqx.combine(p, static)

        circle_masks = []
        for r in all_radii:
            new_mask = jnp.abs(X_data - x_mean) <= r
            circle_masks.append(new_mask.flatten())

        ## Outward circle idex is know. We want to plot th edata in the annulus between this circle and the previous one (if circle_idx>0)
        in_circle_idx = circle_idx-1 if circle_idx > 0 else 0
        circle_masks = jnp.array(circle_masks)
        ring_mask = jnp.logical_and(circle_masks[circle_idx], ~circle_masks[in_circle_idx])
        X_circle = X_data[ring_mask]

        # circle_mask = circle_masks[circle_idx]
        # X_circle = X_data[circle_mask]

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

# X_circles of shape (N_circle, 1), y_pred of shape (N_circle, 1)

for circle_idx, (X_circle, y_pred) in train_preds_cc.items():
    ax.scatter(X_circle, y_pred, c='green', s=1, alpha=0.3)
for circle_idx, (X_circle, y_pred) in test_preds_cc.items():
    ax.scatter(X_circle, y_pred, c='red', s=1, alpha=0.3)

# ax.set_title(f"Model Predictions Corresponding to Circles (Circle-Specific Models)")
ax.set_title(f"Model Predictions At Final Time Step")
ax.set_ylim(jnp.min(Y_train_full)-1, jnp.max(Y_train_full)+1)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(plots_path / "model_predictions_n_circles_circle_specific.png")
plt.show()