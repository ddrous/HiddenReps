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

# Set plotting style
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

## Jax config stop on NaNs
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)


# --- 1. CONFIGURATION ---
TRAIN = True  
RUN_DIR = "."

CONFIG = {
    # "seed": time.time_ns() % (2**32 - 1),
    "seed": 2028,

    "x_range": [-4.0, 4.0],  # Wider range to see the sine wave repeat
    "segments": 5,           # Split into 5 distinct vertical strips
    "train_seg_ids": [1, 2, 3], # Train on the middle

    # Data & MLP Hyperparameters
    "data_samples": 2000,
    "noise_std": 0.005,
    "segments": 11,
    "x_range": [0.175, 1.5],
    "train_seg_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8], 
    "width_size": 24,
    "mlp_batch_size": 64,

    # Expansion Hyperparameters
    "n_circles": 30*2,           
    "warmup_steps": 0,

    # --- TRANSFORMER HYPERPARAMETERS ---
    "lr": 5e-5,      
    "transformer_epochs": 7000*5,
    "print_every": 2500,
    "transformer_batch_size": 1,      

    # New Params
    "transformer_d_model": 128*4,    # Embedding Dimension
    "transformer_n_heads": 1,      # Number of Heads
    "transformer_n_layers": 1,     # Number of Transformer Blocks
    "transformer_d_ff": 1//1,       # Feedforward dimension inside block: TODO: not needed atm, see forward pass.
    "transformer_substeps": 50,     # Number of micro-steps per macro step
    "kl_weight": 1e-1,          # Weight on KL divergence loss

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



def gen_data_linear(seed, n_samples, n_segments=3, local_structure="random", 
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


def gen_data_air():
    data_path = "air_quality.csv"  # Update this path if needed
    df = pd.read_csv(data_path)
    
    ## Normalise the dataframe (Optional, but can help with visualization)
    scaler = StandardScaler()
    df[['PT08.S3.NOx.', 'PT08.S5.O3.']] = scaler.fit_transform(df[['PT08.S3.NOx.', 'PT08.S5.O3.']])

    ## x corresponds to O3 sensor, y corresponds to NOx sensor
    X = df['PT08.S5.O3.'].values
    Y = df['PT08.S3.NOx.'].values

    ## Split in two segments based on O3 values (arbitrary threshold at 1.0 after normalization). seg 0 for train, seg 1 for test
    segs = (X > 1.0).astype(int)
    data = np.column_stack((X, Y))
    return data, segs

run_path = setup_run_dir()
artefacts_path = run_path / "artefacts"
plots_path = run_path / "plots"

if TRAIN:
    SEED = CONFIG["seed"]
    data, segs = gen_data(SEED, CONFIG["data_samples"], CONFIG["segments"], CONFIG["x_range"], CONFIG["noise_std"])

    # data, segs = gen_data_linear(SEED, CONFIG["data_samples"], n_segments=CONFIG["segments"], 
    #                   local_structure="gradual_increase", x_range=CONFIG["x_range"], 
    #                   slope=0.5, base_intercept=-0.4, step_size=0.1, noise_std=CONFIG["noise_std"])

    # data, segs = gen_data_air()

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

x_mean = jnp.min(X_train_full)
# x_mean = jnp.min(X_train_full)
print(f"Data Center (Mean): {x_mean:.4f}")

# Precompute masks
dists = jnp.abs(X_train_full - x_mean).flatten()
radii = jnp.linspace(0.0, jnp.max(dists) + 0.01, CONFIG["n_circles"])
circle_masks = jnp.stack([dists <= r for r in radii]) 
# circle_masks = jnp.stack([dists < r for r in radii])        ## TODO: stricktly less than, because radius[0]=0.0 and we don't want any data in there

delta_radius = radii[1] - radii[0]
fake_radii = jnp.arange(radii[-1], radii[-1] + (CONFIG["transformer_target_step"]-CONFIG["n_circles"])*delta_radius + 0.01, delta_radius)
all_radii = jnp.concatenate([radii, fake_radii])

#%%
# --- 4. MODEL DEFINITIONS ---

width_size = CONFIG["width_size"]
real_output_size = 1

class MLPModel(eqx.Module):
    layers: list
    def __init__(self, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        ## Overwite k1 to k3 with fixed seeds for reproducibility TODO
        # k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)

        # self.layers = [eqx.nn.Linear(1, width_size, key=k1), jax.nn.relu,
        #             #    eqx.nn.Linear(width_size, width_size, key=k2), jax.nn.relu,
        #             #    eqx.nn.Linear(width_size, width_size, key=k4), jax.nn.relu,
        #                eqx.nn.Linear(width_size, 1, key=k3)]

        self.layers = [eqx.nn.Linear(1, 1, use_bias=True, key=k1)]

    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x
    def predict(self, x):
        return jax.vmap(self)(x)


# ## Let's just test the Unet1D forward pass
# random_input = jnp.ones((1273, 1))  # (channels, length)
# Unet = Unet1D(
#     in_shape=(1273, 1),
#     out_shape=(1273, 1),
#     base_chans=16,
#     levels=3,
#     use_normalization=False,
#     key=jax.random.PRNGKey(0),
#     cond_dim=None,
#     use_conditioning=False
# )
# output = Unet(0.0, random_input, None)
# print(f"Input shape: {random_input.shape}")
# print(f"Output shape: {output.shape}")

## Number of params
def count_params(module):
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array)))
    return param_count

# print("Unet1D Parameter Count:", count_params(Unet))





#%%


class LinearRNN(eqx.Module):
    A: jax.Array
    B: jax.Array
    y_init: jax.Array  # Contains [y_0, y_{-1}]
    data_dim: int

    def __init__(self, data_dim, hidden_dim, key):
        self.data_dim = data_dim
        
        # Initialize A as Identity and B as Zeros
        problem_dim = data_dim*2
        self.A = jnp.eye(problem_dim)
        self.B = jnp.zeros((problem_dim, problem_dim))
        
        # y_init stores the two initial points needed for second-order recurrence
        # Shape: (2, data_dim)
        self.y_init = jax.random.normal(key, shape=(2, problem_dim)) * 0e-2

        # # y_init_mean = jnp.zeros((data_dim,)) 
        # y_init_mean = jax.random.normal(key, shape=(data_dim,)) * 1e-2
        # y_init_sigma = jnp.ones((data_dim,))*(-3)
        # y_init = jnp.concatenate([y_init_mean, y_init_sigma, jnp.zeros(problem_dim-2*data_dim,)])  # Shape: (data_dim*3)
        # self.y_init = jnp.repeat(y_init[None, :], 2, axis=0)  # Shape: (2, data_dim*3)

    def __call__(self, y0, steps, key):
        # Initial state for the scan: (y_{t-1}, y_{t-2})
        # We've initialized y_init such that y_init[0] is y_0 and y_init[1] is y_{-1}
        init_state = (self.y_init[0], self.y_init[1])

        # init_state1 = jnp.concatenate([y0, jnp.zeros((self.data_dim*3 - y0.shape[0],))])
        # init_state = (init_state1, init_state1)

        def scan_fn(state, _):
            y_prev1, y_prev2 = state
            
            # Recurrence: y_t = A y_{t-1} + B(y_{t-1} - y_{t-2})
            # y_next = self.A @ y_prev1 + self.B @ (y_prev1 - y_prev2)
            y_next = self.A @ y_prev1
            
            # New state shifts: y_next becomes y_{t-1}, y_prev1 becomes y_{t-2}
            new_state = (y_next, y_prev1)
            return new_state, y_next

        # Use jax.lax.scan to iterate over the number of steps
        _, ys = jax.lax.scan(scan_fn, init_state, None, length=steps)

        # ## Sample from predicted mean and stddev
        # means = ys[:, :self.data_dim]
        # stddevs = ys[:, self.data_dim:2*self.data_dim]
        # eps = jax.random.normal(key, shape=stddevs.shape)
        # ys = means + eps * stddevs

        return ys

# Example usage:
# key = jax.random.PRNGKey(0)
# model = LinearRNN(data_dim=16, key=key)
# output = model(steps=10) # Shape: (10, 16)



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
        eps = jax.random.normal(gen_key, shape=f.shape) * 1e-2
        x0_batch_list.append(eps)

        # x0_batch_list.append(f*0.0)
        # x0_batch_list.append(f/100.0)
        # x0_batch_list.append(f/1000.0)

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

# # # 3. Init NeuralODE Model
# tf_model = NeuralODE(
#     data_dim=input_dim,
#     hidden_dim=CONFIG["transformer_d_model"],
#     key=k_tf
# )

# 3. Init LinearRNN Model
tf_model = LinearRNN(
    data_dim=input_dim,
    hidden_dim=CONFIG["transformer_d_model"],
    key=k_tf
)

print(f"Transformer / Neural ODE Parameter Count: {count_params(tf_model)}")

# opt = optax.adam(CONFIG["lr"]) 
opt = optax.adabelief(CONFIG["lr"]) 
# opt = optax.chain(
#     optax.clip(1e-5),
#     optax.adabelief(CONFIG["lr"]),
# )
opt_state = opt.init(eqx.filter(tf_model, eqx.is_array))

#%%
# --- 6. END-TO-END TRAINING LOOP ---

# def negative_log_likelihood(theta_mu, theta_sigma, X, Y=None):
#     """Computes the negative log-likelihood of the observed data under a Gaussian model.
#         We use Bayesian inference intending to maginalise over all possible thetas that could have generated the data.
#         We use a the Laplace approximation, linearising the model around the predicted theta_mu and using the predicted theta_sigma as the variance of the Gaussian. 
#     """

#     # print("Actual values ot theta and sigma:", theta_mu, theta_sigma)

#     ## First, we want to write f(x;\theta) = f(x;theta_mu) + grad_theta f(x;theta_mu) @ (theta - theta_mu)
#     def model_fn(theta, x):
#         params = unflatten_pytree(theta, shapes, treedef, mask)
#         model = eqx.combine(params, static)
#         return model.predict(x)     ## Shape (nb_data_points, output_dim)

#     mean_y_pred = model_fn(theta_mu, X)  # Shape: (n_data_points, output_dim)
#     jacobian = eqx.filter_jacfwd(model_fn)(theta_mu, X)  # Shape: (n_data_points, output_dim, input_dim)

#     ## Next we calculate the covaraance matrix Cov(y) = J @ Cov(theta) @ J^T + sigma^2 I
#     # theta_cov = jnp.diag(theta_sigma**2)  # Shape: (input_dim, input_dim)
#     theta_cov = jnp.diag(theta_sigma**2) + jnp.eye(theta_sigma.shape[0]) * 1e-6   # Add small value to diagonal for numerical stability

#     # cov_y_pred = jacobian @ theta_cov @ jnp.transpose(jacobian, (0, 2, 1)) + jnp.eye(mean_y_pred.shape[0]) * 1e-6  # Shape: (n_data_points, output_dim, output_dim)
#     cov_y_pred = jacobian @ theta_cov @ jnp.transpose(jacobian, (0, 2, 1)) 

#     # print("Covy squeezed:", cov_y_pred.squeeze())

#     if Y is None:
#         return None, mean_y_pred, cov_y_pred

#     ## Finally, we compute the negative log-likelihood of the observed data under the predicted Gaussian distribution
#     diff = Y - mean_y_pred  # Shape: (n_data_points, output_dim)
#     inv_cov_y_pred = jnp.linalg.inv(cov_y_pred)  # Shape: (n_data_points, output_dim, output_dim)

#     # We should have one term for each data point, but we can average them together
#     diff_expanded = diff[:, :, None]  # Shape: (n_data_points, output_dim, 1)
#     nll = 0.5 * jnp.sum(diff_expanded * (inv_cov_y_pred @ diff_expanded), axis=(1, 2)) + 0.5 * jnp.log(jnp.linalg.det(cov_y_pred) + 1e-6)

#     return nll, mean_y_pred, cov_y_pred


def negative_log_likelihood(theta_mu, log_theta_sigma, X, Y, use_inverse=True, noise_sigma=1e-3):
    """
    Computes the NLL of the observed data under a linearized Laplace approximation.
    
    Args:
        theta_mu: Mean parameters (flat vector).
        log_theta_sigma: Log standard deviation of parameters (flat vector).
        X: Inputs (N, input_dim).
        Y: Targets (N, output_dim).
        use_inverse: If True, uses direct matrix inversion (simpler but less stable).
                     If False, uses Cholesky decomposition (standard, stable).
        [shapes, treedef, mask, static]: Context variables for model reconstruction.
    """
    
    # --- 0. Helper to reconstruct and predict ---
    def model_fn(theta, x):
        # Reconstruct params from flat vector
        params = unflatten_pytree(theta, shapes, treedef, mask)
        model = eqx.combine(params, static)
        return model.predict(x)     ## Vectorized predict: Shape (N, p)

    N = X.shape[0]

    # --- 1. Linearize Model (Mean & Jacobian) ---
    # mean_y: (N, p)
    # jacobian: (N, p, d) where p is output_dim, d is param_dim
    mean_y = model_fn(theta_mu, X)
    jacobian = jax.jacfwd(model_fn)(theta_mu, X)

    p = mean_y.shape[-1] # output dimension

    # --- 2. Construct Output Covariance ---
    # Theta variance (diagonal)
    # theta_var = jnp.exp(log_theta_sigma * 2) + 1e-6
    # theta_var = jnp.exp(log_theta_sigma * 2)
    theta_var = log_theta_sigma ** 2
    # theta_std = jnp.sqrt(theta_var)
    theta_std = log_theta_sigma

    # Optimization: Instead of full matrix multiply J @ Cov @ J.T,
    # we scale J by std_dev and do J_scaled @ J_scaled.T
    # scaled_jacobian shape: (N, p, d)
    scaled_jacobian = jacobian * theta_std[None, None, :]
    
    # Cov_y = J_scaled @ J_scaled^T + Noise
    # Result shape: (N, p, p)
    cov_y = jnp.einsum('npd,nqd->npq', scaled_jacobian, scaled_jacobian)
    
    # Add observation noise (Nugget) for stability
    noise_variance = noise_sigma ** 2
    cov_y = cov_y + jnp.eye(p) * noise_variance

    if Y is None:
        return None, mean_y, cov_y

    # --- 3. Compute Residuals ---
    diff = Y - mean_y # (N, p)

    # --- 4. Calculate NLL (Branching Logic) ---
    
    if use_inverse:
        # METHOD A: Direct Inverse
        # 1. Invert Covariance
        cov_inv = jnp.linalg.inv(cov_y) # (N, p, p)
        
        # 2. Quadratic Form: 0.5 * (y-mu)^T @ inv @ (y-mu)
        # diff is (N, p), we need (N, 1, p) @ (N, p, p) @ (N, p, 1) -> (N, 1, 1)
        # Using einsum for cleaner batch operation:
        quad_form = 0.5 * jnp.einsum('np,npq,nq->n', diff, cov_inv, diff)
        
        # 3. Log Determinant
        _, log_det = jnp.linalg.slogdet(cov_y) # (N,)
        log_det_term = 0.5 * log_det
        
    else:
        # METHOD B: Cholesky Solve (Numerically Stable)
        # 1. Cholesky Decomposition: cov = L @ L.T
        L = jnp.linalg.cholesky(cov_y) # (N, p, p)
        
        # 2. Solve for Quadratic Form
        # We want: diff^T @ Cov^-1 @ diff
        # Since Cov = L @ L.T, Cov^-1 = L^-T @ L^-1
        # Let z = L^-1 @ diff  =>  L @ z = diff
        
        # Solve Lz = diff (Triangular solve)
        # L is lower triangular. shape (N, p, p), diff is (N, p) -> (N, p, 1)
        # We need to reshape diff for generic solve or use vmap
        def solve_single(Li, diffi):
            # z = Li \ diffi
            z = jax.lax.linalg.triangular_solve(Li, diffi[:, None], lower=True, left_side=True)
            return jnp.squeeze(z) # back to vector

        z = jax.vmap(solve_single)(L, diff)
        
        # Quad form is 0.5 * ||z||^2
        quad_form = 0.5 * jnp.sum(z**2, axis=-1)
        
        # 3. Log Determinant using Cholesky
        # log(det(Cov)) = 2 * sum(log(diag(L)))
        log_det_term = jnp.sum(jnp.log(jnp.diagonal(L, axis1=1, axis2=2)), axis=-1)

    # --- 5. Combine Terms ---
    all_nll = quad_form + log_det_term
    
    # Add constant terms: 0.5 * N * p * log(2*pi)
    # constant_term = 0.5 * N * p * jnp.log(2 * jnp.pi)
    # all_nll += 0.5 * N * p * jnp.log(2 * jnp.pi)
    
    return all_nll, mean_y, cov_y

def compute_kl_divergence(mu, sigma):
    """
    Computes KL(q||p) where q = N(mu, sigma^2) and p = N(0, 1).
    Formula: 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
    """
    # Ensure sigma is positive and adds stability
    var = sigma**2 + 1e-6
    log_var = jnp.log(var)
    
    kl = 0.5 * jnp.sum(var + mu**2 - 1.0 - log_var)
    return kl

def compute_kl_divergence_emp(mu, sigma, emp_means):
    """
    Computes KL(q||p) where q = N(mu, sigma^2) and p = N(empirical_means, I).
    """
    var = sigma**2 + 1e-6
    log_var = jnp.log(var)
    
    kl = 0.5 * jnp.sum(var + (mu - emp_means)**2 - 1.0 - log_var)
    return kl

def compute_output_kl(pred_mean, pred_cov, prior_mean=0.0, prior_sigma=3.0):
    """
    Computes KL divergence between predicted Gaussian and a fixed Prior Gaussian.
    
    Args:
        pred_mean: (N, D) Predicted means
        pred_cov: (N, D, D) Predicted covariance matrices
        prior_mean: Scalar or (D,) target mean (default 0.0)
        prior_sigma: Scalar target standard deviation (default 3.0)
    """
    N, D = pred_mean.shape
    
    # 1. Define Prior Parameters
    # Covariance of prior is diagonal: sigma^2 * I
    prior_var = prior_sigma ** 2
    
    # 2. Compute Terms
    
    # A) Trace Term: tr(Sigma_prior^-1 @ Sigma_pred)
    # Since Sigma_prior is scalar diagonal, this is just tr(Sigma_pred) / prior_var
    # We can just sum the diagonal elements of pred_cov
    tr_term = jnp.trace(pred_cov, axis1=-2, axis2=-1) / prior_var
    
    # B) Quadratic Term: (mu_1 - mu_0)^T @ Sigma_prior^-1 @ (mu_1 - mu_0)
    # Similarly, divides by prior_var
    diff = pred_mean - prior_mean
    quad_term = jnp.sum(diff**2, axis=-1) / prior_var
    
    # C) Log Det Term: log(|Sigma_prior| / |Sigma_pred|)
    # log(|Sigma_prior|) = D * log(prior_var)
    log_det_prior = D * jnp.log(prior_var)
    
    # Use slogdet for stability on the predicted covariance
    _, log_det_pred = jnp.linalg.slogdet(pred_cov)
    
    log_det_term = log_det_prior - log_det_pred
    
    # 3. Combine
    # KL = 0.5 * (trace + quad + log_det - dimensionality)
    kl = 0.5 * (tr_term + quad_term + log_det_term - D)
    
    return jnp.mean(kl) # Average over batch
    # return kl

def get_functional_loss(flat_w, step_idx, key=None):
    # Unflatten MLP
    # params = unflatten_pytree(flat_w, shapes, treedef, mask)
    # model = eqx.combine(params, static)
    
    # y_pred = model.predict(X_train_full)
    # residuals = (y_pred - Y_train_full) ** 2

    ## y_pred contrains means and stddev, and we want a NLL loss
    # y_pred = model.predict(X_train_full)
    # means = y_pred[:, 0:real_output_size]
    # stddev = y_pred[:, real_output_size:real_output_size*2]
    # residuals = 0.5 * jnp.log(2 * jnp.pi * (stddev ** 2 + 1e-6)) + 0.5 * ((Y_train_full - means) ** 2) / (stddev ** 2 + 1e-6)
    # # residuals = ((Y_train_full - means) ** 2) / (stddev ** 2 + 1e-6)

    ## Let's compute teh NL by approximate marginalisation over \theta
    means_theta, sigmas_theta = flat_w[:input_dim], flat_w[input_dim:2*input_dim]
    # residuals, _, _ = negative_log_likelihood(means_theta, sigmas_theta, X_train_full, Y_train_full)

    theta = jax.random.normal(key, shape=sigmas_theta.shape) * sigmas_theta + means_theta  # Sample theta from predicted distribution
    # theta = means_theta  ## Let's just use the mean for now, to see if it can learn anything at all. TODO
    params = unflatten_pytree(theta, shapes, treedef, mask)
    model = eqx.combine(params, static)
    residuals = (model.predict(X_train_full) - Y_train_full) ** 2

    ## Compute KL divergence between predicted theta distribution and prior (N(0, I))
    # kl_div = compute_kl_divergence(means_theta, sigmas_theta)

    ## Compute KL divergence between predicted y distribution and N(Y_train_full, I)
    _, mean_y, cov_y = negative_log_likelihood(means_theta, sigmas_theta, X_train_full, Y=None, use_inverse=True, noise_sigma=1e-3)
    std_y = jnp.sqrt(jnp.diagonal(cov_y, axis1=1, axis2=2))
    # kl_div = compute_kl_divergence_emp(mean_y.flatten(), std_y.flatten(), Y_train_full.flatten())
    # kl_div = compute_kl_divergence(mean_y.flatten(), std_y.flatten())
    kl_div = compute_output_kl(mean_y, cov_y, prior_mean=jnp.mean(Y_train_full), prior_sigma=3.0)

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

    ## Add the KL loss
    final_loss = final_loss + CONFIG["kl_weight"] * kl_div
    # final_loss = final_loss + 1e-4 * kl_div

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
    keys = jax.random.split(key, x0_batch.shape[0])
    preds_batch = jax.vmap(model, in_axes=(0, None, 0))(x0_batch, total_steps, keys) # (Batch, Steps, D*3)

    # ## Extract and sample from the predicted mean and stddev
    # means = preds_batch[:, :, :input_dim]
    # stddevs = preds_batch[:, :, input_dim:2*input_dim]
    # eps = jax.random.normal(key, shape=stddevs.shape)
    # # preds_batch = means + eps * stddevs
    # preds_batch = means

    # step_indices = jnp.arange(total_steps)
    # preds_batch_data = preds_batch

    # step_indices = jnp.array([0, CONFIG["n_circles"]//2, CONFIG["n_circles"]-1])       ## TODO
    # step_indices = jnp.array([CONFIG["n_circles"]//2, CONFIG["n_circles"]-1])             ## TODO
    # chose_from = jnp.arange(1, CONFIG["n_circles"])
    # step_indices = jax.random.choice(key, CONFIG["n_circles"], shape=(25,), replace=False)
    # ## Always add 0
    # step_indices = jnp.concatenate([jnp.array([0]), step_indices])

    step_indices = jnp.arange(CONFIG["n_circles"])
    # step_indices = jax.random.choice(key, CONFIG["n_circles"], shape=(25,), replace=False)
    # step_indices = jnp.sort(step_indices)
    preds_batch_data = preds_batch[:, step_indices, :]

    keys = jax.random.split(key, len(step_indices))
    def loss_per_seq(seq):
        return jax.vmap(get_functional_loss)(seq, step_indices, keys)
    losses_batch = jax.vmap(loss_per_seq)(preds_batch_data) # (Batch, Steps)
    # total_data_loss = jnp.mean(jnp.sum(losses_batch, axis=1))
    total_data_loss = jnp.mean(jnp.mean(losses_batch, axis=1))

    # Consistency Loss
    step_indices = jnp.arange(1, CONFIG["transformer_target_step"])
    keys = jax.random.split(key, len(step_indices)-2)
    preds_batch_cons = preds_batch[:, step_indices, :]
    def cons_loss_per_seq(seq):
        # return jax.vmap(get_disconsistency_loss)(seq[:-1], seq[1:], step_indices[:-1], keys)
        return jax.vmap(get_consistency_loss)(seq[1:-1], seq[2:], step_indices[1:-1], keys)
    # cons_losses_batch = jax.vmap(cons_loss_per_seq)(preds_batch_cons) 
    # total_cons_loss = jnp.mean(jnp.sum(cons_losses_batch, axis=1))

    # total_loss = total_data_loss + CONFIG["consistency_loss_weight"]*total_cons_loss

    total_loss = total_data_loss

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
    # total_loss += 1e-1 * jax.nn.relu(max_val - 2.0)

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
    final_batch_traj = jax.vmap(tf_model, in_axes=(0, None, None))(x0_batch, CONFIG["transformer_target_step"], eval_key)
    
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

weight_dim = input_dim
# --- 7. VISUALIZATION ---
print("\n=== Generating Dashboards ===")
x_grid = jnp.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], 300)[:, None]

# --- DASHBOARD 2: FUNCTIONAL EVOLUTION ---
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 2)

## Loss history is NLL, so can be zero or negative. Shift it up for log plotting
# shifted_loss_history = loss_history - np.min(loss_history) + 1e-2

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(loss_history, color='teal', linewidth=2)
# ax1.set_yscale('symlog')
ax1.set_yscale('log')
ax1.set_title("Training NLL Loss (Shifted to Avoid Zero)")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
traj_train_losses = []
traj_test_losses = []
for i in range(len(final_traj)):
    w = final_traj[i, :weight_dim]
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
    w = final_traj[i, :input_dim]
    p = unflatten_pytree(w, shapes, treedef, mask)
    m = eqx.combine(p, static)
    label = "Limit" if i==n_steps-1 else None
    alpha = 1.0 if i==n_steps-1 else 0.1
    ax3.plot(x_grid, m.predict(x_grid), color=cmap(i/n_steps), alpha=alpha, linewidth=1.5, label=label)

ax3.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1)
w_reg = final_traj[CONFIG["regularization_step"], :weight_dim]
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
        w = final_batch_traj[b, step_idx, :weight_dim]
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
fig, ax = plt.subplots(figsize=(10, 4), nrows=1, ncols=1) 
traj_seed_0 = final_batch_traj[0]

## Extrat mean and stddev trajectories
mean_traj = traj_seed_0[:, :weight_dim]
stddev_traj = traj_seed_0[:, weight_dim:2*weight_dim]
eps = jax.random.normal(eval_key, shape=stddev_traj.shape)
traj_seed_0 = mean_traj + eps * stddev_traj
# traj_seed_0 = mean_traj

# plot_ids = np.arange(100)  # First 100 parameters
nb_plots = min(100, traj_seed_0.shape[1])
plot_ids = jax.random.choice(jax.random.PRNGKey(42), traj_seed_0.shape[1], shape=(nb_plots,), replace=False)

# plot_up_to = CONFIG["n_circles"]
plot_up_to = CONFIG["transformer_target_step"]

for idx in plot_ids:
    ## Plot the difference x_t - x_(t-1)
    traj = traj_seed_0[:plot_up_to, idx]
    # traj_diff = jnp.concatenate([jnp.array([0.0]), traj[1:] - traj[:-1]])
    # ax.plot(np.arange(plot_up_to), traj_diff, linewidth=1.5, label=f"Param {idx}")

    ax.plot(np.arange(plot_up_to), traj, linewidth=1.5, label=f"Param {idx}")



# ## Plot vertical lines (One for the n_circles step, one for the regularization step
# ax.axvline(CONFIG["n_circles"], color='k', linestyle='--', label="End of Data")
# ax.axvline(CONFIG["regularization_step"], color='red', linestyle=':', label="Reg Step")

ax.set_title("Parameter Trajectories")
ax.set_xlabel("Sequence Step")
ax.set_ylabel("Parameter Value")
plt.tight_layout()
plt.savefig(plots_path / "extended_parameter_trajectories.png")
plt.show()

#%% Model Predictions Corresponding to Circles plot
print("Generating n_circles Model Prediction Plot (Circle-Specific Models)...")

## Increase the font size of everything in the plot
plt.rcParams.update({'font.size': 14})

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
        w = final_batch_traj[0, circle_idx, :weight_dim]    ## Only using the means  
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
        y_pred = m.predict(X_circle)[:, 0:real_output_size]
        circle_losses[circle_idx] = (X_circle, y_pred)
    return circle_losses

train_preds_cc = predict_circle_specific_loss(final_batch_traj, X_train_full)
test_preds_cc = predict_circle_specific_loss(final_batch_traj, X_test)

fig, ax = plt.subplots(figsize=(10, 6))     
# ax.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1, label="Train Data")
ax.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1)
# ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1, label="Test Data")
ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1)

# X_circles of shape (N_circle, 1), y_pred of shape (N_circle, 1)

for circle_idx, (X_circle, y_pred) in train_preds_cc.items():
    # print(f"Circle {circle_idx}: X_circle shape: {X_circle.shape}, y_pred shape: {y_pred.shape}")
    ax.scatter(X_circle, y_pred, c='green', s=1, alpha=0.3, label="Train" if circle_idx==1 else None)
for circle_idx, (X_circle, y_pred) in test_preds_cc.items():
    ax.scatter(X_circle, y_pred, c='red', s=1, alpha=0.3, label="Test" if circle_idx==CONFIG["n_circles"]-1 else None)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
# ax.set_title(f"Model Predictions Corresponding to Circles (Circle-Specific Models)")
ax.set_title(f"Model Predictions")
ax.set_ylim(jnp.min(Y_train_full)-1, jnp.max(Y_train_full)+1)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(plots_path / "model_predictions_n_circles_circle_specific.png")
# plt.show()

plt.draw()

## Save as PDF
plt.savefig(plots_path / "model_predictions_n_circles_circle_specific.pdf")













# %%

print("Generating n_circles Model Prediction Plot (Circle-Specific Models)...")

def predict_circle_uncertainty(final_batch_traj, X_data):
    """
    Computes mean and uncertainty for circle-specific linear models.
    Assumes final_batch_traj shape is (Batch, Steps, 4) -> [mu_slope, mu_bias, std_slope, std_bias]
    """
    circle_stats = {}
    
    # Pre-calculate masks for all circles to ensure consistency
    circle_masks = []
    for r in all_radii:
        new_mask = jnp.abs(X_data - x_mean) <= r
        circle_masks.append(new_mask.flatten())
    circle_masks = jnp.array(circle_masks)

    for circle_idx in range(CONFIG["transformer_target_step"]):
    # for circle_idx in [100, CONFIG["transformer_target_step"]-1]:
        # Extract Mean and Std from the final dimension (Dim*2 = 4)
        # We assume the ordering: [mu_slope, mu_intercept, sigma_slope, sigma_intercept]
        # Adjust indices [0, 1] and [2, 3] if your specific flattening order differs.

        # --- Annulus Logic ---
        # If circle_idx is 0, we take the first circle mask.
        # If circle_idx > 0, we take (Current Circle) AND (NOT Previous Circle)
        if circle_idx == 0:
            ring_mask = circle_masks[circle_idx]
        else:
            in_circle_idx = circle_idx - 1
            ring_mask = jnp.logical_and(circle_masks[circle_idx], ~circle_masks[in_circle_idx])

        X_circle = X_data[ring_mask]

        if X_circle.shape[0] == 0:
            continue

        w = final_batch_traj[0, circle_idx, :]    ## Only using the means  
        # p = unflatten_pytree(w, shapes, treedef, mask)
        # m = eqx.combine(p, static)
        # y_pred = m.predict(X_circle)

        # Extract Mean and Std Predictions
        # y_mean = y_pred[:, 0:real_output_size]
        # y_std = y_pred[:, real_output_size:real_output_size*2]

        # Calculte the means and stddev using nll loss
        theta_mean = w[:weight_dim]
        theta_std = w[weight_dim:2*weight_dim]
        _, y_mean, cov_y_pred = negative_log_likelihood(theta_mean, theta_std, X_circle, Y=None, use_inverse=True, noise_sigma=1e-3)
        y_std = jnp.sqrt(jnp.diagonal(cov_y_pred, axis1=1, axis2=2))[:, 0]  # Extract stddev for the output dimension

        # ## Let's assume the model is linear: y = theta_0 * x + theta_1. Then the mean and stddev of y can be calculated directly from the mean and stddev of theta.
        # theta_mean = w[:weight_dim]
        # theta_std = w[weight_dim:2*weight_dim]
        # slope_mean, intercept_mean = theta_mean
        # slope_std, intercept_std = theta_std
        # y_mean = slope_mean * X_circle.flatten() + intercept_mean
        # y_std = jnp.sqrt((X_circle.flatten() * slope_std) ** 2 + intercept_std ** 2)  # Propagate uncertainty through the linear model

        # Store data for plotting
        circle_stats[circle_idx] = (X_circle, y_mean, y_std)

    return circle_stats

# Run inference
train_stats = predict_circle_uncertainty(final_batch_traj, X_train_full)
test_stats = predict_circle_uncertainty(final_batch_traj, X_test)

#%%
# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))

# 1. Plot Background Data
ax.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1, label="Train Data")
ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1, label="Test Data")

# Helper function to plot bands
def plot_uncertainty_bands(stats_dict, color_mean, color_band, label_prefix):
    added_label = False
    
    for circle_idx, (X_seg, y_mu, y_sigma) in stats_dict.items():
        # We must sort X to plot lines and fill_between correctly
        sort_indices = jnp.argsort(X_seg.flatten())
        X_sorted = X_seg[sort_indices].flatten()
        mu_sorted = y_mu[sort_indices].flatten()
        sigma_sorted = y_sigma[sort_indices].flatten()

        # print("Aff distance from the mean are:", jnp.abs(X_sorted - x_mean).flatten()   )
        
        # Label only the first segment to avoid cluttering the legend
        lbl = f"{label_prefix} Mean" if not added_label else None
        
        # Plot Mean
        # ax.plot(X_sorted, mu_sorted, c=color_mean, linewidth=2, alpha=0.8, label=lbl)

        ## Scatter plot mean instead of line plot
        ax.scatter(X_sorted, mu_sorted, c=color_mean, s=5, alpha=0.8, label=lbl)
        
        # Plot Uncertainty (Mean +/- 2 Std)
        # ax.fill_between(
        #     X_sorted, 
        #     mu_sorted - 2 * sigma_sorted, 
        #     mu_sorted + 2 * sigma_sorted, 
        #     color=color_band, 
        #     alpha=0.3,
        #     label=f"{label_prefix} Uncertainty" if not added_label else None
        # )
        # added_label = True

        # print("Min and Max of sigma for circle", circle_idx, "are:", jnp.min(sigma_sorted), jnp.max(sigma_sorted))

        ## We can't use fill_between with scatter, so for each point, we plot a vertical line
        multiplier = 2
        for x_pt, mu_pt, sigma_pt in zip(X_sorted, mu_sorted, sigma_sorted):
            ax.vlines(x_pt, mu_pt - multiplier * sigma_pt, mu_pt + multiplier * sigma_pt, color=color_band, alpha=0.1, label=f"{label_prefix} Uncertainty" if not added_label else None)

            added_label = True

# 2. Plot Model Inference (Mean + 2 Std)
# We usually only plot the Test predictions for clarity, but you can enable both.
# Here I plot Test predictions in Red/Pink and Train in Green (Optional)

# Plotting Test Set Inference (High contrast)
plot_uncertainty_bands(test_stats, color_mean='red', color_band='red', label_prefix="Test Model")

# Uncomment below if you also want to see the fit on Training data segments
plot_uncertainty_bands(train_stats, color_mean='green', color_band='green', label_prefix="Train Model")

ax.set_title(r"(d) Model Predictions with Uncertainty ($\mu \pm 2 \sigma$)", fontsize=24)
ax.set_ylim(jnp.min(Y_train_full)-1, jnp.max(Y_train_full)+1)
ax.grid(True, alpha=0.3)
# ax.legend()

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

plt.tight_layout()
plt.savefig(plots_path / "model_predictions_uncertainty.png")
# plt.show()
plt.draw()
## Save as PDF as well
plt.savefig(plots_path / "model_predictions_uncertainty.pdf")

# %%
