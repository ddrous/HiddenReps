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
    "data_samples": 10000,
    "noise_std": 0.005,
    "segments": 11,
    "x_range": [-1.5, 1.5],
    "train_seg_ids": [2, 3, 4, 5, 6, 7, 8], 
    "width_size": 24,
    "mlp_batch_size": 64,

    # Expansion Hyperparameters
    "n_circles": 30*10,           
    "warmup_steps": 0,
    
    # --- TRANSFORMER HYPERPARAMETERS ---
    "lr": 5e-5,      
    "transformer_epochs": 25000,
    "print_every": 2500,
    "transformer_batch_size": 1,      

    # New Params
    "transformer_d_model": 128*4,    # Embedding Dimension
    "transformer_n_heads": 1,      # Number of Heads
    "transformer_n_layers": 1,     # Number of Transformer Blocks
    "transformer_d_ff": 1//1,       # Feedforward dimension inside block: TODO: not needed atm, see forward pass.
    "transformer_substeps": 50,     # Number of micro-steps per macro step
    
    "transformer_target_step": 60*10,    # Total steps to unroll
    "scheduled_loss_weight": False,

    ## Consistency Loss Config
    "n_synthetic_points": 512,
    "consistency_loss_weight": 0.0,

    # Regularization Config
    "regularization_step": 40*10,     
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

x_mean = jnp.mean(X_train_full)
# x_mean = jnp.min(X_train_full)
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

        self.layers = [eqx.nn.Linear(1, 2, use_bias=True, key=k1)]

    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x
    def predict(self, x):
        return jax.vmap(self)(x)





class ResidualBlock(eqx.Module):
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    norm1: Optional[eqx.nn.GroupNorm]
    norm2: Optional[eqx.nn.GroupNorm]
    emb_proj: Optional[eqx.nn.Linear]
    act: typing.Callable
    use_conditioning: bool
    use_normalization: bool

    def __init__(self, in_channels, out_channels, emb_dim, use_conditioning, use_normalization, key):
        self.use_conditioning = use_conditioning
        self.use_normalization = use_normalization
        
        k1, k2, k3 = jax.random.split(key, 3)
        
        # V-Net style: Input + Conv-Act-Conv(Input)
        self.conv1 = eqx.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same", key=k1)
        self.conv2 = eqx.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same", key=k2)
        
        if use_normalization:
            self.norm1 = eqx.nn.GroupNorm(min(8, out_channels), out_channels)
            self.norm2 = eqx.nn.GroupNorm(min(8, out_channels), out_channels)
        else:
            self.norm1 = self.norm2 = None
            
        if use_conditioning:
            self.emb_proj = eqx.nn.Linear(emb_dim, out_channels, key=k3)
        else:
            self.emb_proj = None
            
        self.act = jax.nn.silu

    def __call__(self, x, emb: Optional[jax.Array] = None):
        h = x
        
        # Block 1
        if self.use_normalization: h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        # Inject conditioning
        if self.use_conditioning and emb is not None:
            # emb: (emb_dim,) -> proj -> (out_channels,) -> (out_channels, 1)
            h = h + self.emb_proj(emb)[:, None]
        
        # Block 2
        if self.use_normalization: h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        # Residual alignment
        if x.shape[0] != h.shape[0]:
            # Project x to match hidden channels if necessary
            x = eqx.nn.Conv1d(x.shape[0], h.shape[0], kernel_size=1, key=jax.random.PRNGKey(0))(x)
            
        return x + h

class Unet1D(eqx.Module):
    # Projections
    t_proj: Optional[eqx.nn.Linear] # Simplified time embedding
    cond_proj: Optional[eqx.nn.Linear]
    
    # Layers
    init_conv: eqx.nn.Conv1d
    down_blocks: list
    down_samples: list
    mid_block: ResidualBlock
    up_samples: list
    up_blocks: list
    final_conv: eqx.nn.Conv1d
    
    # Config
    use_conditioning: bool

    def __init__(self, 
                 in_shape: Tuple[int, int], 
                 out_shape: Tuple[int, int],
                 base_chans: int, 
                 levels: int, 
                 use_normalization: bool = False,
                 key: jax.random.PRNGKey = jax.random.PRNGKey(0), 
                 cond_dim: Optional[int] = None, 
                 use_conditioning: bool = False):
        
        self.use_conditioning = use_conditioning
        c_in, _ = in_shape
        c_out, _ = out_shape
        keys = jax.random.split(key, 100)
        
        # --- Conditioning ---
        emb_dim = 0
        if self.use_conditioning:
            if cond_dim is None: raise ValueError("cond_dim required if use_conditioning=True")
            emb_dim = base_chans * 4
            # Simple Linear time embedding (can be replaced with Sinusoidal if needed)
            self.t_proj = eqx.nn.Linear(1, emb_dim, key=keys[0]) 
            self.cond_proj = eqx.nn.Linear(cond_dim, emb_dim, key=keys[1])
        else:
            self.t_proj = self.cond_proj = None

        # --- Encoder ---
        self.init_conv = eqx.nn.Conv1d(c_in, base_chans, kernel_size=3, padding="same", key=keys[2])
        
        self.down_blocks = []
        self.down_samples = []
        
        curr_ch = base_chans
        for i in range(levels):
            # ResBlock keeps channels same, Downsample doubles them usually, 
            # but standard UNet: Conv(C->C) -> Down(C->2C) is common, 
            # or Down(C->C) -> Conv(C->2C).
            # Here: Block(C->C) -> Down(C->C) -> Next Level starts with 2C expansion logic?
            # Simpler: Block(C->C) -> Down(C->2C)
            
            self.down_blocks.append(
                ResidualBlock(curr_ch, curr_ch, emb_dim, use_conditioning, use_normalization, keys[10+i])
            )
            # Downsample doubles channels
            self.down_samples.append(
                eqx.nn.Conv1d(curr_ch, curr_ch * 2, kernel_size=3, stride=2, padding=1, key=keys[20+i])
            )
            curr_ch *= 2

        # --- Middle ---
        self.mid_block = ResidualBlock(curr_ch, curr_ch, emb_dim, use_conditioning, use_normalization, keys[50])

        # --- Decoder ---
        self.up_samples = []
        self.up_blocks = []
        
        for i in range(levels):
            # Upsample halves channels (2C -> C)
            target_ch = curr_ch // 2
            self.up_samples.append(
                eqx.nn.ConvTranspose1d(curr_ch, target_ch, kernel_size=4, stride=2, padding=1, key=keys[60+i])
            )
            # Block takes (C_skip + C_up) -> C_target
            self.up_blocks.append(
                ResidualBlock(target_ch * 2, target_ch, emb_dim, use_conditioning, use_normalization, keys[70+i])
            )
            curr_ch = target_ch
            
        self.final_conv = eqx.nn.Conv1d(base_chans, c_out, kernel_size=1, key=keys[90])

    def __call__(self, t, y, arg):
        # 1. Conditioning
        emb = None
        if self.use_conditioning:
            t = jnp.array([t]) if jnp.ndim(t) == 0 else t.reshape(1)
            t_emb = jax.nn.silu(self.t_proj(t))
            c_emb = jax.nn.silu(self.cond_proj(arg))
            emb = t_emb + c_emb

        # 2. Encoder
        x = self.init_conv(y)
        skips = []
        
        for block, down in zip(self.down_blocks, self.down_samples):
            x = block(x, emb)
            skips.append(x)
            x = down(x)
            
        # 3. Middle
        x = self.mid_block(x, emb)
        
        # 4. Decoder
        # Reverse skips to match upsamples
        for up, block, skip in zip(self.up_samples, self.up_blocks, reversed(skips)):
            x = up(x)
            
            # --- FIX: Handle Shape Mismatch ---
            # If x is smaller than skip (due to odd length downsampling), pad x
            if x.shape[-1] < skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                x = jnp.pad(x, ((0,0), (0, diff)))
            # If x is larger (rare with correct padding but possible), crop x
            elif x.shape[-1] > skip.shape[-1]:
                x = x[:, :skip.shape[-1]]
                
            x = jnp.concatenate([x, skip], axis=0) # Channel concat
            x = block(x, emb)
            
        return self.final_conv(x)


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

class NeuralODE(eqx.Module):
    func: eqx.nn.MLP
    data_dim: int
    # embd_dim: int
    # init_embd: eqx.nn.Linear

    y0: jax.Array
     
    def __init__(self, data_dim, hidden_dim, key):
        # self.embd_dim = 32
        # self.init_embd = eqx.nn.Linear(data_dim, self.embd_dim, key=key)

        self.func = eqx.nn.MLP(
            in_size=data_dim*4, 
            out_size=data_dim*4, 
            width_size=hidden_dim, 
            # width_size=int(data_dim*2.5), 
            depth=4,
            activation=jax.nn.softplus,
            key=key
        )

        # self.y0 = jax.random.normal(key, shape=(data_dim*2,)) * 1e-2

        # self.func = eqx.nn.Linear(data_dim*3, data_dim*3, use_bias=False, key=key)
        # ## Initialize func to near identity
        # self.func = eqx.tree_at(lambda func: func.weight, func, jnp.eye(data_dim))

        self.y0 = jax.random.normal(key, shape=(data_dim*4,)) * 1e-2

        ## Define func as a small Unet1D
        ## Data_dim must be a multiple of 8. Let's pick the closest multiple of 8 greater than data_dim
        self.data_dim = data_dim
        # data_dim = ((data_dim + 7) // 8) * 8

        # self.func = Unet1D(
        #     in_shape=(data_dim, 1),
        #     out_shape=(data_dim, 1),
        #     base_chans=32,
        #     levels=3,
        #     use_normalization=False,
        #     key=key,
        #     cond_dim=None,
        #     use_conditioning=False
        # )


    def __call__(self, y0, steps, key=None):
        # Neural ODE logic differs: it doesn't take 'future' for teacher forcing usually.
        # It integrates from the last history point.
        
        # Initial condition: last point of history
        # y0 = history[-1]
                
        # Integration times
        # Assuming data step size is 1.0 (arbitrary units matching index)
        # ts = jnp.arange(steps + 1)
        ts = jnp.linspace(0.0, 1.0, (steps + CONFIG["warmup_steps"]))

        # data_dim_8 = ((self.data_dim + 7) // 8) * 8
        # # print("All shapes invloved in ODE solve:", y0.shape, data_dim_8, self.data_dim)
        # # Pad y0 to data_dim_8
        # if y0.shape[0] < data_dim_8:
        #     pad_width = data_dim_8 - y0.shape[0]
        #     y0 = jnp.pad(y0, (0, pad_width))

        def ode_func(t, y, args):
            # y = jnp.concatenate([y, jnp.array([t])])  # Append time to state
            return self.func(y)

            # y = y[:, None]
            # out = self.func(t, y, args)
            # # print("Inside ODE func shapes:", y.shape, out.shape)
            # return out.flatten()

            # y = jnp.concatenate([y, self.init_embd(y0)])  # Append fixed embedding to state
            # return self.func(y)

        
        term = diffrax.ODETerm(ode_func)
        solver = diffrax.Tsit5()
        # Step size controller
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        

        ## y0 must containt mean and stddev
        # y0 = jnp.concatenate([y0, jnp.zeros_like(y0)])  # Initial stddev = 0.0 
        y0 = self.y0

        sol = diffrax.diffeqsolve(
            term, solver, t0=ts[0], t1=ts[-1], dt0=0.1, y0=y0,
            # saveat=diffrax.SaveAt(ts=ts[1:]), # Save at 1, 2, ... steps
            saveat=diffrax.SaveAt(ts=ts[CONFIG["warmup_steps"]::1]), # Save at 1, 2, ... steps
            stepsize_controller=stepsize_controller,
            max_steps=4096*1
        )
        
        out = sol.ys # Shape (steps, dim)
        # return sol.ys[:, :self.data_dim]  # Remove padding if any
        # return out

        ## Usethe reparametrisation trick to sample from predicted mean and stddev
        # out of shape (steps, data_dim*2)
        means = out[:, :self.data_dim]
        stddevs = out[:, self.data_dim:2*self.data_dim]
        eps = jax.random.normal(key, shape=stddevs.shape)
        samples = means + eps * stddevs
        # samples = means
        return samples


class LinearRNN(eqx.Module):
    A: jax.Array
    B: jax.Array
    y_init: jax.Array  # Contains [y_0, y_{-1}]
    data_dim: int

    def __init__(self, data_dim, hidden_dim, key):
        self.data_dim = data_dim
        
        # Initialize A as Identity and B as Zeros
        problem_dim = data_dim*3
        self.A = jnp.eye(problem_dim)
        self.B = jnp.zeros((problem_dim, problem_dim))
        
        # y_init stores the two initial points needed for second-order recurrence
        # Shape: (2, data_dim)
        self.y_init = jax.random.normal(key, shape=(2, problem_dim)) * 0e-2

    def __call__(self, y0, steps, key):
        # Initial state for the scan: (y_{t-1}, y_{t-2})
        # We've initialized y_init such that y_init[0] is y_0 and y_init[1] is y_{-1}
        init_state = (self.y_init[0], self.y_init[1])

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

def get_functional_loss(flat_w, step_idx, key=None):
    # Unflatten MLP
    params = unflatten_pytree(flat_w, shapes, treedef, mask)
    model = eqx.combine(params, static)
    
    # y_pred = model.predict(X_train_full)
    # residuals = (y_pred - Y_train_full) ** 2

    ## y_pred contrains means and stddev, and we want a NLL loss
    y_pred = model.predict(X_train_full)
    means = y_pred[:, 0:real_output_size]
    stddev = y_pred[:, real_output_size:real_output_size*2]
    residuals = 0.5 * jnp.log(2 * jnp.pi * (stddev ** 2 + 1e-6)) + 0.5 * ((Y_train_full - means) ** 2) / (stddev ** 2 + 1e-6)
    # residuals = ((Y_train_full - means) ** 2) / (stddev ** 2 + 1e-6)

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
    keys = jax.random.split(key, x0_batch.shape[0])
    preds_batch = jax.vmap(model, in_axes=(0, None, 0))(x0_batch, total_steps, keys) # (Batch, Steps, D*3)

    ## Extract and sample from the predicted mean and stddev
    means = preds_batch[:, :, :input_dim]
    stddevs = preds_batch[:, :, input_dim:2*input_dim]
    eps = jax.random.normal(key, shape=stddevs.shape)
    # preds_batch = means + eps * stddevs
    preds_batch = means

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
shifted_loss_history = loss_history - np.min(loss_history) + 1e-2

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(shifted_loss_history, color='teal', linewidth=2)
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
fig, ax = plt.subplots(figsize=(7, 10), nrows=1, ncols=1) 
traj_seed_0 = final_batch_traj[0]

## Extrat mean and stddev trajectories
mean_traj = traj_seed_0[:, :weight_dim]
stddev_traj = traj_seed_0[:, weight_dim:2*weight_dim]
eps = jax.random.normal(eval_key, shape=stddev_traj.shape)
# traj_seed_0 = mean_traj + eps * stddev_traj
traj_seed_0 = mean_traj

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
ax.scatter(X_train_full, Y_train_full, c='blue', s=10, alpha=0.1, label="Train Data")
ax.scatter(X_test, Y_test, c='orange', s=10, alpha=0.1, label="Test Data")

# X_circles of shape (N_circle, 1), y_pred of shape (N_circle, 1)

for circle_idx, (X_circle, y_pred) in train_preds_cc.items():
    # print(f"Circle {circle_idx}: X_circle shape: {X_circle.shape}, y_pred shape: {y_pred.shape}")
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

        w = final_batch_traj[0, circle_idx, :weight_dim]    ## Only using the means  
        p = unflatten_pytree(w, shapes, treedef, mask)
        m = eqx.combine(p, static)
        y_pred = m.predict(X_circle)

        # Extract Mean and Std Predictions
        y_mean = y_pred[:, 0:real_output_size]
        y_std = y_pred[:, real_output_size:real_output_size*2]

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
        multiplier = 25
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

ax.set_title(r"Model Predictions with Uncertainty ($\mu \pm 2\sigma$)")
ax.set_ylim(jnp.min(Y_train_full)-1, jnp.max(Y_train_full)+1)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(plots_path / "model_predictions_uncertainty.png")
plt.show()

# %%
