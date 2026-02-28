#%% Cell 1: Imports, Utilities, and Configuration
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
import numpy as np
import time
import datetime
from pathlib import Path
import shutil
import sys
import glob
import subprocess
from tqdm import tqdm
from jax.flatten_util import ravel_pytree
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset, Sampler

import seaborn as sns
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 150  # Crisp visualizations

try:
    import xarray as xr
except ImportError:
    raise ImportError("Please install xarray and netcdf4: pip install xarray netcdf4")

def count_trainable_params(model):
    """Utility to count trainable parameters in an Equinox module."""
    def count_params(x):
        if isinstance(x, jnp.ndarray) and x.dtype in [jnp.float32, jnp.float64]:
            return x.size
        return 0
    param_counts = jax.tree_util.tree_map(count_params, model)
    return sum(jax.tree_util.tree_leaves(param_counts))

# --- Configuration ---
TRAIN = True
RUN_DIR = "./" if not TRAIN else None

SINGLE_BATCH = False
USE_NLL_LOSS = False

CONFIG = {
    "seed": 2026,
    "nb_epochs": 1,
    "print_every": 1,
    "batch_size": 2 if SINGLE_BATCH else 64,
    "learning_rate": 1e-7 if USE_NLL_LOSS else 1e-4,
    "p_forcing": 0.0,
    "inf_context_ratio": 0.5,
    "use_nll_loss": USE_NLL_LOSS,
    "aux_encoder_loss": False,
    "aux_loss_weight": 0.1,
    "seq_len": 24, # OpenSTL setting (12 pre -> 12 aft)

    # --- Architecture Params ---
    "num_layers": 1,
    "use_hypernet": True,
    "use_controlnet": True,

    # --- Architecture Params ---
    "dynamics_space": 961,
    "lam_space": 64,
    "split_forward": True,

    # --- Plateau Scheduler Config ---
    "lr_patience": 500,      
    "lr_cooldown": 0,       
    "lr_factor": 0.5,        
    "lr_rtol": 1e-3,         
    "lr_accum_size": 5,     
    "lr_min_scale": 1e-1     
}

key = jax.random.PRNGKey(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

def setup_run_dir(base_dir="experiments"):
    if TRAIN:
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        run_path = Path(base_dir) / timestamp
        run_path.mkdir(parents=True, exist_ok=True)

        current_script = Path(__file__)
        if current_script.exists():
            shutil.copy(current_script, run_path / "main.py")
        
        (run_path / "artefacts").mkdir(exist_ok=True)
        (run_path / "plots").mkdir(exist_ok=True)
        return run_path
    else:
        return Path(RUN_DIR)

run_path = setup_run_dir()
artefacts_path = run_path / "artefacts"
plots_path = run_path / "plots"

#%% Cell 2: PyTorch Data Loading & Plotting Helpers

class WeatherBenchTemperature(Dataset):
    def __init__(self, data_path="./data/WeatherBench", split="train", download=False, seq_len=24, mean=None, std=None):
        self.data_path = data_path
        self.split = split
        self.seq_len = seq_len
        
        if download:
            self._download_and_extract()
            
        if split == "train":
            years = [str(y) for y in range(1979, 2016)]
        elif split == "val":
            years = ['2016']
        elif split == "test":
            years = ['2017', '2018']
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'.")
            
        file_patterns = [os.path.join(data_path, f"*{y}*.nc") for y in years]
        files_to_load = []
        for pat in file_patterns:
            files_to_load.extend(glob.glob(pat))
            
        if len(files_to_load) == 0:
            raise FileNotFoundError(f"No .nc files found in {data_path}. Try download=True.")
            
        print(f"[{split.upper()}] Loading data into memory...")
        dataset = xr.open_mfdataset(sorted(files_to_load), combine='by_coords')
        raw_data = dataset.get('t2m').values 
        
        if split == "train":
            self.mean = raw_data.mean() if mean is None else mean
            self.std = raw_data.std() if std is None else std
        else:
            self.mean = raw_data.mean() if mean is None else mean
            self.std = raw_data.std() if std is None else std
            
        norm_data = (raw_data - self.mean) / self.std
        self.data = np.expand_dims(norm_data, axis=1).astype(np.float32)
        print(f"[{split.upper()}] Loaded Tensor Shape: {self.data.shape}")

    def _download_and_extract(self):
        if len(glob.glob(os.path.join(self.data_path, "*.nc"))) > 0: return
        os.makedirs(self.data_path, exist_ok=True)
        zip_path = os.path.join(self.data_path, "2m_temperature.zip")
        url = "https://dataserv.ub.tum.de/public.php/dav/files/m1524895/5.625deg/2m_temperature/?accept=zip"
        subprocess.run(["wget", url, "-O", zip_path], check=True)
        subprocess.run(["unzip", "-q", zip_path, "-d", self.data_path], check=True)
        os.remove(zip_path)

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        seq = self.data[idx : idx + self.seq_len]
        return torch.from_numpy(seq)


def numpy_collate(batch):
    # Stack list of (T, C, H, W) tensors into (B, T, C, H, W)
    videos = torch.stack(batch).numpy()
    # Transpose to (B, T, H, W, C) for the JAX models
    videos = np.transpose(videos, (0, 1, 3, 4, 2))
    return videos

print("Loading WeatherBench Dataset...")
try:
    data_path = './data/WeatherBench/2m_temperature' if TRAIN else "../../data/WeatherBench/2m_temperature"
    train_dataset = WeatherBenchTemperature(data_path=data_path, split="train", download=False, seq_len=CONFIG["seq_len"])
    
    if SINGLE_BATCH:
        training_subset = Subset(train_dataset, range(CONFIG["batch_size"]))
        train_loader = DataLoader(training_subset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=numpy_collate, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=numpy_collate, drop_last=False)

    sample_batch = next(iter(train_loader))
    B, nb_frames, H, W, C = sample_batch.shape
    print(f"Batched Video shape: {sample_batch.shape}")
except Exception as e:
    print(f"Could not load WeatherBench: {e}")
    raise e

# Dummy coords for pipeline compatibility with the WARP eval signature
y_coords = jnp.linspace(-1, 1, H)
x_coords = jnp.linspace(-1, 1, W)
X_grid, Y_grid = jnp.meshgrid(x_coords, y_coords)
coords_grid = jnp.stack([X_grid, Y_grid], axis=-1) 

def get_vmin_vmax(v1, v2):
    return min(v1.min(), v2.min()), max(v1.max(), v2.max())

def sbimshow(img, title="", ax=None, vmin=None, vmax=None):
    if ax is None:
        plt.imshow(img, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
        plt.title(title, fontsize=10)
        plt.axis('off')
    else:
        ax.imshow(img, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

def plot_pred_ref_videos(video, ref_video, title="Render", save_name=None):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    t_mid = len(video) // 2
    t_end = len(video) - 1
    vmin, vmax = get_vmin_vmax(video[..., :C], ref_video[..., :C])
    
    sbimshow(video[0, ..., 0], title=f"{title} t=0h", ax=ax[0, 0], vmin=vmin, vmax=vmax)
    sbimshow(video[t_mid, ..., 0], title=f"{title} t={t_mid}h", ax=ax[0, 1], vmin=vmin, vmax=vmax)
    sbimshow(video[t_end, ..., 0], title=f"{title} t={t_end}h", ax=ax[0, 2], vmin=vmin, vmax=vmax)

    sbimshow(ref_video[0, ..., 0], title="Ref t=0h", ax=ax[1, 0], vmin=vmin, vmax=vmax)
    sbimshow(ref_video[t_mid, ..., 0], title=f"Ref t={t_mid}h", ax=ax[1, 1], vmin=vmin, vmax=vmax)
    sbimshow(ref_video[t_end, ..., 0], title=f"Ref t={t_end}h", ax=ax[1, 2], vmin=vmin, vmax=vmax)
    plt.tight_layout()
    if save_name:
        plt.savefig(plots_path / save_name)
    plt.show()

def plot_pred_ref_videos_rollout(video, ref_video, title="Render", save_name=None):
    nb_frames = video.shape[0]
    # vmin, vmax = get_vmin_vmax(video[..., :C], ref_video[..., :C])
    vmin, vmax = get_vmin_vmax(ref_video[..., :C], ref_video[..., :C])

    if video.shape[-1] == 1:
        fig, axes = plt.subplots(2, 1+(nb_frames//2), figsize=(20, 6))
        indices_to_plot = list(np.arange(0, nb_frames, 2)) + [nb_frames-1]
        for i, idx in enumerate(indices_to_plot):
            sbimshow(video[idx, ..., 0], title=f"{title} t={idx}h", ax=axes[0, i], vmin=vmin, vmax=vmax)
            sbimshow(ref_video[idx, ..., 0], title=f"Ref t={idx}h", ax=axes[1, i], vmin=vmin, vmax=vmax)
    else:
        fig, axes = plt.subplots(3, 1+(nb_frames//2), figsize=(20, 7))
        indices_to_plot = list(np.arange(0, nb_frames, 2)) + [nb_frames-1]
        for i, idx in enumerate(indices_to_plot):
            sbimshow(video[idx, ..., 0], title=f"Mean t={idx}h", ax=axes[0, i], vmin=vmin, vmax=vmax)
            # Standard deviation plot doesn't share limits with temperature
            sbimshow(video[idx, ..., 1], title=f"Std t={idx}h", ax=axes[1, i]) 
            sbimshow(ref_video[idx, ..., 0], title=f"Ref t={idx}h", ax=axes[2, i], vmin=vmin, vmax=vmax)

    plt.tight_layout()
    if save_name:
        plt.savefig(plots_path / save_name)
    plt.show()

def animate_side_by_side(pred_video, ref_video, title="Prediction vs Ground Truth", interval=200):
    """HTML5 video for Predicted vs Reference Weather Sequences."""
    seq_len = pred_video.shape[0]
    # vmin, vmax = get_vmin_vmax(pred_video[..., :C], ref_video[..., :C])
    vmin, vmax = get_vmin_vmax(ref_video[..., :C], ref_video[..., :C])
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    
    im_pred = axes[0].imshow(pred_video[0, ..., 0], cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[0].axis('off')
    title_pred = axes[0].set_title(f"Predicted | t=0h")
    
    im_ref = axes[1].imshow(ref_video[0, ..., 0], cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[1].axis('off')
    title_ref = axes[1].set_title(f"Ground Truth | t=0h")
    
    cbar = fig.colorbar(im_ref, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label("Norm. Temperature")
    plt.suptitle(title, y=1.05)
    
    def update(frame):
        im_pred.set_array(pred_video[frame, ..., 0])
        title_pred.set_text(f"Predicted | t={frame}h")
        im_ref.set_array(ref_video[frame, ..., 0])
        title_ref.set_text(f"Ground Truth | t={frame}h")
        return [im_pred, title_pred, im_ref, title_ref]
        
    ani = animation.FuncAnimation(fig, update, frames=seq_len, interval=interval, blit=False)
    plt.close(fig)
    return HTML(ani.to_jshtml())


#%% Cell 3: Model Definition
def fourier_encode(x, num_freqs):
    freqs = 2.0 ** jnp.arange(num_freqs)
    angles = x[..., None] * freqs[None, None, :] * jnp.pi
    angles = angles.reshape(*x.shape[:-1], -1)
    return jnp.concatenate([x, jnp.sin(angles), jnp.cos(angles)], axis=-1)

class CNNEncoder(eqx.Module):
    layers: list

    def __init__(self, in_channels, out_dim, spatial_shape, key, hidden_width=16, depth=3):
        H, W = spatial_shape
        keys = jax.random.split(key, depth + 1)
        
        conv_layers = []
        current_in = in_channels
        current_out = hidden_width
        
        for i in range(depth):
            conv_layers.append(
                eqx.nn.Conv2d(current_in, current_out, kernel_size=3, stride=2, padding=1, key=keys[i])
            )
            current_in = current_out
            current_out *= 2
            
        # Compute flattened dimension dynamically
        h_small = H // (2**depth)
        w_small = W // (2**depth)
        flat_dim = current_in * h_small * w_small
        
        self.layers = conv_layers + [eqx.nn.Linear(flat_dim, out_dim, key=keys[depth])]
        
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = x.reshape(-1)
        x = self.layers[-1](x)
        return x

class CNNDecoder(eqx.Module):
    fc: eqx.nn.Linear
    deconv_layers: list
    hidden_width: int
    h_small: int
    w_small: int

    def __init__(self, in_dim, out_channels, spatial_shape, key, hidden_width=16, depth=3):
        H, W = spatial_shape
        self.hidden_width = hidden_width * (2 ** (depth - 1))
        self.h_small = H // (2 ** depth)
        self.w_small = W // (2 ** depth)

        keys = jax.random.split(key, depth + 1)
        self.fc = eqx.nn.Linear(in_dim, self.hidden_width * self.h_small * self.w_small, key=keys[0])

        layers = []
        current_in = self.hidden_width
        for i in range(depth):
            current_out = current_in // 2 if i < depth - 1 else out_channels
            layers.append(
                eqx.nn.ConvTranspose2d(current_in, current_out, kernel_size=4, stride=2, padding=1, key=keys[i+1])
            )
            current_in = current_out
        self.deconv_layers = layers

    def __call__(self, z):
        x = jax.nn.relu(self.fc(z))
        x = x.reshape(self.hidden_width, self.h_small, self.w_small)
        for i, layer in enumerate(self.deconv_layers):
            x = layer(x)
            if i < len(self.deconv_layers) - 1:
                x = jax.nn.relu(x)
        return x

class LAM(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, dyn_dim, lam_dim, key):
        self.mlp = eqx.nn.MLP(dyn_dim * 2, lam_dim, width_size=dyn_dim, depth=3, key=key)
        
    def __call__(self, z_prev, z_target):
        return self.mlp(jnp.concatenate([z_prev, z_target], axis=-1))

class ForwardDynamics(eqx.Module):
    mlp_A: Optional[eqx.nn.MLP]
    mlp_B: Optional[eqx.nn.MLP]
    giant_mlp: Optional[eqx.nn.MLP]
    split_forward: bool = eqx.field(static=True)

    def __init__(self, dyn_dim, lam_dim, split_forward, key):
        self.split_forward = split_forward
        k1, k2, k3 = jax.random.split(key, 3)
        if split_forward:
            self.mlp_A = eqx.nn.MLP(dyn_dim, dyn_dim, width_size=dyn_dim*2, depth=3, key=k1)
            self.mlp_B = eqx.nn.MLP(lam_dim, dyn_dim, width_size=dyn_dim*2, depth=3, key=k2)
            self.giant_mlp = None
        else:
            self.mlp_A = None
            self.mlp_B = None
            self.giant_mlp = eqx.nn.MLP(dyn_dim + lam_dim, dyn_dim, width_size=dyn_dim*2, depth=2, key=k3)

    def __call__(self, z_prev, a):
        if self.split_forward:
            return self.mlp_A(z_prev) + self.mlp_B(a)
        else:
            return self.giant_mlp(jnp.concatenate([z_prev, a], axis=-1))

class WorldModel(eqx.Module):
    encoder: CNNEncoder
    decoder: CNNDecoder
    lam: LAM
    forward_dyn: ForwardDynamics
    dyn_dim: int = eqx.field(static=True)
    lam_dim: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)
    split_forward: bool = eqx.field(static=True)

    def __init__(self, frame_shape, dyn_dim, lam_dim, split_forward, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        H, W, C = frame_shape
        self.frame_shape = frame_shape
        self.dyn_dim = dyn_dim
        self.lam_dim = lam_dim
        self.split_forward = split_forward

        self.encoder = CNNEncoder(in_channels=C, out_dim=dyn_dim, spatial_shape=(H, W), key=k1, hidden_width=64, depth=4)
        self.decoder = CNNDecoder(in_dim=dyn_dim, out_channels=C*2 if CONFIG["use_nll_loss"] else C, spatial_shape=(H, W), key=k2, hidden_width=64, depth=4)
        self.lam = LAM(dyn_dim, lam_dim, key=k3)
        self.forward_dyn = ForwardDynamics(dyn_dim, lam_dim, split_forward, key=k4)

    @property
    def A(self):
        """Maintains compatibility with your Cell 5 visualisation."""
        if self.split_forward:
            return self.forward_dyn.mlp_A.layers[-1].weight
        else:
            return self.forward_dyn.giant_mlp.layers[-1].weight

    def _get_preds_single(self, ref_video, p_forcing, key, inf_context_ratio):
        T = ref_video.shape[0]
        H, W, C = self.frame_shape
        init_frame = ref_video[0]

        # 1. Initialize abstract state from the very first frame
        z_init = self.encoder(jnp.transpose(init_frame, (2, 0, 1)))

        def scan_step(z_prev, scan_inputs):
            gt_curr_frame, step_idx = scan_inputs
            k = jax.random.fold_in(key, step_idx)
            k, subk = jax.random.split(k)

            # 8. Decode to pixels
            pred_out = self.decoder(z_prev)
            pred_out = jnp.transpose(pred_out, (1, 2, 0))

            # 4. Coin flip logic
            t_ratio = step_idx / T
            is_context = t_ratio < inf_context_ratio
            is_forced = jax.random.bernoulli(subk, p_forcing)
            use_gt = jnp.logical_or(is_context, is_forced)

            # 2. Encode ground truth next frame
            z_next_gt = self.encoder(jnp.transpose(gt_curr_frame, (2, 0, 1)))

            # 6. Predict action
            a_gt = self.lam(z_prev, z_next_gt)

            # 5. Decide whether to use GT action or zero action
            a_t = jnp.where(use_gt, a_gt, jnp.zeros_like(a_gt))

            # 7. Step forward in latent space
            z_next = self.forward_dyn(z_prev, a_t)

            if CONFIG["use_nll_loss"]:
                mean, std = pred_out[..., :C], pred_out[..., C:]
                std = jax.nn.softplus(std) + 1e-4
                pred_out = jnp.concatenate([mean, std], axis=-1)

            return z_next, pred_out

        # Match old sequence length behavior
        scan_inputs = (jnp.concatenate([ref_video[1:], ref_video[-1:]], axis=0), jnp.arange(1, T+1))
        _, pred_video = jax.lax.scan(scan_step, z_init, scan_inputs)

        return pred_video

    def __call__(self, ref_videos, p_forcing, keys, coords_grid, inf_context_ratio, precompute_ref_diffs=False):
        # coords_grid and precompute_ref_diffs are ignored as they apply to the old weight-space model
        is_single = (ref_videos.ndim == 4)
        if is_single:
            ref_videos = ref_videos[None, ...]
            keys = keys[None, ...] if keys.ndim == 1 else keys
            
        batched_fn = jax.vmap(self._get_preds_single, in_axes=(0, None, 0, None))
        preds = batched_fn(ref_videos, p_forcing, keys, inf_context_ratio)
        
        if is_single:
            return preds[0]
        return preds

@eqx.filter_jit
def evaluate(m, batch, p_forcing, keys, coords, context_ratio, precompute_ref_diffs=False):
    return m(batch, p_forcing, keys, coords, context_ratio, precompute_ref_diffs)


#%% Cell 4: Initialization & Training/Loading Logic
key, subkey = jax.random.split(key)

# Pass the new hyperparameters into WorldModel initialization
model = WorldModel(
    frame_shape=(H, W, C), 
    dyn_dim=CONFIG["dynamics_space"], 
    lam_dim=CONFIG["lam_space"],
    split_forward=CONFIG["split_forward"],
    key=subkey
)
A_init = model.A.copy()

print(f"Total Trainable Parameters in WorldModel: {count_trainable_params(model)}")

count_A = count_trainable_params(model.forward_dyn.mlp_A)
count_B = count_trainable_params(model.forward_dyn.mlp_B)
count_lam = count_trainable_params(model.lam)
count_encoder = count_trainable_params(model.encoder)
count_decoder = count_trainable_params(model.decoder)
print(" - in the Encoder:", count_encoder)
print(" - in the Decoder:", count_decoder)
print(" - in the Forward Dynamics A:", count_A)
print(" - in the Forward Dynamics B:", count_B)
print(" - in the Inv. Dynamics (LAM):", count_lam)

if TRAIN:
    print(f"\n🚀 Starting WorldModel Training -> Saving to {run_path}")
    
    # --- Chain Adam with the Plateau Scheduler ---
    optimizer = optax.chain(
        optax.adam(CONFIG["learning_rate"]),
        optax.contrib.reduce_on_plateau(
            patience=CONFIG["lr_patience"],
            cooldown=CONFIG["lr_cooldown"],
            factor=CONFIG["lr_factor"],
            rtol=CONFIG["lr_rtol"],
            accumulation_size=CONFIG["lr_accum_size"],
            min_scale=CONFIG["lr_min_scale"]
        )
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, opt_state, keys, ref_videos, coords_grid, p_forcing):
        def loss_fn(m):
            k_full, k_init = jax.random.split(keys[0], 2)
            
            # 1. Standard Forward Pass
            pred_videos = m(ref_videos, p_forcing, keys, coords_grid, 0.0, precompute_ref_diffs=False)

            # --- SEQUENCE LOSS---
            pred_selected = pred_videos
            ref_selected = ref_videos
            
            if CONFIG["use_nll_loss"]:
                pred_means, pred_stds = pred_selected[..., :C], pred_selected[..., C:]
                nll = 0.5 * jnp.log(2 * jnp.pi * pred_stds**2) + 0.5 * ((ref_selected - pred_means)**2 / (pred_stds**2 + 1e-8))
                loss_full = jnp.mean(nll)
            else:
                loss_full = jnp.mean((pred_selected - ref_selected)**2)

            if CONFIG["aux_encoder_loss"]:
                # --- AUXILIARY ENCODER LOSS ---
                latent_z = eqx.filter_vmap(m.encoder)(jnp.transpose(ref_videos, (0, 1, 4, 2, 3)))
                recon_frames = eqx.filter_vmap(m.decoder)(latent_z)
                recon_frames = jnp.transpose(recon_frames, (0, 1, 3, 4, 2))
                ae_loss = jnp.mean((recon_frames - ref_videos)**2)
            else:
                ae_loss = 0.0

            return loss_full

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
        
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array), value=loss_val
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    all_losses = []
    lr_scales = []
    start_time = time.time()

    for epoch in range(CONFIG["nb_epochs"]):
        epoch_losses = []
        
        # for batch_idx, batch_videos in enumerate(train_loader):
        pbar = tqdm(train_loader)
        for batch_idx, batch_videos in enumerate(pbar):
            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, batch_videos.shape[0])
            
            model, opt_state, loss = train_step(model, opt_state, batch_keys, batch_videos, coords_grid, CONFIG["p_forcing"])
            epoch_losses.append(loss)
            
            current_scale = optax.tree_utils.tree_get(opt_state, "scale")
            lr_scales.append(current_scale)

            if batch_idx % 10 == 0:
                pbar.set_description(f"Epoch {epoch+1}/{CONFIG['nb_epochs']} - Loss: {loss:.4f} - LR Scale: {current_scale:.4f}")

        all_losses.extend(epoch_losses)

        if not SINGLE_BATCH and ((epoch+1) % CONFIG["print_every"] == 0 or (epoch+1) == CONFIG["nb_epochs"] - 1):
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1}/{CONFIG['nb_epochs']} - Avg Loss: {avg_epoch_loss:.4f} - LR Scale: {current_scale:.4f}", flush=True)

        if epoch in [4, CONFIG["nb_epochs"]//2, 2*CONFIG["nb_epochs"]//3]:
            eqx.tree_serialise_leaves(artefacts_path / f"tf_model_ep{epoch+1}.eqx", model)

        if (epoch+1) % (max(CONFIG["nb_epochs"]//10, 1)) == 0:
            val_keys = jax.random.split(key, sample_batch.shape[0])
            val_videos = evaluate(model, sample_batch, 0.0, val_keys, coords_grid, CONFIG["inf_context_ratio"], precompute_ref_diffs=False)
            plot_pred_ref_videos_rollout(val_videos[0], 
                                        sample_batch[0], 
                                        title=f"Pred", 
                                        save_name=f"pred_ref_epoch{epoch+1}.png")

    wall_time = time.time() - start_time
    print("\nWall time for WorldModel training in h:m:s:", time.strftime("%H:%M:%S", time.gmtime(wall_time)))
    
    eqx.tree_serialise_leaves(artefacts_path / "tf_model_final.eqx", model)
    np.save(artefacts_path / "loss_history.npy", np.array(all_losses))
    np.save(artefacts_path / "lr_history.npy", np.array(lr_scales))

else:
    print(f"\n📥 Loading WorldModel model from {artefacts_path}")
    model = eqx.tree_deserialise_leaves(artefacts_path / "tf_model_final.eqx", model)
    try:
        all_losses = np.load(artefacts_path / "loss_history.npy").tolist()
        lr_scales = np.load(artefacts_path / "lr_history.npy").tolist()
    except FileNotFoundError:
        all_losses = []
        lr_scales = []
        print("Warning: loss_history.npy not found.")

#%% Cell 5: Final Visualizations
print("\n=== Generating Dashboards ===")

if len(all_losses) > 0:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color1 = 'teal'
    ax1.plot(all_losses, color=color1, alpha=0.8, label="Total Loss")
    if not CONFIG["use_nll_loss"]:
        ax1.set_yscale('log')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)
    
    ax2 = ax1.twinx()  
    color2 = 'crimson'
    if len(lr_scales) > 0:
        ax2.plot(lr_scales, color=color2, linewidth=2, label="LR Scale Multiplier")
        ax2.set_ylabel("LR Scale", color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title("Training Loss and Adaptive Learning Rate Decay")
    fig.tight_layout()
    plt.savefig(plots_path / "loss_and_lr_history.png")
    plt.show()

A_final = model.A
subsample_step = max(1, 1) 
vmin, vmax = -1e-4, 1e-4

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
im1 = axes[0].imshow(A_init[::subsample_step, ::subsample_step], cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title(f"Recurrence Matrix A (Init)\nSubsampled step={subsample_step}")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(A_final[::subsample_step, ::subsample_step], cmap='viridis', vmin=vmin, vmax=vmax)
axes[1].set_title(f"Recurrence Matrix A (Final)\nSubsampled step={subsample_step}")
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig(plots_path / "recurrence_matrix_A.png")
plt.show()

#%% Cell 6: Testing & Animations
test_dataset = WeatherBenchTemperature(
    data_path=data_path, 
    split="test", 
    download=False, 
    seq_len=CONFIG["seq_len"],
    mean=train_dataset.mean,
    std=train_dataset.std
)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=numpy_collate, drop_last=False)
sample_batch = next(iter(test_loader))

print("Batch shape for evaluation:", sample_batch.shape)

pad_length = CONFIG["seq_len"] - sample_batch.shape[1]
if pad_length > 0:
    sample_batch = jnp.concatenate([sample_batch, np.zeros((sample_batch.shape[0], pad_length, H, W, C), dtype=sample_batch.dtype)], axis=1)

val_keys = jax.random.split(key, sample_batch.shape[0])
final_videos = evaluate(model, sample_batch, 0.0, val_keys, coords_grid, 0.5, precompute_ref_diffs=False)

if CONFIG["use_nll_loss"]:
    print(f"Final Predicted Video Mean Pixel Value Range: min={final_videos[...,:C].min():.4f}, max={final_videos[...,:C].max():.4f}")
    print(f"Final Predicted Video Std Pixel Value Range: min={final_videos[...,C:].min():.4f}, max={final_videos[...,C:].max():.4f}")

#%% Generate Final Outputs
test_seq_id = np.random.randint(0, sample_batch.shape[0])

# 1. Rollout Visualizer
plot_pred_ref_videos_rollout(
    final_videos[test_seq_id], 
    sample_batch[test_seq_id],
    title=f"Pred", 
    save_name=f"inference_forecast_rollout_seq{test_seq_id}.png"
)

# 2. Render Side-by-Side Video
video_html = animate_side_by_side(
    final_videos[test_seq_id], 
    sample_batch[test_seq_id]
)
display(video_html)

# %%
os.system(f"cp -r nohup.log {run_path}/nohup.log")
#%%