#%% Cell 1: Imports, Utilities, and Configuration
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from pathlib import Path
import shutil
import sys
from tqdm import tqdm
from jax.flatten_util import ravel_pytree
from typing import Optional

import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, Sampler

import seaborn as sns
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

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

SINGLE_BATCH = True
USE_NLL_LOSS = False

CONFIG = {
    "seed": 2026,
    "nb_epochs": 250,
    "print_every": 1,
    "batch_size": 1 if SINGLE_BATCH else 32,
    "learning_rate": 1e-7 if USE_NLL_LOSS else 1e-5,
    "p_forcing": 0.5,
    "inf_context_ratio": 0.5,
    "nb_loss_steps_full": 20,
    "nb_loss_steps_init": 20,
    "rec_feat_dim": 128,
    "root_width": 12,
    "root_depth": 5,
    "num_fourier_freqs": 6,
    "use_nll_loss": USE_NLL_LOSS,

    # --- Architecture Params ---
    "num_layers": 1,
    "use_hypernet": True,
    "use_controlnet": True,

    # --- Architecture Params ---
    "dynamics_space": 128,
    "lam_space": 32,
    "split_forward": True,

    # --- Plateau Scheduler Config ---
    "lr_patience": 500,      
    "lr_cooldown": 0,       
    "lr_factor": 0.5,        
    "lr_rtol": 1e-3,         
    "lr_accum_size": 5,     
    "lr_min_scale": 1e-2     
}

key = jax.random.PRNGKey(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

def setup_run_dir(base_dir="experiments"):
    if TRAIN:
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        run_path = Path(base_dir) / timestamp
        run_path.mkdir(parents=True, exist_ok=True)

        ## Copy this script to the run directory
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
def numpy_collate(batch):
    if isinstance(batch[0], tuple):
        videos = torch.stack([b[0] for b in batch]).numpy()
    else:
        videos = torch.stack(batch).numpy()
    
    if videos.ndim == 4:
        videos = np.expand_dims(videos, axis=-1)
    elif videos.ndim == 5 and videos.shape[2] == 1:
        videos = np.transpose(videos, (0, 1, 3, 4, 2))

    videos = videos.astype(np.float32)
    if videos.max() > 2.0:
        videos = videos / 255.0

    ## Rescale to [-1, 1]
    # videos = videos * 2.0 - 1.0

    return videos

print("Loading Moving MNIST Dataset...")
try:
    data_path = './data' if TRAIN else '../../data'
    dataset = datasets.MovingMNIST(root=data_path, split=None, download=True)
    # sampler = SlidingWindowBatchSampler(
    #     dataset_size=len(dataset),
    #     batch_size=CONFIG["batch_size"],
    #     shuffle=True,       # shuffles the global ordering once per epoch
    #     drop_last=True,
    # )
    # sampler = RepeatingBatchSampler(
    #     dataset_size=len(dataset),
    #     batch_size=CONFIG["batch_size"],
    #     new_batch_every=5,   # each batch is repeated for 5 consecutive steps before
    #     shuffle=True,       # shuffles the global ordering once per epoch
    #     drop_last=True,
    # )
    if SINGLE_BATCH:
        training_subset = Subset(dataset, range(CONFIG["batch_size"]))
        train_loader = DataLoader(
            training_subset, 
            batch_size=CONFIG["batch_size"]//1, 
            shuffle=False, 
            collate_fn=numpy_collate, 
            drop_last=True
        )
    else:
        train_loader = DataLoader(dataset, 
                                  batch_size=CONFIG["batch_size"], 
                                  shuffle=True, 
                                  collate_fn=numpy_collate, 
                                  drop_last=True)
        # train_loader = DataLoader(dataset, 
        #                           batch_sampler=sampler, 
        #                           collate_fn=numpy_collate)

    sample_batch = next(iter(train_loader))
    B, nb_frames, H, W, C = sample_batch.shape
    print(f"Batched Video shape: {sample_batch.shape}")
except Exception as e:
    print(f"Could not load MovingMNIST: {e}")
    raise e

y_coords = jnp.linspace(-1, 1, H)
x_coords = jnp.linspace(-1, 1, W)
X_grid, Y_grid = jnp.meshgrid(x_coords, y_coords)
coords_grid = jnp.stack([X_grid, Y_grid], axis=-1) 

def sbimshow(img, title="", ax=None):
    img = np.clip(img, 0.0, 1.0)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if ax is None:
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    else:
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

def plot_pred_ref_videos(video, ref_video, title="Render", save_name=None):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    t_mid = len(video) // 2
    t_end = len(video) - 1
    
    sbimshow(video[0], title=f"{title} t=0", ax=ax[0, 0])
    sbimshow(video[t_mid], title=f"{title} t={t_mid}", ax=ax[0, 1])
    sbimshow(video[t_end], title=f"{title} t={t_end}", ax=ax[0, 2])

    sbimshow(ref_video[0], title="Ref t=0", ax=ax[1, 0])
    sbimshow(ref_video[t_mid], title=f"Ref t={t_mid}", ax=ax[1, 1])
    sbimshow(ref_video[t_end], title=f"Ref t={t_end}", ax=ax[1, 2])
    plt.tight_layout()
    if save_name:
        plt.savefig(plots_path / save_name)
    plt.show()

def plot_pred_ref_videos_rollout(video, ref_video, title="Render", save_name=None):
    nb_frames = video.shape[0]

    ## Rescale to [0,1] if in [-1,1]
    rescale = False
    if ref_video[..., :C].min() < 0.0:
        rescale = True
        ref_video = (ref_video + 1.0) / 2.0

    if video.shape[-1] == 1:
        fig, axes = plt.subplots(2, 1+(nb_frames//2), figsize=(20, 6))
        indices_to_plot = list(np.arange(0, nb_frames, 2)) + [nb_frames-1]
        for i, idx in enumerate(indices_to_plot):
            video_to_plot = video[idx] if not rescale else (video[idx] + 1.0) / 2.0
            sbimshow(video_to_plot, title=f"{title} t={idx}", ax=axes[0, i])
            sbimshow(ref_video[idx], title=f"Ref t={idx}", ax=axes[1, i])
    else:
        ## We want to plot uncertainties as well
        fig, axes = plt.subplots(3, 1+(nb_frames//2), figsize=(20, 7))
        indices_to_plot = list(np.arange(0, nb_frames, 2)) + [nb_frames-1]
        for i, idx in enumerate(indices_to_plot):
            video_to_plot = video[idx, ..., :C] if not rescale else (video[idx, ..., :C] + 1.0) / 2.0
            sbimshow(video_to_plot, title=f"Mean t={idx}", ax=axes[0, i])
            sbimshow(video[idx, ..., C:], title=f"Std t={idx}", ax=axes[1, i])
            sbimshow(ref_video[idx], title=f"Ref t={idx}", ax=axes[2, i])

    plt.tight_layout()
    if save_name:
        plt.savefig(plots_path / save_name)
    plt.show()

#%% Cell 3: Model Definition
def fourier_encode(x, num_freqs):
    freqs = 2.0 ** jnp.arange(num_freqs)
    angles = x[..., None] * freqs[None, None, :] * jnp.pi
    angles = angles.reshape(*x.shape[:-1], -1)
    return jnp.concatenate([x, jnp.sin(angles), jnp.cos(angles)], axis=-1)

class RootMLP(eqx.Module):
    layers: list

    def __init__(self, in_size, out_size, width, depth, key):
        keys = jax.random.split(key, depth + 1)
        self.layers = [eqx.nn.Linear(in_size, width, key=keys[0])]
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i+1]))
        self.layers.append(eqx.nn.Linear(width, out_size, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
            # x = jnp.sin(layer(x))
        return self.layers[-1](x)

class CNNEncoder(eqx.Module):
    layers: list

    def __init__(self, in_channels, out_dim, spatial_shape, key, hidden_width=16, depth=3):
        H, W = spatial_shape
        keys = jax.random.split(key, depth + 1)
        
        conv_layers = []
        current_in_channels = in_channels
        current_out_channels = hidden_width
        
        for i in range(depth):
            conv_layers.append(
                eqx.nn.Conv2d(current_in_channels, current_out_channels, kernel_size=3, stride=2, padding=1, key=keys[i])
            )
            current_in_channels = current_out_channels
            current_out_channels *= 2
            
        dummy_x = jnp.zeros((in_channels, H, W))
        for layer in conv_layers:
            dummy_x = layer(dummy_x)

        flat_dim = dummy_x.reshape(-1).shape[0]
        self.layers = conv_layers + [eqx.nn.Linear(flat_dim, out_dim, key=keys[depth])]
        
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = jax.nn.relu(x)
        x = x.reshape(-1)
        x = self.layers[-1](x)
        return x

class LinearRNNLayer(eqx.Module):
    A: jax.Array
    B: jax.Array
    # C: jax.Array = eqx.field(static=True)  # For potential FiLM conditioning, not used in this version

    def __init__(self, in_dim, out_dim, key):
        k_A, k_B = jax.random.split(key)

        ## define a MLP
        # self.A = eqx.nn.MLP(out_dim, out_dim, width_size=out_dim*2, depth=2, activation=jax.nn.softplus, key=k_A)
        # self.A = eqx.nn.MLP(out_dim, out_dim, width_size=out_dim*2, depth=2, key=k_A)
        self.A = eqx.nn.MLP(out_dim, out_dim, width_size=int(out_dim*1.5), depth=2, key=k_A)

        # self.A = jnp.eye(out_dim)
        # self.A = jax.random.normal(k_A, (out_dim, out_dim)) * 0.01
        # self.A = jax.random.normal(k_A, (out_dim, out_dim)) * 0

        ## A should be a transport operator close to identiy. Initilialise as a transport matrix
        # self.A = jnp.eye(out_dim) + jax.random.normal(k_A, (out_dim, out_dim)) * 0.0001
        
        # self.B = jnp.zeros((out_dim, CONFIG["rec_feat_dim"]))
        # self.B = jnp.zeros((out_dim, in_dim))
        self.B = eqx.nn.MLP(in_dim, out_dim, width_size=(out_dim+in_dim)//2, depth=2, key=k_B)
        # self.B = eqx.nn.MLP(in_dim, out_dim, width_size=out_dim*2, depth=2, activation=jax.nn.softplus, key=k_B)

        # self.C = jnp.zeros((out_dim, in_dim))  # Not used in this version, but reserved for potential FiLM conditioning

#%% Cell 3: Model Definition
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
            
        # Compute flattened dimension
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
        self.mlp = eqx.nn.MLP(dyn_dim * 2, lam_dim, width_size=dyn_dim, depth=2, key=key)
        
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
            self.mlp_A = eqx.nn.MLP(dyn_dim, dyn_dim, width_size=dyn_dim*2, depth=2, key=k1)
            self.mlp_B = eqx.nn.MLP(lam_dim, dyn_dim, width_size=dyn_dim*2, depth=2, key=k2)
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

class WARP(eqx.Module):
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

        self.encoder = CNNEncoder(in_channels=C, out_dim=dyn_dim, spatial_shape=(H, W), key=k1)
        self.decoder = CNNDecoder(in_dim=dyn_dim, out_channels=C*2 if CONFIG["use_nll_loss"] else C, spatial_shape=(H, W), key=k2)
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

            # 2. Encode ground truth next frame
            z_next_gt = self.encoder(jnp.transpose(gt_curr_frame, (2, 0, 1)))

            # 3. Model's own prediction of the next latent space (Assumes null action)
            z_next_base = self.forward_dyn(z_prev, jnp.zeros(self.lam_dim))

            # 4. Coin flip logic
            t_ratio = step_idx / (T - 1)
            is_context = t_ratio <= inf_context_ratio
            is_forced = jax.random.bernoulli(subk, p_forcing)
            use_gt = jnp.logical_or(is_context, is_forced)

            # 5. Decide target representation for LAM
            z_target = jnp.where(use_gt, z_next_gt, z_next_base)

            # 6. Predict action
            a_t = self.lam(z_prev, z_target)

            # 7. Step forward in latent space
            z_next = self.forward_dyn(z_prev, a_t)

            # 8. Decode to pixels
            pred_out = self.decoder(z_next)
            pred_out = jnp.transpose(pred_out, (1, 2, 0))

            if CONFIG["use_nll_loss"]:
                mean, std = pred_out[..., :C], pred_out[..., C:]
                std = jax.nn.softplus(std) + 1e-4
                pred_out = jnp.concatenate([mean, std], axis=-1)

            return z_next, pred_out

        # Match old sequence length behavior
        scan_inputs = (jnp.concatenate([ref_video[1:], ref_video[-1:]], axis=0), jnp.arange(T))
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

# Pass the new hyperparameters into WARP initialization
model = WARP(
    frame_shape=(H, W, C), 
    dyn_dim=CONFIG["dynamics_space"], 
    lam_dim=CONFIG["lam_space"],
    split_forward=CONFIG["split_forward"],
    key=subkey
)
A_init = model.A.copy()

print(f"Total Trainable Parameters in WARP: {count_trainable_params(model)}")

if TRAIN:
    print(f"\n🚀 Starting WARP Training -> Saving to {run_path}")
    
    # --- Chain Adam with the Plateau Scheduler ---
    optimizer = optax.chain(
        optax.adam(CONFIG["learning_rate"]),
        # optax.clip_by_global_norm(1e-3),  # Gradient clipping for stability
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
            
            # 1. Standard Forward Pass (Default behavior, precompute_ref_diffs is False)
            pred_videos = m(ref_videos, p_forcing, keys, coords_grid, 0.0, precompute_ref_diffs=False)

            # --- SEQUENCE LOSS---
            full_indices = jax.random.choice(k_full, ref_videos.shape[1], shape=(CONFIG["nb_loss_steps_full"],), replace=False)
            pred_selected = pred_videos[:, full_indices]
            ref_selected = ref_videos[:, full_indices]
            
            if CONFIG["use_nll_loss"]:
                # pred_means, pred_stds = pred_selected[..., :C], jax.nn.softplus(pred_selected[..., C:])
                pred_means, pred_stds = pred_selected[..., :C], pred_selected[..., C:]
                # pred_stds = jnp.clip(pred_stds, 1e-4, 10.0)
                nll = 0.5 * jnp.log(2 * jnp.pi * pred_stds**2) + 0.5 * ((ref_selected - pred_means)**2 / (pred_stds**2 + 1e-8))
                loss_full = jnp.mean(nll)
            else:
                loss_full = jnp.mean((pred_selected - ref_selected)**2)

            return loss_full

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
        
        # --- Pass the loss value into the optimizer update ---
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array), value=loss_val
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    all_losses = []
    lr_scales = []
    start_time = time.time()

    for epoch in range(CONFIG["nb_epochs"]):
        # if not SINGLE_BATCH:
        #     print(f"\nEPOCH: {epoch+1}", flush=True)
        epoch_losses = []
        
        # pbar = tqdm(train_loader)
        # for batch_idx, batch_videos in enumerate(pbar):
        for batch_idx, batch_videos in enumerate(train_loader):
            key, subkey = jax.random.split(key)
            # batch_keys = jax.random.split(subkey, CONFIG["batch_size"])
            batch_keys = jax.random.split(subkey, batch_videos.shape[0])
            
            model, opt_state, loss = train_step(model, opt_state, batch_keys, batch_videos, coords_grid, CONFIG["p_forcing"])
            epoch_losses.append(loss)
            
            # --- Extract and record the current learning rate scale ---
            current_scale = optax.tree_utils.tree_get(opt_state, "scale")
            lr_scales.append(current_scale)

            # if not SINGLE_BATCH and (batch_idx % CONFIG["print_every"]) == 0:
            #     pbar.set_description(f"Loss: {loss:.4f} | LR Scale: {current_scale:.4f}")

            # ## Stop after one bbatch if not in SINGLE_BATCH mode
            # if not SINGLE_BATCH:        ## TODO: fix this
            #     break

        all_losses.extend(epoch_losses)

        if not SINGLE_BATCH and ((epoch+1) % CONFIG["print_every"] == 0 or (epoch+1) == CONFIG["nb_epochs"] - 1):
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1}/{CONFIG['nb_epochs']} - Avg Loss: {avg_epoch_loss:.4f} - LR Scale: {current_scale:.4f}", flush=True)

        # Periodically save model over the training process
        if epoch in [4, CONFIG["nb_epochs"]//2, 2*CONFIG["nb_epochs"]//3]:
            eqx.tree_serialise_leaves(artefacts_path / f"tf_model_ep{epoch+1}.eqx", model)

        ## Generate intermediate visualizations at the end of each epoch
        if (epoch+1) % (max(CONFIG["nb_epochs"]//10, 1)) == 0:
            # val_keys = jax.random.split(key, CONFIG["batch_size"])
            val_keys = jax.random.split(key, sample_batch.shape[0])
            # Evaluate using standard autoregressive behavior
            val_videos = evaluate(model, sample_batch, 0.0, val_keys, coords_grid, CONFIG["inf_context_ratio"], precompute_ref_diffs=False)
            plot_pred_ref_videos_rollout(val_videos[0], 
                                        sample_batch[0], 
                                        title=f"Pred", 
                                        save_name=f"pred_ref_epoch{epoch+1}.png")

    wall_time = time.time() - start_time
    print("\nWall time for WARP training in h:m:s:", time.strftime("%H:%M:%S", time.gmtime(wall_time)))
    
    # Save final artifacts
    eqx.tree_serialise_leaves(artefacts_path / "tf_model_final.eqx", model)
    np.save(artefacts_path / "loss_history.npy", np.array(all_losses))
    np.save(artefacts_path / "lr_history.npy", np.array(lr_scales))

else:
    print(f"\n📥 Loading WARP model from {artefacts_path}")
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
    
    # Plot standard loss history on the left axis
    color1 = 'teal'
    ax1.plot(all_losses, color=color1, alpha=0.8, label="Total Loss")
    if CONFIG["use_nll_loss"]:
        # ax1.set_yscale('symlog', linthresh=1e-4)
        # ax1.set_yscale('log')
        pass
    else:
        ax1.set_yscale('log')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)
    
    # Plot LR Scale on the right axis to see when reductions triggered
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

# Plot Matrix A
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

#%%
# sample_batch = next(iter(train_loader))

## sample a batch from the test_loader (create it first)
testing_subset = datasets.MovingMNIST(root=data_path, split="test", download=True)
test_loader = DataLoader(testing_subset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=numpy_collate, drop_last=True)
sample_batch = next(iter(test_loader))

# print(f"Sample batch shape for inference: {sample_batch.shape}")

## Test sequences have length 10, so we all zeros to make it 20
pad_length = 20 - sample_batch.shape[1]
sample_batch = jnp.concatenate([sample_batch, np.zeros((sample_batch.shape[0], pad_length, H, W, C), dtype=sample_batch.dtype)], axis=1)

# Run inference utilizing the newly embedded `__call__` interface (handles batched transparently)
val_keys = jax.random.split(key, sample_batch.shape[0])
final_videos = evaluate(model, sample_batch, 0.0, val_keys, coords_grid, 0.5, precompute_ref_diffs=False)

## Print Min and Max pixel values
if CONFIG["use_nll_loss"]:
    print(f"Final Predicted Video Mean Pixel Value Range: min={final_videos[...,:C].min():.4f}, max={final_videos[...,:C].max():.4f}")
    print(f"Final Predicted Video Std Pixel Value Range: min={final_videos[...,C:].min():.4f}, max={final_videos[...,C:].max():.4f}")

#%%
test_seq_id = np.random.randint(0, sample_batch.shape[0])

plot_pred_ref_videos_rollout(
    final_videos[test_seq_id], 
    sample_batch[test_seq_id],
    title=f"Pred", 
    save_name="inference_forecast_rollout.png"
)

# %%
# Next idea: Use a random ambstract space, and see the difference in performance.

## Copy the nohup.log folder to the run folder
os.system(f"cp -r nohup.log {run_path}/nohup.log")

