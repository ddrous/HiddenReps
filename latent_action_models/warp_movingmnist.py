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
from torch.utils.data import DataLoader, Subset

import seaborn as sns
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

def count_trainable_params(model):
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
    "nb_epochs": 2500,
    "print_every": 1,
    "batch_size": 2 if SINGLE_BATCH else 256*1,
    "learning_rate": 1e-4 if USE_NLL_LOSS else 1e-4,
    "p_forcing": 0.5,
    "inf_context_ratio": 0.5,
    "use_nll_loss": USE_NLL_LOSS,
    "aux_encoder_loss": False,
    "aux_loss_weight": 1,
    "aux_loss_num_steps": 4,

    # --- Architecture Params ---
    "lam_space": 32,
    "mem_space": 128*2,
    "split_forward": True,
    "root_width": 12,
    "root_depth": 5,
    "num_fourier_freqs": 6,
    "use_time_in_root": False,

    # --- Plateau Scheduler Config ---
    "lr_patience": 400,      
    "lr_cooldown": 100,       
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
    return videos

print("Loading Moving MNIST Dataset...")
try:
    data_path = './data' if TRAIN else '../../data'
    # dataset = datasets.MovingMNIST(root=data_path, split=None, download=True)

    ## Manulally load train and test splits to have more control over batching and shuffling
    mov_mnist_arrays = np.load(data_path + "/MovingMNIST/mnist_test_seq.npy")
    print(f"Original loaded MovingMNIST shape: {mov_mnist_arrays.shape} (T, N, H, W)")
    ## SPlit, 8000 train, 2000 test
    train_arrays = mov_mnist_arrays[:, :8000]
    test_arrays = mov_mnist_arrays[:, 8000:]

    ## Create PyTorch dataset
    class MovingMNISTDataset(torch.utils.data.Dataset):
        def __init__(self, data_array):
            self.data_array = data_array

        def __len__(self):
            return self.data_array.shape[1]

        def __getitem__(self, idx):
            video = self.data_array[:, idx]  # Shape (T, H, W)
            video = np.expand_dims(video, axis=-1)  # Add channel dimension -> (T, H, W, 1)
            return torch.from_numpy(video.astype(np.float32))

    dataset = MovingMNISTDataset(train_arrays)

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
                                  drop_last=False)

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

def plot_pred_ref_videos_rollout(video, ref_video, title="Render", save_name=None):
    nb_frames = video.shape[0]

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
        return self.layers[-1](x)

class CNNEncoder(eqx.Module):
    layers: list

    def __init__(self, in_channels, out_dim, spatial_shape, key, hidden_width=16, depth=4):
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
            
        dummy_x = jnp.zeros((in_channels, H, W))
        for layer in conv_layers:
            dummy_x = layer(dummy_x)

        flat_dim = dummy_x.reshape(-1).shape[0]
        self.layers = conv_layers + [eqx.nn.Linear(flat_dim, out_dim, key=keys[depth])]
        
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = x.reshape(-1)
        x = self.layers[-1](x)
        return x

class LAM(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, dyn_dim, lam_dim, key):
        self.mlp = eqx.nn.MLP(dyn_dim * 2, lam_dim, width_size=dyn_dim*1, depth=2, key=key)
        
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

    def get_latents(self, z_prev, a):
        if self.split_forward:
            return self.mlp_A(z_prev), self.mlp_B(a)
        else:
            return NotImplementedError("get_latents is only implemented for split_forward=True")


class MemoryModule(eqx.Module):
    """
    A Gated Memory Module.
    Allows for dynamic remembering of latent actions over time.
    """
    self_mlp: eqx.nn.MLP
    candidate_mlp: eqx.nn.MLP
    decoder_mlp: eqx.nn.MLP

    def __init__(self, lam_dim, mem_dim, key):
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Encoding requires the action, the current memory state, and time.
        # enc_in_dim = lam_dim + mem_dim + 1
        enc_in_dim = lam_dim + 0
        
        # int_dim = (mem_dim + lam_dim) // 2
        int_dim = mem_dim * 2

        # 1. The Gate Network: Outputs values squashed between 0 and 1 (via sigmoid)
        # It decides WHAT to forget and WHAT to let in.
        self.self_mlp = eqx.nn.MLP(mem_dim, mem_dim, width_size=mem_dim*2, depth=2, key=k1)
        
        # 2. The Candidate Network: Outputs values squashed between -1 and 1 (via tanh)
        # It represents the NEW information extracted from the current action.
        self.candidate_mlp = eqx.nn.MLP(enc_in_dim, mem_dim, width_size=int_dim, depth=2, key=k2)

        # Decoding just needs the memory state and time.
        dec_in_dim = mem_dim + 0
        self.decoder_mlp = eqx.nn.MLP(dec_in_dim, lam_dim, width_size=int_dim, depth=2, key=k3)

    def reset(self, mem_dim):
        """Returns an empty initial memory state."""
        return jnp.zeros((mem_dim,))

    def encode(self, m, a, t):
        """
        Ingests a new action 'a' at time 't', updating memory 'm'.
        """
        # Combine the new action, current memory, and time
        # x = jnp.concatenate([a, m, jnp.array([t], dtype=a.dtype)], axis=0)
        # x = jnp.concatenate([a, jnp.array([t], dtype=a.dtype)], axis=0)
        x = a
        
        # gate ~ 1.0 means "overwrite with new candidate"
        # gate ~ 0.0 means "ignore new action, keep old memory"
        # gate = jax.nn.sigmoid(self.gate_mlp(x))

        # The proposed new memory content based on this event
        # candidate = jax.nn.tanh(self.candidate_mlp(x))
        # candidate = self.candidate_mlp(x)
        
        # Smooth interpolation: Forget part of the old, write part of the new.
        # This guarantees the memory 'm' remains bounded between -1 and 1 over time.
        # m_new = (1.0 - gate) * m + gate * candidate

        # m_new = m + candidate

        m_new = self.self_mlp(m) + self.candidate_mlp(x)

        return m_new
        # return jax.nn.tanh(m_new)

    def decode(self, m, t):
        """
        Queries the memory 'm' to extract an action for time 't'.
        """
        # x = jnp.concatenate([m, jnp.array([t], dtype=m.dtype)], axis=0)
        x = m
        return self.decoder_mlp(x)


class WARP(eqx.Module):
    encoder: CNNEncoder
    lam: LAM
    forward_dyn: ForwardDynamics
    theta_base: jax.Array

    memory: Optional[MemoryModule]
    
    unravel_fn: callable = eqx.field(static=True)
    d_theta: int = eqx.field(static=True)
    lam_dim: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)
    split_forward: bool = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)
    mem_dim: int = eqx.field(static=True)

    def __init__(self, root_width, root_depth, num_freqs, frame_shape, lam_dim, mem_dim, split_forward, key):
        k_root, k_enc, k_lam, k_fwd, k_mem = jax.random.split(key, 5)
        self.frame_shape = frame_shape
        self.num_freqs = num_freqs
        self.lam_dim = lam_dim
        self.split_forward = split_forward
        H, W, C = frame_shape

        # Set up implicit renderer (decoder)
        coord_dim = 2 + 2 * 2 * num_freqs 
        root_out_dim = C * 2 if CONFIG["use_nll_loss"] else C
        add_time = 1 if CONFIG["use_time_in_root"] else 0
        template_root = RootMLP(coord_dim+add_time, root_out_dim, root_width, root_depth, k_root)
        
        flat_params, self.unravel_fn = ravel_pytree(template_root)
        self.d_theta = flat_params.shape[0]
        self.theta_base = flat_params

        # Set up JEPA dynamics components
        self.encoder = CNNEncoder(in_channels=C, out_dim=self.d_theta, spatial_shape=(H, W), key=k_enc, hidden_width=64)
        self.lam = LAM(self.d_theta, lam_dim, key=k_lam)
        self.forward_dyn = ForwardDynamics(self.d_theta, lam_dim, split_forward, key=k_fwd)

        self.mem_dim = mem_dim
        self.memory = MemoryModule(self.lam_dim, self.mem_dim, key=k_mem)

    @property
    def A(self):
        if self.split_forward:
            return self.forward_dyn.mlp_A.layers[-1].weight
        else:
            return self.forward_dyn.giant_mlp.layers[-1].weight

    def render_pixels(self, theta, coords):
        def render_pt(theta, coord):
            root = self.unravel_fn(theta)
            if CONFIG["use_time_in_root"]:
                encoded_coord = jnp.concatenate([coord[:1], fourier_encode(coord[1:], self.num_freqs)], axis=-1)      ##@TODO: maybe add time coord here in the future?
            else:
                encoded_coord = fourier_encode(coord[1:], self.num_freqs)
            out = root(encoded_coord)
            if CONFIG["use_nll_loss"]:
                mean, std = out[:C], out[C:]
                std = jax.nn.softplus(std) + 1e-4
                return jnp.concatenate([mean, std], axis=-1)
            return out
        # return jax.vmap(render_pt)(thetas, coords)
        return jax.vmap(render_pt, in_axes=(None, 0))(theta, coords)

    def render_frame(self, theta_offset, coords_grid):
        H, W, C = self.frame_shape
        flat_coords = coords_grid.reshape(-1, 3)
        
        # Add the base weights before rendering!
        theta = theta_offset + self.theta_base
        # thetas_frame = jnp.tile(theta, (H*W, 1))
        
        pred_flat = self.render_pixels(theta, flat_coords)
        return pred_flat.reshape(H, W, -1)

    # def _get_preds_single(self, ref_video, p_forcing, key, coords_grid, inf_context_ratio):
    #     T = ref_video.shape[0]
    #     init_frame = ref_video[0]

    #     # 1. Initialize offset from first frame
    #     z_init = self.encoder(jnp.transpose(init_frame, (2, 0, 1)))

    #     def scan_step(z_prev, scan_inputs):
    #         gt_curr_frame, step_idx = scan_inputs
    #         k = jax.random.fold_in(key, step_idx)
    #         k, subk = jax.random.split(k)

    #         # Render pixels explicitly using our helper
    #         pred_out = self.render_frame(z_prev, coords_grid)

    #         # Determine if we are forcing towards ground truth this step
    #         t_ratio = step_idx / T
    #         is_context = t_ratio < inf_context_ratio
    #         # is_forced = jax.random.bernoulli(subk, p_forcing)       #TODO: use mode="high" for small p
    #         # use_gt = jnp.logical_or(is_context, is_forced)
    #         # is_forced = False
    #         use_gt = is_context         ##TODO: basically, is_forced = False

    #         # Encode the ground truth future
    #         # z_next_gt = self.encoder(jnp.transpose(gt_curr_frame, (2, 0, 1)))
    #         z_next_gt = jax.lax.stop_gradient(self.encoder(jnp.transpose(gt_curr_frame, (2, 0, 1))))

    #         # Compute the required action to hit GT
    #         a_gt = self.lam(z_prev, z_next_gt)

    #         # THE FIX: If using GT, use the LAM's action. Otherwise, default to 0.
    #         a_t = jnp.where(use_gt, a_gt, jnp.zeros(self.lam_dim))

    #         # SINGLE forward dynamics call handles both the forced and autoregressive paths!
    #         z_next = self.forward_dyn(z_prev, a_t)

    #         return z_next, (z_next, pred_out)

    #     scan_inputs = (jnp.concatenate([ref_video[1:], ref_video[-1:]], axis=0), jnp.arange(1, T+1))
    #     _, (pred_latents, pred_video) = jax.lax.scan(scan_step, z_init, scan_inputs)

    #     return pred_latents, pred_video

    def _get_preds_single(self, ref_video, p_forcing, key, coords_grid, context_ratio):
        T = ref_video.shape[0]
        init_frame = ref_video[0]

        # 1. Initialize offset from first frame
        z_init = self.encoder(jnp.transpose(init_frame, (2, 0, 1)))

        m_init = jnp.zeros(self.mem_dim)

        # 2. Add the Equinox checkpointing decorator here!
        # Recompute internal activations during the backward pass.
        @eqx.filter_checkpoint
        def scan_step(carry, scan_inputs):
            z_t, m_t = carry
            o_tp1, step_idx = scan_inputs
            # subk = jax.random.fold_in(key, step_idx)

            # --- Rendering INSIDE the checkpointed step ---
            # Concatenate the time t to the coordinates before rendering
            time_coord = jnp.array([(step_idx-1)/ (T-1)], dtype=z_t.dtype)
            # print("SHapes in scan_step:", z_t.shape, m_t.shape, o_tp1.shape, coords_grid.shape)
            coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)

            pred_out = self.render_frame(z_t, coords_grid_t)

            # Determine if we are forcing towards ground truth this step
            t_ratio = step_idx / T
            is_context = t_ratio < context_ratio
            # is_forced = jax.random.bernoulli(subk, p_forcing)       #TODO: use mode="high" for small p
            # use_gt = jnp.logical_or(is_context, is_forced)
            use_gt = is_context

            # ONLY compute encoder and LAM when use_gt is True
            a_t = jax.lax.cond(
                use_gt,
                lambda: self.lam(
                    z_t, 
                    jax.lax.stop_gradient(self.encoder(jnp.transpose(o_tp1, (2, 0, 1))))
                ),
                lambda: self.memory.decode(m_t, None)
            )

            ## Encode into the memory with C
            m_tp1 = self.memory.encode(m_t, a_t, None)

            # SINGLE forward dynamics call handles both the forced and autoregressive paths!
            z_tp1 = self.forward_dyn(z_t, a_t)

            return (z_tp1, m_tp1), (z_t, pred_out)

        scan_inputs = (jnp.concatenate([ref_video[1:], ref_video[-1:]], axis=0), jnp.arange(1, T+1))
        
        # 3. Execute the scan as normal, collecting both latents and rendered frames
        _, (pred_latents, pred_video) = jax.lax.scan(scan_step, (z_init, m_init), scan_inputs)

        return pred_latents, pred_video

    def __call__(self, ref_videos, p_forcing, keys, coords_grid, inf_context_ratio, precompute_ref_diffs=False):
        is_single = (ref_videos.ndim == 4)
        if is_single:
            ref_videos = ref_videos[None, ...]
            keys = keys[None, ...] if keys.ndim == 1 else keys
            
        batched_fn = jax.vmap(self._get_preds_single, in_axes=(0, None, 0, None, None))
        pred_latents, pred_videos = batched_fn(ref_videos, p_forcing, keys, coords_grid, inf_context_ratio)
        
        if is_single:
            return pred_latents[0], pred_videos[0]
        return pred_latents, pred_videos

@eqx.filter_jit
def evaluate(m, batch, p_forcing, keys, coords, context_ratio, precompute_ref_diffs=False):
    return m(batch, p_forcing, keys, coords, context_ratio, precompute_ref_diffs)

#%% Cell 4: Initialization & Training/Loading Logic
key, subkey = jax.random.split(key)

model = WARP(
    root_width=CONFIG["root_width"],
    root_depth=CONFIG["root_depth"],
    num_freqs=CONFIG["num_fourier_freqs"],
    frame_shape=(H, W, C), 
    lam_dim=CONFIG["lam_space"],
    mem_dim=CONFIG["mem_space"],
    split_forward=CONFIG["split_forward"],
    key=subkey
)
A_init = model.A.copy()


print(f"Dynamics Weight Space Dimension (d_theta): {model.d_theta}")

print(f"Total Trainable Parameters in WARP: {count_trainable_params(model)}")

count_A = count_trainable_params(model.forward_dyn.mlp_A)
count_B = count_trainable_params(model.forward_dyn.mlp_B)
count_lam = count_trainable_params(model.lam)
count_memory = count_trainable_params(model.memory)
count_encoder = count_trainable_params(model.encoder)
theta_base = count_trainable_params(model.theta_base)
print(" - in the Encoder:", count_encoder)
print(" - in the base theta:", theta_base)
print(" - in the Memory Module:", count_memory)
print(" - in the Forward Dynamics A:", count_A)
print(" - in the Forward Dynamics B:", count_B)
print(" - in the Inv. Dynamics (LAM):", count_lam)

if TRAIN:
    print(f"\n🚀 Starting WARP Training -> Saving to {run_path}")
    
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
            
            # Forward pass: Extract both predicted thetas and rendered pixels
            # pred_thetas, pred_videos = m(ref_videos, p_forcing, keys, coords_grid, 0.0, precompute_ref_diffs=False)
            # pred_thetas, pred_videos = m(ref_videos, p_forcing, keys, coords_grid, CONFIG["inf_context_ratio"], precompute_ref_diffs=False)

            # context_ratio = jax.random.uniform(k_full, minval=0.0, maxval=1.0)
            # context_ratio = jax.random.uniform(k_full, minval=0.25, maxval=1.0)
            context_ratio = CONFIG["inf_context_ratio"]
            _, pred_videos = m(ref_videos, p_forcing, keys, coords_grid, context_ratio, precompute_ref_diffs=False)

            # --- 1. LATENT (WEIGHT-SPACE) DYNAMICS LOSS (Primary) ---
            # latent_loss = jnp.mean((pred_thetas - target_thetas_shifted)**2)
            rec_loss = jnp.mean((pred_videos - ref_videos)**2)

            # --- 2. AUTOENCODING LOSS (Auxiliary) ---
            if CONFIG["aux_encoder_loss"]:

                indices = jax.random.choice(k_init, ref_videos.shape[1], shape=(CONFIG["aux_loss_num_steps"],), replace=False)

                # Encode ground truth to target thetas
                ref_videos_enc = jnp.transpose(ref_videos[:, indices], (0, 1, 4, 2, 3))
                target_thetas = jax.vmap(jax.vmap(m.encoder))(ref_videos_enc)

                # Render Target Thetas -> Match GT Pixels
                # We vmap our fast render_frame function across Batches and Time steps
                ## Concat t=0 to the coords
                coords_grid_t0 = jnp.concatenate([jnp.zeros_like(coords_grid[..., :1]), coords_grid], axis=-1)
                batched_render = jax.vmap(jax.vmap(lambda theta: m.render_frame(theta, coords_grid_t0)))
                decoded_pixels = batched_render(target_thetas)

                if CONFIG["use_nll_loss"]:
                    decoded_mean = decoded_pixels[..., :C]
                    ae_loss = jnp.mean((decoded_mean - ref_videos[:, indices])**2)
                else:
                    ae_loss = jnp.mean((decoded_pixels - ref_videos[:, indices])**2)
            else:
                ae_loss = 0.0

            total_loss = rec_loss + CONFIG["aux_loss_weight"] * ae_loss

            return total_loss

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
        
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array), value=loss_val
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    @eqx.filter_jit
    def compute_p_forcing(epoch, schedule="linear", start=1.0, end=0.0):
        ### if linear decay, make if JAX numpy compatible and compute outside of train_step
        if schedule == "linear":
            p = start + (end - start) * (epoch / CONFIG["nb_epochs"])
            # return jnp.array(p, dtype=jnp.float32).item()
            return p
        elif schedule == "constant":
            return start
        elif schedule == "exponential":
            decay_rate = (end / start) ** (1 / CONFIG["nb_epochs"])
            p = start * (decay_rate ** epoch)
            # return jnp.array(p, dtype=jnp.float32).item()
            return p
        elif schedule == "step":
            ## 5 steps for now, can be made more flexible
            possible_ps = jnp.linspace(start, end, num=5)
            step_size = CONFIG["nb_epochs"] // 5
            step_idx = epoch // step_size
            return possible_ps[min(step_idx, len(possible_ps)-1)]

    all_losses = []
    lr_scales = []
    start_time = time.time()

    for epoch in range(CONFIG["nb_epochs"]):
        epoch_losses = []
        for batch_idx, batch_videos in enumerate(train_loader):
            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, batch_videos.shape[0])
            
            model, opt_state, loss = train_step(model, opt_state, batch_keys, batch_videos, coords_grid, CONFIG["p_forcing"])

            # p_forcing = compute_p_forcing(epoch, schedule="step", start=CONFIG["p_forcing"], end=0.0)
            # model, opt_state, loss = train_step(model, opt_state, batch_keys, batch_videos, coords_grid, p_forcing)

            epoch_losses.append(loss)

            current_scale = optax.tree_utils.tree_get(opt_state, "scale")
            lr_scales.append(current_scale)

        all_losses.extend(epoch_losses)

        if not SINGLE_BATCH and ((epoch+1) % CONFIG["print_every"] == 0 or (epoch+1) == CONFIG["nb_epochs"] - 1):
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1}/{CONFIG['nb_epochs']} - Avg Loss: {avg_epoch_loss:.4f} - LR Scale: {current_scale:.4f}", flush=True)

        if epoch in [4, CONFIG["nb_epochs"]//2, 2*CONFIG["nb_epochs"]//3]:
            eqx.tree_serialise_leaves(artefacts_path / f"model_ep{epoch+1}.eqx", model)

        if (epoch+1) % (max(CONFIG["nb_epochs"]//10, 1)) == 0:
            val_keys = jax.random.split(key, sample_batch.shape[0])
            _, val_videos = evaluate(model, sample_batch, 0.0, val_keys, coords_grid, CONFIG["inf_context_ratio"], precompute_ref_diffs=False)
            plot_pred_ref_videos_rollout(val_videos[0], 
                                        sample_batch[0], 
                                        title=f"Pred", 
                                        save_name=f"pred_ref_epoch{epoch+1}.png")

    wall_time = time.time() - start_time
    print("\nWall time for WARP training in h:m:s:", time.strftime("%H:%M:%S", time.gmtime(wall_time)))
    
    eqx.tree_serialise_leaves(artefacts_path / "model_final.eqx", model)
    np.save(artefacts_path / "loss_history.npy", np.array(all_losses))
    np.save(artefacts_path / "lr_history.npy", np.array(lr_scales))

else:
    print(f"\n📥 Loading WARP model from {artefacts_path}")
    model = eqx.tree_deserialise_leaves(artefacts_path / "model_final.eqx", model)
    try:
        all_losses = np.load(artefacts_path / "loss_history.npy").tolist()
        lr_scales = np.load(artefacts_path / "lr_history.npy").tolist()
    except FileNotFoundError:
        all_losses = []
        lr_scales = []
        print("Warning: loss_history.npy not found.")

#%% Cell 5: Final Visualizations
print("\n=== Generating Dashboards ===")

# print(len(all_losses), "loss points collected during training.")

if len(all_losses) > 0:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color1 = 'teal'
    ax1.plot(all_losses, color=color1, alpha=0.8, label="Total Loss")
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

#%%

# testing_subset = datasets.MovingMNIST(root=data_path, split="test", download=True)

testing_subset = MovingMNISTDataset(test_arrays)
test_loader = DataLoader(testing_subset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=numpy_collate, drop_last=False)
sample_batch = next(iter(test_loader))

# sample_batch = next(iter(train_loader))

print("Batch shape for evaluation:", sample_batch.shape, flush=True)

pad_length = 20 - sample_batch.shape[1]
sample_batch = jnp.concatenate([sample_batch, np.zeros((sample_batch.shape[0], pad_length, H, W, C), dtype=sample_batch.dtype)], axis=1)

val_keys = jax.random.split(key, sample_batch.shape[0])
_, final_videos = evaluate(model, sample_batch, 0.0, val_keys, coords_grid, 0.5, precompute_ref_diffs=False)

if CONFIG["use_nll_loss"]:
    print(f"Final Predicted Video Mean Pixel Value Range: min={final_videos[...,:C].min():.4f}, max={final_videos[...,:C].max():.4f}")
    print(f"Final Predicted Video Std Pixel Value Range: min={final_videos[...,C:].min():.4f}, max={final_videos[...,C:].max():.4f}")

#%%
test_seq_id = np.random.randint(0, sample_batch.shape[0])

plot_pred_ref_videos_rollout(
    final_videos[test_seq_id], 
    sample_batch[test_seq_id],
    title=f"Pred", 
    save_name=f"inference_forecast_rollout_seq{test_seq_id}.png"
)

# %%
os.system(f"cp -r nohup.log {run_path}/nohup.log")

#%%

