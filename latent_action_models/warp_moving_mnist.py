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

import torch
from torchvision import datasets
from torch.utils.data import DataLoader

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
RUN_DIR = "./experiments/YOUR_RUN_FOLDER"  # Specify if TRAIN = False

CONFIG = {
    "seed": 2027,
    "nb_epochs": 50,
    "batch_size": 12,
    "learning_rate": 1e-7,
    "print_every": 5,
    "p_forcing": 0.0,
    "inf_context_ratio": 0.5,
    "rec_feat_dim": 1024,
    "root_width": 24,
    "root_depth": 2,
    "num_fourier_freqs": 3
}

key = jax.random.PRNGKey(CONFIG["seed"])

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
        
    ## Subsample the video, and rescalle between -1 and 1
    videos = videos[:, :, ::2, ::2]
    videos = videos * 2.0 - 1.0

    return videos

print("Loading Moving MNIST Dataset...")
try:
    dataset = datasets.MovingMNIST(root='./data', split=None, download=True)
    train_loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=numpy_collate, drop_last=True)
    
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

class WARP(eqx.Module):
    A: jax.Array
    B: jax.Array
    hypernet_phi: CNNEncoder
    controlnet_psi: CNNEncoder
    
    root_structure: RootMLP = eqx.field(static=True)
    unravel_fn: callable = eqx.field(static=True)
    d_theta: int = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)

    def __init__(self, root_width, root_depth, num_freqs, frame_shape, key):
        k_root, k_A, k_B, k_phi, k_psi = jax.random.split(key, 5)
        self.num_freqs = num_freqs
        self.frame_shape = frame_shape
        H, W, C = frame_shape
        
        coord_dim = 2 + 2 * 2 * num_freqs 
        template_root = RootMLP(coord_dim, 3, root_width, root_depth, k_root)

        flat_params, self.unravel_fn = ravel_pytree(template_root)
        self.d_theta = flat_params.shape[0]
        self.root_structure = template_root
        
        self.hypernet_phi = CNNEncoder(in_channels=C, out_dim=self.d_theta, spatial_shape=(H, W), key=k_phi, hidden_width=32, depth=4)
        self.controlnet_psi = CNNEncoder(in_channels=C, out_dim=CONFIG["rec_feat_dim"], spatial_shape=(H, W), key=k_psi, hidden_width=16, depth=3)
        
        self.A = jnp.eye(self.d_theta)
        self.B = jnp.zeros((self.d_theta, CONFIG["rec_feat_dim"]))

    def render_pixels(self, thetas, coords):
        def render_pt(theta, coord):
            root = self.unravel_fn(theta)
            encoded_coord = fourier_encode(coord, self.num_freqs)
            out = root(encoded_coord)
            
            gray_fg = jax.nn.sigmoid(out[0:1])
            gray_bg = jax.nn.sigmoid(out[1:2])
            alpha   = jax.nn.sigmoid(out[2:3])
            return alpha * gray_fg + (1.0 - alpha) * gray_bg
        return jax.vmap(render_pt)(thetas, coords)

    def _get_thetas_and_preds_single(self, ref_video, p_forcing, key, coords_grid, inf_context_ratio):
        H, W, C = self.frame_shape
        flat_coords = coords_grid.reshape(-1, 2)
        T = ref_video.shape[0]
        
        init_gt_frame = ref_video[0]
        init_gt_frame_chw = jnp.transpose(init_gt_frame, (2, 0, 1)) 
        theta_0 = self.hypernet_phi(init_gt_frame_chw)
        
        def scan_step(state, scan_inputs):
            gt_curr_frame, step_idx = scan_inputs
            theta, prev_frame_selected, k = state
            k, subk = jax.random.split(k)
            
            thetas_frame = jnp.tile(theta, (H*W, 1))
            pred_flat = self.render_pixels(thetas_frame, flat_coords)
            pred_frame = pred_flat.reshape(H, W, C)
            
            t_ratio = step_idx / (T - 1)
            is_context = t_ratio <= inf_context_ratio
            is_forced = jax.random.bernoulli(subk, p_forcing)
            
            use_gt = jnp.logical_or(is_context, is_forced)
            frame_t = jnp.where(use_gt, gt_curr_frame, pred_frame)
            
            frame_t_feats = self.controlnet_psi(jnp.transpose(frame_t, (2, 0, 1)))
            prev_frame_selected_feats = self.controlnet_psi(jnp.transpose(prev_frame_selected, (2, 0, 1)))

            dx_feat = (frame_t_feats - prev_frame_selected_feats) / jnp.sqrt(frame_t_feats.size)
            theta_next = self.A @ theta + self.B @ dx_feat
            
            new_state = (theta_next, frame_t, subk)
            return new_state, pred_frame
            
        init_frame = jnp.zeros((H, W, C))
        init_state = (theta_0, init_frame, key)
        scan_inputs = (ref_video, jnp.arange(T))
        _, pred_video = jax.lax.scan(scan_step, init_state, scan_inputs)
        return pred_video

    def __call__(self, ref_videos, p_forcing, keys, coords_grid, inf_context_ratio):
        """
        Embeds the batched forward process natively. 
        If a single video is provided (ndim==4), it automatically inflates the batch dim.
        """
        is_single = (ref_videos.ndim == 4)
        if is_single:
            ref_videos = ref_videos[None, ...]
            keys = keys[None, ...] if keys.ndim == 1 else keys
            
        # Execute mapped function across batch dimension
        batched_fn = jax.vmap(self._get_thetas_and_preds_single, in_axes=(0, None, 0, None, None))
        preds = batched_fn(ref_videos, p_forcing, keys, coords_grid, inf_context_ratio)
        
        if is_single:
            return preds[0]
        return preds

#%% Cell 4: Initialization & Training/Loading Logic
key, subkey = jax.random.split(key)
model = WARP(CONFIG["root_width"], CONFIG["root_depth"], CONFIG["num_fourier_freqs"], (H, W, C), subkey)
A_init = model.A.copy()

print(f"Total Trainable Parameters in WARP: {count_trainable_params(model)}")

if TRAIN:
    print(f"\n🚀 Starting WARP Training -> Saving to {run_path}")
    
    # scheduler = optax.exponential_decay(CONFIG["learning_rate"], transition_steps=len(train_loader)*CONFIG["nb_epochs"], decay_rate=0.1)
    scheduler = CONFIG["learning_rate"]
    optimizer = optax.adam(scheduler)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, opt_state, keys, ref_videos, coords_grid, p_forcing):
        def loss_fn(m):
            pred_videos = m(ref_videos, p_forcing, keys, coords_grid, 0.0)
            loss_full = jnp.mean(jnp.abs(pred_videos[:, 1:] - ref_videos[:, 1:]))

            # loss_t0 = jnp.mean(jnp.abs(pred_videos[:, 0] - ref_videos[:, 0]))
            # return loss_full + 1.0 * loss_t0

            loss_full = jnp.mean(jnp.abs(pred_videos - ref_videos))
            return loss_full

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    all_losses = []
    start_time = time.time()

    for epoch in range(CONFIG["nb_epochs"]):
        print(f"\nEPOCH: {epoch+1}")
        epoch_losses = []
        
        pbar = tqdm(train_loader)
        for batch_idx, batch_videos in enumerate(pbar):
            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, CONFIG["batch_size"])
            
            model, opt_state, loss = train_step(model, opt_state, batch_keys, batch_videos, coords_grid, CONFIG["p_forcing"])
            epoch_losses.append(loss)
            
            if batch_idx % CONFIG["print_every"] == 0:
                pbar.set_description(f"Loss: {loss:.4f}")
                
        all_losses.extend(epoch_losses)
        
        # Periodically save model over the training process
        eqx.tree_serialise_leaves(artefacts_path / f"tf_model_ep{epoch+1}.eqx", model)

    wall_time = time.time() - start_time
    print("\nWall time for WARP training in h:m:s:", time.strftime("%H:%M:%S", time.gmtime(wall_time)))
    
    # Save final artifacts
    eqx.tree_serialise_leaves(artefacts_path / "tf_model_final.eqx", model)
    np.save(artefacts_path / "loss_history.npy", np.array(all_losses))

else:
    print(f"\n📥 Loading WARP model from {artefacts_path}")
    model = eqx.tree_deserialise_leaves(artefacts_path / "tf_model_final.eqx", model)
    try:
        all_losses = np.load(artefacts_path / "loss_history.npy").tolist()
    except FileNotFoundError:
        all_losses = []
        print("Warning: loss_history.npy not found.")

#%% Cell 5: Final Visualizations
print("\n=== Generating Dashboards ===")

# Run inference utilizing the newly embedded `__call__` interface (handles batched transparently)
val_keys = jax.random.split(key, CONFIG["batch_size"])
final_videos = eqx.filter_jit(model)(sample_batch, 0.0, val_keys, coords_grid, CONFIG["inf_context_ratio"])
# final_videos = eqx.filter_jit(model)(sample_batch, 1.0, val_keys, coords_grid, 1.0)

plot_pred_ref_videos(
    final_videos[0], 
    sample_batch[0], 
    title=f"Final (Ctx Ratio={CONFIG['inf_context_ratio']})", 
    save_name="inference_context_forecast.png"
)

if len(all_losses) > 0:
    plt.figure(figsize=(8, 4))
    plt.plot(all_losses)
    plt.yscale('log')
    plt.title("Moving MNIST Batched Loss History")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_path / "loss_history.png")
    plt.show()

# Plot Matrix A
A_final = model.A
# subsample_step = max(1, model.d_theta // 1) 
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