#%% Cell 1: Imports and Configuration
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from jax.flatten_util import ravel_pytree

import torch
from torchvision import datasets
from torch.utils.data import DataLoader

import seaborn as sns
sns.set(style="white", context="talk")

# --- Configuration ---
SEED = 2026
key = jax.random.PRNGKey(SEED)

NB_EPOCHS = 10
BATCH_SIZE = 12
LEARNING_RATE = 1e-4
PRINT_EVERY = 5

P_FORCING = 0.5            # Used purely as base Teacher Forcing during training
INF_CONTEXT_RATIO = 0.5    # Proportion of frames used as context during inference

REC_FEAT_DIM = 256
ROOT_WIDTH = 32
ROOT_DEPTH = 2
NUM_FOURIER_FREQS = 12

#%% Cell 2: PyTorch Data Loading & Plotting Helpers
def numpy_collate(batch):
    """ Converts PyTorch tensors from MovingMNIST to batched JAX-compatible arrays. """
    # Moving MNIST dataset natively returns (T, C, H, W)
    if isinstance(batch[0], tuple):
        videos = torch.stack([b[0] for b in batch]).numpy()
    else:
        videos = torch.stack(batch).numpy()
    
    # Ensure shape is (B, T, H, W, C) where C=1
    if videos.ndim == 4: # (B, T, H, W)
        videos = np.expand_dims(videos, axis=-1)
    elif videos.ndim == 5 and videos.shape[2] == 1: # (B, T, C, H, W)
        videos = np.transpose(videos, (0, 1, 3, 4, 2))
        
    videos = videos.astype(np.float32)
    if videos.max() > 2.0:
        videos = videos / 255.0  # Normalize to [0, 1]
        
    return videos

# --- Load Moving MNIST ---
print("Loading Moving MNIST Dataset...")
try:
    dataset = datasets.MovingMNIST(root='./data', split=None, download=True)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=numpy_collate, drop_last=True)
    
    # Get one batch to extract shapes
    sample_batch = next(iter(train_loader))
    B, nb_frames, H, W, C = sample_batch.shape
    print(f"Batched Video shape: {sample_batch.shape}")
    
except Exception as e:
    print(f"Could not load MovingMNIST: {e}. Please ensure network access.")
    raise e

# Precompute Normalized Spatial Coordinates
y_coords = jnp.linspace(-1, 1, H)
x_coords = jnp.linspace(-1, 1, W)
X_grid, Y_grid = jnp.meshgrid(x_coords, y_coords)
coords_grid = jnp.stack([X_grid, Y_grid], axis=-1)  # [H, W, 2]

# --- Plotting ---
def sbimshow(img, title="", ax=None):
    img = np.clip(img, 0.0, 1.0)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1) # Broadcast Grayscale to RGB for plotting
        
    if ax is None:
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    else:
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

def plot_pred_ref_videos(video, ref_video, title="Render"):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    t_mid = len(video) // 2
    t_end = len(video) - 1
    
    sbimshow(video[0], title=f"{title} t=0", ax=ax[0, 0])
    sbimshow(video[t_mid], title=f"{title} t={t_mid}", ax=ax[0, 1])
    sbimshow(video[t_end], title=f"{title} t={t_end} (Forecast)", ax=ax[0, 2])

    sbimshow(ref_video[0], title="Ref t=0", ax=ax[1, 0])
    sbimshow(ref_video[t_mid], title=f"Ref t={t_mid}", ax=ax[1, 1])
    sbimshow(ref_video[t_end], title=f"Ref t={t_end}", ax=ax[1, 2])
    plt.tight_layout()
    plt.show()

def plot_pred_video(video, title="Render"):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    t_mid = len(video) // 2
    t_end = len(video) - 1
    
    sbimshow(video[0], title=f"{title} t=0", ax=ax[0])
    sbimshow(video[t_mid], title=f"{title} t={t_mid}", ax=ax[1])
    sbimshow(video[t_end], title=f"{title} t={t_end} (Forecast)", ax=ax[2])
    plt.tight_layout()
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
    psi_net: CNNEncoder
    
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
        # Output is 3 for Grayscale (Foreground, Background, Alpha)
        template_root = RootMLP(coord_dim, 3, root_width, root_depth, k_root)

        flat_params, self.unravel_fn = ravel_pytree(template_root)
        self.d_theta = flat_params.shape[0]
        self.root_structure = template_root
        
        self.hypernet_phi = CNNEncoder(in_channels=C, out_dim=self.d_theta, spatial_shape=(H, W), key=k_phi, hidden_width=32, depth=4)
        self.psi_net = CNNEncoder(in_channels=C, out_dim=REC_FEAT_DIM, spatial_shape=(H, W), key=k_psi, hidden_width=16, depth=3)
        
        self.A = jnp.eye(self.d_theta)
        self.B = jnp.zeros((self.d_theta, REC_FEAT_DIM))

        print(f"Model Initialized:")
        print(f"  d_theta (State Size): {self.d_theta}")
        print(f"  Matrix A Shape: {self.A.shape}")
        print(f"  Matrix B Shape: {self.B.shape}")

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

    def get_thetas_and_preds(self, ref_video, p_forcing, key, coords_grid, inf_context_ratio):
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
            
            # 1. Render current frame
            thetas_frame = jnp.tile(theta, (H*W, 1))
            pred_flat = self.render_pixels(thetas_frame, flat_coords)
            pred_frame = pred_flat.reshape(H, W, C)
            
            # 2. Context Window & Teacher Forcing Logic
            t_ratio = step_idx / (T - 1)
            is_context = t_ratio <= inf_context_ratio
            is_forced = jax.random.bernoulli(subk, p_forcing)
            
            use_gt = jnp.logical_or(is_context, is_forced)
            frame_t = jnp.where(use_gt, gt_curr_frame, pred_frame)
            
            # 3. Compute dense features via CNN \psi
            frame_t_feats = self.psi_net(jnp.transpose(frame_t, (2, 0, 1)))
            prev_frame_selected_feats = self.psi_net(jnp.transpose(prev_frame_selected, (2, 0, 1)))

            # Variance scaling for stable gradients in the recurrence
            dx_feat = (frame_t_feats - prev_frame_selected_feats) / jnp.sqrt(frame_t_feats.size)

            # 4. Weight-space Recurrence Update
            theta_next = self.A @ theta + self.B @ dx_feat
            
            new_state = (theta_next, frame_t, subk)
            return new_state, pred_frame
            
        init_frame = jnp.zeros((H, W, C))
        init_state = (theta_0, init_frame, key)
        
        scan_inputs = (ref_video, jnp.arange(T))
        _, pred_video = jax.lax.scan(scan_step, init_state, scan_inputs)
        return pred_video

#%% Cell 4: Batched Initialization & Training
key, subkey = jax.random.split(key)
model = WARP(ROOT_WIDTH, ROOT_DEPTH, NUM_FOURIER_FREQS, (H, W, C), subkey)
A_init = model.A.copy()

# Vectorize the model forward pass across the batch dimension (Axis 0)
@eqx.filter_vmap(in_axes=(None, 0, None, 0, None, None))
def batched_forward(model, ref_videos, p_forcing, keys, coords_grid, inf_context_ratio):
    return model.get_thetas_and_preds(ref_videos, p_forcing, keys, coords_grid, inf_context_ratio)

scheduler = optax.exponential_decay(LEARNING_RATE, transition_steps=len(train_loader)*NB_EPOCHS, decay_rate=0.1)
optimizer = optax.adam(scheduler)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

@eqx.filter_jit
def train_step(model, opt_state, keys, ref_videos, coords_grid, p_forcing):
    def loss_fn(m):
        # During training, INF_CONTEXT_RATIO is strictly 0.0
        pred_videos = batched_forward(m, ref_videos, p_forcing, keys, coords_grid, 0.0)
        
        loss_full = jnp.mean(jnp.abs(pred_videos[:, 1:] - ref_videos[:, 1:]))
        loss_t0 = jnp.mean(jnp.abs(pred_videos[:, 0] - ref_videos[:, 0]))
        
        return loss_full + 1.0 * loss_t0

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

all_losses = []
start_time = time.time()

for epoch in range(NB_EPOCHS):
    print(f"\nEPOCH: {epoch+1}")
    epoch_losses = []
    
    pbar = tqdm(train_loader)
    for batch_idx, batch_videos in enumerate(pbar):
        key, subkey = jax.random.split(key)
        batch_keys = jax.random.split(subkey, BATCH_SIZE)
        
        # Train Step
        model, opt_state, loss = train_step(model, opt_state, batch_keys, batch_videos, coords_grid, P_FORCING)
        epoch_losses.append(loss)
        
        if batch_idx % PRINT_EVERY == 0:
            pbar.set_description(f"Loss: {loss:.4f}")
            
    all_losses.extend(epoch_losses)
    
    # Render inference progress at end of epoch using the last batch and 50% Context Ratio
    val_keys = jax.random.split(key, BATCH_SIZE)
    current_videos = eqx.filter_jit(batched_forward)(model, batch_videos, 0.0, val_keys, coords_grid, INF_CONTEXT_RATIO)
    plot_pred_video(current_videos[0], title=f"Epoch {epoch+1} Render")

wall_time = time.time() - start_time
print("\nWall time for WARP training in h:m:s:", time.strftime("%H:%M:%S", time.gmtime(wall_time)))

#%% Cell 5: Final Visualizations
# Generate final validation video using 50% context ratio and zero forcing
val_keys = jax.random.split(key, BATCH_SIZE)
final_videos = eqx.filter_jit(batched_forward)(model, sample_batch, 0.0, val_keys, coords_grid, INF_CONTEXT_RATIO)

plot_pred_ref_videos(final_videos[0], sample_batch[0], title=f"Final (Ctx Ratio={INF_CONTEXT_RATIO})")

# Plot Loss
plt.figure(figsize=(8, 4))
plt.plot(all_losses)
plt.yscale('log')
plt.title("Moving MNIST Batched Loss History")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Plot Matrix A
A_final = model.A
subsample_step = max(1, model.d_theta // 10) 
vmin, vmax = -1e-4, 1e-4

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im1 = axes[0].imshow(A_init[::subsample_step, ::subsample_step], cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title(f"Recurrence Matrix A (Init)\nSubsampled step={subsample_step}")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(A_final[::subsample_step, ::subsample_step], cmap='viridis', vmin=vmin, vmax=vmax)
axes[1].set_title(f"Recurrence Matrix A (Final)\nSubsampled step={subsample_step}")
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

def count_trainable_params(model):
    def count_params(x):
        if isinstance(x, jnp.ndarray) and x.dtype in [jnp.float32, jnp.float64]:
            return x.size
        else:
            return 0

    param_counts = jax.tree_util.tree_map(count_params, model)
    total_params = sum(jax.tree_util.tree_leaves(param_counts))
    return total_params

print(f"Total Trainable Parameters in WARP: {count_trainable_params(model)}")