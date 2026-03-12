#%%
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import time
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import seaborn as sns
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

TRAIN = True

# --- Configuration ---
CONFIG = {
    "seed": 2026,
    "nb_epochs": 50,       # Adjust based on convergence
    "batch_size": 256,     # Frames per batch (larger since no temporal axis)
    "learning_rate": 1e-4,
    "root_width": 12,
    "root_depth": 5,
    "num_fourier_freqs": 6,
    "hidden_width": 32,    # CNN hidden width
    "cnn_depth": 4,        # CNN depth
}

key = jax.random.PRNGKey(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

#%%
# --- PyTorch Data Loading (Frames instead of Videos) ---
def numpy_collate(batch):
    frames = torch.stack(batch).numpy()
    if frames.ndim == 3:
        frames = np.expand_dims(frames, axis=-1)
    frames = frames.astype(np.float32)
    if frames.max() > 2.0:
        frames = frames / 255.0
    return frames

print("Loading Moving MNIST/MiniGrid Dataset for Pre-training...")
data_path = './data'
try:
    minigrid_arrays = np.load(data_path + "/MiniGrid/minigrid.npy")
    train_size = int(0.8 * minigrid_arrays.shape[0])
    train_arrays = minigrid_arrays[:train_size]
    test_arrays = minigrid_arrays[train_size:]
    
    # Flatten the dataset to consider every frame independently: (N*T, H, W, C)
    all_train_frames = train_arrays.reshape(-1, *train_arrays.shape[2:])
    
    class FrameDataset(torch.utils.data.Dataset):
        def __init__(self, data_array):
            self.data_array = data_array
        def __len__(self):
            return self.data_array.shape[0]
        def __getitem__(self, idx):
            frame = self.data_array[idx]
            return torch.from_numpy(frame.astype(np.float32))

    dataset = FrameDataset(all_train_frames)
    train_loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=numpy_collate)

    sample_batch = next(iter(train_loader))
    B, H, W, C = sample_batch.shape
    print(f"Batched Frames shape: {sample_batch.shape}")
except Exception as e:
    print(f"Could not load MiniGrid: {e}")
    raise e

y_coords = jnp.linspace(-1, 1, H)
x_coords = jnp.linspace(-1, 1, W)
X_grid, Y_grid = jnp.meshgrid(x_coords, y_coords)
coords_grid = jnp.stack([X_grid, Y_grid], axis=-1) 

# --- Model Definition (Autoencoder Wrapper) ---
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
    def __init__(self, in_channels, out_dim, spatial_shape, key, hidden_width=8, depth=4):
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

class Autoencoder(eqx.Module):
    encoder: CNNEncoder
    theta_base: jax.Array
    unravel_fn: callable = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)

    def __init__(self, root_width, root_depth, num_freqs, frame_shape, key):
        k_root, k_enc = jax.random.split(key, 2)
        self.frame_shape = frame_shape
        self.num_freqs = num_freqs
        H, W, C = frame_shape

        coord_dim = 2 + 2 * 2 * num_freqs
        template_root = RootMLP(coord_dim, C, root_width, root_depth, k_root)
        
        flat_params, self.unravel_fn = ravel_pytree(template_root)
        d_theta = flat_params.shape[0]
        self.theta_base = flat_params

        self.encoder = CNNEncoder(in_channels=C, out_dim=d_theta, spatial_shape=(H, W), key=k_enc, 
                                  hidden_width=CONFIG["hidden_width"], depth=CONFIG["cnn_depth"])

    def render_frame(self, theta_offset, coords_grid):
        H, W, C = self.frame_shape
        flat_coords = coords_grid.reshape(-1, 3)
        theta = theta_offset + self.theta_base

        def render_pt(th, coord):
            root = self.unravel_fn(th)
            encoded_coord = fourier_encode(coord[1:], self.num_freqs)
            return root(encoded_coord)
            
        pred_flat = jax.vmap(render_pt, in_axes=(None, 0))(theta, flat_coords)
        return pred_flat.reshape(H, W, -1)

# --- Initialization & Training ---
key, subkey = jax.random.split(key)
model = Autoencoder(CONFIG["root_width"], CONFIG["root_depth"], CONFIG["num_fourier_freqs"], 
                    (H, W, C), key=subkey)

optimizer = optax.adam(CONFIG["learning_rate"])
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

@eqx.filter_jit
def train_step(model, opt_state, batch_frames, coords_grid):
    def loss_fn(m):
        # 1. Forward pass encoder (B, H, W, C) -> (B, C, H, W)
        frames_enc = jnp.transpose(batch_frames, (0, 3, 1, 2))
        thetas = jax.vmap(m.encoder)(frames_enc)

        # 2. Add dummy time dimension (t=0) to coords
        coords_grid_t0 = jnp.concatenate([jnp.zeros_like(coords_grid[..., :1]), coords_grid], axis=-1)

        # 3. Reconstruct via RootMLP
        batched_render = jax.vmap(lambda theta: m.render_frame(theta, coords_grid_t0))
        reconstructed = batched_render(thetas)

        # # 4. Compute MSE
        # return jnp.mean((reconstructed - batch_frames)**2)

        def ssim(x, y, data_range=1.0):
            C1 = (0.01 * data_range) ** 2
            C2 = (0.03 * data_range) ** 2

            mu_x = jnp.mean(x)
            mu_y = jnp.mean(y)
            sigma_x = jnp.var(x)
            sigma_y = jnp.var(y)
            sigma_xy = jnp.mean((x - mu_x) * (y - mu_y))

            ssim_numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
            ssim_denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

            return ssim_numerator / ssim_denominator

        # 5. Compare SSIM instead
        return 1.0 - jnp.mean(jax.vmap(ssim)(reconstructed, batch_frames))

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


if TRAIN:
    print(f"\n🚀 Starting Separate Autoencoder Training")
    start_time = time.time()

    for epoch in range(CONFIG["nb_epochs"]):
        epoch_losses = []
        for batch_frames in train_loader:
            model, opt_state, loss = train_step(model, opt_state, batch_frames, coords_grid)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{CONFIG['nb_epochs']} - Avg MSE Loss: {avg_loss:.6f}")

    print("\nWall time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # --- Serialization ---
    # We ONLY save the `.encoder` component exactly.
    eqx.tree_serialise_leaves("minigrid_enc.eqx", (model.encoder, model.theta_base))
    # eqx.tree_serialise_leaves("pretrained_thetabase.eqx", model.theta_base)
    print("✅ Saved isolated CNNEncoder to 'pretrained_encoder.eqx'")

else:
    print("⏭️  Skipping training, loading pretrained encoder...")
    try:
        loaded_encoder, loaded_theta_base = eqx.tree_deserialise_leaves("minigrid_enc.eqx", (model.encoder, model.theta_base))
        model = eqx.tree_at(lambda m: m.encoder, model, loaded_encoder)
        model = eqx.tree_at(lambda m: m.theta_base, model, loaded_theta_base)
        print("✅ Successfully loaded pretrained encoder and theta base!")
    except Exception as e:
        print(f"❌ Failed to load pretrained encoder: {e}")
        raise e


#%% Plot loss function
if TRAIN:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epoch_losses, label='MSE Loss')
    ax.set_xlabel('Train Step')
    ax.set_ylabel('Loss')
    ax.set_title('Autoencoder Training Loss')
    ax.legend()
    plt.show()

#%% Visualize some reconstructions
predicted_frames = []
test_loader = DataLoader(FrameDataset(test_arrays.reshape(-1, *test_arrays.shape[2:])), batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=numpy_collate)

batch_frames = next(iter(test_loader))
frames_enc = jnp.transpose(batch_frames, (0, 3, 1, 2))
thetas = jax.vmap(model.encoder)(frames_enc)
coords_grid_t0 = jnp.concatenate([jnp.zeros_like(coords_grid[..., :1]), coords_grid], axis=-1)
batched_render = jax.vmap(lambda theta: model.render_frame(theta, coords_grid_t0))
reconstructed = batched_render(thetas)
predicted_frames.append(reconstructed[:5])  # Take first 5 for visualization

predicted_frames = jnp.concatenate(predicted_frames, axis=0)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
ids_to_show = list(np.random.choice(predicted_frames.shape[0], 5, replace=False))
for i in ids_to_show:
    print("Predicted frame range:", predicted_frames[i].min(), predicted_frames[i].max())
    axes[1, i].imshow(batch_frames[i])
    axes[1, i].set_title(f"Ref")
    axes[1, i].axis('off')
    
    axes[0, i].imshow(predicted_frames[i])
    axes[0, i].set_title(f"Pred idx={i}")
    axes[0, i].axis('off')
plt.suptitle("Autoencoder Reconstructions on MiniGrid Test Set")
plt.tight_layout()
plt.show()

# %%
