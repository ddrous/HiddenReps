#%% Cell 1: Imports and Configuration
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
    "nb_epochs": 100,       # Adjust based on convergence
    "batch_size": 256,     # Frames per batch
    "learning_rate": 1e-4,
    "root_width": 12,
    "root_depth": 5,
    "num_fourier_freqs": 6,
    "hidden_width": 64,
    "cnn_depth": 4,
}

key = jax.random.PRNGKey(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

#%% Cell 2: Data Loading (Frames instead of Videos)
def numpy_collate(batch):
    frames = torch.stack(batch).numpy()
    frames = frames.astype(np.float32)
    if frames.max() > 2.0:
        frames = frames / 255.0
    return frames

print("Loading Moving MNIST Dataset for Pre-training...")
data_path = './data'
try:
    mov_mnist_arrays = np.load(data_path + "/MovingMNIST/mnist_test_seq.npy")
    print(f"Original loaded MovingMNIST shape: {mov_mnist_arrays.shape} (T, N, H, W)")
    
    # Strictly split: 8000 train, 2000 test
    train_arrays = mov_mnist_arrays[:, :8000]
    test_arrays = mov_mnist_arrays[:, 8000:]
    
    # Flatten temporal and batch axes to treat every frame independently: (T*N, H, W)
    all_train_frames = train_arrays.reshape(-1, *train_arrays.shape[2:])
    all_train_frames = np.expand_dims(all_train_frames, axis=-1)  # Add channel dimension -> (T*N, H, W, 1)
    
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
    print(f"Could not load Moving MNIST: {e}")
    raise e

y_coords = jnp.linspace(-1, 1, H)
x_coords = jnp.linspace(-1, 1, W)
X_grid, Y_grid = jnp.meshgrid(x_coords, y_coords)
coords_grid = jnp.stack([X_grid, Y_grid], axis=-1) 

#%% Cell 3: Model Definition (Autoencoder Wrapper)
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

#%% Cell 4: Initialization & Training Loop
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

        # 4. Compute MSE Loss (Preferred over SSIM for sparse MNIST digits)
        return jnp.mean((reconstructed - batch_frames)**2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

if TRAIN:
    print(f"\n🚀 Starting Separate Autoencoder Training for Moving MNIST")
    start_time = time.time()

    epoch_losses_history = []
    for epoch in range(CONFIG["nb_epochs"]):
        epoch_losses = []
        for batch_frames in train_loader:
            model, opt_state, loss = train_step(model, opt_state, batch_frames, coords_grid)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        epoch_losses_history.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{CONFIG['nb_epochs']} - Avg MSE Loss: {avg_loss:.6f}")

    print("\nWall time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # --- Serialization ---
    # Save precisely the `.encoder` component and the `theta_base` arrays for Moving MNIST
    eqx.tree_serialise_leaves("movingmnist_enc.eqx", (model.encoder, model.theta_base))
    print("✅ Saved isolated CNNEncoder to 'movingmnist_enc.eqx'")

else:
    print("⏭️  Skipping training, loading pretrained encoder...")
    try:
        loaded_encoder, loaded_theta_base = eqx.tree_deserialise_leaves("movingmnist_enc.eqx", (model.encoder, model.theta_base))
        model = eqx.tree_at(lambda m: m.encoder, model, loaded_encoder)
        model = eqx.tree_at(lambda m: m.theta_base, model, loaded_theta_base)
        print("✅ Successfully loaded pretrained encoder and theta base!")
    except Exception as e:
        print(f"❌ Failed to load pretrained encoder: {e}")
        raise e

#%% Cell 5: Plotting and Evaluation
if TRAIN:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epoch_losses_history, label='MSE Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title('Autoencoder Training Loss (Moving MNIST)')
    ax.legend()
    plt.show()

# Visualize reconstructions on unseen Test data
print("Generating evaluation reconstructions from the Test Split...")
test_frames_flat = test_arrays.reshape(-1, *test_arrays.shape[2:])
test_frames_flat = np.expand_dims(test_frames_flat, axis=-1)

test_loader = DataLoader(FrameDataset(test_frames_flat), batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=numpy_collate)
batch_frames = next(iter(test_loader))

frames_enc = jnp.transpose(batch_frames, (0, 3, 1, 2))
thetas = jax.vmap(model.encoder)(frames_enc)
coords_grid_t0 = jnp.concatenate([jnp.zeros_like(coords_grid[..., :1]), coords_grid], axis=-1)
batched_render = jax.vmap(lambda theta: model.render_frame(theta, coords_grid_t0))
reconstructed = batched_render(thetas)

# Helper function to convert 1-channel to 3-channel for reliable matplotlib rendering
def prepare_img(img):
    img = np.clip(img, 0.0, 1.0)
    return np.repeat(img, 3, axis=-1)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
ids_to_show = list(np.random.choice(reconstructed.shape[0], 5, replace=False))

for col_idx, batch_idx in enumerate(ids_to_show):
    # Reference Ground Truth
    axes[1, col_idx].imshow(prepare_img(batch_frames[batch_idx]))
    axes[1, col_idx].set_title(f"Ref")
    axes[1, col_idx].axis('off')
    
    # Predicted Reconstruction
    axes[0, col_idx].imshow(prepare_img(reconstructed[batch_idx]))
    axes[0, col_idx].set_title(f"Pred idx={batch_idx}")
    axes[0, col_idx].axis('off')

plt.suptitle("Autoencoder Reconstructions on Moving MNIST Test Set")
plt.tight_layout()
plt.show()
