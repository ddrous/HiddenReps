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
from torch.utils.data import DataLoader, Subset

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

SINGLE_BATCH = False

CONFIG = {
    "seed": 2026,
    "nb_epochs": 100,
    "batch_size": 2 if not SINGLE_BATCH else 8,
    "learning_rate": 1e-5, 
    "print_every": 100,
    "p_forcing": 0.25,
    "inf_context_ratio": 0.5,
    "nb_loss_steps_full": 12,
    
    # --- LSTM Dynamics Config ---
    "lstm_input_dim": 128,   # Size of the encoded frame vector
    "lstm_hidden_dim": 128,  # Size of the Custom LSTM hidden state
    
    "root_width": 32,
    "root_depth": 2,
    "num_fourier_freqs": 6,

    # --- Plateau Scheduler Config ---
    "lr_patience": 500,      
    "lr_cooldown": 0,       
    "lr_factor": 0.5,        
    "lr_rtol": 1e-3,         
    "lr_accum_size": 5,     
    "lr_min_scale": 1e-5     
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
    # RAW PIXELS: Retaining range [0, 255] for categorical classification
    return videos

print("Loading Moving MNIST Dataset...")
try:
    data_path = './data' if TRAIN else '../../data'
    dataset = datasets.MovingMNIST(root=data_path, split=None, download=True)
    if SINGLE_BATCH:
        testing_subset = Subset(dataset, range(CONFIG["batch_size"]))
        train_loader = DataLoader(
            testing_subset, 
            batch_size=CONFIG["batch_size"], 
            shuffle=False, 
            collate_fn=numpy_collate, 
            drop_last=True
        )
    else:
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
    # Normalize to [0, 1] purely for Matplotlib rendering
    img = np.clip(img / 255.0, 0.0, 1.0)
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
    fig, axes = plt.subplots(2, 1+(nb_frames//2), figsize=(20, 6))
    indices_to_plot = list(np.arange(0, nb_frames, 2)) + [nb_frames-1]
    for i, idx in enumerate(indices_to_plot):
        sbimshow(video[idx], title=f"{title} t={idx}", ax=axes[0, i])
        sbimshow(ref_video[idx], title=f"Ref t={idx}", ax=axes[1, i])

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

# --- NEW: Custom Latent Dynamics Implementation ---
class CustomLSTMCell(eqx.Module):
    weight_ih: eqx.nn.Linear
    weight_hh: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, key):
        k1, k2 = jax.random.split(key)
        self.weight_ih = eqx.nn.Linear(input_size, 4 * hidden_size, key=k1)
        self.weight_hh = eqx.nn.Linear(hidden_size, 4 * hidden_size, use_bias=False, key=k2)

    def __call__(self, x, state):
        h, c = state
        gates = self.weight_ih(x) + self.weight_hh(h)
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f + 1.0)  # Forget gate bias initialized to 1 for long-range stability
        g = jax.nn.tanh(g)
        o = jax.nn.sigmoid(o)
        
        c_next = f * c + i * g
        h_next = o * jax.nn.tanh(c_next)
        
        return h_next, c_next

class ARModel(eqx.Module):
    """Wrapper that can easily swap between LSTM, GRU, or other Recurrent logic."""
    cell: eqx.Module
    cell_type: str = eqx.field(static=True)

    def __init__(self, input_dim, hidden_dim, cell_type="lstm", *, key):
        self.cell_type = cell_type.lower()
        if self.cell_type == "lstm":
            self.cell = CustomLSTMCell(input_dim, hidden_dim, key=key)
        elif self.cell_type == "gru":
            raise NotImplementedError("Custom GRU Cell can be added here seamlessly.")
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

    def __call__(self, x, state):
        return self.cell(x, state)

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
    """Encodes the 2D frame down to a flat vector for the AR Model."""
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
    # 1. Base Weights
    theta_base: jax.Array
    
    # 2. Dynamics Model Components
    frame_encoder: CNNEncoder
    ar_model: ARModel
    decoder: eqx.nn.MLP
    
    # 3. Initial States
    h_init: jax.Array
    c_init: jax.Array

    # Constants & Structure
    root_structure: RootMLP = eqx.field(static=True)
    unravel_fn: callable = eqx.field(static=True)
    d_theta: int = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)

    def __init__(self, root_width, root_depth, num_freqs, frame_shape, lstm_in_dim, lstm_hid_dim, key):
        k_root, k_enc, k_lstm, k_dec = jax.random.split(key, 4)
        self.num_freqs = num_freqs
        self.frame_shape = frame_shape
        H, W, C = frame_shape
        
        coord_dim = 2 + 2 * 2 * num_freqs 
        
        # Root outputs 256 dimensions for CE class logits
        template_root = RootMLP(coord_dim, 256, root_width, root_depth, k_root)

        flat_params, self.unravel_fn = ravel_pytree(template_root)
        self.d_theta = flat_params.shape[0]
        self.root_structure = template_root
        
        # Learnable parameters
        self.theta_base = flat_params
        self.h_init = jnp.zeros(lstm_hid_dim)
        self.c_init = jnp.zeros(lstm_hid_dim)

        # Architectural Modules
        self.frame_encoder = CNNEncoder(
            in_channels=C, out_dim=lstm_in_dim, spatial_shape=(H, W), key=k_enc, hidden_width=16, depth=3
        )
        
        self.ar_model = ARModel(
            input_dim=lstm_in_dim, hidden_dim=lstm_hid_dim, cell_type="lstm", key=k_lstm
        )
        
        self.decoder = eqx.nn.MLP(
            in_size=lstm_hid_dim, out_size=self.d_theta * 2, width_size=256, depth=2, key=k_dec
        )

    def render_pixels(self, thetas, coords):
        def render_pt(theta, coord):
            root = self.unravel_fn(theta)
            encoded_coord = fourier_encode(coord, self.num_freqs)
            out = root(encoded_coord)
            return out # Returns raw logits
        return jax.vmap(render_pt)(thetas, coords)

    def _get_thetas_and_preds_single(self, ref_video, p_forcing, key, coords_grid, inf_context_ratio):
        H, W, C = self.frame_shape
        flat_coords = coords_grid.reshape(-1, 2)
        T = ref_video.shape[0]

        def scan_step(state, scan_inputs):
            h, c, prev_frame_selected, k = state
            gt_curr_frame, step_idx = scan_inputs
            k, subk = jax.random.split(k)

            # 1. Encode previous frame into AR input space
            prev_frame_chw = jnp.transpose(prev_frame_selected, (2, 0, 1))
            ar_input = self.frame_encoder(prev_frame_chw)

            # 2. Update Autoregressive Dynamics
            h_next, c_next = self.ar_model(ar_input, (h, c))

            # 3. Decode state to Weight Space Modulation
            theta_decoded = self.decoder(h_next)
            theta_mu, theta_scale = jnp.split(theta_decoded, 2, axis=-1)
            
            # FiLM the learned base parameters
            theta_scaled = self.theta_base * (1 + theta_scale) + theta_mu

            # 4. Render Categorical Logits
            thetas_frame = jnp.tile(theta_scaled, (H*W, 1))
            logits_flat = self.render_pixels(thetas_frame, flat_coords)
            logits_frame = logits_flat.reshape(H, W, 256)
            
            # Extract expected pixel value [0, 255] for autoregressive feedback & plotting
            probs_flat = jax.nn.softmax(logits_flat, axis=-1)
            pred_pixels_flat = jnp.sum(probs_flat * jnp.arange(256), axis=-1, keepdims=True)
            pred_frame = pred_pixels_flat.reshape(H, W, C)
            
            # 5. Context/Teacher Forcing Decision
            t_ratio = step_idx / (T - 1)
            is_context = t_ratio <= inf_context_ratio
            is_forced = jax.random.bernoulli(subk, p_forcing)
            
            use_gt = jnp.logical_or(is_context, is_forced)
            frame_for_next_step = jnp.where(use_gt, gt_curr_frame, pred_frame)
            
            new_state = (h_next, c_next, frame_for_next_step, subk)
            return new_state, (logits_frame, pred_frame)

        # Initial conditions: Feed a zeroed-out frame to the dynamics model to predict step 0. 
        init_frame = jnp.zeros_like(ref_video[0])
        init_state = (self.h_init, self.c_init, init_frame, key)
        
        scan_inputs = (ref_video, jnp.arange(T))
        _, (pred_logits, pred_pixels) = jax.lax.scan(scan_step, init_state, scan_inputs)

        return pred_logits, pred_pixels

    def __call__(self, ref_videos, p_forcing, keys, coords_grid, inf_context_ratio):
        is_single = (ref_videos.ndim == 4)
        if is_single:
            ref_videos = ref_videos[None, ...]
            keys = keys[None, ...] if keys.ndim == 1 else keys
            
        batched_fn = jax.vmap(self._get_thetas_and_preds_single, in_axes=(0, None, 0, None, None))
        pred_logits, pred_pixels = batched_fn(ref_videos, p_forcing, keys, coords_grid, inf_context_ratio)
        
        if is_single:
            return pred_logits[0], pred_pixels[0]
        return pred_logits, pred_pixels

@eqx.filter_jit
def evaluate(m, batch, p_forcing, keys, coords, context_ratio):
    return m(batch, p_forcing, keys, coords, context_ratio)

#%% Cell 4: Initialization & Training
key, subkey = jax.random.split(key)
model = WARP(
    CONFIG["root_width"], 
    CONFIG["root_depth"], 
    CONFIG["num_fourier_freqs"], 
    (H, W, C), 
    CONFIG["lstm_input_dim"],
    CONFIG["lstm_hidden_dim"],
    subkey
)

print(f"Total Trainable Parameters in WARP: {count_trainable_params(model)}")

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
            k_full = keys[0]
            
            # Forward Pass returns (Logits, Pixels)
            pred_logits, _ = m(ref_videos, p_forcing, keys, coords_grid, 0.0)

            # Sample sequence steps for CE Loss
            full_indices = jax.random.choice(k_full, ref_videos.shape[1], shape=(CONFIG["nb_loss_steps_full"],), replace=False)
            
            pred_logits_selected = pred_logits[:, full_indices]
            ref_selected = ref_videos[:, full_indices]
            
            # Format targets to integers and squeeze out the single channel dim
            labels_seq = ref_selected.astype(jnp.int32).squeeze(-1) 
            
            loss_full = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                logits=pred_logits_selected, 
                labels=labels_seq
            ))

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
        if not SINGLE_BATCH:
            print(f"\nEPOCH: {epoch+1}")
        
        epoch_losses = []
        # for batch_idx, batch_videos in enumerate(train_loader):
        pbar = tqdm(train_loader)
        for batch_idx, batch_videos in enumerate(pbar):
            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, CONFIG["batch_size"])
            
            model, opt_state, loss = train_step(model, opt_state, batch_keys, batch_videos, coords_grid, CONFIG["p_forcing"])
            epoch_losses.append(loss)
            
            current_scale = optax.tree_utils.tree_get(opt_state, "scale")
            lr_scales.append(current_scale)

            # if not SINGLE_BATCH and (batch_idx % CONFIG["print_every"]) == 0:
            #     print(f"Batch {batch_idx} | Loss: {loss:.4f} | LR Scale: {current_scale:.4f}")

            if not SINGLE_BATCH and (batch_idx % CONFIG["print_every"]) == 0:
                pbar.set_description(f"Loss: {loss:.4f} | LR Scale: {current_scale:.4f}")

        all_losses.extend(epoch_losses)
        
        # Save checkpoints
        if epoch in [4, CONFIG["nb_epochs"]//2, 2*CONFIG["nb_epochs"]//3]:
            eqx.tree_serialise_leaves(artefacts_path / f"tf_model_ep{epoch+1}.eqx", model)

        # Generate eval frames periodically
        if (epoch+1) % (CONFIG["nb_epochs"]//10)==0 or epoch == 0:
            val_keys = jax.random.split(key, CONFIG["batch_size"])
            _, val_pixels = evaluate(model, sample_batch, 0.0, val_keys, coords_grid, CONFIG["inf_context_ratio"])
            plot_pred_ref_videos_rollout(val_pixels[0], 
                                        sample_batch[0], 
                                        title=f"Pred", 
                                        save_name=f"pred_ref_epoch{epoch+1}.png")

    wall_time = time.time() - start_time
    print("\nWall time for WARP training in h:m:s:", time.strftime("%H:%M:%S", time.gmtime(wall_time)))
    
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
    
    color1 = 'teal'
    ax1.plot(all_losses, color=color1, alpha=0.8, label="Cross-Entropy Loss")
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

#%%
sample_batch = next(iter(train_loader))

val_keys = jax.random.split(key, CONFIG["batch_size"])
_, final_pixels = evaluate(model, sample_batch, 0.0, val_keys, coords_grid, 0.5)

plot_pred_ref_videos_rollout(
    final_pixels[0], 
    sample_batch[0], 
    title=f"Pred", 
    save_name="inference_forecast_rollout.png"
)
