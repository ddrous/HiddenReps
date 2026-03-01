#%% Cell 1: Imports, Utilities, and Configuration
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import diffrax
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
    "nb_epochs": 1000,
    "print_every": 100,
    "batch_size": 1 if SINGLE_BATCH else 32,
    "learning_rate": 1e-7 if USE_NLL_LOSS else 1e-5,
    "inf_context_ratio": 0.5,
    "nb_loss_steps_full": 20,
    
    # --- Architecture Params ---
    "rec_feat_dim": 32,
    "root_width": 12,
    "root_depth": 5,
    "num_fourier_freqs": 6,
    "use_nll_loss": USE_NLL_LOSS,
    
    # --- NCF & ODE Config ---
    "weight_space_ode": False, # Toggle between Weight-Space ODE and Abstract Latent ODE
    "ode_state_dim": 128,      # Used only if weight_space_ode is False
    "context_dim": 64,
    "taylor_order": 1,         # 0: standard, 1: first-order, 2: second-order NCF
    "vf_hidden_dim": 128,      # Vector field MLP width
    "dataset_size": 10000,     # Moving MNIST Train size
    
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

class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to return (data, index) for mapping contexts."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx], idx

def numpy_collate(batch):
    # Separate the videos and the indices
    videos = torch.stack([b[0] for b in batch]).numpy()
    indices = np.array([b[1] for b in batch])
    
    if videos.ndim == 4:
        videos = np.expand_dims(videos, axis=-1)
    elif videos.ndim == 5 and videos.shape[2] == 1:
        videos = np.transpose(videos, (0, 1, 3, 4, 2))
        
    videos = videos.astype(np.float32)
    if videos.max() > 2.0:
        videos = videos / 255.0

    return videos, indices

print("Loading Moving MNIST Dataset...")
try:
    data_path = './data' if TRAIN else '../../data'
    base_dataset = datasets.MovingMNIST(root=data_path, split=None, download=True)
    dataset = IndexedDataset(base_dataset)
    
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

    sample_batch, sample_indices = next(iter(train_loader))
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

class CNNDecoder(eqx.Module):
    proj: eqx.nn.Linear
    layers: list
    base_shape: tuple
    
    def __init__(self, in_dim, out_channels, spatial_shape, key, hidden_width=16, depth=3):
        H, W = spatial_shape
        # Calculate the feature map shape before the first transpose convolution
        self.base_shape = (hidden_width * (2 ** (depth - 1)), H // (2 ** depth), W // (2 ** depth))
        
        keys = jax.random.split(key, depth + 2)
        flat_base_dim = self.base_shape[0] * self.base_shape[1] * self.base_shape[2]
        
        self.proj = eqx.nn.Linear(in_dim, flat_base_dim, key=keys[0])
        
        conv_layers = []
        current_in = self.base_shape[0]
        
        for i in range(depth):
            current_out = current_in // 2 if i < depth - 1 else hidden_width
            conv_layers.append(
                eqx.nn.ConvTranspose2d(current_in, current_out, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[i+1])
            )
            current_in = current_out
            
        conv_layers.append(
            eqx.nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1, key=keys[-1])
        )
        self.layers = conv_layers

    def __call__(self, x):
        x = self.proj(x)
        x = x.reshape(self.base_shape)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = jax.nn.relu(x)
        x = self.layers[-1](x) # Shape: [out_channels, H, W]
        return jnp.transpose(x, (1, 2, 0)) # Shape: [H, W, out_channels]

class ContextFlowVectorField(eqx.Module):
    """The Vector field for Neural ODE, augmented with NCF Taylor expansion."""
    mlp: eqx.Module
    taylor_order: int = eqx.field(static=True)

    def __init__(self, state_dim, context_dim, hidden_dim, taylor_order, key):
        self.taylor_order = taylor_order
        in_size = 1 + state_dim + context_dim
        self.mlp = eqx.nn.MLP(in_size=in_size, out_size=state_dim, width_size=hidden_dim, 
                              depth=3, activation=jax.nn.swish, key=key)

    def __call__(self, t, y, args):
        ctx, ctx_ = args  

        def vf(xi):
            t_arr = jnp.array([t])
            in_feat = jnp.concatenate([t_arr, y, xi])
            return self.mlp(in_feat)

        if self.taylor_order == 0:
            return vf(ctx)
        elif self.taylor_order == 1:
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx - xi_,))[1]
            return vf(ctx_) + gradvf(ctx_)
        elif self.taylor_order == 2:
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx - xi_,))[1]
            scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx - ctx_,))[1]
            return vf(ctx_) + 1.5 * gradvf(ctx_) + 0.5 * scd_order_term
        else:
            raise ValueError("Taylor order must be 0, 1, or 2.")

class NCFOdeModel(eqx.Module):
    hypernet_phi: CNNEncoder
    vector_field: ContextFlowVectorField
    context_embedding: eqx.nn.Embedding
    
    # Conditional decoding branches
    weight_space_ode: bool = eqx.field(static=True)
    root_structure: Optional[RootMLP] = eqx.field(static=True)
    unravel_fn: Optional[callable] = eqx.field(static=True)
    cnn_decoder: Optional[CNNDecoder]
    
    state_dim: int = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)

    def __init__(self, root_width, root_depth, num_freqs, frame_shape, context_dim, dataset_size, taylor_order, vf_hidden_dim, weight_space_ode, ode_state_dim, key):
        k_root, k_phi, k_vf, k_emb, k_dec = jax.random.split(key, 5)
        self.num_freqs = num_freqs
        self.frame_shape = frame_shape
        self.weight_space_ode = weight_space_ode
        H, W, C = frame_shape
        out_c_dim = C * 2 if CONFIG["use_nll_loss"] else C
        
        if self.weight_space_ode:
            coord_dim = 2 + 2 * 2 * num_freqs 
            template_root = RootMLP(coord_dim, out_c_dim, root_width, root_depth, k_root)
            flat_params, self.unravel_fn = ravel_pytree(template_root)
            self.state_dim = flat_params.shape[0]
            self.root_structure = template_root
            self.cnn_decoder = None
        else:
            self.state_dim = ode_state_dim
            self.root_structure = None
            self.unravel_fn = None
            self.cnn_decoder = CNNDecoder(in_dim=self.state_dim, out_channels=out_c_dim, spatial_shape=(H, W), key=k_dec, hidden_width=16, depth=3)

        # Hypernetwork embeds image 0 into the appropriate abstract space
        self.hypernet_phi = CNNEncoder(in_channels=C, out_dim=self.state_dim, spatial_shape=(H, W), key=k_phi, hidden_width=64, depth=3)
        
        self.context_embedding = eqx.nn.Embedding(dataset_size, context_dim, key=k_emb)
        self.context_embedding = eqx.tree_at(lambda m: m.weight, self.context_embedding, jnp.zeros_like(self.context_embedding.weight))

        self.vector_field = ContextFlowVectorField(state_dim=self.state_dim, context_dim=context_dim, 
                                                   hidden_dim=vf_hidden_dim, taylor_order=taylor_order, key=k_vf)

    def render_pixels(self, thetas, coords):
        def render_pt(theta, coord):
            root = self.unravel_fn(theta)
            encoded_coord = fourier_encode(coord, self.num_freqs)
            return root(encoded_coord)
        return jax.vmap(render_pt)(thetas, coords)

    def rollout_ode(self, h0, ctx, ctx_, T):
        t_eval = jnp.arange(0.0, float(T))
        term = diffrax.ODETerm(self.vector_field)
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=t_eval)
        
        sol = diffrax.diffeqsolve(
            term, solver, t0=t_eval[0], t1=t_eval[-1], dt0=0.1,
            y0=h0, args=(ctx, ctx_), saveat=saveat
        )
        return sol.ys # Shape: [T, state_dim]

    def _get_preds_single(self, ref_video, ctx, ctx_, coords_grid):
        H, W, C = self.frame_shape
        flat_coords = coords_grid.reshape(-1, 2)
        T = ref_video.shape[0]
        init_frame = ref_video[0]

        h0 = self.hypernet_phi(jnp.transpose(init_frame, (2, 0, 1)))
        h_traj = self.rollout_ode(h0, ctx, ctx_, T)

        def render_frame(state_vec):
            if self.weight_space_ode:
                thetas_frame = jnp.tile(state_vec, (H * W, 1))
                pred_flat = self.render_pixels(thetas_frame, flat_coords)
                out_dim = C * 2 if CONFIG["use_nll_loss"] else C
                pred_img = pred_flat.reshape(H, W, out_dim)
            else:
                pred_img = self.cnn_decoder(state_vec)

            if CONFIG["use_nll_loss"]:
                mean, std = pred_img[..., :C], pred_img[..., C:]
                std = jax.nn.softplus(std) + 1e-4
                return jnp.concatenate([mean, std], axis=-1)
            else:
                return pred_img

        pred_video = jax.vmap(render_frame)(h_traj)
        return pred_video

    def __call__(self, ref_videos, indices, coords_grid, key):
        ctxs = jax.vmap(self.context_embedding)(indices)
        perm_idx = jax.random.permutation(key, ctxs.shape[0])
        ctxs_pool = ctxs[perm_idx]

        batched_fn = jax.vmap(self._get_preds_single, in_axes=(0, 0, 0, None))
        return batched_fn(ref_videos, ctxs, ctxs_pool, coords_grid)

    def inference_rollout(self, ref_video, ctx, coords_grid):
        return self._get_preds_single(ref_video, ctx, ctx, coords_grid)

#%% Cell 4: Initialization & Training/Loading Logic

key, subkey = jax.random.split(key)

model = NCFOdeModel(
    CONFIG["root_width"], 
    CONFIG["root_depth"], 
    CONFIG["num_fourier_freqs"], 
    (H, W, C), 
    CONFIG["context_dim"],
    CONFIG["dataset_size"],
    CONFIG["taylor_order"],
    CONFIG["vf_hidden_dim"],
    CONFIG["weight_space_ode"],
    CONFIG["ode_state_dim"],
    subkey
)

print(f"Total Trainable Parameters in NCF-ODE WARP: {count_trainable_params(model)}")

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
    def train_step(model, opt_state, ref_videos, indices, coords_grid, step_key):
        def loss_fn(m):
            pred_videos = m(ref_videos, indices, coords_grid, step_key)
            
            k_full, _ = jax.random.split(step_key, 2)
            full_indices = jax.random.choice(k_full, ref_videos.shape[1], shape=(CONFIG["nb_loss_steps_full"],), replace=False)
            
            pred_selected = pred_videos[:, full_indices]
            ref_selected = ref_videos[:, full_indices]
            
            if CONFIG["use_nll_loss"]:
                pred_means, pred_stds = pred_selected[..., :C], pred_selected[..., C:]
                nll = 0.5 * jnp.log(2 * jnp.pi * pred_stds**2) + 0.5 * ((ref_selected - pred_means)**2 / (pred_stds**2 + 1e-8))
                loss_full = jnp.mean(nll)
            else:
                loss_full = jnp.mean((pred_selected - ref_selected)**2)

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
        
        for batch_idx, (batch_videos, batch_indices) in enumerate(train_loader):
            key, subkey = jax.random.split(key)
            
            model, opt_state, loss = train_step(model, opt_state, batch_videos, batch_indices, coords_grid, subkey)
            epoch_losses.append(loss)
            
            current_scale = optax.tree_utils.tree_get(opt_state, "scale")
            lr_scales.append(current_scale)

            if not SINGLE_BATCH:
                break

        all_losses.extend(epoch_losses)

        if not SINGLE_BATCH and ((epoch+1) % CONFIG["print_every"] == 0 or (epoch+1) == CONFIG["nb_epochs"] - 1):
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1}/{CONFIG['nb_epochs']} - Avg Loss: {avg_epoch_loss:.4f} - LR Scale: {current_scale:.4f}", flush=True)

        if epoch in [4, CONFIG["nb_epochs"]//2, 2*CONFIG["nb_epochs"]//3]:
            eqx.tree_serialise_leaves(artefacts_path / f"tf_model_ep{epoch+1}.eqx", model)

        if (epoch+1) % (max(CONFIG["nb_epochs"]//10, 1)) == 0:
            val_keys, step_k = jax.random.split(key)
            val_videos = model(sample_batch, sample_indices, coords_grid, step_k)
            plot_pred_ref_videos_rollout(val_videos[0], 
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
        all_losses, lr_scales = [], []
        print("Warning: loss_history.npy not found.")

#%% Cell 5: Final Visualizations & Test Inference Adaptation

print("\n=== Generating Dashboards ===")

if len(all_losses) > 0:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1 = 'teal'
    ax1.plot(all_losses, color=color1, alpha=0.8, label="Total Loss")
    if not CONFIG["use_nll_loss"]:
        ax1.set_yscale('log')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss", color=color1)
    ax1.grid(True)
    
    ax2 = ax1.twinx()  
    color2 = 'crimson'
    if len(lr_scales) > 0:
        ax2.plot(lr_scales, color=color2, linewidth=2, label="LR Scale Multiplier")
        ax2.set_ylabel("LR Scale", color=color2)
    
    plt.title("Training Loss and Adaptive Learning Rate Decay")
    fig.tight_layout()
    plt.savefig(plots_path / "loss_and_lr_history.png")
    plt.show()

def adapt_context_for_inference(model, video, coords, context_ratio=0.5, steps=50, lr=1e-2):
    T = video.shape[0]
    num_context_frames = max(1, int(T * context_ratio))
    ctx = jnp.zeros((CONFIG["context_dim"],))

    def loss_fn(c):
        preds = model.inference_rollout(video, c, coords)
        
        if CONFIG["use_nll_loss"]:
            pred_means, pred_stds = preds[:num_context_frames, ..., :C], preds[:num_context_frames, ..., C:]
            nll = 0.5 * jnp.log(2 * jnp.pi * pred_stds**2) + 0.5 * ((video[:num_context_frames] - pred_means)**2 / (pred_stds**2 + 1e-8))
            return jnp.mean(nll)
        else:
            return jnp.mean((preds[:num_context_frames] - video[:num_context_frames])**2)

    opt = optax.adam(lr)
    opt_state = opt.init(ctx)

    @eqx.filter_jit
    def step_fn(carry, _):
        c, o_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(c)
        updates, o_state = opt.update(grads, o_state)
        c = optax.apply_updates(c, updates)
        return (c, o_state), loss

    (final_ctx, _), losses = jax.lax.scan(step_fn, (ctx, opt_state), None, length=steps)
    print(f"Test-time context adaptation loss: start={losses[0]:.4f} -> end={losses[-1]:.4f}")
    return final_ctx

test_base_dataset = datasets.MovingMNIST(root=data_path, split="test", download=True)
test_dataset = IndexedDataset(test_base_dataset)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=numpy_collate, drop_last=True)

test_batch, test_indices = next(iter(test_loader))
pad_length = 20 - test_batch.shape[1]
test_batch = jnp.concatenate([test_batch, np.zeros((test_batch.shape[0], pad_length, H, W, C), dtype=test_batch.dtype)], axis=1)

test_seq_id = np.random.randint(0, test_batch.shape[0])
test_video = test_batch[test_seq_id]

print("\nAdapting zeroed context vector on new test sequence...")
adapted_ctx = adapt_context_for_inference(model, test_video, coords_grid, CONFIG["inf_context_ratio"])

final_video_pred = model.inference_rollout(test_video, adapted_ctx, coords_grid)

if CONFIG["use_nll_loss"]:
    print(f"Final Predicted Video Mean Range: min={final_video_pred[...,:C].min():.4f}, max={final_video_pred[...,:C].max():.4f}")
    print(f"Final Predicted Video Std Range: min={final_video_pred[...,C:].min():.4f}, max={final_video_pred[...,C:].max():.4f}")

plot_pred_ref_videos_rollout(
    final_video_pred, 
    test_video,
    title=f"Pred", 
    save_name="inference_forecast_rollout.png"
)