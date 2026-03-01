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

SINGLE_BATCH = False
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


class WARP(eqx.Module):
    rnn_layers: list
    theta_base: jax.Array
    hypernet_phi: Optional[CNNEncoder]
    controlnet_psi: Optional[CNNEncoder]

    use_hypernet: bool = eqx.field(static=True)
    use_controlnet: bool = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    root_structure: RootMLP = eqx.field(static=True)
    unravel_fn: callable = eqx.field(static=True)
    d_theta: int = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)

    def __init__(self, root_width, root_depth, num_freqs, frame_shape, num_layers, use_hypernet, use_controlnet, key):
        k_root, k_rnn, k_phi, k_psi = jax.random.split(key, 4)
        self.num_freqs = num_freqs
        self.frame_shape = frame_shape
        self.num_layers = num_layers
        self.use_hypernet = use_hypernet
        self.use_controlnet = use_controlnet
        H, W, C = frame_shape

        coord_dim = 2 + 2 * 2 * num_freqs 
        root_out_dim = C * 2 if CONFIG["use_nll_loss"] else C
        template_root = RootMLP(coord_dim, root_out_dim, root_width, root_depth, k_root)

        flat_params, self.unravel_fn = ravel_pytree(template_root)
        self.d_theta = flat_params.shape[0]
        self.root_structure = template_root
        self.theta_base = flat_params

        if use_hypernet:
            self.hypernet_phi = CNNEncoder(in_channels=C, out_dim=self.d_theta*1, spatial_shape=(H, W), key=k_phi, hidden_width=64, depth=4)

            ## Initialise the weight and biases at zero
            # self.hypernet_phi = jax.tree_map(lambda x: jnp.zeros_like(x), self.hypernet_phi)
            # self.hypernet_phi = jax.tree_map(lambda x: x*0.001, self.hypernet_phi)

        else:
            self.hypernet_phi = None

        # ## TODO: the B space is the weight space:
        # CONFIG["rec_feat_dim"] = self.d_theta*2

        if use_controlnet:
            self.controlnet_psi = CNNEncoder(in_channels=C*2, out_dim=CONFIG["rec_feat_dim"], spatial_shape=(H, W), key=k_psi, hidden_width=32, depth=3)
            in_dim = CONFIG["rec_feat_dim"]
        else:
            self.controlnet_psi = None
            in_dim = H * W * C * 1
            
        rnn_layers = []
        keys = jax.random.split(k_rnn, num_layers)

        if num_layers == 1:
            rnn_layers.append(LinearRNNLayer(in_dim, self.d_theta, keys[0]))
        else:
            # First layer
            rnn_layers.append(LinearRNNLayer(in_dim, CONFIG["rec_feat_dim"], keys[0]))
            # Intermediate layers
            for i in range(1, num_layers - 1):
                rnn_layers.append(LinearRNNLayer(CONFIG["rec_feat_dim"], CONFIG["rec_feat_dim"], keys[i]))
            # Last layer maps strictly back into weight space
            rnn_layers.append(LinearRNNLayer(CONFIG["rec_feat_dim"], self.d_theta, keys[-1]))

        self.rnn_layers = rnn_layers

    @property
    def A(self):
        """Property to keep evaluation/plotting code transparently compatible."""
        return self.rnn_layers[-1].A if type(self.rnn_layers[-1].A) == jnp.ndarray else self.rnn_layers[-1].A.layers[-1].weight

    @property
    def B(self):
        return self.rnn_layers[-1].B if type(self.rnn_layers[-1].B) == jnp.ndarray else self.rnn_layers[-1].B.layers[-1].weight

    def render_pixels(self, thetas, coords):
        def render_pt(theta, coord):
            root = self.unravel_fn(theta)
            encoded_coord = fourier_encode(coord, self.num_freqs)
            out = root(encoded_coord)
            
            # gray_fg = jax.nn.sigmoid(out[0:1])
            # gray_bg = jax.nn.sigmoid(out[1:2])
            # alpha   = jax.nn.sigmoid(out[2:3])

            # gray_fg = out[0:1]
            # gray_bg = out[1:2]

            # ## Alpha should be hard thresholded to encourage more discrete rendering and reduce blurriness
            # alpha = jax.nn.sigmoid(out[2:3])

            # return alpha * gray_fg + (1.0 - alpha) * gray_bg

            # return out[0:1]
            
            # return out

            if CONFIG["use_nll_loss"]:
                mean, std = out[:C], out[C:]
                # std = jnp.clip(std, 1e-4, 10.0)
                # std = jnp.maximum(jax.nn.softplus(std), 1e-4)
                # std = jnp.maximum(jax.nn.softplus(std), 0.5)
                # std = jnp.maximum(std, 0.5)
                # std = jax.nn.relu(std) + 1e-4
                std = jax.nn.softplus(std) + 1e-4
                return jnp.concatenate([mean, std], axis=-1)
            else:
                return out

        return jax.vmap(render_pt)(thetas, coords)

    def _get_thetas_and_preds_single(self, ref_video, p_forcing, key, coords_grid, inf_context_ratio, precompute_ref_diffs=False):
        H, W, C = self.frame_shape
        flat_coords = coords_grid.reshape(-1, 2)
        T = ref_video.shape[0]
        init_frame = ref_video[0]

        # Shared Rendering function
        def render_frame(theta):
            # ## split and compute theta with the FiLM projection
            # theta_mu, theta_scale = jnp.split(theta, 2, axis=-1)
            # theta_scaled = self.theta_base*(1+theta_scale) + theta_mu

            # thetas_frame = jnp.tile(theta, (H*W, 1))
            thetas_frame = jnp.tile(theta+self.theta_base, (H*W, 1))
            pred_flat = self.render_pixels(thetas_frame, flat_coords)

            if CONFIG["use_nll_loss"]:
                # pred_means, pred_stds = pred_flat[:, :C], jax.nn.softplus(pred_flat[:, C:])
                # pred_flat = pred_means.reshape(H, W, C)
                return pred_flat.reshape(H, W, C*2)
            else:
                return pred_flat.reshape(H, W, C)

        # Shared Initialization of multi-layer hidden states
        h_inits = []
        for i in range(self.num_layers):
            is_last = (i == self.num_layers - 1)
            if is_last and self.use_hypernet and self.hypernet_phi is not None:
                h_inits.append(self.hypernet_phi(jnp.transpose(init_frame, (2, 0, 1))))
                # h_inits.append(self.hypernet_phi(jnp.transpose(init_frame, (2, 0, 1))) + self.theta_base)
            elif is_last:
                h_inits.append(self.theta_base)
            else:
                h_inits.append(jnp.zeros(self.rnn_layers[i].B.shape[0]))

        if precompute_ref_diffs:
            # ==========================================================
            # FAST PATH: Decoupled Sequence Scans (No Autoregressive Rendering)
            # ==========================================================
            shifted_ref_video = jnp.concatenate([ref_video[1:], ref_video[-1:]], axis=0)
            dx_frames = shifted_ref_video - ref_video
            
            if self.use_controlnet and self.controlnet_psi is not None:
                dx_feats = jax.vmap(self.controlnet_psi)(jnp.transpose(dx_frames, (0, 3, 1, 2)))
            else:
                dx_feats = dx_frames.reshape(T, -1) / jnp.sqrt(H * W * C)

            curr_input_seq = dx_feats
            
            for i, layer in enumerate(self.rnn_layers):
                is_last = (i == self.num_layers - 1)
                h_init = h_inits[i]
                
                def fast_scan(h, dx):
                    h_next = layer.A @ h + layer.B @ dx
                    return h_next, h_next

                _, h_seq = jax.lax.scan(fast_scan, h_init, curr_input_seq)

                if not is_last:
                    # Precompute the sequential differences for the next layer
                    h_seq_prev = jnp.concatenate([h_init[None, ...], h_seq[:-1]], axis=0)
                    curr_input_seq = h_seq - h_seq_prev
                else:
                    final_thetas = jnp.concatenate([h_init[None, ...], h_seq[:-1]], axis=0)

            pred_video = jax.vmap(render_frame)(final_thetas)
            return pred_video

        else:
            # ==========================================================
            # DEFAULT PATH: True Autoregressive Step-by-Step Scan
            # ==========================================================
            def scan_step(state, scan_inputs):
                gt_curr_frame, step_idx = scan_inputs
                h_states, prev_frame_selected, k = state
                k, subk = jax.random.split(k)

                # 1. Render the prediction for the CURRENT step from the LAST layer (theta)
                theta = h_states[-1]
                pred_frame = render_frame(theta)

                if CONFIG["use_nll_loss"]:
                    # pred_frame_mean, pred_frame_std = pred_frame[..., :C], jax.nn.softplus(pred_frame[..., C:])
                    pred_frame_mean, pred_frame_std = pred_frame[..., :C], pred_frame[..., C:]
                    pred_frame_sample = pred_frame_mean + pred_frame_std * jax.random.normal(subk, pred_frame_mean.shape)
                    prev_frame_selected = prev_frame_selected[:, :, :C]
                else:
                    pred_frame_sample = pred_frame

                # 2. First Layer Coin Flip
                t_ratio = step_idx / (T - 1)
                is_context = t_ratio <= inf_context_ratio
                is_forced = jax.random.bernoulli(subk, p_forcing)
                use_gt = jnp.logical_or(is_context, is_forced)

                frame_t = jnp.where(use_gt, gt_curr_frame, pred_frame_sample)

                # 3. Input computation for Layer 1
                dx0 = frame_t - prev_frame_selected
                if self.use_controlnet and self.controlnet_psi is not None:
                    # dx_feat = self.controlnet_psi(jnp.transpose(dx0, (2, 0, 1)))

                    concat = jnp.concatenate([frame_t, prev_frame_selected], axis=-1)
                    dx_feat = self.controlnet_psi(jnp.transpose(concat, (2, 0, 1)))

                    # ft_feats = self.controlnet_psi(jnp.transpose(frame_t, (2, 0, 1)))
                    # ps_feats = self.controlnet_psi(jnp.transpose(prev_frame_selected, (2, 0, 1)))
                    # dx_feat = ft_feats - ps_feats
                else:
                    # dx_feat = dx0.flatten() / jnp.sqrt(dx0.size)
                    dx_feat = dx0.flatten()

                    # ft_feats = self.hypernet_phi(jnp.transpose(frame_t, (2, 0, 1)))
                    # ps_feats = self.hypernet_phi(jnp.transpose(prev_frame_selected, (2, 0, 1)))
                    # dx_feat = jnp.concatenate([ft_feats, ps_feats], axis=-1)


                # 4. Deep Linear Recurrence cascade (Precomputing differences inline)
                new_h_states = []
                curr_dx = dx_feat
                
                for i, layer in enumerate(self.rnn_layers):
                    h_prev = h_states[i]

                    # h_next = layer.A @ h_prev + layer.B @ curr_dx

                    h_next = layer.A(h_prev) + layer.B(curr_dx)

                    # h_next = layer.B @ curr_dx

                    # h_next = layer.A @ h_prev + layer.B @ ft_feats - layer.C @ ps_feats

                    # h_next = jax.nn.tanh(layer.A @ h_prev + layer.B @ curr_dx)
                    new_h_states.append(h_next)
                    
                    # Precompute difference from this layer to act as the pure linear input for the next layer
                    curr_dx = h_next - h_prev
                
                new_state = (new_h_states, frame_t, subk)
                # new_state = (new_h_states, gt_curr_frame, subk)
                return new_state, pred_frame

            init_state = (h_inits, init_frame, key)
            scan_inputs = (jnp.concatenate([ref_video[1:], ref_video[-1:]], axis=0), jnp.arange(T))
            _, pred_video = jax.lax.scan(scan_step, init_state, scan_inputs)

            return pred_video

    def __call__(self, ref_videos, p_forcing, keys, coords_grid, inf_context_ratio, precompute_ref_diffs=False):
        is_single = (ref_videos.ndim == 4)
        if is_single:
            ref_videos = ref_videos[None, ...]
            keys = keys[None, ...] if keys.ndim == 1 else keys
            
        batched_fn = jax.vmap(self._get_thetas_and_preds_single, in_axes=(0, None, 0, None, None, None))
        preds = batched_fn(ref_videos, p_forcing, keys, coords_grid, inf_context_ratio, precompute_ref_diffs)
        
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
    CONFIG["root_width"], 
    CONFIG["root_depth"], 
    CONFIG["num_fourier_freqs"], 
    (H, W, C), 
    CONFIG["num_layers"],
    CONFIG["use_hypernet"],
    CONFIG["use_controlnet"],
    subkey
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

