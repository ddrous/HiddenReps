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
TRAIN_PHASE_0 = True  # Optional pretraining of Encoder + Base Theta via autoencoding
TRAIN_PHASE_1 = True  # Train IDM, FDM, and Base Theta
TRAIN_PHASE_2 = True  # Train GCM (Memory Model) to match IDM
RUN_DIR = "./" if (not TRAIN_PHASE_1 or not TRAIN_PHASE_2) else None

SINGLE_BATCH = False
USE_NLL_LOSS = False

CONFIG = {
    "seed": 42,
    
    # Phase 1 Params
    "p1_nb_epochs": 10000,        
    "p1_learning_rate": 1e-4 if USE_NLL_LOSS else 1e-5,
    "reverse_video_aug": False,
    "static_video_aug": True,
    "action_l1_reg": 0.00,     # Skipped automatically if discrete_actions=True
    "mse_weight": 1.0,        
    "aux_encoder_loss": False,
    "aux_loss_weight": 1.0,
    "aux_loss_num_steps": 4,

    # Phase 2 Params
    "p2_nb_epochs": 10000,
    "p2_learning_rate": 1e-5,

    "print_every": 10,
    "batch_size": 2 if SINGLE_BATCH else 8,
    "inf_context_ratio": 0.5,
    "use_nll_loss": USE_NLL_LOSS,

    # --- Architecture Params ---
    "lam_space": 8,            # Embedding dimension size
    "mem_space": 256,
    "icl_decoding": True,
    "discrete_actions": True,  # Key difference for MiniGrid!
    "num_actions": 4,
    "split_forward": True,
    "root_width": 12,
    "root_depth": 5,
    "num_fourier_freqs": 6,
    "use_time_in_root": False,
    "pretrain_encoder": TRAIN_PHASE_0, 

    # --- Plateau Scheduler Config ---
    "lr_patience": 300,      
    "lr_cooldown": 100,       
    "lr_factor": 0.5,        
    "lr_rtol": 1e-3,         
    "lr_accum_size": 5,     
    "lr_min_scale": 1e-0     
}

key = jax.random.PRNGKey(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

def setup_run_dir(base_dir="experiments"):
    if TRAIN_PHASE_1 or TRAIN_PHASE_2:
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

# Path to explicitly load Phase 1 weights if Phase 1 is skipped
P1_LOAD_PATH = artefacts_path / "model_phase1_final.eqx" 

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

print("Loading MiniGrid Dataset...")
try:
    data_path = './data' if (TRAIN_PHASE_1 or TRAIN_PHASE_2) else '../../data'

    ## Manually load train and test splits
    minigrid_arrays = np.load(data_path + "/MiniGrid/minigrid.npy")
    ## Only use 12 steps
    minigrid_arrays = minigrid_arrays[:, :12]
    print(f"Original loaded MiniGrid shape: {minigrid_arrays.shape} (N, T, H, W, 3)")
    
    ## Split, 80/20 train test
    train_size = int(0.8 * minigrid_arrays.shape[0])
    train_arrays = minigrid_arrays[:train_size]
    test_arrays = minigrid_arrays[train_size:]

    ## Create PyTorch dataset
    class MiniGridDataset(torch.utils.data.Dataset):
        def __init__(self, data_array):
            self.data_array = data_array

        def __len__(self):
            return self.data_array.shape[0]

        def __getitem__(self, idx):
            video = self.data_array[idx]  # Shape (T, H, W, 3)
            return torch.from_numpy(video.astype(np.float32))

    dataset = MiniGridDataset(train_arrays)

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
    print(f"Could not load MiniGrid: {e}")
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

    if video.shape[-1] == C:
        fig, axes = plt.subplots(2, nb_frames, figsize=(2.5*nb_frames, 2.5*2))
        indices_to_plot = np.arange(0, nb_frames, 1)
        for i, idx in enumerate(indices_to_plot):
            video_to_plot = video[idx] if not rescale else (video[idx] + 1.0) / 2.0
            sbimshow(video_to_plot, title=f"{title} t={idx}", ax=axes[0, i])
            ref_idx = min(idx, ref_video.shape[0]-1)
            sbimshow(ref_video[ref_idx], title=f"Ref t={ref_idx}", ax=axes[1, i])
    else:
        fig, axes = plt.subplots(3, nb_frames, figsize=(2.5*nb_frames, 3.5*2))
        indices_to_plot = np.arange(0, nb_frames, 1)
        for i, idx in enumerate(indices_to_plot):
            video_to_plot = video[idx, ..., :C] if not rescale else (video[idx, ..., :C] + 1.0) / 2.0
            sbimshow(video_to_plot, title=f"Mean t={idx}", ax=axes[0, i])
            sbimshow(video[idx, ..., C:], title=f"Std t={idx}", ax=axes[1, i])
            ref_idx = min(idx, ref_video.shape[0]-1)
            sbimshow(ref_video[ref_idx], title=f"Ref t={ref_idx}", ax=axes[2, i])

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)
    plt.show()
    plt.close()

#%% Visualize Augmentations (Ported from MovingMNIST)
print("\n=== Visualizing Video Augmentations ===")

# 1. Grab a single video and duplicate it to make a test batch of 4 identical videos
test_vid = sample_batch[0:1] # Shape: (1, T, H, W, C)
test_batch = jnp.repeat(test_vid, 4, axis=0)

# 2. Force deterministic boolean arrays to test every combination
# Row 0: Original
# Row 1: Reversed
# Row 2: Static Front
# Row 3: Static Back
do_reverse = jnp.array([False, True, False, False])
do_static  = jnp.array([False, False, True, True])
add_to_front = jnp.array([False, False, True, False]) 

# --- Apply Reverse Augmentation ---
aug_batch = jax.vmap(lambda rev, vid: jax.lax.cond(
    rev, 
    lambda v: jnp.flip(v, axis=0), 
    lambda v: v, 
    vid
))(do_reverse, test_batch)

# --- Apply Static Augmentation ---
nb_frames = aug_batch.shape[1]
repeat_frames = nb_frames // 4

def static_aug(add_front, v):
    return jax.lax.cond(
        add_front, 
        lambda v_in: jnp.concatenate([jnp.repeat(v_in[:1], repeats=repeat_frames, axis=0), v_in[1:nb_frames-repeat_frames+1]], axis=0),
        lambda v_in: jnp.concatenate([v_in[:nb_frames-repeat_frames], jnp.repeat(v_in[nb_frames-repeat_frames:nb_frames-repeat_frames+1], repeats=repeat_frames, axis=0)], axis=0),
        v
    )

aug_batch = jax.vmap(lambda apply_stat, add_front, vid: jax.lax.cond(
    apply_stat,
    lambda v: static_aug(add_front, v),
    lambda v: v,
    vid
))(do_static, add_to_front, aug_batch)

# --- Plot the Results ---
fig, axes = plt.subplots(4, nb_frames, figsize=(nb_frames * 1.5, 4 * 1.5))
row_titles = ["Original", "Reversed", "Static (Front)", "Static (Back)"]

for row in range(4):
    for t in range(nb_frames):
        ax = axes[row, t]
        img = aug_batch[row, t]
        
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
            
        ax.imshow(np.clip(img, 0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        
        if t == 0:
            ax.set_ylabel(row_titles[row], fontsize=14, rotation=0, labelpad=60, ha='center', va='center', fontweight='bold')
        if row == 0:
            ax.set_title(f"t={t}")

plt.suptitle("Effect of Temporal Video Augmentations on MiniGrid", y=1.00, fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig(plots_path / "minigrid_augmentations.png")
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

    def __init__(self, in_channels, out_dim, spatial_shape, key, hidden_width=32, depth=4):
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
            self.giant_mlp = eqx.nn.MLP(dyn_dim + lam_dim, dyn_dim, width_size=dyn_dim*2, depth=3, key=k3)

    def __call__(self, z_prev, a):
        if self.split_forward:
            return self.mlp_A(z_prev) + self.mlp_B(a)
        else:
            return self.giant_mlp(jnp.concatenate([z_prev, a], axis=-1))

class TransformerBlock(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(self, d_model, num_heads, key):
        k1, k2 = jax.random.split(key)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=d_model,
            use_query_bias=True, use_key_bias=True,
            use_value_bias=True, use_output_bias=True, key=k1
        )
        self.mlp = eqx.nn.MLP(d_model, d_model, width_size=d_model * 4, depth=1, key=k2)
        self.ln1 = eqx.nn.LayerNorm(d_model)
        self.ln2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, mask):
        x_norm = jax.vmap(self.ln1)(x)
        attn_out = self.attn(x_norm, x_norm, x_norm, mask=mask)
        x = x + attn_out
        x = x + jax.vmap(self.mlp)(jax.vmap(self.ln2)(x))
        return x

class InverseDynamics(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, dyn_dim, lam_dim, key, num_actions=None):
        if num_actions: ## Discrete case
            self.mlp = eqx.nn.MLP(dyn_dim * 2, num_actions, width_size=dyn_dim*1, depth=3, key=key)
        else:
            self.mlp = eqx.nn.MLP(dyn_dim * 2, lam_dim, width_size=dyn_dim*1, depth=3, key=key)
        
    def __call__(self, z_prev, z_target):
        return self.mlp(jnp.concatenate([z_prev, z_target], axis=-1))


class VanillaRNNCell(eqx.Module):
    """A standard Elman RNN cell for lightweight baselining."""
    weight_ih: eqx.nn.Linear
    weight_hh: eqx.nn.Linear

    def __init__(self, input_size: int, hidden_size: int, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key)
        self.weight_ih = eqx.nn.Linear(input_size, hidden_size, use_bias=True, key=k1)
        self.weight_hh = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=k2)

    def __call__(self, input: jax.Array, hidden: jax.Array) -> jax.Array:
        return jax.nn.tanh(self.weight_ih(input) + self.weight_hh(hidden))


class MemoryModule(eqx.Module):
    """
    Recurrent Memory Module for Latent Actions.
    Uses either a Vanilla RNN, GRU, or LSTM as the core memory architecture.
    """
    d_model: int
    rnn_type: str = eqx.field(static=True)
    lam_dim: int = eqx.field(static=True)
    num_actions: Optional[int] = eqx.field(static=True)
    
    rnn_cell: eqx.Module
    action_decoder: eqx.nn.MLP

    def __init__(self, lam_dim, mem_dim, latent_dim, key, rnn_type="GRU", num_actions=None, **kwargs):
        self.lam_dim = lam_dim
        self.d_model = mem_dim
        self.rnn_type = rnn_type.upper()
        self.num_actions = num_actions
        
        k1, k2 = jax.random.split(key, 2)
        
        input_dim = latent_dim + lam_dim
        if self.rnn_type == "LSTM":
            self.rnn_cell = eqx.nn.LSTMCell(input_dim, self.d_model, key=k1)
        elif self.rnn_type == "GRU":
            self.rnn_cell = eqx.nn.GRUCell(input_dim, self.d_model, key=k1)
        elif self.rnn_type == "RNN":
            self.rnn_cell = VanillaRNNCell(input_dim, self.d_model, key=k1)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Must be 'LSTM', 'GRU', or 'RNN'.")

        decode_input_dim = self.d_model + latent_dim
        out_dim = num_actions if num_actions is not None else lam_dim
        
        self.action_decoder = eqx.nn.MLP(
            in_size=decode_input_dim, out_size=out_dim, 
            width_size=self.d_model * 1, depth=1, key=k2
        )

    def reset(self, T):
        if self.rnn_type == "LSTM":
            return (jnp.zeros((self.d_model,)), jnp.zeros((self.d_model,)))
        else:
            return jnp.zeros((self.d_model,))

    def encode(self, state, step_idx, z, a):
        rnn_input = jnp.concatenate([z, a], axis=-1)
        new_state = self.rnn_cell(rnn_input, state)
        return new_state

    def decode(self, state, step_idx, z_current):
        if self.rnn_type == "LSTM":
            h = state[0]
        else:
            h = state
            
        decode_input = jnp.concatenate([h, z_current], axis=-1)
        return self.action_decoder(decode_input)


class LAM(eqx.Module):
    """ Action Model holding both the IDM (Phase 1) and GCM (Phase 2). """
    idm: InverseDynamics
    gcm: Optional[eqx.Module]
    discrete_actions: bool = eqx.field(static=True)
    action_embedding: Optional[eqx.nn.Embedding]

    def __init__(self, dyn_dim, lam_dim, mem_dim, max_len, num_heads, num_blocks, num_actions, key, phase=1):
        k1, k2 = jax.random.split(key)
        self.discrete_actions = num_actions is not None

        self.idm = InverseDynamics(dyn_dim, lam_dim, key=k1, num_actions=num_actions if self.discrete_actions else None)
        
        # Instantiate GCM in Phase 2
        if phase == 2:
            self.gcm = MemoryModule(lam_dim, mem_dim, dyn_dim, key=k2, rnn_type="GRU", num_actions=num_actions if self.discrete_actions else None)
        else:
            self.gcm = None

        if self.discrete_actions:
            self.action_embedding = eqx.nn.Embedding(num_actions, lam_dim, key=k2)
        else:
            self.action_embedding = None

    def discretise_action(self, logits):
        soft_probs = jax.nn.softmax(logits, axis=-1)
        hard_idx = jnp.argmax(logits, axis=-1)
        hard_probs = jax.nn.one_hot(hard_idx, num_classes=logits.shape[-1])
        ste_probs = soft_probs + jax.lax.stop_gradient(hard_probs - soft_probs)
        action = jnp.dot(ste_probs, self.action_embedding.weight)
        return action

    def inverse_dynamics(self, z_prev, z_target):
        if not self.discrete_actions:
            return self.idm(z_prev, z_target)
        else:
            logits = self.idm(z_prev, z_target)
            return self.discretise_action(logits)
    
    def inverse_dynamics_logits(self, z_prev, z_target):
        if not self.discrete_actions:
            raise ValueError("IDM does not produce logits in continuous action setting.")
        else:
            return self.idm(z_prev, z_target)

    def decode_memory(self, buffer, step_idx, z_current):
        if self.gcm is None:
            raise ValueError("GCM is not initialized in Phase 1.")
        if not self.discrete_actions:
            return self.gcm.decode(buffer, step_idx, z_current)
        else:
            logits = self.gcm.decode(buffer, step_idx, z_current)
            return self.discretise_action(logits)
        
    def decode_memory_logits(self, buffer, step_idx, z_current):
        if self.gcm is None:
            raise ValueError("GCM is not initialized in Phase 1.")
        if not self.discrete_actions:
            raise ValueError("GCM does not produce logits in continuous action setting.")
        else:
            return self.gcm.decode(buffer, step_idx, z_current)

    def encode_memory(self, buffer, step_idx, z_current, a):
        if self.gcm is None:
            raise ValueError("GCM is not initialized in Phase 1.")
        return self.gcm.encode(buffer, step_idx, z_current, a)
    
    def reset_memory(self, T):
        if self.gcm is None:
            raise ValueError("GCM is not initialized in Phase 1.")
        return self.gcm.reset(T)

class WARP(eqx.Module):
    encoder: CNNEncoder
    forward_dyn: ForwardDynamics
    theta_base: jax.Array
    action_model: LAM

    unravel_fn: callable = eqx.field(static=True)
    d_theta: int = eqx.field(static=True)
    lam_dim: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)
    split_forward: bool = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)
    mem_dim: int = eqx.field(static=True)
    phase: int = eqx.field(static=True)

    def __init__(self, root_width, root_depth, num_freqs, frame_shape, lam_dim, mem_dim, split_forward, key, phase=1):
        k_root, k_enc, k_lam, k_fwd, k_mem = jax.random.split(key, 5)
        self.frame_shape = frame_shape
        self.num_freqs = num_freqs
        self.lam_dim = lam_dim
        self.split_forward = split_forward
        self.phase = phase
        H, W, C = frame_shape

        coord_dim = 2 + 2 * 2 * num_freqs 
        root_out_dim = C * 2 if CONFIG["use_nll_loss"] else C
        add_time = 1 if CONFIG["use_time_in_root"] else 0
        template_root = RootMLP(coord_dim+add_time, root_out_dim, root_width, root_depth, k_root)
        
        flat_params, self.unravel_fn = ravel_pytree(template_root)
        self.d_theta = flat_params.shape[0]
        self.theta_base = flat_params

        # 32 Hidden width for MiniGrid
        self.encoder = CNNEncoder(in_channels=C, out_dim=self.d_theta, spatial_shape=(H, W), key=k_enc, hidden_width=32, depth=4)

        if CONFIG["pretrain_encoder"] and self.phase == 1:
            try:
                self.encoder, self.theta_base = eqx.tree_deserialise_leaves("minigrid_enc.eqx", (self.encoder, self.theta_base))
            except:
                print("Warning: minigrid_enc.eqx not found. Starting from scratch.")

        self.forward_dyn = ForwardDynamics(self.d_theta, lam_dim, split_forward, key=k_fwd)
        self.mem_dim = mem_dim

        num_actions = CONFIG["num_actions"] if CONFIG["discrete_actions"] else None
        self.action_model = LAM(self.d_theta, lam_dim, mem_dim, max_len=12, num_heads=4, num_blocks=4, num_actions=num_actions, key=k_lam, phase=self.phase)

    def render_pixels(self, theta, coords):
        def render_pt(theta, coord):
            root = self.unravel_fn(theta)
            if CONFIG["use_time_in_root"]:
                encoded_coord = jnp.concatenate([coord[:1], fourier_encode(coord[1:], self.num_freqs)], axis=-1)
            else:
                encoded_coord = fourier_encode(coord[1:], self.num_freqs)
            out = root(encoded_coord)
            if CONFIG["use_nll_loss"]:
                mean, std = out[:C], out[C:]
                std = jax.nn.softplus(std) + 1e-4
                return jnp.concatenate([mean, std], axis=-1)
            return out
        return jax.vmap(render_pt, in_axes=(None, 0))(theta, coords)

    def render_frame(self, theta_offset, coords_grid):
        H, W, C = self.frame_shape
        flat_coords = coords_grid.reshape(-1, 3)
        
        if not CONFIG["pretrain_encoder"]:
            theta = theta_offset + self.theta_base
        else:
            theta = theta_offset + jax.lax.stop_gradient(self.theta_base)

        pred_flat = self.render_pixels(theta, flat_coords)
        return pred_flat.reshape(H, W, -1)

    # -------------------------------------------------------------------------------------
    # PHASE 1 FORWARD: IDM Forcing (GCM is ignored/None)
    # -------------------------------------------------------------------------------------
    def phase1_forward(self, ref_video, coords_grid, render):
        T = ref_video.shape[0]
        init_frame = ref_video[0]

        z_init = self.encoder(jnp.transpose(init_frame, (2, 0, 1)))
        if CONFIG["pretrain_encoder"]:
            z_init = jax.lax.stop_gradient(z_init)

        @eqx.filter_checkpoint
        def scan_step(z_t, scan_inputs):
            o_tp1, step_idx = scan_inputs

            if render:
                time_coord = jnp.array([(step_idx-1)/(T-1)], dtype=z_t.dtype)
                coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
                pred_out = self.render_frame(z_t, coords_grid_t)
            else:
                pred_out = None

            z_tp1_enc = self.encoder(jnp.transpose(o_tp1, (2, 0, 1)))
            if CONFIG["pretrain_encoder"]:
                z_tp1_enc = jax.lax.stop_gradient(z_tp1_enc)

            a_t = self.action_model.inverse_dynamics(z_t, z_tp1_enc)
            z_tp1 = self.forward_dyn(z_t, a_t)

            return z_tp1, (a_t, (z_tp1, z_tp1_enc), pred_out)

        scan_inputs = (jnp.concatenate([ref_video[1:], jnp.zeros_like(ref_video[:1])], axis=0), jnp.arange(1, T+1))
        _, (actions, latents, pred_video) = jax.lax.scan(scan_step, z_init, scan_inputs)

        return actions, latents, pred_video

    # -------------------------------------------------------------------------------------
    # PHASE 2 FORWARD: Action Matching (IDM/Base is frozen via stop_gradient)
    # -------------------------------------------------------------------------------------
    def phase2_forward(self, ref_video):
        T = ref_video.shape[0]
        init_frame = ref_video[0]

        # Explicitly freeze via stop_gradient
        z_init = jax.lax.stop_gradient(self.encoder(jnp.transpose(init_frame, (2, 0, 1))))
        m_init = self.action_model.reset_memory(T)

        @eqx.filter_checkpoint
        def scan_step(carry, scan_inputs):
            z_t, m_t = carry
            o_tp1, step_idx = scan_inputs

            z_tp1_enc = jax.lax.stop_gradient(self.encoder(jnp.transpose(o_tp1, (2, 0, 1))))
            
            # Ground truth action from IDM (frozen)
            logits_target = jax.lax.stop_gradient(self.action_model.inverse_dynamics_logits(z_t, z_tp1_enc))
            a_target = jax.lax.stop_gradient(self.action_model.discretise_action(logits_target))

            # Predicted action from GCM
            logits_pred = self.action_model.decode_memory_logits(m_t, step_idx, jax.lax.stop_gradient(z_t))

            # Update memory buffer using target action (Teacher Forcing)
            m_tp1 = self.action_model.encode_memory(m_t, step_idx, jax.lax.stop_gradient(z_t), a_target)

            # Step dynamics (frozen)
            z_tp1 = jax.lax.stop_gradient(self.forward_dyn(z_t, a_target))

            return (z_tp1, m_tp1), (logits_pred, logits_target)

        scan_inputs = (ref_video[1:], jnp.arange(1, T))
        _, (logits_preds, logits_targets) = jax.lax.scan(scan_step, (z_init, m_init), scan_inputs)

        return logits_preds, logits_targets

    # -------------------------------------------------------------------------------------
    # INFERENCE ROLLOUT: Context-Conditioned Autoregressive Generation
    # -------------------------------------------------------------------------------------
    def inference_rollout(self, ref_video, coords_grid, context_ratio=0.0):
        T = ref_video.shape[0]
        init_frame = ref_video[0]
        
        z_init = self.encoder(jnp.transpose(init_frame, (2, 0, 1)))
        m_init = self.action_model.reset_memory(T)

        @eqx.filter_checkpoint
        def scan_step(carry, scan_inputs):
            z_t, m_t = carry
            o_tp1, step_idx = scan_inputs

            time_coord = jnp.array([(step_idx-1)/(T-1)], dtype=z_t.dtype)
            coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
            pred_out = self.render_frame(z_t, coords_grid_t)

            # Determine if we are still in the context window
            is_context = (step_idx / T) < context_ratio

            # Conditionally choose action: IDM (Teacher Forcing) vs GCM (Autoregressive)
            a_t = jax.lax.cond(
                is_context,
                lambda: self.action_model.inverse_dynamics(
                    z_t, 
                    self.encoder(jnp.transpose(o_tp1, (2, 0, 1)))
                ),
                lambda: self.action_model.decode_memory(m_t, step_idx, z_t)
            )

            m_tp1 = self.action_model.encode_memory(m_t, step_idx, z_t, a_t)
            z_tp1 = self.forward_dyn(z_t, a_t)

            return (z_tp1, m_tp1), (a_t, z_t, pred_out)

        # Pass the future ground truth frames into the scan so the IDM can use them
        scan_inputs = (jnp.concatenate([ref_video[1:], jnp.zeros_like(ref_video[:1])], axis=0), jnp.arange(1, T+1))
        _, (actions, pred_latents, pred_video) = jax.lax.scan(scan_step, (z_init, m_init), scan_inputs)
        
        return actions, pred_latents, pred_video

#%% Cell 4: Phase 1 Training (Base Model & IDM)
if TRAIN_PHASE_1:
    print(f"\n噫 [PHASE 1] Starting Base Training (IDM + FDM + maybe Enc) -> Saving to {run_path}")
    key, subkey = jax.random.split(key)

    model_p1 = WARP(
        root_width=CONFIG["root_width"], root_depth=CONFIG["root_depth"],
        num_freqs=CONFIG["num_fourier_freqs"], frame_shape=(H, W, C), 
        lam_dim=CONFIG["lam_space"], mem_dim=CONFIG["mem_space"],
        split_forward=CONFIG["split_forward"], key=subkey, phase=1
    )
    
    print(f"Total Trainable Parameters in Phase 1 WARP: {count_trainable_params(model_p1)}")

    optimizer_p1 = optax.chain(
        optax.adam(CONFIG["p1_learning_rate"]),
        optax.contrib.reduce_on_plateau(
            patience=CONFIG["lr_patience"], cooldown=CONFIG["lr_cooldown"],
            factor=CONFIG["lr_factor"], rtol=CONFIG["lr_rtol"],
            accumulation_size=CONFIG["lr_accum_size"], min_scale=CONFIG["lr_min_scale"]
        )
    )
    opt_state_p1 = optimizer_p1.init(eqx.filter(model_p1, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step_p1(model, opt_state, keys, in_videos, coords_grid):
        def loss_fn(m):
            k_aug, k_init = jax.random.split(keys[0], 2)
            ref_videos = in_videos
            
            # 1. Reverse Video Augmentation
            if CONFIG["reverse_video_aug"]:
                do_reverse = jax.random.bernoulli(k_aug, 0.5, shape=(ref_videos.shape[0],))
                ref_videos = jax.vmap(lambda rev, vid: jax.lax.cond(rev, lambda v: jnp.flip(v, axis=0), lambda v: v, vid))(do_reverse, ref_videos)

            ## 1.5 Static Augmentation 
            if CONFIG["static_video_aug"]:
                add_to_front = jax.random.bernoulli(k_init, 0.5, shape=(ref_videos.shape[0],))
                nb_frames = ref_videos.shape[1]
                repeat_frames = nb_frames // 4
                ref_videos = jax.vmap(lambda add_front, vid: jax.lax.cond(add_front, 
                                                                            # Add static frames at the front
                                                                            lambda v_in: jnp.concatenate([jnp.repeat(v_in[:1], repeats=repeat_frames, axis=0), v_in[1:nb_frames-repeat_frames+1]], axis=0),
                                                                            # Add static frames at the back
                                                                            lambda v_in: jnp.concatenate([v_in[:nb_frames-repeat_frames], jnp.repeat(v_in[nb_frames-repeat_frames:nb_frames-repeat_frames+1], repeats=repeat_frames, axis=0)], axis=0),
                                                                                vid))(add_to_front, ref_videos)


            batched_fn = jax.vmap(m.phase1_forward, in_axes=(0, None, None))
            actions, (pred_lats, gt_lats), pred_videos = batched_fn(ref_videos, coords_grid, False)

            # # 2. SSIM + MSE Loss
            # def ssim(x, y, data_range=1.0):
            #     C1, C2 = (0.01 * data_range)**2, (0.03 * data_range)**2
            #     mu_x, mu_y = jnp.mean(x), jnp.mean(y)
            #     sigma_x, sigma_y = jnp.var(x), jnp.var(y)
            #     sigma_xy = jnp.mean((x - mu_x) * (y - mu_y))
            #     return ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
            
            # # ssim_loss = 1.0 - jnp.mean(jax.vmap(jax.vmap(ssim))(pred_videos, ref_videos))
            # mse_loss = jnp.mean((pred_videos - ref_videos)**2)
            
            mse_loss = jnp.mean((pred_lats[:, :-1] - gt_lats[:, :-1])**2)       ## @TODO: assumes a well-pretrained encoder

            # Choose loss combination
            # rec_loss = ssim_loss + CONFIG["mse_weight"] * mse_loss
            rec_loss = mse_loss

            # 3. L1 Continuous Action Regularisation
            action_l1_loss = 0.0
            if not CONFIG["discrete_actions"] and CONFIG["action_l1_reg"] > 0:
                action_l1_loss = CONFIG["action_l1_reg"] * jnp.mean(jnp.abs(actions))

            # 4. Aux Autoencoding Loss
            ae_loss = 0.0
            if CONFIG["aux_encoder_loss"]:
                indices = jax.random.choice(k_init, ref_videos.shape[1], shape=(CONFIG["aux_loss_num_steps"],), replace=False)
                ref_videos_enc = jnp.transpose(ref_videos[:, indices], (0, 1, 4, 2, 3))
                target_thetas = jax.vmap(jax.vmap(m.encoder))(ref_videos_enc)
                coords_grid_t0 = jnp.concatenate([jnp.zeros_like(coords_grid[..., :1]), coords_grid], axis=-1)
                batched_render = jax.vmap(jax.vmap(lambda theta: m.render_frame(theta, coords_grid_t0)))
                decoded_pixels = batched_render(target_thetas)
                ae_loss = jnp.mean((decoded_pixels - ref_videos[:, indices])**2)

            total_loss = rec_loss + action_l1_loss + CONFIG["aux_loss_weight"] * ae_loss
            return total_loss

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer_p1.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array), value=loss_val)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    all_losses_p1 = []
    lr_scales_p1 = []
    start_time = time.time()

    sample_videos_vis = next(iter(train_loader))[:1]

    for epoch in range(CONFIG["p1_nb_epochs"]):
        epoch_losses = []
        for batch_idx, batch_videos in enumerate(train_loader):
            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, batch_videos.shape[0])
            model_p1, opt_state_p1, loss = train_step_p1(model_p1, opt_state_p1, batch_keys, batch_videos, coords_grid)
            epoch_losses.append(loss)

            lr_scale_val = optax.tree_utils.tree_get(opt_state_p1, "scale")
            lr_scales_p1.append(lr_scale_val)

        all_losses_p1.extend(epoch_losses)

        if not SINGLE_BATCH and ((epoch+1) % CONFIG["print_every"] == 0 or (epoch+1) == CONFIG["p1_nb_epochs"]):
            avg_loss = np.mean(epoch_losses)
            print(f"Phase 1 - Epoch {epoch+1}/{CONFIG['p1_nb_epochs']} - Avg Loss: {avg_loss:.6f} - LR Scale: {lr_scale_val:.4f}", flush=True)

        ## Save checkpoints and visualizations
        if (epoch+1) % (CONFIG["p1_nb_epochs"]//2) == 0 or (epoch+1) == CONFIG["p1_nb_epochs"]:
            eqx.tree_serialise_leaves(artefacts_path / f"model_phase1_epoch{epoch+1}.eqx", model_p1)

        if (epoch+1) % (CONFIG["p1_nb_epochs"]//10) == 0 or (epoch+1) == CONFIG["p1_nb_epochs"]:
            _, _, pred_videos = jax.vmap(model_p1.phase1_forward, in_axes=(0, None, None))(sample_videos_vis, coords_grid, True)
            for i in range(pred_videos.shape[0]):
                plot_pred_ref_videos_rollout(pred_videos[i], sample_videos_vis[i], f"Pred", plots_path / f"p1_vis_epoch{epoch+1}_sample{i}.png")

    print("\nPhase 1 Wall time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    
    # Save Phase 1 artifacts
    eqx.tree_serialise_leaves(artefacts_path / "model_phase1_final.eqx", model_p1)
    np.save(artefacts_path / "loss_history_p1.npy", np.array(all_losses_p1))
    np.save(artefacts_path / "lr_history_p1.npy", np.array(lr_scales_p1))

    # Phase 1 Dashboard
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(all_losses_p1, color='teal', alpha=0.8, label="Phase 1 Loss")
    ax1.set_yscale('log')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss", color='teal')
    ax2 = ax1.twinx()  
    ax2.plot(lr_scales_p1, color='crimson', linewidth=2, label="LR Scale")
    plt.title("Phase 1: Base Model Training Loss")
    fig.tight_layout()
    plt.savefig(plots_path / "p1_loss_history.png")
    plt.show()

#%% Cell 5: Phase 2 Training (GCM Matching)
if TRAIN_PHASE_2:
    print(f"\n噫 [PHASE 2] Starting GCM Training (Action Matching) -> Saving to {run_path}")
    key, subkey = jax.random.split(key)

    # 1. Initialize fresh Phase 2 model (with GCM enabled)
    model_p2 = WARP(
        root_width=CONFIG["root_width"], root_depth=CONFIG["root_depth"],
        num_freqs=CONFIG["num_fourier_freqs"], frame_shape=(H, W, C), 
        lam_dim=CONFIG["lam_space"], mem_dim=CONFIG["mem_space"],
        split_forward=CONFIG["split_forward"], key=subkey, phase=2
    )

    # 2. Transplant weights from Phase 1
    print("踏 Loading Base weights from Phase 1...")
    dummy_p1 = WARP(
        root_width=CONFIG["root_width"], root_depth=CONFIG["root_depth"],
        num_freqs=CONFIG["num_fourier_freqs"], frame_shape=(H, W, C), 
        lam_dim=CONFIG["lam_space"], mem_dim=CONFIG["mem_space"],
        split_forward=CONFIG["split_forward"], key=subkey, phase=1
    )
    # Load from the fresh save or the explicit P1 path
    load_path = artefacts_path / "model_phase1_final.eqx" if TRAIN_PHASE_1 else P1_LOAD_PATH
    dummy_p1 = eqx.tree_deserialise_leaves(load_path, dummy_p1)
    
    model_p2 = eqx.tree_at(lambda m: m.encoder, model_p2, dummy_p1.encoder)
    model_p2 = eqx.tree_at(lambda m: m.forward_dyn, model_p2, dummy_p1.forward_dyn)
    model_p2 = eqx.tree_at(lambda m: m.theta_base, model_p2, dummy_p1.theta_base)
    model_p2 = eqx.tree_at(lambda m: m.action_model.idm, model_p2, dummy_p1.action_model.idm)

    # 3. Partition parameters: Freeze everything except GCM
    # First, create a mask where absolutely everything is False
    filter_spec = jax.tree_util.tree_map(lambda _: False, model_p2)
    
    # Next, compute the proper gradient mask (True for float arrays) using the ACTUAL model
    gcm_mask = jax.tree_util.tree_map(eqx.is_inexact_array, model_p2.action_model.gcm)
    
    # Graft the active GCM mask into our all-False filter_spec
    filter_spec = eqx.tree_at(lambda m: m.action_model.gcm, filter_spec, gcm_mask)
    
    diff_model_p2, static_model_p2 = eqx.partition(model_p2, filter_spec)

    print(f"Trainable Parameters in Phase 2 (GCM only): {count_trainable_params(diff_model_p2)}")

    optimizer_p2 = optax.chain(
        optax.adam(CONFIG["p2_learning_rate"]),
        optax.contrib.reduce_on_plateau(
            patience=CONFIG["lr_patience"], cooldown=CONFIG["lr_cooldown"],
            factor=CONFIG["lr_factor"], rtol=CONFIG["lr_rtol"],
            accumulation_size=CONFIG["lr_accum_size"], min_scale=CONFIG["lr_min_scale"]
        )
    )
    opt_state_p2 = optimizer_p2.init(diff_model_p2)

    @eqx.filter_jit
    def train_step_p2(diff_m, static_m, opt_state, ref_videos):
        def loss_fn(d_model):
            # Recombine model for forward pass
            m = eqx.combine(d_model, static_m)
            
            # Action matching phase forward
            batched_fn = jax.vmap(m.phase2_forward, in_axes=(0,))
            logits_preds, logits_targets = batched_fn(ref_videos)
            
            # L1 Match Loss (GCM matching IDM)
            # total_loss = jnp.mean(jnp.abs(a_preds - a_targets))

            ## Cross-Entropy Loss for discrete actions
            ## Ground thruth actions are argmax of target logits
            gt_actions = jax.lax.stop_gradient(jnp.argmax(logits_targets, axis=-1))     #@TODO: we could just use the softmax, to match the same uncertainties?
            ce_loss = optax.softmax_cross_entropy(logits_preds, jax.nn.one_hot(gt_actions, num_classes=logits_preds.shape[-1]))
            total_loss = jnp.mean(ce_loss)

            return total_loss

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(diff_m)
        updates, opt_state = optimizer_p2.update(grads, opt_state, diff_m, value=loss_val)
        diff_m = eqx.apply_updates(diff_m, updates)
        return diff_m, opt_state, loss_val

    all_losses_p2 = []
    lr_scales_p2 = []
    start_time = time.time()

    for epoch in range(CONFIG["p2_nb_epochs"]):
        epoch_losses = []
        for batch_idx, batch_videos in enumerate(train_loader):
            diff_model_p2, opt_state_p2, loss = train_step_p2(diff_model_p2, static_model_p2, opt_state_p2, batch_videos)
            epoch_losses.append(loss)
            lr_scales_p2.append(optax.tree_utils.tree_get(opt_state_p2, "scale"))

        all_losses_p2.extend(epoch_losses)

        if not SINGLE_BATCH and ((epoch+1) % CONFIG["print_every"] == 0 or (epoch+1) == CONFIG["p2_nb_epochs"]):
            avg_loss = np.mean(epoch_losses)
            print(f"Phase 2 - Epoch {epoch+1}/{CONFIG['p2_nb_epochs']} - Avg Loss: {avg_loss:.6f}", flush=True)

        ## Visualize the Phase 2 predictions
        if (epoch+1) % (CONFIG["p2_nb_epochs"]//10) == 0 or (epoch+1) == CONFIG["p2_nb_epochs"]:
            model_vis = eqx.combine(diff_model_p2, static_model_p2)
            # Generating visualizations mid-training for Phase 2 tracking
            # _, _, pred_videos = jax.vmap(model_vis.phase1_forward, in_axes=(0, None))(sample_videos_vis, coords_grid)
            _, _, pred_videos = jax.vmap(model_vis.inference_rollout, in_axes=(0, None, None))(sample_videos_vis, coords_grid, 0.0)
            for i in range(pred_videos.shape[0]):
                plot_pred_ref_videos_rollout(pred_videos[i], sample_videos_vis[i], f"Pred", plots_path / f"p2_vis_epoch{epoch+1}_sample{i}.png")

    print("\nPhase 2 Wall time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    
    # Save Phase 2 artifacts
    model_final = eqx.combine(diff_model_p2, static_model_p2)
    eqx.tree_serialise_leaves(artefacts_path / "model_phase2_final.eqx", model_final)
    np.save(artefacts_path / "loss_history_p2.npy", np.array(all_losses_p2))
    np.save(artefacts_path / "lr_history_p2.npy", np.array(lr_scales_p2))

    # Phase 2 Dashboard
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(all_losses_p2, color='darkorange', alpha=0.8, label="Phase 2 Loss")
    ax1.set_yscale('log')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss", color='darkorange')
    ax2 = ax1.twinx()  
    ax2.plot(lr_scales_p2, color='royalblue', linewidth=2, label="LR Scale")
    plt.title("Phase 2: GCM Action Matching Loss")
    fig.tight_layout()
    plt.savefig(plots_path / "p2_loss_history.png")
    plt.show()

#%% Cell 6: Evaluation & Plotting
print("\n=== Evaluating Phase 2 Model ===")

# If we skipped training, load the Phase 2 model
if not TRAIN_PHASE_2:
    print(f"踏 Loading completed Phase 2 WARP model from {artefacts_path}")
    key, subkey = jax.random.split(key)
    model_final = WARP(
        root_width=CONFIG["root_width"], root_depth=CONFIG["root_depth"],
        num_freqs=CONFIG["num_fourier_freqs"], frame_shape=(H, W, C), 
        lam_dim=CONFIG["lam_space"], mem_dim=CONFIG["mem_space"],
        split_forward=CONFIG["split_forward"], key=subkey, phase=2
    )
    model_final = eqx.tree_deserialise_leaves(artefacts_path / "model_phase2_final.eqx", model_final)

@eqx.filter_jit
def evaluate(m, batch, coords, context_ratio):
    batched_fn = jax.vmap(m.inference_rollout, in_axes=(0, None, None))
    return batched_fn(batch, coords, context_ratio)

testing_subset = MiniGridDataset(test_arrays)
test_loader = DataLoader(testing_subset, batch_size=CONFIG["batch_size"]*10, shuffle=False, collate_fn=numpy_collate, drop_last=False)
sample_batch = next(iter(test_loader))

print("Batch shape for evaluation:", sample_batch.shape, flush=True)

# Padding shape up to 12 for MiniGrid testing if needed
pad_length = 12 - sample_batch.shape[1]
if pad_length > 0:
    sample_batch = jnp.concatenate([sample_batch, np.zeros((sample_batch.shape[0], pad_length, H, W, C), dtype=sample_batch.dtype)], axis=1)

# Evaluation using context-conditioned autoregressive rollout
final_actions, _, final_videos = evaluate(model_final, sample_batch, coords_grid, CONFIG["inf_context_ratio"])

if CONFIG["use_nll_loss"]:
    print(f"Final Predicted Video Mean Pixel Value Range: min={final_videos[...,:C].min():.4f}, max={final_videos[...,:C].max():.4f}")
    print(f"Final Predicted Video Std Pixel Value Range: min={final_videos[...,C:].min():.4f}, max={final_videos[...,C:].max():.4f}")

#%% MiniGrid Visualization Suite
test_seq_id = np.random.randint(0, sample_batch.shape[0])
print(f"Plotting rollout for test sequence ID: {test_seq_id}")

plot_pred_ref_videos_rollout(
    final_videos[test_seq_id], 
    sample_batch[test_seq_id, 1:], # Ground truth targets (shifted by 1 for causality)
    title=f"Pred", 
    save_name=plots_path / f"inference_forecast_rollout_seq{test_seq_id}.png"
)

#%%
if TRAIN_PHASE_1 or TRAIN_PHASE_2:
    os.system(f"cp -r nohup.log {run_path}/nohup.log")

#%% Latent Action Heatmap
print("Predicted Latent Actions for the selected test sequence:")

plt.figure(figsize=(12, 6))
# Transpose to have dimensions on Y axis and time on X axis
sns.heatmap(final_actions[test_seq_id].T, cmap="coolwarm", center=0, 
            annot=True, fmt=".2f", cbar_kws={'label': 'Latent Action Value'}, annot_kws={'size': 8})
plt.xlabel("Time Step (t)")
plt.ylabel(f"Latent Dimension (0-{model_final.lam_dim-1})")
plt.title(f"Latent Action Heatmap for Sequence {test_seq_id}")
plt.tight_layout()
plt.savefig(plots_path / f"action_heatmap_seq{test_seq_id}.png")
plt.show()

#%% Action Dimension Variance
# Flatten across batch and time: shape becomes (B * T, lam_dim)
all_actions_flat = final_actions.reshape(-1, model_final.lam_dim)
action_variances = np.var(all_actions_flat, axis=0)

plt.figure(figsize=(10, 4))
plt.bar(range(model_final.lam_dim), action_variances, color='teal')
plt.xlabel("Latent Dimension")
plt.ylabel("Variance across all data")
plt.title("Latent Action Dimension Importance (Variance)")
plt.xticks(range(model_final.lam_dim))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "action_dimension_variance.png")
plt.show()
