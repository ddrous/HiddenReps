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
    "nb_epochs": 200*1,        ## 20 hours !
    "print_every": 10,
    "batch_size": 2 if SINGLE_BATCH else 256*1,
    "learning_rate": 1e-4 if USE_NLL_LOSS else 1e-4,
    "reverse_video_aug": True,
    "static_video_aug": True,
    "action_l1_reg": 0.00,     # L1 regularisation on continuous actions to limit info capacity
    "p_forcing": 0.5,
    "inf_context_ratio": 0.5,
    "use_nll_loss": USE_NLL_LOSS,
    "aux_encoder_loss": False,
    "aux_loss_weight": 1,
    "aux_loss_num_steps": 4,

    # --- Architecture Params ---
    "lam_space": 4,
    "mem_space": 128*2,
    "icl_decoding": True,
    "discrete_actions": False,
    "split_forward": True,
    "root_width": 12,
    "root_depth": 5,
    "num_fourier_freqs": 6,
    "use_time_in_root": False,
    "pretrain_encoder": False, 

    # --- Plateau Scheduler Config ---
    "lr_patience": 400,      
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
    if TRAIN:
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        run_path = Path(base_dir) / timestamp
        run_path.mkdir(parents=True, exist_ok=True)

        current_script = Path(__file__)
        if current_script.exists():
            # shutil.copy(current_script, run_path / "main.py")
            ## Maintain the name
            shutil.copy(current_script, run_path / current_script.name)
        
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

    ## Manually load train and test splits to have more control over batching and shuffling
    mov_mnist_arrays = np.load(data_path + "/MovingMNIST/mnist_test_seq.npy")
    print(f"Original loaded MovingMNIST shape: {mov_mnist_arrays.shape} (T, N, H, W)")
    ## Split, 8000 train, 2000 test
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
        fig, axes = plt.subplots(2, nb_frames, figsize=(2*nb_frames, 2*2))
        indices_to_plot = list(np.arange(0, nb_frames))

        for i, idx in enumerate(indices_to_plot):
            video_to_plot = video[idx] if not rescale else (video[idx] + 1.0) / 2.0
            sbimshow(video_to_plot, title=f"Pred t={idx}", ax=axes[0, i])
            # Handle offsets for plotting ref against predicted properly (sometimes diff by 1)
            ref_idx = min(idx, ref_video.shape[0]-1)
            sbimshow(ref_video[ref_idx], title=f"Ref t={ref_idx}", ax=axes[1, i])
    else:
        fig, axes = plt.subplots(3, nb_frames, figsize=(2*nb_frames, 3*2))
        indices_to_plot = list(np.arange(0, nb_frames))

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


#%% Visualize Augmentations
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
add_to_front = jnp.array([False, False, True, False]) # Only matters if do_static is True

# --- Apply Reverse Augmentation ---
aug_batch = jax.vmap(lambda rev, vid: jax.lax.cond(
    rev, 
    lambda v: jnp.flip(v, axis=0), 
    lambda v: v, 
    vid
))(do_reverse, test_batch)

# --- Apply Static Augmentation (With Bug Fix) ---
nb_frames = aug_batch.shape[1]
repeat_frames = nb_frames // 4

def static_aug(add_front, v):
    return jax.lax.cond(
        add_front, 
        # FIX: Take the remaining frames from the START of the sequence, not the end
        lambda v_in: jnp.concatenate([jnp.repeat(v_in[:1], repeats=repeat_frames, axis=0), v_in[1:nb_frames-repeat_frames+1]], axis=0),
        # False branch was already correct
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
        
        # Handle 1-channel grayscale for RGB imshow
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
            
        ax.imshow(np.clip(img, 0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Row and Column labels
        if t == 0:
            ax.set_ylabel(row_titles[row], fontsize=14, rotation=0, labelpad=60, ha='center', va='center', fontweight='bold')
        if row == 0:
            ax.set_title(f"t={t}")

plt.suptitle("Effect of Temporal Video Augmentations", y=1.00, fontsize=20, fontweight='bold')
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

class LAM(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, dyn_dim, lam_dim, key):
        self.mlp = eqx.nn.MLP(dyn_dim * 2, lam_dim, width_size=dyn_dim*1, depth=3, key=key)
        
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
            self.mlp_A = eqx.nn.MLP(dyn_dim, dyn_dim, width_size=dyn_dim*1, depth=5, key=k1)
            self.mlp_B = eqx.nn.MLP(lam_dim, dyn_dim, width_size=dyn_dim*1, depth=5, key=k2)
            self.giant_mlp = None
        else:
            self.mlp_A = None
            self.mlp_B = None
            self.giant_mlp = eqx.nn.MLP(dyn_dim + lam_dim, dyn_dim, width_size=dyn_dim*1, depth=5, key=k3)

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


# ==============================================================================
# --- TRANSFORMER MEMORY MODULE (From MiniGrid Implementation) ---
# ==============================================================================
class TransformerBlock(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(self, d_model, num_heads, key):
        k1, k2 = jax.random.split(key)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=d_model,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=k1
        )
        self.mlp = eqx.nn.MLP(d_model, d_model, width_size=d_model * 4, depth=1, key=k2)
        self.ln1 = eqx.nn.LayerNorm(d_model)
        self.ln2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, mask):
        # Pre-LN architecture
        x_norm = jax.vmap(self.ln1)(x)
        attn_out = self.attn(x_norm, x_norm, x_norm, mask=mask)
        x = x + attn_out
        x = x + jax.vmap(self.mlp)(jax.vmap(self.ln2)(x))
        return x


class MemoryModule(eqx.Module):
    """
    Autoregressive Transformer Memory Module for Latent Actions.
    Supports continuous actions (MLP) or discrete learned embeddings via In-Context Learning (ICL).
    """
    d_model: int
    max_len: int
    pos_emb: jax.Array
    blocks: tuple
    proj_in: eqx.nn.Linear
    
    # Static fields for control flow and shapes
    discrete_actions: bool = eqx.field(static=True)
    lam_dim: int = eqx.field(static=True)
    icl_decoding: bool = eqx.field(static=True)
    
    # Standard decoding components
    action_mlp: Optional[eqx.nn.MLP]
    
    # ICL decoding components
    action_embedding: Optional[eqx.nn.Embedding]
    output_proj: Optional[eqx.nn.Linear]

    def __init__(self, lam_dim, mem_dim, latent_dim, key, max_len=12, num_heads=4, num_blocks=4, num_actions=4):
        self.max_len = max_len
        self.discrete_actions = CONFIG["discrete_actions"]
        self.icl_decoding = CONFIG["icl_decoding"]
        self.lam_dim = lam_dim
        ## d_model is the closest power of 2 after latent_dim + lam_dim for better transformer performance
        # self.d_model = 2 ** int(jnp.ceil(jnp.log2(latent_dim + lam_dim)))
        self.d_model = mem_dim
        
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        
        # 1. Project concatenated [state, action] tokens into model dimension
        self.proj_in = eqx.nn.Linear(latent_dim + lam_dim, self.d_model, key=k1)
        
        # 2. Learnable Positional Embeddings
        self.pos_emb = jax.random.normal(k2, (max_len, self.d_model)) * 0.02
        
        # 3. Causal Transformer Blocks
        block_keys = jax.random.split(k3, num_blocks)
        self.blocks = tuple(TransformerBlock(self.d_model, num_heads, bk) for bk in block_keys)

        # 4. Decoder Heads
        if self.icl_decoding:
            self.action_mlp = None
            if self.discrete_actions:
                # Learned discrete actions (num_actions -> lam_dim)
                self.action_embedding = eqx.nn.Embedding(num_actions, lam_dim, key=k5)
                # Projects transformer output to action logits
                self.output_proj = eqx.nn.Linear(self.d_model, num_actions, key=k6)
            else:
                # Continuous actions with ICL decoding still use an MLP head
                self.action_embedding = None
                self.output_proj = eqx.nn.Linear(self.d_model, lam_dim, key=k6)
        else:
            self.action_mlp = eqx.nn.MLP(self.d_model + latent_dim, lam_dim, width_size=self.d_model * 2, depth=3, key=k4)
            self.action_embedding = None
            self.output_proj = None

    def reset(self, T):
        """Returns an empty fixed-size buffer for JAX scan"""
        return jnp.zeros((T, self.d_model))

    def encode(self, buffer, step_idx, z, a):
        """
        Dynamically injects the new token at the correct sequence index.
        """
        token = self.proj_in(jnp.concatenate([z, a], axis=-1))
        return buffer.at[step_idx - 1].set(token)

    def decode(self, buffer, step_idx, z_current):
        """
        Computes the causal context from the buffer and predicts the current action.
        """
        T = buffer.shape[0]
        
        if self.icl_decoding:
            # --- ICL Decoding Path ---
            
            # 1. Create query token: [z_current, zeros]
            zero_action = jnp.zeros((self.lam_dim,), dtype=z_current.dtype)
            query_token = self.proj_in(jnp.concatenate([z_current, zero_action], axis=-1))
            
            # 2. Inject query token temporally into the buffer for the current step
            temp_buffer = buffer.at[step_idx - 1].set(query_token)
            
            # 3. Apply positional embeddings and causal attention
            x = temp_buffer + self.pos_emb[:T]
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))
            
            for block in self.blocks:
                x = block(x, mask)

            # 4. Extract context specifically at the current query step
            context = x[step_idx - 1]

            if self.discrete_actions:
                # 1. Get the raw logits from the transformer context
                logits = self.output_proj(context)

                # 2. Calculate the soft probabilities (this gives us our gradients!)
                soft_probs = jax.nn.softmax(logits, axis=-1)

                # 3. Calculate the strictly hard, discrete one-hot vector (this is what we want for the forward pass)
                hard_idx = jnp.argmax(logits, axis=-1)
                hard_probs = jax.nn.one_hot(hard_idx, num_classes=logits.shape[-1])

                # 4. The Magic STE Formula:
                ste_probs = soft_probs + jax.lax.stop_gradient(hard_probs - soft_probs)

                # 5. Lookup the embedding
                action = jnp.dot(ste_probs, self.action_embedding.weight)
            else:
                # If not using discrete actions, directly project to continuous action space
                action = self.output_proj(context)

            return action

        else:
            # --- Original Continuous MLP Decoding Path ---
            def compute_context():
                x = buffer + self.pos_emb[:T]
                mask = jnp.tril(jnp.ones((T, T), dtype=bool))
                for block in self.blocks:
                    x = block(x, mask)
                return x[step_idx - 2]
                
            context = jax.lax.cond(
                step_idx > 1,
                compute_context,
                lambda: jnp.zeros(self.d_model)
            )
            
            return self.action_mlp(jnp.concatenate([context, z_current], axis=-1))


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

        # Set up JEPA dynamics components. Retaining MovingMNIST larger hidden_width
        self.encoder = CNNEncoder(in_channels=C, out_dim=self.d_theta, spatial_shape=(H, W), key=k_enc, hidden_width=64, depth=4)

        if CONFIG.get("pretrain_encoder", False):
            ## Load the pretrained encoder weights from the autoencoding phase
            try:
                self.encoder, self.theta_base = eqx.tree_deserialise_leaves("movingmnist_enc.eqx", (self.encoder, self.theta_base))
            except:
                print("Warning: could not load movingmnist_enc.eqx")

        self.lam = LAM(self.d_theta, lam_dim, key=k_lam)
        self.forward_dyn = ForwardDynamics(self.d_theta, lam_dim, split_forward, key=k_fwd)

        self.mem_dim = mem_dim
        # self.memory = MemoryModule(self.lam_dim, self.mem_dim, self.d_theta, key=k_mem, max_len=20)
        self.memory = MemoryModule(self.lam_dim, self.mem_dim, self.d_theta, key=k_mem, max_len=20, num_heads=4, num_blocks=4, num_actions=4)

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
        
        # Add the base weights before rendering
        if not CONFIG.get("pretrain_encoder", False):
            theta = theta_offset + self.theta_base
        else:
            theta = theta_offset + jax.lax.stop_gradient(self.theta_base)

        pred_flat = self.render_pixels(theta, flat_coords)
        return pred_flat.reshape(H, W, -1)

    def _get_preds_single(self, ref_video, p_forcing, key, coords_grid, context_ratio):
        T = ref_video.shape[0]
        init_frame = ref_video[0]

        # 1. Initialize offset from first frame
        z_init = self.encoder(jnp.transpose(init_frame, (2, 0, 1)))
        if CONFIG.get("pretrain_encoder", False):
            z_init = jax.lax.stop_gradient(z_init)

        # 2. Initialize fixed buffer for Transformer memory
        m_init = self.memory.reset(T)

        is_context_init = False

        # 3. Add the Equinox checkpointing decorator here
        @eqx.filter_checkpoint
        def scan_step(carry, scan_inputs):
            z_t, m_t, was_context = carry
            o_tp1, step_idx = scan_inputs

            # --- Rendering INSIDE the checkpointed step ---
            time_coord = jnp.array([(step_idx-1)/ (T-1)], dtype=z_t.dtype)
            coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)

            pred_out = self.render_frame(z_t, coords_grid_t)

            # Determine if we are forcing towards ground truth this step
            # is_context = (step_idx / T) < context_ratio
            is_context = True

            # ONLY compute encoder and LAM when use_gt is True
            a_t = jax.lax.cond(
                is_context,
                lambda: self.lam(
                    z_t, 
                    jax.lax.stop_gradient(self.encoder(jnp.transpose(o_tp1, (2, 0, 1))))
                ),
                lambda: self.memory.decode(m_t, step_idx, z_t)
            )

            ## Encode into the memory with C
            m_tp1 = self.memory.encode(m_t, step_idx, z_t, a_t)

            # SINGLE forward dynamics call handles both the forced and autoregressive paths
            z_tp1 = self.forward_dyn(z_t, a_t)

            return (z_tp1, m_tp1, is_context), (a_t, z_t, pred_out)

        scan_inputs = (jnp.concatenate([ref_video[1:], ref_video[-1:]], axis=0), jnp.arange(1, T+1))
        
        # 4. Execute the scan as normal, collecting both latents and rendered frames
        _, (pred_actions, pred_latents, pred_video) = jax.lax.scan(scan_step, (z_init, m_init, is_context_init), scan_inputs)

        return pred_actions, pred_latents, pred_video


    def __call__(self, ref_videos, p_forcing, keys, coords_grid, inf_context_ratio, precompute_ref_diffs=False):
        is_single = (ref_videos.ndim == 4)
        if is_single:
            ref_videos = ref_videos[None, ...]
            keys = keys[None, ...] if keys.ndim == 1 else keys
            
        batched_fn = jax.vmap(self._get_preds_single, in_axes=(0, None, 0, None, None))
        pred_actions, pred_latents, pred_videos = batched_fn(ref_videos, p_forcing, keys, coords_grid, inf_context_ratio)
        
        if is_single:
            return pred_latents[0], pred_videos[0]
        return pred_actions, pred_latents, pred_videos

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
            
            # Use randomized uniform context ratio for training, mirroring minigrid
            context_ratio = jax.random.uniform(k_full, minval=0.0, maxval=1.0)
            # context_ratio = CONFIG["inf_context_ratio"]
            pred_actions, _, pred_videos = m(ref_videos, p_forcing, keys, coords_grid, context_ratio, precompute_ref_diffs=False)

            ## --- 1. LATENT (WEIGHT-SPACE) DYNAMICS LOSS (Primary) ---
            rec_loss = jnp.mean((pred_videos - ref_videos)**2)

            # --- 2. AUTOENCODING LOSS (Auxiliary) ---
            if CONFIG["aux_encoder_loss"]:
                indices = jax.random.choice(k_init, ref_videos.shape[1], shape=(CONFIG["aux_loss_num_steps"],), replace=False)

                # Encode ground truth to target thetas
                ref_videos_enc = jnp.transpose(ref_videos[:, indices], (0, 1, 4, 2, 3))
                target_thetas = jax.vmap(jax.vmap(m.encoder))(ref_videos_enc)

                # Render Target Thetas -> Match GT Pixels
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

            # action_loss = jnp.mean(jnp.abs(pred_actions))
            # total_loss = total_loss + 0.01 * action_loss

            return total_loss

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
        
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array), value=loss_val
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    @eqx.filter_jit
    def compute_p_forcing(epoch, schedule="linear", start=1.0, end=0.0):
        if schedule == "linear":
            p = start + (end - start) * (epoch / CONFIG["nb_epochs"])
            return p
        elif schedule == "constant":
            return start
        elif schedule == "exponential":
            decay_rate = (end / start) ** (1 / CONFIG["nb_epochs"])
            p = start * (decay_rate ** epoch)
            return p
        elif schedule == "step":
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
            epoch_losses.append(loss)

            current_scale = optax.tree_utils.tree_get(opt_state, "scale")
            lr_scales.append(current_scale)

        all_losses.extend(epoch_losses)

        if not SINGLE_BATCH and ((epoch+1) % CONFIG["print_every"] == 0 or (epoch+1) == CONFIG["nb_epochs"] - 1):
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1}/{CONFIG['nb_epochs']} - Avg Loss: {avg_epoch_loss:.6f} - LR Scale: {current_scale:.4f}", flush=True)

        if epoch in [4, CONFIG["nb_epochs"]//2, 2*CONFIG["nb_epochs"]//3]:
            eqx.tree_serialise_leaves(artefacts_path / f"model_ep{epoch+1}.eqx", model)

        if (epoch+1) % (max(CONFIG["nb_epochs"]//10, 1)) == 0:
            val_keys = jax.random.split(key, sample_batch.shape[0])
            _, _, val_videos = evaluate(model, sample_batch, 0.0, val_keys, coords_grid, CONFIG["inf_context_ratio"], precompute_ref_diffs=False)
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

#%% Evaluate on Testing Dataset
testing_subset = MovingMNISTDataset(test_arrays)
test_loader = DataLoader(testing_subset, batch_size=CONFIG["batch_size"]*10, shuffle=False, collate_fn=numpy_collate, drop_last=False)
sample_batch = next(iter(test_loader))

print("Batch shape for evaluation:", sample_batch.shape, flush=True)

# Padding shape up to 20 for Moving MNIST testing if needed
pad_length = 20 - sample_batch.shape[1]
if pad_length > 0:
    sample_batch = jnp.concatenate([sample_batch, np.zeros((sample_batch.shape[0], pad_length, H, W, C), dtype=sample_batch.dtype)], axis=1)

val_keys = jax.random.split(key, sample_batch.shape[0])
final_actions, _, final_videos = evaluate(model, sample_batch, 0.0, val_keys, coords_grid, 2.0, precompute_ref_diffs=False)

if CONFIG["use_nll_loss"]:
    print(f"Final Predicted Video Mean Pixel Value Range: min={final_videos[...,:C].min():.4f}, max={final_videos[...,:C].max():.4f}")
    print(f"Final Predicted Video Std Pixel Value Range: min={final_videos[...,C:].min():.4f}, max={final_videos[...,C:].max():.4f}")

#%% Generate final forecast rollout
test_seq_id = np.random.randint(0, sample_batch.shape[0])
print(f"\nGenerating final forecast rollout visualization for test sequence ID: {test_seq_id}")

plot_pred_ref_videos_rollout(
    final_videos[test_seq_id], 
    sample_batch[test_seq_id],
    title=f"Pred", 
    save_name=f"inference_forecast_rollout_seq{test_seq_id}.png"
)

# %% Save nohup
os.system(f"cp -r nohup.log {run_path}/nohup.log")


# %%

#%% Generate final forecast rollout
test_seq_id = np.random.randint(0, sample_batch.shape[0])
test_seq_id = 203
print(f"\nGenerating final forecast rollout visualization for test sequence ID: {test_seq_id}")

plot_pred_ref_videos_rollout(
    final_videos[test_seq_id], 
    sample_batch[test_seq_id],
    title=f"Pred", 
    save_name=f"inference_forecast_rollout_seq{test_seq_id}.png"
)




#%% 1. Action Variance (Finding the Joystick Dimensions)
all_actions_flat = final_actions.reshape(-1, model.lam_dim)
action_variances = np.var(all_actions_flat, axis=0)

plt.figure(figsize=(10, 4))
plt.bar(range(model.lam_dim), action_variances, color='teal')
plt.xlabel("Latent Dimension")
plt.ylabel("Variance across all data")
plt.title("Latent Action Dimension Importance (Variance)")
plt.xticks(range(model.lam_dim))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "action_dimension_variance_mnist.png")
plt.show()

# Extract the top 4 most active dimensions
top_dims = np.argsort(action_variances)[-4:][::-1]
print(f"Top 4 most active latent dimensions: {top_dims}")

#%% 2. Continuous Action Trajectories over Time
seq_actions = final_actions[test_seq_id] # Shape: (T, lam_dim)
T_steps = seq_actions.shape[0]

plt.figure(figsize=(12, 6))
colors = ['crimson', 'dodgerblue', 'forestgreen', 'darkorange']

for i, dim in enumerate(top_dims):
    plt.plot(range(T_steps), seq_actions[:, dim], marker='o', linewidth=2, 
             color=colors[i], label=f"Dim {dim} (Rank {i+1})")

plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.xlabel("Time Step (t)")
plt.ylabel("Latent Action Value (Velocity Proxy)")
plt.title(f"Continuous Action Evolution for Sequence {test_seq_id}\n(Sudden changes indicate wall bounces)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / f"action_lines_seq{test_seq_id}.png")
plt.show()

#%% 3. Dual Joystick Visualization (Phase Portrait)
# Assume the top 4 dims split cleanly into two 2D joysticks. 
# (Note: The network might entangle them slightly, but PCA/ICA could decouple them if needed. 
# For now, we pair Rank 1 & 2, and Rank 3 & 4).

joy1_x, joy1_y = top_dims[0], top_dims[1]
joy2_x, joy2_y = top_dims[2], top_dims[3]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

def plot_joystick(ax, x_dim, y_dim, title, color_map="Blues"):
    x_vals = seq_actions[:, x_dim]
    y_vals = seq_actions[:, y_dim]
    
    # Plot the path
    ax.plot(x_vals, y_vals, color='gray', alpha=0.5, linestyle='--')
    
    # Scatter points colored by time
    scatter = ax.scatter(x_vals, y_vals, c=range(T_steps), cmap=color_map, s=80, edgecolor='black', zorder=5)
    
    # Annotate time steps
    for t in range(T_steps):
        ax.annotate(str(t), (x_vals[t], y_vals[t]), xytext=(5,5), textcoords="offset points", fontsize=9)
        
    ax.axhline(0, color='black', linewidth=1, alpha=0.3)
    ax.axvline(0, color='black', linewidth=1, alpha=0.3)
    ax.set_xlabel(f"Latent Dim {x_dim}")
    ax.set_ylabel(f"Latent Dim {y_dim}")
    ax.set_title(title)
    return scatter

sc1 = plot_joystick(axes[0], joy1_x, joy1_y, "Joystick 1 (Top Dims 1 & 2)", "Reds")
sc2 = plot_joystick(axes[1], joy2_x, joy2_y, "Joystick 2 (Top Dims 3 & 4)", "Blues")

fig.colorbar(sc1, ax=axes[0], label="Time Step (t)")
fig.colorbar(sc2, ax=axes[1], label="Time Step (t)")

plt.suptitle(f"Continuous Latent Velocity phase portraits for Sequence {test_seq_id}")
plt.tight_layout()
plt.savefig(plots_path / f"joystick_phase_seq{test_seq_id}.png")
plt.show()

#%% 4. Continuous PCA Trajectory
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
actions_2d = pca.fit_transform(all_actions_flat)

plt.figure(figsize=(10, 8))
# Plot the background distribution, colored by magnitude (speed)
magnitudes = np.linalg.norm(all_actions_flat[:, top_dims], axis=1)
scatter = plt.scatter(actions_2d[:, 0], actions_2d[:, 1], c=magnitudes, 
                      cmap='viridis', alpha=0.2, s=15)
plt.colorbar(scatter, label="Velocity Magnitude (Norm of Top 4 Dims)")

# Get the 2D coordinates for just our specific test sequence
seq_actions_2d = pca.transform(final_actions[test_seq_id])

# Plot the trajectory with arrows
plt.plot(seq_actions_2d[:, 0], seq_actions_2d[:, 1], color='red', linewidth=2)
plt.scatter(seq_actions_2d[:, 0], seq_actions_2d[:, 1], c=range(T_steps), cmap='autumn', s=60, edgecolor='black', zorder=5)

for t in range(T_steps):
    plt.annotate(f"t={t}", (seq_actions_2d[t, 0], seq_actions_2d[t, 1]), 
                 textcoords="offset points", xytext=(5,5), ha='center', fontsize=9, fontweight='bold')

plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title(f"Continuous Action Trajectory (PCA) for Sequence {test_seq_id}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / f"action_pca_continuous_seq{test_seq_id}.png")
plt.show()


#%%
import pandas as pd
import seaborn as sns

# Grab the top 4 active dimensions from all sequences
top_actions = all_actions_flat[:, top_dims]

# Put them in a DataFrame for easy plotting
df_actions = pd.DataFrame(
    top_actions, 
    columns=[f"Dim {d} (Rank {i+1})" for i, d in enumerate(top_dims)]
)

# A pairplot plots every dimension against every other dimension
plt.figure(figsize=(12, 12))
sns.pairplot(df_actions, kind="hist", diag_kind="kde", corner=True, 
             plot_kws={'cmap': 'viridis', 'cbar': False})
plt.suptitle("Pairplot of Top 4 Action Dimensions\n(Looking for geometric structure / corners)", y=1.02)
plt.show()

#%%
def plot_action_perturbation(model, ref_video, dim_to_perturb, magnitudes=[-5.0, 0.0, 5.0]):
    T = ref_video.shape[0]
    z_init = model.encoder(jnp.transpose(ref_video[0], (2, 0, 1)))
    
    fig, axes = plt.subplots(len(magnitudes), T, figsize=(40, 2.0 * len(magnitudes)))
    
    for row, mag in enumerate(magnitudes):
        # 1. Autoregressive rollout with our hijacked action
        def scan_step(carry, step_idx):
            z_t, m_t = carry
            
            # Get the natural action the model WANTS to take
            a_t_natural = model.memory.decode(m_t, step_idx, z_t)
            
            # Hijack the action! Add our magnitude to the specific dimension
            a_t_hijacked = a_t_natural.at[dim_to_perturb].add(mag)
            
            # Step physics forward
            m_tp1 = model.memory.encode(m_t, step_idx, z_t, a_t_hijacked)
            # z_tp1 = model.forward_dyn(z_t, a_t_hijacked)
            z_tp1 = model.forward_dyn.mlp_A(z_t) + model.forward_dyn.mlp_B(a_t_hijacked)
            # z_tp1 = model.forward_dyn.mlp_A(z_t)
            # z_tp1 = model.forward_dyn.mlp_B(a_t_hijacked)
            # z_tp1 = model.forward_dyn.mlp_A(z_t) + model.forward_dyn.mlp_B(jnp.zeros_like(a_t_hijacked))

            # Render
            time_coord = jnp.array([(step_idx-1)/(T-1)], dtype=z_t.dtype)
            coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)
            frame = model.render_frame(z_t, coords_grid_t)
            
            return (z_tp1, m_tp1), frame

        m_init = model.memory.reset(T)
        _, frames = jax.lax.scan(scan_step, (z_init, m_init), jnp.arange(1, T+1))
        
        # 2. Plot the resulting hijacked video
        for t in range(T):
            ax = axes[row, t]
            img = np.clip(frames[t, ..., :C], 0, 1) if CONFIG["use_nll_loss"] else np.clip(frames[t], 0, 1)
            if img.shape[-1] == 1: img = np.repeat(img, 3, axis=-1)
            
            ax.imshow(img)
            if t == 0:
                # ax.set_ylabel(f"Perturb Dim {dim_to_perturb}\nby {mag}", fontsize=12, fontweight='bold')
                ax.set_ylabel(f"by {mag}", fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t={t}")

    plt.suptitle(f"Latent Perturbation Test on Action Dimension {dim_to_perturb}", fontsize=16)
    plt.tight_layout()
    plt.show()

# Run this for your top dimensions!
plot_action_perturbation(model, sample_batch[test_seq_id], dim_to_perturb=top_dims[3], magnitudes=[-3.0, 0.0, 3.0])


#%%
import numpy as np
from scipy.ndimage import label, center_of_mass

def extract_true_velocities(video):
    """
    Extracts (v_x1, v_y1, v_x2, v_y2) from a single Moving MNIST video.
    video shape: (T, H, W, C) or (T, H, W)
    """
    T = video.shape[0]
    centroids = np.zeros((T, 2, 2)) # (Time, Digit, [y, x])
    
    for t in range(T):
        frame = video[t].squeeze()
        # Threshold to find the digits
        binary_mask = frame > 0.1 
        labeled_array, num_features = label(binary_mask)
        
        if num_features >= 2:
            # Find center of mass for the two largest blobs
            coms = center_of_mass(frame, labeled_array, index=[1, 2])
            centroids[t, 0] = coms[0]
            centroids[t, 1] = coms[1]
        else:
            # If they overlap, just repeat the last known distinct positions
            centroids[t] = centroids[t-1] if t > 0 else np.array([[16, 16], [48, 48]])

    # To maintain consistent identities (Digit 1 vs Digit 2) across frames,
    # sort them by their y-coordinate (or x-coordinate)
    centroids = np.sort(centroids, axis=1)

    # Velocity is the derivative of position (difference between frames)
    # Pad the first frame with 0 velocity
    velocities = np.zeros((T, 4))
    velocities[1:] = (centroids[1:] - centroids[:-1]).reshape(-1, 4)
    velocities[0] = velocities[1] # Assume constant initial velocity
    
    return velocities # Shape: (T, 4) -> [v_y1, v_x1, v_y2, v_x2]

# Extract for your specific test sequence
true_vels = extract_true_velocities(sample_batch[test_seq_id])
latent_acts = final_actions[test_seq_id] # Your model's 4D actions

#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame combining latent actions and true velocities
data = np.hstack([latent_acts, true_vels])
columns = [f"Latent {i}" for i in range(4)] + ["True v_y1", "True v_x1", "True v_y2", "True v_x2"]
df = pd.DataFrame(data, columns=columns)

plt.figure(figsize=(8, 6))
correlation_matrix = df.corr().iloc[0:4, 4:8] # Just Latent vs True

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Pearson Correlation: Latent Actions vs True Physical Velocities")
plt.tight_layout()
plt.show()
# %%

from sklearn.cross_decomposition import CCA

# Initialize CCA to find 4 canonical components
cca = CCA(n_components=4)
cca.fit(latent_acts, true_vels)
latent_c, true_vel_c = cca.transform(latent_acts, true_vels)

# Calculate the correlation for each rotated component
cca_corrs = [np.corrcoef(latent_c[:, i], true_vel_c[:, i])[0, 1] for i in range(4)]

plt.figure(figsize=(8, 4))
plt.bar([f"Component {i+1}" for i in range(4)], cca_corrs, color="purple")
plt.ylim(0, 1.1)
plt.ylabel("Canonical Correlation")
plt.title("CCA: How much velocity information is in the latent space?")
for i, v in enumerate(cca_corrs):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
plt.show()

#%%
from sklearn.linear_model import Ridge

# Train a simple linear decoder to predict True Velocities from Latent Actions
decoder = Ridge(alpha=1.0)
decoder.fit(latent_acts, true_vels)
predicted_vels = decoder.predict(latent_acts)

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
titles = ["Velocity Y (Digit 1)", "Velocity X (Digit 1)", "Velocity Y (Digit 2)", "Velocity X (Digit 2)"]

for i in range(4):
    axes[i].plot(true_vels[:, i], label="Ground Truth Velocity", color="black", linewidth=2, linestyle="--")
    axes[i].plot(predicted_vels[:, i], label="Decoded from Latent Action", color="red", linewidth=2, alpha=0.7)
    axes[i].set_ylabel("Velocity")
    axes[i].set_title(titles[i])
    axes[i].legend(loc="upper right")
    axes[i].grid(alpha=0.3)

axes[3].set_xlabel("Time Step (t)")
plt.suptitle("Decoding Physical Velocity directly from Latent Actions", fontsize=14)
plt.tight_layout()
plt.show()
# %%
