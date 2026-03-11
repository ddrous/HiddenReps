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
    "nb_epochs": 4000*4,
    "print_every": 10,
    "batch_size": 2 if SINGLE_BATCH else 16*4,
    "learning_rate": 1e-4 if USE_NLL_LOSS else 1e-5,
    "p_forcing": 0.5,
    "inf_context_ratio": 0.5,
    "use_nll_loss": USE_NLL_LOSS,
    "aux_encoder_loss": False,
    "aux_loss_weight": 1,
    "aux_loss_num_steps": 4,

    # --- Architecture Params ---
    "lam_space": 2,
    "mem_space": 128*2,     ## Not used if discrete actions
    "icl_decoding": True,
    "discrete_actions": True,
    "split_forward": True,
    "root_width": 12,
    "root_depth": 5,
    "num_fourier_freqs": 6,
    "use_time_in_root": False,
    "pretrain_encoder": True,

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

    ## Manulally load train and test splits to have more control over batching and shuffling
    minigrid_arrays = np.load(data_path + "/MiniGrid/minigrid.npy")
    ## Only use 12 steps
    minigrid_arrays = minigrid_arrays[:, :12]       ## @TODO: maybe use all 16 steps in the future?
    print(f"Original loaded MiniGrid shape: {minigrid_arrays.shape} (N, T, H, W, 3)")
    ## SPlit, 8000 train, 2000 test
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
        fig, axes = plt.subplots(2, nb_frames, figsize=(30, 6))
        indices_to_plot = np.arange(0, nb_frames, 1)
        for i, idx in enumerate(indices_to_plot):
            video_to_plot = video[idx] if not rescale else (video[idx] + 1.0) / 2.0
            sbimshow(video_to_plot, title=f"{title} t={idx}", ax=axes[0, i])
            sbimshow(ref_video[idx], title=f"Ref t={idx}", ax=axes[1, i])
    else:
        fig, axes = plt.subplots(3, nb_frames, figsize=(30, 7))
        indices_to_plot = np.arange(0, nb_frames, 1)
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

    def get_latents(self, z_prev, a):
        if self.split_forward:
            return self.mlp_A(z_prev), self.mlp_B(a)
        else:
            return NotImplementedError("get_latents is only implemented for split_forward=True")


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

class InverseDynamics(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, dyn_dim, lam_dim, key, num_actions=None):
        if num_actions: ## Discrete case
            self.mlp = eqx.nn.MLP(dyn_dim * 2, num_actions, width_size=dyn_dim*1, depth=3, key=key)
        else:
            self.mlp = eqx.nn.MLP(dyn_dim * 2, lam_dim, width_size=dyn_dim*1, depth=3, key=key)
        
    def __call__(self, z_prev, z_target):
        return self.mlp(jnp.concatenate([z_prev, z_target], axis=-1))


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
    lam_dim: int = eqx.field(static=True)
    icl_decoding: bool = eqx.field(static=True)
    
    # Standard decoding components
    action_mlp: Optional[eqx.nn.MLP]
    
    # ICL decoding components
    output_proj: Optional[eqx.nn.Linear]

    def __init__(self, lam_dim, mem_dim, latent_dim, key, max_len=12, num_heads=4, num_blocks=4, num_actions=4):
        self.max_len = max_len
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
            if num_actions:
                self.output_proj = eqx.nn.Linear(self.d_model, num_actions, key=k6)
            else:
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

            return self.output_proj(context)

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


class LAM(eqx.Module):
    """ A LAM is made up of a IDM and a DMM, it produces the action based on context. It also stores discrete embeddings if needed """
    idm: InverseDynamics
    dmm: MemoryModule

    discrete_actions: bool = eqx.field(static=True)
    action_embedding: Optional[eqx.nn.Embedding]

    def __init__(self, dyn_dim, lam_dim, mem_dim, max_len, num_heads, num_blocks, num_actions, key):
        k1, k2 = jax.random.split(key)
        self.discrete_actions = num_actions is not None

        self.idm = InverseDynamics(dyn_dim, 
                                   lam_dim, 
                                   key=k1, 
                                   num_actions=num_actions if self.discrete_actions else None)
        self.dmm = MemoryModule(lam_dim, 
                                mem_dim, 
                                dyn_dim, 
                                key=k2, 
                                num_heads=num_heads, 
                                num_blocks=num_blocks, 
                                max_len=max_len, 
                                num_actions=num_actions if self.discrete_actions else None)

        if self.discrete_actions:
            self.action_embedding = eqx.nn.Embedding(num_actions, lam_dim, key=k2)
        else:
            self.action_embedding = None

    def __call__(self, context):
        return NameError("The module cannot be called. Please call either the MemoryModule directly for the autoregressive path, or the IDM for the forced path.")

    def discretise_action(self, logits):
        # 2. Calculate the soft probabilities (this gives us our gradients!)
        soft_probs = jax.nn.softmax(logits, axis=-1)

        # 3. Calculate the strictly hard, discrete one-hot vector (this is what we want for the forward pass)
        hard_idx = jnp.argmax(logits, axis=-1)
        hard_probs = jax.nn.one_hot(hard_idx, num_classes=logits.shape[-1])

        # 4. The Magic STE Formula:
        ste_probs = soft_probs + jax.lax.stop_gradient(hard_probs - soft_probs)

        # 5. Lookup the embedding
        action = jnp.dot(ste_probs, self.action_embedding.weight)

        return action

    def inverse_dynamics(self, z_prev, z_target):
        if not self.discrete_actions:
            return self.idm(z_prev, z_target)
        else:
            logits = self.idm(z_prev, z_target)
            return self.discretise_action(logits)

    def decode_memory(self, buffer, step_idx, z_current):
        if not self.discrete_actions:
            return self.dmm.decode(buffer, step_idx, z_current)
        else:
            logits = self.dmm.decode(buffer, step_idx, z_current)
            return self.discretise_action(logits)

    def encode_memory(self, buffer, step_idx, z_current, a):
        return self.dmm.encode(buffer, step_idx, z_current, a)
    
    def reset_memory(self, T):
        return self.dmm.reset(T)



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
        self.encoder = CNNEncoder(in_channels=C, out_dim=self.d_theta, spatial_shape=(H, W), key=k_enc, hidden_width=32, depth=4)

        if CONFIG["pretrain_encoder"]:
            ## Load the pretrained encoder weights from the autoencoding phase
            self.encoder, self.theta_base = eqx.tree_deserialise_leaves("minigrid_enc.eqx", (self.encoder, self.theta_base))

        self.forward_dyn = ForwardDynamics(self.d_theta, lam_dim, split_forward, key=k_fwd)

        self.mem_dim = mem_dim

        # self.lam = LAM(self.d_theta, lam_dim, key=k_lam)
        # self.memory = MemoryModule(self.lam_dim, self.mem_dim, self.d_theta, key=k_mem, max_len=12, num_heads=4, num_blocks=4, num_actions=4)

        num_actions = 4 if CONFIG["discrete_actions"] else None
        self.action_model = LAM(self.d_theta, lam_dim, mem_dim, max_len=12, num_heads=4, num_blocks=4, num_actions=num_actions, key=k_lam)

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
        if not CONFIG["pretrain_encoder"]:
            theta = theta_offset + self.theta_base
            # theta = theta_offset
        else:
            theta = theta_offset + jax.lax.stop_gradient(self.theta_base)

        pred_flat = self.render_pixels(theta, flat_coords)
        return pred_flat.reshape(H, W, -1)

    def _get_preds_single(self, ref_video, p_forcing, key, coords_grid, context_ratio):
        T = ref_video.shape[0]
        init_frame = ref_video[0]

        # 1. Initialize offset from first frame
        z_init = self.encoder(jnp.transpose(init_frame, (2, 0, 1)))
        if CONFIG["pretrain_encoder"]:
            z_init = jax.lax.stop_gradient(z_init)

        # 2. Initialize fixed buffer for Transformer memory
        m_init = self.action_model.reset_memory(T)

        ## Flip a coin at the start of each step to decide whether to force towards GT or not
        # is_context_init = jax.random.bernoulli(key, p=0.5).astype(bool)
        is_context_init = False

        # 3. Add the Equinox checkpointing decorator here!
        # Recompute internal activations during the backward pass.
        @eqx.filter_checkpoint
        def scan_step(carry, scan_inputs):
            z_t, m_t, was_context = carry
            o_tp1, step_idx = scan_inputs
            # subk = jax.random.fold_in(key, step_idx)

            # --- Rendering INSIDE the checkpointed step ---
            # Concatenate the time t to the coordinates before rendering
            time_coord = jnp.array([(step_idx-1)/ (T-1)], dtype=z_t.dtype)
            coords_grid_t = jnp.concatenate([jnp.full_like(coords_grid[..., :1], time_coord), coords_grid], axis=-1)

            pred_out = self.render_frame(z_t, coords_grid_t)

            # Determine if we are forcing towards ground truth this step
            # is_context = jnp.logical_not(was_context)
            ## Calculate based n step_idx even or not
            # is_context = step_idx % 2 == 1
            is_context = (step_idx / T) < context_ratio

            a_t = jax.lax.cond(
                is_context,
                lambda: self.action_model.inverse_dynamics(
                    z_t, 
                    jax.lax.stop_gradient(self.encoder(jnp.transpose(o_tp1, (2, 0, 1))))
                ),
                lambda: self.action_model.decode_memory(m_t, step_idx, z_t)
            )

            ## Encode into the memory with C
            m_tp1 = self.action_model.encode_memory(m_t, step_idx, z_t, a_t)

            # SINGLE forward dynamics call handles both the forced and autoregressive paths!
            z_tp1 = self.forward_dyn(z_t, a_t)

            return (z_tp1, m_tp1, is_context), (a_t, z_t, pred_out)

        scan_inputs = (jnp.concatenate([ref_video[1:], ref_video[-1:]], axis=0), jnp.arange(1, T+1))
        
        # 4. Execute the scan as normal, collecting both latents and rendered frames
        _, (actions, pred_latents, pred_video) = jax.lax.scan(scan_step, (z_init, m_init, is_context_init), scan_inputs)

        return actions, pred_latents, pred_video


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

print(f"Dynamics Weight Space Dimension (d_theta): {model.d_theta}")

print(f"Total Trainable Parameters in WARP: {count_trainable_params(model)}")

count_A = count_trainable_params(model.forward_dyn.mlp_A)
count_B = count_trainable_params(model.forward_dyn.mlp_B)
count_lam = count_trainable_params(model.action_model.idm)
count_memory = count_trainable_params(model.action_model.dmm)
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

            context_ratio = jax.random.uniform(k_full, minval=0.0, maxval=1.0)
            # context_ratio = jax.random.uniform(k_full, minval=0.25, maxval=1.0)
            # context_ratio = CONFIG["inf_context_ratio"]
            _, _, pred_videos = m(ref_videos, p_forcing, keys, coords_grid, context_ratio, precompute_ref_diffs=False)

            ## --- 1. LATENT (WEIGHT-SPACE) DYNAMICS LOSS (Primary) ---
            # latent_loss = jnp.mean((pred_thetas - target_thetas_shifted)**2)
            rec_loss = jnp.mean((pred_videos - ref_videos)**2)

            # # --- 1. SSIM LOSS (Primary) ---
            # # Compute SSIM for each frame and average over time and batch
            # def ssim(x, y, data_range=1.0):
            #     C1 = (0.01 * data_range) ** 2
            #     C2 = (0.03 * data_range) ** 2

            #     mu_x = jnp.mean(x)
            #     mu_y = jnp.mean(y)
            #     sigma_x = jnp.var(x)
            #     sigma_y = jnp.var(y)
            #     sigma_xy = jnp.mean((x - mu_x) * (y - mu_y))

            #     ssim_numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
            #     ssim_denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

            #     return ssim_numerator / ssim_denominator
            # ssim_loss = 1.0 - jnp.mean(jax.vmap(lambda x, y: ssim(x, y, data_range=1.0), in_axes=(0, 0))(pred_videos, ref_videos))
            # rec_loss = ssim_loss

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

#%%

# testing_subset = datasets.MiniGrid(root=data_path, split="test", download=True)

testing_subset = MiniGridDataset(test_arrays)
test_loader = DataLoader(testing_subset, batch_size=CONFIG["batch_size"]*10, shuffle=False, collate_fn=numpy_collate, drop_last=False)
sample_batch = next(iter(test_loader))

# sample_batch = next(iter(train_loader))

print("Batch shape for evaluation:", sample_batch.shape, flush=True)

pad_length = nb_frames - sample_batch.shape[1]
sample_batch = jnp.concatenate([sample_batch, np.zeros((sample_batch.shape[0], pad_length, H, W, C), dtype=sample_batch.dtype)], axis=1)

val_keys = jax.random.split(key, sample_batch.shape[0])
final_actions, _, final_videos = evaluate(model, sample_batch, 0.0, val_keys, coords_grid, 0.0, precompute_ref_diffs=False)

if CONFIG["use_nll_loss"]:
    print(f"Final Predicted Video Mean Pixel Value Range: min={final_videos[...,:C].min():.4f}, max={final_videos[...,:C].max():.4f}")
    print(f"Final Predicted Video Std Pixel Value Range: min={final_videos[...,C:].min():.4f}, max={final_videos[...,C:].max():.4f}")

#%%
test_seq_id = np.random.randint(0, sample_batch.shape[0])
test_seq_id = 28
# test_seq_id = 128
print(f"Plotting rollout for test sequence ID: {test_seq_id}")

plot_pred_ref_videos_rollout(
    final_videos[test_seq_id], 
    sample_batch[test_seq_id],
    title=f"Pred", 
    save_name=f"inference_forecast_rollout_seq{test_seq_id}.png"
)

# %%
os.system(f"cp -r nohup.log {run_path}/nohup.log")

#%%

## Print the actions for the first test sequence
print("Predicted Latent Actions for the first test sequence:")
# print(final_actions[test_seq_id])

import seaborn as sns

plt.figure(figsize=(12, 6))
# Transpose to have dimensions on Y axis and time on X axis
sns.heatmap(final_actions[test_seq_id].T, cmap="coolwarm", center=0, 
            annot=True, fmt=".2f", cbar_kws={'label': 'Latent Action Value'}, annot_kws={'size': 8})
plt.xlabel("Time Step (t)")
plt.ylabel("Latent Dimension (0-9)")
plt.title(f"Latent Action Heatmap for Sequence {test_seq_id}")
plt.tight_layout()
plt.savefig(plots_path / f"action_heatmap_seq{test_seq_id}.png")
plt.show()


#%%
# Flatten across batch and time: shape becomes (B * T, 10)
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
plt.savefig(plots_path / "action_dimension_variance.png")
plt.show()


#%%
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Project 10D actions to 2D for visualization
pca = PCA(n_components=2)
actions_2d = pca.fit_transform(all_actions_flat)

# 2. Cluster the actions. MiniGrid usually has ~3-4 relevant actions for simple navigation
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(all_actions_flat)

# 3. Plot the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(actions_2d[:, 0], actions_2d[:, 1], c=cluster_labels, 
                      cmap='tab10', alpha=0.6, s=20)
plt.legend(*scatter.legend_elements(), title="Action Clusters")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("Latent Action Space Projection (PCA)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "action_pca_clusters.png")
plt.show()


#%%
plt.figure(figsize=(10, 8))
# Plot the background distribution in grey
plt.scatter(actions_2d[:, 0], actions_2d[:, 1], c='lightgrey', alpha=0.3, s=10)

# Get the 2D coordinates for just our specific test sequence
seq_actions_2d = pca.transform(final_actions[test_seq_id])

# Plot the trajectory with arrows
plt.plot(seq_actions_2d[:, 0], seq_actions_2d[:, 1], marker='o', color='green', markersize=6, linewidth=2)

# Annotate the time steps
for t in range(seq_actions_2d.shape[0]):
    plt.annotate(f"t={t}", (seq_actions_2d[t, 0], seq_actions_2d[t, 1]), 
                 textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"Action Trajectory for Sequence {test_seq_id} in Latent Space")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / f"action_trajectory_seq{test_seq_id}.png")
plt.show()


# %%
