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
from tqdm import tqdm

import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

import seaborn as sns
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

def count_trainable_params(model):
    """Utility to count trainable parameters in an Equinox module."""
    def count_params(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.inexact):
            return x.size
        return 0
    param_counts = jax.tree_util.tree_map(count_params, model)
    return sum(jax.tree_util.tree_leaves(param_counts))

# --- Configuration ---
TRAIN = True
RUN_DIR = "./" if not TRAIN else None
SINGLE_BATCH = True

CONFIG = {
    "seed": 2026,
    "nb_epochs": 1000,
    "batch_size": 4 if not SINGLE_BATCH else 1,
    "learning_rate": 1e-6,   # Paper starting learning rate [cite: 343]
    "print_every": 100,
    "p_forcing": 1.0,        # Scheduled sampling probability 
    
    # PredRNN++ Architecture settings [cite: 220, 221]
    "hidden_channels": [64, 64, 64, 64], # Paper uses [128, 64, 64, 64], scaled down for safety
    "kernel_size": 5,        
    "ghu_channels": 64,      # Paper uses 128
    "seq_len_in": 10,        # 10 frames in [cite: 347]
    "seq_len_out": 10,       # 10 frames out [cite: 347]

    # Plateau Scheduler Config
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
def numpy_collate(batch):
    if isinstance(batch[0], tuple):
        videos = torch.stack([b[0] for b in batch]).numpy()
    else:
        videos = torch.stack(batch).numpy()
    
    if videos.ndim == 4:
        videos = np.expand_dims(videos, axis=-1)
    elif videos.ndim == 5 and videos.shape[2] == 1:
        # moving from (B, T, C, H, W) to (B, T, H, W, C)
        videos = np.transpose(videos, (0, 1, 3, 4, 2))
        
    videos = videos.astype(np.float32)
    if videos.max() > 2.0:
        videos = videos / 255.0  

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

class CausalLSTM(eqx.Module):
    """
    Causal LSTM unit containing dual memories: temporal and spatial.
    The spatial memory is updated as a function of the temporal memory in a cascaded mechanism[cite: 113, 128].
    """
    W1: eqx.nn.Conv2d
    W2: eqx.nn.Conv2d
    W3: eqx.nn.Conv2d
    W4: eqx.nn.Conv2d
    W5: eqx.nn.Conv2d
    hidden_dim: int

    def __init__(self, in_channels, hidden_dim, kernel_size, key):
        self.hidden_dim = hidden_dim
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        pad = kernel_size // 2
        
        # W1 * [X_t, H_{t-1}, C_{t-1}] -> g_t, i_t, f_t 
        self.W1 = eqx.nn.Conv2d(in_channels + 2 * hidden_dim, 3 * hidden_dim, kernel_size, padding=pad, key=k1)

        # # After defining self.W1:
        # bias1_patched = self.W1.bias.at[2 * hidden_dim :].set(1.0)
        # self.W1 = eqx.tree_at(lambda m: m.bias, self.W1, bias1_patched)

        # W2 * [X_t, C_t, M_{t-1}] -> g'_t, i'_t, f'_t 
        self.W2 = eqx.nn.Conv2d(in_channels + 2 * hidden_dim, 3 * hidden_dim, kernel_size, padding=pad, key=k2)

        # # After defining self.W2:
        # bias2_patched = self.W2.bias.at[2 * hidden_dim :].set(1.0)
        # self.W2 = eqx.tree_at(lambda m: m.bias, self.W2, bias2_patched)

        # W3 is a 1x1 convolution for the spatial memory transition [cite: 133]
        self.W3 = eqx.nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0, key=k3)
        # W4 * [X_t, C_t, M_t] -> o_t 

        self.W4 = eqx.nn.Conv2d(in_channels + 2 * hidden_dim, hidden_dim, kernel_size, padding=pad, key=k4)

        # W5 is a 1x1 convolution for final hidden state gating [cite: 133]
        self.W5 = eqx.nn.Conv2d(2 * hidden_dim, hidden_dim, 1, padding=0, key=k5)

    def __call__(self, x, h_prev, c_prev, m_prev):
        # 1. Temporal Memory Update 
        w1_out = self.W1(jnp.concatenate([x, h_prev, c_prev], axis=0))
        g_t_raw, i_t_raw, f_t_raw = jnp.split(w1_out, 3, axis=0)
        
        g_t = jnp.tanh(g_t_raw)
        i_t = jax.nn.sigmoid(i_t_raw)
        f_t = jax.nn.sigmoid(f_t_raw)
        
        c_t = f_t * c_prev + i_t * g_t
        
        # 2. Spatial Memory Update 
        w2_out = self.W2(jnp.concatenate([x, c_t, m_prev], axis=0))
        g_t_prime_raw, i_t_prime_raw, f_t_prime_raw = jnp.split(w2_out, 3, axis=0)
        
        g_t_prime = jnp.tanh(g_t_prime_raw)
        i_t_prime = jax.nn.sigmoid(i_t_prime_raw)
        f_t_prime = jax.nn.sigmoid(f_t_prime_raw)
        
        m_t = f_t_prime * jnp.tanh(self.W3(m_prev)) + i_t_prime * g_t_prime
        
        # 3. Final Hidden State 
        o_t = jax.nn.sigmoid(self.W4(jnp.concatenate([x, c_t, m_t], axis=0))) # Note: Paper says tanh in eq 1, but o_t is traditionally sigmoid. Using sigmoid for stable gating.
        h_t = o_t * jnp.tanh(self.W5(jnp.concatenate([c_t, m_t], axis=0)))
        
        return h_t, c_t, m_t
    
class GHU(eqx.Module):
    """
    Gradient Highway Unit to allow adaptive learning between transformed inputs and long-term hidden states.
    Alleviates vanishing gradients in deep-in-time sequences[cite: 144, 147].
    """
    W_p: eqx.nn.Conv2d
    W_s: eqx.nn.Conv2d
    
    def __init__(self, in_channels, hidden_dim, kernel_size, key):
        k1, k2 = jax.random.split(key, 2)
        pad = kernel_size // 2
        # W_p and W_s applied to [X_t, Z_{t-1}] 
        self.W_p = eqx.nn.Conv2d(in_channels + hidden_dim, hidden_dim, kernel_size, padding=pad, key=k1)
        self.W_s = eqx.nn.Conv2d(in_channels + hidden_dim, hidden_dim, kernel_size, padding=pad, key=k2)

    def __call__(self, x, z_prev):
        concat_xz = jnp.concatenate([x, z_prev], axis=0)
        p_t = jnp.tanh(self.W_p(concat_xz))
        s_t = jax.nn.sigmoid(self.W_s(concat_xz))
        z_t = s_t * p_t + (1 - s_t) * z_prev
        return z_t

class PredRNNPP(eqx.Module):
    causal_lstms: list
    ghu: GHU
    final_conv: eqx.nn.Conv2d
    layers: int
    frame_shape: tuple

    def __init__(self, in_channels, hidden_channels, ghu_channels, kernel_size, frame_shape, key):
        self.layers = len(hidden_channels)
        self.frame_shape = frame_shape
        self.causal_lstms = []
        
        keys = jax.random.split(key, self.layers + 2)
        
        # Build 4 Causal LSTMs
        for i in range(self.layers):
            c_in = in_channels if i == 0 else hidden_channels[i-1]
            if i == 1: 
                c_in = ghu_channels # GHU sits between layer 0 and 1 [cite: 150]
                
            self.causal_lstms.append(
                CausalLSTM(c_in, hidden_channels[i], kernel_size, keys[i])
            )
            
        # Insert GHU after the bottom Causal LSTM [cite: 150]
        self.ghu = GHU(hidden_channels[0], ghu_channels, kernel_size, keys[-2])
        
        # Final projection to image space (C channels)
        self.final_conv = eqx.nn.Conv2d(hidden_channels[-1], in_channels, 1, padding=0, key=keys[-1])

    def _forward_single(self, ref_video, p_forcing, key):
        T, H, W, C = ref_video.shape
        
        # Initialize zero states for all layers
        h_states = [jnp.zeros((dim, H, W)) for dim in CONFIG["hidden_channels"]]
        c_states = [jnp.zeros((dim, H, W)) for dim in CONFIG["hidden_channels"]]
        m_states = [jnp.zeros((dim, H, W)) for dim in CONFIG["hidden_channels"]]
        z_state = jnp.zeros((CONFIG["ghu_channels"], H, W))
        
        def scan_step(state, scan_inputs):
            h_st, c_st, m_st, z_st, prev_pred, k = state
            gt_curr_frame, step_idx = scan_inputs
            k, subk = jax.random.split(k)
            
            # Scheduled sampling
            use_gt = jax.random.bernoulli(subk, p_forcing) | (step_idx < CONFIG["seq_len_in"])
            x_t = jnp.where(use_gt, gt_curr_frame, prev_pred)
            
            # (H, W, C) -> (C, H, W)
            x_t = jnp.transpose(x_t, (2, 0, 1))
            
            # Layer 0 [cite: 152]
            # Uses M from top layer of previous timestep (m_st[-1])
            h_st[0], c_st[0], m_st[0] = self.causal_lstms[0](x_t, h_st[0], c_st[0], m_st[-1])
            
            # GHU [cite: 152]
            z_st = self.ghu(h_st[0], z_st)
            
            # Layer 1 [cite: 153]
            h_st[1], c_st[1], m_st[1] = self.causal_lstms[1](z_st, h_st[1], c_st[1], m_st[0])
            
            # Subsequent layers [cite: 153]
            for i in range(2, self.layers):
                h_st[i], c_st[i], m_st[i] = self.causal_lstms[i](h_st[i-1], h_st[i], c_st[i], m_st[i-1])
                
            # Decode to image space
            out_frame_chw = self.final_conv(h_st[-1])
            # PredRNN++ usually doesn't strictly sigmoid at the end if standardized, but for image [0,1] it's useful
            out_frame = jax.nn.sigmoid(jnp.transpose(out_frame_chw, (1, 2, 0)))
            
            new_state = (h_st, c_st, m_st, z_st, out_frame, subk)
            return new_state, out_frame

        init_state = (h_states, c_states, m_states, z_state, ref_video[0], key)
        # We roll forward from t=0. 
        # For evaluation beyond inputs, we just provide zeros for the scan inputs and rely on the model prediction.
        scan_inputs = (ref_video, jnp.arange(T))
        _, pred_video = jax.lax.scan(scan_step, init_state, scan_inputs)

        return pred_video

    def __call__(self, ref_videos, p_forcing, keys):
        is_single = (ref_videos.ndim == 4)
        if is_single:
            ref_videos = ref_videos[None, ...]
            keys = keys[None, ...] if keys.ndim == 1 else keys
            
        batched_fn = jax.vmap(self._forward_single, in_axes=(0, None, 0))
        preds = batched_fn(ref_videos, p_forcing, keys)
        
        if is_single:
            return preds[0]
        return preds

@eqx.filter_jit
def evaluate(m, batch, p_forcing, keys):
    return m(batch, p_forcing, keys)

#%% Cell 4: Initialization & Training/Loading Logic
key, subkey = jax.random.split(key)
model = PredRNNPP(
    in_channels=C, 
    hidden_channels=CONFIG["hidden_channels"], 
    ghu_channels=CONFIG["ghu_channels"], 
    kernel_size=CONFIG["kernel_size"], 
    frame_shape=(H, W, C), 
    key=subkey
)

print(f"Total Trainable Parameters in PredRNN++: {count_trainable_params(model)}")

if TRAIN:
    print(f"\n🚀 Starting PredRNN++ Training -> Saving to {run_path}")
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # <-- Add this critical line
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
    def train_step(model, opt_state, keys, ref_videos, p_forcing):
        def loss_fn(m):
            pred_videos = m(ref_videos, p_forcing, keys)
            
            # Predict sequences beyond context length
            target_videos = ref_videos[:, 1:]
            pred_videos = pred_videos[:, :-1]
            
            # L1 + L2 loss as proposed in the paper 
            l1_loss = jnp.mean(jnp.abs(pred_videos - target_videos))
            l2_loss = jnp.mean((pred_videos - target_videos)**2)
            
            return l1_loss + l2_loss

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
        
        pbar = tqdm(train_loader)
        for batch_idx, batch_videos in enumerate(pbar):
        # for batch_idx, batch_videos in enumerate(train_loader):
            key, subkey = jax.random.split(key)
            batch_keys = jax.random.split(subkey, CONFIG["batch_size"])
            
            model, opt_state, loss = train_step(model, opt_state, batch_keys, batch_videos, CONFIG["p_forcing"])
            epoch_losses.append(loss)
            
            current_scale = optax.tree_utils.tree_get(opt_state, "scale")
            lr_scales.append(current_scale)

            if (batch_idx + 1) % CONFIG["print_every"] == 0:
                ## Updated tqdm description to show current LR scale and loss
                pbar.set_description(f"Epoch {epoch+1} | Loss: {loss:.4f} | LR Scale: {current_scale:.4f}")

        all_losses.extend(epoch_losses)
        
        if epoch in [4, CONFIG["nb_epochs"]//2, 2*CONFIG["nb_epochs"]//3]:
            eqx.tree_serialise_leaves(artefacts_path / f"tf_model_ep{epoch+1}.eqx", model)

        if (epoch+1) % max(1, (CONFIG["nb_epochs"]//10)) == 0:
            val_keys = jax.random.split(key, CONFIG["batch_size"])
            # Evaluate with zero forcing to test autoregressive generation
            val_videos = evaluate(model, sample_batch, 0.0, val_keys)
            plot_pred_ref_videos_rollout(val_videos[0], 
                                        sample_batch[0], 
                                        title=f"Pred", 
                                        save_name=f"pred_ref_epoch{epoch+1}.png")

    wall_time = time.time() - start_time
    print("\nWall time for PredRNN++ training:", time.strftime("%H:%M:%S", time.gmtime(wall_time)))
    
    eqx.tree_serialise_leaves(artefacts_path / "tf_model_final.eqx", model)
    np.save(artefacts_path / "loss_history.npy", np.array(all_losses))
    np.save(artefacts_path / "lr_history.npy", np.array(lr_scales))

else:
    print(f"\n📥 Loading PredRNN++ model from {artefacts_path}")
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
    ax1.plot(all_losses, color=color1, alpha=0.8, label="Total Loss (L1 + L2)")
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
    
    plt.title("PredRNN++ Training Loss and LR Decay")
    fig.tight_layout()
    plt.savefig(plots_path / "loss_and_lr_history.png")
    plt.show()

#%% Inference Forecast
sample_batch = next(iter(train_loader))

# Zero forcing: completely autoregressive after input frames
val_keys = jax.random.split(key, CONFIG["batch_size"])
final_videos = evaluate(model, sample_batch, 0.0, val_keys)

plot_pred_ref_videos_rollout(
    final_videos[0], 
    sample_batch[0], 
    title=f"Pred", 
    save_name="inference_forecast_rollout.png"
)