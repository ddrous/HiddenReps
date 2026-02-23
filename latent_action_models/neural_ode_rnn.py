#%% Cell 1: Imports, Utilities, and Configuration
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
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
SINGLE_BATCH = False

CONFIG = {
    "seed": 2026,
    "nb_epochs": 20,
    "batch_size": 16 if not SINGLE_BATCH else 16,
    "learning_rate": 1e-5,   
    "print_every": 100,
    "seq_len_in": 10,        # Context frames used for training/encoding
    "seq_len_out": 20,       # Total frames to generate at test time (10 recon + 10 pred)
    
    # Architecture settings
    "latent_dim": 64,        # Dimension of the ODE latent space z(t)
    "rnn_hidden_dim": 256,   # Hidden state size of the GRU encoder
    "ode_hidden_dim": 256,   # Hidden width of the vector field MLP
    "kl_weight": 1e-4,       # Weight for the KL-divergence loss
    
    # Plateau Scheduler Config
    "lr_patience": 300,      
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
        if idx < ref_video.shape[0]:
            sbimshow(ref_video[idx], title=f"Ref t={idx}", ax=axes[1, i])
        else:
            axes[1, i].axis('off')

    plt.tight_layout()
    if save_name:
        plt.savefig(plots_path / save_name)
    plt.show()

#%% Cell 3: Model Definition
class CNNEncoder(eqx.Module):
    """Encodes a single frame into a flattened spatial representation."""
    layers: list
    out_dim: int

    def __init__(self, in_channels, out_dim, key):
        self.out_dim = out_dim
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1, key=k1),
            jax.nn.relu,
            eqx.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, key=k2),
            jax.nn.relu,
            eqx.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, key=k3),
            jax.nn.relu,
            eqx.nn.Conv2d(128, out_dim, kernel_size=4, stride=2, padding=1, key=k4),
            jax.nn.relu
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.reshape(-1) # Flatten

class CNNDecoder(eqx.Module):
    """Decodes a latent vector back into a pixel-space image."""
    linear: eqx.nn.Linear
    layers: list
    reshape_dim: tuple

    def __init__(self, in_dim, out_channels, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.reshape_dim = (128, 4, 4)
        self.linear = eqx.nn.Linear(in_dim, 128 * 4 * 4, key=k1)
        
        self.layers = [
            eqx.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, key=k2),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, key=k3),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, key=k4),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1, key=k5),
            jax.nn.sigmoid
        ]

    def __call__(self, z):
        x = self.linear(z)
        x = x.reshape(self.reshape_dim)
        for layer in self.layers:
            x = layer(x)
        return x

class VectorField(eqx.Module):
    """The MLP defining the continuous dynamics dz/dt = f_theta(z, t)."""
    mlp: eqx.nn.MLP
    
    def __init__(self, latent_dim, hidden_dim, key):
        self.mlp = eqx.nn.MLP(
            in_size=latent_dim + 1,  # z(t) and t
            out_size=latent_dim,
            width_size=hidden_dim,
            depth=3,
            activation=jax.nn.softplus,
            key=key
        )
        
    def __call__(self, t, z, args):
        # Concatenate time t with state z
        t_vec = jnp.ones((1,)) * t
        tz = jnp.concatenate([z, t_vec], axis=-1)
        return self.mlp(tz)

class RNN_ODE_Model(eqx.Module):
    encoder: CNNEncoder
    rnn: eqx.nn.GRUCell
    proj_mu: eqx.nn.Linear
    proj_logvar: eqx.nn.Linear
    vector_field: VectorField
    decoder: CNNDecoder
    
    latent_dim: int
    rnn_hidden_dim: int

    def __init__(self, frame_shape, latent_dim, rnn_hidden_dim, ode_hidden_dim, key):
        H, W, C = frame_shape
        self.latent_dim = latent_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        
        k_enc, k_rnn, k_mu, k_logvar, k_vf, k_dec = jax.random.split(key, 6)
        
        # We need an intermediate dimension out of the CNN before the RNN
        cnn_out_channels = 256 

        # 4 stride-2 convolutions reduce 64x64 -> 4x4
        cnn_flat_dim = cnn_out_channels * 4 * 4  
        
        self.encoder = CNNEncoder(C, cnn_out_channels, k_enc)
        
        # Initialize GRU to accept the flattened 4096-dim vector
        self.rnn = eqx.nn.GRUCell(cnn_flat_dim, rnn_hidden_dim, key=k_rnn)
        
        self.proj_mu = eqx.nn.Linear(rnn_hidden_dim, latent_dim, key=k_mu)

        self.proj_logvar = eqx.nn.Linear(rnn_hidden_dim, latent_dim, key=k_logvar)
        
        self.vector_field = VectorField(latent_dim, ode_hidden_dim, k_vf)
        self.decoder = CNNDecoder(latent_dim, C, k_dec)

    def _forward_single(self, ref_video, context_len, eval_len, key):
        # 1. Encode Context Frames through CNN + RNN
        # ref_video shape: (T, H, W, C) -> Map to (T, C, H, W) for Equinox convolutions
        video_chw = jnp.transpose(ref_video, (0, 3, 1, 2))
        
        def encode_step(h, x):
            cnn_feat = self.encoder(x)
            h_next = self.rnn(cnn_feat, h)
            return h_next, None

        h_init = jnp.zeros((self.rnn_hidden_dim,))
        # Scan over context frames to accumulate temporal information
        h_final, _ = jax.lax.scan(encode_step, h_init, video_chw[:context_len])
        
        # 2. Extract Stochastic Latent Distribution
        mu = self.proj_mu(h_final)
        logvar = self.proj_logvar(h_final)
        std = jnp.exp(0.5 * logvar)
        
        # Reparameterization trick
        eps = jax.random.normal(key, shape=(self.latent_dim,))
        z0 = mu + eps * std
        
        # 3. Solve Neural ODE Forward in Time
        solver = diffrax.Tsit5()
        dt0 = 0.1
        # Save trajectory exactly at discrete frame indices
        saveat = diffrax.SaveAt(ts=jnp.arange(float(eval_len)))
        
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.vector_field),
            solver,
            t0=0.0,
            t1=float(eval_len - 1),
            dt0=dt0,
            y0=z0,
            saveat=saveat,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-3)
        )
        z_trajectory = sol.ys # Shape: (eval_len, latent_dim)
        
        # 4. Decode Trajectory Back to Pixel Space
        decode_vmap = jax.vmap(self.decoder)
        pred_chw = decode_vmap(z_trajectory)
        
        # Convert back to (T, H, W, C)
        pred_videos = jnp.transpose(pred_chw, (0, 2, 3, 1))
        
        # Return predictions and KL Divergence terms
        kl_div = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar))
        
        return pred_videos, kl_div

    def __call__(self, ref_videos, context_len, eval_len, keys):
        is_single = (ref_videos.ndim == 4)
        if is_single:
            ref_videos = ref_videos[None, ...]
            keys = keys[None, ...] if keys.ndim == 1 else keys
            
        batched_fn = jax.vmap(self._forward_single, in_axes=(0, None, None, 0))
        preds, kl_divs = batched_fn(ref_videos, context_len, eval_len, keys)
        
        if is_single:
            return preds[0], kl_divs[0]
        return preds, kl_divs

@eqx.filter_jit
def evaluate(m, batch, context_len, eval_len, keys):
    preds, _ = m(batch, context_len, eval_len, keys)
    return preds

#%% Cell 4: Initialization & Training/Loading Logic
key, subkey = jax.random.split(key)
model = RNN_ODE_Model(
    frame_shape=(H, W, C), 
    latent_dim=CONFIG["latent_dim"], 
    rnn_hidden_dim=CONFIG["rnn_hidden_dim"],
    ode_hidden_dim=CONFIG["ode_hidden_dim"],
    key=subkey
)

print(f"Total Trainable Parameters in RNN-ODE: {count_trainable_params(model)}")

if TRAIN:
    print(f"\n🚀 Starting RNN-ODE Training -> Saving to {run_path}")
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), 
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
    def train_step(model, opt_state, keys, ref_videos):
        def loss_fn(m):
            # Training is framed purely as reconstruction of the first 10 frames
            # Evaluation will later predict the next 10 frames based on the continuous dynamic.
            pred_videos, kl_divs = m(ref_videos, context_len=CONFIG["seq_len_in"], eval_len=CONFIG["seq_len_in"], keys=keys)
            
            # ELBO Maximization (min -ELBO)
            # L2 Pixel loss between the reconstructed output and original input
            recon_loss = jnp.mean((pred_videos - ref_videos[:, :CONFIG["seq_len_in"]])**2)
            
            # KL divergence constraint
            kl_loss = jnp.mean(kl_divs)
            
            total_loss = recon_loss + CONFIG["kl_weight"] * kl_loss
            return total_loss, (recon_loss, kl_loss)

        (loss_val, (recon_val, kl_val)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array), value=loss_val
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val, recon_val, kl_val

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
            
            model, opt_state, loss, recon, kl = train_step(model, opt_state, batch_keys, batch_videos)
            epoch_losses.append(loss)
            
            current_scale = optax.tree_utils.tree_get(opt_state, "scale")
            lr_scales.append(current_scale)

            if not SINGLE_BATCH and batch_idx % CONFIG["print_every"] == 0:
                pbar.set_description(f"Loss: {loss:.4f} (Recon: {recon:.4f}, KL: {kl:.4f}), LR Scale: {current_scale:.4f}")

        all_losses.extend(epoch_losses)
        
        if epoch in [4, CONFIG["nb_epochs"]//2, 2*CONFIG["nb_epochs"]//3]:
            eqx.tree_serialise_leaves(artefacts_path / f"tf_model_ep{epoch+1}.eqx", model)

        if (epoch+1) % max(1, (CONFIG["nb_epochs"]//10)) == 0:
            val_keys = jax.random.split(key, CONFIG["batch_size"])
            # Extrapolate to 20 frames by letting the Neural ODE solve forward in time
            val_videos = evaluate(model, sample_batch, CONFIG["seq_len_in"], CONFIG["seq_len_out"], val_keys)
            plot_pred_ref_videos_rollout(val_videos[0], 
                                        sample_batch[0], 
                                        title=f"Pred", 
                                        save_name=f"pred_ref_epoch{epoch+1}.png")

    wall_time = time.time() - start_time
    print("\nWall time for RNN-ODE training:", time.strftime("%H:%M:%S", time.gmtime(wall_time)))
    
    eqx.tree_serialise_leaves(artefacts_path / "tf_model_final.eqx", model)
    np.save(artefacts_path / "loss_history.npy", np.array(all_losses))
    np.save(artefacts_path / "lr_history.npy", np.array(lr_scales))

else:
    print(f"\n📥 Loading RNN-ODE model from {artefacts_path}")
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
    ax1.plot(all_losses, color=color1, alpha=0.8, label="Total Loss (ELBO)")
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
    
    plt.title("RNN-ODE Training Loss and LR Decay")
    fig.tight_layout()
    plt.savefig(plots_path / "loss_and_lr_history.png")
    plt.show()

#%% Inference Forecast (Extrapolation)
sample_batch = next(iter(train_loader))

val_keys = jax.random.split(key, CONFIG["batch_size"])
# Extrapolate beyond the 10 context frames up to 20 total frames
final_videos = evaluate(model, sample_batch, CONFIG["seq_len_in"], CONFIG["seq_len_out"], val_keys)

plot_pred_ref_videos_rollout(
    final_videos[2], 
    sample_batch[2], 
    title=f"Pred", 
    save_name="inference_forecast_rollout.png"
)