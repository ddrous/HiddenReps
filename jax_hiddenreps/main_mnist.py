#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Modern matplotlib configurations
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.constrained_layout.use'] = True

H, W = 32, 32

# ==========================================
# 1. UTILITIES & DYNAMIC INR SETUP
# ==========================================
def get_run_path(base_dir="./runs"):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    run_id = f"fbae_mnist_2D_{int(time.time())}"
    run_path = Path(base_dir) / run_id
    run_path.mkdir()
    (run_path / "plots").mkdir()
    return run_path

def count_params(model):
    """Calculates the total number of trainable parameters in an Equinox module."""
    return sum(x.size for x in jax.tree_util.tree_leaves(model) if eqx.is_array(x))

# --- DYNAMIC INR PARAMETER EXTRACTION ---
# Reverted to standard ReLU as requested.
dummy_key = jax.random.PRNGKey(0)
dummy_inr = eqx.nn.MLP(in_size=16, out_size=1, width_size=16, depth=4, activation=jax.nn.relu, key=dummy_key)

inr_params, STATIC_INR = eqx.partition(dummy_inr, eqx.is_array)
flat_params, UNFLATTEN_FN = jax.flatten_util.ravel_pytree(inr_params)
TARGET_PARAM_COUNT = len(flat_params) # Will be ~609 parameters

# def render_image(theta, coords):
#     """Reconstructs the 32x32 image from the flat weight vector."""
#     restored_params = UNFLATTEN_FN(theta)
#     inr = eqx.combine(restored_params, STATIC_INR)
#     def forward_pixel(c):
#         # return jax.nn.sigmoid(inr(c))[0] 
#         return inr(c)[0] 
#     pixels = jax.vmap(forward_pixel)(coords)
#     # return pixels.reshape(1, 32, 32) 
#     return pixels.reshape(1, H, W) 

def render_image(theta, coords):
    """Reconstructs the HxW image from the flat weight vector using Fourier features."""
    restored_params = UNFLATTEN_FN(theta)
    inr = eqx.combine(restored_params, STATIC_INR)
    
    def forward_pixel(c):
        # 1. Define frequency bands (e.g., 4 bands -> 16D feature vector)
        num_freqs = 4
        freq_bands = 2.0 ** jnp.arange(num_freqs)
        
        # 2. Project coordinates: c is (2,), freq_bands is (4,) -> shape (2, 4)
        scaled_c = c[:, None] * freq_bands * jnp.pi 
        
        # 3. Apply Sin/Cos and flatten into a 1D feature array
        fourier_features = jnp.concatenate([
            jnp.sin(scaled_c).flatten(), 
            jnp.cos(scaled_c).flatten()
        ])
        
        # 4. Pass the high-dimensional representation to the INR
        return inr(fourier_features)[0] 
        
    pixels = jax.vmap(forward_pixel)(coords)
    return pixels.reshape(1, H, W)

# ==========================================
# 2. PYTORCH TO JAX DATALOADER (MNIST)
# ==========================================
def numpy_collate(batch):
    if isinstance(batch[0], torch.Tensor):
        return np.stack([x.numpy() for x in batch])
    elif isinstance(batch[0], tuple):
        return tuple(numpy_collate([samples[i] for samples in batch]) for i in range(len(batch[0])))
    else:
        return np.array(batch)

class MNISTDataHandler:
    def __init__(self, batch_size=256, subset_size=None, seed=42):
        self.batch_size = batch_size
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        
        full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        full_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        if subset_size:
            rng = np.random.default_rng(seed)
            train_idx = rng.choice(len(full_train), subset_size, replace=False)
            full_train = Subset(full_train, train_idx)
            
        self.train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate, drop_last=True)
        self.val_loader = DataLoader(full_test, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate, drop_last=True)
        
        xs = jnp.linspace(-1, 1, H)
        ys = jnp.linspace(-1, 1, W)
        X, Y = jnp.meshgrid(xs, ys)
        self.coords = jnp.stack([X.flatten(), Y.flatten()], axis=-1)

    def get_full_eval_set(self, split='val', max_samples=3000):
        loader = self.train_loader if split == 'train' else self.val_loader
        images, labels = [], []
        count = 0
        for x, y in loader:
            images.append(x)
            labels.append(y)
            count += len(x)
            if count >= max_samples: break
        return jnp.concatenate(images)[:max_samples], np.concatenate(labels)[:max_samples]

# ==========================================
# 3. SHARED ARCHITECTURE COMPONENTS
# ==========================================
class CNNBackbone(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d

    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(1, 32, 4, stride=2, padding=1, key=k1)  # -> 16x16
        self.conv2 = eqx.nn.Conv2d(32, 64, 4, stride=2, padding=1, key=k2) # -> 8x8
        self.conv3 = eqx.nn.Conv2d(64, 128, 4, stride=2, padding=1, key=k3) # -> 4x4

    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        return x.flatten() # Outputs 2048

class CNNEncoder(eqx.Module):
    """Wraps the backbone to project to the desired latent dimension."""
    backbone: CNNBackbone
    linear: eqx.nn.Linear

    def __init__(self, latent_dim=2, key=None):
        k1, k2 = jax.random.split(key)
        self.backbone = CNNBackbone(k1)
        self.linear = eqx.nn.Linear(128 * 4 * 4, latent_dim, key=k2)

    def __call__(self, x):
        features = self.backbone(x)
        return self.linear(features)

class CNNDecoder(eqx.Module):
    linear_up: eqx.nn.Linear
    deconv1: eqx.nn.ConvTranspose2d
    deconv2: eqx.nn.ConvTranspose2d
    deconv3: eqx.nn.ConvTranspose2d

    def __init__(self, latent_dim=2, key=None):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.linear_up = eqx.nn.Linear(latent_dim, 128 * 4 * 4, key=k1)
        self.deconv1 = eqx.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, key=k2) 
        self.deconv2 = eqx.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, key=k3) 
        self.deconv3 = eqx.nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, key=k4)  

    def __call__(self, z):
        h = jax.nn.relu(self.linear_up(z))
        h = h.reshape((128, 4, 4))
        h = jax.nn.relu(self.deconv1(h))
        h = jax.nn.relu(self.deconv2(h))
        return jax.nn.sigmoid(self.deconv3(h))

# ==========================================
# 4. BASELINES & F-BAE MODELS
# ==========================================
class DenseAE(eqx.Module):
    """Baseline 1: Standard Dense Autoencoder (~300k params)"""
    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.encoder = eqx.nn.MLP(H*W, 2, width_size=128, depth=2, key=k1)
        self.decoder = eqx.nn.MLP(2, H*W, width_size=128, depth=2, key=k2)

    def __call__(self, x, coords=None, key=None):
        x_flat = x.flatten()
        z = self.encoder(x_flat)
        out = jax.nn.sigmoid(self.decoder(z))
        return out.reshape(1, H, W), z

class CNNAE(eqx.Module):
    """Baseline 2: Standard CNN Autoencoder (~350k params)"""
    encoder: CNNEncoder # Shared!
    decoder: CNNDecoder

    def __init__(self, key):
        self.encoder = CNNEncoder(latent_dim=2, key=key)
        self.decoder = CNNDecoder(latent_dim=2, key=key)

    def __call__(self, x, coords=None, key=None):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

class CNN_VAE(eqx.Module):
    """Baseline 3: Variational Autoencoder (~350k params)"""
    encoder: CNNEncoder
    decoder: CNNDecoder

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.encoder = CNNEncoder(latent_dim=4, key=k1) # 2 mu, 2 logvar
        self.decoder = CNNDecoder(latent_dim=2, key=k2)

    def __call__(self, x, coords=None, key=None):
        h = self.encoder(x)
        mu, logvar = h[:2], h[2:]
        if key is not None:
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(key, std.shape)
            z = mu + eps * std
        else:
            z = mu 
        out = self.decoder(z)
        return out, z, mu, logvar

class DirectHyperNet(eqx.Module):
    """Baseline 4: Unconstrained Functional Generation (~380k params)"""
    encoder: CNNEncoder
    projector: eqx.nn.Linear
    theta_base: jax.Array

    def __init__(self, dummy_inr_params, key):
        k1, k2 = jax.random.split(key)
        self.encoder = CNNEncoder(latent_dim=128, key=k1) # Extractor
        self.projector = eqx.nn.Linear(128, TARGET_PARAM_COUNT, key=k2) # Mapper
        self.theta_base = dummy_inr_params

    def __call__(self, x, coords, key=None):
        features = self.encoder(x)
        delta_theta = self.projector(features)
        theta = self.theta_base + delta_theta 
        x_hat = render_image(theta, coords)
        return x_hat, theta 

class FunctionalBottleneckAE(eqx.Module):
    """Our Method: CNN -> 2D Latent -> INR Weights (~330k params)"""
    encoder: CNNEncoder # Shared!
    weight_generator: eqx.nn.MLP
    theta_base: jax.Array

    def __init__(self, dummy_inr_params, key):
        self.encoder = CNNEncoder(latent_dim=2, key=key)
        # self.encoder = eqx.nn.MLP(H*W, 2, width_size=256, depth=2, key=key) # Strictly 2D latent space for direct comparison
        self.weight_generator = eqx.nn.MLP(in_size=2, out_size=TARGET_PARAM_COUNT, width_size=256, depth=2, key=key)
        self.theta_base = dummy_inr_params

    def __call__(self, x, coords, key=None):
        # x = x.flatten()
        z = self.encoder(x)
        delta_theta = self.weight_generator(z)
        theta = self.theta_base + delta_theta
        x_hat = render_image(theta, coords)
        return x_hat, z

# ==========================================
# 5. TRAINING ENGINES
# ==========================================
def create_trainer(model_instance, lr=1e-3, is_vae=False):
    optimizer = optax.adam(lr) 
    filter_spec = jax.tree_util.tree_map(eqx.is_array, model_instance)
    
    diff_model, static_model = eqx.partition(model_instance, filter_spec)
    opt_state = optimizer.init(diff_model)

    if is_vae:
        @eqx.filter_value_and_grad
        def loss_fn(diff, static, x, coords, key):
            model = eqx.combine(diff, static)
            keys = jax.random.split(key, x.shape[0])
            x_hat, _, mu, logvar = jax.vmap(model, in_axes=(0, None, 0))(x, coords, keys)
            recon_loss = jnp.mean((x_hat - x) ** 2)
            kl_loss = -0.5 * jnp.mean(jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1))
            return recon_loss + (0.01 * kl_loss)
    else:
        @eqx.filter_value_and_grad
        def loss_fn(diff, static, x, coords, key):
            model = eqx.combine(diff, static)
            x_hat, _ = jax.vmap(model, in_axes=(0, None, None))(x, coords, None)
            return jnp.mean((x_hat - x) ** 2)

    @eqx.filter_jit
    def make_step(diff, static, opt_state, x, coords, key):
        loss, grads = loss_fn(diff, static, x, coords, key)
        updates, opt_state = optimizer.update(grads, opt_state, diff)
        diff = eqx.apply_updates(diff, updates)
        return diff, opt_state, loss

    @eqx.filter_jit
    def evaluate(diff, static, x, coords):
        model = eqx.combine(diff, static)
        if is_vae:
            x_hat, _, _, _ = jax.vmap(model, in_axes=(0, None, None))(x, coords, None)
        else:
            x_hat, _ = jax.vmap(model, in_axes=(0, None, None))(x, coords, None)
        return jnp.mean((x_hat - x) ** 2)

    return diff_model, static_model, opt_state, make_step, evaluate

# ==========================================
# 6. EXECUTION SCRIPT
# ==========================================
#%%
config = {
    "lr": 1e-4,
    "batch_size": 128,
    "epochs": 50,
    "seed": 42
}

key = jax.random.PRNGKey(config["seed"])
keys = jax.random.split(key, 6)

dm = MNISTDataHandler(batch_size=config["batch_size"], seed=config["seed"], subset_size=None)

models = {
    "Dense_AE": DenseAE(keys[1]),
    "CNN_AE": CNNAE(key=keys[2]),
    "CNN_VAE": CNN_VAE(keys[3]),
    "Direct_HyperNet": DirectHyperNet(dummy_inr_params=flat_params, key=keys[4]),
    "Functional_Bottleneck_AE": FunctionalBottleneckAE(dummy_inr_params=flat_params, key=keys[5])
}

print(f"\nTarget INR Parameters: {TARGET_PARAM_COUNT}")
print("-" * 50)
print("Architectural Complexity:")
for name, model in models.items():
    print(f"  {name:25s}: {count_params(model):,} params")
print("-" * 50)

#%%
# History structure configured to your exact tracking specifications
history = {m: {'train_loss': [], 'val_steps': [], 'val_loss': []} for m in models.keys()}
run_dir = get_run_path()
print(f"\n🚀 Initiating Training. Logs at: {run_dir.name}")

trainers = {}
for name, model in models.items():
    diff, stat, opt, step_fn, eval_fn = create_trainer(model, config["lr"], is_vae=(name == "CNN_VAE"))
    trainers[name] = {"diff": diff, "static": stat, "opt": opt, "step": step_fn, "eval": eval_fn}

start_time = time.time()
global_step = 0

for epoch in range(config["epochs"]):
    # --- Training Loop ---
    for x, _ in dm.train_loader:
        global_step += 1
        x_jax = jnp.array(x)
        key, step_key = jax.random.split(key)
        
        for name, t in trainers.items():
            t["diff"], t["opt"], loss = t["step"](t["diff"], t["static"], t["opt"], x_jax, dm.coords, step_key)
            # Append directly; index implicitly acts as the training step
            history[name]['train_loss'].append(float(loss))
            
    # --- Validation Loop ---
    val_losses = {m: [] for m in models.keys()}
    for x_val, _ in dm.val_loader:
        x_val_jax = jnp.array(x_val)
        for name, t in trainers.items():
            v_loss = t["eval"](t["diff"], t["static"], x_val_jax, dm.coords)
            val_losses[name].append(float(v_loss))
            
    print(f"Epoch {epoch+1:02d}/{config['epochs']} | Val MSE: ", end="")
    for name in trainers.keys():
        avg_val = np.mean(val_losses[name])
        # Record the global step to correctly align the val point over the train curves
        history[name]['val_steps'].append(global_step)
        history[name]['val_loss'].append(avg_val)
        print(f"{name} [{avg_val:.4f}] | ", end="")
    print()

print(f"\n⏱ Training complete in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
final_models = {n: eqx.combine(t["diff"], t["static"]) for n, t in trainers.items()}

# ==========================================
# 7. VISUAL RECONSTRUCTIONS
# ==========================================
#%%
print("\n--- Visualizing Reconstructions ---")
test_images, test_labels = dm.get_full_eval_set('val', max_samples=8)

fig, axes = plt.subplots(len(models) + 1, 8, figsize=(16, 2 * (len(models) + 1)))
fig.suptitle("Model Reconstructions (Unseen Val Data)", fontsize=18)

for i in range(8):
    axes[0, i].imshow(test_images[i, 0], cmap='gray')
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title("Ground Truth", loc='left', fontsize=12)

for row_idx, (name, model) in enumerate(final_models.items(), start=1):
    batched_model = jax.vmap(model, in_axes=(0, None, None))
    outputs = batched_model(test_images, dm.coords, None) 
    recons = outputs[0]
    
    for i in range(8):
        axes[row_idx, i].imshow(np.array(recons[i, 0]), cmap='gray')
        axes[row_idx, i].axis('off')
        if i == 0: axes[row_idx, i].set_title(name, loc='left', fontsize=12)

plt.savefig(run_dir / "plots" / "reconstructions.png", dpi=150)
plt.show()

# ==========================================
# 8. STEP-WISE LOSS PLOTTING
# ==========================================
#%%
print("\n--- Plotting Step-wise Loss Curves ---")
fig, axes = plt.subplots(1, 5, figsize=(28, 5))
fig.suptitle("Training & Validation MSE by Training Step", fontsize=18)

for ax, name in zip(axes, models.keys()):
    # Get the raw lists
    train_losses = history[name]['train_loss']
    val_steps = history[name]['val_steps']
    val_losses = history[name]['val_loss']
    
    # Train loss index serves as the x-axis (0 to total_batches * epochs)
    ax.plot(range(len(train_losses)), train_losses, 
            label='Train', linestyle='-', alpha=0.6, linewidth=1.5, color='royalblue')
            
    # Val loss aligned using the recorded global step
    ax.plot(val_steps, val_losses, 
            label='Val', marker='o', linestyle='--', alpha=1.0, color='crimson', markersize=6)
            
    ax.set_title(name.replace('_', ' '), fontsize=14)
    ax.set_xlabel('Train Steps')
    if ax == axes[0]: ax.set_ylabel('MSE Loss')
    ax.set_yscale('log')
    ax.legend(loc="upper right")

plt.savefig(run_dir / "plots" / "step_loss_curves.png", dpi=150)
plt.show()

# ==========================================
# 9. MANIFOLD ANALYSIS 
# ==========================================
#%%
print("\n--- Extracting Representations & Running Manifold Projections ---")
X_eval, y_eval = dm.get_full_eval_set('val', max_samples=2500)
X_flat = np.array(X_eval).reshape(len(X_eval), -1) 

z_dict = {}
for name, model in final_models.items():
    batched_call = jax.vmap(model, in_axes=(0, None, None))
    outputs = batched_call(X_eval, dm.coords, None)
    z_dict[name] = np.array(outputs[1]) # The latent 'z' is always index 1

print("Computing PCA, t-SNE, and UMAP on Raw Pixels & Weight Spaces...")
pca_ambient = PCA(n_components=2).fit_transform(X_flat)
tsne_ambient = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(X_flat)
umap_ambient = umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(X_flat)

theta_nD = z_dict["Direct_HyperNet"]
pca_weights = PCA(n_components=2).fit_transform(theta_nD)
umap_weights = umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(theta_nD)

#%%
cmap = plt.colormaps['tab10'].resampled(10)

fig, axes = plt.subplots(3, 4, figsize=(24, 18))
fig.suptitle('Manifold Analysis: Ambient vs. Structural vs. Functional Reps', fontsize=22)

def scatter_manifold(ax, data, title):
    sc = ax.scatter(data[:, 0], data[:, 1], c=y_eval, cmap=cmap, s=8, alpha=0.7)
    ax.set_title(title, fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    return sc

scatter_manifold(axes[0, 0], pca_ambient, "Ambient Pixels -> PCA (2D)")
scatter_manifold(axes[0, 1], tsne_ambient, "Ambient Pixels -> t-SNE (2D)")
scatter_manifold(axes[0, 2], umap_ambient, "Ambient Pixels -> UMAP (2D)")
axes[0, 3].axis('off') 

scatter_manifold(axes[1, 0], z_dict["Dense_AE"], "Dense AE Latents (Native 2D)")
scatter_manifold(axes[1, 1], z_dict["CNN_AE"], "CNN AE Latents (Native 2D)")
scatter_manifold(axes[1, 2], z_dict["CNN_VAE"], "CNN-VAE Latents (Native 2D)") 
axes[1, 3].axis('off')

scatter_manifold(axes[2, 0], pca_weights, "Direct HyperNet Weights -> PCA (2D)")
scatter_manifold(axes[2, 1], umap_weights, "Direct HyperNet Weights -> UMAP (2D)")
sc = scatter_manifold(axes[2, 2], z_dict["Functional_Bottleneck_AE"], "F-BAE Latents (Native 2D)")
axes[2, 3].axis('off')

cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), ticks=range(10), fraction=0.015, pad=0.04)
cbar.set_label('MNIST Digit')

plt.savefig(run_dir / "plots" / "manifold_comparison.png", dpi=200)
plt.show()

# ==========================================
# 10. LATENT SPACE GENERATIVE TRAVERSAL
# ==========================================
#%%
print("\n--- Latent Traversals: Interpolating the Functional Bottleneck ---")
model_fbae = final_models["Functional_Bottleneck_AE"]
z_fbae = z_dict["Functional_Bottleneck_AE"]

x_min, x_max = np.percentile(z_fbae[:, 0], [5, 95])
y_min, y_max = np.percentile(z_fbae[:, 1], [5, 95])

grid_size = 12
x_grid = jnp.linspace(x_min, x_max, grid_size)
y_grid = jnp.linspace(y_min, y_max, grid_size)

grid_images = []
for y in reversed(y_grid): 
    row_images = []
    for x in x_grid:
        z_sample = jnp.array([x, y])
        theta = model_fbae.theta_base + model_fbae.weight_generator(z_sample)
        img = render_image(theta, dm.coords)
        row_images.append(np.array(img[0])) 
    grid_images.append(np.concatenate(row_images, axis=1))

full_grid = np.concatenate(grid_images, axis=0)

plt.figure(figsize=(10, 10))
plt.imshow(full_grid, cmap='gray')
plt.title("2D Latent Grid Traversal (Functional Bottleneck AE)", fontsize=16)
plt.axis('off')
plt.savefig(run_dir / "plots" / "fbae_latent_traversal.png", dpi=200)
plt.show()

print(f"\n✅ All tasks completed successfully. Artifacts saved in {run_dir.name}")
# %%