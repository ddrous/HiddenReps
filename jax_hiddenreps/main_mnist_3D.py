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
    run_id = f"fbae_mnist_3D_{int(time.time())}"
    run_path = Path(base_dir) / run_id
    run_path.mkdir()
    (run_path / "plots").mkdir()
    return run_path

def count_params(model):
    """Calculates the total number of trainable parameters in an Equinox module."""
    return sum(x.size for x in jax.tree_util.tree_leaves(model) if eqx.is_array(x))

# --- DYNAMIC INR PARAMETER EXTRACTION ---
# In_size=16 accommodates the 4-band Fourier feature projection
dummy_key = jax.random.PRNGKey(0)
dummy_inr = eqx.nn.MLP(in_size=16, out_size=1, width_size=16, depth=4, activation=jax.nn.relu, key=dummy_key)

inr_params, STATIC_INR = eqx.partition(dummy_inr, eqx.is_array)
flat_params, UNFLATTEN_FN = jax.flatten_util.ravel_pytree(inr_params)
TARGET_PARAM_COUNT = len(flat_params) 

def render_image(theta, coords):
    """Reconstructs the HxW image from the flat weight vector using Fourier features."""
    restored_params = UNFLATTEN_FN(theta)
    inr = eqx.combine(restored_params, STATIC_INR)
    
    def forward_pixel(c):
        # 1. Define frequency bands (4 bands -> 16D feature vector)
        num_freqs = 4
        freq_bands = 2.0 ** jnp.arange(num_freqs)
        
        # 2. Project coordinates
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
            transforms.Resize((H, W)),
            transforms.ToTensor(),
        ])
        
        full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        full_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        if subset_size:
            rng = np.random.default_rng(seed)
            train_idx = rng.choice(len(full_train), subset_size, replace=False)
            full_train = Subset(full_train, train_idx)
            
        self.train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate, drop_last=False)
        self.val_loader = DataLoader(full_test, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate, drop_last=False)
        
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
# 3. ARCHITECTURE COMPONENTS (Updated for 3D)
# ==========================================
class CNNBackbone(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d

    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(1, 32, 4, stride=2, padding=1, key=k1)  
        self.conv2 = eqx.nn.Conv2d(32, 64, 4, stride=2, padding=1, key=k2) 
        self.conv3 = eqx.nn.Conv2d(64, 128, 4, stride=2, padding=1, key=k3) 

    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        return x.flatten() 

class CNNEncoder(eqx.Module):
    backbone: CNNBackbone
    linear: eqx.nn.Linear

    # Default latent_dim is now 3
    def __init__(self, latent_dim=3, key=None):
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

    def __init__(self, latent_dim=3, key=None):
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
    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP
    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.encoder = eqx.nn.MLP(H*W, 3, width_size=128, depth=2, key=k1) # 3D
        self.decoder = eqx.nn.MLP(3, H*W, width_size=128, depth=2, key=k2) # 3D
    def __call__(self, x, coords=None, key=None):
        x_flat = x.flatten()
        z = self.encoder(x_flat)
        out = jax.nn.sigmoid(self.decoder(z))
        return out.reshape(1, H, W), z

class CNNAE(eqx.Module):
    encoder: CNNEncoder 
    decoder: CNNDecoder
    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.encoder = CNNEncoder(latent_dim=3, key=k1)
        self.decoder = CNNDecoder(latent_dim=3, key=k2)
    def __call__(self, x, coords=None, key=None):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

class CNN_VAE(eqx.Module):
    encoder: CNNEncoder
    decoder: CNNDecoder
    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.encoder = CNNEncoder(latent_dim=6, key=k1) # 3 mu, 3 logvar
        self.decoder = CNNDecoder(latent_dim=3, key=k2)
    def __call__(self, x, coords=None, key=None):
        h = self.encoder(x)
        mu, logvar = h[:3], h[3:]
        if key is not None:
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(key, std.shape)
            z = mu + eps * std
        else:
            z = mu 
        out = self.decoder(z)
        return out, z, mu, logvar

class DirectHyperNet(eqx.Module):
    encoder: CNNEncoder
    projector: eqx.nn.Linear
    theta_base: jax.Array
    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.encoder = CNNEncoder(latent_dim=128, key=k1) 
        self.projector = eqx.nn.Linear(128, TARGET_PARAM_COUNT, key=k2) 
        self.theta_base = jnp.zeros(TARGET_PARAM_COUNT)
    def __call__(self, x, coords, key=None):
        features = self.encoder(x)
        delta_theta = self.projector(features)
        theta = self.theta_base + delta_theta 
        x_hat = render_image(theta, coords)
        return x_hat, theta 

class FunctionalBottleneckAE(eqx.Module):
    encoder: CNNEncoder 
    weight_generator: eqx.nn.MLP
    theta_base: jax.Array
    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.encoder = CNNEncoder(latent_dim=3, key=k1)
        self.weight_generator = eqx.nn.MLP(in_size=3, out_size=TARGET_PARAM_COUNT, width_size=256, depth=2, key=k2)
        self.theta_base = jnp.zeros(TARGET_PARAM_COUNT)
    def __call__(self, x, coords, key=None):
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
    "epochs": 10,
    "seed": 42
}

key = jax.random.PRNGKey(config["seed"])
keys = jax.random.split(key, 6)

dm = MNISTDataHandler(batch_size=config["batch_size"], seed=config["seed"], subset_size=None)

# FBAE and CNNAE independently instantiated, but with identical architectures and capacities
models = {
    "Dense_AE": DenseAE(keys[0]),
    "CNN_AE": CNNAE(key=keys[1]),
    "CNN_VAE": CNN_VAE(keys[2]),
    "Direct_HyperNet": DirectHyperNet(keys[3]),
    "Functional_Bottleneck_AE": FunctionalBottleneckAE(key=keys[4])
}

print(f"\nTarget INR Parameters: {TARGET_PARAM_COUNT}")
print("-" * 50)
print("Architectural Complexity:")
for name, model in models.items():
    print(f"  {name:25s}: {count_params(model):,} params")
print("-" * 50)

#%%
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
    for x, _ in dm.train_loader:
        global_step += 1
        x_jax = jnp.array(x)
        key, step_key = jax.random.split(key)
        
        for name, t in trainers.items():
            t["diff"], t["opt"], loss = t["step"](t["diff"], t["static"], t["opt"], x_jax, dm.coords, step_key)
            history[name]['train_loss'].append(float(loss))
            
    val_losses = {m: [] for m in models.keys()}
    for x_val, _ in dm.val_loader:
        x_val_jax = jnp.array(x_val)
        for name, t in trainers.items():
            v_loss = t["eval"](t["diff"], t["static"], x_val_jax, dm.coords)
            val_losses[name].append(float(v_loss))
            
    print(f"Epoch {epoch+1:02d}/{config['epochs']} | Val MSE: ", end="")
    for name in trainers.keys():
        avg_val = np.mean(val_losses[name])
        history[name]['val_steps'].append(global_step)
        history[name]['val_loss'].append(avg_val)
        print(f"{name} [{avg_val:.4f}] | ", end="")
    print()

print(f"\n⏱ Training complete in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
final_models = {n: eqx.combine(t["diff"], t["static"]) for n, t in trainers.items()}

# ==========================================
# 7. VISUAL RECONSTRUCTIONS & LOSS
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

#%%
print("\n--- Plotting Step-wise Loss Curves ---")
fig, axes = plt.subplots(1, 5, figsize=(28, 5))
fig.suptitle("Training & Validation MSE by Training Step", fontsize=18)

for ax, name in zip(axes, models.keys()):
    train_losses = history[name]['train_loss']
    val_steps = history[name]['val_steps']
    val_losses = history[name]['val_loss']
    
    if len(train_losses) <= 1:
        ax.scatter(range(len(train_losses)), train_losses, color='royalblue', marker='x', label='Train')
    else:
        ax.plot(range(len(train_losses)), train_losses, label='Train', linestyle='-', alpha=0.6, linewidth=1.5, color='royalblue')
            
    ax.plot(val_steps, val_losses, label='Val', marker='o', linestyle='--', alpha=1.0, color='crimson', markersize=6)
            
    ax.set_title(name.replace('_', ' '), fontsize=14)
    ax.set_xlabel('Train Steps')
    if ax == axes[0]: ax.set_ylabel('MSE Loss')
    ax.set_yscale('log')
    ax.legend(loc="upper right")

plt.savefig(run_dir / "plots" / "step_loss_curves.png", dpi=150)
plt.show()

# ==========================================
# 8. MANIFOLD ANALYSIS (Native 3D)
# ==========================================
#%%
print("\n--- Extracting Representations ---")
X_eval, y_eval = dm.get_full_eval_set('val', max_samples=2500)
X_flat = np.array(X_eval).reshape(len(X_eval), -1) 

z_dict = {}
for name, model in final_models.items():
    batched_call = jax.vmap(model, in_axes=(0, None, None))
    outputs = batched_call(X_eval, dm.coords, None)
    z_dict[name] = np.array(outputs[1]) 

print("\n--- Plotting Native 3D Latent Spaces ---")
cmap = plt.colormaps['tab10'].resampled(10)

fig = plt.figure(figsize=(24, 6))
fig.suptitle('Native 3D Representations on Unseen Data', fontsize=22)

# Only plot models that have a genuine 3D bottleneck
bottleneck_models = ["Dense_AE", "CNN_AE", "CNN_VAE", "Functional_Bottleneck_AE"]

for idx, name in enumerate(bottleneck_models, 1):
    ax = fig.add_subplot(1, 4, idx, projection='3d')
    z_3d = z_dict[name]
    sc = ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], c=y_eval, cmap=cmap, s=8, alpha=0.7)
    
    ax.set_title(name.replace('_', ' '), fontsize=14)
    ax.set_xlabel("z1"); ax.set_ylabel("z2"); ax.set_zlabel("z3")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

cbar = fig.colorbar(sc, ax=fig.axes, ticks=range(10), fraction=0.015, pad=0.04)
cbar.set_label('MNIST Digit')
plt.savefig(run_dir / "plots" / "native_3d_manifolds.png", dpi=200)
plt.show()

# ==========================================
# 9. MANIFOLD ANALYSIS (Massive 2D Compressions)
# ==========================================
#%%
print("\n--- Computing Aggressive 2D Reductions Across All Feature Spaces ---")

# Gather all feature spaces (from 1024D ambient to 600D weights to 3D latents)
feature_spaces = {
    "Ambient Pixels": X_flat,
    "Dense AE (3D)": z_dict["Dense_AE"],
    "CNN AE (3D)": z_dict["CNN_AE"],
    "CNN VAE (3D)": z_dict["CNN_VAE"],
    "Direct HyperNet (Weights)": z_dict["Direct_HyperNet"],
    "Functional Bottleneck (3D)": z_dict["Functional_Bottleneck_AE"]
}

algorithms = {
    "PCA": PCA(n_components=2),
    "t-SNE": TSNE(n_components=2, perplexity=40, random_state=42),
    "UMAP": umap.UMAP(n_components=2, random_state=42, n_jobs=1)
}

reductions = {alg: {} for alg in algorithms.keys()}

# Compute all reductions
for algo_name, algo in algorithms.items():
    print(f"Fitting {algo_name}...")
    for space_name, data in feature_spaces.items():
        reductions[algo_name][space_name] = algo.fit_transform(data)

def plot_giant_reduction_grid(algo_name, results_dict, filename):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{algo_name} 2D Projection Comparison Across All Models', fontsize=22)
    
    for ax, (space_name, data_2d) in zip(axes.flatten(), results_dict.items()):
        sc = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=y_eval, cmap=cmap, s=5, alpha=0.7)
        ax.set_title(space_name, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
        
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), ticks=range(10), fraction=0.015, pad=0.04)
    cbar.set_label('MNIST Digit')
    plt.savefig(run_dir / "plots" / filename, dpi=200)
    plt.show()

plot_giant_reduction_grid("PCA", reductions["PCA"], "giant_pca_grid.png")
plot_giant_reduction_grid("t-SNE", reductions["t-SNE"], "giant_tsne_grid.png")
plot_giant_reduction_grid("UMAP", reductions["UMAP"], "giant_umap_grid.png")

# ==========================================
# 10. LATENT SPACE GENERATIVE TRAVERSAL (3D Slices)
# ==========================================
#%%
print("\n--- 3D Latent Traversals: Interpolating the Functional Bottleneck ---")
model_fbae = final_models["Functional_Bottleneck_AE"]
z_fbae = z_dict["Functional_Bottleneck_AE"]

# To visualize 3D, we create a 2D (z1, z2) grid and render it at 3 different depth slices (z3)
z1_min, z1_max = np.percentile(z_fbae[:, 0], [5, 95])
z2_min, z2_max = np.percentile(z_fbae[:, 1], [5, 95])

# Choose 3 slices for depth (z3): 10th percentile, 50th (median), 90th percentile
z3_slices = np.percentile(z_fbae[:, 2], [10, 50, 90])
slice_labels = ["10th %ile Depth", "Median Depth", "90th %ile Depth"]

grid_size = 10
x_grid = jnp.linspace(z1_min, z1_max, grid_size)
y_grid = jnp.linspace(z2_min, z2_max, grid_size)

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle("F-BAE 3D Latent Grid Traversals (Slicing along Z3)", fontsize=22)

for ax_idx, z3_val in enumerate(z3_slices):
    grid_images = []
    for y in reversed(y_grid): 
        row_images = []
        for x in x_grid:
            z_sample = jnp.array([x, y, z3_val]) # Construct the full 3D coordinate
            theta = model_fbae.theta_base + model_fbae.weight_generator(z_sample)
            img = render_image(theta, dm.coords)
            row_images.append(np.array(img[0])) 
        grid_images.append(np.concatenate(row_images, axis=1))

    full_grid = np.concatenate(grid_images, axis=0)
    axes[ax_idx].imshow(full_grid, cmap='gray')
    axes[ax_idx].set_title(f"Slice {ax_idx+1}: z3 = {slice_labels[ax_idx]}", fontsize=16)
    axes[ax_idx].axis('off')

plt.savefig(run_dir / "plots" / "fbae_3d_latent_traversal.png", dpi=200)
plt.show()

print(f"\n✅ All tasks completed successfully. Artifacts saved in {run_dir.name}")
# %%