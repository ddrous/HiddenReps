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
import json
import os
import urllib.request
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

sns.set_theme(style="whitegrid")

# ==========================================
# 1. UTILITIES & DYNAMIC INR SETUP
# ==========================================
def get_run_path(base_dir="./runs"):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    run_id = f"run_{int(time.time())}"
    run_path = Path(base_dir) / run_id
    run_path.mkdir()
    (run_path / "plots").mkdir()
    return run_path

def plot_step_loss_curves(history, run_dir, title="Autoencoder Loss Curves (Per Step)"):
    plt.figure(figsize=(12, 6))
    for model_name, hist in history.items():
        # Plot every single batch step
        plt.plot(hist['train_step'], label=f"{model_name}", alpha=0.7, linewidth=1.0)
    plt.title(title, fontsize=14)
    plt.xlabel("Training Step (Batch)")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(run_dir / "plots" / "loss_curves_steps.png", dpi=100)
    plt.show()

# --- DYNAMIC INR PARAMETER EXTRACTION ---
# Define the deep, narrow INR structure
dummy_key = jax.random.PRNGKey(0)
dummy_inr = eqx.nn.MLP(in_size=2, out_size=1, width_size=12, depth=5, activation=jax.nn.relu, key=dummy_key)

# Extract only the learnable arrays (weights/biases) and separate the static structure (activation functions, etc.)
inr_params, STATIC_INR = eqx.partition(dummy_inr, eqx.is_array)

# Flatten the parameters into a 1D vector and get the exact reconstruction function
flat_params, UNFLATTEN_FN = jax.flatten_util.ravel_pytree(inr_params)
TARGET_PARAM_COUNT = len(flat_params)
print(f"Dynamically calculated Target INR Parameters: {TARGET_PARAM_COUNT}")

def render_image(theta, coords):
    """
    Takes a 1D vector `theta`, unflattens it back into the Deep/Narrow MLP,
    and evaluates it over the given (x,y) coordinates to render the image.
    """
    # 1. Reconstruct the PyTree of weights using the dynamically generated function
    restored_params = UNFLATTEN_FN(theta)
    # 2. Re-combine weights with the static architecture
    inr = eqx.combine(restored_params, STATIC_INR)
    
    # 3. Define the per-pixel forward pass (with Sigmoid to output [0, 1])
    def forward_pixel(c):
        return jax.nn.sigmoid(inr(c))[0] 
    
    # 4. Map over all 4096 coordinates
    pixels = jax.vmap(forward_pixel)(coords)
    return pixels.reshape(64, 64, 1)

# ==========================================
# 2. DATA HANDLER (dSprites with NPZ)
# ==========================================
class DSpritesDataHandler:
    ### Accessible at https://github.com/google-deepmind/dsprites-dataset
    def __init__(self, batch_size=64, subset_size=10000, seed=42):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        
        # Checking for standard or user-specified NPZ
        self.filepaths = ['dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz']
        
        self._load_and_split(subset_size)
        
        # Precompute coordinate grid [-1, 1] for the INR evaluation
        xs = jnp.linspace(-1, 1, 64)
        ys = jnp.linspace(-1, 1, 64)
        X, Y = jnp.meshgrid(xs, ys)
        self.coords = jnp.stack([X.flatten(), Y.flatten()], axis=-1)

    def _load_and_split(self, subset_size):
        filepath = next((fp for fp in self.filepaths if os.path.exists(fp)), None)
        
        if filepath is None:
            print("Downloading dSprites dataset (~25MB)...")
            url = "https://github.com/google-deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
            filepath = self.filepaths[0]
            urllib.request.urlretrieve(url, filepath)
            print("Download complete.")

        print(f"Loading dSprites data from {filepath}...")
        data = np.load(filepath, allow_pickle=False)
        
        # Extract full data
        imgs = data['imgs']
        latents = data['latents_values']
        print(f"Full dataset loaded. Total samples: {len(imgs)}")
        
        # Subsample to keep memory reasonable
        idx = self.rng.choice(len(latents), subset_size, replace=False)
        latents_sub = latents[idx]
        imgs_sub = imgs[idx]
        
        # Create ID Holdout Gap (PosX is index 4, ranges [0, 1])
        gap_mask = (latents_sub[:, 4] > 0.4) & (latents_sub[:, 4] < 0.6)
        
        self.train_data = np.expand_dims(imgs_sub[~gap_mask], -1).astype(np.float32)
        self.train_latents = latents_sub[~gap_mask].astype(np.float32)
        
        self.test_data = np.expand_dims(imgs_sub[gap_mask], -1).astype(np.float32)
        self.test_latents = latents_sub[gap_mask].astype(np.float32)
            
        print(f"Data ready. Train size: {len(self.train_data)} | Held-out ID Test size: {len(self.test_data)}")

    def get_iterator(self):
        idx = np.arange(len(self.train_data))
        np.random.shuffle(idx)
        for i in range(0, len(self.train_data), self.batch_size):
            batch = idx[i:i + self.batch_size]
            yield jnp.array(self.train_data[batch]), jnp.array(self.train_latents[batch])

    def get_full_data(self, split='train', max_samples=2500):
        d, l = (self.train_data, self.train_latents) if split == 'train' else (self.test_data, self.test_latents)
        size = min(len(d), max_samples)
        return jnp.array(d[:size]), jnp.array(l[:size])


# ==========================================
# 3. ENCODER ARCHITECTURES
# ==========================================
class CNNEncoder(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    linear: eqx.nn.Linear

    def __init__(self, latent_dim=6, key=None):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(1, 16, 4, stride=2, padding=1, key=k1) # 64 -> 32
        self.conv2 = eqx.nn.Conv2d(16, 32, 4, stride=2, padding=1, key=k2) # 32 -> 16
        self.conv3 = eqx.nn.Conv2d(32, 64, 4, stride=2, padding=1, key=k3) # 16 -> 8
        self.linear = eqx.nn.Linear(64 * 8 * 8, latent_dim, key=k4)

    def __call__(self, x):
        # Convert (64, 64, 1) to (1, 64, 64) for Equinox Conv2d
        x = jnp.transpose(x, (2, 0, 1))
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = x.flatten()
        return self.linear(x)

# ==========================================
# 4. MODELS
# ==========================================
class AbstractAE(eqx.Module):
    """Standard Baseline: CNN -> 6D -> Transposed CNN"""
    encoder: CNNEncoder
    linear_up: eqx.nn.Linear
    deconv1: eqx.nn.ConvTranspose2d
    deconv2: eqx.nn.ConvTranspose2d
    deconv3: eqx.nn.ConvTranspose2d

    def __init__(self, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.encoder = CNNEncoder(latent_dim=6, key=k1)
        self.linear_up = eqx.nn.Linear(6, 64 * 8 * 8, key=k2)
        self.deconv1 = eqx.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, key=k3) 
        self.deconv2 = eqx.nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, key=k4) 
        self.deconv3 = eqx.nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1, key=k5)  

    def __call__(self, x, coords=None):
        z = self.encoder(x)
        h = jax.nn.relu(self.linear_up(z))
        h = h.reshape((64, 8, 8))
        h = jax.nn.relu(self.deconv1(h))
        h = jax.nn.relu(self.deconv2(h))
        out = jax.nn.sigmoid(self.deconv3(h))
        return jnp.transpose(out, (1, 2, 0)), z

class DirectHyperNet(eqx.Module):
    """CNN directly to Weights (No explicit 6D bottleneck)"""
    hyper_encoder: CNNEncoder
    theta_base: jax.Array
    learn_base: bool

    def __init__(self, key, learn_base=True):
        self.hyper_encoder = CNNEncoder(latent_dim=TARGET_PARAM_COUNT, key=key)
        self.theta_base = jnp.zeros(TARGET_PARAM_COUNT)
        self.learn_base = learn_base

    def __call__(self, x, coords):
        delta_theta = self.hyper_encoder(x)
        # Combine learned global base with predicted sample offset
        theta = (self.theta_base + delta_theta) if self.learn_base else delta_theta
        x_hat = render_image(theta, coords)
        return x_hat, theta 

class BottleneckHyperNet(eqx.Module):
    """CNN -> 6D Latent -> Weights (Rigorous Structure mapping offset)"""
    latent_encoder: CNNEncoder
    weight_generator: eqx.nn.MLP
    theta_base: jax.Array
    learn_base: bool

    def __init__(self, key, learn_base=True):
        k1, k2 = jax.random.split(key)
        self.latent_encoder = CNNEncoder(latent_dim=6, key=k1)
        self.weight_generator = eqx.nn.MLP(6, TARGET_PARAM_COUNT, 128, 2, key=k2)
        self.theta_base = jnp.zeros(TARGET_PARAM_COUNT)
        self.learn_base = learn_base

    def __call__(self, x, coords):
        z = self.latent_encoder(x) # Strict 6D Bottleneck
        delta_theta = self.weight_generator(z)
        theta = (self.theta_base + delta_theta) if self.learn_base else delta_theta
        x_hat = render_image(theta, coords)
        return x_hat, z

# ==========================================
# 5. TRAINING LOGIC
# ==========================================
def create_trainer(model_instance, lr):
    optimizer = optax.adam(lr)
    
    filter_spec = jax.tree_util.tree_map(eqx.is_array, model_instance)
    # Freeze theta_base if user configured it
    if hasattr(model_instance, 'learn_base') and not model_instance.learn_base:
        filter_spec = eqx.tree_at(lambda m: m.theta_base, filter_spec, False)

    diff_model, static_model = eqx.partition(model_instance, filter_spec)
    opt_state = optimizer.init(diff_model)

    @eqx.filter_value_and_grad
    def loss_fn(diff, static, x, coords):
        model = eqx.combine(diff, static)
        # vmap over batch dimension (0) for x, coords are shared (None)
        batched_call = jax.vmap(model, in_axes=(0, None))
        x_hat, _ = batched_call(x, coords)
        return jnp.mean((x_hat - x) ** 2)

    @eqx.filter_jit
    def make_step(diff, static, opt_state, x, coords):
        loss, grads = loss_fn(diff, static, x, coords)
        updates, opt_state = optimizer.update(grads, opt_state, diff)
        diff = eqx.apply_updates(diff, updates)
        return diff, opt_state, loss

    return diff_model, static_model, opt_state, make_step

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
#%%
config = {
    "lr": 1e-5,
    "batch_size": 16,
    "epochs": 100,
    "val_every": 10,
    "subset_size": 737280, 
    "seed": 2026
}

key = jax.random.PRNGKey(config["seed"])
k1, k2, k3 = jax.random.split(key, 3)

dm = DSpritesDataHandler(batch_size=config["batch_size"], subset_size=config["subset_size"], seed=config["seed"])

models = {
    "Abstract_AE": AbstractAE(k1),
    "Direct_HyperNet": DirectHyperNet(k2, learn_base=True),
    "Bottleneck_HyperNet": BottleneckHyperNet(k3, learn_base=True)
}

history = {m: {'train_step': [], 'epoch_avg': []} for m in models.keys()}
run_dir = get_run_path()
print(f"\n🚀 Training Models. Run ID: {run_dir.name}")

trainers = {}
for name, model in models.items():
    diff, stat, opt, step_fn = create_trainer(model, config["lr"])
    trainers[name] = {"diff": diff, "static": stat, "opt": opt, "step": step_fn}

start_time = time.time()
for epoch in range(config["epochs"]):
    batch_losses = {m: [] for m in models.keys()}
    
    for x, _ in dm.get_iterator():
        for name, t in trainers.items():
            t["diff"], t["opt"], loss = t["step"](t["diff"], t["static"], t["opt"], x, dm.coords)
            loss_val = float(loss)
            batch_losses[name].append(loss_val)
            history[name]['train_step'].append(loss_val) # Store per step for granular plot
            
    if epoch % config["val_every"] == 0 or epoch == config["epochs"] - 1:
        print(f"Epoch {epoch:02d} | ", end="")
        for name in trainers.keys():
            avg_loss = np.mean(batch_losses[name])
            history[name]['epoch_avg'].append(avg_loss)
            print(f"{name}: [L: {avg_loss:.4f}] | ", end="")
        print()

print(f"\n⏱ Training completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

final_models = {n: eqx.combine(t["diff"], t["static"]) for n, t in trainers.items()}
plot_step_loss_curves(history, run_dir, title="dSprites Training Loss (Per Step)")




# ==========================================
# 7.5 ANALYSIS & PLOTTING LATENTS (2x4 Grid)
# ==========================================
#%%
print("\n--- Extracting Functional Representations ---")
X_full, true_latents_full = dm.get_full_data('train', max_samples=2500)
X_flat = np.array(X_full).reshape(len(X_full), -1) 

batched_abs = jax.vmap(final_models["Abstract_AE"], in_axes=(0, None))
_, z_abs = batched_abs(X_full, dm.coords)
z_abs = np.array(z_abs)

batched_dir = jax.vmap(final_models["Direct_HyperNet"], in_axes=(0, None))
_, theta_nD = batched_dir(X_full, dm.coords)
theta_nD = np.array(theta_nD)

batched_bot = jax.vmap(final_models["Bottleneck_HyperNet"], in_axes=(0, None))
_, z_explicit = batched_bot(X_full, dm.coords)
z_explicit = np.array(z_explicit)

print("Running PCA, t-SNE, and UMAP on Ambient Pixels (4096D) & Weight Spaces...")
# Fit Ambient PCA to 6 dimensions to allow flexible slicing
pca_ambient = PCA(n_components=6).fit_transform(X_flat)
tsne_ambient = TSNE(n_components=3, perplexity=30, random_state=42, method='barnes_hut').fit_transform(X_flat)
umap_ambient = umap.UMAP(n_components=6, random_state=42).fit_transform(X_flat)

# Option A: rigorously extract 6 dimensions from the Target Weights first
pca_model_6d = PCA(n_components=6).fit(theta_nD)
pca_weights_6d = pca_model_6d.transform(theta_nD)

# Extract standard t-SNE in 2D
tsne_weights = TSNE(n_components=3, perplexity=30, random_state=42, method='barnes_hut').fit_transform(theta_nD)

#%%
# -------------------------------------------------------------------------
# CONFIGURATION: Select which two dimensions to plot for each representation
# For 6D arrays (true, ambient_pca, z_abs, pca_weights, z_explicit): pick any 0-5
# For 2D arrays (t-SNE, UMAP): leave as (0, 1)
# -------------------------------------------------------------------------
plot_dims = {
    "true": (4, 5),          # dSprites factors (4=PosX, 5=PosY)
    "ambient_pca": (4, 5),
    "ambient_tsne": (1, 2),
    "ambient_umap": (4, 5),
    "z_abs": (4, 5),
    "pca_weights": (4, 5),
    "tsne_weights": (1, 2),
    "z_explicit": (0, 1)
}

print("Generating the Functional Latent Analysis Grid...")
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('Evaluating Representation Paradigms on dSprites Manifold', fontsize=16)

# PosX (index 4) dictates the color mapping to show spatial preservation
color_metric = true_latents_full[:, 4] 

plots = [
    # Top Row: Ambient & Ground Truth
    (true_latents_full[:, list(plot_dims["true"])], f"True Physical Latents (Dims {plot_dims['true'][0]}, {plot_dims['true'][1]})", axes[0, 0]),
    (pca_ambient[:, list(plot_dims["ambient_pca"])], f"Ambient PCA (Dims {plot_dims['ambient_pca'][0]}, {plot_dims['ambient_pca'][1]})", axes[0, 1]),
    (tsne_ambient[:, list(plot_dims["ambient_tsne"])], f"Ambient t-SNE (Dims {plot_dims['ambient_tsne'][0]}, {plot_dims['ambient_tsne'][1]})", axes[0, 2]),
    (umap_ambient[:, list(plot_dims["ambient_umap"])], f"Ambient UMAP (Dims {plot_dims['ambient_umap'][0]}, {plot_dims['ambient_umap'][1]})", axes[0, 3]),
    
    # Bottom Row: Model Latents explicitly sliced
    (z_abs[:, list(plot_dims["z_abs"])], f"Baseline: Abstract AE (Dims {plot_dims['z_abs'][0]}, {plot_dims['z_abs'][1]})", axes[1, 0]),
    (pca_weights_6d[:, list(plot_dims["pca_weights"])], f"Option A: PCA Weights (Dims {plot_dims['pca_weights'][0]}, {plot_dims['pca_weights'][1]})", axes[1, 1]),
    (tsne_weights[:, list(plot_dims["tsne_weights"])], f"Option A: t-SNE Target Weights (Dims {plot_dims['tsne_weights'][0]}, {plot_dims['tsne_weights'][1]})", axes[1, 2]),
    (z_explicit[:, list(plot_dims["z_explicit"])], f"Option B: Explicit 6D Bottleneck (Dims {plot_dims['z_explicit'][0]}, {plot_dims['z_explicit'][1]})", axes[1, 3]),
]

for data, title, ax in plots:
    ax.scatter(data[:, 0], data[:, 1], c=color_metric, cmap='viridis', s=15, alpha=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(run_dir / "plots" / "dsprites_latent_analysis.png", dpi=150)
plt.show()




# ==========================================
# 8. ID GENERALIZATION TESTING
# ==========================================
#%%
print("\n--- Running ID Generalization Test (Interpolating PosX Gap) ---")
X_test_full, test_latents_full = dm.get_full_data('test', max_samples=1000)

_, z_abs_test = batched_abs(X_test_full, dm.coords)
_, z_explicit_test = batched_bot(X_test_full, dm.coords)
_, theta_nD_test = batched_dir(X_test_full, dm.coords)

# Project test weights using the strictly fitted 6D PCA model
pca_weights_test_6d = pca_model_6d.transform(np.array(theta_nD_test))

# Slice out the chosen dimensions uniformly for test data based on the config
sliced_z_abs = z_abs[:, list(plot_dims["z_abs"])]
sliced_z_abs_test = np.array(z_abs_test)[:, list(plot_dims["z_abs"])]

sliced_pca_weights = pca_weights_6d[:, list(plot_dims["pca_weights"])]
sliced_pca_weights_test = pca_weights_test_6d[:, list(plot_dims["pca_weights"])]

sliced_z_explicit = z_explicit[:, list(plot_dims["z_explicit"])]
sliced_z_explicit_test = np.array(z_explicit_test)[:, list(plot_dims["z_explicit"])]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('ID Generalization: Safely Bridging the PosX Gap', fontsize=16)

id_plots = [
    (sliced_z_abs, sliced_z_abs_test, f"Abstract AE Interpolation (Dims {plot_dims['z_abs'][0]}, {plot_dims['z_abs'][1]})", axes[0]),
    (sliced_pca_weights, sliced_pca_weights_test, f"Option A (PCA Weights) Interpolation (Dims {plot_dims['pca_weights'][0]}, {plot_dims['pca_weights'][1]})", axes[1]),
    (sliced_z_explicit, sliced_z_explicit_test, f"Option B (Bottleneck) Interpolation (Dims {plot_dims['z_explicit'][0]}, {plot_dims['z_explicit'][1]})", axes[2]),
]

for train_z, test_z, title, ax in id_plots:
    # Plot Training Data lightly in the background
    ax.scatter(np.array(train_z)[:, 0], np.array(train_z)[:, 1], 
               c='lightgrey', s=10, alpha=0.3, label='Train Clusters')
    # Overlay Unseen Interpolation points
    ax.scatter(np.array(test_z)[:, 0], np.array(test_z)[:, 1], 
               c=test_latents_full[:, 4], cmap='magma', s=15, alpha=0.9, label='Unseen Interpolation Data')
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

axes[2].legend(loc='lower right')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(run_dir / "plots" / "id_generalization_test.png", dpi=150)
plt.show()
print("✅ ID Generalization complete.")
# %%
