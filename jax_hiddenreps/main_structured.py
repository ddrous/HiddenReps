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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

sns.set_theme(style="whitegrid")

# ==========================================
# 1. UTILITIES & CONFIG
# ==========================================
TARGET_PARAM_COUNT = 83 # 1*16 + 16 + 16*3 + 3

def get_run_path(base_dir="./runs"):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    run_id = f"run_{int(time.time())}"
    run_path = Path(base_dir) / run_id
    run_path.mkdir()
    (run_path / "plots").mkdir()
    return run_path

def plot_loss_curves(history, run_dir, title="Autoencoder Loss Curves"):
    plt.figure(figsize=(10, 6))
    for model_name, hist in history.items():
        plt.plot(hist['train'], label=f"{model_name} (Train)", linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(run_dir / "plots" / "loss_curves.png", dpi=100)
    plt.show()

# ==========================================
# 2. DATA HANDLER (Disjoint Spirals & ID Testing)
# ==========================================
class SpiralDataHandler:
    def __init__(self, batch_size=64, num_points=1500, num_clusters=6, val_split=0.2, cluster_spread=0.2, seed=2018):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        
        points_per_cluster = num_points // num_clusters
        center_thetas = self.rng.uniform(0.5 * np.pi, 3.5 * np.pi, num_clusters)
        center_heights = self.rng.uniform(-2.0, 2.0, num_clusters)

        all_data, all_labels, all_latents = [], [], []
        for i in range(num_clusters):
            theta = self.rng.normal(loc=center_thetas[i], scale=cluster_spread, size=points_per_cluster)
            height = self.rng.normal(loc=center_heights[i], scale=cluster_spread, size=points_per_cluster)
            theta = np.clip(theta, 0, 4 * np.pi)
            
            x = theta * jnp.cos(theta) + self.rng.normal(0, 0.05, points_per_cluster)
            y = height + self.rng.normal(0, 0.05, points_per_cluster)
            z = theta * jnp.sin(theta) + self.rng.normal(0, 0.05, points_per_cluster)

            all_data.append(np.vstack([x, y, z]).T)
            all_labels.append(np.full(points_per_cluster, i))
            all_latents.append(np.vstack([theta, height]).T)

        data, labels, latents = np.vstack(all_data), np.concatenate(all_labels), np.vstack(all_latents)
        perm = self.rng.permutation(len(data))
        data, labels, latents = data[perm], labels[perm], latents[perm]
        
        self.means, self.stds = data.mean(axis=0), data.std(axis=0)
        self.stds[self.stds < 1e-6] = 1.0 
        
        norm_data = ((data - self.means) / self.stds).astype(np.float32)
        val_len = int(val_split * len(norm_data))
        
        self.train_data, self.train_labels = norm_data[val_len:], labels[val_len:]
        self.true_latents = latents[val_len:].astype(np.float32)

    def get_iterator(self):
        idx = np.arange(len(self.train_data))
        np.random.shuffle(idx)
        for i in range(0, len(self.train_data), self.batch_size):
            batch = idx[i:i + self.batch_size]
            yield jnp.array(self.train_data[batch]), jnp.array(self.train_labels[batch])

    def get_full_data(self):
        return jnp.array(self.train_data), jnp.array(self.train_labels)

    def generate_id_test_manifold(self, num_points=2000):
        """Generates continuous unseen data across the whole valid 2D manifold for ID testing."""
        theta = self.rng.uniform(0.5 * np.pi, 3.5 * np.pi, num_points)
        height = self.rng.uniform(-2.0, 2.0, num_points)
        
        x = theta * jnp.cos(theta)
        y = height
        z = theta * jnp.sin(theta)
        
        test_data_raw = np.vstack([x, y, z]).T
        test_latents = np.vstack([theta, height]).T
        test_data = ((test_data_raw - self.means) / self.stds).astype(np.float32)
        return jnp.array(test_data), test_latents

# ==========================================
# 3. FUNCTIONAL BASIS EVALUATOR (The Decoder)
# ==========================================
def render_target_network(theta, tau=1.0):
    """
    Evaluates a tiny neural network defined by the weights `theta`
    at a canonical coordinate `tau`.
    """
    w1 = theta[0:16].reshape((16, 1))
    b1 = theta[16:32]
    w2 = theta[32:80].reshape((3, 16))
    b2 = theta[80:83]
    
    tau_arr = jnp.array([tau])
    hidden = jax.nn.relu(w1 @ tau_arr + b1)
    out = w2 @ hidden + b2
    return out

# ==========================================
# 4. MODELS (Abstract & Hypernetworks)
# ==========================================
class AbstractAE(eqx.Module):
    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.encoder = eqx.nn.MLP(3, 2, 128, 4, key=k1)
        self.decoder = eqx.nn.MLP(2, 3, 128, 4, key=k2)

    def __call__(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class DirectHyperNet(eqx.Module):
    hyper_encoder: eqx.nn.MLP
    means: jax.Array
    stds: jax.Array

    def __init__(self, key, means, stds):
        self.hyper_encoder = eqx.nn.MLP(3, TARGET_PARAM_COUNT, 128, 3, key=key)
        self.means = jnp.array(means)
        self.stds = jnp.array(stds)

    def __call__(self, x):
        theta = self.hyper_encoder(x)
        x_hat_raw = render_target_network(theta, tau=1.0)
        x_hat = (x_hat_raw - self.means) / self.stds
        return x_hat, theta

class BottleneckHyperNet(eqx.Module):
    latent_encoder: eqx.nn.MLP
    weight_generator: eqx.nn.MLP
    means: jax.Array
    stds: jax.Array

    def __init__(self, key, means, stds):
        k1, k2 = jax.random.split(key)
        self.latent_encoder = eqx.nn.MLP(3, 2, 64, 2, key=k1)
        self.weight_generator = eqx.nn.MLP(2, TARGET_PARAM_COUNT, 64, 2, key=k2)
        self.means = jnp.array(means)
        self.stds = jnp.array(stds)

    def __call__(self, x):
        z = self.latent_encoder(x)
        theta = self.weight_generator(z)
        x_hat_raw = render_target_network(theta, tau=1.0)
        x_hat = (x_hat_raw - self.means) / self.stds
        return x_hat, z

# ==========================================
# 5. TRAINING LOGIC
# ==========================================
def create_trainer(model_instance, lr):
    optimizer = optax.adam(lr)
    
    filter_spec = jax.tree_util.tree_map(eqx.is_array, model_instance)
    if hasattr(model_instance, 'means'):
        filter_spec = eqx.tree_at(lambda m: m.means, filter_spec, False)
    if hasattr(model_instance, 'stds'):
        filter_spec = eqx.tree_at(lambda m: m.stds, filter_spec, False)

    diff_model, static_model = eqx.partition(model_instance, filter_spec)
    opt_state = optimizer.init(diff_model)

    @eqx.filter_value_and_grad
    def loss_fn(diff, static, x):
        model = eqx.combine(diff, static)
        x_hat, _ = jax.vmap(model)(x)
        return jnp.mean((x_hat - x) ** 2)

    @eqx.filter_jit
    def make_step(diff, static, opt_state, x):
        loss, grads = loss_fn(diff, static, x)
        updates, opt_state = optimizer.update(grads, opt_state, diff)
        diff = eqx.apply_updates(diff, updates)
        return diff, opt_state, loss

    return diff_model, static_model, opt_state, make_step

# ==========================================
# 6. MAIN EXECUTION & PRE-TRAINING VISUALIZATION
# ==========================================
#%%
config = {
    "lr": 1e-3,
    "batch_size": 64,
    "epochs": 150,
    "val_every": 50,
    "num_points": 2000,
    "seed": 42
}

key = jax.random.PRNGKey(config["seed"])
k1, k2, k3 = jax.random.split(key, 3)

print("--- Initializing Data ---")
dm = SpiralDataHandler(num_points=config["num_points"], seed=config["seed"])

# ---- PRE-TRAINING 3D MANIFOLD PLOT ----
print("Visualizing 3D Data Manifold (Train vs Test)...")
X_train_raw = (dm.train_data * dm.stds) + dm.means
X_test_norm, _ = dm.generate_id_test_manifold(num_points=1000)
X_test_raw = (np.array(X_test_norm) * dm.stds) + dm.means

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test_raw[:, 0], X_test_raw[:, 1], X_test_raw[:, 2], 
           c='grey', s=5, alpha=0.3, label='ID Test Data (Continuous Manifold)')
scatter = ax.scatter(X_train_raw[:, 0], X_train_raw[:, 1], X_train_raw[:, 2], 
           c=dm.train_labels, cmap='tab10', s=15, alpha=0.9, label='Training Data (Disjoint Clusters)')
ax.set_title("3D Spiral Manifold: Training Clusters vs Unseen Test Data", fontsize=14)
legend1 = ax.legend(loc='upper right')
ax.add_artist(legend1)
plt.show()
# ---------------------------------------

models = {
    "Abstract_AE": AbstractAE(k1),
    "Direct_HyperNet": DirectHyperNet(k2, dm.means, dm.stds),
    "Bottleneck_HyperNet": BottleneckHyperNet(k3, dm.means, dm.stds)
}

history = {m: {'train': []} for m in models.keys()}
run_dir = get_run_path()
print(f"🚀 Training Models. Run ID: {run_dir.name}")

trainers = {}
for name, model in models.items():
    diff, stat, opt, step_fn = create_trainer(model, config["lr"])
    trainers[name] = {"diff": diff, "static": stat, "opt": opt, "step": step_fn}

start_time = time.time()
for epoch in range(config["epochs"]):
    batch_losses = {m: [] for m in models.keys()}
    
    for x, _ in dm.get_iterator():
        for name, t in trainers.items():
            t["diff"], t["opt"], loss = t["step"](t["diff"], t["static"], t["opt"], x)
            batch_losses[name].append(float(loss))
            
    if epoch % config["val_every"] == 0 or epoch == config["epochs"] - 1:
        print(f"Epoch {epoch:04d} | ", end="")
        for name, t in trainers.items():
            avg_loss = np.mean(batch_losses[name])
            history[name]['train'].append(avg_loss)
            print(f"{name}: [L: {avg_loss:.4f}] | ", end="")
        print()

print(f"\n⏱ Training completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

final_models = {n: eqx.combine(t["diff"], t["static"]) for n, t in trainers.items()}
plot_loss_curves(history, run_dir, title="Functional Basis Training Loss")

# ==========================================
# 7. ANALYSIS & PLOTTING THE LATENT OPTIONS
# ==========================================
#%%
import umap
print("\n--- Extracting Functional Representations ---")
X_full, y_full = dm.get_full_data()

_, z_abs = jax.vmap(final_models["Abstract_AE"])(X_full)
z_abs = np.array(z_abs)

_, theta_nD = jax.vmap(final_models["Direct_HyperNet"])(X_full)
theta_nD = np.array(theta_nD)

print(f"Running PCA and t-SNE on the {TARGET_PARAM_COUNT}D Weight Space...")
pca_weights = PCA(n_components=2).fit_transform(theta_nD)
tsne_weights = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(theta_nD)

_, z_explicit = jax.vmap(final_models["Bottleneck_HyperNet"])(X_full)
z_explicit = np.array(z_explicit)

print("Running PCA, t-SNE, and UMAP on the Original 3D Ambient Space...")
pca_ambient = PCA(n_components=2).fit_transform(X_full)
tsne_ambient = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_full)
umap_ambient = umap.UMAP(n_components=2, random_state=42).fit_transform(X_full)

print("Generating the Functional Latent Analysis...")
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('Evaluating Functional Basis Representations vs Ambient Baselines', fontsize=16)

plots = [
    # Top Row: Ambient & Ground Truth
    (dm.true_latents, "True Physical Latents", axes[0, 0]),
    (pca_ambient, "Ambient PCA (3D -> 2D)", axes[0, 1]),
    (tsne_ambient, "Ambient t-SNE (3D -> 2D)", axes[0, 2]),
    (umap_ambient, "Ambient UMAP (3D -> 2D)", axes[0, 3]),
    
    # Bottom Row: Model Latents
    (z_abs, "Baseline: Abstract AE (Entangled / Unstable)", axes[1, 0]),
    (pca_weights, f"Option A: PCA on {TARGET_PARAM_COUNT}D Target Weights", axes[1, 1]),
    (tsne_weights, f"Option A: t-SNE on {TARGET_PARAM_COUNT}D Target Weights", axes[1, 2]),
    (z_explicit, "Option B: Explicit 2D Bottleneck", axes[1, 3]),
]

for data, title, ax in plots:
    ax.scatter(data[:, 0], data[:, 1], c=y_full, cmap='tab10', s=15, alpha=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(run_dir / "plots" / "functional_latent_analysis.png", dpi=150)
plt.show()

# ==========================================
# 8. ID GENERALIZATION TESTING
# ==========================================
#%%
print("\n--- Running ID Generalization Test (Interpolation) ---")
# Get continuous test data that fills the gaps between training clusters
X_test, test_true_latents = dm.generate_id_test_manifold(num_points=1500)

# Pass test data through models
_, z_abs_test = jax.vmap(final_models["Abstract_AE"])(X_test)
_, z_explicit_test = jax.vmap(final_models["Bottleneck_HyperNet"])(X_test)

# To project test data via PCA, we must use the PCA fitted on the training weights
_, theta_nD_test = jax.vmap(final_models["Direct_HyperNet"])(X_test)
pca_model = PCA(n_components=2).fit(theta_nD) # Re-fit to grab the model object
pca_weights_test = pca_model.transform(np.array(theta_nD_test))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('ID Generalization: Interpolating Continuous Unseen Data', fontsize=16)

id_plots = [
    (z_abs, z_abs_test, "Abstract AE Interpolation", axes[0]),
    (pca_weights, pca_weights_test, f"Option A (PCA) Interpolation", axes[1]),
    (z_explicit, z_explicit_test, "Option B (Bottleneck) Interpolation", axes[2]),
]

for train_z, test_z, title, ax in id_plots:
    # Plot Training Clusters lightly in the background
    ax.scatter(np.array(train_z)[:, 0], np.array(train_z)[:, 1], 
               c=y_full, cmap='tab10', s=10, alpha=0.2, label='Train Clusters')
    # Overlay Test Interpolation points
    ax.scatter(np.array(test_z)[:, 0], np.array(test_z)[:, 1], 
               c='black', s=5, alpha=0.6, label='Test Data (Interpolation)')
    
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

axes[2].legend(loc='lower right')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(run_dir / "plots" / "id_generalization_test.png", dpi=150)
plt.show()
print("✅ ID Generalization complete. Observe how Option B smoothly fills the gaps!")