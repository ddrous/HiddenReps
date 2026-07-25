"""
Tiny Implicit Neural Representation (INR) that overfits a single image.
JAX + Equinox + Optax.

Pipeline:
  1. Load the source image, upscale it to 224x224 (RGB).
  2. Build a normalized (x, y) coordinate grid in [-1, 1].
  3. Optionally encode coordinates with Fourier features (USE_FOURIER_FEATURES flag).
  4. Train a small SIREN-style MLP (the INR) to map coords -> RGB, minimizing MSE.
  5. Plot the loss curve, the reconstruction vs. ground truth, and a
     2D loss-landscape slice around the final weights (random directions,
     "filter normalized" the way Li et al. 2018 do it).
"""

import os

os.environ.setdefault(
    "JAX_PLATFORMS", "cpu"
)  # safe default; CPU is plenty for a 224x224 INR

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# %%
# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

IMG_SIZE = 224
USE_FOURIER_FEATURES = False  # <-- flip this to False to feed raw (x, y) coords instead
NUM_FOURIER_FREQS = 1  # only used when USE_FOURIER_FEATURES = True
INR_WIDTH = 12
INR_DEPTH = 6
NUM_STEPS = 5000
LR = 1e-3
SEED = 0
DTYPE = (
    jnp.float32
)  # float32 throughout; bf16 causes needless precision headaches for a demo

# INPUT_IMAGE_PATH = "pusht.webp"
INPUT_IMAGE_PATH = "ogbench.png"
# INPUT_IMAGE_PATH = "tworoom.png"
# INPUT_IMAGE_PATH = "reacher.png"
OUT_DIR = "./"
os.makedirs(OUT_DIR, exist_ok=True)

# %%
# --------------------------------------------------------------------------------------
# Data: load image, upscale to 224x224, build coordinate grid
# --------------------------------------------------------------------------------------


def load_and_upscale(path, size):
    im = Image.open(path).convert("RGB")
    im = im.resize((size, size), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    return arr


def make_coord_grid(size, dtype):
    # Pixel-center coordinates normalized to [-1, 1]
    ys = jnp.linspace(-1.0, 1.0, size, dtype=dtype)
    xs = jnp.linspace(-1.0, 1.0, size, dtype=dtype)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")
    coords = jnp.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)
    return coords


# %%
# --------------------------------------------------------------------------------------
# Tiny INR with (optional) Fourier features
# --------------------------------------------------------------------------------------


def fourier_encode(coords, num_freqs):
    """
    coords: (..., 2) grid coordinates.
    Returns: (..., 2*2*num_freqs) encoded features without flattening spatial dims.
    """
    freqs = (2.0 ** jnp.arange(num_freqs)).astype(coords.dtype)
    pi = jnp.array(jnp.pi, dtype=coords.dtype)

    # Broadcast to shape (..., 2, num_freqs)
    args = coords[..., None] * freqs * pi

    # Concatenate sin/cos and flatten ONLY the feature dimension
    feature_shape = coords.shape[:-1] + (-1,)
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1).reshape(
        feature_shape
    )


def encode_coords(coords, use_fourier, num_freqs):
    """Single switch point: Fourier-encode, or pass raw (x, y) straight through."""
    if use_fourier:
        return fourier_encode(coords, num_freqs)
    return coords  # raw (..., 2) coords, untouched


def compute_in_dim(use_fourier, num_freqs):
    return 2 * 2 * num_freqs if use_fourier else 2


class INR(eqx.Module):
    layers: list
    raw_omega: jnp.ndarray  # shape (depth-1,), unconstrained/trainable
    omega_min: float = eqx.field(static=True)
    omega_max: float = eqx.field(static=True)

    def __init__(
        self, in_dim, out_dim, width, depth, key, omega_min=50.0, omega_max=60.0
    ):
        keys = jax.random.split(key, depth)
        dims = [in_dim] + [width] * (depth - 1) + [out_dim]
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(depth)
        ]
        self.omega_min = omega_min
        self.omega_max = omega_max
        # placeholder; siren_init_learnable_omega overwrites this with a sensible starting point
        self.raw_omega = jnp.zeros((depth - 1,))

    def get_omega(self, i):
        # sigmoid squashes the trainable scalar into (0, 1), then rescale into [omega_min, omega_max]
        # frac = jax.nn.sigmoid(self.raw_omega[i])

        # frac = jnp.clip(self.raw_omega[i], 0.0, 1.0)

        frac = jnp.sin(self.raw_omega[i]) ** 2

        # frac = self.raw_omega[i]

        return self.omega_min + frac * (self.omega_max - self.omega_min)

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                omega = self.get_omega(i)
                x = jnp.sin(omega * x)
        return x


def siren_init(model, key, init_omega=30.0):
    """
    Initialize weights with SIREN's principled uniform scheme (using init_omega
    as the effective frequency for the bound calculation), and initialize the
    learnable raw_omega scalars so every layer STARTS at init_omega exactly
    (it's then free to drift within [omega_min, omega_max] during training).
    """
    layers = model.layers
    keys = jax.random.split(key, len(layers))
    new_layers = []
    for i, (layer, k) in enumerate(zip(layers, keys)):
        fan_in = layer.weight.shape[1]
        bound = 1.0 / fan_in if i == 0 else np.sqrt(6.0 / fan_in) / init_omega
        new_weight = jax.random.uniform(
            k, layer.weight.shape, minval=-bound, maxval=bound
        )
        layer = eqx.tree_at(lambda l: l.weight, layer, new_weight)
        new_layers.append(layer)
    model = eqx.tree_at(lambda m: m.layers, model, new_layers)

    # Solve for the raw (pre-sigmoid) value that maps to init_omega exactly:
    # init_omega = omega_min + sigmoid(raw) * (omega_max - omega_min)
    frac = (init_omega - model.omega_min) / (model.omega_max - model.omega_min)
    frac = float(np.clip(frac, 1e-4, 1 - 1e-4))  # avoid inf at the logit boundaries
    raw_init_value = float(np.log(frac / (1 - frac)))  # inverse sigmoid (logit)
    raw_omega = jnp.full((len(layers) - 1,), raw_init_value)
    model = eqx.tree_at(lambda m: m.raw_omega, model, raw_omega)

    return model


# class INR(eqx.Module):
#     layers: list
#     omega_0: float = 30.0   # frequency multiplier, standard SIREN default

#     def __init__(self, in_dim, out_dim, width, depth, key, omega_min=50.0, omega_max=60.0):
#         keys = jax.random.split(key, depth)
#         dims = [in_dim] + [width] * (depth - 1) + [out_dim]
#         self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(depth)]

#     def __call__(self, x):
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i < len(self.layers) - 1:
#                 x = jnp.sin(self.omega_0 * x)
#         return x


# def siren_init(model, key, omega_0=30.0, init_omega=30.0):
#     """Re-initialize weights with SIREN's principled uniform scheme.
#     Call this once, right after constructing INR(...)."""
#     layers = model.layers
#     keys = jax.random.split(key, len(layers))
#     new_layers = []
#     for i, (layer, k) in enumerate(zip(layers, keys)):
#         fan_in = layer.weight.shape[1]
#         bound = 1.0 / fan_in if i == 0 else np.sqrt(6.0 / fan_in) / omega_0
#         new_weight = jax.random.uniform(k, layer.weight.shape, minval=-bound, maxval=bound)
#         layer = eqx.tree_at(lambda l: l.weight, layer, new_weight)
#         new_layers.append(layer)
#     return eqx.tree_at(lambda m: m.layers, model, new_layers)


# %%
# --------------------------------------------------------------------------------------
# Train / loss utilities
# --------------------------------------------------------------------------------------


def make_forward(use_fourier, num_freqs):
    """Returns a fn (model, coords) -> predicted RGB image, vmapped over H and W."""

    def predict_pixel(model, coord):
        feats = encode_coords(coord, use_fourier, num_freqs)
        return jax.nn.sigmoid(model(feats))  # squash to [0, 1] so it's a valid image

    # vmap over width, then over height
    predict_row = jax.vmap(predict_pixel, in_axes=(None, 0))
    predict_image = jax.vmap(predict_row, in_axes=(None, 0))
    return predict_image


def make_loss_fn(forward_fn):
    def loss_fn(model, coords, target):
        pred = forward_fn(model, coords)
        return jnp.mean((pred - target) ** 2)

    return loss_fn


@eqx.filter_jit
def train_step(model, opt_state, coords, target, optimizer, loss_fn):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, coords, target)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train(model, coords, target, num_steps, lr, use_fourier, num_freqs):
    forward_fn = make_forward(use_fourier, num_freqs)
    loss_fn = make_loss_fn(forward_fn)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    losses = np.zeros(num_steps, dtype=np.float64)
    for step in range(num_steps):
        model, opt_state, loss = train_step(
            model, opt_state, coords, target, optimizer, loss_fn
        )
        losses[step] = float(loss)
        if step % 500 == 0 or step == num_steps - 1:
            print(
                f"step {step:5d}  mse {float(loss):.6f}  psnr {psnr(float(loss)):.2f} dB"
            )

    return model, losses, forward_fn


def psnr(mse, max_val=1.0):
    mse = max(mse, 1e-12)
    return 10.0 * np.log10((max_val**2) / mse)


# %%
# --------------------------------------------------------------------------------------
# Loss landscape visualization (filter-normalized random directions, Li et al. 2018)
# --------------------------------------------------------------------------------------


def flatten_params(model):
    params = eqx.filter(model, eqx.is_array)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    return leaves, treedef


def random_filter_normalized_direction(leaves, key):
    """One random direction per leaf, norm-matched per-leaf to the leaf's own norm
    (a simplified per-tensor version of the 'filter normalization' trick)."""
    keys = jax.random.split(key, len(leaves))
    direction = []
    for leaf, k in zip(leaves, keys):
        d = jax.random.normal(k, leaf.shape, dtype=leaf.dtype)
        d_norm = jnp.linalg.norm(d) + 1e-12
        leaf_norm = jnp.linalg.norm(leaf)
        d = d / d_norm * leaf_norm  # match this tensor's norm
        direction.append(d)
    return direction


def loss_landscape(model, coords, target, loss_fn, key, span=1.0, n=25):
    params, static = eqx.partition(model, eqx.is_array)
    leaves, treedef = jax.tree_util.tree_flatten(params)

    k1, k2 = jax.random.split(key)
    dir1 = random_filter_normalized_direction(leaves, k1)
    dir2 = random_filter_normalized_direction(leaves, k2)

    alphas = np.linspace(-span, span, n)
    betas = np.linspace(-span, span, n)
    grid = np.zeros((n, n), dtype=np.float64)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            new_leaves = [
                leaf + a * d1 + b * d2 for leaf, d1, d2 in zip(leaves, dir1, dir2)
            ]
            new_params = jax.tree_util.tree_unflatten(treedef, new_leaves)
            perturbed_model = eqx.combine(new_params, static)
            grid[j, i] = float(loss_fn(perturbed_model, coords, target))

    return alphas, betas, grid


# %%
# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

key = jax.random.PRNGKey(SEED)
model_key, land_key = jax.random.split(key)

print("Loading + upscaling image...")
target_np = load_and_upscale(INPUT_IMAGE_PATH, IMG_SIZE)
target = jnp.asarray(target_np, dtype=DTYPE)
coords = make_coord_grid(IMG_SIZE, DTYPE)

in_dim = compute_in_dim(USE_FOURIER_FEATURES, NUM_FOURIER_FREQS)
print(f"Fourier features: {USE_FOURIER_FEATURES}  ->  input dim: {in_dim}")

# model = INR(in_dim=in_dim, out_dim=3, width=INR_WIDTH, depth=INR_DEPTH, key=model_key)
# model = siren_init(model, model_key, omega_0=30.0)

model = INR(
    in_dim=in_dim,
    out_dim=3,
    width=INR_WIDTH,
    depth=INR_DEPTH,
    key=model_key,
    omega_min=10.0,
    omega_max=60.0,
)
model = siren_init(model, model_key, init_omega=30.0)

n_params = sum(
    p.size for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
)
print(f"INR parameter count: {n_params}")

# %%
print("Training...")
model, losses, forward_fn = train(
    model,
    coords,
    target,
    NUM_STEPS,
    LR,
    use_fourier=USE_FOURIER_FEATURES,
    num_freqs=NUM_FOURIER_FREQS,
)

# Final reconstruction
recon = np.array(forward_fn(model, coords))
recon = np.clip(recon, 0.0, 1.0)

final_mse = float(np.mean((recon - target_np) ** 2))
print(f"Final MSE: {final_mse:.6f}  PSNR: {psnr(final_mse):.2f} dB")

# %%
# ---------------- Plot 1: target vs reconstruction ----------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(target_np)
axes[0].set_title("Target (224x224)")
axes[0].axis("off")

axes[1].imshow(recon)
axes[1].set_title(f"INR reconstruction\nPSNR {psnr(final_mse):.2f} dB")
axes[1].axis("off")

diff = np.abs(recon - target_np)
im = axes[2].imshow(diff, cmap="inferno", vmin=0, vmax=diff.max())
axes[2].set_title("Abs. error")
axes[2].axis("off")
fig.colorbar(im, ax=axes[2], fraction=0.046)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "reconstruction.png"), dpi=150)
plt.show()

# %%
# ---------------- Plot 2: training loss curve ----------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(losses)
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("MSE (log scale)")
ax.set_title("INR training loss")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150)
plt.show()

# %%
# ---------------- Plot 3: 2D loss landscape around the trained weights ----------------
print("Computing loss landscape (this takes a little while)...")
loss_fn = make_loss_fn(forward_fn)
alphas, betas, grid = loss_landscape(
    model, coords, target, loss_fn, land_key, span=1.0, n=25
)

fig = plt.figure(figsize=(11, 4.5))

ax1 = fig.add_subplot(1, 2, 1)
cs = ax1.contourf(alphas, betas, np.log10(grid + 1e-8), levels=30, cmap="viridis")
ax1.plot(0, 0, marker="*", color="red", markersize=14, label="trained solution")
ax1.set_xlabel("direction 1")
ax1.set_ylabel("direction 2")
ax1.set_title("Loss landscape (log10 MSE)")
ax1.legend(loc="upper right")
fig.colorbar(cs, ax=ax1, fraction=0.046)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
A, B = np.meshgrid(alphas, betas)
ax2.plot_surface(
    A, B, np.log10(grid + 1e-8), cmap="viridis", linewidth=0, antialiased=True
)
ax2.set_xlabel("direction 1")
ax2.set_ylabel("direction 2")
ax2.set_zlabel("log10 MSE")
ax2.set_title("Loss landscape (surface)")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_landscape.png"), dpi=150)
plt.show()

print("Done. Saved:")
print(" -", os.path.join(OUT_DIR, "reconstruction.png"))
print(" -", os.path.join(OUT_DIR, "loss_curve.png"))
print(" -", os.path.join(OUT_DIR, "loss_landscape.png"))
