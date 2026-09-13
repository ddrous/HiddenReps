#%% 0) Imports, configuration, and experiment constants
"""Sequential Bayes Transport for the two-moons benchmark.

This variant has no replay buffer. It maintains one particle cloud for the observed datum x_o and
mixes it with fresh uniform interpolation clouds during training. At every optimizer step:

    1. theta targets are resampled from the current x_o cloud and passed through the simulator;
    2. each row uses either that current cloud or a fresh uniform interpolation cloud as its prior;
    3. the model is updated with the simulator-supervised energy-score objective; and
    4. the updated model transports the current cloud at x_o, producing the prior cloud for the
       next optimizer step.

The initial cloud is sampled exactly from U([-1,1]^2).  All cloud transitions are detached: the
optimizer differentiates through only the current step, never through the history of sequential
clouds.  The script follows the cell-oriented style of bayes_transport_two_moons.py and intentionally
has no main() function.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import seaborn as sns

sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "DejaVu Sans",
    "axes.titlepad": 8.0,
    "axes.labelpad": 6.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.18,
})

Array = jax.Array


@dataclass
class Config:
    # Reproducibility / outputs
    seed: int = 2028
    output_dir: str = "plots/bayes_transport_two_moons_seq"

    # Exact two-moons benchmark from Greenberg et al. (2019), Appendix A.5.1
    prior_low: float = -1.0
    prior_high: float = 1.0
    radial_mean: float = 0.1
    radial_std: float = 0.01
    crescent_x_offset: float = 0.25
    observed_x1: float = 0.0
    observed_x2: float = 0.0

    # A single fixed-size sequential cloud is shared by every inference problem in a batch.
    batch_size: int = 128
    particles: int = 16 * 4

    # Particle transport architecture, matching bayes_transport_two_moons.py.
    hidden_dim: int = 64 * 2
    heads: int = 4
    mlp_ratio: int = 4
    posterior_depth: int = 4
    posterior_conditioning: str = "adaln"  # {"cross_attention", "adaln"}
    max_normalized_displacement: float = 6.0
    attention_dropout_rate: float = 0.0

    # x=(x1,x2) is represented as two coordinate-labelled observation tokens.
    likelihood_hidden_dim: int = 64
    likelihood_heads: int = 4
    likelihood_mlp_ratio: int = 4
    likelihood_depth: int = 3
    normalize_observations: bool = True
    observation_scale: float = 1.0

    # Optimisation
    training_steps: int = 1_000
    learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip_norm: float = 5000.0
    log_every: int = 250

    # Per-row training prior mixture.
    x_observed_prior_probability: float = 0.2
    prior_interpolation_tau_min: float = 0.0
    prior_interpolation_tau_max: float = 1.5

    # Diagnostics only; none of these values enter the training objective.
    posterior_grid_size: int = 420
    exact_reference_samples: int = 10_000
    sliced_wasserstein_projections: int = 128
    prior_predictive_plot_samples: int = 30_000
    snapshot_simulation_budgets: tuple[int, ...] = (1000, 5000, 10_000)


CFG = Config()
OUT = Path(CFG.output_dir)
OUT.mkdir(parents=True, exist_ok=True)

if CFG.posterior_conditioning not in {"cross_attention", "adaln"}:
    raise ValueError("posterior_conditioning must be 'cross_attention' or 'adaln'.")
if CFG.hidden_dim % CFG.heads != 0:
    raise ValueError("hidden_dim must be divisible by heads.")
if CFG.likelihood_hidden_dim % CFG.likelihood_heads != 0:
    raise ValueError("likelihood_hidden_dim must be divisible by likelihood_heads.")
if CFG.prior_low >= CFG.prior_high:
    raise ValueError("prior_low must be smaller than prior_high.")
if CFG.radial_std <= 0.0 or CFG.observation_scale <= 0.0:
    raise ValueError("radial_std and observation_scale must be positive.")
if CFG.particles < 2:
    raise ValueError("particles must be at least 2 for the energy score.")
if CFG.batch_size < 1 or CFG.training_steps < 1:
    raise ValueError("batch_size and training_steps must be positive.")
if not 0.0 <= CFG.x_observed_prior_probability <= 1.0:
    raise ValueError("x_observed_prior_probability must lie in [0,1].")
if not 0.0 <= CFG.prior_interpolation_tau_min <= CFG.prior_interpolation_tau_max:
    raise ValueError("prior_interpolation_tau_min/max must satisfy 0 <= min <= max.")

PRIOR_CENTER = 0.5 * (CFG.prior_low + CFG.prior_high)
PRIOR_STD = (CFG.prior_high - CFG.prior_low) / math.sqrt(12.0)
X_OBS = np.asarray([CFG.observed_x1, CFG.observed_x2], dtype=np.float32)

print("JAX devices:", jax.devices())
print("Output directory:", OUT.resolve())
print(json.dumps(asdict(CFG), indent=2))
print("Observed datum:", X_OBS)
print("Sequential particle count:", CFG.particles)


#%% 1) Exact prior and two-moons simulator

def sample_exact_prior_np(
    rng: np.random.Generator,
    n: int,
    cfg: Config = CFG,
) -> np.ndarray:
    """Sample theta ~ U([prior_low, prior_high]^2)."""
    return rng.uniform(cfg.prior_low, cfg.prior_high, size=(int(n), 2)).astype(np.float32)


def sample_from_particle_cloud_np(
    rng: np.random.Generator,
    cloud: np.ndarray,
    n: int,
) -> np.ndarray:
    """Resample simulator parameters from the current empirical sequential prior."""
    cloud = np.asarray(cloud, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] != 2 or len(cloud) < 2:
        raise ValueError("cloud must have shape [M,2] with M >= 2.")
    ids = rng.integers(0, len(cloud), size=int(n))
    return cloud[ids].copy()


def sample_interpolated_training_prior_np(
    rng: np.random.Generator,
    theta_target: np.ndarray,
    n_particles: int,
    cfg: Config = CFG,
) -> tuple[np.ndarray, float]:
    """Sample a fresh uniform cloud interpolated toward the simulator target."""
    uniform_cloud = sample_exact_prior_np(rng, n_particles, cfg)
    tau = float(rng.uniform(cfg.prior_interpolation_tau_min, cfg.prior_interpolation_tau_max))
    target = np.asarray(theta_target, dtype=np.float32).reshape(1, 2)
    cloud = (1.0 - tau) * uniform_cloud + tau * target
    return cloud.astype(np.float32), tau


def make_training_prior_batch_np(
    rng: np.random.Generator,
    mode_rng: np.random.Generator,
    theta_target: np.ndarray,
    observed_cloud: np.ndarray,
    cfg: Config = CFG,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Choose an observed cloud or an independent uniform interpolation per row."""
    theta_target = np.asarray(theta_target, dtype=np.float32)
    observed_cloud = np.asarray(observed_cloud, dtype=np.float32)
    if theta_target.ndim != 2 or theta_target.shape[1] != 2:
        raise ValueError("theta_target must have shape [B,2].")
    if observed_cloud.shape != (cfg.particles, 2):
        raise ValueError(f"observed_cloud must have shape [{cfg.particles},2].")

    prior = np.empty((len(theta_target), cfg.particles, 2), dtype=np.float32)
    observed_mask = mode_rng.random(len(theta_target)) < cfg.x_observed_prior_probability
    interpolation_mask = ~observed_mask
    prior[observed_mask] = observed_cloud

    interpolation_tau = np.full(len(theta_target), np.nan, dtype=np.float32)
    for index in np.flatnonzero(interpolation_mask):
        prior[index], interpolation_tau[index] = sample_interpolated_training_prior_np(
            rng, theta_target[index], cfg.particles, cfg
        )

    return prior, {
        "observed_cloud_used": observed_mask.astype(np.float32),
        "interpolation_used": interpolation_mask.astype(np.float32),
        "interpolation_tau": interpolation_tau,
    }


def simulate_two_moons_batch_np(
    rng: np.random.Generator,
    theta: np.ndarray,
    cfg: Config = CFG,
) -> np.ndarray:
    """Vectorized two-moons simulator. theta is [B,2] and the result is [B,2]."""
    theta = np.asarray(theta, dtype=np.float32)
    if theta.ndim != 2 or theta.shape[1] != 2:
        raise ValueError("theta must have shape [B,2].")

    b = theta.shape[0]
    a = rng.uniform(-0.5 * math.pi, 0.5 * math.pi, size=b).astype(np.float32)
    r = rng.normal(cfg.radial_mean, cfg.radial_std, size=b).astype(np.float32)
    p1 = r * np.cos(a) + np.float32(cfg.crescent_x_offset)
    p2 = r * np.sin(a)

    theta_sum = theta[:, 0] + theta[:, 1]
    shift1 = -np.abs(theta_sum) / np.float32(math.sqrt(2.0))
    shift2 = (-theta[:, 0] + theta[:, 1]) / np.float32(math.sqrt(2.0))
    return np.column_stack([p1 + shift1, p2 + shift2]).astype(np.float32)


def simulate_two_moons_np(
    rng: np.random.Generator,
    theta: np.ndarray,
    cfg: Config = CFG,
) -> np.ndarray:
    """Single-theta convenience wrapper."""
    return simulate_two_moons_batch_np(rng, np.asarray(theta).reshape(1, 2), cfg)[0]


# Show x_o against the exact-prior predictive distribution before training.
_plot_rng = np.random.default_rng(CFG.seed + 101)
_plot_theta = sample_exact_prior_np(_plot_rng, CFG.prior_predictive_plot_samples)
_plot_x = simulate_two_moons_batch_np(_plot_rng, _plot_theta)

fig, ax = plt.subplots(figsize=(7.5, 7.0))
h = ax.hist2d(_plot_x[:, 0], _plot_x[:, 1], bins=150, cmap="viridis", density=True)
ax.scatter(
    [X_OBS[0]], [X_OBS[1]], marker="*", s=260, c="white",
    edgecolors="black", linewidths=1.2, label=r"observed $x_o=(0,0)$",
)
ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", title="Prior-predictive simulator output and observed datum")
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="best")
fig.colorbar(h[3], ax=ax, label="prior-predictive density")
fig.tight_layout()
fig.savefig(OUT / "00_observed_x_prior_predictive.png", dpi=180, bbox_inches="tight")
plt.show()


#%% 2) JAX + Equinox observation and particle Transformers

def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]


class ObservationBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout_p: float, *, key: Array):
        k_attn, k_ff1, k_ff2 = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=dim,
            key_size=dim,
            value_size=dim,
            output_size=dim,
            dropout_p=dropout_p,
            key=k_attn,
        )
        self.ff_in = eqx.nn.Linear(dim, mlp_dim, key=k_ff1)
        self.ff_out = eqx.nn.Linear(mlp_dim, dim, key=k_ff2)

    def __call__(self, tokens: Array, *, key: Array | None = None, inference: bool = False) -> Array:
        h = _layernorm_tokens(self.norm1, tokens)
        tokens = tokens + self.attention(h, h, h, key=key, inference=inference)
        h = _layernorm_tokens(self.norm2, tokens)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        return tokens + _linear_tokens(self.ff_out, h)


class TwoMoonsObservationEmbedder(eqx.Module):
    """Encode x=(x1,x2) as two labelled tokens [value, one-hot coordinate id]."""

    input_projection: eqx.nn.Linear
    blocks: tuple[ObservationBlock, ...]
    final_norm: eqx.nn.LayerNorm
    normalize: bool = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(self, cfg: Config, *, key: Array):
        keys = jax.random.split(key, cfg.likelihood_depth + 1)
        self.input_projection = eqx.nn.Linear(3, cfg.likelihood_hidden_dim, key=keys[0])
        self.blocks = tuple(
            ObservationBlock(
                cfg.likelihood_hidden_dim,
                cfg.likelihood_heads,
                cfg.likelihood_mlp_ratio * cfg.likelihood_hidden_dim,
                cfg.attention_dropout_rate,
                key=keys[i + 1],
            )
            for i in range(cfg.likelihood_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(cfg.likelihood_hidden_dim)
        self.normalize = bool(cfg.normalize_observations)
        self.scale = float(cfg.observation_scale)

    def __call__(self, x: Array, *, key: Array | None = None, inference: bool = False) -> Array:
        x = jnp.reshape(x, (2,))
        if self.normalize:
            x = x / self.scale
        coord_id = jnp.eye(2, dtype=x.dtype)
        token_features = jnp.concatenate([x[:, None], coord_id], axis=-1)
        tokens = _linear_tokens(self.input_projection, token_features)
        block_keys = None if key is None else jax.random.split(key, len(self.blocks))
        for i, block in enumerate(self.blocks):
            block_key = None if block_keys is None else block_keys[i]
            tokens = block(tokens, key=block_key, inference=inference)
        return _layernorm_tokens(self.final_norm, tokens)


class AdaLNParticleBlock(eqx.Module):
    norm_attn: eqx.nn.LayerNorm
    norm_ff: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear
    modulation: eqx.nn.Linear

    def __init__(
        self,
        hidden: int,
        conditioning_dim: int,
        heads: int,
        mlp_dim: int,
        dropout_p: float,
        *,
        key: Array,
    ):
        k_attn, k_ff1, k_ff2, k_mod = jax.random.split(key, 4)
        self.norm_attn = eqx.nn.LayerNorm(hidden)
        self.norm_ff = eqx.nn.LayerNorm(hidden)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden,
            key_size=hidden,
            value_size=hidden,
            output_size=hidden,
            dropout_p=dropout_p,
            key=k_attn,
        )
        self.ff_in = eqx.nn.Linear(hidden, mlp_dim, key=k_ff1)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden, key=k_ff2)
        modulation = eqx.nn.Linear(conditioning_dim, 6 * hidden, key=k_mod)
        modulation = eqx.tree_at(lambda layer: layer.weight, modulation, jnp.zeros_like(modulation.weight))
        modulation = eqx.tree_at(lambda layer: layer.bias, modulation, jnp.zeros_like(modulation.bias))
        self.modulation = modulation

    def __call__(
        self,
        particles: Array,
        conditioning: Array,
        *,
        key: Array | None = None,
        inference: bool = False,
    ) -> Array:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = jnp.split(
            self.modulation(jax.nn.silu(conditioning)), 6, axis=-1
        )
        h = _modulate(_layernorm_tokens(self.norm_attn, particles), shift_a, scale_a)
        particles = particles + gate_a[None, :] * self.attention(
            h, h, h, key=key, inference=inference
        )
        h = _modulate(_layernorm_tokens(self.norm_ff, particles), shift_f, scale_f)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        return particles + gate_f[None, :] * _linear_tokens(self.ff_out, h)


class CrossAttentionParticleBlock(eqx.Module):
    norm_self: eqx.nn.LayerNorm
    norm_cross: eqx.nn.LayerNorm
    memory_norm: eqx.nn.LayerNorm
    norm_ff: eqx.nn.LayerNorm
    self_attention: eqx.nn.MultiheadAttention
    cross_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(
        self,
        hidden: int,
        memory_dim: int,
        heads: int,
        mlp_dim: int,
        dropout_p: float,
        *,
        key: Array,
    ):
        k_self, k_cross, k_ff1, k_ff2 = jax.random.split(key, 4)
        self.norm_self = eqx.nn.LayerNorm(hidden)
        self.norm_cross = eqx.nn.LayerNorm(hidden)
        self.memory_norm = eqx.nn.LayerNorm(memory_dim)
        self.norm_ff = eqx.nn.LayerNorm(hidden)
        self.self_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=hidden, key_size=hidden, value_size=hidden,
            output_size=hidden, dropout_p=dropout_p, key=k_self,
        )
        self.cross_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=hidden, key_size=memory_dim, value_size=memory_dim,
            output_size=hidden, dropout_p=dropout_p, key=k_cross,
        )
        self.ff_in = eqx.nn.Linear(hidden, mlp_dim, key=k_ff1)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden, key=k_ff2)

    def __call__(
        self,
        particles: Array,
        observation_memory: Array,
        *,
        key: Array | None = None,
        inference: bool = False,
    ) -> Array:
        if key is None:
            self_key = cross_key = None
        else:
            self_key, cross_key = jax.random.split(key)

        h = _layernorm_tokens(self.norm_self, particles)
        particles = particles + self.self_attention(h, h, h, key=self_key, inference=inference)
        q = _layernorm_tokens(self.norm_cross, particles)
        memory = _layernorm_tokens(self.memory_norm, observation_memory)
        particles = particles + self.cross_attention(
            q, memory, memory, key=cross_key, inference=inference
        )
        h = _layernorm_tokens(self.norm_ff, particles)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        return particles + _linear_tokens(self.ff_out, h)


class ConditionalParticleTransport(eqx.Module):
    """Identity-initialized particle transport conditioned on one 2-D observation."""

    observation_embedder: TwoMoonsObservationEmbedder
    particle_in: eqx.nn.Linear
    blocks: tuple[Any, ...]
    final_norm: eqx.nn.LayerNorm
    displacement_head: eqx.nn.Linear
    conditioning_type: str = eqx.field(static=True)
    max_displacement: float = eqx.field(static=True)
    prior_center: float = eqx.field(static=True)
    prior_std: float = eqx.field(static=True)

    def __init__(self, cfg: Config, *, key: Array):
        keys = jax.random.split(key, cfg.posterior_depth + 4)
        self.observation_embedder = TwoMoonsObservationEmbedder(cfg, key=keys[0])
        self.particle_in = eqx.nn.Linear(2, cfg.hidden_dim, key=keys[1])
        block_cls = AdaLNParticleBlock if cfg.posterior_conditioning == "adaln" else CrossAttentionParticleBlock
        self.blocks = tuple(
            block_cls(
                cfg.hidden_dim,
                cfg.likelihood_hidden_dim,
                cfg.heads,
                cfg.mlp_ratio * cfg.hidden_dim,
                cfg.attention_dropout_rate,
                key=keys[2 + i],
            )
            for i in range(cfg.posterior_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(cfg.hidden_dim)
        head = eqx.nn.Linear(cfg.hidden_dim, 2, key=keys[-1])
        head = eqx.tree_at(lambda layer: layer.weight, head, jnp.zeros_like(head.weight))
        head = eqx.tree_at(lambda layer: layer.bias, head, jnp.zeros_like(head.bias))
        self.displacement_head = head
        self.conditioning_type = str(cfg.posterior_conditioning)
        self.max_displacement = float(cfg.max_normalized_displacement)
        self.prior_center = float(PRIOR_CENTER)
        self.prior_std = float(PRIOR_STD)

    def __call__(
        self,
        prior_theta: Array,
        x: Array,
        *,
        key: Array | None = None,
        inference: bool = False,
    ) -> Array:
        if key is None:
            obs_key = transport_key = None
        else:
            obs_key, transport_key = jax.random.split(key)

        memory = self.observation_embedder(x, key=obs_key, inference=inference)
        z0 = (prior_theta - self.prior_center) / self.prior_std
        particles = _linear_tokens(self.particle_in, z0)
        block_keys = None if transport_key is None else jax.random.split(transport_key, len(self.blocks))

        if self.conditioning_type == "adaln":
            conditioning = jnp.mean(memory, axis=0)
            for i, block in enumerate(self.blocks):
                block_key = None if block_keys is None else block_keys[i]
                particles = block(particles, conditioning, key=block_key, inference=inference)
        else:
            for i, block in enumerate(self.blocks):
                block_key = None if block_keys is None else block_keys[i]
                particles = block(particles, memory, key=block_key, inference=inference)

        particles = _layernorm_tokens(self.final_norm, particles)
        delta = self.max_displacement * jnp.tanh(_linear_tokens(self.displacement_head, particles))
        return self.prior_center + self.prior_std * (z0 + delta)


#%% 3) Proper multivariate energy score and optimizer step

_ENERGY_NORM_EPS = 1e-12


def _stable_l2_norm(x: Array, axis: int = -1) -> Array:
    eps = jnp.asarray(_ENERGY_NORM_EPS, dtype=x.dtype)
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis) + eps)


def energy_score_terms(posterior: Array, target_theta: Array) -> tuple[Array, Array, Array]:
    """Empirical 2-D energy score: E||Y-theta*|| - 0.5 E||Y-Y'||."""
    attraction = jnp.mean(_stable_l2_norm(posterior - target_theta[None, :], axis=-1))
    pairwise = posterior[:, None, :] - posterior[None, :, :]
    repulsion = jnp.mean(_stable_l2_norm(pairwise, axis=-1))
    return attraction - 0.5 * repulsion, attraction, repulsion


def batch_metrics(posterior: Array, target_theta: Array) -> dict[str, Array]:
    score, attraction, repulsion = jax.vmap(energy_score_terms)(posterior, target_theta)
    means = jnp.mean(posterior, axis=1)
    centered = posterior - means[:, None, :]
    outside = jnp.any(
        (posterior < CFG.prior_low) | (posterior > CFG.prior_high), axis=-1
    )
    return {
        "loss": jnp.mean(score),
        "energy_score": jnp.mean(score),
        "attraction": jnp.mean(attraction),
        "repulsion": jnp.mean(repulsion),
        "mean_error": jnp.mean(_stable_l2_norm(means - target_theta, axis=-1)),
        "covariance_trace": jnp.mean(jnp.mean(jnp.sum(centered**2, axis=-1), axis=1)),
        "outside_prior_fraction": jnp.mean(outside.astype(jnp.float32)),
    }


def transport_objective(
    model: ConditionalParticleTransport,
    prior_theta: Array,
    x_batch: Array,
    target_theta: Array,
    dropout_key: Array,
):
    row_keys = jax.random.split(dropout_key, prior_theta.shape[0])
    posterior = jax.vmap(
        lambda prior, x, key: model(prior, x, key=key, inference=False)
    )(prior_theta, x_batch, row_keys)
    metrics = batch_metrics(posterior, target_theta)
    return metrics["loss"], (metrics, posterior)


_loss_and_grad = eqx.filter_value_and_grad(transport_objective, has_aux=True)


def make_train_step(optimizer: optax.GradientTransformation):
    @eqx.filter_jit
    def step(model, opt_state, prior_theta, x_batch, target_theta, dropout_key):
        (loss, (metrics, posterior)), grads = _loss_and_grad(
            model, prior_theta, x_batch, target_theta, dropout_key
        )
        params = eqx.filter(model, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return model, opt_state, loss, metrics, posterior, grad_norm

    return step


def evaluate_transport(
    model: ConditionalParticleTransport,
    prior_particles: np.ndarray,
    x: np.ndarray = X_OBS,
) -> np.ndarray:
    """Deterministically transport one detached particle cloud at one observation."""
    prior_particles = np.asarray(prior_particles, dtype=np.float32)
    if prior_particles.ndim != 2 or prior_particles.shape[1] != 2:
        raise ValueError("prior_particles must have shape [M,2].")
    result = model(
        jnp.asarray(prior_particles),
        jnp.asarray(x, dtype=jnp.float32),
        key=None,
        inference=True,
    )
    result_np = np.asarray(jax.device_get(result), dtype=np.float32)
    if result_np.shape != prior_particles.shape or not np.all(np.isfinite(result_np)):
        raise FloatingPointError("Sequential posterior cloud has an invalid shape or non-finite values.")
    return result_np


def save_model(path: Path, model: ConditionalParticleTransport, cfg: Config = CFG) -> None:
    eqx.tree_serialise_leaves(Path(path), model)
    with Path(path).with_suffix(".json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_model(path: Path, cfg: Config = CFG) -> ConditionalParticleTransport:
    template = ConditionalParticleTransport(cfg, key=jax.random.key(cfg.seed))
    return eqx.tree_deserialise_leaves(Path(path), template)


model = ConditionalParticleTransport(CFG, key=jax.random.key(CFG.seed))
optimizer = optax.chain(
    optax.clip_by_global_norm(CFG.grad_clip_norm),
    optax.adamw(CFG.learning_rate, weight_decay=CFG.weight_decay),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
train_step = make_train_step(optimizer)

print("Model initialized.")
print("Conditioning:", CFG.posterior_conditioning)
print("Training batch size:", CFG.batch_size)


#%% 4) Exact-posterior diagnostic helpers (never used by training)

def normal_pdf_sd_np(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float64) - float(mean)) / float(std)
    return np.exp(-0.5 * z**2) / (float(std) * math.sqrt(2.0 * math.pi))


def crescent_p_density_np(p: np.ndarray, cfg: Config = CFG) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    u = p[..., 0] - float(cfg.crescent_x_offset)
    v = p[..., 1]
    rho = np.sqrt(u**2 + v**2)
    signed_r = np.where(u >= 0.0, rho, -rho)
    density = normal_pdf_sd_np(signed_r, cfg.radial_mean, cfg.radial_std) / (
        math.pi * np.maximum(rho, 1e-12)
    )
    return np.where(rho > 1e-12, density, 0.0)


def two_moons_likelihood_density_np(
    x: np.ndarray,
    theta: np.ndarray,
    cfg: Config = CFG,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(2)
    theta = np.asarray(theta, dtype=np.float64)
    theta_sum = theta[..., 0] + theta[..., 1]
    shift1 = -np.abs(theta_sum) / math.sqrt(2.0)
    shift2 = (-theta[..., 0] + theta[..., 1]) / math.sqrt(2.0)
    p_required = np.stack([x[0] - shift1, x[1] - shift2], axis=-1)
    return crescent_p_density_np(p_required, cfg)


def exact_posterior_grid(x: np.ndarray = X_OBS, cfg: Config = CFG):
    t1 = np.linspace(cfg.prior_low, cfg.prior_high, cfg.posterior_grid_size, dtype=np.float64)
    t2 = np.linspace(cfg.prior_low, cfg.prior_high, cfg.posterior_grid_size, dtype=np.float64)
    g1, g2 = np.meshgrid(t1, t2, indexing="xy")
    density = two_moons_likelihood_density_np(x, np.stack([g1, g2], axis=-1), cfg)
    normalizer = np.sum(density) * (t1[1] - t1[0]) * (t2[1] - t2[0])
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise FloatingPointError("Ground-truth posterior grid failed to normalize.")
    return t1, t2, density / normalizer


def sample_from_grid_posterior(
    rng: np.random.Generator,
    theta1: np.ndarray,
    theta2: np.ndarray,
    density: np.ndarray,
    n: int,
) -> np.ndarray:
    probabilities = np.asarray(density, dtype=np.float64).reshape(-1)
    probabilities /= probabilities.sum()
    ids = rng.choice(len(probabilities), size=int(n), replace=True, p=probabilities)
    i2, i1 = np.unravel_index(ids, density.shape)
    samples = np.column_stack([theta1[i1], theta2[i2]])
    samples[:, 0] += rng.uniform(-0.5, 0.5, len(samples)) * (theta1[1] - theta1[0])
    samples[:, 1] += rng.uniform(-0.5, 0.5, len(samples)) * (theta2[1] - theta2[0])
    return samples.astype(np.float32)


def posterior_summary(samples: np.ndarray, cfg: Config = CFG) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=np.float64)
    return {
        "mean": np.mean(samples, axis=0).tolist(),
        "cov": np.cov(samples.T).tolist(),
        "outside_prior_fraction": float(np.mean(np.any(
            (samples < cfg.prior_low) | (samples > cfg.prior_high), axis=1
        ))),
    }


def sliced_wasserstein(
    a: np.ndarray,
    b: np.ndarray,
    n_projections: int,
) -> float:
    rng = np.random.default_rng(456)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b), 5000)
    a = a[rng.choice(len(a), n, replace=False)]
    b = b[rng.choice(len(b), n, replace=False)]
    directions = rng.normal(size=(int(n_projections), 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = [
        np.mean(np.abs(np.sort(a @ direction) - np.sort(b @ direction)))
        for direction in directions
    ]
    return float(np.mean(distances))


def rolling_mean(values: np.ndarray, window: int = 100) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return values.copy()
    window = max(1, min(int(window), len(values)))
    smooth = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), smooth])


THETA1_GRID, THETA2_GRID, EXACT_DENSITY = exact_posterior_grid(X_OBS, CFG)
_exact_rng = np.random.default_rng(CFG.seed + 500)
EXACT_SAMPLES = sample_from_grid_posterior(
    _exact_rng, THETA1_GRID, THETA2_GRID, EXACT_DENSITY, CFG.exact_reference_samples
)


#%% 5) TRAIN SEQUENTIALLY — one shared prior cloud, updated at x_o after every step
# The exact likelihood and ground-truth posterior above are diagnostic-only and are never referenced
# in this cell.

train_rng = np.random.default_rng(CFG.seed + 10_001)
mode_rng = np.random.default_rng(CFG.seed + 20_003)
dropout_key = jax.random.key(CFG.seed + 30_007)

# This is the only initial input cloud.  It becomes the first proposal and is retained for plots.
INITIAL_PRIOR_CLOUD = sample_exact_prior_np(train_rng, CFG.particles)
sequential_prior_cloud = INITIAL_PRIOR_CLOUD.copy()

history = {name: [] for name in (
    "step",
    "simulations_seen",
    "energy_score",
    "attraction",
    "repulsion",
    "mean_error",
    "covariance_trace",
    "outside_prior_fraction",
    "grad_norm",
    "proposal_mean_1",
    "proposal_mean_2",
    "proposal_std_1",
    "proposal_std_2",
    "proposal_update_rms",
    "observed_cloud_fraction",
    "interpolation_fraction",
    "mean_interpolation_tau",
)}
snapshot_clouds: dict[int, tuple[int, np.ndarray]] = {}

for step in range(1, CFG.training_steps + 1):
    # The current empirical x_o posterior is the proposal from which this step's simulator
    # parameters are drawn.
    theta_target = sample_from_particle_cloud_np(train_rng, sequential_prior_cloud, CFG.batch_size)
    x_batch = simulate_two_moons_batch_np(train_rng, theta_target)

    # Each row either starts from the current x_o cloud or from its own fresh interpolation.
    prior_batch, prior_info = make_training_prior_batch_np(
        train_rng,
        mode_rng,
        theta_target,
        sequential_prior_cloud,
        CFG,
    )

    dropout_key, step_key = jax.random.split(dropout_key)
    model, opt_state, loss, metrics, _, grad_norm = train_step(
        model,
        opt_state,
        jnp.asarray(prior_batch),
        jnp.asarray(x_batch),
        jnp.asarray(theta_target),
        step_key,
    )

    # This detached posterior at x_o is the one and only prior cloud for the next step.
    previous_cloud = sequential_prior_cloud
    sequential_prior_cloud = evaluate_transport(model, previous_cloud, X_OBS)
    update_rms = float(np.sqrt(np.mean((sequential_prior_cloud - previous_cloud) ** 2)))

    host = jax.device_get(metrics)
    proposal_std = np.std(sequential_prior_cloud, axis=0)
    interpolation_tau = prior_info["interpolation_tau"]
    finite_tau = interpolation_tau[np.isfinite(interpolation_tau)]
    scalar_values = {
        "step": float(step),
        "simulations_seen": float(step * CFG.batch_size),
        "energy_score": float(host["energy_score"]),
        "attraction": float(host["attraction"]),
        "repulsion": float(host["repulsion"]),
        "mean_error": float(host["mean_error"]),
        "covariance_trace": float(host["covariance_trace"]),
        "outside_prior_fraction": float(host["outside_prior_fraction"]),
        "grad_norm": float(jax.device_get(grad_norm)),
        "proposal_mean_1": float(np.mean(sequential_prior_cloud[:, 0])),
        "proposal_mean_2": float(np.mean(sequential_prior_cloud[:, 1])),
        "proposal_std_1": float(proposal_std[0]),
        "proposal_std_2": float(proposal_std[1]),
        "proposal_update_rms": update_rms,
        "observed_cloud_fraction": float(np.mean(prior_info["observed_cloud_used"])),
        "interpolation_fraction": float(np.mean(prior_info["interpolation_used"])),
        "mean_interpolation_tau": float(np.mean(finite_tau)) if len(finite_tau) else np.nan,
    }
    for name, value in scalar_values.items():
        history[name].append(value)

    previous_simulations = (step - 1) * CFG.batch_size
    current_simulations = step * CFG.batch_size
    for budget in CFG.snapshot_simulation_budgets:
        if budget not in snapshot_clouds and previous_simulations < budget <= current_simulations:
            snapshot_clouds[budget] = (current_simulations, sequential_prior_cloud.copy())

    if step == 1 or step % CFG.log_every == 0 or step == CFG.training_steps:
        print(
            f"step {step:6d}/{CFG.training_steps} | "
            f"sims {current_simulations:9,d} | "
            f"ES {scalar_values['energy_score']:.5f} | "
            f"mean-err {scalar_values['mean_error']:.4f} | "
            f"cloud std ({scalar_values['proposal_std_1']:.4f}, "
            f"{scalar_values['proposal_std_2']:.4f}) | "
            f"prior observed/interpolated "
            f"{scalar_values['observed_cloud_fraction']:.2f}/"
            f"{scalar_values['interpolation_fraction']:.2f} | "
            f"update-rms {update_rms:.3e} | "
            f"grad {scalar_values['grad_norm']:.3e}"
        )

BT_POSTERIOR = sequential_prior_cloud.copy()
save_model(OUT / "bayes_transport_two_moons_seq.eqx", model, CFG)

with (OUT / "training_history.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(history.keys()))
    writer.writeheader()
    for i in range(len(history["step"])):
        writer.writerow({name: history[name][i] for name in history})

np.save(OUT / "initial_prior_cloud.npy", INITIAL_PRIOR_CLOUD)
np.save(OUT / "sequential_posterior_cloud.npy", BT_POSTERIOR)
np.save(OUT / "ground_truth_posterior_samples.npy", EXACT_SAMPLES)
np.save(OUT / "observed_x.npy", X_OBS)


#%% 6) Training and final-posterior diagnostics

steps = np.asarray(history["step"])
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

loss_values = np.asarray(history["energy_score"])
axes[0, 0].plot(steps, loss_values, alpha=0.3, label="energy score")
axes[0, 0].plot(steps, rolling_mean(loss_values), linewidth=2, label="100-step mean")
axes[0, 0].set(title="Training objective", xlabel="optimizer step", ylabel="energy score")
axes[0, 0].legend()

axes[0, 1].plot(steps, history["proposal_std_1"], label=r"std($\theta_1$)")
axes[0, 1].plot(steps, history["proposal_std_2"], label=r"std($\theta_2$)")
axes[0, 1].set(title=r"Sequential $x_o$ cloud scale", xlabel="optimizer step", ylabel="standard deviation")
axes[0, 1].legend()

axes[1, 0].plot(steps, history["proposal_mean_1"], label=r"mean($\theta_1$)")
axes[1, 0].plot(steps, history["proposal_mean_2"], label=r"mean($\theta_2$)")
axes[1, 0].set(title=r"Sequential $x_o$ cloud location", xlabel="optimizer step", ylabel="mean")
axes[1, 0].legend()

axes[1, 1].semilogy(steps, np.maximum(history["proposal_update_rms"], 1e-12))
axes[1, 1].set(title="Cloud change after each update", xlabel="optimizer step", ylabel="RMS displacement")

fig.tight_layout()
fig.savefig(OUT / "10_training_diagnostics.png", dpi=180, bbox_inches="tight")
plt.show()

levels = np.sort(np.asarray([
    np.quantile(EXACT_DENSITY[EXACT_DENSITY > 0], 0.50),
    np.quantile(EXACT_DENSITY[EXACT_DENSITY > 0], 0.80),
    np.quantile(EXACT_DENSITY[EXACT_DENSITY > 0], 0.95),
]))

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
axes[0].scatter(INITIAL_PRIOR_CLOUD[:, 0], INITIAL_PRIOR_CLOUD[:, 1], s=14, alpha=0.6)
axes[0].set_title("Initial exact-prior cloud")

axes[1].imshow(
    EXACT_DENSITY,
    origin="lower",
    extent=[CFG.prior_low, CFG.prior_high, CFG.prior_low, CFG.prior_high],
    cmap="viridis",
    aspect="equal",
)
axes[1].set_title(r"Ground truth $p(\theta\mid x_o)$")

axes[2].contour(THETA1_GRID, THETA2_GRID, EXACT_DENSITY, levels=levels, cmap="viridis")
axes[2].scatter(BT_POSTERIOR[:, 0], BT_POSTERIOR[:, 1], s=14, alpha=0.6)
axes[2].set_title("Final sequential posterior cloud")

for ax in axes:
    ax.set_xlim(CFG.prior_low, CFG.prior_high)
    ax.set_ylim(CFG.prior_low, CFG.prior_high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")

fig.tight_layout()
fig.savefig(OUT / "20_sequential_posterior.png", dpi=200, bbox_inches="tight")
plt.show()

print("\nGround-truth posterior summary:")
print(json.dumps(posterior_summary(EXACT_SAMPLES), indent=2))
print("\nSequential Bayes-Transport posterior summary:")
print(json.dumps(posterior_summary(BT_POSTERIOR), indent=2))
print(
    "Sliced Wasserstein distance:",
    sliced_wasserstein(EXACT_SAMPLES, BT_POSTERIOR, CFG.sliced_wasserstein_projections),
)


#%% 7) Optional snapshot panel

if snapshot_clouds:
    ordered_snapshots = sorted(snapshot_clouds.items())
    fig, axes = plt.subplots(1, len(ordered_snapshots) + 1, figsize=(5 * (len(ordered_snapshots) + 1), 4.8))
    axes = np.atleast_1d(axes)
    axes[0].imshow(
        EXACT_DENSITY,
        origin="lower",
        extent=[CFG.prior_low, CFG.prior_high, CFG.prior_low, CFG.prior_high],
        cmap="viridis",
        aspect="equal",
    )
    axes[0].set_title("Ground truth")

    for ax, (budget, (actual_sims, cloud)) in zip(axes[1:], ordered_snapshots):
        ax.contour(THETA1_GRID, THETA2_GRID, EXACT_DENSITY, levels=levels, cmap="viridis")
        ax.scatter(cloud[:, 0], cloud[:, 1], s=14, alpha=0.6)
        ax.set_title(f"budget {budget:,}\nactual {actual_sims:,}")

    for ax in axes:
        ax.set_xlim(CFG.prior_low, CFG.prior_high)
        ax.set_ylim(CFG.prior_low, CFG.prior_high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")

    fig.tight_layout()
    fig.savefig(OUT / "30_sequential_snapshots.png", dpi=200, bbox_inches="tight")
    plt.show()
