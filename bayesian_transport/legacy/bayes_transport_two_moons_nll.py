#%% 0) Imports, configuration, and experiment constants
"""Notebook-style Bayes Transport on the Greenberg et al. (2019) two-moons benchmark.

Run this file one #%% cell at a time in VS Code / Spyder / Jupyter-compatible editors.
There is intentionally no main() function.

Training is simulator-supervised and likelihood-free:
    theta* ~ U([-1,1]^2)
    x ~ simulator(theta*)
    q_psi(theta | x) = one of two mutually-exclusive normalized density models:
        "gmm" -> small MLP-conditioned full-covariance Gaussian mixture
        "mlp" -> small conditional MLP RealNVP flow (no GMM)
    q_psi is trained with the proper target NLL -log q_psi(theta* | x)
    prior particles ~ one of three mutually-exclusive TRAINING sources
    posterior particles = T_phi(prior particles, x)
    transport loss = multivariate empirical energy score against theta*

The observed datum used in the paper is x_o=(0,0). Evaluation ALWAYS starts Bayes Transport
from the exact paper prior U([-1,1]^2); training-only interpolation and replay never enter evaluation.

The implementation keeps the same Bayes-Transport design used in the previous experiment:
    * Equinox/JAX particle Transformer;
    * selectable AdaLN or cross-attention conditioning;
    * identity-initialized displacement head;
    * a selectable normalized conditional density: MLP flow or MLP-conditioned GMM;
    * proper energy-score training for the particle transport;
    * optional training-only prior interpolation and historical-posterior replay;
    * dense training diagnostics and final posterior diagnostics.

The two-moons observation x is a single 2-D vector, not a variable-length sequence. To retain the
same conditioning architecture without inventing fake repeated observations, the observation encoder
represents x1 and x2 as two dimension-labelled tokens. Both tokens are available to each other; there
is no causal mask because x1 and x2 are coordinates of one observation, not a time sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
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
from scipy.stats import gaussian_kde

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "DejaVu Sans",
    "axes.titlepad": 8.0,
    "axes.labelpad": 6.0,
})

# Uncomment only when debugging numerical issues. It substantially slows JAX execution.
# jax.config.update("jax_debug_nans", True)

Array = jax.Array


@dataclass
class Config:
    # Reproducibility / outputs
    seed: int = 2028
    output_dir: str = "plots/bayes_transport_two_moons_v3"

    # Exact two-moons benchmark from Greenberg et al. (2019), Appendix A.5.1
    prior_low: float = -1.0
    prior_high: float = 1.0
    radial_mean: float = 0.1
    radial_std: float = 0.01
    crescent_x_offset: float = 0.25
    observed_x1: float = 0.0
    observed_x2: float = 0.0

    # Here batch_size is a conventional simulator-training batch:
    # 128 independent theta* values, each generating ONE 2-D two-moons observation x.
    batch_size: int = 128

    # Particle transport -- intentionally kept very close to the previous script.
    num_particles: int = 16 * 2
    eval_particles: int = 1024 // 2
    hidden_dim: int = 64 * 4
    heads: int = 4
    mlp_ratio: int = 4
    posterior_depth: int = 5
    posterior_conditioning: str = "cross_attention"  # {"cross_attention", "adaln"}
    max_normalized_displacement: float = 6.0
    attention_dropout_rate: float = 0.0

    # Observation encoder. x=(x1,x2) is represented as two labelled tokens.
    likelihood_hidden_dim: int = 64
    likelihood_heads: int = 4
    likelihood_mlp_ratio: int = 4
    likelihood_depth: int = 3
    normalize_observations: bool = True
    observation_scale: float = 1.0

    # Bayes Transport optimisation -- preserved from the supplied/latest setup.
    training_steps: int = 25_000
    learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip_norm: float = 5.0
    log_every: int = 1250

    # Normalized posterior-density model trained by NLL.
    # Exactly ONE mode is active:
    #   "gmm": x -> parameters of a full-covariance Gaussian mixture (current v2 behavior).
    #   "mlp": a conditional RealNVP density whose coupling functions are small MLPs; no GMM.
    #
    # Both are genuine normalized densities, so -log q_psi(theta*|x) is a valid log score.
    # The existing hidden size/depth are reused in BOTH modes. GMM-only hyperparameters are
    # ignored in "mlp" mode; nothing else in the experiment changes.
    posterior_density_mode: str = "mlp"  # {"gmm", "mlp"}
    posterior_density_components: int = 50
    posterior_density_hidden_dim: int = 64
    posterior_density_depth: int = 3
    posterior_density_min_scale: float = 1e-3
    posterior_density_max_abs_correlation: float = 0.95

    # Three mutually-exclusive TRAINING prior sources:
    #   interpolation with probability p_interp;
    #   historical posterior replay with probability p_buffer;
    #   exact evaluation prior U([-1,1]^2) with residual probability 1-p_interp-p_buffer.
    #
    # The cloud BEFORE interpolation can be selected independently:
    #   "uniform"  -> exact U([-1,1]^2) geometry;
    #   "gaussian" -> moment-matched N(0, 1/3 I), i.e. same mean/covariance as U([-1,1]^2).
    # Evaluation is ALWAYS exact uniform, regardless of this setting.
    interpolation_base_cloud: str = "uniform"  # {"uniform", "gaussian"}
    prior_interpolation_probability: float = 0.25
    prior_interpolation_tau_min: float = 0.0
    prior_interpolation_tau_max: float = 2.0
    truth_anchor_probability: float = 1.0
    historical_output_prior_probability: float = 0.25
    historical_output_buffer_capacity: int = 2048

    # Exact posterior / diagnostic grids
    posterior_grid_size: int = 420
    exact_reference_samples: int = 10_000
    kde_grid_size: int = 220
    sliced_wasserstein_projections: int = 128

    # Figure-1-style simulator budgets from the paper. With minibatches of 128, the snapshot is
    # taken at the first optimizer step whose cumulative simulator count reaches/exceeds each target.
    # Thus 1000 becomes 1024 calls, 5000 becomes 5120, etc.; the plot reports the actual count.
    figure1_simulation_budgets: tuple[int, ...] = (1000, 5000, 10_000)

    # Prior-predictive plot at the very start. Diagnostic only; not used for training.
    prior_predictive_plot_samples: int = 30_000


CFG = Config()
OUT = Path(CFG.output_dir)
OUT.mkdir(parents=True, exist_ok=True)

if CFG.posterior_conditioning not in {"cross_attention", "adaln"}:
    raise ValueError("posterior_conditioning must be 'cross_attention' or 'adaln'.")
if CFG.hidden_dim % CFG.heads != 0:
    raise ValueError("hidden_dim must be divisible by heads.")
if CFG.likelihood_hidden_dim % CFG.likelihood_heads != 0:
    raise ValueError("likelihood_hidden_dim must be divisible by likelihood_heads.")
if CFG.interpolation_base_cloud not in {"uniform", "gaussian"}:
    raise ValueError("interpolation_base_cloud must be 'uniform' or 'gaussian'.")
if not 0.0 <= CFG.prior_interpolation_probability <= 1.0:
    raise ValueError("prior_interpolation_probability must lie in [0,1].")
if not 0.0 <= CFG.historical_output_prior_probability <= 1.0:
    raise ValueError("historical_output_prior_probability must lie in [0,1].")
if CFG.prior_interpolation_probability + CFG.historical_output_prior_probability > 1.0 + 1e-12:
    raise ValueError(
        "prior_interpolation_probability + historical_output_prior_probability must be <= 1. "
        "The residual probability is reserved for the exact evaluation prior."
    )
if not 0.0 <= CFG.truth_anchor_probability <= 1.0:
    raise ValueError("truth_anchor_probability must lie in [0,1].")
if not 0.0 <= CFG.prior_interpolation_tau_min <= CFG.prior_interpolation_tau_max:
    raise ValueError("prior_interpolation_tau_min/max must satisfy 0 <= min <= max.")
if CFG.prior_low >= CFG.prior_high:
    raise ValueError("prior_low must be smaller than prior_high.")
if CFG.radial_std <= 0.0 or CFG.observation_scale <= 0.0:
    raise ValueError("radial_std and observation_scale must be positive.")
if CFG.posterior_density_mode not in {"gmm", "mlp"}:
    raise ValueError("posterior_density_mode must be 'gmm' or 'mlp'.")
if CFG.posterior_density_components < 1:
    raise ValueError("posterior_density_components must be >= 1.")
if CFG.posterior_density_hidden_dim < 1 or CFG.posterior_density_depth < 1:
    raise ValueError("posterior_density_hidden_dim/depth must be >= 1.")
if CFG.posterior_density_min_scale <= 0.0:
    raise ValueError("posterior_density_min_scale must be positive.")
if not 0.0 < CFG.posterior_density_max_abs_correlation < 1.0:
    raise ValueError("posterior_density_max_abs_correlation must lie in (0,1).")

TRAIN_EXACT_PRIOR_PROBABILITY = (
    1.0
    - CFG.prior_interpolation_probability
    - CFG.historical_output_prior_probability
)

PRIOR_CENTER = 0.5 * (CFG.prior_low + CFG.prior_high)
PRIOR_STD = (CFG.prior_high - CFG.prior_low) / math.sqrt(12.0)
X_OBS = np.asarray([CFG.observed_x1, CFG.observed_x2], dtype=np.float32)

plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.18,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

print("JAX devices:", jax.devices())
print("Output directory:", OUT.resolve())
print(json.dumps(asdict(CFG), indent=2))
print(
    "Training prior-source probabilities: "
    f"interpolate={CFG.prior_interpolation_probability:.3f}, "
    f"buffer={CFG.historical_output_prior_probability:.3f}, "
    f"exact-test-uniform={TRAIN_EXACT_PRIOR_PROBABILITY:.3f}"
)
print(f"Interpolation base cloud: {CFG.interpolation_base_cloud}")
print("Evaluation prior: exact Uniform([-1,1]^2)")
print("Observed datum:", X_OBS)
print("Posterior density mode:", CFG.posterior_density_mode)


#%% 1) Exact two-moons simulator + FIRST plot: prior-predictive x and observed x_o

def sample_exact_prior_np(rng: np.random.Generator, n: int) -> np.ndarray:
    """Exact paper prior: theta ~ U([-1,1]^2)."""
    return rng.uniform(
        CFG.prior_low,
        CFG.prior_high,
        size=(int(n), 2),
    ).astype(np.float32)


def simulate_two_moons_batch_np(
    rng: np.random.Generator,
    theta: np.ndarray,
) -> np.ndarray:
    """Vectorized two-moons simulator. theta has shape [B,2], returns x with shape [B,2]."""
    theta = np.asarray(theta, dtype=np.float32)
    if theta.ndim != 2 or theta.shape[1] != 2:
        raise ValueError("theta must have shape [B,2].")

    b = theta.shape[0]
    a = rng.uniform(-0.5 * math.pi, 0.5 * math.pi, size=b).astype(np.float32)
    r = rng.normal(CFG.radial_mean, CFG.radial_std, size=b).astype(np.float32)

    p1 = r * np.cos(a) + np.float32(CFG.crescent_x_offset)
    p2 = r * np.sin(a)

    theta_sum = theta[:, 0] + theta[:, 1]
    shift1 = -np.abs(theta_sum) / np.float32(math.sqrt(2.0))
    shift2 = (-theta[:, 0] + theta[:, 1]) / np.float32(math.sqrt(2.0))

    x1 = p1 + shift1
    x2 = p2 + shift2
    return np.column_stack([x1, x2]).astype(np.float32)


def simulate_two_moons_np(rng: np.random.Generator, theta: np.ndarray) -> np.ndarray:
    """Single-theta convenience wrapper."""
    theta = np.asarray(theta, dtype=np.float32).reshape(1, 2)
    return simulate_two_moons_batch_np(rng, theta)[0]


# Plot the observed data FIRST. Because x_o is a single 2-D point, show it against the
# prior-predictive simulator distribution rather than pretending there are repeated observations.
_plot_rng = np.random.default_rng(CFG.seed + 101)
_plot_theta = sample_exact_prior_np(_plot_rng, CFG.prior_predictive_plot_samples)
_plot_x = simulate_two_moons_batch_np(_plot_rng, _plot_theta)

fig, ax = plt.subplots(figsize=(7.5, 7.0))
h = ax.hist2d(
    _plot_x[:, 0],
    _plot_x[:, 1],
    bins=150,
    cmap="viridis",
    density=True,
)
ax.scatter(
    [X_OBS[0]], [X_OBS[1]],
    marker="*", s=260, c="white", edgecolors="black", linewidths=1.2,
    label=r"observed $x_o=(0,0)$",
)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Two-moons prior-predictive simulator output and observed datum")
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="best")
fig.colorbar(h[3], ax=ax, label="prior-predictive density")
fig.tight_layout()
fig.savefig(OUT / "00_observed_x_prior_predictive.png", dpi=180, bbox_inches="tight")
plt.show()


#%% 2) Training-only prior-source mixture and historical posterior buffer

def sample_interpolation_base_cloud_np(
    rng: np.random.Generator,
    n: int,
    cfg: Config = CFG,
) -> np.ndarray:
    """Cloud used BEFORE C_tau interpolation. Evaluation never calls this helper."""
    if cfg.interpolation_base_cloud == "uniform":
        return sample_exact_prior_np(rng, n)

    # Moment-matched Gaussian: same mean and marginal variance as U([prior_low,prior_high]).
    return rng.normal(
        loc=PRIOR_CENTER,
        scale=PRIOR_STD,
        size=(int(n), 2),
    ).astype(np.float32)


def sample_interpolated_training_prior_np(
    rng: np.random.Generator,
    theta_target: np.ndarray,
    cfg: Config = CFG,
) -> tuple[np.ndarray, float]:
    """Training-only C_tau=(1-tau)Z+tau*anchor with one shared tau per particle cloud."""
    z = sample_interpolation_base_cloud_np(rng, cfg.num_particles, cfg)

    if rng.random() < cfg.truth_anchor_probability:
        anchor = np.asarray(theta_target, dtype=np.float32).reshape(2)
    else:
        anchor = sample_exact_prior_np(rng, 1)[0]

    tau = float(rng.uniform(cfg.prior_interpolation_tau_min, cfg.prior_interpolation_tau_max))
    cloud = (1.0 - tau) * z + tau * anchor[None, :]
    return cloud.astype(np.float32), tau


class HistoricalPosteriorBuffer:
    """Training-only nearest-posterior replay keyed by the current simulated 2-D observation x.

    Each slot stores only:
        x from an earlier simulator call;
        the detached posterior cloud achieved for that x.

    On replay, ONLY the current prior cloud is replaced. The current x and current theta* target
    remain untouched, matching the intent of the original Bayes-Transport replay mechanism.
    """

    def __init__(self, capacity: int, num_particles: int):
        self.capacity = int(capacity)
        self.num_particles = int(num_particles)
        self.x = np.empty((self.capacity, 2), dtype=np.float32)
        self.clouds = np.empty((self.capacity, self.num_particles, 2), dtype=np.float32)
        self.size = 0
        self.next_index = 0

    def __len__(self) -> int:
        return int(self.size)

    @property
    def active_x(self) -> np.ndarray:
        return self.x[: self.size]

    @property
    def active_clouds(self) -> np.ndarray:
        return self.clouds[: self.size]

    def add_batch(self, x: np.ndarray, clouds: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        clouds = np.asarray(clouds, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("x must have shape [B,2].")
        if clouds.shape != (len(x), self.num_particles, 2):
            raise ValueError(
                f"clouds must have shape {(len(x), self.num_particles, 2)}, got {clouds.shape}."
            )
        for xi, cloud in zip(x, clouds):
            self.x[self.next_index] = xi
            self.clouds[self.next_index] = cloud
            self.next_index = (self.next_index + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def nearest_batch(self, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            raise ValueError("HistoricalPosteriorBuffer is empty.")
        x_query = np.asarray(x_query, dtype=np.float32).reshape(-1, 2)
        # Fixed observation scale avoids unstable standardization when the buffer is still tiny.
        delta = (x_query[:, None, :] - self.active_x[None, :, :]) / float(CFG.observation_scale)
        d2 = np.sum(delta**2, axis=-1)
        ids = np.argmin(d2, axis=1)
        distances = np.sqrt(d2[np.arange(len(x_query)), ids])
        return self.active_clouds[ids].copy(), distances.astype(np.float32)


def make_training_prior_batch_np(
    rng: np.random.Generator,
    mode_rng: np.random.Generator,
    theta_target: np.ndarray,
    x_batch: np.ndarray,
    replay_buffer: HistoricalPosteriorBuffer,
    cfg: Config = CFG,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Choose exactly one prior source independently for each training inference problem."""
    theta_target = np.asarray(theta_target, dtype=np.float32)
    x_batch = np.asarray(x_batch, dtype=np.float32)
    b = theta_target.shape[0]

    prior = np.empty((b, cfg.num_particles, 2), dtype=np.float32)
    u = mode_rng.random(b)
    p_interp = cfg.prior_interpolation_probability
    p_buffer = cfg.historical_output_prior_probability

    interp_mask = u < p_interp
    requested_buffer_mask = (u >= p_interp) & (u < p_interp + p_buffer)
    buffer_mask = requested_buffer_mask & (len(replay_buffer) > 0)
    exact_mask = ~(interp_mask | buffer_mask)

    tau = np.zeros(b, dtype=np.float32)
    replay_distance = np.full(b, np.nan, dtype=np.float32)

    for i in np.flatnonzero(interp_mask):
        prior[i], tau[i] = sample_interpolated_training_prior_np(rng, theta_target[i], cfg)

    buffer_ids = np.flatnonzero(buffer_mask)
    if len(buffer_ids):
        clouds, dist = replay_buffer.nearest_batch(x_batch[buffer_ids])
        prior[buffer_ids] = clouds
        replay_distance[buffer_ids] = dist

    exact_ids = np.flatnonzero(exact_mask)
    if len(exact_ids):
        prior[exact_ids] = sample_exact_prior_np(rng, len(exact_ids) * cfg.num_particles).reshape(
            len(exact_ids), cfg.num_particles, 2
        )

    info = {
        "interpolation_used": interp_mask.astype(np.float32),
        "buffer_used": buffer_mask.astype(np.float32),
        "exact_prior_used": exact_mask.astype(np.float32),
        "interpolation_tau": tau,
        "replay_distance": replay_distance,
    }
    return prior, info


#%% 3) JAX + Equinox model: 2-D observation encoder + posterior particle Transformer

def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]


class ObservationBlock(eqx.Module):
    """Self-attention block over the two labelled observation-coordinate tokens."""

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
        token_features = jnp.concatenate([x[:, None], coord_id], axis=-1)  # [2,3]
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
        modulation = eqx.tree_at(lambda l: l.weight, modulation, jnp.zeros_like(modulation.weight))
        modulation = eqx.tree_at(lambda l: l.bias, modulation, jnp.zeros_like(modulation.bias))
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
            num_heads=heads,
            query_size=hidden,
            key_size=hidden,
            value_size=hidden,
            output_size=hidden,
            dropout_p=dropout_p,
            key=k_self,
        )
        self.cross_attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden,
            key_size=memory_dim,
            value_size=memory_dim,
            output_size=hidden,
            dropout_p=dropout_p,
            key=k_cross,
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
            q,
            memory,
            memory,
            key=cross_key,
            inference=inference,
        )

        h = _layernorm_tokens(self.norm_ff, particles)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        return particles + _linear_tokens(self.ff_out, h)


class ConditionalParticleTransport(eqx.Module):
    """Identity-initialized 2-D prior -> posterior particle transport conditioned on x in R^2."""

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
        # Exact identity transport at initialization.
        head = eqx.tree_at(lambda l: l.weight, head, jnp.zeros_like(head.weight))
        head = eqx.tree_at(lambda l: l.bias, head, jnp.zeros_like(head.bias))
        self.displacement_head = head

        self.conditioning_type = str(cfg.posterior_conditioning)
        self.max_displacement = float(cfg.max_normalized_displacement)
        self.prior_center = float(PRIOR_CENTER)
        self.prior_std = float(PRIOR_STD)

    def _standardize(self, theta: Array) -> Array:
        return (theta - self.prior_center) / self.prior_std

    def _unstandardize(self, z: Array) -> Array:
        return self.prior_center + self.prior_std * z

    def __call__(
        self,
        prior_theta: Array,
        x: Array,
        *,
        key: Array | None = None,
        inference: bool = False,
    ) -> Array:
        if key is None:
            obs_key = None
            transport_key = None
        else:
            obs_key, transport_key = jax.random.split(key)

        memory = self.observation_embedder(x, key=obs_key, inference=inference)  # [2,C]
        z0 = self._standardize(prior_theta)
        particles = _linear_tokens(self.particle_in, z0)
        block_keys = None if transport_key is None else jax.random.split(transport_key, len(self.blocks))

        if self.conditioning_type == "adaln":
            # Symmetric summary of both observation-coordinate tokens.
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
        return self._unstandardize(z0 + delta)


#%% 4) Version 3: proper NLL density objective + proper energy-score transport objective

class PosteriorDensityGMM(eqx.Module):
    """Conditional MLP -> full-covariance Gaussian mixture q_psi(theta | x).

    This is the original v2 density model. The MLP maps x to mixture weights,
    component means, component standard deviations and correlations.
    """

    hidden_layers: tuple[eqx.nn.Linear, ...]
    output_layer: eqx.nn.Linear

    n_components: int = eqx.field(static=True)
    min_scale: float = eqx.field(static=True)
    max_abs_correlation: float = eqx.field(static=True)
    observation_scale: float = eqx.field(static=True)
    prior_center: float = eqx.field(static=True)
    prior_half_width: float = eqx.field(static=True)

    def __init__(self, cfg: Config, *, key: Array):
        depth = int(cfg.posterior_density_depth)
        keys = jax.random.split(key, depth + 1)
        hidden = int(cfg.posterior_density_hidden_dim)

        layers = []
        in_dim = 2
        for i in range(depth):
            layers.append(eqx.nn.Linear(in_dim, hidden, key=keys[i]))
            in_dim = hidden
        self.hidden_layers = tuple(layers)

        # Per component: 1 mixture logit + 2 means + 2 scales + 1 correlation = 6 values.
        self.output_layer = eqx.nn.Linear(
            hidden,
            6 * int(cfg.posterior_density_components),
            key=keys[-1],
        )
        self.n_components = int(cfg.posterior_density_components)
        self.min_scale = float(cfg.posterior_density_min_scale)
        self.max_abs_correlation = float(cfg.posterior_density_max_abs_correlation)
        self.observation_scale = float(cfg.observation_scale)
        self.prior_center = float(PRIOR_CENTER)
        self.prior_half_width = 0.5 * float(cfg.prior_high - cfg.prior_low)

    def _raw_parameters(self, x: Array) -> Array:
        h = jnp.reshape(x, (2,)) / self.observation_scale
        for layer in self.hidden_layers:
            h = jax.nn.gelu(layer(h))
        return self.output_layer(h)

    def parameters(self, x: Array) -> tuple[Array, Array, Array, Array]:
        raw = self._raw_parameters(x).reshape(self.n_components, 6)
        logits = raw[:, 0]

        # Keep component centres inside the known prior support while retaining smooth gradients.
        means = self.prior_center + self.prior_half_width * jnp.tanh(raw[:, 1:3])

        # Positive marginal scales and bounded correlations give valid covariance matrices.
        scales = self.min_scale + jax.nn.softplus(raw[:, 3:5])
        rho = self.max_abs_correlation * jnp.tanh(raw[:, 5])
        return logits, means, scales, rho

    def log_prob(self, x: Array, theta: Array) -> Array:
        """Normalized log q_psi(theta | x). theta may have shape [2] or [...,2]."""
        logits, means, scales, rho = self.parameters(x)
        theta = jnp.asarray(theta)

        dx = theta[..., None, 0] - means[:, 0]
        dy = theta[..., None, 1] - means[:, 1]
        sx = scales[:, 0]
        sy = scales[:, 1]
        one_minus_rho2 = jnp.maximum(1.0 - rho**2, 1e-6)

        zx = dx / sx
        zy = dy / sy
        mahal = (zx**2 - 2.0 * rho * zx * zy + zy**2) / one_minus_rho2
        log_det_half = jnp.log(sx) + jnp.log(sy) + 0.5 * jnp.log(one_minus_rho2)
        component_log_prob = -jnp.log(2.0 * jnp.pi) - log_det_half - 0.5 * mahal
        log_weights = jax.nn.log_softmax(logits)
        return jax.scipy.special.logsumexp(component_log_prob + log_weights, axis=-1)


class AffineCouplingMLP(eqx.Module):
    """Small MLP used by one conditional RealNVP affine coupling layer."""

    hidden_layers: tuple[eqx.nn.Linear, ...]
    output_layer: eqx.nn.Linear
    observation_scale: float = eqx.field(static=True)
    max_log_scale: float = eqx.field(static=True)

    def __init__(self, cfg: Config, *, key: Array):
        # Reuse the EXISTING density hidden size/depth; no optimizer/model-size hyperparameters changed.
        depth = int(cfg.posterior_density_depth)
        hidden = int(cfg.posterior_density_hidden_dim)
        keys = jax.random.split(key, depth + 1)

        layers = []
        # Input = normalized x1, x2 plus the untouched theta coordinate.
        in_dim = 3
        for i in range(depth):
            layers.append(eqx.nn.Linear(in_dim, hidden, key=keys[i]))
            in_dim = hidden
        self.hidden_layers = tuple(layers)
        self.output_layer = eqx.nn.Linear(hidden, 2, key=keys[-1])
        self.observation_scale = float(cfg.observation_scale)

        # Fixed numerical bound, derived as an implementation safeguard rather than a tuned hyperparameter.
        self.max_log_scale = 2.0

    def __call__(self, x: Array, conditioner: Array) -> tuple[Array, Array]:
        x = jnp.reshape(x, (2,)) / self.observation_scale
        inp = jnp.concatenate([x, jnp.reshape(conditioner, (1,))], axis=0)
        h = inp
        for layer in self.hidden_layers:
            h = jax.nn.gelu(layer(h))
        raw_shift, raw_log_scale = self.output_layer(h)
        # Bounding log-scale prevents numerical explosions while retaining an exact invertible density.
        log_scale = self.max_log_scale * jnp.tanh(raw_log_scale)
        return raw_shift, log_scale


class PosteriorDensityMLP(eqx.Module):
    """Pure neural normalized density: conditional 2-D RealNVP with MLP coupling functions.

    This is the "mlp" mode. There is NO Gaussian mixture. A standard Gaussian base is transformed
    by alternating conditional affine couplings. Because the transformation is invertible and its
    Jacobian determinant is known exactly, log_prob is a genuine normalized density.
    """

    couplings: tuple[AffineCouplingMLP, ...]
    prior_center: float = eqx.field(static=True)
    prior_std: float = eqx.field(static=True)

    def __init__(self, cfg: Config, *, key: Array):
        # Use twice the existing posterior_density_depth as the number of alternating coupling layers.
        # This keeps mode capacity tied to the supplied hyperparameters rather than introducing a new knob.
        n_couplings = max(2, 2 * int(cfg.posterior_density_depth))
        keys = jax.random.split(key, n_couplings)
        self.couplings = tuple(AffineCouplingMLP(cfg, key=k) for k in keys)
        self.prior_center = float(PRIOR_CENTER)
        self.prior_std = float(PRIOR_STD)

    def _standardize(self, theta: Array) -> Array:
        return (theta - self.prior_center) / self.prior_std

    def log_prob(self, x: Array, theta: Array) -> Array:
        """Exact normalized log q_psi(theta|x). theta may have shape [2] or [...,2]."""
        theta = jnp.asarray(theta)

        def one_log_prob(th: Array) -> Array:
            # Invert the flow from theta -> z. Coupling parity alternates transformed coordinate.
            y = self._standardize(jnp.reshape(th, (2,)))
            log_abs_det_inverse = jnp.asarray(
                -2.0 * math.log(self.prior_std), dtype=y.dtype
            )

            for layer_index in range(len(self.couplings) - 1, -1, -1):
                coupling = self.couplings[layer_index]
                transformed = layer_index % 2
                conditioner_idx = 1 - transformed

                shift, log_scale = coupling(x, y[conditioner_idx])
                transformed_value = (y[transformed] - shift) * jnp.exp(-log_scale)
                y = y.at[transformed].set(transformed_value)
                log_abs_det_inverse = log_abs_det_inverse - log_scale

            base_log_prob = -0.5 * jnp.sum(y**2) - jnp.log(2.0 * jnp.pi)
            return base_log_prob + log_abs_det_inverse

        flat = theta.reshape((-1, 2))
        values = jax.vmap(one_log_prob)(flat)
        return values.reshape(theta.shape[:-1])


def make_posterior_density_model(cfg: Config, *, key: Array):
    """Instantiate exactly one normalized density family."""
    if cfg.posterior_density_mode == "gmm":
        return PosteriorDensityGMM(cfg, key=key)
    if cfg.posterior_density_mode == "mlp":
        return PosteriorDensityMLP(cfg, key=key)
    raise ValueError(f"Unknown posterior_density_mode={cfg.posterior_density_mode!r}.")


def density_target_objective(
    density_model,
    x_batch: Array,
    target_theta: Array,
):
    """Proper logarithmic score for normalized q_psi(theta|x), whichever density mode is active."""
    target_log_prob = jax.vmap(density_model.log_prob)(x_batch, target_theta)
    target_nll = -jnp.mean(target_log_prob)
    metrics = {
        "target_nll": target_nll,
        "mean_target_log_prob": jnp.mean(target_log_prob),
    }
    return target_nll, metrics


_density_loss_and_grad = eqx.filter_value_and_grad(density_target_objective, has_aux=True)


def make_density_train_step(optimizer: optax.GradientTransformation):
    @eqx.filter_jit
    def step(density_model, opt_state, x_batch, target_theta):
        (loss, metrics), grads = _density_loss_and_grad(
            density_model,
            x_batch,
            target_theta,
        )
        params = eqx.filter(density_model, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        density_model = eqx.apply_updates(density_model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return density_model, opt_state, loss, metrics, grad_norm

    return step


# Numerical stabilizer only; not a model/training hyperparameter.
# The pairwise energy-score matrix contains exact zero diagonal differences. The ordinary
# Euclidean norm has an undefined derivative at exactly zero, so use a float32-safe norm.
_ENERGY_NORM_EPS = 1e-12


def _stable_l2_norm(x: Array, axis: int = -1) -> Array:
    eps = jnp.asarray(_ENERGY_NORM_EPS, dtype=x.dtype)
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis) + eps)


def energy_score_terms(posterior: Array, target_theta: Array) -> tuple[Array, Array, Array]:
    """2-D empirical energy score of one particle cloud against one simulator-known theta*.

    The cloud itself is interpreted as an empirical predictive distribution with equal particle
    weights. This is the same attraction-minus-repulsion construction used in the earlier
    Bayes-Transport version, with a tiny numerical stabilizer for finite gradients at zero.
    """
    attraction = jnp.mean(
        _stable_l2_norm(posterior - target_theta[None, :], axis=-1)
    )
    pairwise = posterior[:, None, :] - posterior[None, :, :]
    repulsion = jnp.mean(_stable_l2_norm(pairwise, axis=-1))
    score = attraction - 0.5 * repulsion
    return score, attraction, repulsion


def transport_batch_metrics(
    posterior: Array,
    target_theta: Array,
) -> dict[str, Array]:
    """Proper particle-distribution diagnostics and energy-score transport objective.

    posterior has shape [B,M,2], target_theta has shape [B,2]. For every simulator problem,
    theta* is the realized outcome and the transported cloud is the forecast distribution.
    """
    score, attraction, repulsion = jax.vmap(energy_score_terms)(posterior, target_theta)

    means = jnp.mean(posterior, axis=1)
    mean_error = _stable_l2_norm(means - target_theta, axis=-1)
    centered = posterior - means[:, None, :]
    covariance_trace = jnp.mean(jnp.sum(centered**2, axis=-1), axis=1)
    outside = jnp.any(
        (posterior < CFG.prior_low) | (posterior > CFG.prior_high),
        axis=-1,
    )

    return {
        "loss": jnp.mean(score),
        "energy_score": jnp.mean(score),
        "attraction": jnp.mean(attraction),
        "repulsion": jnp.mean(repulsion),
        "mean_error": jnp.mean(mean_error),
        "covariance_trace": jnp.mean(covariance_trace),
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
        lambda p, x, k: model(p, x, key=k, inference=False)
    )(prior_theta, x_batch, row_keys)

    metrics = transport_batch_metrics(posterior, target_theta)
    return metrics["loss"], (metrics, posterior)


_transport_loss_and_grad = eqx.filter_value_and_grad(transport_objective, has_aux=True)


def make_transport_train_step(optimizer: optax.GradientTransformation):
    @eqx.filter_jit
    def step(model, opt_state, prior_theta, x_batch, target_theta, dropout_key):
        (loss, (metrics, posterior)), grads = _transport_loss_and_grad(
            model,
            prior_theta,
            x_batch,
            target_theta,
            dropout_key,
        )
        params = eqx.filter(model, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return model, opt_state, loss, metrics, posterior, grad_norm

    return step


# Same transport architecture/hyperparameters as the supplied script.
model = ConditionalParticleTransport(CFG, key=jax.random.key(CFG.seed))
density_model = make_posterior_density_model(CFG, key=jax.random.key(CFG.seed + 1))

transport_optimizer = optax.chain(
    optax.clip_by_global_norm(CFG.grad_clip_norm),
    optax.adamw(CFG.learning_rate, weight_decay=CFG.weight_decay),
)
density_optimizer = optax.chain(
    optax.clip_by_global_norm(CFG.grad_clip_norm),
    optax.adamw(CFG.learning_rate, weight_decay=CFG.weight_decay),
)

transport_opt_state = transport_optimizer.init(eqx.filter(model, eqx.is_array))
density_opt_state = density_optimizer.init(eqx.filter(density_model, eqx.is_array))
transport_train_step = make_transport_train_step(transport_optimizer)
density_train_step = make_density_train_step(density_optimizer)

print("Transport model initialized.")
print("Conditioning:", CFG.posterior_conditioning)
print("Training batch size (independent simulator pairs):", CFG.batch_size)
print("Training particles per inference problem:", CFG.num_particles)
if CFG.posterior_density_mode == "gmm":
    print(
        "Posterior density mode: GMM | "
        f"conditioning MLP hidden={CFG.posterior_density_hidden_dim} x {CFG.posterior_density_depth}, "
        f"components={CFG.posterior_density_components}"
    )
else:
    print(
        "Posterior density mode: MLP flow | "
        f"coupling MLP hidden={CFG.posterior_density_hidden_dim} x {CFG.posterior_density_depth}, "
        f"coupling layers={max(2, 2 * CFG.posterior_density_depth)}"
    )


#%% 5) Ground-truth posterior, plotting, metrics, and checkpoint utilities

def normal_pdf_sd_np(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float64) - float(mean)) / float(std)
    return np.exp(-0.5 * z**2) / (float(std) * math.sqrt(2.0 * math.pi))


def crescent_p_density_np(p: np.ndarray, cfg: Config = CFG) -> np.ndarray:
    """Exact density of the intermediate p=(r cos a + .25, r sin a).

    With a in [-pi/2,pi/2], each nonzero point has exactly one valid signed-r branch:
    positive r when p1-.25 >= 0 and negative r otherwise. The polar Jacobian contributes 1/|r|.
    Negative-r mass is tiny for N(.1,.01^2), but including it makes this diagnostic density exact
    for the simulator as written rather than silently truncating r at zero.
    """
    p = np.asarray(p, dtype=np.float64)
    u = p[..., 0] - float(cfg.crescent_x_offset)
    v = p[..., 1]
    rho = np.sqrt(u**2 + v**2)
    rho_safe = np.maximum(rho, 1e-12)
    signed_r = np.where(u >= 0.0, rho, -rho)
    density = normal_pdf_sd_np(signed_r, cfg.radial_mean, cfg.radial_std) / (math.pi * rho_safe)
    return np.where(rho > 1e-12, density, 0.0)


def two_moons_likelihood_density_np(
    x: np.ndarray,
    theta: np.ndarray,
    cfg: Config = CFG,
) -> np.ndarray:
    """Closed diagnostic density implied by the toy simulator; training does NOT use this function."""
    x = np.asarray(x, dtype=np.float64).reshape(2)
    theta = np.asarray(theta, dtype=np.float64)
    theta_sum = theta[..., 0] + theta[..., 1]
    shift1 = -np.abs(theta_sum) / math.sqrt(2.0)
    shift2 = (-theta[..., 0] + theta[..., 1]) / math.sqrt(2.0)
    p_required = np.stack([x[0] - shift1, x[1] - shift2], axis=-1)
    return crescent_p_density_np(p_required, cfg)


def exact_posterior_grid(x: np.ndarray = X_OBS, cfg: Config = CFG):
    """Numerically normalized ground-truth posterior on the exact square prior support."""
    t1 = np.linspace(cfg.prior_low, cfg.prior_high, cfg.posterior_grid_size, dtype=np.float64)
    t2 = np.linspace(cfg.prior_low, cfg.prior_high, cfg.posterior_grid_size, dtype=np.float64)
    g1, g2 = np.meshgrid(t1, t2, indexing="xy")
    theta = np.stack([g1, g2], axis=-1)

    # Prior is constant on the grid support, so posterior shape is proportional to likelihood.
    density = two_moons_likelihood_density_np(x, theta, cfg)
    d1 = t1[1] - t1[0]
    d2 = t2[1] - t2[0]
    z = np.sum(density) * d1 * d2
    if not np.isfinite(z) or z <= 0.0:
        raise FloatingPointError("Ground-truth posterior grid failed to normalize.")
    density = density / z
    return t1, t2, density


def sample_from_grid_posterior(
    rng: np.random.Generator,
    theta1: np.ndarray,
    theta2: np.ndarray,
    density: np.ndarray,
    n: int,
) -> np.ndarray:
    p = np.asarray(density, dtype=np.float64).reshape(-1)
    p /= p.sum()
    ids = rng.choice(len(p), size=int(n), replace=True, p=p)
    i2, i1 = np.unravel_index(ids, density.shape)
    samples = np.column_stack([theta1[i1], theta2[i2]]).astype(np.float64)
    d1 = theta1[1] - theta1[0]
    d2 = theta2[1] - theta2[0]
    samples[:, 0] += rng.uniform(-0.5 * d1, 0.5 * d1, size=len(samples))
    samples[:, 1] += rng.uniform(-0.5 * d2, 0.5 * d2, size=len(samples))
    return samples.astype(np.float32)


def credible_density_levels(density: np.ndarray, masses=(0.50, 0.80, 0.95)) -> np.ndarray:
    flat = np.asarray(density, dtype=np.float64).reshape(-1)
    order = np.argsort(flat)[::-1]
    sorted_d = flat[order]
    cumulative = np.cumsum(sorted_d)
    cumulative /= cumulative[-1]
    levels = []
    for mass in masses:
        idx = min(int(np.searchsorted(cumulative, mass)), len(sorted_d) - 1)
        levels.append(sorted_d[idx])
    return np.asarray(sorted(set(levels)), dtype=np.float64)


def save_model(path: Path, model: ConditionalParticleTransport, cfg: Config = CFG) -> None:
    """Save transport model (kept for compatibility with v1 checkpoints)."""
    path = Path(path)
    eqx.tree_serialise_leaves(path, model)
    with path.with_suffix(".json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_model(path: Path, cfg: Config = CFG) -> ConditionalParticleTransport:
    template = ConditionalParticleTransport(cfg, key=jax.random.key(cfg.seed))
    return eqx.tree_deserialise_leaves(Path(path), template)


def save_density_model(path: Path, density_model) -> None:
    eqx.tree_serialise_leaves(Path(path), density_model)


def load_density_model(path: Path, cfg: Config = CFG):
    template = make_posterior_density_model(cfg, key=jax.random.key(cfg.seed + 1))
    return eqx.tree_deserialise_leaves(Path(path), template)


def evaluate_bt(
    model: ConditionalParticleTransport,
    prior_particles: np.ndarray,
    x: np.ndarray = X_OBS,
) -> np.ndarray:
    return np.asarray(
        jax.device_get(
            model(
                jnp.asarray(prior_particles),
                jnp.asarray(x),
                key=None,
                inference=True,
            )
        ),
        dtype=np.float32,
    )


def rolling_mean(x: np.ndarray, window: int = 100) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return x.copy()
    window = max(1, min(int(window), len(x)))
    kernel = np.ones(window, dtype=np.float64) / window
    y = np.convolve(x, kernel, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), y])


def plot_training_diagnostics(history: dict[str, list[float]], cfg: Config = CFG) -> None:
    step = np.asarray(history["step"])
    sims = np.asarray(history["simulations_seen"])

    fig, axes = plt.subplots(4, 2, figsize=(15, 18))

    ax = axes[0, 0]
    energy = np.asarray(history["energy_score"])
    ax.plot(step, energy, alpha=0.30, label="Transport energy score")
    ax.plot(step, rolling_mean(energy, 100), linewidth=2, label="100-step mean")
    ax.set_title("Bayes Transport proper-score loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("energy score")
    ax.legend()

    ax = axes[0, 1]
    target_nll = np.asarray(history["target_nll"])
    ax.plot(step, target_nll, alpha=0.30, label=r"Density NLL $-\log q_\psi(\theta^*|x)$")
    ax.plot(step, rolling_mean(target_nll, 100), linewidth=2, label="100-step mean")
    ax.set_title(f"Normalized density proper-score loss ({cfg.posterior_density_mode})")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("negative log probability")
    ax.legend()

    ax = axes[1, 0]
    tg = np.asarray(history["transport_grad_norm"])
    dg = np.asarray(history["density_grad_norm"])
    ax.plot(step, np.maximum(tg, 1e-16), label="Transport grad norm")
    ax.plot(step, np.maximum(dg, 1e-16), label="Density-model grad norm")
    ax.set_yscale("log")
    ax.set_title("Gradient norms")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("global norm")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(step, history["attraction"], label="Attraction")
    ax.plot(step, history["repulsion"], label="Repulsion")
    ax.set_title("Energy-score components")
    ax.set_xlabel("optimizer step")
    ax.legend()

    ax = axes[2, 0]
    ax.plot(step, history["mean_error"])
    ax.set_title(r"Posterior-cloud mean error to simulator-known $\theta^*$")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Euclidean error")

    ax = axes[2, 1]
    ax.plot(step, history["covariance_trace"], label="Cloud covariance trace")
    ax.plot(step, history["outside_prior_fraction"], label="Fraction outside [-1,1]^2")
    ax.set_title("Posterior cloud geometry")
    ax.set_xlabel("optimizer step")
    ax.legend()

    ax = axes[3, 0]
    ax.plot(step, history["mean_target_log_prob"], label=r"Mean target log $q_\psi(\theta^*|x)$")
    ax.set_title("Density-model fit to simulator targets")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("log probability")
    ax.legend()

    ax = axes[3, 1]
    ax.plot(step, history["interpolation_fraction"], label="Interpolation fraction")
    ax.plot(step, history["buffer_fraction"], label="Buffer fraction")
    ax.plot(step, history["exact_prior_fraction"], label="Exact-prior fraction")
    ax.set_title("Training prior-source mixture")
    ax.set_xlabel("optimizer step")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    fig.suptitle(
        f"Bayes Transport two-moons v3 diagnostics | final simulations={int(sims[-1]):,}",
        fontsize=17,
    )
    fig.tight_layout()
    fig.savefig(OUT / "10_training_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].plot(sims, energy, alpha=0.30, label="Transport energy score")
    axes[0].plot(sims, rolling_mean(energy, 100), linewidth=2, label="100-step mean")
    axes[0].set_xlabel("cumulative simulator calls")
    axes[0].set_ylabel("energy score")
    axes[0].set_title("Transport loss versus simulation budget")
    axes[0].legend()

    axes[1].plot(sims, target_nll, alpha=0.30, label="Density target NLL")
    axes[1].plot(sims, rolling_mean(target_nll, 100), linewidth=2, label="100-step mean")
    axes[1].set_xlabel("cumulative simulator calls")
    axes[1].set_ylabel("negative log probability")
    axes[1].set_title("Density loss versus simulation budget")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT / "11_losses_vs_simulations.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_prior_diagnostics(prior_samples: np.ndarray, cfg: Config = CFG) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].scatter(prior_samples[:, 0], prior_samples[:, 1], s=10, alpha=0.30)
    axes[0].set_xlim(cfg.prior_low, cfg.prior_high)
    axes[0].set_ylim(cfg.prior_low, cfg.prior_high)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel(r"$\theta_1$")
    axes[0].set_ylabel(r"$\theta_2$")
    axes[0].set_title(r"Exact evaluation prior $U([-1,1]^2)$")

    for d, label in enumerate([r"$\theta_1$", r"$\theta_2$"]):
        ax = axes[d + 1]
        ax.hist(prior_samples[:, d], bins=40, density=True, alpha=0.55)
        ax.hlines(
            1.0 / (cfg.prior_high - cfg.prior_low),
            cfg.prior_low,
            cfg.prior_high,
            linewidth=2,
            label="Exact uniform density",
        )
        ax.set_xlim(cfg.prior_low - 0.1, cfg.prior_high + 0.1)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.set_title(f"Prior marginal {label}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "20_exact_evaluation_prior.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_training_prior_examples(cfg: Config = CFG) -> None:
    rng = np.random.default_rng(cfg.seed + 700)
    theta_anchor = np.asarray([0.35, -0.20], dtype=np.float32)
    taus = [cfg.prior_interpolation_tau_min, 0.5 * (cfg.prior_interpolation_tau_min + cfg.prior_interpolation_tau_max), cfg.prior_interpolation_tau_max]
    base = sample_interpolation_base_cloud_np(rng, 1500, cfg)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
    axes[0].scatter(base[:, 0], base[:, 1], s=5, alpha=0.25)
    axes[0].set_title(f"Interpolation base: {cfg.interpolation_base_cloud}")
    for ax, tau in zip(axes[1:], taus):
        cloud = (1.0 - tau) * base + tau * theta_anchor[None, :]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=5, alpha=0.25)
        ax.scatter([theta_anchor[0]], [theta_anchor[1]], marker="*", s=120)
        ax.set_title(rf"$\tau={tau:.2f}$")
    for ax in axes:
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Training-only prior interpolation geometry")
    fig.tight_layout()
    fig.savefig(OUT / "21_training_prior_interpolation.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_samples_on_exact_contours(
    samples: np.ndarray,
    theta1_grid: np.ndarray,
    theta2_grid: np.ndarray,
    density: np.ndarray,
    title: str,
    filename: str,
) -> None:
    levels = credible_density_levels(density)
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.contour(theta1_grid, theta2_grid, density, levels=levels, linewidths=2, cmap="viridis")
    ax.scatter(samples[:, 0], samples[:, 1], s=12, alpha=0.35, label="Bayes Transport particles")
    ax.set_xlim(CFG.prior_low, CFG.prior_high)
    ax.set_ylim(CFG.prior_low, CFG.prior_high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.show()


def kde_density_on_grid(
    samples: np.ndarray,
    theta1_grid: np.ndarray,
    theta2_grid: np.ndarray,
) -> np.ndarray:
    """Smooth particle density for Figure-1-style visualization only."""
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if len(samples) < 3:
        return np.zeros((len(theta2_grid), len(theta1_grid)), dtype=np.float64)

    g1, g2 = np.meshgrid(theta1_grid, theta2_grid, indexing="xy")
    points = np.vstack([g1.ravel(), g2.ravel()])

    try:
        kde = gaussian_kde(samples.T, bw_method="scott")
    except np.linalg.LinAlgError:
        # Diagnostic-only jitter fallback for an accidentally near-singular particle cloud.
        rng = np.random.default_rng(991)
        jittered = samples + rng.normal(0.0, 1e-4, size=samples.shape)
        kde = gaussian_kde(jittered.T, bw_method="scott")

    return kde(points).reshape(g1.shape)


def plot_true_posterior_viridis(
    theta1_grid: np.ndarray,
    theta2_grid: np.ndarray,
    density: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    im = ax.imshow(
        density,
        origin="lower",
        extent=[theta1_grid[0], theta1_grid[-1], theta2_grid[0], theta2_grid[-1]],
        cmap="viridis",
        aspect="equal",
    )
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title(r"Ground-truth $p(\theta\mid x_o)$ for $x_o=(0,0)$")
    fig.colorbar(im, ax=ax, label="posterior density")
    fig.tight_layout()
    fig.savefig(OUT / "30_true_posterior_viridis.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_figure1_style(
    true_theta1: np.ndarray,
    true_theta2: np.ndarray,
    true_density: np.ndarray,
    snapshot_samples: dict[int, tuple[int, np.ndarray]],
    cfg: Config = CFG,
) -> None:
    """Single-row analogue of Figure 1: true posterior + Bayes Transport training snapshots."""
    snapshot_budgets = [b for b in cfg.figure1_simulation_budgets if b in snapshot_samples]
    ncols = 1 + len(snapshot_budgets)
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.1), squeeze=False)
    axes = axes.ravel()

    # Use per-panel max normalization, matching Figure 1's emphasis on posterior SHAPE.
    d0 = true_density / max(float(np.max(true_density)), 1e-12)
    axes[0].imshow(
        d0,
        origin="lower",
        extent=[cfg.prior_low, cfg.prior_high, cfg.prior_low, cfg.prior_high],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    axes[0].set_title("True posterior")

    kde_axis = np.linspace(cfg.prior_low, cfg.prior_high, cfg.kde_grid_size)
    for ax, budget in zip(axes[1:], snapshot_budgets):
        sims_seen, samples = snapshot_samples[budget]
        d = kde_density_on_grid(samples, kde_axis, kde_axis)
        d = d / max(float(np.max(d)), 1e-12)
        ax.imshow(
            d,
            origin="lower",
            extent=[cfg.prior_low, cfg.prior_high, cfg.prior_low, cfg.prior_high],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        ax.set_title(f"BT target N={budget:,}\nactual N={sims_seen:,}")

    for ax in axes:
        ax.set_xlim(cfg.prior_low, cfg.prior_high)
        ax.set_ylim(cfg.prior_low, cfg.prior_high)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    fig.suptitle("Two moons — Bayes Transport analogue of Figure 1", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "31_figure1_style_bayes_transport_viridis.png", dpi=220, bbox_inches="tight")
    plt.show()


def plot_final_density_comparison(
    theta1_grid: np.ndarray,
    theta2_grid: np.ndarray,
    exact_density: np.ndarray,
    bt_samples: np.ndarray,
    cfg: Config = CFG,
) -> None:
    kde_axis = np.linspace(cfg.prior_low, cfg.prior_high, cfg.kde_grid_size)
    bt_density = kde_density_on_grid(bt_samples, kde_axis, kde_axis)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].imshow(
        exact_density / np.max(exact_density),
        origin="lower",
        extent=[cfg.prior_low, cfg.prior_high, cfg.prior_low, cfg.prior_high],
        cmap="viridis",
        vmin=0,
        vmax=1,
        aspect="equal",
    )
    axes[0].set_title("Ground-truth posterior")

    axes[1].imshow(
        bt_density / max(float(np.max(bt_density)), 1e-12),
        origin="lower",
        extent=[cfg.prior_low, cfg.prior_high, cfg.prior_low, cfg.prior_high],
        cmap="viridis",
        vmin=0,
        vmax=1,
        aspect="equal",
    )
    axes[1].set_title("Bayes Transport KDE")

    levels = credible_density_levels(exact_density)
    axes[2].contour(theta1_grid, theta2_grid, exact_density, levels=levels, linewidths=1.7, cmap="viridis")
    axes[2].scatter(bt_samples[:, 0], bt_samples[:, 1], s=7, alpha=0.3)
    axes[2].set_title("BT particles + exact contours")

    for ax in axes:
        ax.set_xlim(cfg.prior_low, cfg.prior_high)
        ax.set_ylim(cfg.prior_low, cfg.prior_high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.grid(False)

    fig.tight_layout()
    fig.savefig(OUT / "32_final_density_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_marginals(exact_samples: np.ndarray, bt_samples: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for d, label in enumerate([r"$\theta_1$", r"$\theta_2$"]):
        axes[d].hist(
            exact_samples[:, d], bins=70, density=True, histtype="step", linewidth=2,
            label="Ground truth"
        )
        axes[d].hist(
            bt_samples[:, d], bins=70, density=True, histtype="step", linewidth=2,
            label="Bayes Transport"
        )
        axes[d].set_xlim(CFG.prior_low, CFG.prior_high)
        axes[d].set_xlabel(label)
        axes[d].set_ylabel("density")
        axes[d].set_title(f"Posterior marginal {label}")
        axes[d].legend()
    fig.tight_layout()
    fig.savefig(OUT / "33_posterior_marginals.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_prior_to_posterior_transport(prior: np.ndarray, posterior: np.ndarray) -> None:
    n = min(len(prior), len(posterior), 220)
    ids = np.linspace(0, len(prior) - 1, n, dtype=int)
    p0 = prior[ids]
    p1 = posterior[ids]

    fig, ax = plt.subplots(figsize=(7.3, 7.0))
    ax.scatter(p0[:, 0], p0[:, 1], s=15, alpha=0.35, label="Prior particles")
    ax.scatter(p1[:, 0], p1[:, 1], s=15, alpha=0.45, label="Posterior particles")
    ax.quiver(
        p0[:, 0], p0[:, 1],
        p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1],
        angles="xy", scale_units="xy", scale=1.0, alpha=0.25, width=0.002,
    )
    ax.set_xlim(CFG.prior_low - 0.35, CFG.prior_high + 0.35)
    ax.set_ylim(CFG.prior_low - 0.35, CFG.prior_high + 0.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title("Learned prior-to-posterior particle displacement at $x_o$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "34_prior_to_posterior_transport.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_posterior_predictive(bt_samples: np.ndarray, cfg: Config = CFG) -> None:
    rng = np.random.default_rng(cfg.seed + 8800)
    ids = rng.choice(len(bt_samples), size=min(5000, len(bt_samples)), replace=True)
    x_pp = simulate_two_moons_batch_np(rng, bt_samples[ids])

    fig, ax = plt.subplots(figsize=(7.3, 7.0))
    h = ax.hist2d(x_pp[:, 0], x_pp[:, 1], bins=120, cmap="viridis", density=True)
    ax.scatter([X_OBS[0]], [X_OBS[1]], marker="*", s=250, c="white", edgecolors="black", linewidths=1.2)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Posterior predictive simulator outputs from Bayes Transport particles")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(h[3], ax=ax, label="predictive density")
    fig.tight_layout()
    fig.savefig(OUT / "35_posterior_predictive.png", dpi=180, bbox_inches="tight")
    plt.show()


def posterior_density_on_grid(
    density_model,
    x: np.ndarray,
    theta1_grid: np.ndarray,
    theta2_grid: np.ndarray,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Evaluate the active normalized learned q_psi(theta|x) on a diagnostic grid."""
    g1, g2 = np.meshgrid(theta1_grid, theta2_grid, indexing="xy")
    points = np.column_stack([g1.ravel(), g2.ravel()]).astype(np.float32)
    xj = jnp.asarray(x, dtype=jnp.float32)
    logp_chunks = []
    for start in range(0, len(points), int(chunk_size)):
        chunk = jnp.asarray(points[start:start + int(chunk_size)])
        logp = jax.vmap(lambda th: density_model.log_prob(xj, th))(chunk)
        logp_chunks.append(np.asarray(jax.device_get(logp), dtype=np.float64))
    logp = np.concatenate(logp_chunks).reshape(g1.shape)
    return np.exp(logp)


def plot_density_model_comparison(
    density_model,
    theta1_grid: np.ndarray,
    theta2_grid: np.ndarray,
    exact_density: np.ndarray,
    x: np.ndarray = X_OBS,
) -> None:
    learned = posterior_density_on_grid(
        density_model, x, theta1_grid, theta2_grid
    )
    learned_shape = learned / max(float(np.max(learned)), 1e-12)
    exact_shape = exact_density / max(float(np.max(exact_density)), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].imshow(
        exact_shape,
        origin="lower",
        extent=[theta1_grid[0], theta1_grid[-1], theta2_grid[0], theta2_grid[-1]],
        cmap="viridis", vmin=0, vmax=1, aspect="equal",
    )
    axes[0].set_title("Ground-truth posterior")

    axes[1].imshow(
        learned_shape,
        origin="lower",
        extent=[theta1_grid[0], theta1_grid[-1], theta2_grid[0], theta2_grid[-1]],
        cmap="viridis", vmin=0, vmax=1, aspect="equal",
    )
    axes[1].set_title(r"Learned density $q_\psi(\theta|x_o)$")

    axes[2].imshow(
        learned_shape - exact_shape,
        origin="lower",
        extent=[theta1_grid[0], theta1_grid[-1], theta2_grid[0], theta2_grid[-1]],
        cmap="coolwarm", aspect="equal",
    )
    axes[2].set_title("Learned shape - exact shape")

    for ax in axes:
        ax.set_xlim(CFG.prior_low, CFG.prior_high)
        ax.set_ylim(CFG.prior_low, CFG.prior_high)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUT / "36_density_mlp_vs_exact.png", dpi=200, bbox_inches="tight")
    plt.show()


def posterior_summary(samples: np.ndarray) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=np.float64)
    return {
        "mean": np.mean(samples, axis=0),
        "cov": np.cov(samples.T),
        "outside_prior_fraction": float(np.mean(np.any((samples < CFG.prior_low) | (samples > CFG.prior_high), axis=1))),
    }


def energy_distance_samples(a: np.ndarray, b: np.ndarray, max_n: int = 2500) -> float:
    rng = np.random.default_rng(123)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]
    ab = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).mean()
    aa = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1).mean()
    bb = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=-1).mean()
    return float(2.0 * ab - aa - bb)


def sliced_wasserstein(a: np.ndarray, b: np.ndarray, n_proj: int = 128) -> float:
    rng = np.random.default_rng(456)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b), 5000)
    a = a[rng.choice(len(a), n, replace=False)]
    b = b[rng.choice(len(b), n, replace=False)]
    directions = rng.normal(size=(n_proj, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return float(np.mean([
        np.mean(np.abs(np.sort(a @ u) - np.sort(b @ u)))
        for u in directions
    ]))


def rbf_mmd2(a: np.ndarray, b: np.ndarray, max_n: int = 2500) -> float:
    """Biased RBF MMD^2 with pooled median-distance bandwidth; useful for comparison with the paper."""
    rng = np.random.default_rng(789)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]

    pooled = np.concatenate([a, b], axis=0)
    m = min(len(pooled), 1200)
    probe = pooled[rng.choice(len(pooled), m, replace=False)]
    d2_probe = np.sum((probe[:, None, :] - probe[None, :, :]) ** 2, axis=-1)
    tri = d2_probe[np.triu_indices(m, k=1)]
    positive = tri[tri > 0]
    sigma2 = float(np.median(positive)) if len(positive) else 1.0
    sigma2 = max(sigma2, 1e-8)

    def kernel(x, y):
        d2 = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
        return np.exp(-0.5 * d2 / sigma2)

    return float(kernel(a, a).mean() + kernel(b, b).mean() - 2.0 * kernel(a, b).mean())


def comparison_metrics(exact: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    e = posterior_summary(exact)
    c = posterior_summary(candidate)
    return {
        "mean_error": float(np.linalg.norm(c["mean"] - e["mean"])),
        "covariance_frobenius_error": float(np.linalg.norm(c["cov"] - e["cov"], ord="fro")),
        "outside_prior_fraction": float(c["outside_prior_fraction"]),
        "sliced_wasserstein": sliced_wasserstein(exact, candidate, CFG.sliced_wasserstein_projections),
        "energy_distance": energy_distance_samples(exact, candidate),
        "rbf_mmd2": rbf_mmd2(exact, candidate),
    }


# Ground truth is diagnostic-only. Neither the density-MLP NLL nor the transport energy-score calls this exact density.
THETA1_GRID, THETA2_GRID, EXACT_DENSITY = exact_posterior_grid(X_OBS, CFG)
_exact_rng = np.random.default_rng(CFG.seed + 500)
EXACT_SAMPLES = sample_from_grid_posterior(
    _exact_rng,
    THETA1_GRID,
    THETA2_GRID,
    EXACT_DENSITY,
    CFG.exact_reference_samples,
)
plot_true_posterior_viridis(THETA1_GRID, THETA2_GRID, EXACT_DENSITY)


#%% 6) TRAIN BAYES TRANSPORT v3 — proper score for BOTH learned components
# No analytic likelihood density and no ground-truth posterior density is used in this cell.
# Every row of every optimization batch is a fresh simulator-supervised problem:
#     theta*_b ~ U([-1,1]^2),  x_b ~ simulator(theta*_b).
#
# Two independent proper-scoring-rule updates happen on the SAME fresh simulator batch:
#   (1) density model q_psi: logarithmic score / NLL
#           -log q_psi(theta*_b | x_b)
#   (2) Bayes Transport T_phi: multivariate energy score of the transported particle cloud
#           ES({theta_BT,b,j}, theta*_b)
#
# The learned density q_psi is NOT used in the transport loss. It is trained in parallel as a
# second posterior representation / diagnostic. Thus each component is supervised directly by the
# simulator-known theta* using a scoring rule appropriate to the representation it produces.

train_rng = np.random.default_rng(CFG.seed + 10_001)
mode_rng = np.random.default_rng(CFG.seed + 20_003)
dropout_key = jax.random.key(CFG.seed + 30_007)
replay_buffer = HistoricalPosteriorBuffer(
    CFG.historical_output_buffer_capacity,
    CFG.num_particles,
)

history = {name: [] for name in (
    "step",
    "simulations_seen",
    "energy_score",
    "attraction",
    "repulsion",
    "target_nll",
    "mean_target_log_prob",
    "mean_error",
    "covariance_trace",
    "outside_prior_fraction",
    "transport_grad_norm",
    "density_grad_norm",
    "interpolation_fraction",
    "buffer_fraction",
    "exact_prior_fraction",
    "mean_interpolation_tau",
    "mean_replay_distance",
    "buffer_size",
)}

# Fixed exact evaluation prior used ONLY for non-training diagnostic snapshots.
_snapshot_rng = np.random.default_rng(CFG.seed + 40_009)
FIGURE1_PRIOR_PARTICLES = sample_exact_prior_np(_snapshot_rng, CFG.eval_particles)
figure1_snapshot_samples: dict[int, tuple[int, np.ndarray]] = {}

for step in range(1, CFG.training_steps + 1):
    # Fresh simulator-supervised minibatch shared by both objectives.
    theta_target = sample_exact_prior_np(train_rng, CFG.batch_size)
    x_batch = simulate_two_moons_batch_np(train_rng, theta_target)

    # (1) Proper logarithmic score for the selected normalized density model.
    density_model, density_opt_state, density_loss, density_metrics, density_grad_norm = density_train_step(
        density_model,
        density_opt_state,
        jnp.asarray(x_batch),
        jnp.asarray(theta_target),
    )

    # Mutually-exclusive particle-prior source per row; unchanged from the supplied v2.
    prior_theta, prior_info = make_training_prior_batch_np(
        train_rng,
        mode_rng,
        theta_target,
        x_batch,
        replay_buffer,
        CFG,
    )

    # (2) Proper multivariate energy score for the Bayes-Transport empirical particle distribution.
    # q_psi does not appear anywhere in this objective.
    dropout_key, step_key = jax.random.split(dropout_key)
    model, transport_opt_state, loss, transport_metrics, posterior_batch, transport_grad_norm = transport_train_step(
        model,
        transport_opt_state,
        jnp.asarray(prior_theta),
        jnp.asarray(x_batch),
        jnp.asarray(theta_target),
        step_key,
    )

    posterior_batch_np = np.asarray(jax.device_get(posterior_batch), dtype=np.float32)
    replay_buffer.add_batch(x_batch, posterior_batch_np)

    host_t = jax.device_get(transport_metrics)
    host_d = jax.device_get(density_metrics)
    interp_tau = prior_info["interpolation_tau"]
    replay_dist = prior_info["replay_distance"]
    positive_tau = interp_tau[prior_info["interpolation_used"] > 0]
    finite_replay_dist = replay_dist[np.isfinite(replay_dist)]

    scalar_values = {
        "step": float(step),
        "simulations_seen": float(step * CFG.batch_size),
        "energy_score": float(host_t["energy_score"]),
        "attraction": float(host_t["attraction"]),
        "repulsion": float(host_t["repulsion"]),
        "target_nll": float(host_d["target_nll"]),
        "mean_target_log_prob": float(host_d["mean_target_log_prob"]),
        "mean_error": float(host_t["mean_error"]),
        "covariance_trace": float(host_t["covariance_trace"]),
        "outside_prior_fraction": float(host_t["outside_prior_fraction"]),
        "transport_grad_norm": float(jax.device_get(transport_grad_norm)),
        "density_grad_norm": float(jax.device_get(density_grad_norm)),
        "interpolation_fraction": float(np.mean(prior_info["interpolation_used"])),
        "buffer_fraction": float(np.mean(prior_info["buffer_used"])),
        "exact_prior_fraction": float(np.mean(prior_info["exact_prior_used"])),
        "mean_interpolation_tau": float(np.mean(positive_tau)) if len(positive_tau) else np.nan,
        "mean_replay_distance": float(np.mean(finite_replay_dist)) if len(finite_replay_dist) else np.nan,
        "buffer_size": float(len(replay_buffer)),
    }
    for name, value in scalar_values.items():
        history[name].append(value)

    # Figure-1-style snapshots are evaluation-only: x_o is never part of the gradient update.
    previous_simulations = (step - 1) * CFG.batch_size
    current_simulations = step * CFG.batch_size
    for budget in CFG.figure1_simulation_budgets:
        if (
            budget not in figure1_snapshot_samples
            and previous_simulations < budget <= current_simulations
        ):
            figure1_snapshot_samples[budget] = (
                current_simulations,
                evaluate_bt(model, FIGURE1_PRIOR_PARTICLES, X_OBS),
            )

    if step == 1 or step % CFG.log_every == 0 or step == CFG.training_steps:
        print(
            f"step {step:6d}/{CFG.training_steps} | "
            f"sims {step * CFG.batch_size:9,d} | "
            f"ES {scalar_values['energy_score']:.5f} | "
            f"density-NLL {scalar_values['target_nll']:.5f} | "
            f"mean-err {scalar_values['mean_error']:.4f} | "
            f"grad(T/q) {scalar_values['transport_grad_norm']:.2e}/"
            f"{scalar_values['density_grad_norm']:.2e} | "
            f"prior i/b/e="
            f"{scalar_values['interpolation_fraction']:.2f}/"
            f"{scalar_values['buffer_fraction']:.2f}/"
            f"{scalar_values['exact_prior_fraction']:.2f}"
        )

save_model(OUT / "bayes_transport_two_moons_v3.eqx", model, CFG)
save_density_model(OUT / f"posterior_density_{CFG.posterior_density_mode}_v3.eqx", density_model)

with (OUT / "training_history.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(history.keys()))
    writer.writeheader()
    for i in range(len(history["step"])):
        writer.writerow({k: history[k][i] for k in history})

# End-of-training diagnostics for both proper objectives and both gradient streams.
plot_training_diagnostics(history, CFG)
plot_training_prior_examples(CFG)

# Plot the EXACT test-time prior before plotting the posterior.
_eval_prior_rng = np.random.default_rng(CFG.seed + 60_011)
EVAL_PRIOR_PARTICLES = sample_exact_prior_np(_eval_prior_rng, CFG.eval_particles)
plot_prior_diagnostics(EVAL_PRIOR_PARTICLES, CFG)

# Figure-1 analogue in viridis. If training ended before a requested simulation budget, it is omitted.
plot_figure1_style(
    THETA1_GRID,
    THETA2_GRID,
    EXACT_DENSITY,
    figure1_snapshot_samples,
    CFG,
)

# Inspect the independently learned normalized posterior density itself at x_o.
plot_density_model_comparison(
    density_model,
    THETA1_GRID,
    THETA2_GRID,
    EXACT_DENSITY,
    X_OBS,
)


#%% 7) FINAL EVALUATION on x_o=(0,0): exact prior -> Bayes Transport v3 posterior
# If you skipped the training cell and want to load a checkpoint, first run:
# model = load_model(OUT / "bayes_transport_two_moons_v3.eqx", CFG)
# density_model = load_density_model(OUT / f"posterior_density_{CFG.posterior_density_mode}_v3.eqx", CFG)

if "EVAL_PRIOR_PARTICLES" not in globals():
    _eval_prior_rng = np.random.default_rng(CFG.seed + 60_011)
    EVAL_PRIOR_PARTICLES = sample_exact_prior_np(_eval_prior_rng, CFG.eval_particles)

BT_POSTERIOR = evaluate_bt(model, EVAL_PRIOR_PARTICLES, X_OBS)

plot_samples_on_exact_contours(
    BT_POSTERIOR,
    THETA1_GRID,
    THETA2_GRID,
    EXACT_DENSITY,
    title=r"Bayes Transport at $x_o=(0,0)$ over ground-truth posterior contours",
    filename="40_bt_samples_exact_contours.png",
)

plot_final_density_comparison(
    THETA1_GRID,
    THETA2_GRID,
    EXACT_DENSITY,
    BT_POSTERIOR,
    CFG,
)

plot_marginals(EXACT_SAMPLES, BT_POSTERIOR)
plot_prior_to_posterior_transport(EVAL_PRIOR_PARTICLES, BT_POSTERIOR)
plot_posterior_predictive(BT_POSTERIOR, CFG)
plot_density_model_comparison(density_model, THETA1_GRID, THETA2_GRID, EXACT_DENSITY, X_OBS)

metrics = comparison_metrics(EXACT_SAMPLES, BT_POSTERIOR)
print("\nGround-truth posterior summary:")
print(posterior_summary(EXACT_SAMPLES))
print("\nBayes Transport posterior summary:")
print(posterior_summary(BT_POSTERIOR))
print("\nComparison metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.8f}")

with (OUT / "final_metrics.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
    writer.writeheader()
    writer.writerow(metrics)

np.save(OUT / "evaluation_prior_particles.npy", EVAL_PRIOR_PARTICLES)
np.save(OUT / "bt_posterior_samples_v3.npy", BT_POSTERIOR)
np.save(OUT / "ground_truth_posterior_samples.npy", EXACT_SAMPLES)
np.save(OUT / "observed_x.npy", X_OBS)


#%% 8) Optional compact publication-style panel: prior | exact | learned q | BT particles | BT density
# This cell uses only objects already produced above.

_kde_axis = np.linspace(CFG.prior_low, CFG.prior_high, CFG.kde_grid_size)
_bt_density = kde_density_on_grid(BT_POSTERIOR, _kde_axis, _kde_axis)
_levels = credible_density_levels(EXACT_DENSITY)
_q_density = posterior_density_on_grid(
    density_model, X_OBS, THETA1_GRID, THETA2_GRID
)

fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))

axes[0].scatter(EVAL_PRIOR_PARTICLES[:, 0], EVAL_PRIOR_PARTICLES[:, 1], s=7, alpha=0.30)
axes[0].set_title("Exact test prior")

axes[1].imshow(
    EXACT_DENSITY / np.max(EXACT_DENSITY),
    origin="lower",
    extent=[CFG.prior_low, CFG.prior_high, CFG.prior_low, CFG.prior_high],
    cmap="viridis",
    vmin=0,
    vmax=1,
    aspect="equal",
)
axes[1].set_title("Ground truth")

axes[2].imshow(
    _q_density / max(float(np.max(_q_density)), 1e-12),
    origin="lower",
    extent=[CFG.prior_low, CFG.prior_high, CFG.prior_low, CFG.prior_high],
    cmap="viridis",
    vmin=0,
    vmax=1,
    aspect="equal",
)
axes[2].set_title("Learned density")

axes[3].contour(THETA1_GRID, THETA2_GRID, EXACT_DENSITY, levels=_levels, cmap="viridis", linewidths=1.5)
axes[3].scatter(BT_POSTERIOR[:, 0], BT_POSTERIOR[:, 1], s=7, alpha=0.30)
axes[3].set_title("BT particles")

axes[4].imshow(
    _bt_density / max(float(np.max(_bt_density)), 1e-12),
    origin="lower",
    extent=[CFG.prior_low, CFG.prior_high, CFG.prior_low, CFG.prior_high],
    cmap="viridis",
    vmin=0,
    vmax=1,
    aspect="equal",
)
axes[4].set_title("BT KDE")

for ax in axes:
    ax.set_xlim(CFG.prior_low, CFG.prior_high)
    ax.set_ylim(CFG.prior_low, CFG.prior_high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.grid(False)

fig.suptitle(r"Two moons at $x_o=(0,0)$ — Bayes Transport v3: ES transport + NLL density", fontsize=15)
fig.tight_layout()
fig.savefig(OUT / "50_compact_publication_panel.png", dpi=220, bbox_inches="tight")
plt.show()
