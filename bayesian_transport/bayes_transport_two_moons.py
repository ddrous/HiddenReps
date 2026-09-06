#%% 0) Imports, configuration, and experiment constants
"""Bayes Transport two-moons benchmark — v4 with variable-size particle-set training.

Version 4 preserves the current evaluation/plotting path, including the v2 post-hoc normalized neural
spline-flow density estimator alongside the original crude Gaussian KDE.  The training path can now
randomize the number of particles presented to the particle Transformer, up to one configured maximum.

Run this file one #%% cell at a time in VS Code / Spyder / Jupyter-compatible editors.
There is intentionally no main() function.

Training is simulator-supervised and likelihood-free:
    theta* ~ U([-1,1]^2)
    x ~ simulator(theta*)
    prior particles ~ one of three mutually-exclusive TRAINING sources
    posterior particles = T_phi(prior particles, x)
    loss = multivariate empirical energy score against theta*

The observed datum used in the paper is x_o=(0,0). Evaluation ALWAYS starts Bayes Transport
from the exact paper prior U([-1,1]^2); training-only interpolation and replay never enter evaluation.

The implementation keeps the same Bayes-Transport design used in the previous experiment:
    * Equinox/JAX particle Transformer;
    * selectable AdaLN or cross-attention conditioning;
    * identity-initialized displacement head;
    * proper multivariate energy-score training;
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
    output_dir: str = "plots/bayes_transport_two_moons_v4"

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
    # Only the MAXIMUM training particle count is configured.  When variable_training_particles=True,
    # each optimizer step draws one set size from an automatically-derived geometric ladder ending at
    # max_training_particles.  Setting the flag False always uses the maximum, reproducing the previous
    # fixed-particle-count training path.  Evaluation remains independently controlled by eval_particles.
    max_training_particles: int = 16 * 4*1
    variable_training_particles: bool = True
    eval_particles: int = 16*16*16
    hidden_dim: int = 64 * 2
    heads: int = 4
    mlp_ratio: int = 4
    posterior_depth: int = 4
    posterior_conditioning: str = "adaln"  # {"cross_attention", "adaln"}
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
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5000.0
    log_every: int = 1250

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

    # v2 post-hoc density visualization.  This is NEVER used by the transport training objective.
    # We aggregate several independent test-prior clouds because the particle Transformer couples
    # particles within a cloud; concatenating whole clouds gives a much better empirical marginal
    # without changing the inference problem seen by the model.
    density_estimation_clouds: int = 16
    density_grid_size: int = 260

    # Neural spline flow (NSF) density estimator: an expressive, normalized change-of-variables
    # model fitted only to Bayes-Transport posterior samples after training/evaluation.
    nsf_layers: int = 8
    nsf_bins: int = 12
    nsf_hidden_dim: int = 64
    nsf_tail_bound: float = 4.0
    nsf_learning_rate: float = 2e-3
    nsf_weight_decay: float = 1e-6
    nsf_batch_size: int = 512
    nsf_max_epochs: int = 1200
    nsf_validation_fraction: float = 0.15
    nsf_validation_every: int = 10
    nsf_patience_checks: int = 30

    # Figure-1-style simulator budgets from the paper. With minibatches of 128, the snapshot is
    # taken at the first optimizer step whose cumulative simulator count reaches/exceeds each target.
    # Thus 1000 becomes 1024 calls, 5000 becomes 5120, etc.; the plot reports the actual count.
    figure1_simulation_budgets: tuple[int, ...] = (1000, 5000, 10_000)

    # Prior-predictive plot at the very start. Diagnostic only; not used for training.
    prior_predictive_plot_samples: int = 30_000


def training_particle_count_choices(max_particles: int) -> tuple[int, ...]:
    """Automatically derive a small geometric ladder of JAX-friendly training set sizes.

    There is deliberately no configurable minimum particle count.  At most five distinct shapes are
    used so variable-size training does not trigger a separate JIT compilation for every integer from
    2 to max_particles.  For the default maximum of 256 this returns (16, 32, 64, 128, 256).
    """
    max_particles = int(max_particles)
    if max_particles < 2:
        raise ValueError("max_training_particles must be at least 2.")
    ratios = (1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0)
    choices = {
        min(max_particles, max(2, int(round(max_particles * ratio))))
        for ratio in ratios
    }
    choices.add(max_particles)
    return tuple(sorted(choices))


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
if CFG.max_training_particles < 2:
    raise ValueError("max_training_particles must be at least 2.")
if CFG.density_estimation_clouds < 1 or CFG.density_grid_size < 32:
    raise ValueError("density_estimation_clouds must be >=1 and density_grid_size must be >=32.")
if CFG.nsf_layers < 1 or CFG.nsf_bins < 2 or CFG.nsf_hidden_dim < 4:
    raise ValueError("NSF requires nsf_layers>=1, nsf_bins>=2, and nsf_hidden_dim>=4.")
if CFG.nsf_tail_bound <= 0.0:
    raise ValueError("nsf_tail_bound must be positive.")
if not 0.0 < CFG.nsf_validation_fraction < 0.5:
    raise ValueError("nsf_validation_fraction must lie strictly between 0 and 0.5.")

TRAINING_PARTICLE_COUNTS = (
    training_particle_count_choices(CFG.max_training_particles)
    if CFG.variable_training_particles
    else (int(CFG.max_training_particles),)
)


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
print("Variable training particle count:", CFG.variable_training_particles)
print("Training particle-count choices:", TRAINING_PARTICLE_COUNTS)
print("Maximum training particles:", CFG.max_training_particles)
print("Evaluation prior: exact Uniform([-1,1]^2)")
print("Observed datum:", X_OBS)


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
    n_particles: int,
    cfg: Config = CFG,
) -> tuple[np.ndarray, float]:
    """Training-only C_tau=(1-tau)Z+tau*anchor with one shared tau per particle cloud."""
    n_particles = int(n_particles)
    z = sample_interpolation_base_cloud_np(rng, n_particles, cfg)

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

    v4 additionally records each cloud's actual particle count.  A replay request uses the nearest
    historical cloud with at least the requested number of particles and subsamples it without
    replacement.  Thus variable-size training never pads fake particles or duplicates particles.
    """

    def __init__(self, capacity: int, max_particles: int):
        self.capacity = int(capacity)
        self.max_particles = int(max_particles)
        self.x = np.empty((self.capacity, 2), dtype=np.float32)
        self.clouds = np.empty((self.capacity, self.max_particles, 2), dtype=np.float32)
        self.counts = np.zeros(self.capacity, dtype=np.int32)
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

    @property
    def active_counts(self) -> np.ndarray:
        return self.counts[: self.size]

    def has_cloud_with_at_least(self, n_particles: int) -> bool:
        """Whether replay can supply n_particles without replacement."""
        if self.size == 0:
            return False
        return bool(np.any(self.active_counts >= int(n_particles)))

    def add_batch(self, x: np.ndarray, clouds: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        clouds = np.asarray(clouds, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("x must have shape [B,2].")
        if clouds.ndim != 3 or clouds.shape[0] != len(x) or clouds.shape[2] != 2:
            raise ValueError("clouds must have shape [B,M,2].")
        n_particles = int(clouds.shape[1])
        if not 1 <= n_particles <= self.max_particles:
            raise ValueError(
                f"cloud particle count must lie in [1,{self.max_particles}], got {n_particles}."
            )
        for xi, cloud in zip(x, clouds):
            self.x[self.next_index] = xi
            self.clouds[self.next_index, :n_particles] = cloud
            self.counts[self.next_index] = n_particles
            self.next_index = (self.next_index + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def nearest_batch(
        self,
        x_query: np.ndarray,
        n_particles: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            raise ValueError("HistoricalPosteriorBuffer is empty.")
        n_particles = int(n_particles)
        eligible = np.flatnonzero(self.active_counts >= n_particles)
        if len(eligible) == 0:
            raise ValueError(
                f"HistoricalPosteriorBuffer has no cloud with at least {n_particles} particles."
            )

        x_query = np.asarray(x_query, dtype=np.float32).reshape(-1, 2)
        # Fixed observation scale avoids unstable standardization when the buffer is still tiny.
        candidate_x = self.active_x[eligible]
        delta = (x_query[:, None, :] - candidate_x[None, :, :]) / float(CFG.observation_scale)
        d2 = np.sum(delta**2, axis=-1)
        local_ids = np.argmin(d2, axis=1)
        ids = eligible[local_ids]
        distances = np.sqrt(d2[np.arange(len(x_query)), local_ids])

        result = np.empty((len(x_query), n_particles, 2), dtype=np.float32)
        for row, slot in enumerate(ids):
            stored_count = int(self.counts[slot])
            if stored_count == n_particles:
                # Preserve the old fixed-size replay path exactly when counts match.
                result[row] = self.clouds[slot, :n_particles]
            else:
                subset = rng.choice(stored_count, size=n_particles, replace=False)
                result[row] = self.clouds[slot, subset]
        return result, distances.astype(np.float32)


def make_training_prior_batch_np(
    rng: np.random.Generator,
    mode_rng: np.random.Generator,
    theta_target: np.ndarray,
    x_batch: np.ndarray,
    replay_buffer: HistoricalPosteriorBuffer,
    n_particles: int,
    cfg: Config = CFG,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Choose exactly one prior source independently for each training inference problem."""
    theta_target = np.asarray(theta_target, dtype=np.float32)
    x_batch = np.asarray(x_batch, dtype=np.float32)
    b = theta_target.shape[0]
    n_particles = int(n_particles)
    if not 2 <= n_particles <= cfg.max_training_particles:
        raise ValueError(
            f"training particle count must lie in [2,{cfg.max_training_particles}], got {n_particles}."
        )

    prior = np.empty((b, n_particles, 2), dtype=np.float32)
    u = mode_rng.random(b)
    p_interp = cfg.prior_interpolation_probability
    p_buffer = cfg.historical_output_prior_probability

    interp_mask = u < p_interp
    requested_buffer_mask = (u >= p_interp) & (u < p_interp + p_buffer)
    buffer_available = replay_buffer.has_cloud_with_at_least(n_particles)
    buffer_mask = requested_buffer_mask & buffer_available
    exact_mask = ~(interp_mask | buffer_mask)

    tau = np.zeros(b, dtype=np.float32)
    replay_distance = np.full(b, np.nan, dtype=np.float32)

    for i in np.flatnonzero(interp_mask):
        prior[i], tau[i] = sample_interpolated_training_prior_np(
            rng, theta_target[i], n_particles, cfg
        )

    buffer_ids = np.flatnonzero(buffer_mask)
    if len(buffer_ids):
        clouds, dist = replay_buffer.nearest_batch(
            x_batch[buffer_ids], n_particles, rng
        )
        prior[buffer_ids] = clouds
        replay_distance[buffer_ids] = dist

    exact_ids = np.flatnonzero(exact_mask)
    if len(exact_ids):
        prior[exact_ids] = sample_exact_prior_np(rng, len(exact_ids) * n_particles).reshape(
            len(exact_ids), n_particles, 2
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


#%% 4) Proper scoring rule: stable multivariate energy score + JAX/Optax train step

# Numerical stabilizer only; not a model/training hyperparameter.
# The pairwise ES matrix contains exact zero diagonal differences, and the ordinary Euclidean
# norm has an undefined derivative at exactly zero. This keeps reverse-mode gradients finite.
_ENERGY_NORM_EPS = 1e-12


def _stable_l2_norm(x: Array, axis: int = -1) -> Array:
    eps = jnp.asarray(_ENERGY_NORM_EPS, dtype=x.dtype)
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis) + eps)


def energy_score_terms(posterior: Array, target_theta: Array) -> tuple[Array, Array, Array]:
    """2-D empirical ES: E||Y-theta*|| - 1/2 E||Y-Y'||."""
    attraction = jnp.mean(_stable_l2_norm(posterior - target_theta[None, :], axis=-1))
    pairwise = posterior[:, None, :] - posterior[None, :, :]
    repulsion = jnp.mean(_stable_l2_norm(pairwise, axis=-1))
    return attraction - 0.5 * repulsion, attraction, repulsion


def batch_metrics(posterior: Array, target_theta: Array) -> dict[str, Array]:
    """posterior [B,M,2], target_theta [B,2]."""
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
    metrics = batch_metrics(posterior, target_theta)
    return metrics["loss"], (metrics, posterior)


_loss_and_grad = eqx.filter_value_and_grad(transport_objective, has_aux=True)


def make_train_step(optimizer: optax.GradientTransformation):
    @eqx.filter_jit
    def step(model, opt_state, prior_theta, x_batch, target_theta, dropout_key):
        (loss, (metrics, posterior)), grads = _loss_and_grad(
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


model = ConditionalParticleTransport(CFG, key=jax.random.key(CFG.seed))
optimizer = optax.chain(
    optax.clip_by_global_norm(CFG.grad_clip_norm),
    optax.adamw(CFG.learning_rate, weight_decay=CFG.weight_decay),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
train_step = make_train_step(optimizer)

print("Model initialized.")
print("Conditioning:", CFG.posterior_conditioning)
print("Training batch size (independent simulator pairs):", CFG.batch_size)
print("Maximum training particles per inference problem:", CFG.max_training_particles)
print("Training particle-count choices:", TRAINING_PARTICLE_COUNTS)


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
    path = Path(path)
    eqx.tree_serialise_leaves(path, model)
    with path.with_suffix(".json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_model(path: Path, cfg: Config = CFG) -> ConditionalParticleTransport:
    template = ConditionalParticleTransport(cfg, key=jax.random.key(cfg.seed))
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

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    ax = axes[0, 0]
    loss = np.asarray(history["energy_score"])
    ax.plot(step, loss, alpha=0.35, label="Energy score")
    ax.plot(step, rolling_mean(loss, 100), linewidth=2, label="100-step mean")
    ax.set_title("Training loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("energy score")
    ax.legend()

    ax = axes[0, 1]
    grad = np.asarray(history["grad_norm"])
    ax.plot(step, np.maximum(grad, 1e-16))
    ax.set_yscale("log")
    ax.set_title("Gradient norm")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("global norm")

    ax = axes[1, 0]
    ax.plot(step, history["attraction"], label="Attraction")
    ax.plot(step, history["repulsion"], label="Repulsion")
    ax.set_title("Energy-score components")
    ax.set_xlabel("optimizer step")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(step, history["mean_error"])
    ax.set_title(r"Posterior mean error to simulator-known $\theta^*$")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Euclidean error")

    ax = axes[2, 0]
    ax.plot(step, history["covariance_trace"], label="Cloud covariance trace")
    ax.plot(step, history["outside_prior_fraction"], label="Fraction outside [-1,1]^2")
    ax.set_title("Posterior cloud geometry")
    ax.set_xlabel("optimizer step")
    ax.legend()

    ax = axes[2, 1]
    ax.plot(step, history["interpolation_fraction"], label="Interpolation fraction")
    ax.plot(step, history["buffer_fraction"], label="Buffer fraction")
    ax.plot(step, history["exact_prior_fraction"], label="Exact-prior fraction")
    ax.set_title("Training prior-source mixture")
    ax.set_xlabel("optimizer step")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    fig.suptitle(
        f"Bayes Transport two-moons training diagnostics | final simulations={int(sims[-1]):,}",
        fontsize=17,
    )
    fig.tight_layout()
    fig.savefig(OUT / "10_training_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(sims, loss, alpha=0.35, label="Energy score")
    ax.plot(sims, rolling_mean(loss, 100), linewidth=2, label="100-step mean")
    ax.set_xlabel("cumulative simulator calls")
    ax.set_ylabel("energy score")
    ax.set_title("Training loss versus simulation budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "11_loss_vs_simulations.png", dpi=180, bbox_inches="tight")
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



# -----------------------------------------------------------------------------
# v2: post-hoc neural spline-flow density estimation for posterior visualization
# -----------------------------------------------------------------------------
# The Scott-rule Gaussian KDE above is intentionally retained as a crude baseline.
# The estimator below is a proper normalized density model: a stack of alternating
# coupling transforms whose scalar maps are monotone rational-quadratic splines.
# It is trained by maximum likelihood ONLY on Bayes-Transport posterior particles.
# The exact posterior density is never used to fit or select this estimator.

_NSF_MIN_BIN_WIDTH = 1e-3
_NSF_MIN_BIN_HEIGHT = 1e-3
_NSF_MIN_DERIVATIVE = 1e-3
_LOG_2PI = math.log(2.0 * math.pi)


def _inverse_softplus_np(y: float) -> float:
    y = float(y)
    return math.log(math.expm1(y))


def _init_nsf_conditioner(
    key: Array,
    hidden_dim: int,
    output_dim: int,
    derivative_bias_start: int,
    derivative_bias_value: float,
) -> dict[str, Array]:
    """Small MLP conditioner, initialized close to the identity spline."""
    k1, k2, k3 = jax.random.split(key, 3)
    w1 = jax.random.normal(k1, (1, hidden_dim)) / math.sqrt(1.0)
    w2 = jax.random.normal(k2, (hidden_dim, hidden_dim)) / math.sqrt(float(hidden_dim))
    # Tiny last layer: starts very close to equal-width/equal-height identity bins.
    w3 = 1e-3 * jax.random.normal(k3, (hidden_dim, output_dim)) / math.sqrt(float(hidden_dim))
    b1 = jnp.zeros((hidden_dim,))
    b2 = jnp.zeros((hidden_dim,))
    b3 = jnp.zeros((output_dim,))
    if derivative_bias_start < output_dim:
        b3 = b3.at[derivative_bias_start:].set(derivative_bias_value)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}


def _nsf_conditioner(params: dict[str, Array], context: Array) -> Array:
    context = jnp.reshape(context, (-1, 1))
    h = jax.nn.silu(context @ params["w1"] + params["b1"])
    h = jax.nn.silu(h @ params["w2"] + params["b2"])
    return h @ params["w3"] + params["b3"]


def _rational_quadratic_spline_forward(
    inputs: Array,
    raw_params: Array,
    num_bins: int,
    tail_bound: float,
) -> tuple[Array, Array]:
    """Monotone RQ spline x->y with identity linear tails and exact log|dy/dx|.

    `raw_params` contains K widths, K heights, and K-1 interior derivatives.
    Endpoint derivatives are fixed to one, making the spline join the identity tails
    continuously and yielding a bijection R -> R.
    """
    inputs = jnp.reshape(inputs, (-1,))
    k = int(num_bins)
    bound = float(tail_bound)

    raw_w = raw_params[:, :k]
    raw_h = raw_params[:, k:2 * k]
    raw_d = raw_params[:, 2 * k:]

    available_width = 2.0 * bound - _NSF_MIN_BIN_WIDTH * k
    available_height = 2.0 * bound - _NSF_MIN_BIN_HEIGHT * k
    widths = _NSF_MIN_BIN_WIDTH + available_width * jax.nn.softmax(raw_w, axis=-1)
    heights = _NSF_MIN_BIN_HEIGHT + available_height * jax.nn.softmax(raw_h, axis=-1)

    cumwidths = -bound + jnp.cumsum(widths, axis=-1)
    cumheights = -bound + jnp.cumsum(heights, axis=-1)
    cumwidths = jnp.concatenate([
        -bound * jnp.ones((len(inputs), 1), dtype=inputs.dtype),
        cumwidths,
    ], axis=-1)
    cumheights = jnp.concatenate([
        -bound * jnp.ones((len(inputs), 1), dtype=inputs.dtype),
        cumheights,
    ], axis=-1)
    # Avoid tiny accumulated round-off at the final boundary.
    cumwidths = cumwidths.at[:, -1].set(bound)
    cumheights = cumheights.at[:, -1].set(bound)

    interior_derivatives = _NSF_MIN_DERIVATIVE + jax.nn.softplus(raw_d)
    derivatives = jnp.concatenate([
        jnp.ones((len(inputs), 1), dtype=inputs.dtype),
        interior_derivatives,
        jnp.ones((len(inputs), 1), dtype=inputs.dtype),
    ], axis=-1)

    inside = (inputs >= -bound) & (inputs <= bound)
    # Bin index in {0,...,K-1}; clipping protects the exact right endpoint.
    bin_idx = jnp.sum(inputs[:, None] >= cumwidths[:, 1:-1], axis=-1)
    bin_idx = jnp.clip(bin_idx, 0, k - 1).astype(jnp.int32)

    def gather(a: Array, idx: Array) -> Array:
        return jnp.take_along_axis(a, idx[:, None], axis=1)[:, 0]

    x0 = gather(cumwidths[:, :-1], bin_idx)
    y0 = gather(cumheights[:, :-1], bin_idx)
    w = gather(widths, bin_idx)
    h = gather(heights, bin_idx)
    d0 = gather(derivatives[:, :-1], bin_idx)
    d1 = gather(derivatives[:, 1:], bin_idx)
    delta = h / w

    theta = jnp.clip((inputs - x0) / w, 0.0, 1.0)
    one_minus_theta = 1.0 - theta
    theta_prod = theta * one_minus_theta
    common = d0 + d1 - 2.0 * delta
    denominator = delta + common * theta_prod
    numerator = h * (delta * theta**2 + d0 * theta_prod)
    outputs_inside = y0 + numerator / denominator

    derivative_numerator = delta**2 * (
        d1 * theta**2
        + 2.0 * delta * theta_prod
        + d0 * one_minus_theta**2
    )
    logabsdet_inside = jnp.log(derivative_numerator) - 2.0 * jnp.log(denominator)

    outputs = jnp.where(inside, outputs_inside, inputs)
    logabsdet = jnp.where(inside, logabsdet_inside, 0.0)
    return outputs, logabsdet


def _nsf_forward_standardized(
    params: tuple[dict[str, Array], ...],
    standardized_theta: Array,
    num_bins: int,
    tail_bound: float,
) -> tuple[Array, Array]:
    """Data -> base-space transform and exact per-row log-Jacobian determinant."""
    z = jnp.asarray(standardized_theta)
    total_logdet = jnp.zeros((z.shape[0],), dtype=z.dtype)
    for layer_idx, layer_params in enumerate(params):
        target_dim = layer_idx % 2
        context_dim = 1 - target_dim
        raw = _nsf_conditioner(layer_params, z[:, context_dim])
        transformed, logdet = _rational_quadratic_spline_forward(
            z[:, target_dim], raw, num_bins, tail_bound
        )
        z = z.at[:, target_dim].set(transformed)
        total_logdet = total_logdet + logdet
    return z, total_logdet


def _nsf_standardized_log_prob(
    params: tuple[dict[str, Array], ...],
    standardized_theta: Array,
    num_bins: int,
    tail_bound: float,
) -> Array:
    z, logdet = _nsf_forward_standardized(params, standardized_theta, num_bins, tail_bound)
    base_log_prob = -0.5 * jnp.sum(z**2 + _LOG_2PI, axis=-1)
    return base_log_prob + logdet


def fit_neural_spline_flow_density(
    samples: np.ndarray,
    cfg: Config = CFG,
    seed: int | None = None,
) -> dict[str, Any]:
    """Fit a 2-D neural spline flow by held-out maximum likelihood.

    This is deliberately post-hoc.  Neither X_OBS's analytic likelihood nor EXACT_DENSITY
    appears anywhere in the fitting procedure.  Validation is performed only on held-out
    Bayes-Transport particles.
    """
    samples = np.asarray(samples, dtype=np.float32)
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if len(samples) < 64:
        raise ValueError("Need at least 64 finite posterior samples to fit the neural spline flow.")

    seed = int(cfg.seed + 91_001 if seed is None else seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    n_val = max(32, int(round(cfg.nsf_validation_fraction * len(samples))))
    n_val = min(n_val, len(samples) // 3)
    val_samples = samples[order[:n_val]]
    train_samples = samples[order[n_val:]]

    center = np.mean(train_samples, axis=0).astype(np.float32)
    scale = np.std(train_samples, axis=0, ddof=1).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-3))
    train_z = ((train_samples - center) / scale).astype(np.float32)
    val_z = ((val_samples - center) / scale).astype(np.float32)

    k = int(cfg.nsf_bins)
    output_dim = 3 * k - 1
    derivative_start = 2 * k
    derivative_bias = _inverse_softplus_np(1.0 - _NSF_MIN_DERIVATIVE)
    keys = jax.random.split(jax.random.key(seed), int(cfg.nsf_layers))
    params = tuple(
        _init_nsf_conditioner(
            keys[i],
            int(cfg.nsf_hidden_dim),
            output_dim,
            derivative_start,
            derivative_bias,
        )
        for i in range(int(cfg.nsf_layers))
    )

    optimizer = optax.adamw(cfg.nsf_learning_rate, weight_decay=cfg.nsf_weight_decay)
    opt_state = optimizer.init(params)

    def nll_fn(p, batch):
        return -jnp.mean(_nsf_standardized_log_prob(
            p, batch, int(cfg.nsf_bins), float(cfg.nsf_tail_bound)
        ))

    nll_and_grad = jax.jit(jax.value_and_grad(nll_fn))
    nll_eval = jax.jit(nll_fn)

    @jax.jit
    def update_step(p, state, batch):
        loss, grads = nll_and_grad(p, batch)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, loss

    batch_size = max(32, min(int(cfg.nsf_batch_size), len(train_z)))
    best_params = params
    best_val = float("inf")
    best_epoch = 0
    stale_checks = 0
    train_history: list[tuple[int, float, float]] = []

    for epoch in range(1, int(cfg.nsf_max_epochs) + 1):
        perm = rng.permutation(len(train_z))
        epoch_losses = []
        for start in range(0, len(train_z), batch_size):
            batch_ids = perm[start:start + batch_size]
            batch = jnp.asarray(train_z[batch_ids])
            params, opt_state, batch_loss = update_step(params, opt_state, batch)
            epoch_losses.append(float(jax.device_get(batch_loss)))

        if epoch == 1 or epoch % int(cfg.nsf_validation_every) == 0:
            train_nll = float(np.mean(epoch_losses))
            val_nll = float(jax.device_get(nll_eval(params, jnp.asarray(val_z))))
            if not np.isfinite(train_nll) or not np.isfinite(val_nll):
                raise FloatingPointError(
                    f"Non-finite NSF objective at epoch {epoch}: train={train_nll}, val={val_nll}."
                )
            train_history.append((epoch, train_nll, val_nll))

            if val_nll < best_val - 1e-4:
                best_val = val_nll
                best_epoch = epoch
                best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
                stale_checks = 0
            else:
                stale_checks += 1

            if epoch == 1 or epoch % 100 == 0:
                print(
                    f"NSF epoch {epoch:4d} | train NLL {train_nll:.5f} | "
                    f"validation NLL {val_nll:.5f} | best {best_val:.5f}"
                )

            if stale_checks >= int(cfg.nsf_patience_checks):
                print(f"NSF early stopping at epoch {epoch}; best epoch={best_epoch}.")
                break

    return {
        "params": best_params,
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
        "num_bins": int(cfg.nsf_bins),
        "tail_bound": float(cfg.nsf_tail_bound),
        "best_validation_nll_standardized": float(best_val),
        "best_epoch": int(best_epoch),
        "num_train": int(len(train_z)),
        "num_validation": int(len(val_z)),
        "history": train_history,
    }


def neural_spline_flow_log_density_np(
    estimator: dict[str, Any],
    points: np.ndarray,
    chunk_size: int = 65_536,
) -> np.ndarray:
    """Evaluate the fitted NSF log density in theta coordinates."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    center = np.asarray(estimator["center"], dtype=np.float32)
    scale = np.asarray(estimator["scale"], dtype=np.float32)
    log_standardization_jacobian = -float(np.sum(np.log(scale.astype(np.float64))))

    @jax.jit
    def eval_chunk(z):
        return _nsf_standardized_log_prob(
            estimator["params"],
            z,
            int(estimator["num_bins"]),
            float(estimator["tail_bound"]),
        )

    out = np.empty((len(points),), dtype=np.float64)
    for start in range(0, len(points), int(chunk_size)):
        end = min(start + int(chunk_size), len(points))
        z = (points[start:end] - center) / scale
        logp_std = np.asarray(jax.device_get(eval_chunk(jnp.asarray(z))), dtype=np.float64)
        out[start:end] = logp_std + log_standardization_jacobian
    return out


def neural_spline_flow_density_on_grid(
    estimator: dict[str, Any],
    theta1_grid: np.ndarray,
    theta2_grid: np.ndarray,
) -> np.ndarray:
    g1, g2 = np.meshgrid(theta1_grid, theta2_grid, indexing="xy")
    points = np.column_stack([g1.ravel(), g2.ravel()])
    log_density = neural_spline_flow_log_density_np(estimator, points)
    density = np.exp(np.clip(log_density, -745.0, 700.0))
    return density.reshape(g1.shape)


def collect_bt_density_samples(
    model: ConditionalParticleTransport,
    first_cloud: np.ndarray,
    x: np.ndarray = X_OBS,
    cfg: Config = CFG,
) -> np.ndarray:
    """Aggregate complete evaluation clouds without changing the trained particle-set size."""
    clouds = [np.asarray(first_cloud, dtype=np.float32)]
    rng = np.random.default_rng(cfg.seed + 90_013)
    n_clouds = max(1, int(cfg.density_estimation_clouds))
    for _ in range(1, n_clouds):
        prior = sample_exact_prior_np(rng, cfg.eval_particles)
        clouds.append(evaluate_bt(model, prior, x))
    return np.concatenate(clouds, axis=0).astype(np.float32)


def exact_density_on_grid_axes(
    theta1_grid: np.ndarray,
    theta2_grid: np.ndarray,
    x: np.ndarray = X_OBS,
    cfg: Config = CFG,
) -> np.ndarray:
    """Exact diagnostic posterior evaluated and normalized on an arbitrary rectangular grid."""
    g1, g2 = np.meshgrid(theta1_grid, theta2_grid, indexing="xy")
    theta = np.stack([g1, g2], axis=-1)
    density = two_moons_likelihood_density_np(x, theta, cfg)
    d1 = float(theta1_grid[1] - theta1_grid[0])
    d2 = float(theta2_grid[1] - theta2_grid[0])
    z = float(np.sum(density) * d1 * d2)
    return density / max(z, 1e-300)


def density_grid_diagnostics(
    exact_density: np.ndarray,
    estimated_density: np.ndarray,
    theta1_grid: np.ndarray,
    theta2_grid: np.ndarray,
) -> dict[str, float]:
    """Grid diagnostics that respect the exact posterior's zero density outside [-1,1]^2."""
    exact = np.asarray(exact_density, dtype=np.float64)
    estimated = np.asarray(estimated_density, dtype=np.float64)
    d1 = float(theta1_grid[1] - theta1_grid[0])
    d2 = float(theta2_grid[1] - theta2_grid[0])
    area = d1 * d2

    exact_mass = float(np.sum(exact) * area)
    estimated_square_mass = float(np.sum(estimated) * area)
    outside_mass = max(0.0, 1.0 - estimated_square_mass)
    overlap = float(np.sum(np.sqrt(np.maximum(exact, 0.0) * np.maximum(estimated, 0.0))) * area)
    hellinger2 = max(0.0, 1.0 - overlap / math.sqrt(max(exact_mass, 1e-15)))
    l1_inside = float(np.sum(np.abs(exact - estimated)) * area)
    total_variation = 0.5 * (l1_inside + outside_mass)

    e = exact.reshape(-1)
    q = estimated.reshape(-1)
    if np.std(e) > 0 and np.std(q) > 0:
        corr = float(np.corrcoef(e, q)[0, 1])
    else:
        corr = float("nan")

    return {
        "mass_inside_exact_prior_square": estimated_square_mass,
        "mass_outside_exact_prior_square": outside_mass,
        "hellinger_squared": hellinger2,
        "total_variation_approx": total_variation,
        "grid_density_correlation": corr,
    }


def plot_v2_density_comparison(
    bt_density_samples: np.ndarray,
    nsf_estimator: dict[str, Any],
    cfg: Config = CFG,
) -> tuple[dict[str, float], dict[str, float]]:
    """Heatmap comparison: exact posterior | crude KDE | neural spline flow.

    Every heatmap is normalized by its own maximum only for display, so the viewer compares
    geometry rather than absolute peak height.  Exact credible-density contours are overlaid on
    both estimates to make geometric agreement/mismatch immediately visible.
    """
    axis = np.linspace(cfg.prior_low, cfg.prior_high, cfg.density_grid_size, dtype=np.float64)
    exact = exact_density_on_grid_axes(axis, axis, X_OBS, cfg)
    kde = kde_density_on_grid(bt_density_samples, axis, axis)
    nsf = neural_spline_flow_density_on_grid(nsf_estimator, axis, axis)
    levels = credible_density_levels(exact)

    kde_metrics = density_grid_diagnostics(exact, kde, axis, axis)
    nsf_metrics = density_grid_diagnostics(exact, nsf, axis, axis)

    # Persist the common grid and raw (not display-normalized) densities for reproducible replotting.
    np.save(OUT / "v2_density_axis.npy", axis)
    np.save(OUT / "v2_exact_density_grid.npy", exact)
    np.save(OUT / "v2_kde_density_grid.npy", kde)
    np.save(OUT / "v2_nsf_density_grid.npy", nsf)

    extent = [cfg.prior_low, cfg.prior_high, cfg.prior_low, cfg.prior_high]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.0), sharex=True, sharey=True)

    panels = [
        (exact, "Ground-truth posterior"),
        (kde, "Crude Gaussian KDE (Scott bandwidth)"),
        (nsf, "Neural spline-flow density (v2)"),
    ]
    for i, (ax, (density, title)) in enumerate(zip(axes, panels)):
        shown = density / max(float(np.max(density)), 1e-12)
        im = ax.imshow(
            shown,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            interpolation="nearest",
        )
        if i > 0:
            ax.contour(
                axis,
                axis,
                exact,
                levels=levels,
                colors="white",
                linewidths=1.15,
                alpha=0.95,
            )
        ax.set_title(title)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.set_xlim(cfg.prior_low, cfg.prior_high)
        ax.set_ylim(cfg.prior_low, cfg.prior_high)
        ax.grid(False)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
    cbar.set_label("relative density (panel max = 1)")
    fig.suptitle(
        rf"Two moons at $x_o=(0,0)$ — post-hoc density estimation from {len(bt_density_samples):,} BT particles",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.11, top=0.86, wspace=0.18)
    fig.savefig(OUT / "32_v2_density_heatmaps_kde_vs_nsf.png", dpi=220, bbox_inches="tight")
    plt.show()

    # A second direct-overlay panel: if the NSF follows the exact posterior, the white exact
    # credible contours should sit naturally on the high-density bands of the learned heatmap.
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    shown = nsf / max(float(np.max(nsf)), 1e-12)
    im = ax.imshow(
        shown,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.contour(axis, axis, exact, levels=levels, colors="white", linewidths=1.7)
    # Thin particle overlay keeps the heatmap readable while showing where the empirical mass came from.
    show_n = min(1800, len(bt_density_samples))
    ids = np.linspace(0, len(bt_density_samples) - 1, show_n, dtype=int)
    ax.scatter(
        bt_density_samples[ids, 0],
        bt_density_samples[ids, 1],
        s=3,
        alpha=0.10,
        c="black",
        linewidths=0,
    )
    ax.set_xlim(cfg.prior_low, cfg.prior_high)
    ax.set_ylim(cfg.prior_low, cfg.prior_high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title("v2 NSF heatmap + exact 50/80/95% density contours")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="relative NSF density")
    fig.tight_layout()
    fig.savefig(OUT / "32_v2_nsf_exact_contour_overlay.png", dpi=220, bbox_inches="tight")
    plt.show()

    return kde_metrics, nsf_metrics


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


# Ground truth is diagnostic-only. The Bayes Transport training objective never calls the density below.
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


#%% 6) TRAIN BAYES TRANSPORT — run this cell when ready
# No likelihood density and no ground-truth posterior density is used in this cell.
# Every row of every optimization batch is a fresh simulator-supervised problem:
#     theta*_b ~ U([-1,1]^2),  x_b ~ simulator(theta*_b).

train_rng = np.random.default_rng(CFG.seed + 10_001)
mode_rng = np.random.default_rng(CFG.seed + 20_003)
particle_count_rng = np.random.default_rng(CFG.seed + 25_019)
dropout_key = jax.random.key(CFG.seed + 30_007)
replay_buffer = HistoricalPosteriorBuffer(
    CFG.historical_output_buffer_capacity,
    CFG.max_training_particles,
)

history = {name: [] for name in (
    "step",
    "simulations_seen",
    "training_particles",
    "energy_score",
    "attraction",
    "repulsion",
    "mean_error",
    "covariance_trace",
    "outside_prior_fraction",
    "grad_norm",
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
    # One true set size per minibatch keeps the input unpadded while limiting JAX recompilations to
    # the automatically-derived TRAINING_PARTICLE_COUNTS.  With the flag off this is always the max.
    training_particles = int(particle_count_rng.choice(TRAINING_PARTICLE_COUNTS))

    # Fresh simulator-supervised minibatch.
    theta_target = sample_exact_prior_np(train_rng, CFG.batch_size)
    x_batch = simulate_two_moons_batch_np(train_rng, theta_target)

    # Mutually-exclusive prior source per row.
    prior_theta, prior_info = make_training_prior_batch_np(
        train_rng,
        mode_rng,
        theta_target,
        x_batch,
        replay_buffer,
        training_particles,
        CFG,
    )

    dropout_key, step_key = jax.random.split(dropout_key)
    model, opt_state, loss, metrics, posterior_batch, grad_norm = train_step(
        model,
        opt_state,
        jnp.asarray(prior_theta),
        jnp.asarray(x_batch),
        jnp.asarray(theta_target),
        step_key,
    )

    posterior_batch_np = np.asarray(jax.device_get(posterior_batch), dtype=np.float32)
    replay_buffer.add_batch(x_batch, posterior_batch_np)

    host = jax.device_get(metrics)
    interp_tau = prior_info["interpolation_tau"]
    replay_dist = prior_info["replay_distance"]
    positive_tau = interp_tau[prior_info["interpolation_used"] > 0]
    finite_replay_dist = replay_dist[np.isfinite(replay_dist)]

    scalar_values = {
        "step": float(step),
        "simulations_seen": float(step * CFG.batch_size),
        "training_particles": float(training_particles),
        "energy_score": float(host["energy_score"]),
        "attraction": float(host["attraction"]),
        "repulsion": float(host["repulsion"]),
        "mean_error": float(host["mean_error"]),
        "covariance_trace": float(host["covariance_trace"]),
        "outside_prior_fraction": float(host["outside_prior_fraction"]),
        "grad_norm": float(jax.device_get(grad_norm)),
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
    # Save at the first minibatch that reaches each requested simulation budget.
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
            f"M {training_particles:4d}/{CFG.max_training_particles:<4d} | "
            f"ES {scalar_values['energy_score']:.5f} | "
            f"mean-err {scalar_values['mean_error']:.4f} | "
            f"grad {scalar_values['grad_norm']:.3e} | "
            f"prior fractions i/b/e="
            f"{scalar_values['interpolation_fraction']:.2f}/"
            f"{scalar_values['buffer_fraction']:.2f}/"
            f"{scalar_values['exact_prior_fraction']:.2f}"
        )

save_model(OUT / "bayes_transport_two_moons.eqx", model, CFG)

with (OUT / "training_history.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(history.keys()))
    writer.writeheader()
    for i in range(len(history["step"])):
        writer.writerow({k: history[k][i] for k in history})

# End-of-training diagnostics requested in the previous workflow.
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


#%% 7) FINAL EVALUATION on x_o=(0,0): exact prior -> Bayes Transport posterior
# If you skipped the training cell and want to load a checkpoint, first run:
# model = load_model(OUT / "bayes_transport_two_moons.eqx", CFG)

if "EVAL_PRIOR_PARTICLES" not in globals():
    _eval_prior_rng = np.random.default_rng(CFG.seed + 60_011)
    EVAL_PRIOR_PARTICLES = sample_exact_prior_np(_eval_prior_rng, CFG.eval_particles)

BT_POSTERIOR = evaluate_bt(model, EVAL_PRIOR_PARTICLES, X_OBS)

# v2 density-estimation sample set: preserve the trained set size (CFG.eval_particles) and
# aggregate independent complete clouds.  No simulator calls and no exact-posterior information
# are used here; these are simply additional transports of fresh exact-prior particles at x_o.
BT_DENSITY_SAMPLES = collect_bt_density_samples(model, BT_POSTERIOR, X_OBS, CFG)
print(f"\nv2 density visualization uses {len(BT_DENSITY_SAMPLES):,} BT posterior particles "
      f"from {CFG.density_estimation_clouds} complete evaluation clouds.")

print("Fitting post-hoc neural spline-flow density estimator (BT particles only)...")
NSF_DENSITY_ESTIMATOR = fit_neural_spline_flow_density(BT_DENSITY_SAMPLES, CFG)
print(
    "NSF fit complete | "
    f"best epoch={NSF_DENSITY_ESTIMATOR['best_epoch']} | "
    f"held-out standardized NLL={NSF_DENSITY_ESTIMATOR['best_validation_nll_standardized']:.6f}"
)
with (OUT / "v2_nsf_training_history.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_nll_standardized", "validation_nll_standardized"])
    writer.writerows(NSF_DENSITY_ESTIMATOR["history"])

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

# v2 heatmaps: same BT sample set for the crude KDE and the NSF, with exact contours overlaid.
KDE_DENSITY_METRICS, NSF_DENSITY_METRICS = plot_v2_density_comparison(
    BT_DENSITY_SAMPLES,
    NSF_DENSITY_ESTIMATOR,
    CFG,
)

print("\nv2 density-grid diagnostics (lower Hellinger/TV is better; higher correlation is better):")
for method_name, method_metrics in (("crude KDE", KDE_DENSITY_METRICS), ("neural spline flow", NSF_DENSITY_METRICS)):
    print(f"  {method_name}:")
    for k, v in method_metrics.items():
        print(f"    {k}: {v:.8f}")

with (OUT / "v2_density_metrics.csv").open("w", newline="") as f:
    fieldnames = ["method"] + list(KDE_DENSITY_METRICS.keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"method": "crude_kde_scott", **KDE_DENSITY_METRICS})
    writer.writerow({"method": "neural_spline_flow", **NSF_DENSITY_METRICS})

plot_marginals(EXACT_SAMPLES, BT_POSTERIOR)
plot_prior_to_posterior_transport(EVAL_PRIOR_PARTICLES, BT_POSTERIOR)
plot_posterior_predictive(BT_POSTERIOR, CFG)

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
np.save(OUT / "bt_posterior_samples.npy", BT_POSTERIOR)
np.save(OUT / "bt_density_estimation_samples_v2.npy", BT_DENSITY_SAMPLES)
np.save(OUT / "ground_truth_posterior_samples.npy", EXACT_SAMPLES)
np.save(OUT / "observed_x.npy", X_OBS)


#%% 8) Optional compact publication-style panel (v4): prior | exact | particles | KDE | NSF
# This cell uses only objects already produced above.

_density_axis = np.linspace(CFG.prior_low, CFG.prior_high, CFG.density_grid_size)
_kde_density_v2 = kde_density_on_grid(BT_DENSITY_SAMPLES, _density_axis, _density_axis)
_nsf_density_v2 = neural_spline_flow_density_on_grid(
    NSF_DENSITY_ESTIMATOR,
    _density_axis,
    _density_axis,
)
_levels = credible_density_levels(EXACT_DENSITY)
_exact_density_v2 = exact_density_on_grid_axes(_density_axis, _density_axis)
_exact_levels_v2 = credible_density_levels(_exact_density_v2)

fig, axes = plt.subplots(1, 5, figsize=(20.5, 4.2))

axes[0].scatter(EVAL_PRIOR_PARTICLES[:, 0], EVAL_PRIOR_PARTICLES[:, 1], s=7, alpha=0.30)
axes[0].set_title("Prior particles")

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

# axes[2].contour(THETA1_GRID, THETA2_GRID, EXACT_DENSITY, levels=_levels, cmap="viridis", linewidths=1.5)
axes[2].scatter(BT_POSTERIOR[:, 0], BT_POSTERIOR[:, 1], s=7, alpha=0.30)
axes[2].set_title("Posterior particles")

axes[3].imshow(
    _kde_density_v2 / max(float(np.max(_kde_density_v2)), 1e-12),
    origin="lower",
    extent=[CFG.prior_low, CFG.prior_high, CFG.prior_low, CFG.prior_high],
    cmap="viridis",
    vmin=0,
    vmax=1,
    aspect="equal",
)
# axes[3].contour(_density_axis, _density_axis, _exact_density_v2,
#                 levels=_exact_levels_v2, colors="white", linewidths=0.9)
axes[3].set_title("KDE")

axes[4].imshow(
    _nsf_density_v2 / max(float(np.max(_nsf_density_v2)), 1e-12),
    origin="lower",
    extent=[CFG.prior_low, CFG.prior_high, CFG.prior_low, CFG.prior_high],
    cmap="viridis",
    vmin=0,
    vmax=1,
    aspect="equal",
)
# axes[4].contour(_density_axis, _density_axis, _exact_density_v2,
#                 levels=_exact_levels_v2, colors="white", linewidths=0.9)
axes[4].set_title("Neural spline flow")

for ax in axes:
    ax.set_xlim(CFG.prior_low, CFG.prior_high)
    ax.set_ylim(CFG.prior_low, CFG.prior_high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.grid(False)

fig.suptitle(r"Two moons at $x_o=(0,0)$ — PSPT", fontsize=15)
fig.tight_layout()
fig.savefig(OUT / "50_compact_publication_panel_v4.png", dpi=220, bbox_inches="tight")
plt.show()
