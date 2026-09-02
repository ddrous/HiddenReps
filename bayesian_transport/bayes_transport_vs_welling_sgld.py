#%% 0) Imports, configuration, and experiment constants
"""Notebook-style Bayes Transport vs SGLD on Welling & Teh (2011), Section 5.1.

Run this file one #%% cell at a time in VS Code / Spyder / Jupyter-compatible editors.
There is intentionally no main() function.

Bayes Transport training is simulator-supervised:
    theta* ~ p(theta)
    x_1:O ~ p(x | theta*)
    prior particles ~ p(theta)   [plus optional training-only prior augmentation]
    posterior particles = T_phi(prior particles, x_1:o)
    loss = multivariate empirical energy score against theta*

At evaluation, Bayes Transport and SGLD see the same fixed 100 observations and
Bayes Transport starts from the exact same prior used by SGLD.
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

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "DejaVu Sans",
    "axes.titlepad": 8.0,
    "axes.labelpad": 6.0,
})


## JAX debug nan
# jax.config.update("jax_debug_nans", True)

Array = jax.Array


@dataclass
class Config:
    # Reproducibility / outputs
    seed: int = 2028
    eval_seed: int = 2030
    output_dir: str = "plots/bayes_transport_vs_sgld"

    # Exact Section 5.1 model
    prior_var_theta1: float = 10.0
    prior_var_theta2: float = 1.0
    likelihood_var: float = 2.0
    eval_true_theta1: float = 0.0
    eval_true_theta2: float = 1.0
    eval_observations: int = 100

    # IMPORTANT: here "batch_size" means the number of x observations in one
    # freshly simulated training inference problem. One theta* generates all 128 x's.
    batch_size: int = 128

    # Particle transport
    num_particles: int = 16*2
    eval_particles: int = 1024//2
    hidden_dim: int = 64*4
    heads: int = 4
    mlp_ratio: int = 4
    posterior_depth: int = 5
    posterior_conditioning: str = "adaln"  # {"cross_attention", "adaln"}
    max_normalized_displacement: float = 6.0
    attention_dropout_rate: float = 0.0

    # Causal likelihood Transformer over x_1, ..., x_O
    likelihood_hidden_dim: int = 64
    likelihood_heads: int = 4
    likelihood_mlp_ratio: int = 4
    likelihood_depth: int = 3
    normalize_observations: bool = True

    # Every selected causal prefix is a direct prior -> posterior map.
    # prefix_stride=1 trains on 1,2,...,128 observations from every simulated batch.
    min_observations_per_step: int = 1
    prefix_stride: int = 16

    # Bayes Transport optimisation
    training_steps: int = 10_000
    learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip_norm: float = 5.0
    log_every: int = 1250
    prefix_history_every: int = 250

    # Training prior-source mixture. Evaluation NEVER uses interpolation or replay and always
    # starts from the exact Gaussian Section 5.1 prior used by SGLD. During training the three
    # prior sources are mutually exclusive:
    #   1) interpolate a configurable Gaussian/Uniform base cloud;
    #   2) use a historical posterior cloud from the buffer;
    #   3) use the exact Gaussian evaluation prior.
    # The probability of (3) is the residual
    #     1 - prior_interpolation_probability - historical_output_prior_probability.
    # Therefore the two explicit probabilities below must sum to <= 1.
    interpolation_base_cloud: str = "gaussian"  # {"gaussian", "uniform"}
    prior_interpolation_probability: float = 0.25
    prior_interpolation_tau_min: float = 0.0
    prior_interpolation_tau_max: float = 2.0
    truth_anchor_probability: float = 1.0
    historical_output_prior_probability: float = 0.25
    historical_output_buffer_capacity: int = 2048

    # Exact posterior grid for diagnostics
    grid_theta1_min: float = -2.5
    grid_theta1_max: float = 3.5
    grid_theta2_min: float = -4.0
    grid_theta2_max: float = 4.0
    grid_size: int = 320
    exact_reference_samples: int = 5000

    # SGLD: paper Section 5.1 defaults
    sgld_batch_size: int = 1
    sgld_sweeps: int = 10_000
    sgld_gamma: float = 0.55
    sgld_epsilon_start: float = 1e-2
    sgld_epsilon_end: float = 1e-4
    sgld_burnin_sweeps: int = 1000
    sgld_reference_samples: int = 5000

    # Diagnostic comparison
    sliced_wasserstein_projections: int = 128


CFG = Config()
OUT = Path(CFG.output_dir)
OUT.mkdir(parents=True, exist_ok=True)

if CFG.posterior_conditioning not in {"cross_attention", "adaln"}:
    raise ValueError("posterior_conditioning must be 'cross_attention' or 'adaln'.")
if CFG.hidden_dim % CFG.heads != 0:
    raise ValueError("hidden_dim must be divisible by heads.")
if CFG.likelihood_hidden_dim % CFG.likelihood_heads != 0:
    raise ValueError("likelihood_hidden_dim must be divisible by likelihood_heads.")
if CFG.batch_size < 1 or CFG.eval_observations < 1:
    raise ValueError("Observation counts must be positive.")
if CFG.min_observations_per_step < 1 or CFG.min_observations_per_step > CFG.batch_size:
    raise ValueError("min_observations_per_step must lie in [1, batch_size].")
if CFG.prefix_stride < 1:
    raise ValueError("prefix_stride must be >= 1.")
if CFG.interpolation_base_cloud not in {"gaussian", "uniform"}:
    raise ValueError("interpolation_base_cloud must be 'gaussian' or 'uniform'.")
if not 0.0 <= CFG.prior_interpolation_probability <= 1.0:
    raise ValueError("prior_interpolation_probability must lie in [0, 1].")
if not 0.0 <= CFG.historical_output_prior_probability <= 1.0:
    raise ValueError("historical_output_prior_probability must lie in [0, 1].")
if CFG.prior_interpolation_probability + CFG.historical_output_prior_probability > 1.0 + 1e-12:
    raise ValueError(
        "prior_interpolation_probability + historical_output_prior_probability must be <= 1; "
        "the residual probability is reserved for the exact Gaussian evaluation prior."
    )
if not 0.0 <= CFG.truth_anchor_probability <= 1.0:
    raise ValueError("truth_anchor_probability must lie in [0, 1].")
if not 0.0 <= CFG.prior_interpolation_tau_min <= CFG.prior_interpolation_tau_max:
    raise ValueError("prior_interpolation_tau_min/max must satisfy 0 <= min <= max.")

TRAIN_EXACT_PRIOR_PROBABILITY = (
    1.0
    - CFG.prior_interpolation_probability
    - CFG.historical_output_prior_probability
)

PRIOR_MEAN = np.asarray([0.0, 0.0], dtype=np.float32)
PRIOR_STD = np.asarray(
    [math.sqrt(CFG.prior_var_theta1), math.sqrt(CFG.prior_var_theta2)],
    dtype=np.float32,
)
# Prior-predictive standard deviation of x under the Section 5.1 model:
# Var(theta1) + E[B^2] Var(theta2) + Var(noise), B~Bernoulli(1/2).
OBS_CENTER = 0.0
OBS_SCALE = math.sqrt(
    CFG.prior_var_theta1 + 0.5 * CFG.prior_var_theta2 + CFG.likelihood_var
)

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
    f"exact-evaluation-Gaussian={TRAIN_EXACT_PRIOR_PROBABILITY:.3f}"
)
print(f"Interpolation base cloud: {CFG.interpolation_base_cloud}")


#%% 1) Section 5.1 simulator, fixed evaluation data, and FIRST plot: observed x

def sample_exact_prior_np(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw n iid theta=(theta1,theta2) particles from the exact paper prior."""
    z = rng.normal(size=(int(n), 2)).astype(np.float32)
    return PRIOR_MEAN[None, :] + PRIOR_STD[None, :] * z


def simulate_likelihood_np(
    rng: np.random.Generator,
    theta: np.ndarray,
    n: int,
) -> np.ndarray:
    """Draw iid x from 0.5 N(theta1,sigma_x^2)+0.5 N(theta1+theta2,sigma_x^2)."""
    theta = np.asarray(theta, dtype=np.float32).reshape(2)
    component = rng.integers(0, 2, size=int(n)).astype(np.float32)
    mean = theta[0] + component * theta[1]
    noise = rng.normal(0.0, math.sqrt(CFG.likelihood_var), size=int(n))
    return (mean + noise).astype(np.float32)


def normal_pdf_np(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    return np.exp(-0.5 * (x - mean) ** 2 / var) / math.sqrt(2.0 * math.pi * var)


def mixture_density_np(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64).reshape(2)
    return 0.5 * normal_pdf_np(x, theta[0], CFG.likelihood_var) + 0.5 * normal_pdf_np(
        x, theta[0] + theta[1], CFG.likelihood_var
    )


# This is the one fixed N=100 dataset seen by BOTH Bayes Transport and SGLD at evaluation.
_eval_rng = np.random.default_rng(CFG.eval_seed)
THETA_EVAL_TRUE = np.asarray(
    [CFG.eval_true_theta1, CFG.eval_true_theta2], dtype=np.float32
)
X_EVAL = simulate_likelihood_np(_eval_rng, THETA_EVAL_TRUE, CFG.eval_observations)

# Plot the data FIRST, before defining or training the model.
_x_grid = np.linspace(float(X_EVAL.min()) - 2.0, float(X_EVAL.max()) + 2.0, 600)
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.hist(X_EVAL, bins=18, density=True, alpha=0.45, label="Fixed evaluation data")
ax.plot(
    _x_grid,
    mixture_density_np(_x_grid, THETA_EVAL_TRUE),
    linewidth=2.0,
    label=r"Generating density ($\theta=(0,1)$; reference only)",
)
ax.scatter(X_EVAL, np.zeros_like(X_EVAL), marker="|", s=140, alpha=0.55, label="Observed x")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.set_title("Section 5.1 fixed evaluation dataset: the same 100 x values go to BT and SGLD")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "00_evaluation_data_x.png", dpi=180, bbox_inches="tight")
plt.show()

print("Evaluation x summary:")
print(f"  n={len(X_EVAL)} mean={X_EVAL.mean():.4f} std={X_EVAL.std(ddof=1):.4f}")
print(f"  min={X_EVAL.min():.4f} max={X_EVAL.max():.4f}")


#%% 2) Training-only prior augmentation and historical posterior buffer

def sample_interpolation_base_cloud_np(
    rng: np.random.Generator,
    n: int,
    cfg: Config = CFG,
) -> np.ndarray:
    """Draw the base cloud used ONLY by the training interpolation branch.

    gaussian:
        The exact Section 5.1 Gaussian prior, identical in law to the evaluation/SGLD prior.
    uniform:
        A zero-mean independent Uniform cloud matched to the SAME marginal variances as that
        Gaussian prior. For Uniform(-a,a), Var=a^2/3, so a=sqrt(3)*prior_std.

    Evaluation never calls this function: evaluation always uses sample_exact_prior_np().
    """
    n = int(n)
    if cfg.interpolation_base_cloud == "gaussian":
        return sample_exact_prior_np(rng, n)
    if cfg.interpolation_base_cloud == "uniform":
        half_width = np.sqrt(np.float32(3.0)) * PRIOR_STD
        return rng.uniform(
            low=PRIOR_MEAN - half_width,
            high=PRIOR_MEAN + half_width,
            size=(n, 2),
        ).astype(np.float32)
    raise ValueError(f"Unknown interpolation_base_cloud={cfg.interpolation_base_cloud!r}.")


def sample_interpolated_training_prior_np(
    rng: np.random.Generator,
    theta_target: np.ndarray,
    cfg: Config = CFG,
) -> tuple[np.ndarray, float]:
    """Create one interpolated training cloud from the configured base family.

    C_tau = (1-tau) Z + tau * anchor, with one shared tau for the entire 2-D cloud.
    This function is called only after the mutually-exclusive prior-source router has selected
    the interpolation branch.
    """
    z = sample_interpolation_base_cloud_np(rng, cfg.num_particles, cfg)

    if rng.random() < cfg.truth_anchor_probability:
        anchor = np.asarray(theta_target, dtype=np.float32)
    else:
        anchor = sample_interpolation_base_cloud_np(rng, 1, cfg)[0]

    tau = float(
        rng.uniform(cfg.prior_interpolation_tau_min, cfg.prior_interpolation_tau_max)
    )
    cloud = (1.0 - tau) * z + tau * anchor[None, :]
    return cloud.astype(np.float32), tau


def observation_signature(x: np.ndarray) -> np.ndarray:
    """Observed-data-only key for nearest historical posterior replay."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mean = float(np.mean(x))
    std = float(np.std(x) + 1e-6)
    centered = (x - mean) / std
    skew = float(np.mean(centered**3))
    kurt = float(np.mean(centered**4) - 3.0)
    return np.asarray([mean, math.log(std), skew, kurt], dtype=np.float32)


class HistoricalPosteriorBuffer:
    """Training-only nearest-posterior replay keyed by observed x summaries.

    The buffer stores only an observed-data signature and a detached posterior particle cloud.
    When replay is selected, only the current prior cloud is replaced. The CURRENT simulated
    x batch and CURRENT theta target remain unchanged.
    """

    def __init__(self, capacity: int, num_particles: int):
        self.capacity = int(capacity)
        self.num_particles = int(num_particles)
        self.signatures = np.empty((self.capacity, 4), dtype=np.float32)
        self.clouds = np.empty((self.capacity, self.num_particles, 2), dtype=np.float32)
        self.size = 0
        self.next_index = 0

    def __len__(self) -> int:
        return int(self.size)

    def add(self, signature: np.ndarray, posterior_cloud: np.ndarray) -> None:
        signature = np.asarray(signature, dtype=np.float32).reshape(4)
        posterior_cloud = np.asarray(posterior_cloud, dtype=np.float32)
        if posterior_cloud.shape != (self.num_particles, 2):
            raise ValueError(
                f"posterior_cloud must have shape {(self.num_particles, 2)}, "
                f"got {posterior_cloud.shape}."
            )
        self.signatures[self.next_index] = signature
        self.clouds[self.next_index] = posterior_cloud
        self.next_index = (self.next_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def nearest(self, signature: np.ndarray) -> tuple[np.ndarray, float]:
        if self.size == 0:
            raise ValueError("HistoricalPosteriorBuffer is empty.")
        signature = np.asarray(signature, dtype=np.float32).reshape(4)
        active_sig = self.signatures[: self.size]
        # Standardize each summary dimension across the buffer so no one statistic dominates.
        scale = np.std(active_sig, axis=0) + 1e-4
        d2 = np.sum(((active_sig - signature[None, :]) / scale[None, :]) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return self.clouds[idx].copy(), float(math.sqrt(float(d2[idx])))


#%% 3) JAX + Equinox model: causal likelihood Transformer and posterior particle Transformer

def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]


class CausalObservationBlock(eqx.Module):
    """Transformer block over x_1:O with a strict causal attention mask."""

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
        length = tokens.shape[0]
        idx = jnp.arange(length)
        causal_mask = idx[:, None] >= idx[None, :]
        h = _layernorm_tokens(self.norm1, tokens)
        tokens = tokens + self.attention(
            h, h, h, mask=causal_mask, key=key, inference=inference
        )
        h = _layernorm_tokens(self.norm2, tokens)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        return tokens + _linear_tokens(self.ff_out, h)


class LikelihoodSequenceEmbedder(eqx.Module):
    """Causally contextualize an arbitrary fixed-length sequence of scalar x observations."""

    input_projection: eqx.nn.Linear
    blocks: tuple[CausalObservationBlock, ...]
    final_norm: eqx.nn.LayerNorm
    x_center: float = eqx.field(static=True)
    x_scale: float = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)

    def __init__(self, cfg: Config, *, key: Array):
        keys = jax.random.split(key, cfg.likelihood_depth + 1)
        self.input_projection = eqx.nn.Linear(1, cfg.likelihood_hidden_dim, key=keys[0])
        self.blocks = tuple(
            CausalObservationBlock(
                cfg.likelihood_hidden_dim,
                cfg.likelihood_heads,
                cfg.likelihood_mlp_ratio * cfg.likelihood_hidden_dim,
                cfg.attention_dropout_rate,
                key=keys[i + 1],
            )
            for i in range(cfg.likelihood_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(cfg.likelihood_hidden_dim)
        self.x_center = float(OBS_CENTER)
        self.x_scale = float(OBS_SCALE)
        self.normalize = bool(cfg.normalize_observations)

    def __call__(self, x_observations: Array, *, key: Array | None = None, inference: bool = False) -> Array:
        x = jnp.reshape(x_observations, (-1,))
        if self.normalize:
            x = (x - self.x_center) / self.x_scale
        tokens = _linear_tokens(self.input_projection, x[:, None])
        block_keys = None if key is None else jax.random.split(key, len(self.blocks))
        for i, block in enumerate(self.blocks):
            block_key = None if block_keys is None else block_keys[i]
            tokens = block(tokens, key=block_key, inference=inference)
        return _layernorm_tokens(self.final_norm, tokens)


class AdaLNParticleBlock(eqx.Module):
    """Particle self-attention conditioned by the causal summary token at prefix o."""

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
        modulation = eqx.tree_at(
            lambda l: l.weight, modulation, jnp.zeros_like(modulation.weight)
        )
        modulation = eqx.tree_at(
            lambda l: l.bias, modulation, jnp.zeros_like(modulation.bias)
        )
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
    """Particle self-attention + cross-attention to the active causal likelihood prefix."""

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
        observation_count: Array,
        *,
        key: Array | None = None,
        inference: bool = False,
    ) -> Array:
        if key is None:
            self_key = cross_key = None
        else:
            self_key, cross_key = jax.random.split(key)

        h = _layernorm_tokens(self.norm_self, particles)
        particles = particles + self.self_attention(
            h, h, h, key=self_key, inference=inference
        )

        q = _layernorm_tokens(self.norm_cross, particles)
        memory = _layernorm_tokens(self.memory_norm, observation_memory)

        # Fixed-shape prefix mask makes the whole prefix family vmappable/JIT-friendly.
        # Shape [num_particles, O], as accepted by Equinox MHA just like its causal [O,O] mask.
        count = jnp.clip(observation_count, 1, observation_memory.shape[0]).astype(jnp.int32)
        active = jnp.arange(observation_memory.shape[0]) < count
        cross_mask = jnp.broadcast_to(active[None, :], (particles.shape[0], observation_memory.shape[0]))
        particles = particles + self.cross_attention(
            q,
            memory,
            memory,
            mask=cross_mask,
            key=cross_key,
            inference=inference,
        )

        h = _layernorm_tokens(self.norm_ff, particles)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        return particles + _linear_tokens(self.ff_out, h)


class ConditionalParticleTransport(eqx.Module):
    """Identity-initialized 2-D prior -> posterior particle transport."""

    likelihood_embedder: LikelihoodSequenceEmbedder
    particle_in: eqx.nn.Linear
    blocks: tuple[Any, ...]
    final_norm: eqx.nn.LayerNorm
    displacement_head: eqx.nn.Linear

    conditioning_type: str = eqx.field(static=True)
    max_displacement: float = eqx.field(static=True)
    prior_mean: tuple[float, float] = eqx.field(static=True)
    prior_std: tuple[float, float] = eqx.field(static=True)
    min_observations: int = eqx.field(static=True)
    prefix_stride: int = eqx.field(static=True)

    def __init__(self, cfg: Config, *, key: Array):
        keys = jax.random.split(key, cfg.posterior_depth + 4)
        self.likelihood_embedder = LikelihoodSequenceEmbedder(cfg, key=keys[0])
        self.particle_in = eqx.nn.Linear(2, cfg.hidden_dim, key=keys[1])
        block_cls = (
            AdaLNParticleBlock
            if cfg.posterior_conditioning == "adaln"
            else CrossAttentionParticleBlock
        )
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
        self.prior_mean = (0.0, 0.0)
        self.prior_std = (
            math.sqrt(cfg.prior_var_theta1),
            math.sqrt(cfg.prior_var_theta2),
        )
        self.min_observations = int(cfg.min_observations_per_step)
        self.prefix_stride = int(cfg.prefix_stride)

    def _whiten(self, theta: Array) -> Array:
        mean = jnp.asarray(self.prior_mean, dtype=theta.dtype)
        std = jnp.asarray(self.prior_std, dtype=theta.dtype)
        return (theta - mean[None, :]) / std[None, :]

    def _unwhiten(self, z: Array) -> Array:
        mean = jnp.asarray(self.prior_mean, dtype=z.dtype)
        std = jnp.asarray(self.prior_std, dtype=z.dtype)
        return mean[None, :] + std[None, :] * z

    def encode_observations(
        self,
        x_observations: Array,
        *,
        key: Array | None = None,
        inference: bool = False,
    ) -> Array:
        return self.likelihood_embedder(x_observations, key=key, inference=inference)

    def transport_with_contexts(
        self,
        prior_theta: Array,
        contexts: Array,
        observation_count: Array,
        *,
        key: Array | None = None,
        inference: bool = False,
    ) -> Array:
        z0 = self._whiten(prior_theta)
        particles = _linear_tokens(self.particle_in, z0)
        block_keys = None if key is None else jax.random.split(key, len(self.blocks))

        if self.conditioning_type == "adaln":
            count = jnp.clip(observation_count, 1, contexts.shape[0]).astype(jnp.int32)
            conditioning = contexts[count - 1]
            for i, block in enumerate(self.blocks):
                block_key = None if block_keys is None else block_keys[i]
                particles = block(
                    particles,
                    conditioning,
                    key=block_key,
                    inference=inference,
                )
        else:
            for i, block in enumerate(self.blocks):
                block_key = None if block_keys is None else block_keys[i]
                particles = block(
                    particles,
                    contexts,
                    observation_count,
                    key=block_key,
                    inference=inference,
                )

        particles = _layernorm_tokens(self.final_norm, particles)
        delta = self.max_displacement * jnp.tanh(
            _linear_tokens(self.displacement_head, particles)
        )
        return self._unwhiten(z0 + delta)

    def predict_prefixes(
        self,
        prior_theta: Array,
        x_observations: Array,
        *,
        key: Array | None = None,
        inference: bool = False,
    ) -> tuple[Array, Array]:
        """Encode x once causally, then VMAP the direct transport over observation prefixes."""
        if key is None:
            context_key = None
            prefix_root_key = None
        else:
            context_key, prefix_root_key = jax.random.split(key)

        contexts = self.encode_observations(
            x_observations, key=context_key, inference=inference
        )
        n_obs = int(x_observations.shape[0])
        prefix_counts = jnp.arange(
            self.min_observations,
            n_obs + 1,
            self.prefix_stride,
            dtype=jnp.int32,
        )
        # Always include the full data length even when stride does not divide it exactly.
        if (n_obs - self.min_observations) % self.prefix_stride != 0:
            prefix_counts = jnp.concatenate(
                [prefix_counts, jnp.asarray([n_obs], dtype=jnp.int32)]
            )

        if prefix_root_key is None:
            posterior = jax.vmap(
                lambda count: self.transport_with_contexts(
                    prior_theta,
                    contexts,
                    count,
                    inference=inference,
                )
            )(prefix_counts)
        else:
            prefix_keys = jax.random.split(prefix_root_key, prefix_counts.shape[0])
            posterior = jax.vmap(
                lambda count, k: self.transport_with_contexts(
                    prior_theta,
                    contexts,
                    count,
                    key=k,
                    inference=inference,
                )
            )(prefix_counts, prefix_keys)
        return posterior, prefix_counts

    def posterior_at_count(
        self,
        prior_theta: Array,
        x_observations: Array,
        observation_count: int | Array,
        *,
        key: Array | None = None,
        inference: bool = True,
    ) -> Array:
        contexts = self.encode_observations(
            x_observations, key=key, inference=inference
        )
        return self.transport_with_contexts(
            prior_theta,
            contexts,
            jnp.asarray(observation_count, dtype=jnp.int32),
            key=None,
            inference=inference,
        )


#%% 4) Proper scoring rule: multivariate empirical energy score + JAX/Optax train step

# Numerical stabilizer only; this is not a training hyperparameter.
# jnp.linalg.norm has an undefined gradient at exactly zero. The pairwise
# energy-score matrix always contains zero diagonal entries (Y_i - Y_i),
# which can therefore produce 0/0 -> NaN during reverse-mode autodiff.
_ENERGY_NORM_EPS = 1e-12


def _stable_l2_norm(x: Array, axis: int = -1) -> Array:
    """L2 norm with a finite derivative at x=0 (float32-safe)."""
    eps = jnp.asarray(_ENERGY_NORM_EPS, dtype=x.dtype)
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis) + eps)


def energy_score_terms(posterior: Array, target_theta: Array) -> tuple[Array, Array, Array]:
    """2-D empirical energy score: E||Y-y|| - 1/2 E||Y-Y'||."""
    attraction = jnp.mean(_stable_l2_norm(posterior - target_theta[None, :], axis=-1))
    pairwise = posterior[:, None, :] - posterior[None, :, :]
    repulsion = jnp.mean(_stable_l2_norm(pairwise, axis=-1))
    return attraction - 0.5 * repulsion, attraction, repulsion


def prefix_metrics(
    posterior_by_prefix: Array,
    target_theta: Array,
    prefix_counts: Array,
) -> dict[str, Array]:
    score, attraction, repulsion = jax.vmap(
        lambda cloud: energy_score_terms(cloud, target_theta)
    )(posterior_by_prefix)
    means = jnp.mean(posterior_by_prefix, axis=1)
    mean_error = _stable_l2_norm(means - target_theta[None, :], axis=-1)
    centered = posterior_by_prefix - means[:, None, :]
    covariance_trace = jnp.mean(jnp.sum(centered**2, axis=-1), axis=1)
    return {
        "loss": jnp.mean(score),
        "energy_score": jnp.mean(score),
        "final_energy_score": score[-1],
        "attraction": jnp.mean(attraction),
        "repulsion": jnp.mean(repulsion),
        "final_attraction": attraction[-1],
        "final_repulsion": repulsion[-1],
        "mean_error": jnp.mean(mean_error),
        "final_mean_error": mean_error[-1],
        "covariance_trace": jnp.mean(covariance_trace),
        "final_covariance_trace": covariance_trace[-1],
        "energy_by_o": score,
        "attraction_by_o": attraction,
        "repulsion_by_o": repulsion,
        "mean_error_by_o": mean_error,
        "covariance_trace_by_o": covariance_trace,
        "prefix_counts": prefix_counts,
    }


def transport_objective(
    model: ConditionalParticleTransport,
    prior_theta: Array,
    x_observations: Array,
    target_theta: Array,
    dropout_key: Array,
):
    posterior_by_prefix, prefix_counts = model.predict_prefixes(
        prior_theta,
        x_observations,
        key=dropout_key,
        inference=False,
    )
    metrics = prefix_metrics(posterior_by_prefix, target_theta, prefix_counts)
    return metrics["loss"], (metrics, posterior_by_prefix[-1])


_loss_and_grad = eqx.filter_value_and_grad(transport_objective, has_aux=True)


def make_train_step(optimizer: optax.GradientTransformation):
    @eqx.filter_jit
    def step(model, opt_state, prior_theta, x_observations, target_theta, dropout_key):
        (loss, (metrics, final_posterior)), grads = _loss_and_grad(
            model,
            prior_theta,
            x_observations,
            target_theta,
            dropout_key,
        )
        params = eqx.filter(model, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return model, opt_state, loss, metrics, final_posterior, grad_norm

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
print("Training observation batch size:", CFG.batch_size)
print("Training particles:", CFG.num_particles)


#%% 5) Plotting, exact posterior, metrics, checkpoint, and SGLD utility functions

def save_model(path: Path, model: ConditionalParticleTransport, cfg: Config = CFG) -> None:
    path = Path(path)
    eqx.tree_serialise_leaves(path, model)
    with path.with_suffix(".json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_model(path: Path, cfg: Config = CFG) -> ConditionalParticleTransport:
    template = ConditionalParticleTransport(cfg, key=jax.random.key(cfg.seed))
    return eqx.tree_deserialise_leaves(Path(path), template)


def rolling_mean(x: np.ndarray, window: int = 100) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return x.copy()
    window = max(1, min(int(window), len(x)))
    kernel = np.ones(window, dtype=np.float64) / window
    y = np.convolve(x, kernel, mode="valid")
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, y])


def plot_training_diagnostics(
    history: dict[str, list[float]],
    prefix_history_steps: list[int],
    prefix_history_counts: np.ndarray | None,
    prefix_energy_history: list[np.ndarray],
    cfg: Config = CFG,
) -> None:
    step = np.asarray(history["step"])
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    ax = axes[0, 0]
    loss = np.asarray(history["energy_score"])
    ax.plot(step, loss, alpha=0.35, label="Energy score")
    ax.plot(step, rolling_mean(loss, 100), linewidth=2, label="100-step mean")
    ax.plot(step, history["final_energy_score"], alpha=0.45, label="Final-prefix ES")
    ax.set_title("Training loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("energy score")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(step, history["grad_norm"])
    ax.set_yscale("log")
    ax.set_title("Gradient norm")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("global norm")

    ax = axes[1, 0]
    ax.plot(step, history["attraction"], label="Attraction")
    ax.plot(step, history["repulsion"], label="Repulsion")
    ax.plot(step, history["final_attraction"], alpha=0.5, label="Final attraction")
    ax.plot(step, history["final_repulsion"], alpha=0.5, label="Final repulsion")
    ax.set_title("Energy-score components")
    ax.set_xlabel("optimizer step")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(step, history["mean_error"], label="Mean error, all prefixes")
    ax.plot(step, history["final_mean_error"], label="Mean error, final prefix")
    ax.set_title(r"Posterior mean error to simulator-known $\theta^*$")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Euclidean error")
    ax.legend()

    ax = axes[2, 0]
    ax.plot(step, history["covariance_trace"], label="All prefixes")
    ax.plot(step, history["final_covariance_trace"], label="Final prefix")
    ax.set_title("Particle-cloud covariance trace")
    ax.set_xlabel("optimizer step")
    ax.legend()

    ax = axes[2, 1]
    ax.plot(step, history["replay_used"], alpha=0.35, label="Buffer prior")
    ax.plot(step, history["interpolation_used"], alpha=0.35, label="Interpolated prior")
    ax.plot(step, history["exact_prior_used"], alpha=0.35, label="Exact eval Gaussian prior")
    ax.plot(step, history["buffer_size"], label="Buffer size")
    ax.set_title("Mutually-exclusive training prior source")
    ax.set_xlabel("optimizer step")
    ax.legend()

    fig.suptitle("Bayes Transport training diagnostics", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUT / "10_bt_training_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.show()

    if prefix_energy_history and prefix_history_counts is not None:
        e = np.stack(prefix_energy_history)
        fig, ax = plt.subplots(figsize=(11, 6))
        # Show the first, middle, and last recorded curves for readable evolution.
        ids = sorted(set([0, len(e) // 2, len(e) - 1]))
        for idx in ids:
            ax.plot(
                prefix_history_counts,
                e[idx],
                label=f"train step {prefix_history_steps[idx]}",
            )
        ax.set_xlabel("number of observations in causal prefix")
        ax.set_ylabel("energy score")
        ax.set_title("Loss per observation-prefix length during training")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "11_bt_loss_by_observation_count.png", dpi=180, bbox_inches="tight")
        plt.show()


def log_joint_grid_np(x: np.ndarray, theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """Unnormalized log posterior on a theta1/theta2 grid."""
    t1, t2 = np.meshgrid(theta1, theta2, indexing="xy")
    logp = -0.5 * t1**2 / CFG.prior_var_theta1 - 0.5 * t2**2 / CFG.prior_var_theta2
    var = CFG.likelihood_var
    log_norm = -0.5 * math.log(2.0 * math.pi * var)
    for xi in np.asarray(x, dtype=np.float64):
        l1 = log_norm - 0.5 * (xi - t1) ** 2 / var
        l2 = log_norm - 0.5 * (xi - (t1 + t2)) ** 2 / var
        logp += np.logaddexp(l1, l2) - math.log(2.0)
    return logp


def exact_posterior_grid(x: np.ndarray, cfg: Config = CFG):
    theta1 = np.linspace(cfg.grid_theta1_min, cfg.grid_theta1_max, cfg.grid_size)
    theta2 = np.linspace(cfg.grid_theta2_min, cfg.grid_theta2_max, cfg.grid_size)
    logp = log_joint_grid_np(x, theta1, theta2)
    logp -= np.max(logp)
    density = np.exp(logp)
    d1 = theta1[1] - theta1[0]
    d2 = theta2[1] - theta2[0]
    density /= np.sum(density) * d1 * d2
    return theta1, theta2, density


def sample_from_grid_posterior(
    rng: np.random.Generator,
    theta1: np.ndarray,
    theta2: np.ndarray,
    density: np.ndarray,
    n: int,
) -> np.ndarray:
    p = density.reshape(-1).astype(np.float64)
    p /= p.sum()
    ids = rng.choice(len(p), size=int(n), replace=True, p=p)
    i2, i1 = np.unravel_index(ids, density.shape)
    samples = np.column_stack([theta1[i1], theta2[i2]])
    # Small cell jitter avoids artificial vertical/horizontal grid bands in scatter diagnostics.
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


def prior_density_grid(theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    t1, t2 = np.meshgrid(theta1, theta2, indexing="xy")
    return normal_pdf_np(t1, 0.0, CFG.prior_var_theta1) * normal_pdf_np(
        t2, 0.0, CFG.prior_var_theta2
    )


def plot_prior_diagnostics(prior_samples: np.ndarray) -> None:
    t1 = np.linspace(-10.0, 10.0, 320)
    t2 = np.linspace(-4.0, 4.0, 320)
    density = prior_density_grid(t1, t2)
    levels = credible_density_levels(density)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    ax.contour(t1, t2, density, levels=levels, linewidths=1.5)
    ax.scatter(prior_samples[:, 0], prior_samples[:, 1], s=10, alpha=0.30)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title("Exact evaluation prior: joint samples + contours")

    axes[1].hist(prior_samples[:, 0], bins=40, density=True, alpha=0.5)
    g1 = np.linspace(-10, 10, 500)
    axes[1].plot(g1, normal_pdf_np(g1, 0.0, CFG.prior_var_theta1), linewidth=2)
    axes[1].set_title(r"Prior marginal $\theta_1 \sim N(0,10)$")
    axes[1].set_xlabel(r"$\theta_1$")

    axes[2].hist(prior_samples[:, 1], bins=40, density=True, alpha=0.5)
    g2 = np.linspace(-4, 4, 500)
    axes[2].plot(g2, normal_pdf_np(g2, 0.0, CFG.prior_var_theta2), linewidth=2)
    axes[2].set_title(r"Prior marginal $\theta_2 \sim N(0,1)$")
    axes[2].set_xlabel(r"$\theta_2$")

    fig.tight_layout()
    fig.savefig(OUT / "20_exact_evaluation_prior.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_samples_on_exact_contours(
    samples: np.ndarray,
    theta1_grid: np.ndarray,
    theta2_grid: np.ndarray,
    density: np.ndarray,
    title: str,
    filename: str,
    true_theta: np.ndarray | None = THETA_EVAL_TRUE,
) -> None:
    levels = credible_density_levels(density)
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.contour(theta1_grid, theta2_grid, density, levels=levels, linewidths=2)
    ax.scatter(samples[:, 0], samples[:, 1], s=12, alpha=0.35, label="Samples")
    if true_theta is not None:
        ax.scatter([true_theta[0]], [true_theta[1]], marker="*", s=180, label=r"Generating $\theta$ (reference)")
    ax.set_xlim(CFG.grid_theta1_min, CFG.grid_theta1_max)
    ax.set_ylim(CFG.grid_theta2_min, CFG.grid_theta2_max)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.show()


def plot_marginal_comparison(
    exact_samples: np.ndarray,
    bt_samples: np.ndarray,
    sgld_samples: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for d, label in enumerate([r"$\theta_1$", r"$\theta_2$"]):
        ax = axes[d]
        ax.hist(exact_samples[:, d], bins=55, density=True, histtype="step", linewidth=2, label="Exact grid posterior")
        ax.hist(bt_samples[:, d], bins=55, density=True, histtype="step", linewidth=2, label="Bayes Transport")
        ax.hist(sgld_samples[:, d], bins=55, density=True, histtype="step", linewidth=2, label="SGLD")
        ax.axvline(THETA_EVAL_TRUE[d], linestyle="--", linewidth=1.5, label="Generating value" if d == 0 else None)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.set_title(f"Posterior marginal {label}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "41_posterior_marginals.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_full_comparison(
    prior_samples: np.ndarray,
    exact_samples: np.ndarray,
    bt_samples: np.ndarray,
    sgld_samples: np.ndarray,
    theta1_grid: np.ndarray,
    theta2_grid: np.ndarray,
    density: np.ndarray,
) -> None:
    levels = credible_density_levels(density)
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), sharex=True, sharey=True)
    panels = [
        (prior_samples, "Exact prior"),
        (exact_samples, "Exact numerical posterior"),
        (bt_samples, "Bayes Transport"),
        (sgld_samples, "SGLD"),
    ]
    for ax, (samples, title) in zip(axes.ravel(), panels):
        if title != "Exact prior":
            ax.contour(theta1_grid, theta2_grid, density, levels=levels, linewidths=1.5)
        ax.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.30)
        ax.scatter([THETA_EVAL_TRUE[0]], [THETA_EVAL_TRUE[1]], marker="*", s=120)
        ax.set_title(title)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.set_xlim(CFG.grid_theta1_min, CFG.grid_theta1_max)
        ax.set_ylim(CFG.grid_theta2_min, CFG.grid_theta2_max)
    fig.suptitle("Section 5.1 posterior comparison: same 100 evaluation observations", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUT / "40_full_posterior_comparison.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_bt_prefix_evolution(
    model: ConditionalParticleTransport,
    prior_particles: np.ndarray,
    x_eval: np.ndarray,
    counts=(1, 2, 4, 8, 16, 32, 64, 100),
) -> None:
    counts = [c for c in counts if c <= len(x_eval)]
    ncols = 4
    nrows = math.ceil(len(counts) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.0 * nrows), squeeze=False)

    for ax, count in zip(axes.ravel(), counts):
        t1, t2, d = exact_posterior_grid(x_eval[:count], CFG)
        bt = np.asarray(
            jax.device_get(
                model.posterior_at_count(
                    jnp.asarray(prior_particles),
                    jnp.asarray(x_eval[:count]),
                    count,
                    inference=True,
                )
            )
        )
        ax.contour(t1, t2, d, levels=credible_density_levels(d), linewidths=1.2)
        ax.scatter(bt[:, 0], bt[:, 1], s=8, alpha=0.30)
        ax.scatter([THETA_EVAL_TRUE[0]], [THETA_EVAL_TRUE[1]], marker="*", s=90)
        ax.set_title(f"o = {count}")
        ax.set_xlim(CFG.grid_theta1_min, CFG.grid_theta1_max)
        ax.set_ylim(CFG.grid_theta2_min, CFG.grid_theta2_max)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")

    for ax in axes.ravel()[len(counts):]:
        ax.axis("off")
    fig.suptitle("Bayes Transport causal posterior evolution; contours are exact p(theta | x_1:o)", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT / "32_bt_prefix_evolution.png", dpi=180, bbox_inches="tight")
    plt.show()


def energy_distance_samples(a: np.ndarray, b: np.ndarray, max_n: int = 2000) -> float:
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
    vals = []
    for u in directions:
        pa = np.sort(a @ u)
        pb = np.sort(b @ u)
        vals.append(np.mean(np.abs(pa - pb)))
    return float(np.mean(vals))


def posterior_summary(samples: np.ndarray) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=np.float64)
    return {
        "mean": np.mean(samples, axis=0),
        "cov": np.cov(samples.T),
        "p_theta2_positive": float(np.mean(samples[:, 1] > 0.0)),
    }


def comparison_metrics(exact: np.ndarray, candidate: np.ndarray, name: str) -> dict[str, float]:
    e = posterior_summary(exact)
    c = posterior_summary(candidate)
    return {
        "method": name,
        "mean_error": float(np.linalg.norm(c["mean"] - e["mean"])),
        "covariance_frobenius_error": float(np.linalg.norm(c["cov"] - e["cov"], ord="fro")),
        "p_theta2_positive": float(c["p_theta2_positive"]),
        "p_theta2_positive_error": float(abs(c["p_theta2_positive"] - e["p_theta2_positive"])),
        "sliced_wasserstein": sliced_wasserstein(exact, candidate, CFG.sliced_wasserstein_projections),
        "energy_distance": energy_distance_samples(exact, candidate),
    }


def log_joint_theta_np(theta: np.ndarray, x: np.ndarray) -> float:
    theta = np.asarray(theta, dtype=np.float64).reshape(2)
    logp = -0.5 * theta[0] ** 2 / CFG.prior_var_theta1 - 0.5 * theta[1] ** 2 / CFG.prior_var_theta2
    var = CFG.likelihood_var
    log_norm = -0.5 * math.log(2.0 * math.pi * var)
    m1 = theta[0]
    m2 = theta[0] + theta[1]
    l1 = log_norm - 0.5 * (np.asarray(x) - m1) ** 2 / var
    l2 = log_norm - 0.5 * (np.asarray(x) - m2) ** 2 / var
    return float(logp + np.sum(np.logaddexp(l1, l2) - math.log(2.0)))


def solve_sgld_schedule(n_steps: int, eps_start: float, eps_end: float, gamma: float) -> tuple[float, float]:
    if n_steps < 2:
        raise ValueError("SGLD needs at least two updates.")
    ratio = (eps_start / eps_end) ** (1.0 / gamma)
    b = (n_steps - 1.0) / (ratio - 1.0)
    a = eps_start * b**gamma
    return float(a), float(b)


def _sgld_likelihood_grad_single(theta: Array, xi: Array) -> Array:
    var = jnp.asarray(CFG.likelihood_var, dtype=theta.dtype)
    t1, t2 = theta[0], theta[1]
    r1 = xi - t1
    r2 = xi - t1 - t2
    # Equal mixture weights and equal variances: posterior responsibility of component 2.
    log_a = -0.5 * r1**2 / var
    log_b = -0.5 * r2**2 / var
    resp2 = jax.nn.sigmoid(log_b - log_a)
    g1 = (1.0 - resp2) * r1 / var + resp2 * r2 / var
    g2 = resp2 * r2 / var
    return jnp.stack([g1, g2])


def run_sgld_jax(
    x: Array,
    key: Array,
    theta0: Array,
    *,
    n_steps: int,
    eps_start: float,
    eps_end: float,
    gamma: float,
) -> tuple[Array, Array]:
    """Paper-style batch-size-1 SGLD; returns every state and epsilon."""
    a, b = solve_sgld_schedule(n_steps, eps_start, eps_end, gamma)
    indices = jnp.arange(n_steps, dtype=jnp.int32)
    n_data = x.shape[0]

    def scan_step(carry, t):
        theta, key = carry
        key, k_idx, k_noise = jax.random.split(key, 3)
        idx = jax.random.randint(k_idx, (), 0, n_data)
        xi = x[idx]
        eps = jnp.asarray(a, theta.dtype) * (jnp.asarray(b, theta.dtype) + t) ** (-gamma)
        prior_grad = jnp.stack([
            -theta[0] / CFG.prior_var_theta1,
            -theta[1] / CFG.prior_var_theta2,
        ])
        likelihood_grad = _sgld_likelihood_grad_single(theta, xi)
        stochastic_grad = prior_grad + n_data * likelihood_grad
        noise = jax.random.normal(k_noise, shape=(2,), dtype=theta.dtype)
        theta = theta + 0.5 * eps * stochastic_grad + jnp.sqrt(eps) * noise
        return (theta, key), (theta, eps)

    (_, _), (states, epsilons) = jax.lax.scan(scan_step, (theta0, key), indices)
    return states, epsilons


run_sgld_jit = jax.jit(run_sgld_jax, static_argnames=("n_steps", "eps_start", "eps_end", "gamma"))


def autocorrelation_1d(x: np.ndarray, max_lag: int = 150) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return np.ones(max_lag + 1)
    return np.asarray([
        1.0 if lag == 0 else np.dot(x[:-lag], x[lag:]) / denom
        for lag in range(max_lag + 1)
    ])


def plot_sgld_diagnostics(
    sweep_states: np.ndarray,
    sweep_eps: np.ndarray,
    x_eval: np.ndarray,
) -> None:
    sweeps = np.arange(1, len(sweep_states) + 1)
    log_joint = np.asarray([log_joint_theta_np(t, x_eval) / len(x_eval) for t in sweep_states])
    acf1 = autocorrelation_1d(sweep_states[CFG.sgld_burnin_sweeps :, 0])
    acf2 = autocorrelation_1d(sweep_states[CFG.sgld_burnin_sweeps :, 1])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(sweeps, sweep_states[:, 0], label=r"$\theta_1$")
    axes[0, 0].plot(sweeps, sweep_states[:, 1], label=r"$\theta_2$")
    axes[0, 0].set_title("SGLD sweep-end traces")
    axes[0, 0].set_xlabel("full-data sweep")
    axes[0, 0].legend()

    axes[0, 1].plot(sweeps, sweep_eps)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("SGLD annealed step size")
    axes[0, 1].set_xlabel("full-data sweep")
    axes[0, 1].set_ylabel(r"$\epsilon_t$")

    axes[1, 0].plot(sweeps, log_joint)
    axes[1, 0].set_title("SGLD log joint probability per datum")
    axes[1, 0].set_xlabel("full-data sweep")

    lag = np.arange(len(acf1))
    axes[1, 1].plot(lag, acf1, label=r"$\theta_1$")
    axes[1, 1].plot(lag, acf2, label=r"$\theta_2$")
    axes[1, 1].axhline(0.0, linewidth=1)
    axes[1, 1].set_title("Post-burn-in autocorrelation")
    axes[1, 1].set_xlabel("lag in full-data sweeps")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(OUT / "50_sgld_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.show()


#%% 6) TRAIN BAYES TRANSPORT — run this cell when you are ready
# This cell does not use X_EVAL or THETA_EVAL_TRUE for optimisation.
# Every optimizer step receives a fresh theta* and a fresh batch of 128 x observations.

train_rng = np.random.default_rng(CFG.seed + 10_001)
replay_rng = np.random.default_rng(CFG.seed + 20_003)
dropout_key = jax.random.key(CFG.seed + 30_007)
replay_buffer = HistoricalPosteriorBuffer(
    CFG.historical_output_buffer_capacity,
    CFG.num_particles,
)

history = {name: [] for name in (
    "step",
    "energy_score",
    "final_energy_score",
    "attraction",
    "repulsion",
    "final_attraction",
    "final_repulsion",
    "mean_error",
    "final_mean_error",
    "covariance_trace",
    "final_covariance_trace",
    "grad_norm",
    "replay_used",
    "replay_distance",
    "interpolation_used",
    "interpolation_tau",
    "exact_prior_used",
    "buffer_size",
)}
prefix_history_steps: list[int] = []
prefix_history_counts: np.ndarray | None = None
prefix_energy_history: list[np.ndarray] = []

for step in range(1, CFG.training_steps + 1):
    # Fresh supervised simulation problem from the exact joint p(theta)p(x|theta).
    theta_target = sample_exact_prior_np(train_rng, 1)[0]
    x_batch = simulate_likelihood_np(train_rng, theta_target, CFG.batch_size)

    # Choose exactly ONE training prior source. The two explicit probabilities are disjoint,
    # and the residual probability uses the exact Gaussian prior that BT will see at evaluation.
    # If the buffer branch is selected before the buffer contains anything, fall back to the exact
    # evaluation prior rather than silently switching to interpolation.
    interpolation_used = False
    interpolation_tau = 0.0
    replay_used = False
    replay_distance = np.nan
    exact_prior_used = False
    signature = observation_signature(x_batch)

    prior_mode_u = replay_rng.random()
    p_interp = CFG.prior_interpolation_probability
    p_buffer = CFG.historical_output_prior_probability

    if prior_mode_u < p_interp:
        prior_theta, interpolation_tau = sample_interpolated_training_prior_np(
            train_rng, theta_target, CFG
        )
        interpolation_used = True
    elif prior_mode_u < p_interp + p_buffer and len(replay_buffer) > 0:
        prior_theta, replay_distance = replay_buffer.nearest(signature)
        replay_used = True
    else:
        prior_theta = sample_exact_prior_np(train_rng, CFG.num_particles)
        exact_prior_used = True

    dropout_key, step_key = jax.random.split(dropout_key)
    model, opt_state, loss, metrics, final_posterior, grad_norm = train_step(
        model,
        opt_state,
        jnp.asarray(prior_theta),
        jnp.asarray(x_batch),
        jnp.asarray(theta_target),
        step_key,
    )

    final_posterior_np = np.asarray(jax.device_get(final_posterior), dtype=np.float32)
    replay_buffer.add(signature, final_posterior_np)

    host = jax.device_get(metrics)
    scalar_values = {
        "step": float(step),
        "energy_score": float(host["energy_score"]),
        "final_energy_score": float(host["final_energy_score"]),
        "attraction": float(host["attraction"]),
        "repulsion": float(host["repulsion"]),
        "final_attraction": float(host["final_attraction"]),
        "final_repulsion": float(host["final_repulsion"]),
        "mean_error": float(host["mean_error"]),
        "final_mean_error": float(host["final_mean_error"]),
        "covariance_trace": float(host["covariance_trace"]),
        "final_covariance_trace": float(host["final_covariance_trace"]),
        "grad_norm": float(jax.device_get(grad_norm)),
        "replay_used": float(replay_used),
        "replay_distance": float(replay_distance),
        "interpolation_used": float(interpolation_used),
        "interpolation_tau": float(interpolation_tau),
        "exact_prior_used": float(exact_prior_used),
        "buffer_size": float(len(replay_buffer)),
    }
    for name, value in scalar_values.items():
        history[name].append(value)

    if step == 1 or step % CFG.prefix_history_every == 0 or step == CFG.training_steps:
        prefix_history_steps.append(step)
        prefix_history_counts = np.asarray(jax.device_get(host["prefix_counts"]), dtype=np.int32)
        prefix_energy_history.append(
            np.asarray(jax.device_get(host["energy_by_o"]), dtype=np.float32)
        )

    if step == 1 or step % CFG.log_every == 0 or step == CFG.training_steps:
        print(
            f"step {step:6d}/{CFG.training_steps} | "
            f"ES(all-o) {scalar_values['energy_score']:.5f} | "
            f"ES(final-o) {scalar_values['final_energy_score']:.5f} | "
            f"mean-err(final) {scalar_values['final_mean_error']:.4f} | "
            f"grad {scalar_values['grad_norm']:.3e} | "
            f"prior={'interp' if interpolation_used else ('buffer' if replay_used else 'exact-gaussian')}"
        )

# Save the trained transport and dense scalar diagnostics.
save_model(OUT / "bayes_transport.eqx", model, CFG)
with (OUT / "bt_training_history.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(history.keys()))
    writer.writeheader()
    for i in range(len(history["step"])):
        writer.writerow({k: history[k][i] for k in history})
np.savez_compressed(
    OUT / "bt_prefix_history.npz",
    steps=np.asarray(prefix_history_steps, dtype=np.int32),
    counts=prefix_history_counts,
    energy=np.stack(prefix_energy_history) if prefix_energy_history else np.empty((0, 0)),
)

# Requested end-of-training diagnostics.
plot_training_diagnostics(
    history,
    prefix_history_steps,
    prefix_history_counts,
    prefix_energy_history,
    CFG,
)

# Requested: after training, show the exact evaluation prior BEFORE posterior plots.
_eval_prior_rng = np.random.default_rng(CFG.eval_seed + 100)
EVAL_PRIOR_PARTICLES = sample_exact_prior_np(_eval_prior_rng, CFG.eval_particles)
plot_prior_diagnostics(EVAL_PRIOR_PARTICLES)


#%% 7) BAYES TRANSPORT evaluation on the fixed 100 x values + exact posterior contours
# If you skipped the training cell and want to load a checkpoint, run this line first:
# model = load_model(OUT / "bayes_transport.eqx", CFG)

# Exact numerical posterior for the same fixed evaluation dataset.
THETA1_GRID, THETA2_GRID, EXACT_DENSITY = exact_posterior_grid(X_EVAL, CFG)
_exact_rng = np.random.default_rng(CFG.eval_seed + 200)
EXACT_SAMPLES = sample_from_grid_posterior(
    _exact_rng,
    THETA1_GRID,
    THETA2_GRID,
    EXACT_DENSITY,
    CFG.exact_reference_samples,
)

# Evaluation prior is EXACTLY p(theta) used by SGLD: no interpolation and no replay.
if "EVAL_PRIOR_PARTICLES" not in globals():
    _eval_prior_rng = np.random.default_rng(CFG.eval_seed + 100)
    EVAL_PRIOR_PARTICLES = sample_exact_prior_np(_eval_prior_rng, CFG.eval_particles)

BT_POSTERIOR = np.asarray(
    jax.device_get(
        model.posterior_at_count(
            jnp.asarray(EVAL_PRIOR_PARTICLES),
            jnp.asarray(X_EVAL),
            CFG.eval_observations,
            inference=True,
        )
    ),
    dtype=np.float32,
)

plot_samples_on_exact_contours(
    BT_POSTERIOR,
    THETA1_GRID,
    THETA2_GRID,
    EXACT_DENSITY,
    title="Bayes Transport posterior samples over exact posterior contours",
    filename="30_bt_posterior_vs_exact.png",
)

# Causal diagnostic: how the posterior changes as observations accumulate.
plot_bt_prefix_evolution(
    model,
    EVAL_PRIOR_PARTICLES,
    X_EVAL,
    counts=(1, 2, 4, 8, 16, 32, 64, 100),
)

print("Bayes Transport evaluation summary:")
print(posterior_summary(BT_POSTERIOR))
print("Exact posterior summary:")
print(posterior_summary(EXACT_SAMPLES))


#%% 8) SGLD on the SAME fixed 100 x observations + SGLD diagnostics
# Paper Section 5.1 uses batch size 1 and 10,000 sweeps through N=100 data points.
if CFG.sgld_batch_size != 1:
    raise NotImplementedError("This clean Section 5.1 implementation intentionally matches paper batch size 1.")

SGLD_N_STEPS = CFG.sgld_sweeps * len(X_EVAL)
_sgld_init_rng = np.random.default_rng(CFG.eval_seed + 300)
SGLD_THETA0 = sample_exact_prior_np(_sgld_init_rng, 1)[0]

print(f"Running SGLD for {SGLD_N_STEPS:,} stochastic updates ({CFG.sgld_sweeps:,} sweeps)...")
_sgld_states_jax, _sgld_eps_jax = run_sgld_jit(
    jnp.asarray(X_EVAL),
    jax.random.key(CFG.eval_seed + 301),
    jnp.asarray(SGLD_THETA0),
    n_steps=SGLD_N_STEPS,
    eps_start=CFG.sgld_epsilon_start,
    eps_end=CFG.sgld_epsilon_end,
    gamma=CFG.sgld_gamma,
)
SGLD_ALL_STATES = np.asarray(jax.device_get(_sgld_states_jax), dtype=np.float32)
SGLD_ALL_EPS = np.asarray(jax.device_get(_sgld_eps_jax), dtype=np.float64)

# One state at the end of every full pass through the 100 observations.
_sweep_end_idx = np.arange(len(X_EVAL) - 1, SGLD_N_STEPS, len(X_EVAL))
SGLD_SWEEP_STATES = SGLD_ALL_STATES[_sweep_end_idx]
SGLD_SWEEP_EPS = SGLD_ALL_EPS[_sweep_end_idx]

# Discard burn-in, then epsilon-weighted resample over EVERY retained SGLD update,
# matching the paper's step-size weighting idea more closely than using sweep endpoints alone.
burnin_steps = CFG.sgld_burnin_sweeps * len(X_EVAL)
post_burn_states = SGLD_ALL_STATES[burnin_steps:]
post_burn_eps = SGLD_ALL_EPS[burnin_steps:]
weights = post_burn_eps / post_burn_eps.sum()
_sgld_resample_rng = np.random.default_rng(CFG.eval_seed + 302)
resample_ids = _sgld_resample_rng.choice(
    len(post_burn_states),
    size=CFG.sgld_reference_samples,
    replace=True,
    p=weights,
)
SGLD_POSTERIOR = post_burn_states[resample_ids]

plot_sgld_diagnostics(SGLD_SWEEP_STATES, SGLD_SWEEP_EPS, X_EVAL)
plot_samples_on_exact_contours(
    SGLD_POSTERIOR,
    THETA1_GRID,
    THETA2_GRID,
    EXACT_DENSITY,
    title="SGLD samples over exact posterior contours",
    filename="31_sgld_posterior_vs_exact.png",
)

print("SGLD evaluation summary:")
print(posterior_summary(SGLD_POSTERIOR))


#%% 9) FINAL comparison: exact posterior vs Bayes Transport vs SGLD
# All three panels below refer to the SAME fixed x_1:100 dataset.

plot_full_comparison(
    EVAL_PRIOR_PARTICLES,
    EXACT_SAMPLES,
    BT_POSTERIOR,
    SGLD_POSTERIOR,
    THETA1_GRID,
    THETA2_GRID,
    EXACT_DENSITY,
)
plot_marginal_comparison(EXACT_SAMPLES, BT_POSTERIOR, SGLD_POSTERIOR)

BT_METRICS = comparison_metrics(EXACT_SAMPLES, BT_POSTERIOR, "Bayes Transport")
SGLD_METRICS = comparison_metrics(EXACT_SAMPLES, SGLD_POSTERIOR, "SGLD")

print("\nComparison to exact numerical posterior")
for row in (BT_METRICS, SGLD_METRICS):
    print("\n", row["method"])
    for key, value in row.items():
        if key != "method":
            print(f"  {key:30s}: {value:.6f}")

with (OUT / "comparison_metrics.csv").open("w", newline="") as f:
    fieldnames = list(BT_METRICS.keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(BT_METRICS)
    writer.writerow(SGLD_METRICS)

# Extra direct sample cloud comparison with exact contours.
fig, ax = plt.subplots(figsize=(8, 7.5))
ax.contour(
    THETA1_GRID,
    THETA2_GRID,
    EXACT_DENSITY,
    levels=credible_density_levels(EXACT_DENSITY),
    linewidths=2,
)
ax.scatter(BT_POSTERIOR[:, 0], BT_POSTERIOR[:, 1], s=12, alpha=0.30, label="Bayes Transport")
ax.scatter(SGLD_POSTERIOR[:, 0], SGLD_POSTERIOR[:, 1], s=12, alpha=0.20, label="SGLD")
ax.scatter([THETA_EVAL_TRUE[0]], [THETA_EVAL_TRUE[1]], marker="*", s=180, label="Generating theta")
ax.set_xlim(CFG.grid_theta1_min, CFG.grid_theta1_max)
ax.set_ylim(CFG.grid_theta2_min, CFG.grid_theta2_max)
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.set_title("Direct posterior sample comparison on exact contours")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "42_bt_vs_sgld_samples.png", dpi=180, bbox_inches="tight")
plt.show()

print("\nSaved figures and metrics to:", OUT.resolve())
