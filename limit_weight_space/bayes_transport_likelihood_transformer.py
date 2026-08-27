#%% 0) Imports and experiment overview
"""Conditional empirical-posterior regression from observed (x, y) pairs only.

A uniform cloud of candidate y values is transported by one particle Transformer.
For each observed training pair (x, y*), x is treated as the centre of a configurable
Gaussian empirical likelihood p(x_lik | y*). A configurable block of independent noisy
x_lik observations is causally contextualised by a likelihood Transformer before conditioning
the particle transport. The empirical energy score trains the full y cloud. Its mean is the
point prediction; its empirical quantiles are uncertainty intervals.

Retained from bayes-transport:
  * uniform base prior;
  * shared-tau interpolated training priors;
  * Gaussian-noisified x likelihood observations from the supplied data pairs;
  * causal likelihood Transformer over variable observation prefixes;
  * historical-output replay buffer;
  * identity-initialised particle transport;
  * empirical energy score and cloud diagnostics.

Removed:
  * simulator / synthetic (x,y) generation;
  * dimension embedders;
  * recurrent Bayes rollouts;
  * DEQ / fixed-point / drifting modes;
  * heterogeneous shapes and padding.

The bottom of the file trains the same-condition MLP and GP baselines and both saves
and displays a three-way comparison on exactly the data/split used by nn.py and gp.py.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

try:
    from IPython.display import display as ipy_display
except ImportError:  # Plain Python scripts still work outside Jupyter.
    ipy_display = None

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, DotProduct, RBF, WhiteKernel

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "DejaVu Sans",
    "axes.titlepad": 8.0,
    "axes.labelpad": 6.0,
})

Array = jax.Array


#%% 1) Configuration and data helpers: identical benchmark convention to nn.py / gp.py
@dataclass
class Config:
    seed: int = 2028
    data_samples: int = 2000
    noise_std: float = 0.005
    segments: int = 11
    x_range: tuple[float, float] = (-1.5, 1.5)
    train_seg_ids: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)

    # Particle transport.
    num_particles: int = 16
    eval_particles: int = 64
    hidden_dim: int = 64
    heads: int = 4
    mlp_ratio: int = 4
    posterior_depth: int = 3
    posterior_conditioning: str = "cross_attention"  # {"adaln", "cross_attention"}
    cross_attention_tokens: int = 1  # retained compatibility field; likelihood contexts now supply memory
    max_normalized_displacement: float = 5.0

    # Training: one epoch = one observed-data minibatch / optimizer step.
    epochs: int = 5000
    batch_size: int = 64*4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0
    log_every: int = 1250

    # ONE widest uniform y-prior support. Fresh training interpolation clouds always
    # come from this support. Evaluation may reuse it, use a nearest-training local prior,
    # or use the default cheating/oracle prior centred on the known test label.
    prior_min: float = -2.0
    prior_max: float = 2.0

    # Empirical likelihood attached to each supplied training pair:
    #     x_lik ~ Normal(x_pair, likelihood_x_noise_std^2).
    # The noisy draw x_lik, not the clean pair centre x_pair, conditions the transport.
    # Replay stores the exact realised x_lik so the same evidence can be reused later.
    likelihood_x_noise_std: float = 0.01

    # Causal likelihood Transformer, adapted directly from the location-finding experiment.
    # One observation means one INDEPENDENT Gaussian x_lik draw around the same clean x_pair.
    # Every training step draws max_observations_per_step noisy observations and optimizes EVERY
    # prefix o=min_observations_per_step,...,max_observations_per_step from the SAME prior cloud.
    # Prefixes are direct prior->posterior maps; posterior(o-1) is never fed into posterior(o).
    min_observations_per_step: int = 1
    max_observations_per_step: int = 6
    test_observations_per_step: int = 6
    likelihood_hidden_dim: int = 64
    likelihood_heads: int = 4
    likelihood_mlp_ratio: int = 4
    likelihood_depth: int = 4

    # Evaluation/deployment prior.
    #   "cheating" (DEFAULT): oracle/domain-expert mode. The true evaluation label y* is
    #       assumed known and defines the prior centre. The prior full width is
    #       cheating_prior_width_fraction times the widest pre-interpolation training width.
    #       With prior=[-2,2] and fraction=0.5, this gives U(y*-1, y*+1).
    #   "widest": reuse the full configured training support.
    #   "nearest_training": infer a local support only from nearby known training (x,y) pairs.
    evaluation_prior_mode: str = "widest"  # {"cheating", "widest", "nearest_training"}
    cheating_prior_width_fraction: float = 0.90
    eval_local_prior_k: int = 64
    eval_local_prior_margin: float = 0.25
    eval_local_prior_min_width: float = 1.00
    eval_local_prior_clip_to_global: bool = True

    # Shared-tau interpolation. Strict supervised default is 0: no y* enters the input cloud.
    # Set to 1 to reproduce the original truth-anchored Bayes-transport interpolation ablation.
    truth_anchor_probability: float = 0.0

    # Historical model-output prior replay.
    historical_output_prior_probability: float = 0.50
    historical_output_buffer_capacity: int = 2048

    # Empirical interval extracted from the transported cloud.
    interval_low_q: float = 0.025
    interval_high_q: float = 0.975
    eval_batch_size: int = 256

    # Attached nn.py baseline.
    nn_hidden_size: int = 64
    nn_learning_rate: float = 1e-3
    nn_epochs: int = 3000

    # Attached gp.py baseline.
    gp_restarts: int = 10

    # Plotting. Figures are always saved; when True they are also rendered inline in notebooks.
    show_plots: bool = True

    output_dir: str = "plots/cosine_transport"


CFG = Config()


def validate_config(cfg: Config) -> None:
    if cfg.num_particles < 2 or cfg.eval_particles < 2:
        raise ValueError("Particle counts must be >= 2.")
    if cfg.hidden_dim % cfg.heads != 0:
        raise ValueError("hidden_dim must be divisible by heads.")
    if cfg.posterior_conditioning not in {"adaln", "cross_attention"}:
        raise ValueError("posterior_conditioning must be 'adaln' or 'cross_attention'.")
    if cfg.cross_attention_tokens < 1:
        raise ValueError("cross_attention_tokens must be >= 1.")
    if not 0.0 <= cfg.truth_anchor_probability <= 1.0:
        raise ValueError("truth_anchor_probability must lie in [0,1].")
    if not 0.0 <= cfg.historical_output_prior_probability <= 1.0:
        raise ValueError("historical_output_prior_probability must lie in [0,1].")
    if not cfg.prior_min < cfg.prior_max:
        raise ValueError("prior_min must be strictly smaller than prior_max.")
    if cfg.likelihood_x_noise_std <= 0.0:
        raise ValueError("likelihood_x_noise_std must be > 0 for a non-degenerate Gaussian likelihood.")
    if cfg.min_observations_per_step < 1:
        raise ValueError("min_observations_per_step must be >= 1.")
    if cfg.max_observations_per_step < cfg.min_observations_per_step:
        raise ValueError("max_observations_per_step must be >= min_observations_per_step.")
    if not (cfg.min_observations_per_step <= cfg.test_observations_per_step <= cfg.max_observations_per_step):
        raise ValueError(
            "test_observations_per_step must lie in "
            "[min_observations_per_step, max_observations_per_step]."
        )
    if cfg.likelihood_hidden_dim < 1:
        raise ValueError("likelihood_hidden_dim must be >= 1.")
    if cfg.likelihood_heads < 1 or cfg.likelihood_hidden_dim % cfg.likelihood_heads != 0:
        raise ValueError("likelihood_hidden_dim must be divisible by likelihood_heads.")
    if cfg.likelihood_mlp_ratio < 1 or cfg.likelihood_depth < 1:
        raise ValueError("likelihood_mlp_ratio and likelihood_depth must both be >= 1.")
    if cfg.evaluation_prior_mode not in {"cheating", "widest", "nearest_training"}:
        raise ValueError(
            "evaluation_prior_mode must be 'cheating', 'widest', or 'nearest_training'."
        )
    if not 0.0 < cfg.cheating_prior_width_fraction <= 1.0:
        raise ValueError("cheating_prior_width_fraction must lie in (0,1].")
    if cfg.eval_local_prior_k < 1:
        raise ValueError("eval_local_prior_k must be >= 1.")
    if cfg.eval_local_prior_margin < 0.0:
        raise ValueError("eval_local_prior_margin must be >= 0.")
    if cfg.eval_local_prior_min_width <= 0.0:
        raise ValueError("eval_local_prior_min_width must be > 0.")


def gen_data(cfg: Config = CFG) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exactly the data law and segment split in the attached baselines."""
    np.random.seed(cfg.seed)
    x_min, x_max = cfg.x_range
    x = np.random.uniform(x_min, x_max, cfg.data_samples)
    y = np.cos(10.0 * x) + 0.5 * x
    y += np.random.normal(0.0, cfg.noise_std, cfg.data_samples)
    bins = np.linspace(x_min, x_max, cfg.segments + 1)
    segs = np.clip(np.digitize(x, bins) - 1, 0, cfg.segments - 1)
    train = np.isin(segs, cfg.train_seg_ids)
    return x[train], y[train], x[~train], y[~train]


@dataclass(frozen=True)
class Scaling:
    x_center: float
    x_scale: float
    y_center: float
    y_scale: float
    prior_low: float
    prior_high: float


def make_scaling(x_train: np.ndarray, y_train: np.ndarray, cfg: Config = CFG) -> Scaling:
    """Normalization derived from x_train and the manually configured global y-prior support."""
    del y_train  # y targets do not determine the prior support.
    x_lo, x_hi = float(x_train.min()), float(x_train.max())
    prior_low, prior_high = float(cfg.prior_min), float(cfg.prior_max)
    y_center = 0.5 * (prior_low + prior_high)
    y_half = max(0.5 * (prior_high - prior_low), 1e-6)
    return Scaling(
        x_center=0.5 * (x_lo + x_hi),
        x_scale=max(0.5 * (x_hi - x_lo), 1e-6),
        y_center=y_center,
        y_scale=y_half,
        prior_low=prior_low,
        prior_high=prior_high,
    )




# Larger notebook plotting defaults. These are presentation settings, not model hyperparameters.
plt.rcParams.update({
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
})


def _save_and_show(fig, path: Path, cfg: Config) -> None:
    """Persist a figure and also render it inline when requested."""
    fig.savefig(path, bbox_inches="tight")
    if cfg.show_plots:
        if ipy_display is not None:
            ipy_display(fig)
        else:
            plt.show()
    plt.close(fig)


def plot_data_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    path: Path,
    cfg: Config = CFG,
) -> None:
    """Visualize the fixed train/test split before any model is defined or trained."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x_all = np.concatenate([x_train, x_test])
    y_all = np.concatenate([y_train, y_test])
    ix = np.argsort(x_all)
    ax.plot(x_all[ix], y_all[ix], c="black", linewidth=2.5, alpha=.8, label="Observed cosine relation")
    ax.scatter(x_train, y_train, s=26, alpha=.55, label="Training data")
    ax.scatter(x_test, y_test, s=26, alpha=.55, label="Test/OOD data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Cosine regression data and fixed segment split")
    ax.grid(alpha=.15)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_and_show(fig, path, cfg)


#%% 2) Load data, create output directory, and visualise the benchmark split
validate_config(CFG)
out = Path(CFG.output_dir)
out.mkdir(parents=True, exist_ok=True)

X_train, Y_train, X_test, Y_test = gen_data(CFG)
scaling = make_scaling(X_train, Y_train, CFG)

print("\n--- data / experiment setup: before training ---")
print(f"Train samples: {len(X_train)} | Test/OOD samples: {len(X_test)}")
conditioning_detail = (
    f"{CFG.posterior_conditioning} (causal likelihood-prefix memory)"
    if CFG.posterior_conditioning == "cross_attention"
    else f"{CFG.posterior_conditioning} (final causal likelihood-prefix token)"
)
print(f"Posterior conditioning: {conditioning_detail}")
print(
    f"WIDEST UNIFORM TRAINING PRIOR = U({scaling.prior_low:.3f}, {scaling.prior_high:.3f}) "
    f"| width={scaling.prior_high - scaling.prior_low:.3f} | center={scaling.y_center:.3f}"
)
print(
    f"Gaussian x-likelihood during training: x_lik ~ N(x_pair, {CFG.likelihood_x_noise_std:.4f}^2). "
    "Deployment/evaluation x is NOT noisified."
)
print(
    "Likelihood observation prefixes: "
    f"train o={CFG.min_observations_per_step}..{CFG.max_observations_per_step} "
    f"(one independent noise draw per observation) | deployment o={CFG.test_observations_per_step}"
)
print(
    "Likelihood Transformer: "
    f"hidden={CFG.likelihood_hidden_dim}, heads={CFG.likelihood_heads}, "
    f"depth={CFG.likelihood_depth}, mlp_ratio={CFG.likelihood_mlp_ratio}"
)
print(f"Configured evaluation prior mode: {CFG.evaluation_prior_mode}")
train_inside = np.mean((Y_train >= scaling.prior_low) & (Y_train <= scaling.prior_high))
test_inside = np.mean((Y_test >= scaling.prior_low) & (Y_test <= scaling.prior_high))
print(
    f"Train y: min={Y_train.min():.3f} max={Y_train.max():.3f} "
    f"mean={Y_train.mean():.3f} std={Y_train.std():.3f} | inside widest prior={train_inside:.3%}"
)
print(
    f"Test  y: min={Y_test.min():.3f} max={Y_test.max():.3f} "
    f"mean={Y_test.mean():.3f} std={Y_test.std():.3f} | inside widest prior={test_inside:.3%}"
)
print(
    f"truth_anchor_probability={CFG.truth_anchor_probability:.3f} | "
    f"historical_replay_probability={CFG.historical_output_prior_probability:.3f} | "
    f"train/eval particles={CFG.num_particles}/{CFG.eval_particles}"
)
if train_inside < 1.0 or test_inside < 1.0:
    print("WARNING: some observed y values lie outside the configured widest prior support.")
plot_data_split(X_train, Y_train, X_test, Y_test, out / "data_split.pdf", CFG)



#%% 3) Training utilities: Gaussian likelihood helpers

def sample_likelihood_x_np(
    rng: np.random.Generator,
    x_pair: np.ndarray,
    cfg: Config,
    *,
    num_observations: int | None = None,
) -> np.ndarray:
    """Draw independent x_lik observations for every supplied clean x_pair.

    Returns [B,O]. Observation j is one independent Gaussian noise sample around the same
    clean x_pair. This is the cosine analogue of one independent likelihood observation token.
    """
    x_pair = np.asarray(x_pair, dtype=np.float32).reshape(-1)
    o = cfg.max_observations_per_step if num_observations is None else int(num_observations)
    if o < 1:
        raise ValueError("num_observations must be >= 1.")
    noise = rng.normal(size=(len(x_pair), o)).astype(np.float32)
    return (
        x_pair[:, None] + np.float32(cfg.likelihood_x_noise_std) * noise
    ).astype(np.float32)


def gaussian_likelihood_logpdf_np(
    x_lik: np.ndarray,
    x_pair: np.ndarray,
    std: float,
) -> np.ndarray:
    """Log-density of the configured Gaussian empirical likelihood."""
    z = (np.asarray(x_lik) - np.asarray(x_pair)) / float(std)
    return -0.5 * z**2 - np.log(float(std) * np.sqrt(2.0 * np.pi))




#%% 4) Training model: causal likelihood Transformer + selectable posterior conditioning
def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]


class CausalObservationBlock(eqx.Module):
    """Transformer block over one Omax x-likelihood sequence with a causal mask."""
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, dim: int, heads: int, mlp_dim: int, *, key: Array):
        attn_key, ff1_key, ff2_key = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=dim, key_size=dim, value_size=dim,
            output_size=dim, dropout_p=0.0, key=attn_key,
        )
        self.ff_in = eqx.nn.Linear(dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, dim, key=ff2_key)

    def __call__(self, tokens: Array) -> Array:
        length = tokens.shape[0]
        index = jnp.arange(length)
        causal_mask = index[:, None] >= index[None, :]
        h = _layernorm_tokens(self.norm1, tokens)
        tokens = tokens + self.attention(h, h, h, mask=causal_mask)
        h = _layernorm_tokens(self.norm2, tokens)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        return tokens + _linear_tokens(self.ff_out, h)


class LikelihoodSequenceEmbedder(eqx.Module):
    """Project and causally contextualise a variable prefix of noisy x observations.

    Output token o can depend only on observations 0,...,o. One causal pass therefore provides
    all prefix contexts used on the training step. If Omin=Omax=1, this is intentionally a
    zero-parameter identity pass-through, matching the location-finding implementation.
    """
    input_projection: eqx.nn.Linear | None
    blocks: tuple[CausalObservationBlock, ...]
    final_norm: eqx.nn.LayerNorm | None
    input_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    attention_heads: int = eqx.field(static=True)
    bypass_single_observation: bool = eqx.field(static=True)

    def __init__(self, cfg: Config, *, key: Array, input_dim: int = 1):
        self.input_dim = int(input_dim)
        self.bypass_single_observation = (
            cfg.min_observations_per_step == 1 and cfg.max_observations_per_step == 1
        )
        if self.bypass_single_observation:
            self.hidden_dim = self.input_dim
            self.attention_heads = 0
            self.input_projection = None
            self.blocks = ()
            self.final_norm = None
            return
        self.hidden_dim = int(cfg.likelihood_hidden_dim)
        self.attention_heads = int(cfg.likelihood_heads)
        keys = jax.random.split(key, cfg.likelihood_depth + 1)
        self.input_projection = eqx.nn.Linear(self.input_dim, self.hidden_dim, key=keys[0])
        self.blocks = tuple(
            CausalObservationBlock(
                self.hidden_dim,
                self.attention_heads,
                cfg.likelihood_mlp_ratio * self.hidden_dim,
                key=keys[1 + i],
            )
            for i in range(cfg.likelihood_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(self.hidden_dim)

    def __call__(self, observation_tokens: Array) -> Array:
        if self.bypass_single_observation:
            if observation_tokens.shape[0] != 1:
                raise ValueError("Single-observation likelihood bypass expects exactly one token.")
            return observation_tokens
        if self.input_projection is None or self.final_norm is None:
            raise RuntimeError("Active likelihood Transformer is missing its learned layers.")
        tokens = _linear_tokens(self.input_projection, observation_tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return _layernorm_tokens(self.final_norm, tokens)


class AdaLNParticleBlock(eqx.Module):
    """Particle self-attention conditioned by the causal summary token for observations 1:o."""
    norm_attn: eqx.nn.LayerNorm
    norm_ff: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear
    modulation: eqx.nn.Linear

    def __init__(
        self, hidden: int, conditioning_dim: int, heads: int, mlp_dim: int, *, key: Array
    ):
        k_attn, k_ff1, k_ff2, k_mod = jax.random.split(key, 4)
        self.norm_attn = eqx.nn.LayerNorm(hidden)
        self.norm_ff = eqx.nn.LayerNorm(hidden)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=hidden, key_size=hidden, value_size=hidden,
            output_size=hidden, dropout_p=0.0, key=k_attn,
        )
        self.ff_in = eqx.nn.Linear(hidden, mlp_dim, key=k_ff1)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden, key=k_ff2)
        modulation = eqx.nn.Linear(conditioning_dim, 6 * hidden, key=k_mod)
        modulation = eqx.tree_at(lambda l: l.weight, modulation, jnp.zeros_like(modulation.weight))
        modulation = eqx.tree_at(lambda l: l.bias, modulation, jnp.zeros_like(modulation.bias))
        self.modulation = modulation

    def __call__(self, particles: Array, conditioning: Array) -> Array:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = jnp.split(
            self.modulation(jax.nn.silu(conditioning)), 6, axis=-1
        )
        h = _modulate(_layernorm_tokens(self.norm_attn, particles), shift_a, scale_a)
        particles = particles + gate_a[None, :] * self.attention(h, h, h)
        h = _modulate(_layernorm_tokens(self.norm_ff, particles), shift_f, scale_f)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        return particles + gate_f[None, :] * _linear_tokens(self.ff_out, h)


class CrossAttentionParticleBlock(eqx.Module):
    """Particle self-attention followed by cross-attention to the active causal prefix."""
    norm_self: eqx.nn.LayerNorm
    norm_cross: eqx.nn.LayerNorm
    memory_norm: eqx.nn.LayerNorm
    norm_ff: eqx.nn.LayerNorm
    self_attention: eqx.nn.MultiheadAttention
    cross_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(
        self, hidden: int, memory_dim: int, heads: int, mlp_dim: int, *, key: Array
    ):
        k_self, k_cross, k_ff1, k_ff2 = jax.random.split(key, 4)
        self.norm_self = eqx.nn.LayerNorm(hidden)
        self.norm_cross = eqx.nn.LayerNorm(hidden)
        self.memory_norm = eqx.nn.LayerNorm(memory_dim)
        self.norm_ff = eqx.nn.LayerNorm(hidden)
        self.self_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=hidden, key_size=hidden, value_size=hidden,
            output_size=hidden, dropout_p=0.0, key=k_self,
        )
        self.cross_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=hidden, key_size=memory_dim, value_size=memory_dim,
            output_size=hidden, dropout_p=0.0, key=k_cross,
        )
        self.ff_in = eqx.nn.Linear(hidden, mlp_dim, key=k_ff1)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden, key=k_ff2)

    def __call__(self, particles: Array, observation_memory: Array) -> Array:
        h = _layernorm_tokens(self.norm_self, particles)
        particles = particles + self.self_attention(h, h, h)
        q = _layernorm_tokens(self.norm_cross, particles)
        memory = _layernorm_tokens(self.memory_norm, observation_memory)
        particles = particles + self.cross_attention(q, memory, memory)
        h = _layernorm_tokens(self.norm_ff, particles)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        return particles + _linear_tokens(self.ff_out, h)


class ConditionalParticleTransport(eqx.Module):
    """One direct prior->posterior map conditioned on a causal noisy-x likelihood sequence."""
    particle_in: eqx.nn.Linear
    likelihood_embedder: LikelihoodSequenceEmbedder
    blocks: tuple[Any, ...]
    final_norm: eqx.nn.LayerNorm
    displacement_head: eqx.nn.Linear

    x_center: float = eqx.field(static=True)
    x_scale: float = eqx.field(static=True)
    y_center: float = eqx.field(static=True)
    y_scale: float = eqx.field(static=True)
    max_displacement: float = eqx.field(static=True)
    conditioning_type: str = eqx.field(static=True)
    min_observations: int = eqx.field(static=True)
    max_observations: int = eqx.field(static=True)
    observation_context_dim: int = eqx.field(static=True)

    def __init__(self, cfg: Config, scaling: Scaling, *, key: Array):
        keys = jax.random.split(key, cfg.posterior_depth + 4)
        h = cfg.hidden_dim
        self.likelihood_embedder = LikelihoodSequenceEmbedder(cfg, input_dim=1, key=keys[0])
        self.observation_context_dim = self.likelihood_embedder.hidden_dim
        self.particle_in = eqx.nn.Linear(1, h, key=keys[1])
        block_cls = AdaLNParticleBlock if cfg.posterior_conditioning == "adaln" else CrossAttentionParticleBlock
        self.blocks = tuple(
            block_cls(
                h,
                self.observation_context_dim,
                cfg.heads,
                cfg.mlp_ratio * h,
                key=keys[2 + i],
            )
            for i in range(cfg.posterior_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(h)
        head = eqx.nn.Linear(h, 1, key=keys[-1])
        head = eqx.tree_at(lambda l: l.weight, head, jnp.zeros_like(head.weight))
        head = eqx.tree_at(lambda l: l.bias, head, jnp.zeros_like(head.bias))
        self.displacement_head = head
        self.x_center, self.x_scale = scaling.x_center, scaling.x_scale
        self.y_center, self.y_scale = scaling.y_center, scaling.y_scale
        self.max_displacement = cfg.max_normalized_displacement
        self.conditioning_type = cfg.posterior_conditioning
        self.min_observations = int(cfg.min_observations_per_step)
        self.max_observations = int(cfg.max_observations_per_step)

    def encode_observations(self, x_observations: Array) -> Array:
        """x_observations [O] physical units -> causal contexts [O,C]."""
        xn = (x_observations - self.x_center) / self.x_scale
        return self.likelihood_embedder(xn[:, None])

    def transport_with_contexts(
        self, prior_y: Array, observation_contexts: Array, observation_count: Array
    ) -> Array:
        """Direct prior -> posterior for one selected observation prefix."""
        yn = (prior_y - self.y_center) / self.y_scale
        count = jnp.clip(observation_count, 1, observation_contexts.shape[0]).astype(jnp.int32)

        if self.conditioning_type == "cross_attention":
            # Static branch shapes make memory[:o] legal inside JIT, exactly as in location finding.
            def branch_for(prefix_length: int):
                def transport(args):
                    yn_local, full_memory = args
                    particles = _linear_tokens(self.particle_in, yn_local[:, None])
                    memory = full_memory[:prefix_length]
                    for block in self.blocks:
                        particles = block(particles, memory)
                    return particles
                return transport

            branches = tuple(
                branch_for(prefix_length)
                for prefix_length in range(1, observation_contexts.shape[0] + 1)
            )
            particles = jax.lax.switch(count - 1, branches, (yn, observation_contexts))
        else:
            # Causal token o-1 summarizes exactly observations 1:o.
            conditioning = observation_contexts[count - 1]
            particles = _linear_tokens(self.particle_in, yn[:, None])
            for block in self.blocks:
                particles = block(particles, conditioning)

        particles = _layernorm_tokens(self.final_norm, particles)
        delta = self.max_displacement * jnp.tanh(
            _linear_tokens(self.displacement_head, particles)[:, 0]
        )
        return self.y_center + self.y_scale * (yn + delta)

    def predict_prefixes(self, prior_y: Array, x_observations: Array) -> tuple[Array, Array]:
        """Direct posterior for EVERY configured prefix from the SAME prior cloud.

        This is not a posterior recurrence. The likelihood Transformer is run once over Omax
        observations; then o=Omin,...,Omax independently conditions the same incoming prior.
        """
        contexts = self.encode_observations(x_observations)
        prefix_counts = jnp.arange(self.min_observations, self.max_observations + 1, dtype=jnp.int32)
        posterior_by_prefix = jax.vmap(
            lambda count: self.transport_with_contexts(prior_y, contexts, count)
        )(prefix_counts)
        return posterior_by_prefix, prefix_counts

    def __call__(self, prior_y: Array, x_observations: Array, observation_count: Array) -> Array:
        """Evaluate one selected prefix. x_observations may have any supported static length."""
        contexts = self.encode_observations(x_observations)
        return self.transport_with_contexts(prior_y, contexts, observation_count)


#%% 5) Training objective, interpolated priors, and historical-output replay
def empirical_energy_score_terms(posterior: Array, target: Array) -> tuple[Array, Array, Array]:
    """1-D empirical ES: E|Y-y*| - 1/2 E|Y-Y'|."""
    attraction = jnp.mean(jnp.abs(posterior - target))
    repulsion = jnp.mean(jnp.abs(posterior[:, None] - posterior[None, :]))
    return attraction - 0.5 * repulsion, attraction, repulsion


def prefix_batch_metrics(
    posterior_by_prefix: Array, target: Array, prefix_counts: Array, cfg: Config = CFG
) -> dict[str, Array]:
    """Metrics for [B,P,N] posterior clouds; scalar loss averages all rows and prefixes."""
    mean = jnp.mean(posterior_by_prefix, axis=-1)  # [B,P]
    lo = jnp.quantile(posterior_by_prefix, cfg.interval_low_q, axis=-1)
    hi = jnp.quantile(posterior_by_prefix, cfg.interval_high_q, axis=-1)

    def row_terms(prefix_clouds, yi):
        return jax.vmap(lambda cloud: empirical_energy_score_terms(cloud, yi))(prefix_clouds)

    score, attraction, repulsion = jax.vmap(row_terms)(posterior_by_prefix, target)
    mse = (mean - target[:, None]) ** 2
    coverage = (target[:, None] >= lo) & (target[:, None] <= hi)
    width = hi - lo
    spread = jnp.std(posterior_by_prefix, axis=-1)
    return {
        "loss": jnp.mean(score),
        "energy_score": jnp.mean(score),
        "final_energy_score": jnp.mean(score[:, -1]),
        "attraction": jnp.mean(attraction),
        "repulsion": jnp.mean(repulsion),
        "mean_mse": jnp.mean(mse),
        "final_mean_mse": jnp.mean(mse[:, -1]),
        "coverage_95": jnp.mean(coverage),
        "final_coverage_95": jnp.mean(coverage[:, -1]),
        "interval_width": jnp.mean(width),
        "posterior_std": jnp.mean(spread),
        "energy_by_o": jnp.mean(score, axis=0),
        "mse_by_o": jnp.mean(mse, axis=0),
        "coverage_by_o": jnp.mean(coverage, axis=0),
        "prefix_counts": prefix_counts,
    }


def sample_interpolated_prior_np(
    rng: np.random.Generator,
    target_y: np.ndarray,
    cfg: Config,
    scaling: Scaling,
) -> np.ndarray:
    """Shared-tau C_tau=(1-tau)Z+tau*anchor; no synthetic (x,y) pairs are created."""
    b = len(target_y)
    z = rng.uniform(
        scaling.prior_low, scaling.prior_high, size=(b, cfg.num_particles)
    ).astype(np.float32)
    independent_anchor = rng.uniform(
        scaling.prior_low, scaling.prior_high, size=b
    ).astype(np.float32)
    use_truth = rng.random(b) < cfg.truth_anchor_probability
    anchor = np.where(use_truth, target_y, independent_anchor).astype(np.float32)
    tau = rng.uniform(0.0, 1.0, size=(b, 1)).astype(np.float32)
    return ((1.0 - tau) * z + tau * anchor[:, None]).astype(np.float32)


class HistoricalOutputPriorBuffer:
    """Detached final-prefix clouds plus the exact full noisy-x evidence block."""
    def __init__(self, capacity: int):
        self.entries: deque[dict[str, Any]] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self.entries)

    def add_batch(
        self,
        posterior: np.ndarray,
        x_pair: np.ndarray,
        x_lik: np.ndarray,
        y: np.ndarray,
    ) -> None:
        for cloud, xc, xo, yi in zip(posterior, x_pair, x_lik, y):
            self.entries.append({
                "prior": np.asarray(cloud, dtype=np.float32).copy(),
                "x_pair": np.float32(xc),
                "x_lik": np.asarray(xo, dtype=np.float32).copy(),
                "y": np.float32(yi),
            })

    def mix_into_batch(
        self,
        prior: np.ndarray,
        x_pair: np.ndarray,
        x_lik: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
        cfg: Config,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        if not self.entries or cfg.historical_output_prior_probability <= 0.0:
            return prior, x_pair, x_lik, y, 0
        prior = prior.copy()
        x_pair, x_lik, y = x_pair.copy(), x_lik.copy(), y.copy()
        selected = np.flatnonzero(
            rng.random(len(x_pair)) < cfg.historical_output_prior_probability
        )
        for i in selected:
            entry = self.entries[int(rng.integers(len(self.entries)))]
            prior[i] = entry["prior"]
            x_pair[i] = entry["x_pair"]
            x_lik[i] = entry["x_lik"]
            y[i] = entry["y"]
        return prior, x_pair, x_lik, y, int(len(selected))


#%% 6) Training loop for particle transport
def _transport_objective(
    model: ConditionalParticleTransport,
    prior: Array,
    x_observations: Array,
    y: Array,
    cfg: Config,
) -> tuple[Array, tuple[dict[str, Array], Array]]:
    """Train every configured observation prefix from the same incoming prior cloud."""
    posterior_by_prefix, prefix_counts = jax.vmap(
        lambda p, obs: model.predict_prefixes(p, obs)
    )(prior, x_observations)
    metrics = prefix_batch_metrics(posterior_by_prefix, y, prefix_counts[0], cfg)
    # Historical replay stores the final-prefix output, matching the location-finding design.
    return metrics["loss"], (metrics, posterior_by_prefix[:, -1])


_loss_and_grad = eqx.filter_value_and_grad(_transport_objective, has_aux=True)


def make_train_step(optimizer: optax.GradientTransformation, cfg: Config):
    @eqx.filter_jit
    def step(model, opt_state, prior, x_observations, y):
        (loss, (metrics, final_posterior)), grads = _loss_and_grad(
            model, prior, x_observations, y, cfg
        )
        params = eqx.filter(model, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return model, opt_state, loss, metrics, final_posterior, grad_norm
    return step


def train_transport(
    x_train: np.ndarray,
    y_train: np.ndarray,
    scaling: Scaling,
    cfg: Config = CFG,
    *,
    model: ConditionalParticleTransport | None = None,
) -> tuple[ConditionalParticleTransport, dict[str, list[float]]]:
    if model is None:
        model = ConditionalParticleTransport(cfg, scaling, key=jax.random.key(cfg.seed))
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step = make_train_step(optimizer, cfg)
    rng = np.random.default_rng(cfg.seed + 10_001)
    replay_rng = np.random.default_rng(cfg.seed + 20_003)
    replay = HistoricalOutputPriorBuffer(cfg.historical_output_buffer_capacity)
    history = {name: [] for name in (
        "step", "energy_score", "final_energy_score", "attraction", "repulsion",
        "mean_mse", "final_mean_mse", "coverage_95", "final_coverage_95",
        "interval_width", "posterior_std", "grad_norm", "replay_fraction", "buffer_size",
        "likelihood_noise_rms", "likelihood_mean_logpdf"
    )}

    for step in range(1, cfg.epochs + 1):
        ids = rng.integers(0, len(x_train), size=cfg.batch_size)
        x_pair = x_train[ids].astype(np.float32)
        yb = y_train[ids].astype(np.float32)
        prior = sample_interpolated_prior_np(rng, yb, cfg, scaling)
        # One observation == one independent Gaussian noise sample. Shape [B,Omax].
        x_lik = sample_likelihood_x_np(
            rng, x_pair, cfg, num_observations=cfg.max_observations_per_step
        )
        prior, x_pair, x_lik, yb, n_replay = replay.mix_into_batch(
            prior, x_pair, x_lik, yb, replay_rng, cfg
        )
        model, opt_state, loss, metrics, final_posterior, grad_norm = train_step(
            model, opt_state, jnp.asarray(prior), jnp.asarray(x_lik), jnp.asarray(yb)
        )
        replay.add_batch(
            np.asarray(jax.device_get(final_posterior)), x_pair, x_lik, yb
        )

        likelihood_noise = np.asarray(x_lik) - np.asarray(x_pair)[:, None]
        likelihood_logpdf = gaussian_likelihood_logpdf_np(
            x_lik, np.asarray(x_pair)[:, None], cfg.likelihood_x_noise_std
        )

        # Collect EVERY optimizer step for dense training diagnostics. Printing can stay sparse.
        host = jax.device_get(metrics)
        values = {
            "step": float(step),
            "energy_score": float(host["energy_score"]),
            "final_energy_score": float(host["final_energy_score"]),
            "attraction": float(host["attraction"]),
            "repulsion": float(host["repulsion"]),
            "mean_mse": float(host["mean_mse"]),
            "final_mean_mse": float(host["final_mean_mse"]),
            "coverage_95": float(host["coverage_95"]),
            "final_coverage_95": float(host["final_coverage_95"]),
            "interval_width": float(host["interval_width"]),
            "posterior_std": float(host["posterior_std"]),
            "grad_norm": float(jax.device_get(grad_norm)),
            "replay_fraction": n_replay / cfg.batch_size,
            "buffer_size": float(len(replay)),
            "likelihood_noise_rms": float(np.sqrt(np.mean(likelihood_noise**2))),
            "likelihood_mean_logpdf": float(np.mean(likelihood_logpdf)),
        }
        for name, value in values.items():
            history[name].append(value)

        if step == 1 or step % cfg.log_every == 0 or step == cfg.epochs:
            print(
                f"step {step:5d}/{cfg.epochs} | ES(all-o) {values['energy_score']:.5f} | "
                f"ES(final-o) {values['final_energy_score']:.5f} | "
                f"MSE(final-o) {values['final_mean_mse']:.5f} | "
                f"replay {n_replay}/{cfg.batch_size} | x-noise-rms {values['likelihood_noise_rms']:.4f}"
            )
    return model, history


#%% 7) Baselines, kept in the same experiment for a strict split comparison
class StandardMLP(eqx.Module):
    layers: tuple[Any, ...]

    def __init__(self, cfg: Config, *, key: Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        h = cfg.nn_hidden_size
        self.layers = (
            eqx.nn.Linear(1, h, key=k1), jax.nn.tanh,
            eqx.nn.Linear(h, h, key=k2), jax.nn.tanh,
            eqx.nn.Linear(h, h, key=k3), jax.nn.tanh,
            eqx.nn.Linear(h, 1, key=k4),
        )

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
    *,
    model: StandardMLP | None = None,
) -> StandardMLP:
    if model is None:
        model = StandardMLP(cfg, key=jax.random.key(cfg.seed))
    opt = optax.adam(cfg.nn_learning_rate)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    x, y = jnp.asarray(x_train[:, None]), jnp.asarray(y_train[:, None])

    @eqx.filter_value_and_grad
    def loss_fn(m, xb, yb):
        return jnp.mean((jax.vmap(m)(xb) - yb) ** 2)

    @eqx.filter_jit
    def step(m, state, xb, yb):
        loss, grads = loss_fn(m, xb, yb)
        updates, state = opt.update(grads, state, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, updates), state, loss

    for epoch in range(cfg.nn_epochs):
        model, opt_state, loss = step(model, opt_state, x, y)
        if epoch % 500 == 0 or epoch == cfg.nn_epochs - 1:
            print(f"MLP epoch {epoch:4d}/{cfg.nn_epochs} | MSE {float(loss):.6f}")
    return model


def predict_mlp(model: StandardMLP, x: np.ndarray) -> np.ndarray:
    return np.asarray(jax.device_get(jax.vmap(model)(jnp.asarray(x[:, None])))).reshape(-1)


def make_gp(cfg: Config) -> GaussianProcessRegressor:
    kernel = (
        C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e1))
        + DotProduct(sigma_0=0.0)
        + WhiteKernel(1e-5, (1e-6, 1e-2))
    )
    return GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=cfg.gp_restarts, random_state=cfg.seed
    )


def fit_gp(
    gp: GaussianProcessRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> GaussianProcessRegressor:
    print("Fitting Gaussian Process...")
    gp.fit(x_train[:, None], y_train)
    print(f"Learned kernel: {gp.kernel_}")
    return gp


def count_eqx_parameters(model: eqx.Module) -> int:
    """Number of scalar array parameters/leaves in an Equinox model."""
    return int(sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(model) if eqx.is_array(leaf)))


def print_model_parameter_counts(
    transport: ConditionalParticleTransport,
    mlp: StandardMLP,
    gp: GaussianProcessRegressor,
    n_train: int,
) -> None:
    """Print all model sizes before the first optimizer/model-fit call."""
    print("\n--- model sizes: before any training ---")
    transport_total = count_eqx_parameters(transport)
    likelihood_params = count_eqx_parameters(transport.likelihood_embedder)
    print(
        f"Particle transport ({transport.conditioning_type}): "
        f"{transport_total:,} trainable scalar parameters"
    )
    print(f"  Likelihood Transformer: {likelihood_params:,}")
    print(f"  Posterior transport    : {transport_total - likelihood_params:,}")
    print(f"Standard MLP: {count_eqx_parameters(mlp):,} trainable scalar parameters")
    print(
        f"Gaussian Process: {int(gp.kernel.n_dims)} trainable kernel hyperparameters; "
        f"non-parametric fit uses {int(n_train):,} training observations"
    )





def plot_training(history: dict[str, list[float]], path: Path, cfg: Config = CFG) -> None:
    """Visualize every collected optimizer step; history is intentionally dense, not log-subsampled."""
    s = np.asarray(history["step"])
    fig, axes = plt.subplots(2, 4, figsize=(22, 9.5))

    axes[0, 0].plot(s, history["energy_score"], label="Energy score (all prefixes)")
    axes[0, 0].plot(s, history["final_energy_score"], label="Energy score (final prefix)", alpha=.75)
    axes[0, 0].plot(s, history["attraction"], label="Attraction", alpha=.7)
    axes[0, 0].plot(s, .5 * np.asarray(history["repulsion"]), label="0.5 repulsion", alpha=.7)
    axes[0, 0].set_title("Proper-score objective"); axes[0, 0].legend()

    axes[0, 1].plot(s, history["mean_mse"], label="all prefixes")
    axes[0, 1].plot(s, history["final_mean_mse"], label="final prefix", alpha=.75)
    axes[0, 1].legend()
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Posterior-mean MSE")

    axes[0, 2].plot(s, history["coverage_95"], label="coverage (all prefixes)")
    axes[0, 2].plot(s, history["final_coverage_95"], label="coverage (final prefix)", alpha=.75)
    axes[0, 2].axhline(.95, ls="--", lw=1, label="target 0.95")
    axes[0, 2].set_ylim(-.02, 1.02)
    axes[0, 2].set_title("95% interval coverage"); axes[0, 2].legend()

    axes[0, 3].plot(s, history["interval_width"], label="interval width")
    axes[0, 3].plot(s, history["posterior_std"], label="posterior std", alpha=.8)
    axes[0, 3].set_title("Posterior spread"); axes[0, 3].legend()

    axes[1, 0].plot(s, history["grad_norm"], label="grad norm")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Optimization")

    buffer_fill = np.asarray(history["buffer_size"]) / max(cfg.historical_output_buffer_capacity, 1)
    axes[1, 1].plot(s, history["replay_fraction"], label="replay fraction")
    axes[1, 1].plot(s, buffer_fill, label="buffer fill fraction", alpha=.8)
    axes[1, 1].set_ylim(-.02, 1.02)
    axes[1, 1].set_title("Historical-prior replay"); axes[1, 1].legend()

    axes[1, 2].plot(s, history["likelihood_noise_rms"], label="realised RMS noise")
    axes[1, 2].axhline(cfg.likelihood_x_noise_std, ls="--", lw=1, label="configured sigma_x")
    axes[1, 2].set_title("Gaussian x-likelihood noise"); axes[1, 2].legend()

    axes[1, 3].plot(s, history["likelihood_mean_logpdf"])
    axes[1, 3].set_title("Mean likelihood log-density")

    for ax in axes.flat:
        ax.grid(alpha=.15)
        ax.set_xlabel("optimizer step")
        ax.tick_params(axis="both", labelsize=14)
    fig.suptitle(f"Per-step transport diagnostics ({cfg.posterior_conditioning})", fontsize=22)
    fig.tight_layout()
    _save_and_show(fig, path, cfg)



#%% 8) Build and train all models ONCE
# Build EVERY model before training so sizes are known before the first optimization/fit call.
transport_init = ConditionalParticleTransport(CFG, scaling, key=jax.random.key(CFG.seed))
nn_init = StandardMLP(CFG, key=jax.random.key(CFG.seed))
gp_init = make_gp(CFG)
print_model_parameter_counts(transport_init, nn_init, gp_init, len(X_train))

print("\n--- training particle transport ---")
transport, history = train_transport(
    X_train, Y_train, scaling, CFG, model=transport_init
)

print("\n--- training standard MLP ONCE ---")
nn_model = train_mlp(X_train, Y_train, CFG, model=nn_init)
nn_train = predict_mlp(nn_model, X_train)
nn_test = predict_mlp(nn_model, X_test)

print("\n--- fitting Gaussian Process ONCE ---")
gp = fit_gp(gp_init, X_train, Y_train)
gp_train, gp_train_std = gp.predict(X_train[:, None], return_std=True)
gp_test, gp_test_std = gp.predict(X_test[:, None], return_std=True)

# Baseline metrics are cached here and reused by every later transport-prior evaluation.
mse = lambda y, p: float(np.mean((y - p) ** 2))
baseline_metrics = {
    "mlp": {"train_mse": mse(Y_train, nn_train), "test_mse": mse(Y_test, nn_test)},
    "gp": {
        "train_mse": mse(Y_train, gp_train),
        "test_mse": mse(Y_test, gp_test),
        "train_coverage_95": float(np.mean(np.abs(Y_train - gp_train) <= 1.96 * gp_train_std)),
        "test_coverage_95": float(np.mean(np.abs(Y_test - gp_test) <= 1.96 * gp_test_std)),
    },
}

with (out / "training_history.json").open("w") as f:
    json.dump(history, f, indent=2)
eqx.tree_serialise_leaves(out / "particle_transport.eqx", transport)
plot_training(history, out / "transport_training_diagnostics.pdf", CFG)

print("\nTraining is complete. Re-run only the evaluation cells below to try new prior modes.")



#%% 9) Functions useful for evaluation and test-time prior experiments

# All settings below are EVALUATION-ONLY. They can be changed after the transport, NN, and GP
# have been trained. Nothing in this dataclass enters transport/NN/GP training.
@dataclass
class EvaluationPriorConfig:
    # GP-informed priors reuse the GP that was already fitted ONCE in cell 8.
    gp_interval_z: float = 1.96

    # Smooth kernel/locality calculations. This is NOT a learned hyperparameter.
    kernel_length_scale: float = 0.15
    kernel_eps: float = 1e-12

    # GP-like kernel-support prior: narrow/local near support, widest/global far from support.
    kernel_support_local_width_fraction: float = 0.50

    # Extremely simple nearest-output prior.
    nearest_fixed_width_fraction: float = 0.50

    # Local-variance / local-linear bounds.
    local_std_multiplier: float = 2.0
    local_linear_k: int = 64

    # Simple k-neighbour quantile/envelope rules.
    knn_quantile_low: float = 0.05
    knn_quantile_high: float = 0.95
    simple_margin: float = 0.25
    simple_min_width: float = 1.00

    # Distance-expanded kNN envelope: add this fraction of the widest prior for each
    # kernel-length-scale of distance from the nearest training x, capped at widest width.
    distance_envelope_growth: float = 0.50

    # Split-conformal-style calibration used only to choose a bound width. No model is trained.
    conformal_alpha: float = 0.05
    conformal_calibration_fraction: float = 0.20

    # New evaluation modes may extend beyond the widest training support. Set True if you want
    # every new mode forcibly clipped back to [CFG.prior_min, CFG.prior_max]. Existing
    # nearest_training keeps using CFG.eval_local_prior_clip_to_global exactly as before.
    clip_new_modes_to_global: bool = False


EVAL_PRIOR_CFG = EvaluationPriorConfig()


AVAILABLE_EVAL_PRIOR_MODES = {
    # Existing modes.
    "cheating": "oracle label centre; diagnostic upper benchmark only",
    "widest": "the same fixed widest support used for fresh training clouds",
    "nearest_training": "kNN min/max y envelope + fixed margin (existing method)",

    # GP-informed modes: reuse the already fitted GP; NEVER refit it here.
    "gp_variance_matched": "uniform with GP predictive mean and the same predictive variance",
    "gp_95": "uniform whose bounds equal GP mean +/- configured z * GP predictive std",

    # GP-like but no GP is used or fitted.
    "kernel_support": "kernel local centre; width and centre revert toward the global prior OOD",
    "nearest_distance": "nearest observed y centre; width grows smoothly with distance to support",
    "kernel_local_variance": "kernel local mean/std plus support-dependent OOD widening",
    "local_linear": "local weighted linear extrapolation plus residual/support-dependent width",
    "conformal_kernel": "split-conformal-style constant residual bound around a kernel centre",
    "kernel_conformal": "conformal in-domain width that widens toward the global prior OOD",

    # Very simple, deliberately non-GP ideas: just construct a plausible interval.
    "train_y_range": "same observed training-y min/max envelope everywhere, plus a margin",
    "nearest_y_fixed": "nearest training y as centre with one fixed width everywhere",
    "knn_quantile": "local kNN y quantiles + margin; robust to extreme neighbours",
    "knn_range_distance": "local kNN min/max envelope that widens with distance from support",
}


def print_evaluation_prior_catalog() -> None:
    print("\nAvailable post-training evaluation-prior modes:")
    for name, description in AVAILABLE_EVAL_PRIOR_MODES.items():
        print(f"  {name:22s} : {description}")


def _as_1d_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _global_prior_geometry(scaling: Scaling) -> tuple[float, float, float]:
    low = float(scaling.prior_low)
    high = float(scaling.prior_high)
    return 0.5 * (low + high), high - low, 0.5 * (high - low)


def _ensure_min_width(
    center: np.ndarray,
    half: np.ndarray | float,
    min_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    center = _as_1d_float(center)
    half = np.broadcast_to(np.asarray(half, dtype=np.float64), center.shape).copy()
    half = np.maximum(half, 0.5 * float(min_width))
    return center - half, center + half


def _clip_bounds_to_global_if_requested(
    low: np.ndarray,
    high: np.ndarray,
    scaling: Scaling,
    *,
    clip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    low = _as_1d_float(low)
    high = _as_1d_float(high)
    if not clip:
        return low, high
    width = high - low
    global_width = float(scaling.prior_high - scaling.prior_low)
    width = np.minimum(width, global_width)
    half = 0.5 * width
    center = 0.5 * (low + high)
    center = np.clip(
        center,
        float(scaling.prior_low) + half,
        float(scaling.prior_high) - half,
    )
    return center - half, center + half


def _nearest_indices_and_distances(
    x_query: np.ndarray,
    x_train: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_query = _as_1d_float(x_query)
    x_train = _as_1d_float(x_train)
    if len(x_train) == 0:
        raise ValueError("At least one training pair is required to construct this prior.")
    k = min(max(int(k), 1), len(x_train))
    distances = np.abs(x_query[:, None] - x_train[None, :])
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    nearest_distance = np.min(distances, axis=1)
    return nearest, nearest_distance


def _support_score_from_distance(distance: np.ndarray, length_scale: float) -> np.ndarray:
    """Simple support score in [0,1]: 1 near training x, ->0 away from support."""
    ell = max(float(length_scale), 1e-12)
    d = np.asarray(distance, dtype=np.float64)
    return np.exp(-0.5 * (d / ell) ** 2)


def _kernel_local_moments(
    x_query: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    length_scale: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian-kernel local mean/std plus absolute support score; no fitting/optimization."""
    xq = _as_1d_float(x_query)
    xt = _as_1d_float(x_train)
    yt = _as_1d_float(y_train)
    ell = max(float(length_scale), 1e-12)
    dist = np.abs(xq[:, None] - xt[None, :])
    w = np.exp(-0.5 * (dist / ell) ** 2)
    support = np.max(w, axis=1)
    wsum = np.sum(w, axis=1, keepdims=True)
    normalized = w / np.maximum(wsum, float(eps))
    mean = np.sum(normalized * yt[None, :], axis=1)
    var = np.sum(normalized * (yt[None, :] - mean[:, None]) ** 2, axis=1)

    # If every weight numerically vanishes, normalized weights carry no information.
    dead = (wsum[:, 0] <= float(eps))
    if np.any(dead):
        mean[dead] = float(np.mean(yt))
        var[dead] = float(np.var(yt))
        support[dead] = 0.0
    return mean, np.sqrt(np.maximum(var, 0.0)), support


def _local_linear_center_residual_std(
    x_query: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    k: int,
    length_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tiny weighted local linear fits, solved independently per query; no learned model."""
    xq = _as_1d_float(x_query)
    xt = _as_1d_float(x_train)
    yt = _as_1d_float(y_train)
    nearest, nearest_distance = _nearest_indices_and_distances(xq, xt, k)
    ell = max(float(length_scale), 1e-12)
    centers = np.empty(len(xq), dtype=np.float64)
    residual_std = np.empty(len(xq), dtype=np.float64)

    for i, (x0, ids) in enumerate(zip(xq, nearest)):
        dx = xt[ids] - x0
        yy = yt[ids]
        w = np.exp(-0.5 * (dx / ell) ** 2)
        # Fit y = a + b (x-x0), so a is the prediction exactly at x0.
        A = np.column_stack([np.ones_like(dx), dx])
        sw = np.sqrt(np.maximum(w, 1e-12))
        coef, *_ = np.linalg.lstsq(A * sw[:, None], yy * sw, rcond=None)
        pred_local = A @ coef
        centers[i] = coef[0]
        residual_std[i] = np.sqrt(
            np.sum(w * (yy - pred_local) ** 2) / max(np.sum(w), 1e-12)
        )

    support = _support_score_from_distance(nearest_distance, ell)
    return centers, residual_std, support


def _split_conformal_kernel_components(
    x_query: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    eval_cfg: EvaluationPriorConfig,
    *,
    seed: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Simple deterministic split-conformal-style kernel bound; no fitted regressor.

    This is useful as an intuitive calibration rule. Its usual exchangeability coverage
    interpretation should NOT be assumed for genuinely out-of-support queries.
    """
    xt = _as_1d_float(x_train)
    yt = _as_1d_float(y_train)
    n = len(xt)
    if n < 5:
        raise ValueError("conformal_kernel requires at least five training observations.")
    frac = float(eval_cfg.conformal_calibration_fraction)
    if not 0.0 < frac < 1.0:
        raise ValueError("conformal_calibration_fraction must lie in (0,1).")
    alpha = float(eval_cfg.conformal_alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("conformal_alpha must lie in (0,1).")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_cal = min(max(int(round(frac * n)), 1), n - 2)
    cal_ids = perm[:n_cal]
    ref_ids = perm[n_cal:]

    cal_center, _, _ = _kernel_local_moments(
        xt[cal_ids], xt[ref_ids], yt[ref_ids],
        eval_cfg.kernel_length_scale, eval_cfg.kernel_eps,
    )
    residuals = np.abs(yt[cal_ids] - cal_center)
    # Finite-sample split-conformal quantile, using the conservative 'higher' order statistic.
    level = min(1.0, np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal)
    try:
        q = float(np.quantile(residuals, level, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        q = float(np.quantile(residuals, level, interpolation="higher"))

    query_center, _, support = _kernel_local_moments(
        x_query, xt[ref_ids], yt[ref_ids],
        eval_cfg.kernel_length_scale, eval_cfg.kernel_eps,
    )
    return query_center, q, support


def evaluation_prior_bounds_np(
    x_query: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    scaling: Scaling,
    cfg: Config,
    *,
    mode: str | None = None,
    y_query_oracle: np.ndarray | None = None,
    gp_model: GaussianProcessRegressor | None = None,
    eval_cfg: EvaluationPriorConfig = EVAL_PRIOR_CFG,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one post-training evaluation y-prior interval per query x.

    No mode here retrains the transport, NN, or GP. ``cheating`` deliberately uses the true
    query label. GP modes reuse the already-fitted ``gp_model``. Every other mode is built
    directly from the closed training set with simple deterministic NumPy calculations.
    """
    mode = cfg.evaluation_prior_mode if mode is None else str(mode)
    if mode not in AVAILABLE_EVAL_PRIOR_MODES:
        raise ValueError(
            f"Unknown evaluation prior mode {mode!r}. Available: {sorted(AVAILABLE_EVAL_PRIOR_MODES)}"
        )

    xq = _as_1d_float(x_query)
    xt = _as_1d_float(x_train)
    yt = _as_1d_float(y_train)
    if len(xt) < 1:
        raise ValueError("At least one training pair is required for evaluation-prior construction.")

    global_center, global_width, global_half = _global_prior_geometry(scaling)
    min_width = float(eval_cfg.simple_min_width)

    # ---- Existing modes: preserved exactly. ----
    if mode == "cheating":
        if y_query_oracle is None:
            raise ValueError(
                "cheating evaluation prior requires y_query_oracle (the known evaluation labels)."
            )
        y_oracle = _as_1d_float(y_query_oracle)
        if len(y_oracle) != len(xq):
            raise ValueError("y_query_oracle must have the same length as x_query.")
        cheating_width = float(cfg.cheating_prior_width_fraction) * global_width
        half = 0.5 * cheating_width
        return (y_oracle - half).astype(np.float32), (y_oracle + half).astype(np.float32)

    if mode == "widest":
        low = np.full(len(xq), scaling.prior_low, dtype=np.float32)
        high = np.full(len(xq), scaling.prior_high, dtype=np.float32)
        return low, high

    if mode == "nearest_training":
        nearest, _ = _nearest_indices_and_distances(xq, xt, cfg.eval_local_prior_k)
        local_y = yt[nearest]
        raw_low = np.min(local_y, axis=1)
        raw_high = np.max(local_y, axis=1)
        center = 0.5 * (raw_low + raw_high)
        half = 0.5 * (raw_high - raw_low) + float(cfg.eval_local_prior_margin)
        half = np.maximum(half, 0.5 * float(cfg.eval_local_prior_min_width))
        low, high = center - half, center + half
        low, high = _clip_bounds_to_global_if_requested(
            low, high, scaling, clip=bool(cfg.eval_local_prior_clip_to_global)
        )
        return low.astype(np.float32), high.astype(np.float32)

    # ---- GP-informed priors: reuse the GP fitted once in cell 8. ----
    if mode in {"gp_variance_matched", "gp_95"}:
        if gp_model is None:
            raise ValueError(f"{mode} requires the already-fitted gp_model; it is not refit here.")
        gp_mean, gp_std = gp_model.predict(xq[:, None], return_std=True)
        if mode == "gp_variance_matched":
            # Uniform(mu-h,mu+h) has variance h^2/3, so h=sqrt(3)*sigma matches GP variance.
            half = np.sqrt(3.0) * gp_std
        else:
            half = float(eval_cfg.gp_interval_z) * gp_std
        low, high = _ensure_min_width(gp_mean, half, min_width)
        low, high = _clip_bounds_to_global_if_requested(
            low, high, scaling, clip=eval_cfg.clip_new_modes_to_global
        )
        return low.astype(np.float32), high.astype(np.float32)

    # Shared locality ingredients for several no-fit modes.
    kernel_mean, kernel_std, support = _kernel_local_moments(
        xq, xt, yt, eval_cfg.kernel_length_scale, eval_cfg.kernel_eps
    )
    _, nearest_distance = _nearest_indices_and_distances(xq, xt, 1)

    if mode == "kernel_support":
        # GP-like qualitative behavior without a GP: locally informed near support, then both
        # centre and width continuously revert to the configured global prior as support vanishes.
        local_width = max(
            float(eval_cfg.kernel_support_local_width_fraction) * global_width,
            min_width,
        )
        center = support * kernel_mean + (1.0 - support) * global_center
        width = support * local_width + (1.0 - support) * global_width
        low, high = center - 0.5 * width, center + 0.5 * width

    elif mode == "nearest_distance":
        # Extremely simple: nearest y gives the centre; only nearest-x distance controls width.
        nearest, nearest_distance = _nearest_indices_and_distances(xq, xt, 1)
        center = yt[nearest[:, 0]]
        s = _support_score_from_distance(nearest_distance, eval_cfg.kernel_length_scale)
        local_width = max(float(eval_cfg.nearest_fixed_width_fraction) * global_width, min_width)
        width = s * local_width + (1.0 - s) * global_width
        low, high = center - 0.5 * width, center + 0.5 * width

    elif mode == "kernel_local_variance":
        # Local disagreement sets the ID width; lack of support smoothly restores global width.
        local_half = np.maximum(
            float(eval_cfg.local_std_multiplier) * kernel_std,
            0.5 * min_width,
        )
        half = support * local_half + (1.0 - support) * global_half
        center = support * kernel_mean + (1.0 - support) * global_center
        low, high = center - half, center + half

    elif mode == "local_linear":
        center_local, residual_std, s = _local_linear_center_residual_std(
            xq, xt, yt,
            k=eval_cfg.local_linear_k,
            length_scale=eval_cfg.kernel_length_scale,
        )
        local_half = np.maximum(
            float(eval_cfg.local_std_multiplier) * residual_std + float(eval_cfg.simple_margin),
            0.5 * min_width,
        )
        # Retain extrapolative center while supported; revert toward global center far OOD.
        center = s * center_local + (1.0 - s) * global_center
        half = s * local_half + (1.0 - s) * global_half
        low, high = center - half, center + half

    elif mode in {"conformal_kernel", "kernel_conformal"}:
        center, conformal_half, s = _split_conformal_kernel_components(
            xq, xt, yt, eval_cfg, seed=cfg.seed + 71_001
        )
        conformal_half = max(float(conformal_half), 0.5 * min_width)
        if mode == "conformal_kernel":
            half = np.full(len(xq), conformal_half, dtype=np.float64)
        else:
            # Same conformal-calibrated ID width, but admit ignorance OOD by widening globally.
            half = s * conformal_half + (1.0 - s) * global_half
            center = s * center + (1.0 - s) * global_center
        low, high = center - half, center + half

    # ---- Deliberately simple non-GP bounds. ----
    elif mode == "train_y_range":
        low0 = float(np.min(yt)) - float(eval_cfg.simple_margin)
        high0 = float(np.max(yt)) + float(eval_cfg.simple_margin)
        center0 = 0.5 * (low0 + high0)
        half0 = max(0.5 * (high0 - low0), 0.5 * min_width)
        low = np.full(len(xq), center0 - half0)
        high = np.full(len(xq), center0 + half0)

    elif mode == "nearest_y_fixed":
        nearest, _ = _nearest_indices_and_distances(xq, xt, 1)
        center = yt[nearest[:, 0]]
        width = max(float(eval_cfg.nearest_fixed_width_fraction) * global_width, min_width)
        low, high = center - 0.5 * width, center + 0.5 * width

    elif mode == "knn_quantile":
        nearest, _ = _nearest_indices_and_distances(xq, xt, cfg.eval_local_prior_k)
        local_y = yt[nearest]
        qlo = float(eval_cfg.knn_quantile_low)
        qhi = float(eval_cfg.knn_quantile_high)
        if not 0.0 <= qlo < qhi <= 1.0:
            raise ValueError("knn quantiles must satisfy 0 <= low < high <= 1.")
        raw_low = np.quantile(local_y, qlo, axis=1) - float(eval_cfg.simple_margin)
        raw_high = np.quantile(local_y, qhi, axis=1) + float(eval_cfg.simple_margin)
        center = 0.5 * (raw_low + raw_high)
        half = np.maximum(0.5 * (raw_high - raw_low), 0.5 * min_width)
        low, high = center - half, center + half

    elif mode == "knn_range_distance":
        nearest, nearest_distance = _nearest_indices_and_distances(xq, xt, cfg.eval_local_prior_k)
        local_y = yt[nearest]
        raw_low = np.min(local_y, axis=1) - float(eval_cfg.simple_margin)
        raw_high = np.max(local_y, axis=1) + float(eval_cfg.simple_margin)
        center = 0.5 * (raw_low + raw_high)
        base_width = np.maximum(raw_high - raw_low, min_width)
        ell = max(float(eval_cfg.kernel_length_scale), 1e-12)
        extra = (
            float(eval_cfg.distance_envelope_growth)
            * global_width
            * (nearest_distance / ell)
        )
        width = np.minimum(base_width + extra, global_width)
        low, high = center - 0.5 * width, center + 0.5 * width

    else:  # Defensive; AVAILABLE_EVAL_PRIOR_MODES already checked above.
        raise RuntimeError(f"Unhandled evaluation prior mode: {mode}")

    low, high = _clip_bounds_to_global_if_requested(
        low, high, scaling, clip=eval_cfg.clip_new_modes_to_global
    )
    return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)


def _prior_bounds_summary(low: np.ndarray, high: np.ndarray) -> str:
    width = np.asarray(high) - np.asarray(low)
    center = 0.5 * (np.asarray(low) + np.asarray(high))
    return (
        f"width min/median/max={width.min():.3f}/{np.median(width):.3f}/{width.max():.3f} | "
        f"center min/median/max={center.min():.3f}/{np.median(center):.3f}/{center.max():.3f}"
    )


def make_deployment_observation_block_np(
    x: np.ndarray,
    cfg: Config,
    *,
    num_observations: int | None = None,
) -> np.ndarray:
    """Build clean deployment likelihood observations; NO Gaussian noise is added here.

    Multiple deployment observations are repeated clean measurements of the observed query x.
    This preserves the earlier deployment rule while letting the trained likelihood Transformer
    consume a configurable observation count.
    """
    n_obs = cfg.test_observations_per_step if num_observations is None else int(num_observations)
    if not (cfg.min_observations_per_step <= n_obs <= cfg.max_observations_per_step):
        raise ValueError(
            "Deployment num_observations must lie in the trained prefix range "
            f"[{cfg.min_observations_per_step}, {cfg.max_observations_per_step}]."
        )
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return np.repeat(x[:, None], n_obs, axis=1).astype(np.float32)


@eqx.filter_jit
def _predict_batch(
    model: ConditionalParticleTransport,
    prior: Array,
    x_observations: Array,
    observation_count: Array,
) -> Array:
    return jax.vmap(lambda p, obs: model(p, obs, observation_count))(prior, x_observations)


def predict_transport_from_bounds(
    model: ConditionalParticleTransport,
    x: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    cfg: Config,
    *,
    seed: int,
    prior_mode: str,
    num_observations: int | None = None,
) -> dict[str, np.ndarray]:
    """Predict from already-built prior bounds with clean deployment x observations.

    Prior construction is intentionally separated from transport evaluation so the entire prior
    range can be inspected BEFORE the model sees the test set. Deployment x is NOT noisified.
    `num_observations` selects how many clean x tokens the likelihood Transformer receives.
    """
    x = np.asarray(x).reshape(-1)
    prior_low = np.asarray(prior_low, dtype=np.float32).reshape(-1)
    prior_high = np.asarray(prior_high, dtype=np.float32).reshape(-1)
    if len(prior_low) != len(x) or len(prior_high) != len(x):
        raise ValueError("prior_low/prior_high must have one bound per query x.")
    if np.any(prior_high <= prior_low):
        raise ValueError("Every evaluation prior interval must have positive width.")

    n_obs = cfg.test_observations_per_step if num_observations is None else int(num_observations)
    x_observations = make_deployment_observation_block_np(
        x, cfg, num_observations=n_obs
    )

    rng = np.random.default_rng(seed)
    clouds: list[np.ndarray] = []
    for start in range(0, len(x), cfg.eval_batch_size):
        xb_obs = x_observations[start:start + cfg.eval_batch_size]
        lo = prior_low[start:start + cfg.eval_batch_size, None]
        hi = prior_high[start:start + cfg.eval_batch_size, None]
        u = rng.uniform(0.0, 1.0, (len(xb_obs), cfg.eval_particles)).astype(np.float32)
        prior = (lo + (hi - lo) * u).astype(np.float32)
        cloud = _predict_batch(
            model,
            jnp.asarray(prior),
            jnp.asarray(xb_obs),
            jnp.asarray(n_obs, dtype=jnp.int32),
        )
        clouds.append(np.asarray(jax.device_get(cloud)))
    cloud = np.concatenate(clouds, axis=0)
    return {
        "cloud": cloud,
        "mean": cloud.mean(axis=1),
        "std": cloud.std(axis=1),
        "low": np.quantile(cloud, cfg.interval_low_q, axis=1),
        "high": np.quantile(cloud, cfg.interval_high_q, axis=1),
        "prior_low": prior_low,
        "prior_high": prior_high,
        "prior_width": prior_high - prior_low,
        "prior_mode": str(prior_mode),
        "num_observations": int(n_obs),
    }


def _energy_score_np_1d(cloud: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact 1-D empirical ES without materialising an [B,N,N] array."""
    attraction = np.mean(np.abs(cloud - y[:, None]), axis=1)
    z = np.sort(cloud, axis=1)
    n = z.shape[1]
    coeff = (2.0 * np.arange(n) - n + 1.0)[None, :]
    pairwise_mean = (2.0 / (n * n)) * np.sum(coeff * z, axis=1)
    return attraction - 0.5 * pairwise_mean


def evaluate_transport(pred: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(np.mean((pred["mean"] - y) ** 2)),
        "mae": float(np.mean(np.abs(pred["mean"] - y))),
        "energy_score": float(np.mean(_energy_score_np_1d(pred["cloud"], y))),
        "coverage_95": float(np.mean((y >= pred["low"]) & (y <= pred["high"]))),
        "mean_interval_width": float(np.mean(pred["high"] - pred["low"])),
        "mean_posterior_std": float(np.mean(pred["std"])),
        "mean_prior_width": float(np.mean(pred["prior_width"])),
        "min_prior_width": float(np.min(pred["prior_width"])),
        "max_prior_width": float(np.max(pred["prior_width"])),
    }


def _true_line(ax, x_train, y_train, x_test, y_test):
    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    ix = np.argsort(x)
    ax.plot(x[ix], y[ix], c="black", linewidth=3, label="True Func.")
    ax.set_ylim(-2.5, 2.5)
    ax.grid(alpha=0.1)
    ax.set_xlabel("x")
    ax.tick_params(axis="both", labelsize=15)


def _sorted_contiguous_groups(x: np.ndarray) -> list[np.ndarray]:
    """Split sorted x values across large support gaps so shaded bands do not bridge them."""
    x = np.asarray(x).reshape(-1)
    order = np.argsort(x)
    if len(order) <= 1:
        return [order]
    xs = x[order]
    dx = np.diff(xs)
    positive = dx[dx > 0]
    if len(positive) == 0:
        return [order]
    threshold = max(8.0 * float(np.median(positive)), 0.04 * float(xs.max() - xs.min()))
    split_at = np.flatnonzero(dx > threshold) + 1
    return [g for g in np.split(order, split_at) if len(g)]


def _plot_shaded_interval(
    ax,
    x: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    label: str,
    color: str,
    alpha: float = 0.28,
) -> None:
    """Draw non-overlapping uncertainty regions instead of dense vertical error bars."""
    first = True
    for group in _sorted_contiguous_groups(x):
        order = group[np.argsort(np.asarray(x)[group])]
        ax.fill_between(
            np.asarray(x)[order], np.asarray(low)[order], np.asarray(high)[order],
            color=color, alpha=alpha, linewidth=0,
            label=label if first else None,
        )
        first = False


def _plot_mean(
    ax,
    x: np.ndarray,
    mean: np.ndarray,
    *,
    label: str,
    color: str,
    lw: float = 2.2,
) -> None:
    """Plot predictive means, split across large x gaps so OOD regions are never bridged."""
    first = True
    for group in _sorted_contiguous_groups(x):
        order = group[np.argsort(np.asarray(x)[group])]
        ax.plot(
            np.asarray(x)[order], np.asarray(mean)[order],
            color=color, linewidth=lw,
            label=label if first else None,
        )
        first = False


def plot_evaluation_prior_ranges_from_bounds(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_prior_low: np.ndarray,
    train_prior_high: np.ndarray,
    x_query: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    prior_mode: str,
    path: Path,
    cfg: Config = CFG,
) -> None:
    """Show COMPLETE in-domain and OOD prior intervals before test-model evaluation."""
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # In-domain/train prior support: dark-green known points and a light-green shared region.
    ax.scatter(
        x_train, y_train, s=22, color="darkgreen", alpha=.55,
        label="known in-domain training pairs", zorder=4,
    )
    _plot_shaded_interval(
        ax, x_train, train_prior_low, train_prior_high,
        label="in-domain prior support", color="lightgreen", alpha=.30,
    )
    _plot_mean(
        ax, x_train, train_prior_low,
        label="in-domain prior lower bound", color="darkgreen", lw=1.15,
    )
    _plot_mean(
        ax, x_train, train_prior_high,
        label="in-domain prior upper bound", color="darkgreen", lw=1.15,
    )

    # OOD/test support: both the left and right OOD regions use the SAME red/light-red colors.
    _plot_shaded_interval(
        ax, x_query, prior_low, prior_high,
        label="OOD test prior support", color="lightcoral", alpha=.30,
    )
    _plot_mean(
        ax, x_query, prior_low,
        label="OOD test prior lower bound", color="darkred", lw=1.25,
    )
    _plot_mean(
        ax, x_query, prior_high,
        label="OOD test prior upper bound", color="darkred", lw=1.25,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y prior support")
    ax.set_title(f"PRIOR INSPECTION BEFORE TEST INFERENCE: {prior_mode}")
    ax.grid(alpha=.15)
    ax.legend(loc="best")
    ax.tick_params(axis="both", labelsize=15)
    fig.tight_layout()
    _save_and_show(fig, path, cfg)


def plot_transport(
    x_train, y_train, x_test, y_test, train_pred, test_pred, path: Path, cfg: Config = CFG
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    _true_line(ax, x_train, y_train, x_test, y_test)

    # Restore the original train/test color convention: train=green, OOD test=red.
    # Shaded regions replace the old dense vertical error bars.
    _plot_shaded_interval(
        ax, x_train, train_pred["low"], train_pred["high"],
        label="95% posterior interval (train)", color="lightgreen", alpha=.28,
    )
    _plot_shaded_interval(
        ax, x_test, test_pred["low"], test_pred["high"],
        label="95% posterior interval (test)", color="lightcoral", alpha=.32,
    )
    _plot_mean(
        ax, x_train, train_pred["mean"],
        label="Posterior mean (train)", color="darkgreen", lw=2.0,
    )
    _plot_mean(
        ax, x_test, test_pred["mean"],
        label="Posterior mean (test)", color="darkred", lw=2.6,
    )

    ax.set_title(
        f"Conditional Particle Transport ({cfg.posterior_conditioning}; test prior={test_pred['prior_mode']}; "
        f"o={test_pred.get('num_observations', '?')})",
        fontsize=21,
    )
    ax.set_ylabel("y")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_and_show(fig, path, cfg)


def plot_comparison(
    x_train, y_train, x_test, y_test,
    nn_train, nn_test,
    gp_train, gp_train_std, gp_test, gp_test_std,
    tr_train, tr_test,
    path: Path,
    cfg: Config = CFG,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), sharey=True)
    for ax in axes:
        _true_line(ax, x_train, y_train, x_test, y_test)
    axes[0].set_ylabel("y")

    # NN has no predictive uncertainty in this baseline; preserve green=train, red=OOD test.
    _plot_mean(axes[0], x_train, nn_train, label="NN mean (train)", color="darkgreen", lw=2.0)
    _plot_mean(axes[0], x_test, nn_test, label="NN mean (test)", color="darkred", lw=2.5)
    axes[0].set_title("Standard MLP")
    axes[0].legend(loc="lower right")

    # GP uncertainty uses light-green train and light-red OOD regions instead of vertical bars.
    _plot_shaded_interval(
        axes[1], x_train,
        gp_train - 1.96 * gp_train_std, gp_train + 1.96 * gp_train_std,
        label="GP 95% interval (train)", color="lightgreen", alpha=.28,
    )
    _plot_shaded_interval(
        axes[1], x_test,
        gp_test - 1.96 * gp_test_std, gp_test + 1.96 * gp_test_std,
        label="GP 95% interval (test)", color="lightcoral", alpha=.32,
    )
    _plot_mean(axes[1], x_train, gp_train, label="GP mean (train)", color="darkgreen", lw=2.0)
    _plot_mean(axes[1], x_test, gp_test, label="GP mean (test)", color="darkred", lw=2.5)
    axes[1].set_title("Gaussian Process")
    axes[1].legend(loc="lower right")

    _plot_shaded_interval(
        axes[2], x_train, tr_train["low"], tr_train["high"],
        label="Transport 95% interval (train)", color="lightgreen", alpha=.28,
    )
    _plot_shaded_interval(
        axes[2], x_test, tr_test["low"], tr_test["high"],
        label="Transport 95% interval (test)", color="lightcoral", alpha=.32,
    )
    _plot_mean(
        axes[2], x_train, tr_train["mean"],
        label="Transport mean (train)", color="darkgreen", lw=2.0,
    )
    _plot_mean(
        axes[2], x_test, tr_test["mean"],
        label="Transport mean (test)", color="darkred", lw=2.5,
    )
    axes[2].set_title(
        f"Particle Transport ({cfg.posterior_conditioning}; test prior={tr_test['prior_mode']}; "
        f"o={tr_test.get('num_observations', '?')})"
    )
    axes[2].legend(loc="lower right")

    fig.suptitle("Cosine regression: identical data split", fontsize=22)
    fig.tight_layout()
    _save_and_show(fig, path, cfg)



#%% 10) Choose / inspect a test-time prior mode AFTER training
# Change ONLY EVAL_MODE (and optionally EVAL_PRIOR_CFG fields), then rerun cells 10 and 11.
# The transport, NN, and GP above are NOT retrained.
EVAL_MODE = CFG.evaluation_prior_mode
EVAL_MODE = "cheating"

# Examples -- all available immediately after training:
# EVAL_MODE = "cheating"               # oracle upper benchmark
# EVAL_MODE = "widest"                 # fixed U(CFG.prior_min, CFG.prior_max)
# EVAL_MODE = "nearest_training"        # existing kNN min/max envelope
# EVAL_MODE = "gp_variance_matched"     # GP mean; uniform has same GP predictive variance
# EVAL_MODE = "gp_95"                   # GP predictive 95%-style bounds
# EVAL_MODE = "kernel_support"          # GP-like fallback without fitting/using a GP
# EVAL_MODE = "nearest_distance"        # nearest y + width grows with x-distance
# EVAL_MODE = "kernel_local_variance"   # local kernel mean/std + OOD widening
# EVAL_MODE = "local_linear"            # local linear extrapolation + residual/OOD width
# EVAL_MODE = "conformal_kernel"        # split-conformal-style fixed residual width
# EVAL_MODE = "kernel_conformal"        # conformal width + support-aware OOD widening
# EVAL_MODE = "train_y_range"           # simplest global observed-y envelope
# EVAL_MODE = "nearest_y_fixed"         # nearest y + one fixed width
# EVAL_MODE = "knn_quantile"            # robust local y quantile envelope
# EVAL_MODE = "knn_range_distance"      # kNN range + explicit distance inflation

print_evaluation_prior_catalog()

_test_oracle = Y_test if EVAL_MODE == "cheating" else None
test_prior_low, test_prior_high = evaluation_prior_bounds_np(
    X_test, X_train, Y_train, scaling, CFG,
    mode=EVAL_MODE,
    y_query_oracle=_test_oracle,
    gp_model=gp,                 # reused only by gp_* modes; NEVER refit here
    eval_cfg=EVAL_PRIOR_CFG,
)

# In-domain/train diagnostics stay on the widest prior, exactly as in the evaluation cell below.
# Compute these bounds here as well so BOTH in-domain and OOD prior supports are visible BEFORE
# the trained transport is ever run on X_test.
train_prior_low, train_prior_high = evaluation_prior_bounds_np(
    X_train, X_train, Y_train, scaling, CFG,
    mode="widest",
    gp_model=gp,
    eval_cfg=EVAL_PRIOR_CFG,
)

print(f"\nSelected evaluation prior mode: {EVAL_MODE}")
print("In-domain/train prior statistics: " + _prior_bounds_summary(train_prior_low, train_prior_high))
print("OOD test prior statistics: " + _prior_bounds_summary(test_prior_low, test_prior_high))

# IMPORTANT: inspect the entire in-domain AND OOD prior support BEFORE running the trained
# transport on X_test. Green is always in-domain; red is always OOD on both left and right.
plot_evaluation_prior_ranges_from_bounds(
    X_train, Y_train,
    train_prior_low, train_prior_high,
    X_test, test_prior_low, test_prior_high,
    EVAL_MODE,
    out / f"evaluation_prior_ranges_{EVAL_MODE}.pdf",
    CFG,
)


#%% 11) Evaluate the already-trained models with the selected prior
# Change this AFTER training to test a different number of likelihood observations.
# No Gaussian noise is added at deployment: the clean x query is repeated this many times.
DEPLOYMENT_OBSERVATIONS = CFG.test_observations_per_step
# DEPLOYMENT_OBSERVATIONS = 1
# DEPLOYMENT_OBSERVATIONS = 3
# DEPLOYMENT_OBSERVATIONS = CFG.max_observations_per_step

# Train-set diagnostics remain on the widest prior to avoid a self-informed local/oracle train prior.
# Those bounds were already constructed and displayed in the prior-inspection cell above.
tr_train = predict_transport_from_bounds(
    transport, X_train, train_prior_low, train_prior_high, CFG,
    seed=CFG.seed + 30_001,
    prior_mode="widest",
    num_observations=DEPLOYMENT_OBSERVATIONS,
)

# The test prior bounds above have already been plotted and inspected before this model call.
tr_test = predict_transport_from_bounds(
    transport, X_test, test_prior_low, test_prior_high, CFG,
    seed=CFG.seed + 30_002,
    prior_mode=EVAL_MODE,
    num_observations=DEPLOYMENT_OBSERVATIONS,
)

tr_train_metrics = evaluate_transport(tr_train, Y_train)
tr_test_metrics = evaluate_transport(tr_test, Y_test)

metrics = {
    "particle_transport": {"train": tr_train_metrics, "test": tr_test_metrics},
    **baseline_metrics,
    "evaluation_prior_mode": EVAL_MODE,
    "deployment_observations": int(DEPLOYMENT_OBSERVATIONS),
    "scaling": asdict(scaling),
    "config": asdict(CFG),
}

print("\n--- same-split comparison; baselines were trained only once ---")
print(f"MLP       train MSE={metrics['mlp']['train_mse']:.6f} test MSE={metrics['mlp']['test_mse']:.6f}")
print(f"GP        train MSE={metrics['gp']['train_mse']:.6f} test MSE={metrics['gp']['test_mse']:.6f}")
print(
    f"Transport train MSE={tr_train_metrics['mse']:.6f} test MSE={tr_test_metrics['mse']:.6f} | "
    f"test ES={tr_test_metrics['energy_score']:.6f} cov95={tr_test_metrics['coverage_95']:.3f} | "
    f"prior={EVAL_MODE} mean-width={tr_test_metrics['mean_prior_width']:.3f} | "
    f"likelihood observations={DEPLOYMENT_OBSERVATIONS}"
)

# Mode-specific filenames preserve previous evaluation experiments instead of overwriting them.
with (out / f"metrics_{EVAL_MODE}.json").open("w") as f:
    json.dump(metrics, f, indent=2)

plot_transport(
    X_train, Y_train, X_test, Y_test, tr_train, tr_test,
    out / f"transport_predictions_uncertainty_{EVAL_MODE}.pdf", CFG,
)
plot_comparison(
    X_train, Y_train, X_test, Y_test,
    nn_train, nn_test, gp_train, gp_train_std, gp_test, gp_test_std,
    tr_train, tr_test,
    out / f"final_comparison_{EVAL_MODE}.pdf", CFG,
)
print(f"Saved evaluation outputs to {out.resolve()}")


# %%
