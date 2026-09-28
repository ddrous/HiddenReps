#%% 0) Configuration — edit here to train or reload; experiments have their own cells below
"""Normal mean inference with known observation variance, in notebook-style #%% cells.

    theta ~ N(prior_mean, prior_std**2), x_i | theta ~ N(theta, noise_std**2).

Run cells in order (no main/CLI). Set Config.train=False to reload the latest complete
Gaussian run, or select checkpoint_dir explicitly. Training settings and architecture are
restored from disk. Each experiment cell owns its settings and writes a NEW result folder,
including settings, raw per-trajectory results and figures; rerun that cell after an edit.
No experiment uses another experiment's output. Loss plots also work after reloading.

The one-step architecture, defaults, V-statistic energy score, all-prefix supervision,
finite simulation budget, acquisition/replay schedule, and truth-anchored interpolation
follow bayes_transport_two_moons_history_buffer.py. Only the scalar input/output and
Gaussian scientific prior change. We train BOTH conditioning types at every update,
with fresh-only, interpolation/no-replay, and history-buffer ablations. Shared keys pair
initial observation/particle projections, data, minibatches, modes, fresh particles and
dropout. Different block architectures necessarily have different parameters/counts.
Proposal acquisition, when enabled, uses a symmetric pooled reference from both buffered
models, with the original discrete importance correction. No closed-form posterior enters
training, acquisition, or neural inference; it is used only for diagnostics/controls.

Sequential propagation is a TRANSFER TEST: same-prefix replay and truth interpolation
are not supervision for Bayes updates under arbitrary incoming priors. Neither exact
composition nor a convergence rate is guaranteed. The causal encoder is not invariant to
observation order. Comparisons measure those failures rather than assuming them away.
The empirical energy-score V statistic also has finite-particle bias, as in the source.

Theory: V_n=(prior_std**-2+n/noise_std**2)**-1; prior-predictive posterior-mean
MSE = V_n, and posterior SD ~ n**-1/2. Those rates concern contraction to theta,
NOT neural approximation error to the exact posterior. We plot the two separately.
Exact empirical-to-Normal W2 integrates quantiles (no sampled reference error), but an
empirical cloud still has discretization/sampling error; paired oracle clouds show it.
Bootstrap intervals resample held-out trajectories, preserving model/seed pairing.
With one training seed they are conditional on that fit; use multiple training_seeds
for separate seed-level variability (reported, not pooled as independent trajectories).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any
import csv
import json
import math
import os
import shutil
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.special import ndtr, ndtri
from scipy.spatial import cKDTree


import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "DejaVu Sans",
    "axes.titlepad": 8.0,
    "axes.labelpad": 6.0,
})

print = partial(print, flush=True)
Array = jax.Array



@dataclass
class Config:
    train: bool = True
    checkpoint_dir: str | None = None  # Run directory OR a particular checkpoint_* directory.
    output_dir: str = f"bayesian_transport/runs/gaussian_{datetime.now():%Y-%m-%d_%H-%M-%S-%f}"
    training_seeds: tuple[int, ...] = (2032,)  # e.g. (2032, 2033, 2034) for independent fits.
    prior_mean: float = 0.0
    prior_std: float = 1.0
    noise_std: float = 1.0
    observed_x: float = 0.0  # Acquisition target only; NEVER a held-out observation.
    batch_size: int = 64
    max_training_particles: int = 32
    variable_training_particles: bool = False
    hidden_dim: int = 256
    heads: int = 4
    mlp_ratio: int = 4
    posterior_depth: int = 4
    max_normalized_displacement: float = 6.0
    attention_dropout_rate: float = 0.0
    max_training_observations: int = 8
    observation_sequence_depth: int = 4
    likelihood_hidden_dim: int = 64
    likelihood_heads: int = 4
    likelihood_mlp_ratio: int = 4
    likelihood_depth: int = 4
    normalize_observations: bool = True
    observation_scale: float = 1.0
    simulation_budget: int = 10_000  # Individual scalar observations, NOT rows or prefixes.
    replay_epochs: int = 100
    learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip_norm: float = 5000.0
    log_every: int = 1250
    categorical_proposal_enabled: bool = True
    importance_weights_enabled: bool = True
    categorical_proposal_warmup_steps: int = 8
    categorical_proposal_refresh_every: int = 25
    categorical_proposal_candidate_particles: int = 1024
    categorical_proposal_reference_particles: int = 1024
    categorical_proposal_defensive_epsilon: float = .10
    categorical_proposal_knn: int = 16
    categorical_proposal_bandwidth_scale: float = 1.0
    categorical_proposal_min_bandwidth: float = .03
    prior_interpolation_probability: float = .25
    historical_output_prior_probability: float = .25
    prior_interpolation_tau_min: float = .05
    prior_interpolation_tau_max: float = 1.15
    truth_anchor_probability: float = 1.0


def validate_config(cfg):
    for name in ("batch_size", "hidden_dim", "heads", "mlp_ratio", "posterior_depth",
                 "max_training_observations", "observation_sequence_depth", "likelihood_hidden_dim",
                 "likelihood_heads", "likelihood_mlp_ratio", "likelihood_depth", "simulation_budget",
                 "log_every", "categorical_proposal_refresh_every", "categorical_proposal_knn"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("max_training_particles", "categorical_proposal_candidate_particles",
                 "categorical_proposal_reference_particles"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    for name in ("replay_epochs", "categorical_proposal_warmup_steps"):
        if not isinstance(getattr(cfg, name), int) or getattr(cfg, name) < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("prior_std", "noise_std", "observation_scale", "learning_rate", "grad_clip_norm",
                 "max_normalized_displacement", "categorical_proposal_bandwidth_scale",
                 "categorical_proposal_min_bandwidth"):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if cfg.hidden_dim % cfg.heads or cfg.likelihood_hidden_dim % cfg.likelihood_heads:
        raise ValueError("Transformer widths must be divisible by their head counts")
    if not cfg.training_seeds or len(set(cfg.training_seeds)) != len(cfg.training_seeds):
        raise ValueError("Provide distinct training_seeds")
    if any(not isinstance(s, int) or not 0 <= s < 2**32 - 100000 for s in cfg.training_seeds):
        raise ValueError("Seeds must be nonnegative integers smaller than 2**32 - 100000")
    for name in ("prior_interpolation_probability", "historical_output_prior_probability",
                 "truth_anchor_probability"):
        if not 0 <= getattr(cfg, name) <= 1:
            raise ValueError(f"{name} must lie in [0,1]")
    if cfg.prior_interpolation_probability + cfg.historical_output_prior_probability > 1:
        raise ValueError("Input-source probabilities must sum to <= 1")
    if not 0 <= cfg.attention_dropout_rate < 1 or not 0 < cfg.categorical_proposal_defensive_epsilon <= 1:
        raise ValueError("Invalid dropout or proposal defensive probability")
    if not 0 <= cfg.prior_interpolation_tau_min <= cfg.prior_interpolation_tau_max < float("inf"):
        raise ValueError("Invalid interpolation interval")
    if not all(np.isfinite(v) for v in (cfg.prior_mean, cfg.observed_x, cfg.weight_decay)) or cfg.weight_decay < 0:
        raise ValueError("Location/weight-decay settings must be finite; weight decay nonnegative")


#%% 1) Gaussian conjugacy and simulator — diagnostic posterior never supplies training labels

def sample_prior(rng, shape, cfg):
    return rng.normal(cfg.prior_mean, cfg.prior_std, size=shape).astype(np.float32)


def exact_posterior(observations, cfg, *, prior_mean=None, prior_variance=None):
    """Last axis is observations; empty sequences recover the prior. Float64 diagnostics."""
    x = np.asarray(observations, dtype=np.float64)
    mean = cfg.prior_mean if prior_mean is None else np.asarray(prior_mean)
    variance = cfg.prior_std**2 if prior_variance is None else np.asarray(prior_variance)
    if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
        raise ValueError("Prior variance must be finite and positive")
    precision = 1 / variance + x.shape[-1] / cfg.noise_std**2
    return (mean / variance + x.sum(axis=-1) / cfg.noise_std**2) / precision, 1 / precision


def exact_prefixes(x, cfg):
    x = np.asarray(x, dtype=np.float64)
    n = np.arange(1, x.shape[-1] + 1)
    variance = 1 / (cfg.prior_std**-2 + n / cfg.noise_std**2)
    mean = variance * (cfg.prior_mean / cfg.prior_std**2 + np.cumsum(x, axis=-1) / cfg.noise_std**2)
    return mean, np.broadcast_to(variance, mean.shape)


#%% 2) Same Transformer blocks as the two-moons one-step experiment; scalar input/output
def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]


class ObservationBlock(eqx.Module):
    """Self-attention block over the labelled observation-coordinate token."""

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

    def __call__(self, tokens: Array, *, key: Array | None = None, inference: bool = False,
                 mask: Array | None = None) -> Array:
        h = _layernorm_tokens(self.norm1, tokens)
        tokens = tokens + self.attention(h, h, h, mask=mask, key=key, inference=inference)
        h = _layernorm_tokens(self.norm2, tokens)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        return tokens + _linear_tokens(self.ff_out, h)


class GaussianObservationEmbedder(eqx.Module):
    """Encode scalar x as one labelled token [value, coordinate id]."""

    input_projection: eqx.nn.Linear
    blocks: tuple[ObservationBlock, ...]
    final_norm: eqx.nn.LayerNorm
    normalize: bool = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(self, cfg: Config, *, key: Array):
        keys = jax.random.split(key, cfg.likelihood_depth + 1)
        self.input_projection = eqx.nn.Linear(2, cfg.likelihood_hidden_dim, key=keys[0])
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
        if x.shape != (1,):
            raise ValueError("Each model call requires exactly one observation with shape (1,).")
        if self.normalize:
            x = x / self.scale
        coord_id = jnp.eye(1, dtype=x.dtype)
        token_features = jnp.concatenate([x[:, None], coord_id], axis=-1)  # [1,2]
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


class CausalObservationSequenceEmbedder(eqx.Module):
    """One causal pass yields a fixed-width summary of every iid observation prefix.

    No positional embedding is needed for iid data; the explicit prefix count distinguishes
    repeated identical measurements. Causality does not guarantee exact set permutation invariance.
    """

    count_projection: eqx.nn.Linear
    blocks: tuple[ObservationBlock, ...]
    final_norm: eqx.nn.LayerNorm

    def __init__(self, cfg: Config, *, key: Array):
        keys = jax.random.split(key, cfg.observation_sequence_depth + 1)
        self.count_projection = eqx.nn.Linear(1, cfg.likelihood_hidden_dim, key=keys[0])
        self.blocks = tuple(ObservationBlock(
            cfg.likelihood_hidden_dim, cfg.likelihood_heads,
            cfg.likelihood_mlp_ratio * cfg.likelihood_hidden_dim,
            cfg.attention_dropout_rate, key=keys[i + 1])
            for i in range(cfg.observation_sequence_depth))
        self.final_norm = eqx.nn.LayerNorm(cfg.likelihood_hidden_dim)

    def __call__(self, tokens: Array, *, key: Array | None = None,
                 inference: bool = False) -> Array:
        counts = jnp.arange(1, len(tokens) + 1, dtype=tokens.dtype)
        tokens = tokens + _linear_tokens(self.count_projection, jnp.log(counts)[:, None])
        causal_mask = jnp.arange(len(tokens))[:, None] >= jnp.arange(len(tokens))[None, :]
        keys = None if key is None else jax.random.split(key, len(self.blocks))
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, mask=causal_mask, key=None if keys is None else keys[i],
                           inference=inference)
        return _layernorm_tokens(self.final_norm, tokens)


class ConditionalParticleTransport(eqx.Module):
    """Identity-initialized particle transport conditioned on a cumulative observation summary."""

    observation_embedder: GaussianObservationEmbedder
    observation_sequence_embedder: CausalObservationSequenceEmbedder
    particle_in: eqx.nn.Linear
    blocks: tuple[Any, ...]
    final_norm: eqx.nn.LayerNorm
    displacement_head: eqx.nn.Linear

    conditioning_type: str = eqx.field(static=True)
    max_displacement: float = eqx.field(static=True)
    prior_center: float = eqx.field(static=True)
    prior_std: float = eqx.field(static=True)

    def __init__(self, cfg: Config, conditioning: str, *, key: Array):
        if conditioning not in {"adaln", "cross_attention"}:
            raise ValueError("Unknown conditioning type")
        keys = jax.random.split(key, cfg.posterior_depth + 4)
        self.observation_embedder = GaussianObservationEmbedder(cfg, key=keys[0])
        self.observation_sequence_embedder = CausalObservationSequenceEmbedder(
            cfg, key=jax.random.fold_in(key, 9187))
        self.particle_in = eqx.nn.Linear(1, cfg.hidden_dim, key=keys[1])

        block_cls = AdaLNParticleBlock if conditioning == "adaln" else CrossAttentionParticleBlock
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
        head = eqx.nn.Linear(cfg.hidden_dim, 1, key=keys[-1])
        # Exact identity transport at initialization.
        head = eqx.tree_at(lambda l: l.weight, head, jnp.zeros_like(head.weight))
        head = eqx.tree_at(lambda l: l.bias, head, jnp.zeros_like(head.bias))
        self.displacement_head = head

        self.conditioning_type = str(conditioning)
        self.max_displacement = float(cfg.max_normalized_displacement)
        self.prior_center = float(cfg.prior_mean)
        self.prior_std = float(cfg.prior_std)

    def _standardize(self, theta: Array) -> Array:
        return (theta - self.prior_center) / self.prior_std

    def _unstandardize(self, z: Array) -> Array:
        return self.prior_center + self.prior_std * z

    def observation_contexts(self, observations: Array, *, key: Array | None = None,
                             inference: bool = False) -> Array:
        if observations.ndim != 2 or observations.shape[-1] != 1 or len(observations) < 1:
            raise ValueError("Expected one or more observations with shape [O,1].")
        keys = None if key is None else jax.random.split(key, len(observations) + 1)
        if keys is None:
            encoded = jax.vmap(lambda x: self.observation_embedder(x, inference=inference))(observations)
        else:
            encoded = jax.vmap(lambda x, k: self.observation_embedder(
                x, key=k, inference=inference))(observations, keys[:-1])
        return self.observation_sequence_embedder(
            jnp.mean(encoded, axis=1), key=None if keys is None else keys[-1], inference=inference)

    def predict_prefixes(self, prior_theta: Array, observations: Array, *,
                         key: Array | None = None, inference: bool = False) -> Array:
        """Direct transports for all prefixes [O,N,1]; no output feeds the next prefix.

        A common [N,1] cloud is broadcast, or [O,N,1] supplies each prefix's own replay cloud.
        """
        obs_key, transport_key = (None, None) if key is None else jax.random.split(key)
        contexts = self.observation_contexts(observations, key=obs_key, inference=inference)
        if prior_theta.ndim == 2:
            prior_theta = jnp.broadcast_to(prior_theta, (len(contexts),) + prior_theta.shape)
        if prior_theta.ndim != 3 or prior_theta.shape[0] != len(contexts):
            raise ValueError("Expected a shared [N,1] cloud or one [O,N,1] cloud per prefix.")
        if transport_key is None:
            return jax.vmap(lambda p, c: self._transport(p, c, inference=inference))(prior_theta, contexts)
        keys = jax.random.split(transport_key, len(contexts))
        return jax.vmap(lambda p, c, k: self._transport(
            p, c, key=k, inference=inference))(prior_theta, contexts, keys)

    def __call__(self, prior_theta: Array, x: Array, *, key: Array | None = None,
                 inference: bool = False) -> Array:
        # Existing evaluation calls stay single-observation; blocks are also accepted explicitly.
        observations = x[None, :] if x.ndim == 1 else x
        obs_key, transport_key = (None, None) if key is None else jax.random.split(key)
        context = self.observation_contexts(observations, key=obs_key, inference=inference)[-1]
        return self._transport(prior_theta, context, key=transport_key, inference=inference)

    def _transport(self, prior_theta: Array, conditioning: Array, *, key: Array | None = None,
                   inference: bool = False) -> Array:
        """Return the terminal cloud; all acquisition/training/evaluation use this dispatch."""
        delta = self._residual(prior_theta, conditioning, key=key, inference=inference)
        return self._unstandardize(self._standardize(prior_theta) + delta)

    def _residual(self, theta: Array, conditioning: Array, *,
                  key: Array | None = None, inference: bool = False) -> Array:
        """Bounded normalized displacement, without the identity skip."""
        memory = conditioning[None, :]
        particles = _linear_tokens(self.particle_in, self._standardize(theta))
        block_keys = None if key is None else jax.random.split(key, len(self.blocks))
        for i, block in enumerate(self.blocks):
            block_key = None if block_keys is None else block_keys[i]
            context = conditioning if self.conditioning_type == "adaln" else memory
            particles = block(particles, context, key=block_key, inference=inference)
        particles = _layernorm_tokens(self.final_norm, particles)
        return self.max_displacement * jnp.tanh(_linear_tokens(self.displacement_head, particles))


#%% 3) Matched replay buffers, acquisition, and proper-score optimization

def model_specs():
    return {f"{conditioning}__{variant}": (conditioning, variant)
            for conditioning in ("adaln", "cross_attention")
            for variant in ("fresh", "no_replay", "buffered")}


class SimulationBuffer:
    """One theta/block/weight per row, independent output storage for every model AND prefix."""
    def __init__(self, cfg, names):
        rows = math.ceil(cfg.simulation_budget / cfg.max_training_observations)
        self.theta = np.zeros((rows, 1), np.float32)
        self.x = np.zeros((rows, cfg.max_training_observations, 1), np.float32)
        self.weights = np.ones(rows, np.float32)
        self.counts = np.zeros(rows, np.int32)
        self.clouds = {name: np.zeros((rows, cfg.max_training_observations,
                                       cfg.max_training_particles, 1), np.float32) for name in names}
        self.cloud_counts = {name: np.zeros(rows, np.int32) for name in names}
        self.size = 0

    def add(self, theta, x, weights, counts):
        ids = np.arange(self.size, self.size + len(theta))
        if len(ids) == 0 or ids[-1] >= len(self.theta):
            raise ValueError("Empty batch or exhausted simulation buffer")
        if x.shape != (len(ids), self.x.shape[1], 1) or np.any(counts < 1) or np.any(counts > self.x.shape[1]):
            raise ValueError("Invalid padded observation block/counts")
        self.theta[ids], self.x[ids], self.weights[ids], self.counts[ids] = theta, x, weights, counts
        self.size += len(ids)
        return ids

    def update(self, name, ids, clouds):
        clouds = np.asarray(clouds, dtype=np.float32)
        n = clouds.shape[2]
        if clouds.shape != (len(ids), self.x.shape[1], n, 1) or not 2 <= n <= self.clouds[name].shape[2]:
            raise ValueError("Invalid posterior shape/count")
        if not np.all(np.isfinite(clouds)):
            raise FloatingPointError(f"Nonfinite training cloud for {name}")
        valid = np.arange(self.x.shape[1])[None, :] < self.counts[ids, None]
        self.clouds[name][ids] = 0
        self.clouds[name][ids, :, :n] = np.where(valid[:, :, None, None], clouds, 0)
        self.cloud_counts[name][ids] = n

    def save(self, path):
        np.savez_compressed(path, theta=self.theta[:self.size], x=self.x[:self.size],
                            weights=self.weights[:self.size], counts=self.counts[:self.size],
                            **{f"clouds_{k}": v[:self.size] for k, v in self.clouds.items()},
                            **{f"cloud_counts_{k}": v[:self.size] for k, v in self.cloud_counts.items()})


def training_inputs(buffer, ids, name, variant, particles, cfg, seed):
    # A per-step common seed, with unconditional draws, prevents divergent replay histories
    # or ablation branches from desynchronizing the fresh particles and source choices.
    rng = np.random.default_rng(seed)
    b, o = len(ids), buffer.x.shape[1]
    fresh = sample_prior(rng, (b, particles, 1), cfg)
    base = sample_prior(rng, (b, particles, 1), cfg)
    anchors = sample_prior(rng, (b, 1), cfg)
    anchors = np.where((rng.random(b) < cfg.truth_anchor_probability)[:, None], buffer.theta[ids], anchors)
    tau = rng.uniform(cfg.prior_interpolation_tau_min, cfg.prior_interpolation_tau_max, b)
    interpolated = (1 - tau[:, None, None]) * base + tau[:, None, None] * anchors[:, None]
    u = rng.random(b)
    use_interp = (u < cfg.prior_interpolation_probability) & (variant != "fresh")
    use_replay = ((u >= cfg.prior_interpolation_probability)
                  & (u < cfg.prior_interpolation_probability + cfg.historical_output_prior_probability)
                  & (variant == "buffered") & (buffer.cloud_counts[name][ids] > 0))
    incoming = np.repeat(np.where(use_interp[:, None, None], interpolated, fresh)[:, None], o, axis=1)
    # Separate row keys keep the resampling paired across conditioning types.
    for row in np.flatnonzero(use_replay):
        old_n = int(buffer.cloud_counts[name][ids[row]])
        indices = np.random.default_rng(np.random.SeedSequence([seed, int(row), 91])).choice(
            old_n, particles, replace=particles > old_n)
        incoming[row] = buffer.clouds[name][ids[row]][:, indices]
    return incoming.astype(np.float32), {
        "interpolation_fraction": float(use_interp.mean()), "buffer_fraction": float(use_replay.mean()),
        "fresh_fraction": float((~(use_interp | use_replay)).mean())}


def proposal_acquisition(rng, reference, batch_size, cfg):
    """Defensive categorical acquisition with exact conditional (1/K)/alpha correction."""
    candidates = sample_prior(rng, (cfg.categorical_proposal_candidate_particles, 1), cfg)
    k = min(cfg.categorical_proposal_knn, len(reference))
    distances, _ = cKDTree(reference).query(candidates, k=k)
    distances = np.asarray(distances).reshape(len(candidates), k)
    bandwidth = max(cfg.categorical_proposal_min_bandwidth,
                    cfg.categorical_proposal_bandwidth_scale * float(np.std(reference, ddof=1))
                    * len(reference)**(-1 / 5))  # Scott rule in ONE dimension.
    affinity = np.exp(-.5 * (distances / bandwidth)**2).mean(axis=1)
    focused = affinity / affinity.sum() if affinity.sum() > 0 else np.full(len(candidates), 1 / len(candidates))
    eps = cfg.categorical_proposal_defensive_epsilon
    probabilities = (1 - eps) * focused + eps / len(candidates)
    probabilities /= probabilities.sum()
    ids = rng.choice(len(candidates), batch_size, p=probabilities)
    weights = (1 / len(candidates)) / probabilities[ids]
    if not cfg.importance_weights_enabled:
        weights[:] = 1
    return candidates[ids], weights.astype(np.float32)


def energy_score(cloud, theta):
    # Same stabilized V statistic as the reference; in 1D this is empirical CRPS.
    attraction = jnp.mean(jnp.sqrt((cloud - theta)**2 + 1e-12))
    repulsion = jnp.mean(jnp.sqrt((cloud[:, None] - cloud[None, :])**2 + 1e-12))
    return attraction - .5 * repulsion


def transport_objective(model, incoming, observations, theta, weights, counts, key):
    keys = jax.random.split(key, len(theta))
    posterior = jax.vmap(lambda p, x, k: model.predict_prefixes(p, x, key=k))(
        incoming, observations, keys)
    scores = jax.vmap(jax.vmap(energy_score, in_axes=(0, None)))(posterior, theta)
    valid = jnp.arange(observations.shape[1])[None, :] < counts[:, None]
    loss = jnp.mean(weights * jnp.sum(jnp.where(valid, scores, 0), axis=1) / counts)
    prefix_scores = jnp.sum(jnp.where(valid, weights[:, None] * scores, 0), axis=0) / jnp.maximum(valid.sum(axis=0), 1)
    return loss, (posterior, prefix_scores)


def make_train_step(optimizer):
    @eqx.filter_jit
    def step(model, state, incoming, x, theta, weights, counts, key):
        (loss, aux), grads = eqx.filter_value_and_grad(transport_objective, has_aux=True)(
            model, incoming, x, theta, weights, counts, key)
        updates, state = optimizer.update(grads, state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), state, loss, aux, optax.global_norm(grads)
    return step


@eqx.filter_jit
def evaluate_cloud(model, prior, observations):
    return model(prior, observations, inference=True)


def particle_choices(cfg):
    if not cfg.variable_training_particles:
        return (cfg.max_training_particles,)
    return tuple(sorted({max(2, round(cfg.max_training_particles * r)) for r in (1/16, 1/8, 1/4, 1/2, 1)}))


#%% 4) Persistence and training helpers — atomic complete sets, saved loss/data/configuration

def write_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_schedule(total):
    count = min(total, 10)
    return tuple((i * total + count - 1) // count for i in range(1, count + 1))


def checkpoint_location(root, expected_names):
    candidates = [root] if root.name.startswith("checkpoint_") else list(root.glob("checkpoint_*"))
    complete = []
    for folder in candidates:
        manifest = folder / "complete.json"
        if not manifest.is_file():
            continue
        try:
            info = json.loads(manifest.read_text())
            required = [f"{n}.eqx" for n in expected_names] + ["training_history.csv", "run_config.json", "buffer.npz"]
            if info["models"] == list(expected_names) and all((folder / f).is_file() for f in required):
                complete.append((int(info["step"]), folder))
        except (ValueError, KeyError, TypeError):
            continue
    if not complete:
        raise FileNotFoundError(f"No complete Gaussian checkpoint set in {root}")
    return max(complete)[1]


def matched_checkpoint_locations(out, cfg):
    """Load the latest update complete for ALL seeds; interrupted runs cannot mix budgets."""
    if cfg.checkpoint_dir:
        requested = Path(cfg.checkpoint_dir).expanduser().resolve()
        if requested.name.startswith("checkpoint_"):
            if len(cfg.training_seeds) != 1:
                raise ValueError("Select a run directory for multi-seed comparisons")
            seed = cfg.training_seeds[0]
            if requested.parent != out / f"seed_{seed}":
                raise ValueError("Selected checkpoint does not belong to the saved seed")
            return {seed: checkpoint_location(requested, model_specs())}
    available = {}
    for seed in cfg.training_seeds:
        found = {}
        for candidate in (out / f"seed_{seed}").glob("checkpoint_*"):
            try:
                folder = checkpoint_location(candidate, model_specs())
                info = json.loads((folder / "complete.json").read_text())
                found[int(info["step"])] = folder
            except (FileNotFoundError, ValueError, KeyError):
                continue
        available[seed] = found
    common = set.intersection(*(set(paths) for paths in available.values()))
    if not common:
        raise FileNotFoundError(f"No common completed update across training seeds in {out}")
    step = max(common)
    return {seed: paths[step] for seed, paths in available.items()}


def save_checkpoint(root, step, models, rows, buffer, cfg, progress):
    # Publish the directory only when all six model files, metrics and buffer are present.
    target = root / f"checkpoint_{step:09d}"
    temporary = root / f".checkpoint_{step:09d}.tmp"
    if target.exists() or temporary.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    temporary.mkdir()
    for name, model in models.items():
        eqx.tree_serialise_leaves(temporary / f"{name}.eqx", model)
    write_rows(temporary / "training_history.csv", rows)
    buffer.save(temporary / "buffer.npz")
    (temporary / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (temporary / "complete.json").write_text(json.dumps({"step": step, "models": list(models), **progress}, indent=2))
    os.replace(temporary, target)
    write_rows(root / "training_history.csv", rows)


def script_location():
    name = "bayes_transport_gaussian_conjugate_history_buffer.py"
    candidates = [Path(globals().get("__file__", name)), Path.cwd() / name,
                  Path.cwd() / "bayesian_transport" / name]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Run notebook cells from the project root or bayesian_transport directory")


def setup_run(requested, script):
    base = script.parent if script.parent.name == "bayesian_transport" else Path.cwd() / "bayesian_transport"
    if requested.train:
        cfg = requested
        out = Path(cfg.output_dir).expanduser()
        if not out.is_absolute():
            out = (base.parent / out).resolve()
        validate_config(cfg)
        out.mkdir(parents=True, exist_ok=False)
        cfg = replace(cfg, output_dir=str(out))
        (out / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))
        shutil.copy2(script, out / script.name)
    else:
        if requested.checkpoint_dir:
            out = Path(requested.checkpoint_dir).expanduser().resolve()
            if out.name.startswith("checkpoint_"):
                out = out.parent.parent  # checkpoint -> seed -> run
            elif out.name.startswith("seed_"):
                out = out.parent
        else:
            # Search newest runs, skipping any whose seed sets are not yet complete.
            out = None
            candidates = ([script.parent] if (script.parent / "run_config.json").is_file() else [])
            candidates += sorted((base / "runs").glob("gaussian_*"), reverse=True)
            for candidate in candidates:
                try:
                    saved = Config(**json.loads((candidate / "run_config.json").read_text()))
                    matched_checkpoint_locations(candidate, replace(saved, checkpoint_dir=None))
                    out = candidate
                    break
                except (FileNotFoundError, ValueError, TypeError):
                    continue
            if out is None:
                raise FileNotFoundError("No completed Gaussian run; train first or set checkpoint_dir")
        cfg = Config(**json.loads((out / "run_config.json").read_text()))
        cfg = replace(cfg, train=False, checkpoint_dir=requested.checkpoint_dir, output_dir=str(out))
        validate_config(cfg)
    return cfg, out


def train_or_load(cfg, out):
    fitted, histories, checkpoint_paths = {}, [], {}
    specs = model_specs()
    load_paths = matched_checkpoint_locations(out, cfg) if not cfg.train else {}
    for seed in cfg.training_seeds:
        root = out / f"seed_{seed}"
        models = {}
        for conditioning in ("adaln", "cross_attention"):
            initial = ConditionalParticleTransport(cfg, conditioning, key=jax.random.key(seed))
            for name, (kind, _) in specs.items():
                if kind == conditioning:
                    models[name] = initial  # Immutable trees; updates create independent parameters.
        counts = {name: sum(a.size for a in jax.tree_util.tree_leaves(model) if eqx.is_inexact_array(a))
                  for name, model in models.items()}
        print(f"Seed {seed}; parameter counts: {counts}")
        if not cfg.train:
            folder = load_paths[seed]
            # Config beside checkpoint protects architecture if a run-level config was edited.
            saved = Config(**json.loads((folder / "run_config.json").read_text()))
            fields = ("train", "checkpoint_dir", "output_dir")
            if any(getattr(saved, k) != getattr(cfg, k) for k in asdict(cfg) if k not in fields):
                raise ValueError("Run and checkpoint training configurations disagree")
            models = {name: eqx.tree_deserialise_leaves(folder / f"{name}.eqx", model) for name, model in models.items()}
            with (folder / "training_history.csv").open() as stream:
                rows = [{k: v if k == "model" else float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
            print(f"Loaded {folder}")
        else:
            root.mkdir()
            (root / "parameter_counts.json").write_text(json.dumps(counts, indent=2))
            optimizer = optax.chain(optax.clip_by_global_norm(cfg.grad_clip_norm),
                                    optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay))
            states = {name: optimizer.init(eqx.filter(model, eqx.is_array)) for name, model in models.items()}
            train_step = make_train_step(optimizer)
            buffer = SimulationBuffer(cfg, specs)
            batches = math.ceil(len(buffer.theta) / cfg.batch_size)
            total = batches * (1 + cfg.replay_epochs)
            schedule = checkpoint_schedule(total)
            (root / "checkpoint_schedule.json").write_text(json.dumps({"steps": schedule, "total": total}, indent=2))
            rng = np.random.default_rng(seed + 10001)
            shuffle = np.random.default_rng(seed + 15013)
            proposal_rng = np.random.default_rng(seed + 27011)
            inputs_rng = np.random.default_rng(seed + 31001)
            rows, simulations, reference = [], 0, None
            model_seconds = {name: 0.0 for name in models}
            start_time = perf_counter()
            for step in range(1, total + 1):
                particles = int(inputs_rng.choice(particle_choices(cfg)))
                cloud_seed = int(inputs_rng.integers(0, 2**32))
                epoch = 0
                if buffer.size < len(buffer.theta):
                    b = min(cfg.batch_size, len(buffer.theta) - buffer.size)
                    obs_counts = np.minimum(cfg.max_training_observations,
                                            cfg.simulation_budget - simulations - np.arange(b) * cfg.max_training_observations)
                    if cfg.categorical_proposal_enabled and step > cfg.categorical_proposal_warmup_steps:
                        if reference is None or (step - cfg.categorical_proposal_warmup_steps - 1) % cfg.categorical_proposal_refresh_every == 0:
                            prior = sample_prior(proposal_rng, (cfg.categorical_proposal_reference_particles, 1), cfg)
                            reference = np.concatenate([np.asarray(evaluate_cloud(models[f"{kind}__buffered"],
                                jnp.asarray(prior), jnp.asarray([[cfg.observed_x]], dtype=jnp.float32)))
                                for kind in ("adaln", "cross_attention")])
                        theta, weights = proposal_acquisition(proposal_rng, reference, b, cfg)
                    else:
                        theta, weights = sample_prior(rng, (b, 1), cfg), np.ones(b, np.float32)
                    x = np.zeros((b, cfg.max_training_observations, 1), np.float32)
                    valid = np.arange(cfg.max_training_observations)[None] < obs_counts[:, None]
                    x[valid] = np.repeat(theta, obs_counts, axis=0) + rng.normal(0, cfg.noise_std, (int(obs_counts.sum()), 1))
                    simulations += int(obs_counts.sum())
                    ids = buffer.add(theta, x, weights, obs_counts)
                else:
                    epoch, batch = divmod(step - batches - 1, batches)
                    epoch += 1
                    if batch == 0:
                        order = shuffle.permutation(buffer.size)
                    ids = order[batch * cfg.batch_size:(batch + 1) * cfg.batch_size]
                key = jax.random.fold_in(jax.random.key(seed + 30007), step)
                for name, (_, variant) in specs.items():
                    incoming, info = training_inputs(buffer, ids, name, variant, particles, cfg, cloud_seed)
                    tick = perf_counter()
                    model, state, loss, (clouds, prefixes), grad_norm = train_step(
                        models[name], states[name], jnp.asarray(incoming), jnp.asarray(buffer.x[ids]),
                        jnp.asarray(buffer.theta[ids]), jnp.asarray(buffer.weights[ids]),
                        jnp.asarray(buffer.counts[ids]), key)
                    loss, clouds, prefixes, grad_norm = jax.device_get((loss, clouds, prefixes, grad_norm))
                    model_seconds[name] += perf_counter() - tick  # Includes JIT on first shape/architecture.
                    if not np.isfinite(loss) or not np.isfinite(grad_norm):
                        raise FloatingPointError(f"Nonfinite training metric: {name}, update {step}")
                    models[name], states[name] = model, state
                    buffer.update(name, ids, clouds)
                    rows.append({"seed": seed, "model": name, "step": step, "simulations_seen": simulations,
                                 "replay_epoch": epoch, "particles": particles, "energy_score": float(loss),
                                 "grad_norm": float(grad_norm), "seconds_including_compile": model_seconds[name],
                                 **info, **{f"prefix_{i+1}_score": float(v) for i, v in enumerate(prefixes)},
                                 **{f"prefix_{i+1}_rows": int(np.sum(buffer.counts[ids] > i)) for i in range(len(prefixes))}})
                if step in schedule:
                    save_checkpoint(root, step, models, rows, buffer, cfg,
                                    {"simulations_seen": simulations, "total_steps": total,
                                     "training_seconds": perf_counter() - start_time})
                if step == 1 or step % cfg.log_every == 0 or step == total:
                    print(f"seed {seed} step {step}/{total}, observations {simulations}: " + " | ".join(
                        f"{r['model']}={r['energy_score']:.5f}" for r in rows[-len(specs):]))
            assert simulations == cfg.simulation_budget
            folder = checkpoint_location(root, specs)
        fitted[seed] = models
        histories.extend(rows)
        checkpoint_paths[seed] = str(folder)
    return fitted, histories, checkpoint_paths


#%% 5) TRAIN / LOAD — run once, then rerun any experiment cell independently
CFG, OUT = setup_run(Config(), script_location())
print("JAX devices:", jax.devices(), "\nRun:", OUT)
print("Both conditioning types; six matched fits per seed. Defaults retain the full reference training budget.")
MODELS, TRAINING_HISTORY, CHECKPOINT_PATHS = train_or_load(CFG, OUT)

#%% 6) Reusable evaluation / plotting helpers (no experiment runs in this cell)

def experiment_folder(name, settings):
    for key in ("trajectories", "particles", "observations", "total_observations", "batch_size",
                "block_observations", "repetitions", "bootstrap", "smoothing_window"):
        if key in settings:
            value = settings[key]
            minimum = 2 if key in {"trajectories", "particles", "bootstrap"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
    if "credible_mass" in settings and not 0 < settings["credible_mass"] < 1:
        raise ValueError("credible_mass must lie in (0,1)")
    folder = OUT / "experiments" / f"{name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    folder.mkdir(parents=True)
    (folder / "settings.json").write_text(json.dumps({"experiment": name, "settings": settings,
        "checkpoints": CHECKPOINT_PATHS, "training_config": asdict(CFG),
        "uncertainty": "paired trajectory bootstrap; seed means reported separately; conditional on fitted seeds",
        "versions": {"jax": jax.__version__, "equinox": eqx.__version__, "numpy": np.__version__}}, indent=2))
    return folder


def heldout_data(seed, trajectories, observations, particles, cfg):
    if trajectories < 2 or observations < 1 or particles < 2:
        raise ValueError("Use >=2 trajectories/particles and >=1 observation")
    rng = np.random.default_rng(seed)
    theta = rng.normal(cfg.prior_mean, cfg.prior_std, trajectories)
    x = theta[:, None] + rng.normal(0, cfg.noise_std, (trajectories, observations))
    z = rng.normal(size=(trajectories, particles))
    prior = (cfg.prior_mean + cfg.prior_std * z).astype(np.float32)
    return theta, x, prior, z


def empirical_normal_w2(cloud, mean, variance):
    """Exact W2(empirical measure, Normal), integrating each step of its quantile function.

    Unlike a Gaussian moment approximation this detects non-Gaussian cloud shape, and
    unlike quantile-midpoint matching it retains within-bin reference variance.
    """
    y = np.sort(np.asarray(cloud, np.float64), axis=-1)
    if y.shape[-1] < 2 or not np.all(np.isfinite(y)):
        raise ValueError("W2 requires at least two finite particles")
    z = ndtri(np.linspace(0, 1, y.shape[-1] + 1))
    phi = np.exp(-.5 * z*z) / np.sqrt(2 * np.pi)
    integrals = phi[:-1] - phi[1:]
    centered = y - np.asarray(mean)[..., None]
    squared = np.mean(centered**2, axis=-1) + variance - 2 * np.sqrt(variance) * np.sum(centered * integrals, axis=-1)
    return np.sqrt(np.maximum(squared, 0))


def cloud_w2(a, b):
    """Exact empirical W2 for equal-size clouds (invariant to particle permutation)."""
    if np.shape(a) != np.shape(b):
        raise ValueError("Comparison clouds must have equal shapes")
    return np.sqrt(np.mean((np.sort(a, axis=-1) - np.sort(b, axis=-1))**2, axis=-1))


def cloud_metrics(cloud, truth, mean, variance, credible_mass):
    y = np.asarray(cloud, np.float64)
    if not np.all(np.isfinite(y)):
        raise FloatingPointError("Nonfinite deployment cloud")
    location, sd = y.mean(axis=-1), y.std(axis=-1)
    truth = np.broadcast_to(np.asarray(truth)[:, None], location.shape)
    w2 = empirical_normal_w2(y, mean, variance)
    n = y.shape[-1]
    ordered = np.sort(y, axis=-1)
    half_pairwise = np.sum((2 * np.arange(1, n + 1) - n - 1) * ordered, axis=-1) / n**2
    lower, upper = np.quantile(y, [(1 - credible_mass)/2, (1 + credible_mass)/2], axis=-1)
    # Explicitly a moment-fit KL, NOT KL of an atomic measure to a continuous density (infinite).
    fit_variance = np.maximum(sd**2, np.finfo(np.float64).tiny)
    return {"w2": w2, "relative_w2": w2 / np.sqrt(variance),
            "mean_mse": (location - truth)**2, "mean_reference_mse": (location - mean)**2,
            "posterior_sd": sd, "coverage": ((truth >= lower) & (truth <= upper)).astype(float),
            "crps": np.mean(np.abs(y - truth[..., None]), axis=-1) - half_pairwise,
            "gaussian_fit_kl": .5 * (np.log(variance / fit_variance)
                                    + (fit_variance + (location - mean)**2) / variance - 1)}


def analytic_metrics(truth, mean, variance, mass):
    mean, variance = np.broadcast_arrays(mean, variance)
    residual = np.asarray(truth)[:, None] - mean
    sd = np.sqrt(variance)
    z = residual / sd
    crps = sd * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-z*z/2)/np.sqrt(2*np.pi) - 1/np.sqrt(np.pi))
    zero = np.zeros_like(mean)
    return {"w2": zero, "relative_w2": zero, "mean_mse": residual**2,
            "mean_reference_mse": zero, "posterior_sd": sd,
            "coverage": (np.abs(z) <= ndtri((1 + mass)/2)).astype(float), "crps": crps,
            "gaussian_fit_kl": zero}


@eqx.filter_jit
def _direct_batch(model, priors, observations):
    return jax.vmap(lambda p, x: model.predict_prefixes(p[:, None], x[:, None], inference=True)[..., 0])(
        priors, observations)


@eqx.filter_jit
def _chunk_batch(model, priors, observations, chunk_size):
    # Each observation is used exactly ONCE; no prefix is resent to a carried posterior.
    blocks = observations.reshape(len(observations), -1, chunk_size)
    def trajectory(p, xs):
        def advance(cloud, block):
            updated = model(cloud[:, None], block[:, None], inference=True)[:, 0]
            return updated, updated
        return jax.lax.scan(advance, p, xs)[1]
    return jax.vmap(trajectory)(priors, blocks)


def deploy(model, priors, observations, *, method="joint", chunk_size=1, batch_size=4):
    if batch_size < 1 or chunk_size < 1 or observations.shape[1] % chunk_size:
        raise ValueError("Positive batch/chunk sizes required; chunk size must divide observation count")
    outputs = []
    for start in range(0, len(priors), batch_size):
        p = jnp.asarray(priors[start:start+batch_size], dtype=jnp.float32)
        x = jnp.asarray(observations[start:start+batch_size], dtype=jnp.float32)
        if method == "joint":
            cloud = _direct_batch(model, p, x)
        elif method == "sequential":
            cloud = _chunk_batch(model, p, x, chunk_size)
        else:
            raise ValueError("method must be joint or sequential")
        outputs.append(np.asarray(cloud))
    return np.concatenate(outputs)


def bootstrap_mean(values, indices):
    values = np.asarray(values)
    estimate = values.mean(axis=0)
    boot = values[indices].mean(axis=1)
    low, high = np.quantile(boot, [.025, .975], axis=0)
    return estimate, low, high


def summarize_results(folder, results, axis_values, *, bootstrap, seed, axis_name="observations"):
    """Keep seeds and paired trajectories distinct. Save raw arrays plus paired contrasts."""
    if bootstrap < 2:
        raise ValueError("Use at least two bootstrap replicates")
    axis_values = np.asarray(axis_values)
    first = next(iter(results.values()))
    trajectories = next(iter(first.values())).shape[0]
    indices = np.random.default_rng(seed).integers(trajectories, size=(bootstrap, trajectories))
    arrays, index, grouped, seed_rows = {}, {}, {}, []
    for i, ((fit_seed, model, strategy), metrics) in enumerate(results.items()):
        index[f"result_{i}"] = {"seed": fit_seed, "model": model, "strategy": strategy}
        for metric, values in metrics.items():
            if np.shape(values) != (trajectories, len(axis_values)):
                raise ValueError(f"Incorrect result shape for {model}/{strategy}/{metric}: {np.shape(values)}")
            arrays[f"result_{i}__{metric}"] = values
            grouped.setdefault((model, strategy, metric), []).append(values)
            for j, x in enumerate(axis_values):
                seed_rows.append({"seed": fit_seed, "model": model, "strategy": strategy,
                                  "metric": metric, axis_name: int(x), "mean": float(values[:, j].mean())})
    np.savez_compressed(folder / "raw_metrics.npz", axis=axis_values, **arrays)
    (folder / "result_index.json").write_text(json.dumps(index, indent=2))
    write_rows(folder / "seed_means.csv", seed_rows)
    summary, estimates = [], {}
    for (model, strategy, metric), values in grouped.items():
        # Average independent fits before resampling paired held-out trajectories; this CI
        # is conditional on the fitted ensemble, not a claimed training-population interval.
        mean, low, high = bootstrap_mean(np.mean(values, axis=0), indices)
        estimates[model, strategy, metric] = (mean, low, high)
        for j, x in enumerate(axis_values):
            summary.append({"model": model, "strategy": strategy, "metric": metric, axis_name: int(x),
                            "mean": float(mean[j]), "low": float(low[j]), "high": float(high[j]),
                            "seed_mean_sd": float(np.std([v[:, j].mean() for v in values], ddof=1)) if len(values)>1 else float("nan")})
    write_rows(folder / "summary.csv", summary)
    contrasts = []
    # Pair within seed AND held-out trajectory. Negative delta favors AdaLN for error/CRPS.
    for (model, strategy, metric), values in grouped.items():
        if not model.startswith("adaln__"):
            continue
        other = model.replace("adaln__", "cross_attention__")
        if (other, strategy, metric) not in grouped:
            continue
        differences = []
        for fit_seed in sorted({k[0] for k in results if k[1] == model}):
            differences.append(results[fit_seed, model, strategy][metric] - results[fit_seed, other, strategy][metric])
        mean, low, high = bootstrap_mean(np.mean(differences, axis=0), indices)
        for j, x in enumerate(axis_values):
            contrasts.append({"variant": model.split("__")[1], "strategy": strategy, "metric": metric,
                              axis_name: int(x), "adaln_minus_cross_attention": float(mean[j]),
                              "low": float(low[j]), "high": float(high[j]),
                              "seed_delta_sd": float(np.std([d[:, j].mean() for d in differences], ddof=1)) if len(differences)>1 else float("nan")})
        if metric == "w2":
            print(f"{model.split('__')[1]} / {strategy}, final W2 AdaLN-cross: {mean[-1]:.4g} [{low[-1]:.4g}, {high[-1]:.4g}]")
    write_rows(folder / "paired_conditioning_differences.csv", contrasts)
    print("Saved:", folder)
    return estimates


def plot_metrics(folder, estimates, axis, *, title, xlabel="Observations", theory_variance=None, mass=.95):
    metrics = ("w2", "relative_w2", "mean_mse", "posterior_sd", "coverage", "crps")
    labels = ("W2 to exact Gaussian", "W2 / exact posterior SD", "Mean MSE to latent truth",
              "Posterior SD", f"{100*mass:g}% interval coverage", "CRPS at latent truth")
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, metric, label in zip(axes.flat, metrics, labels):
        for (model, strategy, kind), (mean, low, high) in estimates.items():
            if kind != metric:
                continue
            if metric in {"w2", "relative_w2"} and np.all(mean == 0):
                continue  # Exact zero reference has no representation on a logarithmic axis.
            line, = ax.plot(axis, mean, label=f"{model}: {strategy}", linewidth=1.3)
            ax.fill_between(axis, low, high, color=line.get_color(), alpha=.10)
        if theory_variance is not None and metric in {"mean_mse", "posterior_sd"}:
            theory = theory_variance if metric == "mean_mse" else np.sqrt(theory_variance)
            ax.plot(axis, theory, "k--", linewidth=2, label="Bayes risk V_n" if metric == "mean_mse" else "Exact SD")
        if metric == "coverage":
            ax.axhline(mass, color="k", linestyle="--")
            ax.set_ylim(0, 1.02)
        else:
            ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set(xlabel=xlabel, title=label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    fig.suptitle(title + "\n95% paired trajectory-bootstrap intervals; conditional on fitted seeds")
    fig.tight_layout(rect=(0, .18, 1, .94))
    fig.savefig(folder / "metrics.png", dpi=180, bbox_inches="tight")
    plt.show()


def selected_models(variants):
    if not variants or not set(variants) <= {"fresh", "no_replay", "buffered"}:
        raise ValueError("Select fresh, no_replay, and/or buffered variants")
    return [(seed, name, model) for seed, models in MODELS.items() for name, model in models.items()
            if name.split("__")[1] in variants]


def save_design(folder, theta, x, prior):
    np.savez_compressed(folder / "heldout_design.npz", theta=theta, observations=x, prior=prior)


#%% 7) Loss visualization — editable here; works identically after train=False reload
LOSS_SETTINGS = {"smoothing_window": 100, "show_prefix_scores": True}
_loss_dir = experiment_folder("loss", LOSS_SETTINGS)
_window = int(LOSS_SETTINGS["smoothing_window"])
if _window < 1:
    raise ValueError("smoothing_window must be positive")
_loss_fig, _loss_axes = plt.subplots(1, 3, figsize=(16, 4))
for _seed in CFG.training_seeds:
    for _name in model_specs():
        _rows = [r for r in TRAINING_HISTORY if r["seed"] == _seed and r["model"] == _name]
        if not _rows:
            continue
        _steps = np.asarray([r["step"] for r in _rows])
        for _ax, _metric in zip(_loss_axes, ("energy_score", "grad_norm", "buffer_fraction")):
            _values = np.asarray([r[_metric] for r in _rows])
            _w = min(_window, len(_values))
            _smooth = np.convolve(_values, np.ones(_w)/_w, mode="valid")
            _ax.plot(_steps[_w-1:], _smooth, label=f"{_seed}: {_name}")
            _ax.set(xlabel="Optimizer update", title=_metric.replace("_", " "))
_loss_axes[1].set_yscale("symlog", linthresh=1e-6)
_loss_axes[2].set_ylim(-.02, 1.02)
_loss_axes[0].legend(fontsize=6)
_loss_fig.tight_layout()
_loss_fig.savefig(_loss_dir / "training_loss.png", dpi=180)
plt.show()
if LOSS_SETTINGS["show_prefix_scores"]:
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))
    for _ax, _kind in zip(_axes, ("adaln", "cross_attention")):
        for _prefix in range(1, CFG.max_training_observations+1):
            # Plot one seed's buffered model to avoid hiding prefixes in dozens of curves.
            _rows = [r for r in TRAINING_HISTORY if r["seed"] == CFG.training_seeds[0]
                     and r["model"] == f"{_kind}__buffered" and r[f"prefix_{_prefix}_rows"] > 0]
            _w = min(_window, len(_rows))
            if _w:
                _ax.plot([r["step"] for r in _rows][_w-1:], np.convolve(
                    [r[f"prefix_{_prefix}_score"] for r in _rows], np.ones(_w)/_w, "valid"), label=f"prefix {_prefix}")
        _ax.set(title=f"{_kind}, buffered, seed {CFG.training_seeds[0]}", xlabel="Optimizer update", ylabel="Energy score")
        _ax.legend(fontsize=7)
    _fig.tight_layout()
    _fig.savefig(_loss_dir / "prefix_loss.png", dpi=180)
    plt.show()

#%% 8) Sequential convergence versus joint inference — all hyperparameters for THIS experiment
SEQUENTIAL = {
    "seed": 41001, "trajectories": 64, "observations": 64, "particles": 64,
    "batch_size": 4, "variants": ("fresh", "no_replay", "buffered"),
    "credible_mass": .95, "bootstrap": 500,
    "fit_min_observations": 16, "fit_max_observations": 64,
    "snapshot_counts": (1, 8, 32, 64), "save_clouds": True,
}


def run_sequential(settings):
    s = settings
    mask = ((np.arange(1, s["observations"]+1) >= s["fit_min_observations"])
            & (np.arange(1, s["observations"]+1) <= s["fit_max_observations"]))
    if mask.sum() < 3:
        raise ValueError("The convergence-rate fit needs at least three observation counts")
    folder = experiment_folder("sequential", s)
    theta, x, prior, z = heldout_data(s["seed"], s["trajectories"], s["observations"], s["particles"], CFG)
    save_design(folder, theta, x, prior)
    mean, variance = exact_prefixes(x, CFG)
    axis = np.arange(1, s["observations"]+1)
    results, examples = {}, {}
    oracle = mean[..., None] + np.sqrt(variance[..., None]) * z[:, None]
    results[-1, "oracle_iid", "exact affine transport"] = cloud_metrics(oracle, theta, mean, variance, s["credible_mass"])
    results[-1, "analytic", "closed form"] = analytic_metrics(theta, mean, variance, s["credible_mass"])
    # Optimal equal-weight quantization gives a deterministic finite-N W2 floor.
    edges = ndtri(np.linspace(0, 1, s["particles"]+1))
    phi = np.exp(-edges**2/2)/np.sqrt(2*np.pi)
    centroids = s["particles"] * (phi[:-1] - phi[1:])
    quantized = mean[..., None] + np.sqrt(variance[..., None]) * centroids
    results[-1, "oracle_quantized", "optimal equal-weight atoms"] = cloud_metrics(
        quantized, theta, mean, variance, s["credible_mass"])
    for fit_seed, name, model in selected_models(s["variants"]):
        for strategy in ("joint", "sequential"):
            print(f"Sequential experiment: seed {fit_seed}, {name}, {strategy}")
            cloud = deploy(model, prior, x, method=strategy, batch_size=s["batch_size"])
            results[fit_seed, name, strategy] = cloud_metrics(cloud, theta, mean, variance, s["credible_mass"])
            examples[fit_seed, name, strategy] = cloud[0]
            if s["save_clouds"]:
                np.savez_compressed(folder / f"clouds_{fit_seed}_{name}_{strategy}.npz", particles=cloud)
    estimates = summarize_results(folder, results, axis, bootstrap=s["bootstrap"], seed=s["seed"]+1)
    plot_metrics(folder, estimates, axis, title="Sequential transfer versus fresh-prior joint inference",
                 theory_variance=variance[0], mass=s["credible_mass"])
    # Fit on ensemble-averaged errors with trajectory bootstrap. A fitted neural-error slope
    # is empirical only. Report finite-range theory too; asymptotic slopes need not hold early.
    log_n = np.log(axis[mask])
    centered = log_n - log_n.mean()
    def slope(curve):
        if np.any(curve <= 0) or not np.all(np.isfinite(curve)):
            return np.full(np.shape(curve)[:-1], np.nan)
        return np.sum(np.log(curve) * centered, axis=-1) / np.sum(centered**2)
    boot_ids = np.random.default_rng(s["seed"]+2).integers(len(theta), size=(s["bootstrap"], len(theta)))
    slopes = []
    for model, strategy in sorted({(k[1], k[2]) for k in results}):
        for metric in ("w2", "relative_w2", "mean_mse", "posterior_sd"):
            values = np.mean([v[metric] for k, v in results.items() if k[1:] == (model, strategy)], axis=0)[:, mask]
            fitted = slope(values.mean(axis=0))
            sampled = slope(values[boot_ids].mean(axis=1))
            lo, hi = np.quantile(sampled, [.025, .975])
            asymptotic = -1.0 if metric == "mean_mse" else -.5 if metric == "posterior_sd" else float("nan")
            theory_curve = variance[0, mask] if metric == "mean_mse" else np.sqrt(variance[0, mask])
            finite_theory = float(slope(theory_curve)) if metric in {"mean_mse", "posterior_sd"} else float("nan")
            slopes.append({"model": model, "strategy": strategy, "metric": metric,
                           "slope": float(fitted), "low": float(lo), "high": float(hi),
                           "oracle_asymptotic_slope": asymptotic, "oracle_finite_range_slope": finite_theory})
    write_rows(folder / "convergence_slopes.csv", slopes)
    counts = [n for n in s["snapshot_counts"] if 1 <= n <= len(axis)]
    if counts:
        fig, axes = plt.subplots(1, len(counts), figsize=(5*len(counts), 4), squeeze=False)
        for ax, n in zip(axes[0], counts):
            grid = np.linspace(mean[0,n-1]-4*np.sqrt(variance[0,n-1]), mean[0,n-1]+4*np.sqrt(variance[0,n-1]), 300)
            ax.plot(grid, np.exp(-.5*(grid-mean[0,n-1])**2/variance[0,n-1])/np.sqrt(2*np.pi*variance[0,n-1]), "k", label="exact Gaussian")
            for (fit_seed, name, strategy), cloud in examples.items():
                if fit_seed == CFG.training_seeds[0] and name.endswith("__buffered"):
                    ax.hist(cloud[n-1], bins=20, density=True, histtype="step", label=f"{name}: {strategy}")
            ax.axvline(theta[0], color="k", linestyle=":", label="latent truth")
            ax.set(title=f"n={n}", xlabel="theta")
        axes[0,0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(folder / "posterior_snapshots.png", dpi=180)
        plt.show()
    print(f"Joint prefixes above {CFG.max_training_observations} observations extrapolate beyond training.")
    print("Oracle MSE ~ n^-1 and SD ~ n^-1/2; there is no prescribed neural W2-error slope.")
    return folder


SEQUENTIAL_FOLDER = run_sequential(SEQUENTIAL)


#%% 9) Observations per update — fixed total evidence, matched streams/particles, varying call count
CHUNKING = {
    "seed": 42001, "trajectories": 64, "total_observations": 64, "particles": 64,
    "observations_per_call": (1, 2, 4, 8, 16, 32, 64),  # Must divide total_observations.
    "batch_size": 4, "variants": ("fresh", "no_replay", "buffered"),
    "credible_mass": .95, "bootstrap": 500,
}


def run_chunking(s):
    sizes = np.asarray(s["observations_per_call"], dtype=int)
    if np.any(sizes < 1) or np.any(s["total_observations"] % sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("Provide distinct positive chunk sizes dividing total_observations")
    folder = experiment_folder("observations_per_update", s)
    theta, x, prior, z = heldout_data(s["seed"], s["trajectories"], s["total_observations"], s["particles"], CFG)
    save_design(folder, theta, x, prior)
    m, v = exact_posterior(x, CFG)
    mean = np.repeat(m[:, None], len(sizes), axis=1)
    variance = np.full_like(mean, v)
    results = {(-1, "oracle_iid", "fixed total evidence"): cloud_metrics(
        mean[..., None]+np.sqrt(variance[..., None])*z[:, None], theta, mean, variance, s["credible_mass"])}
    timings = []
    for fit_seed, name, model in selected_models(s["variants"]):
        final = []
        for size in sizes:
            tick = perf_counter()
            cloud = deploy(model, prior, x, method="sequential", chunk_size=int(size), batch_size=s["batch_size"])
            final.append(cloud[:, -1])
            timings.append({"seed": fit_seed, "model": name, "observations_per_call": int(size),
                            "calls_per_trajectory": s["total_observations"] // int(size),
                            "seconds_including_compile": perf_counter()-tick})
        results[fit_seed, name, "chunked"] = cloud_metrics(np.stack(final, axis=1), theta, mean, variance, s["credible_mass"])
    estimates = summarize_results(folder, results, sizes, bootstrap=s["bootstrap"], seed=s["seed"]+1, axis_name="observations_per_call")
    plot_metrics(folder, estimates, sizes, xlabel="Observations per call", mass=s["credible_mass"],
                 title=f"Same {s['total_observations']} observations; different grouping and number of calls")
    write_rows(folder / "call_counts_and_timings.csv", timings)
    return folder


CHUNKING_FOLDER = run_chunking(CHUNKING)


#%% 10) Compositionality and order — two DISTINCT blocks, AB/BA, sequential/joint, exact control
COMPOSITION = {
    "seed": 43001, "trajectories": 128, "particles": 64, "block_observations": 1,
    "history_observations": 0,  # >0 supplies an exact history-posterior input: arbitrary-prior transfer.
    "batch_size": 4, "variants": ("fresh", "no_replay", "buffered"),
    "credible_mass": .95, "bootstrap": 1000,
}


def run_composition(s):
    b, h = s["block_observations"], s["history_observations"]
    if b < 1 or h < 0:
        raise ValueError("Positive block size and nonnegative history length required")
    folder = experiment_folder("composition", s)
    theta, all_x, prior, z = heldout_data(s["seed"], s["trajectories"], h+2*b, s["particles"], CFG)
    history, x = all_x[:, :h], all_x[:, h:]
    start_mean, start_var = exact_posterior(history, CFG)
    incoming = (start_mean[:, None] + np.sqrt(start_var)*z).astype(np.float32)
    save_design(folder, theta, all_x, incoming)
    swapped = np.concatenate([x[:, b:], x[:, :b]], axis=1)
    m, v = exact_posterior(all_x, CFG)
    results, defect_rows, raw_defects = {}, [], {}
    boot = np.random.default_rng(s["seed"]+1).integers(len(theta), size=(s["bootstrap"], len(theta)))
    defects_by_fit = {}
    for fit_seed, name, model in selected_models(s["variants"]):
        clouds = {}
        for order, data in (("AB", x), ("BA", swapped)):
            for method in ("joint", "sequential"):
                key = f"{method}_{order}"
                clouds[key] = deploy(model, incoming, data, method=method, chunk_size=b, batch_size=s["batch_size"])[:, -1]
                results[fit_seed, name, key] = cloud_metrics(clouds[key][:, None], theta, m[:, None], np.full((len(theta),1),v), s["credible_mass"])
        defects = {
            "sequential_commutator": cloud_w2(clouds["sequential_AB"], clouds["sequential_BA"]),
            "joint_order_sensitivity": cloud_w2(clouds["joint_AB"], clouds["joint_BA"]),
            "composition_AB": cloud_w2(clouds["sequential_AB"], clouds["joint_AB"]),
            "composition_BA": cloud_w2(clouds["sequential_BA"], clouds["joint_BA"]),
        }
        defects_by_fit[fit_seed, name] = defects
        np.savez_compressed(folder / f"clouds_{fit_seed}_{name}.npz", **clouds)
        for kind, values in defects.items():
            raw_defects[f"{fit_seed}__{name}__{kind}"] = values
            estimate, low, high = bootstrap_mean(values, boot)
            defect_rows.append({"seed": fit_seed, "model": name, "defect": kind,
                               "mean_w2": float(estimate), "low": float(low), "high": float(high),
                               "mean_relative_w2": float(estimate/np.sqrt(v))})
    # Exact affine updates must agree particlewise, including an arbitrary exact incoming prior.
    def analytic_order(data):
        first_mean, first_var = exact_posterior(data[:, :b], CFG, prior_mean=start_mean, prior_variance=start_var)
        second_mean, second_var = exact_posterior(data[:, b:], CFG, prior_mean=first_mean, prior_variance=first_var)
        first = first_mean[:, None] + np.sqrt(first_var/start_var) * (incoming - start_mean[:, None])
        final = second_mean[:, None] + np.sqrt(second_var/first_var) * (first-first_mean[:, None])
        return final
    exact_ab, exact_ba = analytic_order(x), analytic_order(swapped)
    exact_joint = m[:, None] + np.sqrt(v/start_var) * (incoming-start_mean[:, None])
    np.testing.assert_allclose(exact_ab, exact_ba, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(exact_ab, exact_joint, atol=1e-10, rtol=1e-10)
    write_rows(folder / "composition_defects.csv", defect_rows)
    np.savez_compressed(folder / "raw_defects.npz", **raw_defects,
                        exact_commutator=cloud_w2(exact_ab, exact_ba), exact_composition=cloud_w2(exact_ab, exact_joint))
    paired = []
    for variant in s["variants"]:
        for kind in next(iter(defects_by_fit.values())):
            delta = np.mean([defects_by_fit[seed, f"adaln__{variant}"][kind]
                             - defects_by_fit[seed, f"cross_attention__{variant}"][kind] for seed in CFG.training_seeds], axis=0)
            estimate, low, high = bootstrap_mean(delta, boot)
            paired.append({"variant": variant, "defect": kind, "adaln_minus_cross_attention": float(estimate),
                           "low": float(low), "high": float(high)})
    write_rows(folder / "paired_defect_differences.csv", paired)
    summarize_results(folder, results, [2*b], bootstrap=s["bootstrap"], seed=s["seed"]+1)
    kinds = list(next(iter(defects_by_fit.values())))
    fig, axes = plt.subplots(1, len(kinds), figsize=(18, 5))
    for ax, kind in zip(axes, kinds):
        for i, name in enumerate(sorted({key[1] for key in defects_by_fit})):
            values = np.mean([d[kind] for (seed, n), d in defects_by_fit.items() if n == name], axis=0)
            estimate, low, high = bootstrap_mean(values, boot)
            ax.errorbar(i, estimate, yerr=[[max(0, estimate-low)], [max(0, high-estimate)]], fmt="o", capsize=4)
        ax.set_xticks(range(len({key[1] for key in defects_by_fit})), sorted({key[1] for key in defects_by_fit}), rotation=75, fontsize=7)
        ax.set(title=kind.replace("_", " "), ylabel="Empirical W2 defect (exact = 0)")
        ax.set_ylim(bottom=0)
    fig.suptitle("Order/composition defects; assess posterior accuracy too: an identity map has zero defect")
    fig.tight_layout()
    fig.savefig(folder / "composition_defects.png", dpi=180)
    plt.show()
    return folder


COMPOSITION_FOLDER = run_composition(COMPOSITION)


#%% 11) Particle-count sensitivity — fixed held-out evidence, nested initial random clouds
PARTICLES = {
    "seed": 44001, "trajectories": 64, "observations": 16,
    "particle_counts": (16, 32, 64, 128, 256), "batch_size": 2,
    "variants": ("fresh", "buffered"), "credible_mass": .95, "bootstrap": 500,
}


def run_particles(s):
    sizes = np.asarray(s["particle_counts"], dtype=int)
    if len(sizes) == 0 or np.any(sizes < 2) or len(set(sizes)) != len(sizes):
        raise ValueError("Provide distinct particle counts >=2")
    folder = experiment_folder("particle_count", s)
    theta, x, prior, z = heldout_data(s["seed"], s["trajectories"], s["observations"], int(max(sizes)), CFG)
    save_design(folder, theta, x, prior)
    m, v = exact_posterior(x, CFG)
    results = {}
    def metrics(cloud):
        return cloud_metrics(cloud[:, None], theta, m[:, None], np.full((len(theta),1),v), s["credible_mass"])
    oracle_metrics = [metrics(m[:, None]+np.sqrt(v)*z[:, :n]) for n in sizes]
    results[-1, "oracle_iid", "exact affine transport"] = {key: np.concatenate([a[key] for a in oracle_metrics], axis=1) for key in oracle_metrics[0]}
    for seed, name, model in selected_models(s["variants"]):
        for method in ("joint", "sequential"):
            measured = [metrics(deploy(model, prior[:, :n], x, method=method, batch_size=s["batch_size"])[:, -1]) for n in sizes]
            results[seed, name, method] = {key: np.concatenate([a[key] for a in measured], axis=1) for key in measured[0]}
    estimates = summarize_results(folder, results, sizes, bootstrap=s["bootstrap"], seed=s["seed"]+1, axis_name="particles")
    plot_metrics(folder, estimates, sizes, title=f"Particle-count transfer at {s['observations']} observations", xlabel="Particles", mass=s["credible_mass"])
    return folder


PARTICLES_FOLDER = run_particles(PARTICLES)


#%% 12) Repeated-observation refinement — same datum is NOT new independent evidence
REFINEMENT = {
    "seed": 45001, "trajectories": 64, "particles": 64, "repetitions": 32,
    "batch_size": 4, "variants": ("fresh", "no_replay", "buffered"),
    "credible_mass": .95, "bootstrap": 500,
}


def run_refinement(s):
    folder = experiment_folder("repeated_observation", s)
    theta, x, prior, z = heldout_data(s["seed"], s["trajectories"], 1, s["particles"], CFG)
    save_design(folder, theta, x, prior)
    repeated = np.repeat(x, s["repetitions"], axis=1)
    m, v = exact_posterior(x, CFG)
    mean = np.repeat(m[:, None], s["repetitions"], axis=1)
    variance = np.full_like(mean, v)
    pseudo_mean, pseudo_variance = exact_prefixes(repeated, CFG)
    results, pseudo_rows = {}, []
    results[-1, "oracle_iid", "fixed one-datum posterior"] = cloud_metrics(
        mean[..., None] + np.sqrt(v)*z[:, None], theta, mean, variance, s["credible_mass"])
    for seed, name, model in selected_models(s["variants"]):
        for method in ("sequential", "joint"):
            cloud = deploy(model, prior, repeated, method=method, batch_size=s["batch_size"])
            results[seed, name, method] = cloud_metrics(cloud, theta, mean, variance, s["credible_mass"])
            pseudo_distance = empirical_normal_w2(cloud, pseudo_mean, pseudo_variance)
            # This alternative diagnostic treats duplicated x as independent measurements;
            # it is intentionally NOT used as the correct posterior/calibration reference.
            for i in range(s["repetitions"]):
                pseudo_rows.append({"seed": seed, "model": name, "method": method, "repetitions": i+1,
                                    "w2_to_duplicated_evidence_target": float(pseudo_distance[:, i].mean())})
    write_rows(folder / "duplicated_evidence_diagnostic.csv", pseudo_rows)
    axis = np.arange(1, s["repetitions"]+1)
    estimates = summarize_results(folder, results, axis, bootstrap=s["bootstrap"], seed=s["seed"]+1, axis_name="repetitions")
    plot_metrics(folder, estimates, axis, xlabel="Reuses of the same datum", mass=s["credible_mass"],
                 title="Refinement versus duplicated-input joint calls; reference stays p(theta | x)", theory_variance=variance[0])
    return folder


REFINEMENT_FOLDER = run_refinement(REFINEMENT)
#%%
