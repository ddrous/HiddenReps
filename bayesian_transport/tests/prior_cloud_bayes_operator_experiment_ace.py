#%% 0
"""Prior-cloud Bayes-operator experiment — ACE-style prior meta-amortization.

Goal
----
Test whether a one-step set-to-set model can learn a genuine Bayes update operator
when it is amortized over an extremely broad family of priors.

The prior meta-distribution follows the prior-generation strategy used by ACE
(Chang et al., AISTATS 2025, Appendix B.1), adapted to our particle setting:

    * with probability 0.8: a bounded mixture of Gaussians;
    * otherwise: a flat Uniform prior over the global theta domain;
    * K ~ Geometric(q=0.5) mixture components;
    * for K > 1, equal probability of:
          same means / different stds,
          different means / same std,
          all means and stds different;
    * mixture weights ~ Dirichlet(1,...,1).

ACE represents these priors as normalized histograms on a bounded grid. Here the
same bounded prior law is represented directly by particles. Gaussian mixtures
are therefore truncated and renormalized to the configured theta domain before
we draw the input cloud.

Two supervision semantics are still compared:

1) member:
       theta* is chosen uniformly FROM Theta_in.
   Conditional on Theta_in, the exact Bayesian target is the discrete empirical
   posterior over the input particles.

2) distribution:
       Theta_in and theta* are independent draws from the SAME underlying prior.
   The target is the continuous posterior under that underlying prior (up to the
   finite-cloud uncertainty seen by the network).

Evaluation retains the Gaussian / Uniform closed-form tests, including a Uniform
prior with zero support near theta_true. This remains the strongest diagnostic of
whether the learned update genuinely respects a supplied prior rather than simply
learning x -> theta.

The file is deliberately organized as notebook-style #%% cells. Training,
plotting, and evaluation are separated so evaluation cells can be re-run after
loading checkpoints without retraining.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import math
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.stats import gaussian_kde, norm, truncnorm, wasserstein_distance

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "DejaVu Sans",
    "axes.titlepad": 8.0,
    "axes.labelpad": 6.0,
})


Array = jax.Array
TruthMode = Literal["member", "distribution"]


@dataclass
class Config:
    # Reproducibility / outputs
    seed: int = 2037
    output_dir: str = "plots/prior_cloud_bayes_operator_ace"

    # Likelihood: x | theta ~ N(theta, likelihood_std^2)
    likelihood_std: float = 0.45

    # Training -- kept unchanged from the previous experiment.
    training_steps: int = 60_000
    batch_size: int = 128
    training_particles: int = 64
    learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip_norm: float = 1000.0
    log_every: int = 500

    # One-step particle transport -- kept unchanged.
    hidden_dim: int = 96
    heads: int = 4
    mlp_ratio: int = 4
    depth: int = 4
    global_theta_scale: float = 4.0
    max_normalized_displacement: float = 2.0

    # ------------------------------------------------------------------
    # ACE-style prior meta-distribution.
    # ------------------------------------------------------------------
    # Appendix B.1 of ACE uses 80% smooth Gaussian-mixture priors and 20%
    # flat priors; K ~ Geometric(0.5); and Dirichlet(alpha0=1) weights.
    ace_mixture_probability: float = 0.80
    ace_geometric_q: float = 0.50

    # ACE generates priors on a bounded latent range.  We use that range
    # directly as the support from which the particle cloud is sampled.
    ace_domain_low: float = -4.0
    ace_domain_high: float = 4.0

    # The paper states that the global mean/std hyper-ranges are determined by
    # the latent range, but does not prescribe one universal numerical formula.
    # Here means span the full theta domain and stds are configurable fractions
    # of its width.  These are the only adaptation-specific choices.
    ace_component_std_min_fraction: float = 0.025
    ace_component_std_max_fraction: float = 0.35

    # Diagnostics during training
    diagnostic_every: int = 250
    snapshot_steps: tuple[int, ...] = (100, 500, 1_500, 3_000, 6_000)

    # Evaluation
    eval_particles: int = 1024
    exact_reference_samples: int = 20_000
    density_grid_points: int = 900
    plot_transport_particles: int = 180
    random_ace_eval_priors: int = 6


CFG = Config()
OUT = Path(CFG.output_dir)
OUT.mkdir(parents=True, exist_ok=True)

if CFG.hidden_dim % CFG.heads != 0:
    raise ValueError("hidden_dim must be divisible by heads.")
if not 0.0 <= CFG.ace_mixture_probability <= 1.0:
    raise ValueError("ace_mixture_probability must lie in [0,1].")
if not 0.0 < CFG.ace_geometric_q <= 1.0:
    raise ValueError("ace_geometric_q must lie in (0,1].")
if CFG.ace_domain_low >= CFG.ace_domain_high:
    raise ValueError("ace_domain_low must be smaller than ace_domain_high.")
if not 0.0 < CFG.ace_component_std_min_fraction <= CFG.ace_component_std_max_fraction:
    raise ValueError("ACE std fractions must satisfy 0 < min <= max.")
if CFG.likelihood_std <= 0.0:
    raise ValueError("likelihood_std must be positive.")

plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.18,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "stix",
})

print("JAX devices:", jax.devices())
print("Output directory:", OUT.resolve())
print(json.dumps(asdict(CFG), indent=2))


#%% 1
@dataclass(frozen=True)
class PriorSpec:
    family: str
    label: str
    params: tuple[Any, ...]


@dataclass(frozen=True)
class EvalCase:
    name: str
    prior: PriorSpec
    x_obs: float
    theta_true: float
    note: str = ""


def gaussian_prior(mean: float, std: float, label: str | None = None) -> PriorSpec:
    return PriorSpec("gaussian", label or rf"$\mathcal{{N}}({mean:.2g},{std:.2g}^2)$", (float(mean), float(std)))


def uniform_prior(low: float, high: float, label: str | None = None) -> PriorSpec:
    return PriorSpec("uniform", label or rf"$U[{low:.2g},{high:.2g}]$", (float(low), float(high)))


def gaussian_mixture_prior(
    means: tuple[float, ...],
    stds: tuple[float, ...],
    weights: tuple[float, ...],
    label: str = "Gaussian mixture",
) -> PriorSpec:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return PriorSpec(
        "gaussian_mixture",
        label,
        (tuple(map(float, means)), tuple(map(float, stds)), tuple(map(float, w))),
    )


def bounded_gaussian_mixture_prior(
    means: tuple[float, ...],
    stds: tuple[float, ...],
    weights: tuple[float, ...],
    low: float,
    high: float,
    label: str = "Bounded Gaussian mixture",
) -> PriorSpec:
    """ACE-style smooth prior: Gaussian mixture truncated and renormalized on [low, high]."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return PriorSpec(
        "bounded_gaussian_mixture",
        label,
        (
            tuple(map(float, means)),
            tuple(map(float, stds)),
            tuple(map(float, w)),
            float(low),
            float(high),
        ),
    )


def _bounded_mixture_component_mass(
    means: np.ndarray,
    stds: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    return norm.cdf((high - means) / stds) - norm.cdf((low - means) / stds)


def _sample_truncated_normals_np(
    rng: np.random.Generator,
    means: np.ndarray,
    stds: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Vectorized inverse-CDF draw from N(mean,std^2) truncated to [low,high]."""
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    cdf_low = norm.cdf((low - means) / stds)
    cdf_high = norm.cdf((high - means) / stds)
    u = cdf_low + rng.random(means.shape) * np.maximum(cdf_high - cdf_low, 1e-14)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return means + stds * norm.ppf(u)


def sample_prior_np(rng: np.random.Generator, prior: PriorSpec, n: int) -> np.ndarray:
    n = int(n)
    if prior.family == "gaussian":
        mean, std = prior.params
        return rng.normal(mean, std, size=n).astype(np.float32)

    if prior.family == "uniform":
        low, high = prior.params
        return rng.uniform(low, high, size=n).astype(np.float32)

    if prior.family == "gaussian_mixture":
        means, stds, weights = prior.params
        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        ids = rng.choice(len(weights), size=n, p=weights)
        return rng.normal(means[ids], stds[ids]).astype(np.float32)

    if prior.family == "bounded_gaussian_mixture":
        means, stds, weights, low, high = prior.params
        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        inside_mass = _bounded_mixture_component_mass(means, stds, low, high)
        component_probs = weights * inside_mass
        component_probs /= component_probs.sum()
        ids = rng.choice(len(weights), size=n, p=component_probs)
        draws = _sample_truncated_normals_np(rng, means[ids], stds[ids], low, high)
        return draws.astype(np.float32)

    raise ValueError(f"Unknown prior family {prior.family!r}.")


def prior_density_np(theta: np.ndarray, prior: PriorSpec) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)

    if prior.family == "gaussian":
        mean, std = prior.params
        return norm.pdf(theta, loc=mean, scale=std)

    if prior.family == "uniform":
        low, high = prior.params
        return np.where((theta >= low) & (theta <= high), 1.0 / (high - low), 0.0)

    if prior.family == "gaussian_mixture":
        means, stds, weights = prior.params
        result = np.zeros_like(theta, dtype=np.float64)
        for m, s, w in zip(means, stds, weights):
            result += w * norm.pdf(theta, loc=m, scale=s)
        return result

    if prior.family == "bounded_gaussian_mixture":
        means, stds, weights, low, high = prior.params
        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        z = float(np.sum(weights * _bounded_mixture_component_mass(means, stds, low, high)))
        result = np.zeros_like(theta, dtype=np.float64)
        for m, s, w in zip(means, stds, weights):
            result += w * norm.pdf(theta, loc=m, scale=s)
        return np.where((theta >= low) & (theta <= high), result / max(z, 1e-300), 0.0)

    raise ValueError(f"Unknown prior family {prior.family!r}.")


def likelihood_density_np(x_obs: float, theta: np.ndarray, cfg: Config = CFG) -> np.ndarray:
    return norm.pdf(float(x_obs), loc=np.asarray(theta, dtype=np.float64), scale=cfg.likelihood_std)


def gaussian_posterior_params(mean: float, std: float, x_obs: float, cfg: Config = CFG) -> tuple[float, float]:
    prior_prec = 1.0 / (std**2)
    like_prec = 1.0 / (cfg.likelihood_std**2)
    var = 1.0 / (prior_prec + like_prec)
    post_mean = var * (prior_prec * mean + like_prec * float(x_obs))
    return float(post_mean), float(math.sqrt(var))


def gaussian_mixture_posterior_params(prior: PriorSpec, x_obs: float, cfg: Config = CFG):
    means, stds, weights = prior.params
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    post_means = np.empty_like(means)
    post_stds = np.empty_like(stds)
    evidences = np.empty_like(weights)
    for j, (m, s) in enumerate(zip(means, stds)):
        post_means[j], post_stds[j] = gaussian_posterior_params(m, s, x_obs, cfg)
        evidences[j] = norm.pdf(x_obs, loc=m, scale=math.sqrt(s**2 + cfg.likelihood_std**2))
    post_weights = weights * evidences
    post_weights /= post_weights.sum()
    return post_means, post_stds, post_weights


def bounded_gaussian_mixture_posterior_params(prior: PriorSpec, x_obs: float, cfg: Config = CFG):
    """Exact posterior of an ACE-style bounded Gaussian mixture under the Gaussian likelihood."""
    means, stds, weights, low, high = prior.params
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    post_means = np.empty_like(means)
    post_stds = np.empty_like(stds)
    evidences = np.empty_like(weights)
    for j, (m, s) in enumerate(zip(means, stds)):
        post_means[j], post_stds[j] = gaussian_posterior_params(m, s, x_obs, cfg)
        evidences[j] = norm.pdf(x_obs, loc=m, scale=math.sqrt(s**2 + cfg.likelihood_std**2))

    post_inside = _bounded_mixture_component_mass(post_means, post_stds, low, high)
    post_weights = weights * evidences * post_inside
    post_weights /= post_weights.sum()
    return post_means, post_stds, post_weights, float(low), float(high)


def exact_posterior_density_np(theta: np.ndarray, prior: PriorSpec, x_obs: float, cfg: Config = CFG) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)

    if prior.family == "gaussian":
        mean, std = prior.params
        pm, ps = gaussian_posterior_params(mean, std, x_obs, cfg)
        return norm.pdf(theta, loc=pm, scale=ps)

    if prior.family == "uniform":
        low, high = prior.params
        a = (low - float(x_obs)) / cfg.likelihood_std
        b = (high - float(x_obs)) / cfg.likelihood_std
        z = norm.cdf(b) - norm.cdf(a)
        raw = norm.pdf((theta - float(x_obs)) / cfg.likelihood_std) / cfg.likelihood_std
        return np.where((theta >= low) & (theta <= high), raw / max(float(z), 1e-300), 0.0)

    if prior.family == "gaussian_mixture":
        pmeans, pstds, pweights = gaussian_mixture_posterior_params(prior, x_obs, cfg)
        result = np.zeros_like(theta, dtype=np.float64)
        for m, s, w in zip(pmeans, pstds, pweights):
            result += w * norm.pdf(theta, loc=m, scale=s)
        return result

    if prior.family == "bounded_gaussian_mixture":
        pmeans, pstds, pweights, low, high = bounded_gaussian_mixture_posterior_params(prior, x_obs, cfg)
        result = np.zeros_like(theta, dtype=np.float64)
        for m, s, w in zip(pmeans, pstds, pweights):
            z = norm.cdf((high - m) / s) - norm.cdf((low - m) / s)
            result += w * norm.pdf(theta, loc=m, scale=s) / max(float(z), 1e-300)
        return np.where((theta >= low) & (theta <= high), result, 0.0)

    raise ValueError(f"Unknown prior family {prior.family!r}.")


def sample_exact_posterior_np(
    rng: np.random.Generator,
    prior: PriorSpec,
    x_obs: float,
    n: int,
    cfg: Config = CFG,
) -> np.ndarray:
    n = int(n)

    if prior.family == "gaussian":
        mean, std = prior.params
        pm, ps = gaussian_posterior_params(mean, std, x_obs, cfg)
        return rng.normal(pm, ps, size=n).astype(np.float32)

    if prior.family == "uniform":
        low, high = prior.params
        a = (low - float(x_obs)) / cfg.likelihood_std
        b = (high - float(x_obs)) / cfg.likelihood_std
        return truncnorm.rvs(
            a,
            b,
            loc=float(x_obs),
            scale=cfg.likelihood_std,
            size=n,
            random_state=rng,
        ).astype(np.float32)

    if prior.family == "gaussian_mixture":
        means, stds, weights = gaussian_mixture_posterior_params(prior, x_obs, cfg)
        ids = rng.choice(len(weights), size=n, p=weights)
        return rng.normal(means[ids], stds[ids]).astype(np.float32)

    if prior.family == "bounded_gaussian_mixture":
        means, stds, weights, low, high = bounded_gaussian_mixture_posterior_params(prior, x_obs, cfg)
        ids = rng.choice(len(weights), size=n, p=weights)
        return _sample_truncated_normals_np(rng, means[ids], stds[ids], low, high).astype(np.float32)

    raise ValueError(f"Unknown prior family {prior.family!r}.")


def sample_empirical_posterior_np(
    rng: np.random.Generator,
    input_cloud: np.ndarray,
    x_obs: float,
    n: int,
    cfg: Config = CFG,
) -> tuple[np.ndarray, np.ndarray]:
    input_cloud = np.asarray(input_cloud, dtype=np.float64).reshape(-1)
    weights = likelihood_density_np(x_obs, input_cloud, cfg)
    weights = weights / max(float(weights.sum()), 1e-300)
    ids = rng.choice(len(input_cloud), size=int(n), replace=True, p=weights)
    return input_cloud[ids].astype(np.float32), weights.astype(np.float64)


# -----------------------------------------------------------------------------
# ACE-style prior meta-distribution
# -----------------------------------------------------------------------------

def _ace_std_range(cfg: Config = CFG) -> tuple[float, float]:
    width = float(cfg.ace_domain_high - cfg.ace_domain_low)
    return (
        float(cfg.ace_component_std_min_fraction * width),
        float(cfg.ace_component_std_max_fraction * width),
    )


def sample_ace_prior_spec_np(
    rng: np.random.Generator,
    cfg: Config = CFG,
) -> tuple[PriorSpec, dict[str, Any]]:
    """Draw one prior from the ACE Appendix-B.1 hierarchy, adapted to particle clouds."""
    low = float(cfg.ace_domain_low)
    high = float(cfg.ace_domain_high)

    if rng.random() >= cfg.ace_mixture_probability:
        prior = uniform_prior(low, high, rf"ACE flat $U[{low:g},{high:g}]$")
        return prior, {
            "is_mixture": False,
            "n_components": 0,
            "configuration": "uniform",
        }

    n_components = int(rng.geometric(cfg.ace_geometric_q))
    std_low, std_high = _ace_std_range(cfg)

    if n_components == 1:
        configuration = "single"
        means = np.asarray([rng.uniform(low, high)], dtype=np.float64)
        stds = np.asarray([rng.uniform(std_low, std_high)], dtype=np.float64)
    else:
        configuration = str(rng.choice(("same_mean", "same_std", "all_different")))
        if configuration == "same_mean":
            means = np.full(n_components, rng.uniform(low, high), dtype=np.float64)
            stds = rng.uniform(std_low, std_high, size=n_components)
        elif configuration == "same_std":
            means = rng.uniform(low, high, size=n_components)
            stds = np.full(n_components, rng.uniform(std_low, std_high), dtype=np.float64)
        else:
            means = rng.uniform(low, high, size=n_components)
            stds = rng.uniform(std_low, std_high, size=n_components)

    weights = rng.dirichlet(np.ones(n_components, dtype=np.float64))
    prior = bounded_gaussian_mixture_prior(
        tuple(means), tuple(stds), tuple(weights), low, high,
        label=f"ACE GMM: K={n_components}, {configuration}",
    )
    return prior, {
        "is_mixture": True,
        "n_components": n_components,
        "configuration": configuration,
    }


def _sample_from_ace_batch_parameters_np(
    rng: np.random.Generator,
    is_mixture: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    raw_weights: np.ndarray,
    component_mask: np.ndarray,
    n_draws: int,
    cfg: Config = CFG,
) -> np.ndarray:
    """Vectorized particle draws from a batch of bounded ACE priors."""
    b, max_components = means.shape
    n_draws = int(n_draws)
    low = float(cfg.ace_domain_low)
    high = float(cfg.ace_domain_high)

    draws = rng.uniform(low, high, size=(b, n_draws))
    mixture_rows = np.flatnonzero(is_mixture)
    if len(mixture_rows) == 0:
        return draws.astype(np.float32)

    m = means[mixture_rows]
    s = stds[mixture_rows]
    w = raw_weights[mixture_rows]
    mask = component_mask[mixture_rows]

    inside = norm.cdf((high - m) / s) - norm.cdf((low - m) / s)
    effective = np.where(mask, w * inside, 0.0)
    effective /= np.sum(effective, axis=1, keepdims=True)
    cumulative = np.cumsum(effective, axis=1)
    cumulative[:, -1] = 1.0

    u_component = rng.random((len(mixture_rows), n_draws))
    ids = np.sum(u_component[..., None] > cumulative[:, None, :], axis=-1)
    selected_means = np.take_along_axis(m, ids, axis=1)
    selected_stds = np.take_along_axis(s, ids, axis=1)
    mixture_draws = _sample_truncated_normals_np(
        rng, selected_means, selected_stds, low, high
    )
    draws[mixture_rows] = mixture_draws
    return draws.astype(np.float32)


def sample_ace_training_batch_np(
    rng: np.random.Generator,
    truth_mode: TruthMode,
    cfg: Config = CFG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Vectorized training batch from the ACE-style prior meta-distribution.

    Each row first receives an independently sampled prior.  The input cloud is
    drawn from that prior.  Then:

      member       -> theta* is selected from the finite input cloud;
      distribution -> theta* is an independent draw from the same prior law.
    """
    b = int(cfg.batch_size)
    k_particles = int(cfg.training_particles)
    low = float(cfg.ace_domain_low)
    high = float(cfg.ace_domain_high)
    std_low, std_high = _ace_std_range(cfg)

    is_mixture = rng.random(b) < cfg.ace_mixture_probability
    n_components = np.where(
        is_mixture,
        rng.geometric(cfg.ace_geometric_q, size=b),
        0,
    ).astype(np.int32)
    max_components = max(1, int(np.max(n_components)))
    component_mask = np.arange(max_components)[None, :] < n_components[:, None]

    # configuration: 0=uniform, 1=single, 2=same mean, 3=same std, 4=all different
    configuration = np.zeros(b, dtype=np.int32)
    configuration[is_mixture & (n_components == 1)] = 1
    multi_rows = np.flatnonzero(is_mixture & (n_components > 1))
    if len(multi_rows):
        configuration[multi_rows] = rng.integers(2, 5, size=len(multi_rows))

    means = rng.uniform(low, high, size=(b, max_components))
    stds = rng.uniform(std_low, std_high, size=(b, max_components))

    same_mean_rows = np.flatnonzero(configuration == 2)
    if len(same_mean_rows):
        means[same_mean_rows] = means[same_mean_rows, :1]

    same_std_rows = np.flatnonzero(configuration == 3)
    if len(same_std_rows):
        stds[same_std_rows] = stds[same_std_rows, :1]

    # Dirichlet(1,...,1) via normalized Gamma(1,1) draws, with variable K.
    gamma_draws = rng.gamma(shape=1.0, scale=1.0, size=(b, max_components))
    gamma_draws = np.where(component_mask, gamma_draws, 0.0)
    raw_weights = np.zeros_like(gamma_draws)
    mixture_rows = np.flatnonzero(is_mixture)
    if len(mixture_rows):
        denom = np.sum(gamma_draws[mixture_rows], axis=1, keepdims=True)
        raw_weights[mixture_rows] = gamma_draws[mixture_rows] / denom

    cloud = _sample_from_ace_batch_parameters_np(
        rng,
        is_mixture,
        means,
        stds,
        raw_weights,
        component_mask,
        k_particles,
        cfg,
    )

    if truth_mode == "member":
        ids = rng.integers(0, k_particles, size=b)
        theta_star = cloud[np.arange(b), ids]
    elif truth_mode == "distribution":
        theta_star = _sample_from_ace_batch_parameters_np(
            rng,
            is_mixture,
            means,
            stds,
            raw_weights,
            component_mask,
            1,
            cfg,
        )[:, 0]
    else:
        raise ValueError(f"Unknown truth_mode {truth_mode!r}.")

    x = theta_star + rng.normal(0.0, cfg.likelihood_std, size=b)

    info = {
        "is_mixture": is_mixture.astype(np.float32),
        "n_components": n_components.astype(np.float32),
        "configuration": configuration.astype(np.float32),
        "prior_cloud_mean": np.mean(cloud, axis=1).astype(np.float32),
        "prior_cloud_std": np.std(cloud, axis=1).astype(np.float32),
        "support_low": np.full(b, low, dtype=np.float32),
        "support_high": np.full(b, high, dtype=np.float32),
    }

    return (
        cloud[..., None].astype(np.float32),
        x[:, None].astype(np.float32),
        theta_star[:, None].astype(np.float32),
        info,
    )


# A fixed-x evaluation suite. Using the SAME x makes prior sensitivity visually obvious.
THETA_TRUE = 1.5
X_FIXED = 1.5
EVAL_CASES = [
    EvalCase(
        "gaussian_broad",
        gaussian_prior(0.0, 2.0, "Broad Gaussian"),
        X_FIXED,
        THETA_TRUE,
        "Simple reference prior; most mass lies inside the ACE training domain.",
    ),
    EvalCase(
        "gaussian_shifted_narrow",
        gaussian_prior(-1.5, 0.50, "Shifted narrow Gaussian"),
        X_FIXED,
        THETA_TRUE,
        "Strong prior-likelihood conflict, but Gaussian support remains nonzero everywhere.",
    ),
    EvalCase(
        "uniform_broad",
        uniform_prior(-4.0, 4.0, "Broad Uniform"),
        X_FIXED,
        THETA_TRUE,
        "Exactly the flat-prior branch of the ACE training meta-distribution.",
    ),
    EvalCase(
        "uniform_zero_support",
        uniform_prior(-4.0, -1.0, "Uniform excluding truth"),
        X_FIXED,
        THETA_TRUE,
        "Hard-support OOD stress test: exact posterior has ZERO mass near theta_true.",
    ),
    EvalCase(
        "ace_bimodal",
        bounded_gaussian_mixture_prior(
            (-2.0, 2.0),
            (0.35, 0.35),
            (0.5, 0.5),
            CFG.ace_domain_low,
            CFG.ace_domain_high,
            "ACE-style bimodal GMM",
        ),
        X_FIXED,
        THETA_TRUE,
        "Explicit multimodality from the meta-training prior family.",
    ),
]


# Visualize a collection of priors from the ACE meta-distribution before training.
def plot_training_prior_family_examples(cfg: Config = CFG) -> None:
    rng = np.random.default_rng(cfg.seed + 101)
    n_examples = 12
    ncols = 4
    nrows = math.ceil(n_examples / ncols)
    grid = np.linspace(cfg.ace_domain_low - 0.5, cfg.ace_domain_high + 0.5, 1000)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.1 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(nrows, ncols)

    for i, ax in enumerate(axes.flat):
        if i >= n_examples:
            ax.axis("off")
            continue
        prior, meta = sample_ace_prior_spec_np(rng, cfg)
        samples = sample_prior_np(rng, prior, 2500)
        ax.hist(samples, bins=55, density=True, alpha=0.25, label="particles")
        ax.plot(grid, prior_density_np(grid, prior), linewidth=2.0, label="prior density")
        if meta["is_mixture"]:
            title = f"K={meta['n_components']} | {meta['configuration']}"
        else:
            title = "flat Uniform"
        ax.set_title(title)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("density")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("ACE-style prior meta-distribution represented by particle clouds", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "00_training_prior_family_examples.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_ace_meta_prior_statistics(cfg: Config = CFG) -> None:
    rng = np.random.default_rng(cfg.seed + 102)
    n = 20_000
    is_mix = rng.random(n) < cfg.ace_mixture_probability
    k = np.zeros(n, dtype=int)
    k[is_mix] = rng.geometric(cfg.ace_geometric_q, size=int(np.sum(is_mix)))

    # Config labels conditional on K>1; this mirrors the equal-probability ACE choice.
    multi_n = int(np.sum(k > 1))
    configs = rng.integers(0, 3, size=multi_n) if multi_n else np.asarray([], dtype=int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    axes[0].bar(["GMM", "flat Uniform"], [np.mean(is_mix), np.mean(~is_mix)])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("probability")
    axes[0].set_title("Prior family")

    mixture_k = k[k > 0]
    max_show = min(10, int(np.max(mixture_k)))
    vals = np.arange(1, max_show + 1)
    probs = np.asarray([np.mean(mixture_k == v) for v in vals])
    tail = np.mean(mixture_k > max_show)
    axes[1].bar(vals, probs)
    if tail > 0:
        axes[1].bar([max_show + 1], [tail])
        axes[1].set_xticks(list(vals) + [max_show + 1], [*map(str, vals), f">{max_show}"])
    axes[1].set_xlabel("number of GMM components K")
    axes[1].set_ylabel("probability | GMM")
    axes[1].set_title(r"$K\sim\mathrm{Geometric}(0.5)$")

    cfg_labels = ["same mean", "same std", "all different"]
    cfg_probs = [np.mean(configs == j) if len(configs) else 0.0 for j in range(3)]
    axes[2].bar(cfg_labels, cfg_probs)
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].set_ylabel(r"probability | $K>1$")
    axes[2].set_title("Mixture geometry")

    fig.suptitle("Meta-distribution statistics used for training", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT / "01_ace_meta_prior_statistics.png", dpi=180, bbox_inches="tight")
    plt.show()


plot_training_prior_family_examples(CFG)
plot_ace_meta_prior_statistics(CFG)


#%% 2
def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]


class AdaLNParticleBlock(eqx.Module):
    norm_attn: eqx.nn.LayerNorm
    norm_ff: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear
    modulation: eqx.nn.Linear

    def __init__(self, hidden: int, heads: int, mlp_dim: int, *, key: Array):
        k_attn, k_ff1, k_ff2, k_mod = jax.random.split(key, 4)
        self.norm_attn = eqx.nn.LayerNorm(hidden)
        self.norm_ff = eqx.nn.LayerNorm(hidden)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden,
            key_size=hidden,
            value_size=hidden,
            output_size=hidden,
            key=k_attn,
        )
        self.ff_in = eqx.nn.Linear(hidden, mlp_dim, key=k_ff1)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden, key=k_ff2)
        modulation = eqx.nn.Linear(hidden, 6 * hidden, key=k_mod)
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


class ConditionalParticleTransport1D(eqx.Module):
    particle_in: eqx.nn.Linear
    x_in: eqx.nn.Linear
    x_hidden: eqx.nn.Linear
    blocks: tuple[AdaLNParticleBlock, ...]
    final_norm: eqx.nn.LayerNorm
    displacement_head: eqx.nn.Linear
    global_scale: float = eqx.field(static=True)
    max_displacement: float = eqx.field(static=True)

    def __init__(self, cfg: Config, *, key: Array):
        keys = jax.random.split(key, cfg.depth + 5)
        self.particle_in = eqx.nn.Linear(1, cfg.hidden_dim, key=keys[0])
        self.x_in = eqx.nn.Linear(1, cfg.hidden_dim, key=keys[1])
        self.x_hidden = eqx.nn.Linear(cfg.hidden_dim, cfg.hidden_dim, key=keys[2])
        self.blocks = tuple(
            AdaLNParticleBlock(
                cfg.hidden_dim,
                cfg.heads,
                cfg.mlp_ratio * cfg.hidden_dim,
                key=keys[3 + i],
            )
            for i in range(cfg.depth)
        )
        self.final_norm = eqx.nn.LayerNorm(cfg.hidden_dim)
        head = eqx.nn.Linear(cfg.hidden_dim, 1, key=keys[-1])
        head = eqx.tree_at(lambda l: l.weight, head, jnp.zeros_like(head.weight))
        head = eqx.tree_at(lambda l: l.bias, head, jnp.zeros_like(head.bias))
        self.displacement_head = head
        self.global_scale = float(cfg.global_theta_scale)
        self.max_displacement = float(cfg.max_normalized_displacement)

    def __call__(self, prior_theta: Array, x: Array) -> Array:
        z0 = prior_theta / self.global_scale
        particles = _linear_tokens(self.particle_in, z0)
        xz = jnp.reshape(x / self.global_scale, (1,))
        conditioning = jax.nn.silu(self.x_in(xz))
        conditioning = self.x_hidden(conditioning)
        for block in self.blocks:
            particles = block(particles, conditioning)
        particles = _layernorm_tokens(self.final_norm, particles)
        delta = self.max_displacement * jnp.tanh(_linear_tokens(self.displacement_head, particles))
        return self.global_scale * (z0 + delta)


print("Model cell ready.")


#%% 3
_ENERGY_EPS = 1e-10


def _stable_abs(x: Array) -> Array:
    return jnp.sqrt(jnp.square(x) + _ENERGY_EPS)


def energy_score_terms_1d(posterior: Array, target_theta: Array) -> tuple[Array, Array, Array]:
    posterior = jnp.reshape(posterior, (-1,))
    target = jnp.reshape(target_theta, ())
    attraction = jnp.mean(_stable_abs(posterior - target))
    pairwise = posterior[:, None] - posterior[None, :]
    repulsion = jnp.mean(_stable_abs(pairwise))
    return attraction - 0.5 * repulsion, attraction, repulsion


def transport_objective(
    model: ConditionalParticleTransport1D,
    prior_batch: Array,
    x_batch: Array,
    theta_star: Array,
):
    posterior = jax.vmap(model)(prior_batch, x_batch)
    scores, attractions, repulsions = jax.vmap(energy_score_terms_1d)(posterior, theta_star)
    input_flat = prior_batch[..., 0]
    output_flat = posterior[..., 0]
    displacement = jnp.abs(output_flat - input_flat)
    metrics = {
        "loss": jnp.mean(scores),
        "energy_score": jnp.mean(scores),
        "attraction": jnp.mean(attractions),
        "repulsion": jnp.mean(repulsions),
        "mean_abs_displacement": jnp.mean(displacement),
        "output_std": jnp.mean(jnp.std(output_flat, axis=1)),
    }
    return metrics["loss"], (metrics, posterior)


_loss_and_grad = eqx.filter_value_and_grad(transport_objective, has_aux=True)


def make_train_step(optimizer: optax.GradientTransformation):
    @eqx.filter_jit
    def step(model, opt_state, prior_batch, x_batch, theta_star):
        (loss, (metrics, posterior)), grads = _loss_and_grad(
            model, prior_batch, x_batch, theta_star
        )
        params = eqx.filter(model, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return model, opt_state, loss, metrics, posterior, grad_norm
    return step


def make_model_and_optimizer(seed: int, cfg: Config = CFG):
    model = ConditionalParticleTransport1D(cfg, key=jax.random.key(seed))
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return model, optimizer, opt_state, make_train_step(optimizer)


def save_model(path: Path, model: ConditionalParticleTransport1D, cfg: Config = CFG) -> None:
    path = Path(path)
    eqx.tree_serialise_leaves(path, model)
    with path.with_suffix(".json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_model(path: Path, seed: int, cfg: Config = CFG) -> ConditionalParticleTransport1D:
    template = ConditionalParticleTransport1D(cfg, key=jax.random.key(seed))
    return eqx.tree_deserialise_leaves(Path(path), template)


def evaluate_model(model: ConditionalParticleTransport1D, prior_particles: np.ndarray, x_obs: float) -> np.ndarray:
    prior_particles = np.asarray(prior_particles, dtype=np.float32).reshape(-1, 1)
    output = model(jnp.asarray(prior_particles), jnp.asarray([x_obs], dtype=jnp.float32))
    return np.asarray(jax.device_get(output[:, 0]), dtype=np.float32)


#%% 4
def kde_on_grid(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 3 or float(np.std(samples)) < 1e-10:
        return np.zeros_like(grid, dtype=np.float64)
    try:
        kde = gaussian_kde(samples)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(123)
        kde = gaussian_kde(samples + rng.normal(0.0, 1e-5, size=len(samples)))
    return kde(np.asarray(grid, dtype=np.float64))


def energy_distance_1d(a: np.ndarray, b: np.ndarray, max_points: int = 2500) -> float:
    rng = np.random.default_rng(911)
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) > max_points:
        a = rng.choice(a, size=max_points, replace=False)
    if len(b) > max_points:
        b = rng.choice(b, size=max_points, replace=False)
    ab = np.mean(np.abs(a[:, None] - b[None, :]))
    aa = np.mean(np.abs(a[:, None] - a[None, :]))
    bb = np.mean(np.abs(b[:, None] - b[None, :]))
    return float(2.0 * ab - aa - bb)


def uniform_support_violation(samples: np.ndarray, prior: PriorSpec) -> float:
    if prior.family != "uniform":
        return float("nan")
    low, high = prior.params
    s = np.asarray(samples)
    return float(np.mean((s < low) | (s > high)))


def mean_nearest_input_distance(input_cloud: np.ndarray, output_cloud: np.ndarray) -> float:
    inp = np.sort(np.asarray(input_cloud, dtype=np.float64).reshape(-1))
    out = np.asarray(output_cloud, dtype=np.float64).reshape(-1)
    idx = np.searchsorted(inp, out)
    idx0 = np.clip(idx - 1, 0, len(inp) - 1)
    idx1 = np.clip(idx, 0, len(inp) - 1)
    d = np.minimum(np.abs(out - inp[idx0]), np.abs(out - inp[idx1]))
    return float(np.mean(d))


def evaluate_case_once(
    model: ConditionalParticleTransport1D,
    case: EvalCase,
    rng: np.random.Generator,
    cfg: Config = CFG,
) -> dict[str, Any]:
    input_cloud = sample_prior_np(rng, case.prior, cfg.eval_particles)
    output_cloud = evaluate_model(model, input_cloud, case.x_obs)

    exact_samples = sample_exact_posterior_np(
        rng, case.prior, case.x_obs, cfg.exact_reference_samples, cfg
    )
    empirical_samples, empirical_weights = sample_empirical_posterior_np(
        rng, input_cloud, case.x_obs, cfg.exact_reference_samples, cfg
    )

    metrics = {
        "w1_to_continuous": float(wasserstein_distance(output_cloud, exact_samples)),
        "w1_to_empirical": float(wasserstein_distance(output_cloud, empirical_samples)),
        "energy_to_continuous": energy_distance_1d(output_cloud, exact_samples),
        "mean_error_continuous": float(abs(np.mean(output_cloud) - np.mean(exact_samples))),
        "std_error_continuous": float(abs(np.std(output_cloud) - np.std(exact_samples))),
        "support_violation": uniform_support_violation(output_cloud, case.prior),
        "nearest_input_distance": mean_nearest_input_distance(input_cloud, output_cloud),
    }
    return {
        "case": case,
        "input": input_cloud,
        "output": output_cloud,
        "exact_samples": exact_samples,
        "empirical_samples": empirical_samples,
        "empirical_weights": empirical_weights,
        "metrics": metrics,
    }


# Fixed diagnostic cases during training.
TRAIN_DIAGNOSTIC_CASES = {
    "gaussian": EVAL_CASES[0],
    "uniform": EVAL_CASES[2],
    "zero_support": EVAL_CASES[3],
}


def quick_diagnostic_metrics(
    model: ConditionalParticleTransport1D,
    rng: np.random.Generator,
    cfg: Config = CFG,
) -> dict[str, float]:
    result = {}
    for key, case in TRAIN_DIAGNOSTIC_CASES.items():
        evaluation = evaluate_case_once(model, case, rng, cfg)
        result[f"w1_{key}"] = evaluation["metrics"]["w1_to_continuous"]
        if key == "zero_support":
            result["support_violation_zero_support"] = evaluation["metrics"]["support_violation"]
    return result


#%% 5
def train_one_mode(truth_mode: TruthMode, seed_offset: int = 0, cfg: Config = CFG):
    model_seed = cfg.seed + seed_offset
    model, optimizer, opt_state, train_step = make_model_and_optimizer(model_seed, cfg)
    rng = np.random.default_rng(cfg.seed + 10_000 + seed_offset)
    diagnostic_rng = np.random.default_rng(cfg.seed + 20_000 + seed_offset)

    history = {name: [] for name in (
        "step",
        "energy_score",
        "attraction",
        "repulsion",
        "mean_abs_displacement",
        "output_std",
        "grad_norm",
        "mixture_fraction",
        "mean_components",
        "mean_prior_cloud_std",
        "global_support_violation_train",
        "w1_gaussian",
        "w1_uniform",
        "w1_zero_support",
        "support_violation_zero_support",
    )}
    snapshots: dict[str, np.ndarray] = {}

    for step in range(1, cfg.training_steps + 1):
        prior_batch, x_batch, theta_star, info = sample_ace_training_batch_np(
            rng, truth_mode, cfg
        )
        model, opt_state, loss, metrics, posterior, grad_norm = train_step(
            model,
            opt_state,
            jnp.asarray(prior_batch),
            jnp.asarray(x_batch),
            jnp.asarray(theta_star),
        )

        posterior_np = np.asarray(jax.device_get(posterior[..., 0]), dtype=np.float32)
        low = float(cfg.ace_domain_low)
        high = float(cfg.ace_domain_high)
        train_support_violation = float(np.mean((posterior_np < low) | (posterior_np > high)))

        host = jax.device_get(metrics)
        mixture_mask = info["is_mixture"] > 0.5
        mean_components = (
            float(np.mean(info["n_components"][mixture_mask]))
            if np.any(mixture_mask)
            else float("nan")
        )
        row = {
            "step": float(step),
            "energy_score": float(host["energy_score"]),
            "attraction": float(host["attraction"]),
            "repulsion": float(host["repulsion"]),
            "mean_abs_displacement": float(host["mean_abs_displacement"]),
            "output_std": float(host["output_std"]),
            "grad_norm": float(jax.device_get(grad_norm)),
            "mixture_fraction": float(np.mean(info["is_mixture"])),
            "mean_components": mean_components,
            "mean_prior_cloud_std": float(np.mean(info["prior_cloud_std"])),
            "global_support_violation_train": train_support_violation,
            "w1_gaussian": float("nan"),
            "w1_uniform": float("nan"),
            "w1_zero_support": float("nan"),
            "support_violation_zero_support": float("nan"),
        }

        if step == 1 or step % cfg.diagnostic_every == 0 or step == cfg.training_steps:
            d = quick_diagnostic_metrics(model, diagnostic_rng, cfg)
            row.update(d)

        for name in history:
            history[name].append(row[name])

        if step in cfg.snapshot_steps or step == cfg.training_steps:
            for key, case in TRAIN_DIAGNOSTIC_CASES.items():
                seed_map = {"gaussian": 101, "uniform": 202, "zero_support": 303}
                mode_offset = 0 if truth_mode == "member" else 1000
                fixed_rng = np.random.default_rng(cfg.seed + 70_000 + mode_offset + seed_map[key])
                inp = sample_prior_np(fixed_rng, case.prior, cfg.eval_particles)
                out = evaluate_model(model, inp, case.x_obs)
                snapshots[f"{step}_{key}_input"] = inp.astype(np.float32)
                snapshots[f"{step}_{key}_output"] = out.astype(np.float32)

        if step == 1 or step % cfg.log_every == 0 or step == cfg.training_steps:
            print(
                f"[{truth_mode:12s}] step {step:5d}/{cfg.training_steps} | "
                f"ES {row['energy_score']:.5f} | "
                f"disp {row['mean_abs_displacement']:.3f} | "
                f"grad {row['grad_norm']:.2e} | "
                f"GMM frac {row['mixture_fraction']:.2f} | "
                f"mean K {row['mean_components']:.2f} | "
                f"global-support-viol {row['global_support_violation_train']:.3f}"
            )

    model_path = OUT / f"model_{truth_mode}.eqx"
    save_model(model_path, model, cfg)

    history_path = OUT / f"history_{truth_mode}.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history.keys()))
        writer.writeheader()
        for i in range(len(history["step"])):
            writer.writerow({k: history[k][i] for k in history})

    np.savez_compressed(OUT / f"snapshots_{truth_mode}.npz", **snapshots)
    return model, history, snapshots


# Toggle either flag to reuse saved checkpoints instead of retraining.
TRAIN_MEMBER_MODEL = True
TRAIN_DISTRIBUTION_MODEL = True

if TRAIN_MEMBER_MODEL:
    MEMBER_MODEL, MEMBER_HISTORY, MEMBER_SNAPSHOTS = train_one_mode("member", seed_offset=0, cfg=CFG)
else:
    MEMBER_MODEL = load_model(OUT / "model_member.eqx", CFG.seed, CFG)

if TRAIN_DISTRIBUTION_MODEL:
    DISTRIBUTION_MODEL, DISTRIBUTION_HISTORY, DISTRIBUTION_SNAPSHOTS = train_one_mode(
        "distribution", seed_offset=1, cfg=CFG
    )
else:
    DISTRIBUTION_MODEL = load_model(OUT / "model_distribution.eqx", CFG.seed + 1, CFG)


#%% 6
def load_history_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(data[name], dtype=np.float64) for name in data.dtype.names}


def rolling_mean_nanaware(x: np.ndarray, window: int = 80) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return x.copy()
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        chunk = x[lo:i + 1]
        finite = chunk[np.isfinite(chunk)]
        if len(finite):
            out[i] = np.mean(finite)
    return out


def forward_fill_sparse(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    last = np.nan
    for i in range(len(x)):
        if np.isfinite(x[i]):
            last = x[i]
        elif np.isfinite(last):
            x[i] = last
    return x


def plot_training_diagnostics_both(cfg: Config = CFG) -> None:
    histories = {
        "member": load_history_csv(OUT / "history_member.csv"),
        "distribution": load_history_csv(OUT / "history_distribution.csv"),
    }
    fig, axes = plt.subplots(3, 3, figsize=(18, 12.5))

    for label, h in histories.items():
        step = h["step"]
        axes[0, 0].plot(step, rolling_mean_nanaware(h["energy_score"]), label=label)
        axes[0, 1].plot(step, np.maximum(rolling_mean_nanaware(h["grad_norm"]), 1e-14), label=label)
        axes[0, 2].plot(step, rolling_mean_nanaware(h["mean_abs_displacement"]), label=label)

        axes[1, 0].plot(step, rolling_mean_nanaware(h["global_support_violation_train"]), label=label)
        axes[1, 1].plot(step, rolling_mean_nanaware(h["mixture_fraction"]), label=label)
        axes[1, 2].plot(step, rolling_mean_nanaware(h["mean_components"]), label=label)

        axes[2, 0].plot(step, forward_fill_sparse(h["w1_gaussian"]), label=f"{label}: Gaussian")
        axes[2, 0].plot(step, forward_fill_sparse(h["w1_uniform"]), linestyle="--", label=f"{label}: Uniform")
        axes[2, 1].plot(step, forward_fill_sparse(h["w1_zero_support"]), label=label)
        axes[2, 2].plot(step, forward_fill_sparse(h["support_violation_zero_support"]), label=label)

    axes[0, 0].set_title("Energy-score training loss")
    axes[0, 0].set_ylabel("energy score")
    axes[0, 1].set_title("Gradient norm")
    axes[0, 1].set_yscale("log")
    axes[0, 2].set_title("Mean particle displacement")

    axes[1, 0].set_title("Output outside ACE global support")
    axes[1, 0].set_ylabel("fraction")
    axes[1, 1].set_title("GMM fraction in minibatch")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 2].set_title("Mean number of GMM components")

    axes[2, 0].set_title("Fixed-prior W1 during training")
    axes[2, 0].set_ylabel("Wasserstein-1")
    axes[2, 1].set_title("Zero-support W1 during training")
    axes[2, 1].set_ylabel("Wasserstein-1")
    axes[2, 2].set_title("Zero-support support violation")
    axes[2, 2].set_ylabel("fraction outside supplied Uniform support")

    for ax in axes.flat:
        ax.set_xlabel("optimizer step")
        ax.legend(fontsize=8)
    fig.suptitle("Training diagnostics under ACE-style prior meta-amortization", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUT / "10_training_diagnostics_comparison.png", dpi=190, bbox_inches="tight")
    plt.show()


def plot_training_snapshots(truth_mode: TruthMode, cfg: Config = CFG) -> None:
    data = np.load(OUT / f"snapshots_{truth_mode}.npz")
    snapshot_steps = [s for s in cfg.snapshot_steps if f"{s}_gaussian_output" in data]
    if cfg.training_steps not in snapshot_steps and f"{cfg.training_steps}_gaussian_output" in data:
        snapshot_steps.append(cfg.training_steps)
    case_keys = ["gaussian", "uniform", "zero_support"]

    fig, axes = plt.subplots(len(snapshot_steps), len(case_keys), figsize=(15, 2.7 * len(snapshot_steps)), sharex=False)
    axes = np.atleast_2d(axes)

    for r, step in enumerate(snapshot_steps):
        for c, key in enumerate(case_keys):
            case = TRAIN_DIAGNOSTIC_CASES[key]
            inp = data[f"{step}_{key}_input"]
            out = data[f"{step}_{key}_output"]
            span = max(1.0, float(np.std(inp)) * 4.0)
            lo = min(np.min(inp), np.min(out), case.theta_true, case.x_obs) - 0.5 * span
            hi = max(np.max(inp), np.max(out), case.theta_true, case.x_obs) + 0.5 * span
            if case.prior.family == "uniform":
                pl, ph = case.prior.params
                lo = min(lo, pl - 0.5)
                hi = max(hi, ph + 0.5)
            grid = np.linspace(lo, hi, 700)
            exact = exact_posterior_density_np(grid, case.prior, case.x_obs, cfg)
            kde = kde_on_grid(out, grid)
            prior_d = prior_density_np(grid, case.prior)
            prior_d = prior_d / max(float(prior_d.max()), 1e-12) * max(float(exact.max()), 1e-12) * 0.55

            ax = axes[r, c]
            ax.plot(grid, exact, linewidth=2.0, label="exact posterior")
            ax.plot(grid, kde, linewidth=1.7, label="model")
            ax.plot(grid, prior_d, linestyle=":", linewidth=1.4, label="prior (scaled)")
            ax.axvline(case.theta_true, linestyle="--", linewidth=1.0, label=r"$\theta^*$" if (r == 0 and c == 0) else None)
            if r == 0:
                ax.set_title(case.prior.label)
            if c == 0:
                ax.set_ylabel(f"step {step}\ndensity")
            ax.set_xlabel(r"$\theta$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(f"Training snapshots — {truth_mode} supervision", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / f"11_training_snapshots_{truth_mode}.png", dpi=190, bbox_inches="tight")
    plt.show()


plot_training_diagnostics_both(CFG)
plot_training_snapshots("member", CFG)
plot_training_snapshots("distribution", CFG)


#%% 7
if "MEMBER_MODEL" not in globals():
    MEMBER_MODEL = load_model(OUT / "model_member.eqx", CFG.seed, CFG)
if "DISTRIBUTION_MODEL" not in globals():
    DISTRIBUTION_MODEL = load_model(OUT / "model_distribution.eqx", CFG.seed + 1, CFG)

EVAL_MODELS = {
    "member": MEMBER_MODEL,
    "distribution": DISTRIBUTION_MODEL,
}

EVALUATIONS: dict[str, dict[str, dict[str, Any]]] = {mode: {} for mode in EVAL_MODELS}
for mode_id, (mode, model) in enumerate(EVAL_MODELS.items()):
    for case_id, case in enumerate(EVAL_CASES):
        rng = np.random.default_rng(CFG.seed + 50_000 + 1000 * mode_id + case_id)
        EVALUATIONS[mode][case.name] = evaluate_case_once(model, case, rng, CFG)

rows = []
for mode in EVAL_MODELS:
    for case in EVAL_CASES:
        metrics = EVALUATIONS[mode][case.name]["metrics"]
        rows.append({"mode": mode, "case": case.name, **metrics})

with (OUT / "evaluation_metrics.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\nFinal evaluation metrics")
for row in rows:
    print(row)


#%% 8
def case_plot_grid(case: EvalCase, evaluation: dict[str, Any], cfg: Config = CFG):
    inp = evaluation["input"]
    out = evaluation["output"]
    exact = evaluation["exact_samples"]
    all_values = np.concatenate([inp, out, exact, np.asarray([case.theta_true, case.x_obs])])
    qlo, qhi = np.quantile(all_values, [0.002, 0.998])
    span = max(1.0, qhi - qlo)
    lo = qlo - 0.18 * span
    hi = qhi + 0.18 * span
    if case.prior.family == "uniform":
        pl, ph = case.prior.params
        lo = min(lo, pl - 0.5)
        hi = max(hi, ph + 0.5)
    return np.linspace(lo, hi, cfg.density_grid_points)


def plot_prior_sensitivity_grid(cfg: Config = CFG) -> None:
    fig, axes = plt.subplots(2, len(EVAL_CASES), figsize=(4.0 * len(EVAL_CASES), 7.2), squeeze=False)

    for r, mode in enumerate(("member", "distribution")):
        for c, case in enumerate(EVAL_CASES):
            ev = EVALUATIONS[mode][case.name]
            grid = case_plot_grid(case, ev, cfg)
            exact_d = exact_posterior_density_np(grid, case.prior, case.x_obs, cfg)
            model_d = kde_on_grid(ev["output"], grid)
            prior_d = prior_density_np(grid, case.prior)
            if prior_d.max() > 0:
                prior_d = prior_d / prior_d.max() * max(float(exact_d.max()), 1e-12) * 0.55

            ax = axes[r, c]
            ax.plot(grid, exact_d, linewidth=2.2, label="exact posterior")
            ax.plot(grid, model_d, linewidth=1.9, label="P-SAPT particles (KDE)")
            ax.plot(grid, prior_d, linestyle=":", linewidth=1.4, label="prior (scaled)")
            ax.axvline(case.theta_true, linestyle="--", linewidth=1.0, label=r"$\theta^*$")
            ax.axvline(case.x_obs, linestyle="-.", linewidth=1.0, label=r"$x_o$")
            if case.prior.family == "uniform":
                low, high = case.prior.params
                ax.axvspan(low, high, alpha=0.06)
            if r == 0:
                ax.set_title(case.prior.label)
            if c == 0:
                ax.set_ylabel(f"{mode}\ndensity")
            ax.set_xlabel(r"$\theta$")
            m = ev["metrics"]
            txt = f"W1={m['w1_to_continuous']:.3f}"
            if np.isfinite(m["support_violation"]):
                txt += f"\nsupport viol.={m['support_violation']:.2%}"
            ax.text(0.02, 0.96, txt, transform=ax.transAxes, va="top", fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle(
        rf"Prior sensitivity at fixed $x_o={X_FIXED}$: does the model obey the supplied prior?",
        fontsize=17,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT / "20_prior_sensitivity_grid.png", dpi=200, bbox_inches="tight")
    plt.show()


plot_prior_sensitivity_grid(CFG)


#%% 9
def plot_transport_case(case: EvalCase, cfg: Config = CFG) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    for r, mode in enumerate(("member", "distribution")):
        ev = EVALUATIONS[mode][case.name]
        inp = ev["input"]
        out = ev["output"]
        grid = case_plot_grid(case, ev, cfg)
        exact_d = exact_posterior_density_np(grid, case.prior, case.x_obs, cfg)
        empirical_d = kde_on_grid(ev["empirical_samples"], grid)
        out_d = kde_on_grid(out, grid)

        ax = axes[r, 0]
        ax.plot(grid, exact_d, linewidth=2.1, label="continuous exact posterior")
        ax.plot(grid, empirical_d, linestyle="--", linewidth=1.5, label="empirical-cloud posterior")
        ax.plot(grid, out_d, linewidth=1.8, label="model output")
        ax.axvline(case.theta_true, linestyle=":", linewidth=1.2, label=r"$\theta^*$")
        ax.set_title(f"{mode}: posterior law")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

        ax = axes[r, 1]
        n = min(cfg.plot_transport_particles, len(inp))
        ids = np.linspace(0, len(inp) - 1, n, dtype=int)
        p0 = inp[ids]
        p1 = out[ids]
        y0 = np.zeros(n)
        y1 = np.ones(n)
        for a, b in zip(p0, p1):
            ax.plot([a, b], [0, 1], linewidth=0.55, alpha=0.18)
        ax.scatter(p0, y0, s=10, alpha=0.45, label="input particles")
        ax.scatter(p1, y1, s=10, alpha=0.45, label="output particles")
        ax.axvline(case.theta_true, linestyle="--", linewidth=1.0, label=r"$\theta^*$")
        if case.prior.family == "uniform":
            low, high = case.prior.params
            ax.axvspan(low, high, alpha=0.06, label="prior support")
        ax.set_yticks([0, 1], ["input", "output"])
        ax.set_xlabel(r"$\theta$")
        ax.set_title(f"{mode}: one-step particle transport")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"{case.prior.label} | {case.note}", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / f"21_transport_{case.name}.png", dpi=190, bbox_inches="tight")
    plt.show()


for _case in EVAL_CASES:
    plot_transport_case(_case, CFG)


#%% 10
def plot_zero_support_stress(cfg: Config = CFG) -> None:
    case = next(c for c in EVAL_CASES if c.name == "uniform_zero_support")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    for r, mode in enumerate(("member", "distribution")):
        ev = EVALUATIONS[mode][case.name]
        inp, out = ev["input"], ev["output"]
        low, high = case.prior.params
        grid = np.linspace(low - 1.0, max(case.theta_true, case.x_obs) + 1.0, cfg.density_grid_points)
        exact_d = exact_posterior_density_np(grid, case.prior, case.x_obs, cfg)
        out_d = kde_on_grid(out, grid)

        ax = axes[r, 0]
        ax.plot(grid, exact_d, linewidth=2.2, label="exact posterior")
        ax.plot(grid, out_d, linewidth=1.9, label="model output")
        ax.axvspan(low, high, alpha=0.08, label="supplied prior support")
        ax.axvline(case.theta_true, linestyle="--", linewidth=1.3, label=r"$\theta^*$")
        ax.axvline(case.x_obs, linestyle="-.", linewidth=1.1, label=r"$x_o$")
        ax.set_title(f"{mode}: density")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

        ax = axes[r, 1]
        ax.hist(inp, bins=55, density=True, alpha=0.30, label="input prior cloud")
        ax.hist(out, bins=55, density=True, alpha=0.38, label="output cloud")
        ax.axvspan(low, high, alpha=0.08)
        ax.axvline(case.theta_true, linestyle="--", linewidth=1.3)
        viol = ev["metrics"]["support_violation"]
        ax.set_title(f"{mode}: support violation = {viol:.2%}")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Decisive test: the prior assigns zero support near the generating truth",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(OUT / "22_zero_support_decisive_test.png", dpi=200, bbox_inches="tight")
    plt.show()


plot_zero_support_stress(CFG)


#%% 11
def plot_metric_summary(cfg: Config = CFG) -> None:
    case_names = [c.name for c in EVAL_CASES]
    case_labels = [c.prior.label for c in EVAL_CASES]
    modes = ["member", "distribution"]

    metrics_to_plot = [
        ("w1_to_continuous", "W1 to continuous posterior"),
        ("w1_to_empirical", "W1 to empirical-cloud posterior"),
        ("nearest_input_distance", "Mean output-to-nearest-input distance"),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 5.3))
    x = np.arange(len(case_names))
    width = 0.37

    for ax, (metric, title) in zip(axes, metrics_to_plot):
        for j, mode in enumerate(modes):
            vals = [EVALUATIONS[mode][name]["metrics"][metric] for name in case_names]
            ax.bar(x + (j - 0.5) * width, vals, width=width, label=mode)
        ax.set_xticks(x, case_labels, rotation=28, ha="right")
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.legend()

    fig.suptitle("Which posterior semantics did each model learn?", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT / "23_metric_summary.png", dpi=190, bbox_inches="tight")
    plt.show()


plot_metric_summary(CFG)


#%% 12
# This asks a stronger question than a handful of test cases: does the posterior mean move
# continuously with the supplied prior, as Bayes' rule says it should?
def plot_prior_response_map(cfg: Config = CFG) -> None:
    x_obs = X_FIXED
    gaussian_means = np.linspace(-3.0, 3.0, 25)
    gaussian_std = 0.65
    rng = np.random.default_rng(cfg.seed + 88_001)

    exact_means = []
    member_means = []
    distribution_means = []

    for m in gaussian_means:
        prior = gaussian_prior(float(m), gaussian_std)
        pm, _ = gaussian_posterior_params(float(m), gaussian_std, x_obs, cfg)
        exact_means.append(pm)
        cloud = sample_prior_np(rng, prior, cfg.eval_particles)
        member_means.append(float(np.mean(evaluate_model(MEMBER_MODEL, cloud, x_obs))))
        distribution_means.append(float(np.mean(evaluate_model(DISTRIBUTION_MODEL, cloud, x_obs))))

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.plot(gaussian_means, exact_means, linewidth=2.3, label="exact posterior mean")
    ax.plot(gaussian_means, member_means, marker="o", markersize=3, linewidth=1.4, label="member model")
    ax.plot(gaussian_means, distribution_means, marker="s", markersize=3, linewidth=1.4, label="distribution model")
    ax.axhline(x_obs, linestyle=":", linewidth=1.2, label=r"$x_o$")
    ax.set_xlabel("supplied Gaussian prior mean")
    ax.set_ylabel("posterior mean")
    ax.set_title(rf"Prior-response map at fixed $x_o={x_obs}$, prior std={gaussian_std}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "24_prior_response_map.png", dpi=190, bbox_inches="tight")
    plt.show()


plot_prior_response_map(CFG)


#%% 13
#%% 13) Random ACE-prior generalization panel (no retraining needed)
# This asks whether the network handles fresh priors drawn from the same enormous
# meta-distribution, rather than only the hand-designed evaluation cases above.
def plot_random_ace_prior_generalization(cfg: Config = CFG) -> None:
    if "MEMBER_MODEL" not in globals():
        member_model = load_model(OUT / "model_member.eqx", CFG.seed, CFG)
    else:
        member_model = MEMBER_MODEL
    if "DISTRIBUTION_MODEL" not in globals():
        distribution_model = load_model(OUT / "model_distribution.eqx", CFG.seed + 1, CFG)
    else:
        distribution_model = DISTRIBUTION_MODEL

    rng = np.random.default_rng(cfg.seed + 99_101)
    priors = [sample_ace_prior_spec_np(rng, cfg)[0] for _ in range(cfg.random_ace_eval_priors)]
    models = [("member", member_model), ("distribution", distribution_model)]

    fig, axes = plt.subplots(
        2,
        len(priors),
        figsize=(3.8 * len(priors), 7.0),
        squeeze=False,
    )

    for col, prior in enumerate(priors):
        for row, (mode, model) in enumerate(models):
            local_rng = np.random.default_rng(cfg.seed + 100_000 + 1000 * row + col)
            cloud = sample_prior_np(local_rng, prior, cfg.eval_particles)
            out = evaluate_model(model, cloud, X_FIXED)
            exact = sample_exact_posterior_np(local_rng, prior, X_FIXED, cfg.exact_reference_samples, cfg)

            all_values = np.concatenate([cloud, out, exact, np.asarray([X_FIXED])])
            lo = min(cfg.ace_domain_low - 0.25, float(np.quantile(all_values, 0.002)) - 0.25)
            hi = max(cfg.ace_domain_high + 0.25, float(np.quantile(all_values, 0.998)) + 0.25)
            grid = np.linspace(lo, hi, cfg.density_grid_points)

            exact_d = exact_posterior_density_np(grid, prior, X_FIXED, cfg)
            model_d = kde_on_grid(out, grid)
            prior_d = prior_density_np(grid, prior)
            if prior_d.max() > 0:
                prior_d = prior_d / prior_d.max() * max(float(exact_d.max()), 1e-12) * 0.50

            ax = axes[row, col]
            ax.plot(grid, exact_d, linewidth=2.0, label="exact posterior")
            ax.plot(grid, model_d, linewidth=1.7, label="model KDE")
            ax.plot(grid, prior_d, linestyle=":", linewidth=1.3, label="prior (scaled)")
            ax.axvline(X_FIXED, linestyle="--", linewidth=1.0, label=r"$x_o$")
            ax.axvspan(cfg.ace_domain_low, cfg.ace_domain_high, alpha=0.035)
            w1 = wasserstein_distance(out, exact)
            ax.text(0.03, 0.95, f"W1={w1:.3f}", transform=ax.transAxes, va="top", fontsize=8)
            if row == 0:
                ax.set_title(prior.label, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{mode}\ndensity")
            ax.set_xlabel(r"$\theta$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(
        rf"Fresh priors from the ACE meta-distribution at fixed $x_o={X_FIXED}$",
        fontsize=17,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT / "25_random_ace_prior_generalization.png", dpi=190, bbox_inches="tight")
    plt.show()


plot_random_ace_prior_generalization(CFG)


#%% 14
summary = {
    "hypothesis": (
        "ACE-style meta-amortization over a very broad prior family may teach a genuine "
        "prior-conditioned Bayes operator. The decisive extrapolation test remains the Uniform "
        "prior with zero support near theta_true: a Bayesian updater must keep posterior mass "
        "inside the supplied support."
    ),
    "prior_meta_distribution": {
        "source": "ACE Appendix B.1, adapted from histogram priors to particle clouds",
        "mixture_probability": CFG.ace_mixture_probability,
        "flat_uniform_probability": 1.0 - CFG.ace_mixture_probability,
        "geometric_q": CFG.ace_geometric_q,
        "domain": [CFG.ace_domain_low, CFG.ace_domain_high],
        "std_fraction_range": [
            CFG.ace_component_std_min_fraction,
            CFG.ace_component_std_max_fraction,
        ],
        "mixture_configurations": [
            "same means / different stds",
            "different means / same std",
            "all means and stds different",
        ],
        "mixture_weights": "Dirichlet(alpha0=1)",
    },
    "truth_modes": {
        "member": "theta* is sampled directly from Theta_in; target is empirical-cloud Bayes.",
        "distribution": "theta* is an independent draw from the same underlying prior law as Theta_in.",
    },
    "fixed_x": X_FIXED,
    "theta_true_marker": THETA_TRUE,
    "evaluation_cases": [
        {
            "name": case.name,
            "prior": case.prior.label,
            "note": case.note,
        }
        for case in EVAL_CASES
    ],
}
with (OUT / "experiment_summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

print("\nExperiment complete. Outputs saved under:", OUT.resolve())


