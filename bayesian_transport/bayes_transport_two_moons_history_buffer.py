#%% 0) Imports, configuration, and experiment constants
"""Two moons with changing latent parameters: does posterior-cloud history help?

Standalone companion to bayes_transport_two_moons_proposal_buffer.py; that file is unchanged.
Run #%% cells in order. There is deliberately no main() function.

Every training row contains ONE theta and ONE simulated 2-D observation. Every network call at
training and testing takes ONE 2-D observation and a particle cloud. A configurable sequence of
observations is processed one at a time, carrying information only through the cloud. No sequence
encoder, concatenated observations, test-time gradients, or extra inference simulations are used.

Three identically initialized transports share exactly the same simulator pairs, stored loss
weights, minibatch order, particle counts, and number of updates:
    single_observation: fresh exact-prior inputs only (matched amortized SBI baseline);
    no_replay: fresh prior or original truth-anchored interpolation, without posterior replay;
    buffered: fresh prior, interpolation, or the SAME row's previous posterior.
Input sources are mutually exclusive: default probabilities are 0.25 interpolation, 0.50 stored
posterior, and 0.25 fresh exact prior. On first visits, a requested but unavailable posterior falls
back to a fresh prior. Interpolation retains its own probability and original anchor controls.
The finite simulation buffer stores (x, theta*, loss weight) and separate detached posterior clouds
for each model. Each training call replaces that model's cloud for the selected row. Reusing it
keeps the SAME x, theta*, and weight together, instead of looking up a nearby observation.
After the simulation budget is exhausted, ONLY stored x/theta* pairs are used, still choosing the
input-cloud source on each visit. The target remains theta*, scored with the energy score.
Proposal-targeted simulation acquisition and importance correction are independent opt-ins, both
disabled by default: default simulations come from the uniform prior and all loss weights are one.
If targeting is enabled without correction, the loss follows the acquisition distribution.
No training transitions are introduced. Neither held-out theta nor diagnostic likelihoods enter
learned inference. The matched baseline is a single-observation transport, not external NPE/SNPE.

The original single-datum and categorical-proposal-start evaluations are compared with raw
posterior carry-over and a transition/defensive-refresh version. A same-call-budget x_T-only
refinement control distinguishes benefits from history from benefits of repeated computation.
Repeated conditioning is a learned refinement heuristic, NOT repeated independent evidence.

Crucial distinction: p(theta_T | x_T) and p(theta_T | x_1,...,x_T) are different targets when
parameters are temporally related. We report distance to both numerical grid references (and
the original uniform-prior posterior), plus proper energy scores at held-out theta_T, posterior-mean error, coverage, and paired uncertainty.
Independent parameters are a negative control: their two reference targets coincide exactly.
The original training loss does not teach a Bayes update for arbitrary incoming priors. Therefore
neural history propagation is an empirical transfer experiment, not a guaranteed Bayesian filter.

Evaluation transitions are configurable: identity, brownian, driftBrownian, constantVelocity,
rotation, ou, custom, and relocation, plus optional identity/independent controls. Boundaries reflect into
the original square. These dynamics can change the time-marginal prior, so we also evaluate a
dynamics-only particle control and distance to the original uniform-prior posterior. Numerical filtering
uses only the toy diagnostic likelihood, never for fitting or choosing a neural method.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
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
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist

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
    output_dir: str = "plots/bayes_transport_two_moons_history_buffer"

    # Exact two-moons benchmark from Greenberg et al. (2019), Appendix A.5.1
    prior_low: float = -1.0
    prior_high: float = 1.0
    radial_mean: float = 0.1
    radial_std: float = 0.01
    crescent_x_offset: float = 0.25
    observed_x1: float = 0.0
    observed_x2: float = 0.0

    # Maximum examples per optimizer update; acquisition/replay keep the final partial batch.
    batch_size: int = 128

    # Particle transport -- intentionally kept very close to the previous script.
    # Only the MAXIMUM training particle count is configured.  When variable_training_particles=True,
    # each optimizer step draws one set size from an automatically-derived geometric ladder ending at
    # max_training_particles.  Setting the flag False always uses the maximum, reproducing the previous
    # fixed-particle-count training path.  Evaluation remains independently controlled by eval_particles.
    max_training_particles: int = 16 * 4*1
    variable_training_particles: bool = True
    eval_particles: int = 256  # Repeated sequence evaluation; attention costs O(M^2).
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
    simulation_budget: int = 10_000  # Exact number of fresh training simulator calls.
    replay_epochs: int = 200  # Full shuffled passes AFTER the acquisition stage; may be zero.
    learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip_norm: float = 5000.0
    log_every: int = 1250

    # Particle-native categorical proposal for simulator acquisition.  After a short warm-up,
    # a one-step posterior cloud at x_o is used only to bias WHICH fresh prior candidates receive
    # expensive simulator calls.  The scientific prior supplied to the transport is unchanged.
    # A defensive uniform mixture keeps every candidate selectable and bounds the exact discrete
    # importance weight (1/K)/alpha_k by 1/proposal_defensive_epsilon.
    categorical_proposal_enabled: bool = False  # Broad prior training; no region targeting by default.
    importance_weights_enabled: bool = False  # Opt in to the proposal's discrete risk correction.
    # If proposal targeting is enabled with weights disabled, stored loss weights stay one.
    # Replaying a row always reuses the weight selected at acquisition.
    categorical_proposal_warmup_steps: int = 8
    categorical_proposal_refresh_every: int = 25
    categorical_proposal_candidate_particles: int = 1024
    categorical_proposal_reference_particles: int = 1024
    categorical_proposal_defensive_epsilon: float = 0.10
    categorical_proposal_knn: int = 16
    categorical_proposal_bandwidth_scale: float = 1.0
    categorical_proposal_min_bandwidth: float = 0.03

    # Three mutually-exclusive input sources. The residual probability gives fresh uniform.
    # A requested same-row posterior falls back to fresh uniform on its first visit only.
    prior_interpolation_probability: float = 0.25
    historical_output_prior_probability: float = 0.5
    interpolation_base_cloud: str = "uniform"  # {"uniform", "gaussian"}
    prior_interpolation_tau_min: float = 0.95
    prior_interpolation_tau_max: float = 1.05
    truth_anchor_probability: float = 1.0

    # Exact posterior / diagnostic grids
    posterior_grid_size: int = 420
    exact_reference_samples: int = 10_000
    sliced_wasserstein_projections: int = 128

    # Temporal experiment. T is configurable; no function below assumes T=4.
    sequence_length: int = 4
    evaluation_sequences: int = 24  # Independent held-out trajectories per scenario.
    evaluation_transitions: tuple[str, ...] = (
        "identity", "brownian", "driftBrownian", "constantVelocity", "rotation", "ou", "custom", "relocation",
    )
    evaluation_controls: bool = True  # Ensure identity and independent-parameter controls are included.
    transition_dt: float = 1.0
    brownian_std: float = 0.04  # Diffusion per sqrt(time).
    drift_vector: tuple[float, float] = (0.12, -0.06)
    constant_velocity: tuple[float, float] = (0.16, 0.08)
    rotation_angle: float = 0.30  # Radians per unit time, about transition_center.
    transition_center: tuple[float, float] = (0.0, 0.0)
    ou_rate: float = 0.35
    ou_diffusion: float = 0.12
    custom_amplitude: float = 0.12
    custom_noise_std: float = 0.04
    relocation_noise_std: float = 0.04  # Local diffusion before/after an independent reset.
    history_prior_refresh: float = 0.10  # Fraction of incoming particles refreshed from prior.
    # Forced reset occurs at final time by default; otherwise use a 1-based time in [2,T].
    abrupt_change_time: int | None = None
    metric_particles: int = 512
    bootstrap_replicates: int = 2000
    credible_mass: float = 0.90

    # Opt-in diagnostics make EXTRA simulator calls outside simulation_budget.
    # Leave disabled for a run with a strict total simulator-call budget.
    simulator_diagnostics_enabled: bool = False
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

for _name in ("simulation_budget", "batch_size", "log_every"):
    _value = getattr(CFG, _name)
    if isinstance(_value, bool) or not isinstance(_value, int) or _value < 1:
        raise ValueError(f"{_name} must be a positive integer.")
if isinstance(CFG.replay_epochs, bool) or not isinstance(CFG.replay_epochs, int) or CFG.replay_epochs < 0:
    raise ValueError("replay_epochs must be a non-negative integer.")

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
if CFG.prior_interpolation_probability + CFG.historical_output_prior_probability > 1.0:
    raise ValueError("Interpolation and posterior probabilities must sum to <= 1.")
if not 0.0 <= CFG.truth_anchor_probability <= 1.0:
    raise ValueError("truth_anchor_probability must lie in [0,1].")
if not 0.0 <= CFG.prior_interpolation_tau_min <= CFG.prior_interpolation_tau_max:
    raise ValueError("Interpolation tau bounds must satisfy 0 <= min <= max.")
if not 0.0 <= CFG.historical_output_prior_probability <= 1.0:
    raise ValueError("historical_output_prior_probability must lie in [0,1].")
if CFG.prior_low >= CFG.prior_high:
    raise ValueError("prior_low must be smaller than prior_high.")
if CFG.radial_std <= 0.0 or CFG.observation_scale <= 0.0:
    raise ValueError("radial_std and observation_scale must be positive.")
if CFG.max_training_particles < 2:
    raise ValueError("max_training_particles must be at least 2.")
if CFG.categorical_proposal_warmup_steps < 0:
    raise ValueError("categorical_proposal_warmup_steps must be non-negative.")
if CFG.categorical_proposal_refresh_every < 1:
    raise ValueError("categorical_proposal_refresh_every must be at least 1.")
if CFG.categorical_proposal_candidate_particles < 2:
    raise ValueError("categorical_proposal_candidate_particles must be at least 2.")
if CFG.categorical_proposal_reference_particles < 2:
    raise ValueError("categorical_proposal_reference_particles must be at least 2.")
if not 0.0 < CFG.categorical_proposal_defensive_epsilon <= 1.0:
    raise ValueError("categorical_proposal_defensive_epsilon must lie in (0,1].")
if CFG.categorical_proposal_knn < 1:
    raise ValueError("categorical_proposal_knn must be at least 1.")
if CFG.categorical_proposal_bandwidth_scale <= 0.0:
    raise ValueError("categorical_proposal_bandwidth_scale must be positive.")
if CFG.categorical_proposal_min_bandwidth <= 0.0:
    raise ValueError("categorical_proposal_min_bandwidth must be positive.")
for _name in ("sequence_length", "evaluation_sequences", "metric_particles"):
    _value = getattr(CFG, _name)
    if isinstance(_value, bool) or not isinstance(_value, int) or _value < 2:
        raise ValueError(f"{_name} must be an integer >= 2.")
if CFG.eval_particles < 2 or CFG.posterior_grid_size < 16 or CFG.exact_reference_samples < 2:
    raise ValueError("Use >=2 evaluation/reference particles and a grid size >=16.")
if CFG.bootstrap_replicates < 1 or not 0.0 < CFG.credible_mass < 1.0:
    raise ValueError("Invalid bootstrap_replicates or credible_mass.")
if not 0.0 <= CFG.history_prior_refresh <= 1.0:
    raise ValueError("history_prior_refresh must be in [0,1].")
_allowed_transitions = {"identity", "brownian", "driftBrownian", "constantVelocity", "rotation", "ou", "custom", "relocation"}
if not CFG.evaluation_transitions or not set(CFG.evaluation_transitions) <= _allowed_transitions:
    raise ValueError(f"evaluation_transitions must select from {_allowed_transitions}.")
if len(set(CFG.evaluation_transitions)) != len(CFG.evaluation_transitions):
    raise ValueError("evaluation_transitions must not contain duplicates.")
for _name in ("brownian_std", "ou_rate", "ou_diffusion", "custom_noise_std", "relocation_noise_std"):
    if not np.isfinite(getattr(CFG, _name)) or getattr(CFG, _name) < 0:
        raise ValueError(f"{_name} must be finite and non-negative.")
if not np.isfinite(CFG.transition_dt) or CFG.transition_dt <= 0:
    raise ValueError("transition_dt must be finite and positive.")
for _name in ("rotation_angle", "custom_amplitude"):
    if not np.isfinite(getattr(CFG, _name)):
        raise ValueError(f"{_name} must be finite.")
for _name in ("drift_vector", "constant_velocity", "transition_center"):
    if np.asarray(getattr(CFG, _name)).shape != (2,) or not np.all(np.isfinite(getattr(CFG, _name))):
        raise ValueError(f"{_name} must contain two finite coordinates.")
if CFG.abrupt_change_time is not None and (
    not isinstance(CFG.abrupt_change_time, int)
    or isinstance(CFG.abrupt_change_time, bool)
    or not 2 <= CFG.abrupt_change_time <= CFG.sequence_length
):
    raise ValueError("abrupt_change_time must be None or a 1-based time in [2,sequence_length].")

TRAINING_PARTICLE_COUNTS = (
    training_particle_count_choices(CFG.max_training_particles)
    if CFG.variable_training_particles
    else (int(CFG.max_training_particles),)
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
    "Stored-row input probabilities: "
    f"interpolation={CFG.prior_interpolation_probability:.3f}, "
    f"own-posterior={CFG.historical_output_prior_probability:.3f}, "
    f"fresh-uniform={1.0 - CFG.prior_interpolation_probability - CFG.historical_output_prior_probability:.3f}"
)
print("First visits replace unavailable posterior inputs with fresh priors; interpolation remains available.")
print("Proposal targeting:", CFG.categorical_proposal_enabled, "| Importance correction:", CFG.importance_weights_enabled)
print("Variable training particle count:", CFG.variable_training_particles)
print("Training particle-count choices:", TRAINING_PARTICLE_COUNTS)
print("Maximum training particles:", CFG.max_training_particles)
print("Reset baseline prior: exact uniform; history methods carry posterior particles.")
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


def categorical_proposal_from_posterior_particles_np(
    rng: np.random.Generator,
    posterior_reference: np.ndarray,
    batch_size: int,
    cfg: Config = CFG,
) -> tuple[np.ndarray, np.ndarray]:
    """Select simulator parameters from a categorical distribution over fresh prior candidates.

    Fresh candidates theta_k are iid from the exact prior.  Their categorical probabilities alpha_k
    are determined only by proximity to the current one-step posterior particle cloud at x_o.  The
    returned per-example weights are the exact discrete correction

        w_k = (1/K) / alpha_k,

    so, conditional on the fresh candidate cloud, the weighted proper-score risk equals the risk
    obtained by selecting candidates uniformly.  Averaging over fresh candidate clouds therefore
    recovers the original-prior simulator-training objective without evaluating any density.
    """
    posterior_reference = np.asarray(posterior_reference, dtype=np.float64)
    posterior_reference = posterior_reference[np.all(np.isfinite(posterior_reference), axis=1)]
    if len(posterior_reference) < 2:
        theta = sample_exact_prior_np(rng, batch_size)
        return theta, np.ones(int(batch_size), dtype=np.float32)

    k_candidates = int(cfg.categorical_proposal_candidate_particles)
    candidates = sample_exact_prior_np(rng, k_candidates).astype(np.float64)

    # A particle-native KDE proxy: average Gaussian affinity to the k nearest posterior particles.
    tree = cKDTree(posterior_reference)
    k_nn = min(int(cfg.categorical_proposal_knn), len(posterior_reference))
    distances, _ = tree.query(candidates, k=k_nn)
    if k_nn == 1:
        distances = distances[:, None]

    # Scott-type scale for d=2, with a small floor for very concentrated posterior clouds.
    posterior_scale = float(np.sqrt(np.mean(np.var(posterior_reference, axis=0, ddof=1))))
    scott = float(len(posterior_reference) ** (-1.0 / 6.0))
    bandwidth = max(
        float(cfg.categorical_proposal_min_bandwidth),
        float(cfg.categorical_proposal_bandwidth_scale) * posterior_scale * scott,
    )

    affinity = np.mean(np.exp(-0.5 * (distances / bandwidth) ** 2), axis=1)
    affinity = np.where(np.isfinite(affinity), affinity, 0.0)
    total = float(np.sum(affinity))
    if total <= 0.0:
        focused = np.full(k_candidates, 1.0 / k_candidates, dtype=np.float64)
    else:
        focused = affinity / total

    eps = float(cfg.categorical_proposal_defensive_epsilon)
    alpha = (1.0 - eps) * focused + eps / k_candidates
    alpha /= np.sum(alpha)

    ids = rng.choice(k_candidates, size=int(batch_size), replace=True, p=alpha)
    theta = candidates[ids].astype(np.float32)
    importance = ((1.0 / k_candidates) / alpha[ids]).astype(np.float32)
    return theta, importance


if CFG.simulator_diagnostics_enabled:
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


#%% 2) Paired simulation/posterior buffer and three input-cloud sources

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


class SimulationBuffer:
    """Fixed simulator pairs with a mutable, model-specific posterior for every row.

    x, theta*, and the acquisition importance weight never change after insertion. Detached
    posterior clouds are replaced with the actual output of each training call (before the
    optimizer update), whether its input was fresh or replayed. Models never share outputs.

    counts records real particle counts. When a later call requests fewer particles, subsample
    without replacement; when it requests more, bootstrap the stored empirical cloud. This adds
    no new information, makes the configured replay probability independent of particle count,
    and avoids dropping a requested replay or using a different observation's posterior.
    """

    def __init__(self, capacity: int, max_particles: int, model_names: tuple[str, ...]):
        if capacity < 1 or max_particles < 2:
            raise ValueError("Buffer capacity must be positive and max_particles must be >=2.")
        if not model_names or len(set(model_names)) != len(model_names):
            raise ValueError("Provide distinct model names for posterior storage.")
        self.capacity = int(capacity)
        self.max_particles = int(max_particles)
        self.theta = np.empty((self.capacity, 2), dtype=np.float32)
        self.x = np.empty((self.capacity, 2), dtype=np.float32)
        self.sample_weights = np.empty(self.capacity, dtype=np.float32)
        # Zero-filled unused tails make the saved arrays deterministic; counts defines valid data.
        self.posteriors = {name: np.zeros((self.capacity, self.max_particles, 2), dtype=np.float32)
                           for name in model_names}
        self.posterior_counts = {name: np.zeros(self.capacity, dtype=np.int32) for name in model_names}
        self.posterior_updates = {name: np.zeros(self.capacity, dtype=np.int32) for name in model_names}
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def _row_indices(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Buffer row indices must be a one-dimensional integer array.")
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("Buffer row index is outside the stored dataset.")
        return indices

    def add_batch(self, theta: np.ndarray, x: np.ndarray,
                  sample_weights: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32)
        sample_weights = np.asarray(sample_weights, dtype=np.float32)
        if theta.ndim != 2 or theta.shape[1] != 2 or x.shape != theta.shape:
            raise ValueError("theta and x must both have shape [B,2].")
        if sample_weights.shape != (len(theta),):
            raise ValueError("sample_weights must have shape [B].")
        if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(x))
                and np.all(np.isfinite(sample_weights)) and np.all(sample_weights > 0)):
            raise ValueError("Simulator pairs must be finite and weights positive.")
        end = self.size + len(theta)
        if end > self.capacity:
            raise ValueError("Adding this batch would exceed the simulation budget.")
        indices = np.arange(self.size, end)
        self.theta[indices], self.x[indices] = theta, x
        self.sample_weights[indices] = sample_weights
        self.size = end
        return indices

    def get_batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = self._row_indices(indices)
        return self.theta[indices], self.x[indices], self.sample_weights[indices]

    def update_posteriors(self, model_name: str, indices: np.ndarray, clouds: np.ndarray) -> None:
        indices = self._row_indices(indices)
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Update each buffer row at most once per minibatch.")
        clouds = np.asarray(clouds, dtype=np.float32)
        if clouds.ndim != 3 or clouds.shape[0] != len(indices) or clouds.shape[2] != 2:
            raise ValueError("Posterior clouds must have shape [B,M,2].")
        count = clouds.shape[1]
        if not 2 <= count <= self.max_particles or not np.all(np.isfinite(clouds)):
            raise ValueError("Posterior clouds must be finite with 2 <= M <= max_particles.")
        self.posteriors[model_name][indices] = 0.0
        self.posteriors[model_name][indices, :count] = clouds
        self.posterior_counts[model_name][indices] = count
        self.posterior_updates[model_name][indices] += 1

    def posterior_batch(self, model_name: str, indices: np.ndarray, n_particles: int,
                        rng: np.random.Generator) -> np.ndarray:
        indices = self._row_indices(indices)
        if not 2 <= n_particles <= self.max_particles:
            raise ValueError("Requested cloud size must lie in [2,max_particles].")
        result = np.empty((len(indices), n_particles, 2), dtype=np.float32)
        for row, index in enumerate(indices):
            count = int(self.posterior_counts[model_name][index])
            if count == 0:
                raise ValueError("This row has no posterior for the requested model yet.")
            cloud = self.posteriors[model_name][index, :count]
            if count == n_particles:
                result[row] = cloud
            else:
                ids = rng.choice(count, n_particles, replace=n_particles > count)
                result[row] = cloud[ids]
        return result

    def save(self, path: Path) -> None:
        payload = dict(theta=self.theta[:self.size], x=self.x[:self.size],
                       sample_weights=self.sample_weights[:self.size])
        for name in self.posteriors:
            payload[f"posterior_{name}"] = self.posteriors[name][:self.size]
            payload[f"posterior_counts_{name}"] = self.posterior_counts[name][:self.size]
            payload[f"posterior_updates_{name}"] = self.posterior_updates[name][:self.size]
        np.savez_compressed(path, **payload)


def make_training_prior_batch_np(
    rng: np.random.Generator,
    mode_rng: np.random.Generator,
    buffer: SimulationBuffer,
    indices: np.ndarray,
    model_name: str,
    n_particles: int,
    cfg: Config = CFG,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Choose fresh prior, interpolation, or this exact row's last posterior per example.

    Row IDs bind the replayed cloud to its x/theta*/weight; no observation-neighbour search.
    Only interpolation may use the training target as its anchor, matching the original scheme.
    The three probabilities remain mutually exclusive even when a posterior is not yet available.
    """
    indices = buffer._row_indices(indices)
    if not 2 <= n_particles <= cfg.max_training_particles:
        raise ValueError("Training particle count must lie in [2,max_training_particles].")
    counts = buffer.posterior_counts[model_name][indices]
    available = counts > 0
    u = mode_rng.random(len(indices))
    use_interpolation = u < cfg.prior_interpolation_probability
    use_buffer = ((u >= cfg.prior_interpolation_probability)
                  & (u < cfg.prior_interpolation_probability + cfg.historical_output_prior_probability)
                  & available)
    use_fresh = ~(use_interpolation | use_buffer)
    prior = sample_exact_prior_np(rng, len(indices) * n_particles).reshape(len(indices), n_particles, 2)
    tau = np.zeros(len(indices), dtype=np.float32)
    for row in np.flatnonzero(use_interpolation):
        prior[row], tau[row] = sample_interpolated_training_prior_np(
            rng, buffer.theta[indices[row]], n_particles, cfg)
    selected = np.flatnonzero(use_buffer)
    if len(selected):
        prior[selected] = buffer.posterior_batch(model_name, indices[selected], n_particles, rng)
    return prior, {
        "interpolation_used": use_interpolation.astype(np.float32),
        "interpolation_tau": tau,
        "buffer_used": use_buffer.astype(np.float32),
        "exact_prior_used": use_fresh.astype(np.float32),
        "posterior_available": available.astype(np.float32),
        "posterior_bootstrapped": (use_buffer & (counts < n_particles)).astype(np.float32),
    }


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
        if x.shape != (2,):
            raise ValueError("Each model call requires exactly one observation with shape (2,).")
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


def batch_metrics(
    posterior: Array,
    target_theta: Array,
    sample_weights: Array,
) -> dict[str, Array]:
    """posterior [B,M,2], target_theta [B,2], sample_weights [B]."""
    score, attraction, repulsion = jax.vmap(energy_score_terms)(posterior, target_theta)
    sample_weights = jnp.asarray(sample_weights, dtype=score.dtype)
    weighted_score = sample_weights * score
    means = jnp.mean(posterior, axis=1)
    mean_error = _stable_l2_norm(means - target_theta, axis=-1)
    centered = posterior - means[:, None, :]
    covariance_trace = jnp.mean(jnp.sum(centered**2, axis=-1), axis=1)
    outside = jnp.any(
        (posterior < CFG.prior_low) | (posterior > CFG.prior_high),
        axis=-1,
    )
    return {
        "loss": jnp.mean(weighted_score),
        "energy_score": jnp.mean(weighted_score),
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
    sample_weights: Array,
    dropout_key: Array,
):
    row_keys = jax.random.split(dropout_key, prior_theta.shape[0])
    posterior = jax.vmap(
        lambda p, x, k: model(p, x, key=k, inference=False)
    )(prior_theta, x_batch, row_keys)
    metrics = batch_metrics(posterior, target_theta, sample_weights)
    return metrics["loss"], (metrics, posterior)


_loss_and_grad = eqx.filter_value_and_grad(transport_objective, has_aux=True)


def make_train_step(optimizer: optax.GradientTransformation):
    @eqx.filter_jit
    def step(model, opt_state, prior_theta, x_batch, target_theta, sample_weights, dropout_key):
        (loss, (metrics, posterior)), grads = _loss_and_grad(
            model,
            prior_theta,
            x_batch,
            target_theta,
            sample_weights,
            dropout_key,
        )
        params = eqx.filter(model, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return model, opt_state, loss, metrics, posterior, grad_norm

    return step


# Immutable Equinox initialization is shared; updates create separate model trees.
initial_model = ConditionalParticleTransport(CFG, key=jax.random.key(CFG.seed))
TRAIN_CONFIGS = {
    "single_observation": replace(CFG, prior_interpolation_probability=0.0,
                                  historical_output_prior_probability=0.0),
    "no_replay": replace(CFG, historical_output_prior_probability=0.0),
    "buffered": CFG,
}
models = {name: initial_model for name in TRAIN_CONFIGS}
optimizer = optax.chain(
    optax.clip_by_global_norm(CFG.grad_clip_norm),
    optax.adamw(CFG.learning_rate, weight_decay=CFG.weight_decay),
)
opt_states = {name: optimizer.init(eqx.filter(model, eqx.is_array))
              for name, model in models.items()}
train_step = make_train_step(optimizer)
print("Three matched transports initialized; every network sees one x with shape (2,).")


#%% 5) Diagnostic likelihood and reusable evaluation/checkpoint helpers
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


@eqx.filter_jit
def _evaluate_bt_jax(model, prior_particles, x):
    return model(prior_particles, x, key=None, inference=True)


def evaluate_bt(
    model: ConditionalParticleTransport,
    prior_particles: np.ndarray,
    x: np.ndarray = X_OBS,
) -> np.ndarray:
    return np.asarray(
        jax.device_get(
            _evaluate_bt_jax(model, jnp.asarray(prior_particles), jnp.asarray(x))
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


#%% 6) Temporal simulator and numerical filtering references (diagnostics only)

@dataclass(frozen=True)
class Scenario:
    name: str
    reset_probability: float = 0.0
    forced_reset_time: int | None = None
    # Relocation is a scheduled independent prior reset, unknown to learned history methods.


SCENARIOS = tuple(
    Scenario(name, forced_reset_time=(
        CFG.sequence_length if CFG.abrupt_change_time is None else CFG.abrupt_change_time
    ) if name == "relocation" else None)
    for name in CFG.evaluation_transitions
)
if CFG.evaluation_controls:
    if "identity" not in CFG.evaluation_transitions:
        SCENARIOS += (Scenario("identity"),)
    SCENARIOS += (Scenario("independent", reset_probability=1.0))


def reflect_to_prior(theta: np.ndarray) -> np.ndarray:
    """Fold at both square boundaries; identity in the interior."""
    width = CFG.prior_high - CFG.prior_low
    phase = np.mod(np.asarray(theta) - CFG.prior_low, 2.0 * width)
    return (CFG.prior_low + np.where(phase <= width, phase, 2.0 * width - phase)).astype(np.float32)


def custom_transition_mean_np(theta: np.ndarray, time_index: int) -> np.ndarray:
    """EDIT THIS function for a custom deterministic transition, preserving shape [...,2].

    time_index is the 0-based DESTINATION time (1 for theta_1 -> theta_2). The default is a
    nonlinear sinusoidal displacement field. Independent custom_noise_std noise is added later.
    The same function is used by trajectory simulation, neural prediction, and grid diagnostics.
    """
    phase = 0.25 * time_index * CFG.transition_dt
    displacement = np.stack([np.sin(np.pi * theta[..., 1] + phase),
                             np.cos(np.pi * theta[..., 0] - phase)], axis=-1)
    return theta + CFG.custom_amplitude * CFG.transition_dt * displacement


def transition_mean_np(theta: np.ndarray, time_index: int, scenario: Scenario) -> np.ndarray:
    """Known deterministic part of the evaluation transition, followed by square reflection.

    constantVelocity uses a configured, known fixed velocity (no hidden velocity state).
    rotation uses a known angle/center; reflection handles points leaving the square.
    ou uses the exact unconstrained OU mean/variance, with reflected boundary handling.
    All transitions are discrete evaluation mechanisms, absent from the training objective.
    """
    theta = np.asarray(theta, dtype=np.float64)
    dt = CFG.transition_dt
    center = np.asarray(CFG.transition_center)
    if scenario.name == "driftBrownian":
        mean = theta + dt * np.asarray(CFG.drift_vector)
    elif scenario.name == "constantVelocity":
        mean = theta + dt * np.asarray(CFG.constant_velocity)
    elif scenario.name == "rotation":
        angle = CFG.rotation_angle * dt
        matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        mean = center + (theta - center) @ matrix.T
    elif scenario.name == "ou":
        mean = center + np.exp(-CFG.ou_rate * dt) * (theta - center)
    elif scenario.name == "custom":
        mean = custom_transition_mean_np(theta, time_index)
    else:
        mean = theta
    return reflect_to_prior(mean)


def transition_noise_std(scenario: Scenario) -> float:
    dt = CFG.transition_dt
    if scenario.name in ("brownian", "driftBrownian"):
        return CFG.brownian_std * math.sqrt(dt)
    if scenario.name == "custom":
        return CFG.custom_noise_std * math.sqrt(dt)
    if scenario.name == "relocation":
        return CFG.relocation_noise_std * math.sqrt(dt)
    if scenario.name == "ou":
        variance_factor = (-np.expm1(-2.0 * CFG.ou_rate * dt) / (2.0 * CFG.ou_rate)
                           if CFG.ou_rate > 0.0 else dt)
        return CFG.ou_diffusion * math.sqrt(variance_factor)
    return 0.0


def predict_transition_np(rng: np.random.Generator, theta: np.ndarray,
                          time_index: int, scenario: Scenario) -> np.ndarray:
    if scenario.name == "identity":
        # Exact identity, without numerical folding or a noise draw; x_t is still freshly simulated.
        return np.asarray(theta, dtype=np.float32).copy()
    mean = transition_mean_np(theta, time_index, scenario)
    return reflect_to_prior(mean + rng.normal(0.0, transition_noise_std(scenario), mean.shape))


def simulate_sequence_np(rng: np.random.Generator, scenario: Scenario):
    theta = np.empty((CFG.sequence_length, 2), dtype=np.float32)
    theta[0] = sample_exact_prior_np(rng, 1)[0]
    for t in range(1, CFG.sequence_length):
        reset = t + 1 == scenario.forced_reset_time or rng.random() < scenario.reset_probability
        theta[t] = (sample_exact_prior_np(rng, 1)[0] if reset else
                    predict_transition_np(rng, theta[t - 1], t, scenario))
    # Exactly one independent simulator draw for each theta_t; no replicate observations.
    return theta, simulate_two_moons_batch_np(rng, theta)


def predict_grid_mass(mass: np.ndarray, axis: np.ndarray, time_index: int,
                      scenario: Scenario) -> np.ndarray:
    """Conservative bilinear pushforward, then reflected Gaussian convolution.

    Grid discretization introduces numerical diffusion for deterministic maps; compare finer grids
    before interpreting small gains for rotation/constantVelocity or long sequences.
    """
    spacing = axis[1] - axis[0]
    if scenario.name in ("identity", "stable", "brownian", "relocation", "independent"):
        # Preserve identity exactly; avoid numerical advection for zero deterministic motion.
        std = transition_noise_std(scenario)
        return (normalize_grid_mass(gaussian_filter(mass, std / spacing, mode="reflect", truncate=6.0))
                if std > 0.0 else mass.copy())
    g1, g2 = np.meshgrid(axis, axis, indexing="xy")
    mapped = transition_mean_np(np.stack([g1, g2], axis=-1), time_index, scenario)
    location = np.clip((mapped.reshape(-1, 2) - axis[0]) / spacing, 0, len(axis) - 1)
    lower = np.floor(location).astype(int)
    upper = np.minimum(lower + 1, len(axis) - 1)
    fraction = location - lower
    predicted = np.zeros(mass.size, dtype=np.float64)
    for x_high in (0, 1):
        for y_high in (0, 1):
            ix = upper[:, 0] if x_high else lower[:, 0]
            iy = upper[:, 1] if y_high else lower[:, 1]
            weight = ((fraction[:, 0] if x_high else 1 - fraction[:, 0]) *
                      (fraction[:, 1] if y_high else 1 - fraction[:, 1]))
            predicted += np.bincount(iy * len(axis) + ix, weights=mass.ravel() * weight,
                                     minlength=mass.size)
    predicted = predicted.reshape(mass.shape)
    std = transition_noise_std(scenario)
    if std > 0.0:
        predicted = gaussian_filter(predicted, std / spacing, mode="reflect", truncate=6.0)
    return normalize_grid_mass(predicted)


def normalize_grid_mass(mass: np.ndarray) -> np.ndarray:
    total = float(np.sum(mass))
    if not np.all(np.isfinite(mass)) or np.any(mass < 0.0) or total <= 0.0:
        raise FloatingPointError("Posterior grid lost mass; increase posterior_grid_size.")
    return mass / total


def filtering_grid_references(x_sequence: np.ndarray, scenario: Scenario):
    """Numerical current-only and history posteriors under the true evaluation dynamics.

    Propagate an unconditioned marginal as well as the filter. OU, drift, reflected rotation, and
    custom maps can change the marginal prior: p(theta_t|x_t) must then use that time-t marginal.
    Also return the original uniform-prior posterior to expose this potential distribution shift.
    Relocation's diagnostic oracle knows the reset schedule; learned methods do not.
    """
    if x_sequence.shape != (CFG.sequence_length, 2):
        raise ValueError("Expected sequence_length individual 2-D observations.")
    spacing = (CFG.prior_high - CFG.prior_low) / CFG.posterior_grid_size
    axis = CFG.prior_low + (np.arange(CFG.posterior_grid_size) + 0.5) * spacing
    g1, g2 = np.meshgrid(axis, axis, indexing="xy")
    theta_grid = np.stack([g1, g2], axis=-1)
    uniform = np.full(g1.shape, 1.0 / g1.size)
    posterior, marginal = uniform.copy(), uniform.copy()
    singles, filtered, uniform_posteriors = [], [], []
    for t, x in enumerate(x_sequence):
        if t:
            rho = 1.0 if t + 1 == scenario.forced_reset_time else scenario.reset_probability
            if rho == 1.0:
                posterior, marginal = uniform.copy(), uniform.copy()
            else:
                posterior = (1.0 - rho) * predict_grid_mass(posterior, axis, t, scenario) + rho * uniform
                marginal = (1.0 - rho) * predict_grid_mass(marginal, axis, t, scenario) + rho * uniform
        likelihood = two_moons_likelihood_density_np(x, theta_grid)
        peak = float(np.max(likelihood))
        if peak <= 0.0 or not np.isfinite(peak):
            raise FloatingPointError("Observation unresolved on the diagnostic grid.")
        likelihood = likelihood / peak
        singles.append(normalize_grid_mass(marginal * likelihood))
        uniform_posteriors.append(normalize_grid_mass(likelihood))
        posterior = normalize_grid_mass(posterior * likelihood)
        filtered.append(posterior.copy())
    return axis, np.asarray(singles), np.asarray(filtered), np.asarray(uniform_posteriors)


def sample_grid_mass(rng: np.random.Generator, axis: np.ndarray,
                     mass: np.ndarray, n: int) -> np.ndarray:
    ids = rng.choice(mass.size, n, p=mass.ravel())
    iy, ix = np.unravel_index(ids, mass.shape)
    spacing = axis[1] - axis[0]
    # Midpoint-cell jitter stays inside the prior, including the two boundary cells.
    return (np.column_stack([axis[ix], axis[iy]]) +
            rng.uniform(-spacing / 2, spacing / 2, (n, 2))).astype(np.float32)


def history_input_cloud(rng: np.random.Generator, posterior: np.ndarray,
                        fresh_prior: np.ndarray, scenario: Scenario, time_index: int) -> np.ndarray:
    """Predict particles with assumed dynamics, then defensively refresh from the original prior.

    No current/future x, latent truth, or diagnostic density is inspected here. Folding particles
    back to prior support is part of prediction only; output support violations are still scored.
    """
    rho = 1.0 - (1.0 - scenario.reset_probability) * (1.0 - CFG.history_prior_refresh)
    if rho == 1.0:
        return fresh_prior.copy()
    prediction = predict_transition_np(rng, posterior, time_index, scenario)
    refresh = rng.random(len(prediction)) < rho
    prediction[refresh] = fresh_prior[refresh]
    return prediction


def run_history(model, observations: np.ndarray, priors: np.ndarray,
                scenario: Scenario, *, defensive: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    path = []
    for t, x in enumerate(observations):
        incoming = priors[t] if t == 0 else path[-1]
        if t and defensive:
            incoming = history_input_cloud(rng, incoming, priors[t], scenario, t)
        path.append(evaluate_bt(model, incoming, x))
    return np.asarray(path)


#%% 7) TRAIN — matched simulator budget, fresh acquisition followed by paired dataset replay
# Dataset acquisition is uniform by default because evaluation spans many held-out observations.
# Optional original categorical acquisition focuses on the fixed design datum X_OBS, never on
# evaluation trajectories. Correction is separately optional; stored loss weights and rows are
# shared by ALL models. After acquisition, all observations/targets come from these stored rows.

simulation_buffer = SimulationBuffer(CFG.simulation_budget, CFG.max_training_particles, tuple(models))
train_rng = np.random.default_rng(CFG.seed + 10_001)
shuffle_rng = np.random.default_rng(CFG.seed + 15_013)
particle_rng = np.random.default_rng(CFG.seed + 25_019)
proposal_rng = np.random.default_rng(CFG.seed + 27_011)
# Separate RNGs with common seeds keep cloud/mode choices independent of simulator acquisition.
cloud_rngs = {name: np.random.default_rng(CFG.seed + 31_001) for name in models}
mode_rngs = {name: np.random.default_rng(CFG.seed + 32_001) for name in models}
dropout_key = jax.random.key(CFG.seed + 30_007)
acquisition_steps = math.ceil(CFG.simulation_budget / CFG.batch_size)
TOTAL_TRAINING_STEPS = acquisition_steps * (1 + CFG.replay_epochs)
proposal_posterior_reference = None
training_rows = []
print(f"Shared dataset: {CFG.simulation_budget:,} simulator calls; "
      f"{TOTAL_TRAINING_STEPS:,} optimizer updates PER model.")

for step in range(1, TOTAL_TRAINING_STEPS + 1):
    training_particles = int(particle_rng.choice(TRAINING_PARTICLE_COUNTS))
    replay_epoch = 0
    if len(simulation_buffer) < CFG.simulation_budget:
        batch_n = min(CFG.batch_size, CFG.simulation_budget - len(simulation_buffer))
        if CFG.categorical_proposal_enabled and step > CFG.categorical_proposal_warmup_steps:
            if (proposal_posterior_reference is None or
                (step - CFG.categorical_proposal_warmup_steps - 1) % CFG.categorical_proposal_refresh_every == 0):
                proposal_posterior_reference = evaluate_bt(
                    models["buffered"], sample_exact_prior_np(
                        proposal_rng, CFG.categorical_proposal_reference_particles), X_OBS)
            theta_target, sample_weights = categorical_proposal_from_posterior_particles_np(
                proposal_rng, proposal_posterior_reference, batch_n, CFG)
        else:
            theta_target = sample_exact_prior_np(train_rng, batch_n)
            sample_weights = np.ones(batch_n, dtype=np.float32)
        if not CFG.importance_weights_enabled:
            sample_weights = np.ones(batch_n, dtype=np.float32)
        x_batch = simulate_two_moons_batch_np(train_rng, theta_target)
        batch_indices = simulation_buffer.add_batch(theta_target, x_batch, sample_weights)
    else:
        replay_epoch, batch_in_epoch = divmod(step - acquisition_steps - 1, acquisition_steps)
        replay_epoch += 1
        if batch_in_epoch == 0:
            epoch_indices = shuffle_rng.permutation(len(simulation_buffer))
        start = batch_in_epoch * CFG.batch_size
        batch_indices = epoch_indices[start:start + CFG.batch_size]
        theta_target, x_batch, sample_weights = simulation_buffer.get_batch(batch_indices)

    dropout_key, step_key = jax.random.split(dropout_key)
    for name, cfg in TRAIN_CONFIGS.items():
        incoming, info = make_training_prior_batch_np(
            cloud_rngs[name], mode_rngs[name], simulation_buffer, batch_indices,
            name, training_particles, cfg)
        models[name], opt_states[name], loss, metrics, posterior, grad_norm = train_step(
            models[name], opt_states[name], jnp.asarray(incoming), jnp.asarray(x_batch),
            jnp.asarray(theta_target), jnp.asarray(sample_weights), step_key)
        # Store the detached output of THIS call under the SAME row IDs, for the next visit.
        # Targets/observations/weights stay fixed; gradients never flow through earlier calls.
        simulation_buffer.update_posteriors(name, batch_indices, np.asarray(jax.device_get(posterior)))
        host = jax.device_get(metrics)
        training_rows.append({
            "model": name, "step": step, "simulations_seen": len(simulation_buffer),
            "replay_epoch": replay_epoch, "batch_examples": len(theta_target),
            "training_particles": training_particles, "energy_score": float(host["energy_score"]),
            "grad_norm": float(jax.device_get(grad_norm)),
            "outside_prior_fraction": float(host["outside_prior_fraction"]),
            "interpolation_fraction": float(np.mean(info["interpolation_used"])),
            "mean_loss_weight": float(np.mean(sample_weights)),
            "buffer_fraction": float(np.mean(info["buffer_used"])),
            "fresh_prior_fraction": float(np.mean(info["exact_prior_used"])),
            "posterior_available_fraction": float(np.mean(info["posterior_available"])),
            "posterior_bootstrapped_fraction": float(np.mean(info["posterior_bootstrapped"])),
        })
    # Save only AFTER every model has populated its latest cloud, including the last fresh batch.
    if step == acquisition_steps or step == TOTAL_TRAINING_STEPS:
        simulation_buffer.save(OUT / "simulation_buffer.npz")
    if step == 1 or step % CFG.log_every == 0 or step in (acquisition_steps, TOTAL_TRAINING_STEPS):
        print(f"step {step:6d}/{TOTAL_TRAINING_STEPS} | sims {len(simulation_buffer):,} | "
              f"replay {replay_epoch} | M {training_particles} | " + " | ".join(
                  f"{r['model']} ES {r['energy_score']:.5f} (posterior inputs {r['buffer_fraction']:.2f})" for r in training_rows[-len(models):]))

for name, model in models.items():
    save_model(OUT / f"{name}.eqx", model, TRAIN_CONFIGS[name])
with (OUT / "training_history.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(training_rows[0]))
    writer.writeheader()
    writer.writerows(training_rows)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for name in models:
    rows = [r for r in training_rows if r["model"] == name]
    for ax, metric in zip(axes, ("energy_score", "grad_norm", "outside_prior_fraction", "buffer_fraction")):
        ax.plot([r["step"] for r in rows], rolling_mean(np.array([r[metric] for r in rows])), label=name)
        ax.set(xlabel="Optimizer update", title=metric.replace("_", " "))
        ax.axvline(acquisition_steps, color="black", linestyle=":", alpha=0.5)
axes[0].legend(fontsize=8)
axes[3].axhline(CFG.historical_output_prior_probability, color="black", linestyle="--", alpha=0.5)
axes[3].set_ylim(-0.02, 1.02)
axes[1].set_yscale("symlog", linthresh=1e-5)
fig.suptitle("Matched training — dotted line: start of simulation-dataset replay")
fig.tight_layout()
fig.savefig(OUT / "10_training_diagnostics.png", dpi=180, bbox_inches="tight")
plt.show()


#%% 8) Held-out sequences: one x per call, history only through the incoming particles
# If loading checkpoints, run cells 0–6 and then:
# models = {name: load_model(OUT / f"{name}.eqx", cfg) for name, cfg in TRAIN_CONFIGS.items()}

METHOD_LABELS = {
    "sbi_xT": "SBI: current x only",
    "no_replay_xT": "Interpolation, no cloud replay: current x",
    "marginal_xT": "Buffer: predicted marginal + current x",
    "buffer_xT": "Paired buffer: current x",
    "proposal_xT": "Original proposal start",
    "refine_xT": "Buffer: repeated current x",
    "history_raw": "Buffer: raw history",
    "history_defensive": "Buffer: predicted + refreshed history",
    "no_replay_history": "No cloud replay: predicted + refreshed history",
    "reference_current": "Grid: current x only",
    "reference_history": "Grid: true temporal law",
}
METHOD_CALLS = {name: (CFG.sequence_length if name in (
    "refine_xT", "history_raw", "history_defensive", "no_replay_history") else
    2 if name == "proposal_xT" else 0 if name.startswith("reference_") else 1)
    for name in METHOD_LABELS}


def evaluate_sequence(observations: np.ndarray, scenario: Scenario, seed: int):
    rng = np.random.default_rng(seed)
    priors = sample_exact_prior_np(rng, CFG.sequence_length * CFG.eval_particles).reshape(
        CFG.sequence_length, CFG.eval_particles, 2)
    x_final = observations[-1]
    final = {
        "sbi_xT": evaluate_bt(models["single_observation"], priors[-1], x_final),
        "no_replay_xT": evaluate_bt(models["no_replay"], priors[-1], x_final),
        "buffer_xT": evaluate_bt(models["buffered"], priors[-1], x_final),
    }
    # Dynamics-only control: propagate particles without assimilating any earlier observations.
    # This separates time-marginal prior information from information supplied by observed history.
    marginal_prior = priors[0].copy()
    marginal_rng = np.random.default_rng(seed + 3)
    for t in range(1, CFG.sequence_length):
        marginal_prior = (priors[t].copy() if scenario.reset_probability == 1.0 else
                          predict_transition_np(marginal_rng, marginal_prior, t, scenario))
    final["marginal_xT"] = evaluate_bt(models["buffered"], marginal_prior, x_final)
    proposal, _ = categorical_proposal_from_posterior_particles_np(
        rng, final["buffer_xT"], CFG.eval_particles, CFG)
    final["proposal_xT"] = evaluate_bt(models["buffered"], proposal, x_final)
    # First refinement call is the same buffer_xT evaluation, reused without recomputation.
    refined = [final["buffer_xT"]]
    for _ in range(1, CFG.sequence_length):
        refined.append(evaluate_bt(models["buffered"], refined[-1], x_final))
    final["refine_xT"] = refined[-1]
    paths = {"refine_xT": np.asarray(refined)}
    for name, model_name, defensive in (
        ("history_raw", "buffered", False),
        ("history_defensive", "buffered", True),
        ("no_replay_history", "no_replay", True),
    ):
        paths[name] = run_history(models[model_name], observations, priors, scenario,
                                 defensive=defensive, seed=seed + 1)
        final[name] = paths[name][-1]
    return final, paths


def score_cloud(samples: np.ndarray, truth: np.ndarray, current_reference: np.ndarray,
                history_reference: np.ndarray, uniform_reference: np.ndarray) -> dict[str, float]:
    if not np.all(np.isfinite(samples)):
        raise FloatingPointError("Nonfinite posterior particles encountered.")
    # Deterministic subsampling shared across methods; cap pairwise metric costs.
    rng = np.random.default_rng(917)
    cloud = np.asarray(samples, dtype=np.float64)
    subset = cloud[rng.choice(len(cloud), min(CFG.metric_particles, len(cloud)), replace=False)]
    energy = np.linalg.norm(subset - truth, axis=1).mean() - pdist(subset).sum() / len(subset)**2
    tail = (1.0 - CFG.credible_mass) / 2.0
    lo, hi = np.quantile(cloud, [tail, 1.0 - tail], axis=0)
    return {
        "energy_score": float(energy),
        "mean_squared_error": float(np.sum((cloud.mean(axis=0) - truth)**2)),
        # This is mean marginal coverage, not a joint 2-D credible-region claim.
        "marginal_coverage": float(np.mean((truth >= lo) & (truth <= hi))),
        "marginal_interval_width": float(np.mean(hi - lo)),
        "outside_prior_fraction": float(np.mean(np.any(
            (cloud < CFG.prior_low) | (cloud > CFG.prior_high), axis=1))),
        "sw_uniform": sliced_wasserstein(uniform_reference, cloud, CFG.sliced_wasserstein_projections),
        "sw_current": sliced_wasserstein(current_reference, cloud, CFG.sliced_wasserstein_projections),
        "sw_history": sliced_wasserstein(history_reference, cloud, CFG.sliced_wasserstein_projections),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


metric_rows, example_sequences = [], {}
all_theta, all_x, all_scenario_names = [], [], []
all_samples = {name: [] for name in METHOD_LABELS}
for scenario_id, scenario in enumerate(SCENARIOS):
    for sequence_id in range(CFG.evaluation_sequences):
        sequence_seed = CFG.seed + 100_000 + scenario_id * 10_000 + sequence_id * 10
        theta, observations = simulate_sequence_np(np.random.default_rng(sequence_seed), scenario)
        final, paths = evaluate_sequence(observations, scenario, sequence_seed + 1)
        axis, single_mass, history_mass, uniform_mass = filtering_grid_references(observations, scenario)
        reference_rng = np.random.default_rng(sequence_seed + 2)
        current_reference = sample_grid_mass(reference_rng, axis, single_mass[-1], CFG.exact_reference_samples)
        history_reference = sample_grid_mass(reference_rng, axis, history_mass[-1], CFG.exact_reference_samples)
        uniform_reference = sample_grid_mass(reference_rng, axis, uniform_mass[-1], CFG.exact_reference_samples)
        # Equal-size oracle particle clouds make finite-ensemble truth scores comparable.
        final["reference_current"] = sample_grid_mass(reference_rng, axis, single_mass[-1], CFG.eval_particles)
        final["reference_history"] = sample_grid_mass(reference_rng, axis, history_mass[-1], CFG.eval_particles)
        for name, samples in final.items():
            metric_rows.append({
                "scenario": scenario.name, "sequence": sequence_id, "method": name,
                "sequence_length": CFG.sequence_length, "model_calls": METHOD_CALLS[name],
                **score_cloud(samples, theta[-1], current_reference, history_reference, uniform_reference),
            })
            all_samples[name].append(samples)
        all_theta.append(theta)
        all_x.append(observations)
        all_scenario_names.append(scenario.name)
        if sequence_id == 0:
            example_sequences[scenario.name] = {
                "theta": theta, "x": observations, "final": final, "paths": paths,
                "axis": axis, "single_mass": single_mass, "history_mass": history_mass, "uniform_mass": uniform_mass,
            }
        if sequence_id == 0 or (sequence_id + 1) % 8 == 0 or sequence_id + 1 == CFG.evaluation_sequences:
            print(f"{scenario.name}: {sequence_id + 1}/{CFG.evaluation_sequences} held-out sequences")

write_rows(OUT / "sequence_metrics.csv", metric_rows)
np.savez_compressed(OUT / "held_out_sequences_and_posteriors.npz",
                    theta=np.asarray(all_theta), x=np.asarray(all_x),
                    scenario=np.asarray(all_scenario_names),
                    **{name: np.asarray(samples) for name, samples in all_samples.items()})
with (OUT / "experiment_config.json").open("w") as f:
    json.dump({"config": asdict(CFG), "scenarios": [asdict(s) for s in SCENARIOS],
               "training_simulator_calls_shared": CFG.simulation_budget,
               "optional_diagnostic_simulator_calls": (CFG.prior_predictive_plot_samples
                                                       if CFG.simulator_diagnostics_enabled else 0),
               "evaluation_simulator_calls": len(SCENARIOS) * CFG.evaluation_sequences * CFG.sequence_length,
               "inference_simulator_calls": 0,
               "configured_updates_per_model": math.ceil(CFG.simulation_budget / CFG.batch_size) * (1 + CFG.replay_epochs),
               "method_calls": METHOD_CALLS,
               "uncertainty_scope": "held-out trajectories conditional on one training seed",
               "reference_notes": "Grid approximations; relocation oracle knows reset time; neural methods do not."}, f, indent=2)


#%% 9) Paired improvements with trajectory-bootstrap uncertainty
# Positive gain means lower loss than the comparator on the SAME trajectory. Bootstrap intervals
# cover evaluation trajectories only, not variability across training seeds. No test-based tuning.

SCORED_METRICS = ("energy_score", "mean_squared_error", "sw_uniform", "sw_current", "sw_history",
                  "marginal_coverage", "marginal_interval_width", "outside_prior_fraction")
summary_rows, paired_rows = [], []
bootstrap_rng = np.random.default_rng(CFG.seed + 900_001)
for scenario in SCENARIOS:
    values = {name: {metric: np.asarray([
        row[metric] for row in metric_rows if row["scenario"] == scenario.name and row["method"] == name
    ]) for metric in SCORED_METRICS} for name in METHOD_LABELS}
    bootstrap_ids = bootstrap_rng.integers(0, CFG.evaluation_sequences,
                                          (CFG.bootstrap_replicates, CFG.evaluation_sequences))
    for name in METHOD_LABELS:
        summary_rows.append({"scenario": scenario.name, "method": name,
                             **{metric: float(v.mean()) for metric, v in values[name].items()}})
        for comparator in ("sbi_xT", "buffer_xT", "marginal_xT", "refine_xT", "no_replay_history"):
            for metric in ("energy_score", "mean_squared_error", "sw_uniform", "sw_current", "sw_history"):
                gain = values[comparator][metric] - values[name][metric]
                lo, hi = np.quantile(gain[bootstrap_ids].mean(axis=1), [0.025, 0.975])
                paired_rows.append({
                    "scenario": scenario.name, "method": name, "comparator": comparator,
                    "metric": metric, "mean_gain": float(gain.mean()),
                    "ci_low": float(lo), "ci_high": float(hi), "win_fraction": float(np.mean(gain > 0)),
                })
    print(f"\n{scenario.name} — energy-score gain versus paired buffer/current-x:")
    for row in paired_rows:
        if (row["scenario"] == scenario.name and row["comparator"] == "buffer_xT"
            and row["metric"] == "energy_score" and row["method"] in ("refine_xT", "history_raw", "history_defensive")):
            print(f"  {row['method']:22s}: {row['mean_gain']:+.5f} "
                  f"[95% interval {row['ci_low']:+.5f}, {row['ci_high']:+.5f}]")
write_rows(OUT / "summary_metrics.csv", summary_rows)
write_rows(OUT / "paired_improvements.csv", paired_rows)


#%% 10) Visualize latent paths and the single observation at each time

fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(12, 3.1 * len(SCENARIOS)), squeeze=False)
time = np.arange(1, CFG.sequence_length + 1)
for row, scenario in enumerate(SCENARIOS):
    example = example_sequences[scenario.name]
    for dim in range(2):
        axes[row, 0].plot(time, example["theta"][:, dim], "o-", label=rf"$\theta_{{t,{dim + 1}}}$")
        axes[row, 1].plot(time, example["x"][:, dim], "o-", label=rf"$x_{{t,{dim + 1}}}$")
    for ax in axes[row]:
        ax.set(xlabel="Time t", title=scenario.name.replace("_", " "))
        ax.legend()
        if scenario.forced_reset_time is not None:
            ax.axvline(scenario.forced_reset_time, color="crimson", linestyle=":")
    axes[row, 0].set_ylim(CFG.prior_low - 0.05, CFG.prior_high + 0.05)
fig.suptitle("One held-out trajectory per scenario — one simulated observation at each time")
fig.tight_layout()
fig.savefig(OUT / "20_latent_and_observation_paths.png", dpi=180, bbox_inches="tight")
plt.show()


#%% 11) Final posterior panels: distinct current-only and temporal reference targets

PANEL_METHODS = ("sbi_xT", "buffer_xT", "proposal_xT", "refine_xT", "history_raw", "history_defensive")
for scenario in SCENARIOS:
    example = example_sequences[scenario.name]
    axis = example["axis"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, mass, title in zip(axes.ravel()[:2],
                              (example["single_mass"][-1], example["history_mass"][-1]),
                              (r"Grid $p(\theta_T|x_T)$", r"Grid $p(\theta_T|x_{1:T})$")):
        ax.imshow(mass, origin="lower", extent=[CFG.prior_low, CFG.prior_high] * 2,
                  cmap="viridis", aspect="equal")
        ax.set_title(title)
    for ax, name in zip(axes.ravel()[2:], PANEL_METHODS):
        samples = example["final"][name]
        ax.scatter(samples[:, 0], samples[:, 1], s=7, alpha=0.35)
        ax.contour(axis, axis, example["single_mass"][-1],
                   levels=credible_density_levels(example["single_mass"][-1]),
                   colors="grey", linestyles="--", linewidths=0.8)
        ax.contour(axis, axis, example["history_mass"][-1],
                   levels=credible_density_levels(example["history_mass"][-1]),
                   colors="darkorange", linewidths=0.9)
        ax.set_title(METHOD_LABELS[name], fontsize=10)
    for ax in axes.ravel():
        ax.scatter(*example["theta"][-1], marker="*", s=120, c="crimson", edgecolors="white", zorder=5)
        ax.set(xlim=(CFG.prior_low, CFG.prior_high), ylim=(CFG.prior_low, CFG.prior_high),
               xlabel=r"$\theta_{T,1}$", ylabel=r"$\theta_{T,2}$", aspect="equal")
    fig.suptitle(f"{scenario.name}, T={CFG.sequence_length} — grey: current-only; orange: temporal; star: truth")
    fig.tight_layout()
    fig.savefig(OUT / f"30_final_posteriors_{scenario.name}.png", dpi=180, bbox_inches="tight")
    plt.show()


#%% 12) Refinement over time versus repeatedly processing the final observation
# Marginal bands expose collapse/stale history; raw history sees x_t, refinement sees x_T every call.

fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(14, 3.3 * len(SCENARIOS)), squeeze=False)
for row, scenario in enumerate(SCENARIOS):
    example = example_sequences[scenario.name]
    for dim, ax in enumerate(axes[row]):
        for name in ("history_raw", "history_defensive", "refine_xT"):
            cloud_path = example["paths"][name][:, :, dim]
            tail = (1.0 - CFG.credible_mass) / 2.0
            lo, median, hi = np.quantile(cloud_path, [tail, 0.5, 1.0 - tail], axis=1)
            line, = ax.plot(time, median, "o-", label=METHOD_LABELS[name])
            ax.fill_between(time, lo, hi, color=line.get_color(), alpha=0.12)
        ax.plot(time, example["theta"][:, dim], "k--", label="Latent path")
        ax.axhline(example["theta"][-1, dim], color="crimson", linestyle=":", label="Final latent")
        ax.set(title=f"{scenario.name}: coordinate {dim + 1}", xlabel="Time / refinement call",
               ylim=(CFG.prior_low - 0.1, CFG.prior_high + 0.1))
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "40_history_and_refinement_bands.png", dpi=180, bbox_inches="tight")
plt.show()


#%% 13) Paired history gains: current-only baseline and equal-call refinement control

GAIN_METHODS = ("proposal_xT", "refine_xT", "history_raw", "history_defensive", "no_replay_history")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, comparator in zip(axes, ("buffer_xT", "refine_xT")):
    for j, name in enumerate(GAIN_METHODS):
        rows = [next(r for r in paired_rows if r["scenario"] == s.name and r["method"] == name
                     and r["comparator"] == comparator and r["metric"] == "energy_score") for s in SCENARIOS]
        means = np.asarray([r["mean_gain"] for r in rows])
        low, high = np.array([[r["ci_low"], r["ci_high"]] for r in rows]).T
        # Percentile bootstrap endpoints are shown directly; they need not straddle the estimate.
        positions = np.arange(len(SCENARIOS)) + (j - (len(GAIN_METHODS) - 1) / 2) * 0.13
        line, = ax.plot(positions, means, "o", label=METHOD_LABELS[name])
        ax.vlines(positions, low, high, color=line.get_color(), alpha=0.7)
    ax.axhline(0.0, color="black", linestyle="--")
    ax.set_xticks(np.arange(len(SCENARIOS)), [s.name.replace("_", "\n") for s in SCENARIOS])
    ax.set(title=f"Versus {METHOD_LABELS[comparator]}", ylabel="Energy-score gain (positive = better)")
axes[0].legend(fontsize=7)
fig.suptitle("Paired trajectory means and 95% bootstrap intervals; one training seed")
fig.tight_layout()
fig.savefig(OUT / "50_paired_history_gains.png", dpi=180, bbox_inches="tight")
plt.show()


#%% 14) Target fidelity and calibration — improvement must not mean overconfidence

DISPLAY_METHODS = ("sbi_xT", "buffer_xT", "marginal_xT", "history_raw", "history_defensive", "no_replay_history",
                   "reference_current", "reference_history")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, metric in zip(axes.ravel(), ("sw_current", "sw_history", "marginal_coverage", "marginal_interval_width")):
    for name in DISPLAY_METHODS:
        values = [next(r[metric] for r in summary_rows if r["scenario"] == s.name and r["method"] == name)
                  for s in SCENARIOS]
        ax.plot(np.arange(len(SCENARIOS)), values, "o-", label=METHOD_LABELS[name])
    ax.set_xticks(np.arange(len(SCENARIOS)), [s.name.replace("_", "\n") for s in SCENARIOS])
    ax.set_title({"sw_current": r"Distance to $p(\theta_T|x_T)$ (lower is better)",
                  "sw_history": r"Distance to $p(\theta_T|x_{1:T})$ (lower is better)",
                  "marginal_coverage": "Marginal interval coverage at held-out truth",
                  "marginal_interval_width": "Mean marginal interval width"}[metric])
    if metric == "marginal_coverage":
        ax.axhline(CFG.credible_mass, color="black", linestyle="--", label="Nominal coverage")
        ax.set_ylim(0.0, 1.05)
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "60_target_fidelity_and_calibration.png", dpi=180, bbox_inches="tight")
plt.show()

print("\nResults saved to", OUT.resolve())
print("History is useful only if held-out scores improve; narrow clouds alone are not evidence.")
print("A closer temporal posterior can be farther from the current-only posterior: inspect both targets.")
print("The relocation grid knows the change schedule; the learned history method does not.")
print("Nonstationary marginals can shift the prior: compare marginal_xT and the sw_uniform diagnostic too.")
print("Repeat with other Config.seed values and finer grids before drawing scientific conclusions.")

#%%
