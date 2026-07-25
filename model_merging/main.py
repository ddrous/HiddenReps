# %% [markdown]
# # Anchored Weight-Space Merging vs. Naive Averaging
#
# Idea inspired by NOVA's `z̄ + z_t` decomposition: instead of training two fully
# independent networks and averaging their weights (which suffers from weight-space
# symmetry mismatches), we train a **shared base** `θ̄` plus small **per-domain
# offsets** `θ_A`, `θ_B` initialised at zero. Domain B's offset is trained on top of
# a (optionally frozen) base already fit to domain A. Merging is then just averaging
# the small offsets and adding them back to the shared base.
#
# This script is laid out in `# %%` cells so it can be opened directly as a Jupyter
# notebook (VSCode / Jupytext / `jupyter nbconvert --to notebook`).
#
# Run top to bottom. Each cell that produces a figure will render it inline.

# %%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(0)

# %% [markdown]
# ## 1. Config
#
# `DRIFT_BAR_IN_PHASE_B` is the new hyperparameter you asked for: when `True`,
# `θ̄` is *not* frozen during Phase B — it keeps training (typically with a much
# smaller learning rate than the offset), so it can drift into a shared basin
# that's aware of both domains rather than being locked to whatever Phase A left
# it at. When `False`, we reproduce the original "freeze θ̄ after phase A" scheme.

# %%
STEPS = 2500
LR = 5e-3

# --- NEW: control whether the base is allowed to drift in phase B ---
DRIFT_BAR_IN_PHASE_B = False          # <- flip this to False to reproduce the original scheme
BAR_DRIFT_LR_SCALE = 0.1             # bar's LR in phase B = LR * this scale (only used if drifting)

SEED_DATA_A = 1
SEED_DATA_B = 2
SEED_MODEL_TEMPLATE = 42
SEED_NAIVE_A = 7
SEED_NAIVE_B = 8
SEED_JOINT = 99

# %% [markdown]
# ## 2. Data: quadratic pit, two domains straddling the minimum
#
# Domain A sits on the left of the minimum, Domain B on the right, with a small
# unobserved gap around x=0 (the pit itself is never directly observed by either
# domain).

# %%
def true_fn(x):
    return x ** 2

def make_domain(key, lo, hi, n, noise=0.05):
    kx, kn = jax.random.split(key)
    x = jax.random.uniform(kx, (n,), minval=lo, maxval=hi)
    y = true_fn(x) + noise * jax.random.normal(kn, (n,))
    return x[:, None], y[:, None]

kA = jax.random.PRNGKey(SEED_DATA_A)
kB = jax.random.PRNGKey(SEED_DATA_B)
xA, yA = make_domain(kA, -1.8, -0.2, 200)
xB, yB = make_domain(kB, 0.2, 1.8, 200)

x_full_plot = jnp.linspace(-2.2, 2.2, 400)[:, None]
y_full_plot = true_fn(x_full_plot)

# %%
plt.figure(figsize=(7, 5))
plt.plot(x_full_plot, y_full_plot, "k--", lw=1.5, label="true quadratic $y=x^2$")
plt.scatter(xA, yA, s=12, c="tab:blue", label="Domain A (left of pit)")
plt.scatter(xB, yB, s=12, c="tab:red", label="Domain B (right of pit)")
plt.axvspan(-0.2, 0.2, color="gray", alpha=0.15, label="unseen gap (the pit)")
plt.legend()
plt.title("Data: two domains straddling the minimum")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Model + generic utilities

# %%
def make_mlp(key):
    return eqx.nn.MLP(
        in_size=1, out_size=1, width_size=32, depth=3,
        activation=jax.nn.tanh, key=key,
    )

def zeros_like_pytree(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)

def add_pytrees(t1, t2):
    return jax.tree_util.tree_map(lambda a, b: a + b, t1, t2)

def avg_pytrees(t1, t2):
    return jax.tree_util.tree_map(lambda a, b: 0.5 * (a + b), t1, t2)

def predict(params, static, x):
    model = eqx.combine(params, static)
    return jax.vmap(model)(x)

def mse_loss(params, static, x, y):
    pred = predict(params, static, x)
    return jnp.mean((pred - y) ** 2)

def train(params, static, x, y, steps=2000, lr=1e-2):
    """Plain trainer for a single full parameter pytree (used for baselines)."""
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def step(params, opt_state):
        loss, grads = eqx.filter_value_and_grad(mse_loss)(params, static, x, y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    for i in range(steps):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))
    return params, losses

# %%
template_model = make_mlp(jax.random.PRNGKey(SEED_MODEL_TEMPLATE))
params_template, static = eqx.partition(template_model, eqx.is_array)

# %% [markdown]
# ## 4. Anchored merging method
#
# `train_anchored` optimises a pair `(params_bar, theta)` against
# `predict(bar + theta)`. Per-leaf learning rates are handled via
# `optax.multi_transform`, so `params_bar` and `theta` can each have their own
# learning rate (including LR=0, i.e. fully frozen).

# %%
def train_anchored(params_bar, theta, static, x, y, steps, lr_bar, lr_theta):
    """
    Jointly (or partially) trains params_bar and theta so that
    predict(params_bar + theta, x) fits (x, y).

    lr_bar   : learning rate for the shared base. Use 0.0 to freeze it.
    lr_theta : learning rate for the per-domain offset.
    """
    combo0 = (params_bar, theta)

    # Label each leaf of the (bar, theta) pytree so multi_transform can give
    # bar-leaves and theta-leaves different optimisers/learning rates.
    labels = (
        jax.tree_util.tree_map(lambda _: "bar", params_bar),
        jax.tree_util.tree_map(lambda _: "theta", theta),
    )
    opt = optax.multi_transform(
        {"bar": optax.adam(lr_bar) if lr_bar > 0 else optax.set_to_zero(),
         "theta": optax.adam(lr_theta) if lr_theta > 0 else optax.set_to_zero()},
        labels,
    )
    opt_state = opt.init(combo0)

    def loss_fn(combo, static, x, y):
        pb, th = combo
        summed = add_pytrees(pb, th)
        pred = predict(summed, static, x)
        return jnp.mean((pred - y) ** 2)

    @eqx.filter_jit
    def step(combo, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(combo, static, x, y)
        updates, opt_state = opt.update(grads, opt_state, combo)
        combo = eqx.apply_updates(combo, updates)
        return combo, opt_state, loss

    combo = combo0
    losses = []
    for i in range(steps):
        combo, opt_state, loss = step(combo, opt_state)
        losses.append(float(loss))
    return combo[0], combo[1], losses

# %% [markdown]
# ### Phase A — train θ̄ (random init) + θ_A (zero init) jointly on Domain A

# %%
theta_A_zero = zeros_like_pytree(params_template)
params_bar_afterA, theta_A, lossesA_anchor = train_anchored(
    params_template, theta_A_zero, static, xA, yA,
    steps=STEPS, lr_bar=LR, lr_theta=LR,
)
print("Phase A final loss:", lossesA_anchor[-1])

# %% [markdown]
# ### Phase B — train θ_B (zero init) on Domain B.
#
# `DRIFT_BAR_IN_PHASE_B` controls whether θ̄ keeps updating here (with a scaled-down
# LR) or is fully frozen at its Phase-A value.

# %%
theta_B_zero = zeros_like_pytree(params_template)
bar_lr_phaseB = (LR * BAR_DRIFT_LR_SCALE) if DRIFT_BAR_IN_PHASE_B else 0.0

params_bar_final, theta_B, lossesB_anchor = train_anchored(
    params_bar_afterA, theta_B_zero, static, xB, yB,
    steps=STEPS, lr_bar=bar_lr_phaseB, lr_theta=LR,
)
print(f"Phase B final loss: {lossesB_anchor[-1]:.5f}  "
      f"(bar drift {'ON' if DRIFT_BAR_IN_PHASE_B else 'OFF'}, bar_lr={bar_lr_phaseB})")

# %%
# Merge: average the two offsets, add to the (possibly drifted) shared base
theta_merged = avg_pytrees(theta_A, theta_B)
params_anchored_merged = add_pytrees(params_bar_final, theta_merged)

# Specialists (pre-merge) for reference — note specialist B uses the FINAL bar,
# specialist A uses the bar as it stood right after phase A, since that's what
# it was actually trained against.
params_specialist_A = add_pytrees(params_bar_afterA, theta_A)
params_specialist_B = add_pytrees(params_bar_final, theta_B)

# %% [markdown]
# ## 5. Baselines

# %% [markdown]
# ### Baseline 1: naive independent training (different random inits), then plain averaging
# This is the classic weight-symmetry-mismatch failure case.

# %%
keyA2, keyB2 = jax.random.split(jax.random.PRNGKey(SEED_NAIVE_A + SEED_NAIVE_B))
modelA_naive = make_mlp(jax.random.PRNGKey(SEED_NAIVE_A))
modelB_naive = make_mlp(jax.random.PRNGKey(SEED_NAIVE_B))
paramsA_naive0, _ = eqx.partition(modelA_naive, eqx.is_array)
paramsB_naive0, _ = eqx.partition(modelB_naive, eqx.is_array)

paramsA_naive, lossesA_naive = train(paramsA_naive0, static, xA, yA, steps=STEPS, lr=LR)
paramsB_naive, lossesB_naive = train(paramsB_naive0, static, xB, yB, steps=STEPS, lr=LR)
params_naive_avg = avg_pytrees(paramsA_naive, paramsB_naive)
print("Naive (diff init) - final A/B loss:", lossesA_naive[-1], lossesB_naive[-1])

# %% [markdown]
# ### Baseline 2: naive training from the SAME initialisation, then averaging
# Isolates the effect of init-alignment from the anchoring mechanism itself.

# %%
paramsA_sameinit, lossesA_si = train(params_template, static, xA, yA, steps=STEPS, lr=LR)
paramsB_sameinit, lossesB_si = train(params_template, static, xB, yB, steps=STEPS, lr=LR)
params_sameinit_avg = avg_pytrees(paramsA_sameinit, paramsB_sameinit)
print("Naive (same init) - final A/B loss:", lossesA_si[-1], lossesB_si[-1])

# %% [markdown]
# ### Upper bound: single model jointly trained on both domains at once

# %%
xJoint = jnp.concatenate([xA, xB], axis=0)
yJoint = jnp.concatenate([yA, yB], axis=0)
modelJ = make_mlp(jax.random.PRNGKey(SEED_JOINT))
paramsJ0, _ = eqx.partition(modelJ, eqx.is_array)
params_joint, losses_joint = train(paramsJ0, static, xJoint, yJoint, steps=STEPS, lr=LR)
print("Joint upper bound - final loss:", losses_joint[-1])

# %% [markdown]
# ## 6. Evaluation

# %%
def mse_np(params, static, x, y):
    return float(mse_loss(params, static, x, y))

x_eval_full = jnp.linspace(-2.2, 2.2, 400)[:, None]
y_eval_full = true_fn(x_eval_full)

results = {}

def eval_all(name, params):
    results[name] = {
        "A": mse_np(params, static, xA, yA),
        "B": mse_np(params, static, xB, yB),
        "full": mse_np(params, static, x_eval_full, y_eval_full),
    }

eval_all("Specialist A only (bar_afterA + thetaA)", params_specialist_A)
eval_all("Specialist B only (bar_final + thetaB)", params_specialist_B)
tag = "ANCHORED MERGE (drift ON)" if DRIFT_BAR_IN_PHASE_B else "ANCHORED MERGE (drift OFF)"
eval_all(tag, params_anchored_merged)
eval_all("Naive avg (different init)", params_naive_avg)
eval_all("Naive avg (same init, no anchor)", params_sameinit_avg)
eval_all("Joint training (upper bound)", params_joint)

print(f"\n{'Method':45s} {'MSE@A':>10s} {'MSE@B':>10s} {'MSE@full':>10s}")
for k, v in results.items():
    print(f"{k:45s} {v['A']:10.4f} {v['B']:10.4f} {v['full']:10.4f}")

# %% [markdown]
# ## 7. Visualisations

# %%
xg = np.array(x_eval_full).ravel()
yg = np.array(y_eval_full).ravel()

def pred_np(params):
    return np.array(predict(params, static, x_eval_full)).ravel()

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

ax = axes[0]
ax.plot(xg, yg, "k--", lw=2, label="true $y=x^2$", zorder=1)
ax.scatter(xA, yA, s=8, c="tab:blue", alpha=0.5, label="Domain A data")
ax.scatter(xB, yB, s=8, c="tab:red", alpha=0.5, label="Domain B data")
ax.plot(xg, pred_np(params_specialist_A), c="tab:blue", lw=2, label="Specialist A (bar+θA)")
ax.plot(xg, pred_np(params_specialist_B), c="tab:red", lw=2, label="Specialist B (bar+θB)")
ax.set_title("Specialists before merging (each on its own domain)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8); ax.set_ylim(-0.5, 6)

ax = axes[1]
ax.plot(xg, yg, "k--", lw=2, label="true $y=x^2$", zorder=1)
ax.scatter(xA, yA, s=8, c="tab:blue", alpha=0.3)
ax.scatter(xB, yB, s=8, c="tab:red", alpha=0.3)
ax.plot(xg, pred_np(params_anchored_merged), c="tab:green", lw=2.5, label=tag)
ax.plot(xg, pred_np(params_naive_avg), c="tab:orange", lw=2, ls="-.", label="Naive avg (diff. init)")
ax.plot(xg, pred_np(params_sameinit_avg), c="tab:purple", lw=2, ls=":", label="Naive avg (same init)")
ax.plot(xg, pred_np(params_joint), c="gray", lw=1.5, ls="--", label="Joint training (upper bound)")
ax.set_title("Merged models evaluated on BOTH domains")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8); ax.set_ylim(-0.5, 6)

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(lossesA_anchor, label="Anchored: Phase A (bar+θA jointly)", c="tab:blue")
ax.plot(lossesB_anchor, label=f"Anchored: Phase B (θB{'+bar drift' if DRIFT_BAR_IN_PHASE_B else ', bar frozen'})", c="tab:red")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE (log)")
ax.set_title("Anchored method training curves"); ax.legend(fontsize=8)

ax = axes[1]
ax.plot(lossesA_naive, label="Naive A (diff init)", c="tab:blue", alpha=0.6)
ax.plot(lossesB_naive, label="Naive B (diff init)", c="tab:red", alpha=0.6)
ax.plot(lossesA_si, label="Naive A (same init)", c="tab:blue", ls="--")
ax.plot(lossesB_si, label="Naive B (same init)", c="tab:red", ls="--")
ax.plot(losses_joint, label="Joint (upper bound)", c="gray")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE (log)")
ax.set_title("Baseline training curves"); ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
methods = list(results.keys())
mseA = [results[m]["A"] for m in methods]
mseB = [results[m]["B"] for m in methods]
mseFull = [results[m]["full"] for m in methods]

x_pos = np.arange(len(methods))
w = 0.25
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x_pos - w, mseA, width=w, label="MSE on Domain A", color="tab:blue")
ax.bar(x_pos, mseB, width=w, label="MSE on Domain B", color="tab:red")
ax.bar(x_pos + w, mseFull, width=w, label="MSE on full range", color="gray")
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("MSE")
ax.set_yscale("log")
ax.set_title("Merged/specialist performance comparison (log scale)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Try it: flip `DRIFT_BAR_IN_PHASE_B` above (and re-run from that cell down)
# to compare the frozen-base scheme against the drifting-base scheme directly.
# You can also sweep `BAR_DRIFT_LR_SCALE` (e.g. 0.0, 0.05, 0.1, 0.3, 1.0) to see
# how much drift is helpful before it starts eroding what Phase A learned.