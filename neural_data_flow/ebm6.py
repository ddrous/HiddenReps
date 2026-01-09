#%%
#%%
import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", context="talk")
from pathlib import Path
import time
import json
import datetime
from typing import List, Tuple, Optional, Any

# --- 1. CONFIGURATION ---
TRAIN = True
RUN_DIR = "" 
CONFIG = {
    # "seed": 2026,
    "seed": time.time_ns() % (2**32 - 1),
    "lr_nn": 0.0001,    # Standard Models & PAM NN
    "lr_ctx": 0.01,    # PAM Contexts
    "batch_size": 64,
    "epochs": 300,     
    "context_init": "zero", # "zero" or "random"
    
    # Inner Loop Hyperparameters
    "inner_gd_steps": 20,
    "inner_gd_lr": 0.1, 
    
    # Model Hyperparameters
    "taylor_radius": 0.1,
    "taylor_order_mlp": 2,      # Order for TaylorMLP
    "taylor_order_ebm_x": 0,    # Order for TaylorBaseEBM and TaylorContextEBM (w.r.t input x)
    "taylor_order_ebm_y": 2,    # New: Order for ALL EBMs (w.r.t target y)
    "taylor_order_ebm_c": 0,    # Order for TaylorContextEBM (w.r.t context c)
    
    "context_dim": 1,
    "width_size": 48,
    "noise_std": 0.005,
    "data_samples": 2000,
    "segments": 11,
    "x_range": [-1.5, 1.5],

    # PAM Hyperparameters
    "pam_outer_steps_max": 60,
    "pam_inner_steps_nn": 10,
    "pam_inner_steps_ctx": 10,
    "pam_proximal_reg": 100.0,
    "pam_tol_nn": 1e-3,
    "pam_tol_ctx": 1e-3,
    "print_every": 10
}

#%%
# --- 2. UTILITY FUNCTIONS ---

def get_run_path(base_dir="experiments"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_path = Path(base_dir) / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path

def save_run(run_dir, model, model_name, config, history):
    model_path = run_dir / f"{model_name}.eqx"
    eqx.tree_serialise_leaves(model_path, model)
    with open(run_dir / f"{model_name}_config.json", "w") as f:
        json.dump(config, f, indent=4)
    np.savez(run_dir / f"{model_name}_history.npz", **history)

def flatten_pytree(pytree):
    """ Flatten the leaves of a pytree into a single array. 
        Return the array, the shapes of the leaves and the tree_def. """
    leaves, tree_def = jtree.tree_flatten(pytree)
    if len(leaves) == 0:
        return jnp.array([]), [], tree_def
    flat = jnp.concatenate([x.flatten() for x in leaves])
    shapes = [x.shape for x in leaves]
    return flat, shapes, tree_def

def unflatten_pytree(flat, shapes, tree_def):
    """ Reconstructs a pytree given its leaves flattened, their shapes, and the treedef. """
    leaves_prod = [0] + [np.prod(x) for x in shapes]
    lpcum = np.cumsum(leaves_prod)
    leaves = [flat[lpcum[i-1]:lpcum[i]].reshape(shapes[i-1]) for i in range(1, len(lpcum))]
    return jtree.tree_unflatten(tree_def, leaves)

def params_diff_norm_squared(params1, params2):
    params1 = eqx.filter(params1, eqx.is_array)
    params2 = eqx.filter(params2, eqx.is_array)
    diff_tree = jax.tree_util.tree_map(lambda x, y: x - y, params1, params2)
    diff_flat, _, _ = flatten_pytree(diff_tree)
    if diff_flat.shape[0] == 0: return 0.0
    return (diff_flat.T @ diff_flat) / diff_flat.shape[0]

def params_norm_squared(params):
    params = eqx.filter(params, eqx.is_array)
    flat, _, _ = flatten_pytree(params)
    if flat.shape[0] == 0: return 0.0
    return (flat.T @ flat) / flat.shape[0]

#%%
# --- 3. DATA GENERATION ---

def gen_data(seed, n_samples, n_segments=3, local_structure="random", 
             x_range=[-1, 1], slope=2.0, base_intercept=0.0, 
             step_size=4.0, custom_func=None, noise_std=0.5):
    np.random.seed(seed)
    x_min, x_max = x_range
    segment_boundaries = np.linspace(x_min, x_max, n_segments + 1)
    samples_per_seg = [n_samples // n_segments + (1 if i < n_samples % n_segments else 0) 
                       for i in range(n_segments)]
    all_x, all_y, segment_ids = [], [], []
    
    for i in range(n_segments):
        seg_x_min, seg_x_max = segment_boundaries[i], segment_boundaries[i+1]
        n_seg_samples = samples_per_seg[i]
        x_seg = np.random.uniform(seg_x_min, seg_x_max, n_seg_samples)
        
        b = 0
        if local_structure == "constant": b = base_intercept
        elif local_structure == "random": b = np.random.uniform(-5, 5) 
        elif local_structure == "gradual_increase": b = base_intercept + (i * step_size)
        elif local_structure == "gradual_decrease": b = base_intercept - (i * step_size)
        elif local_structure == "custom":
            if custom_func is None: raise ValueError("Need custom_func")
            b = custom_func((seg_x_min + seg_x_max) / 2)
            
        noise = np.random.normal(0, noise_std, n_seg_samples)
        y_seg = (slope * x_seg) + b + noise
        all_x.append(x_seg)
        all_y.append(y_seg)
        segment_ids.append(np.full(n_seg_samples, i))
        
    data = np.column_stack((np.concatenate(all_x), np.concatenate(all_y)))
    return data, np.concatenate(segment_ids)

class SimpleDataHandler:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(x))
    
    def get_iterator(self):
        np.random.shuffle(self.indices)
        for start_idx in range(0, len(self.x), self.batch_size):
            batch_idx = self.indices[start_idx:start_idx + self.batch_size]
            yield self.x[batch_idx], self.y[batch_idx], batch_idx 

# Generate Data
SEED = CONFIG["seed"]
TRAIN_SEG_IDS = [2, 3, 4, 5, 6, 7, 8]

data, segs = gen_data(SEED, CONFIG["data_samples"], n_segments=CONFIG["segments"], 
                      local_structure="gradual_increase", x_range=CONFIG["x_range"], 
                      slope=0.5, base_intercept=-0.4, step_size=0.2, noise_std=CONFIG["noise_std"])

test_mask = ~np.isin(segs, TRAIN_SEG_IDS)
train_mask = np.isin(segs, TRAIN_SEG_IDS)

train_data, train_segs = data[train_mask], segs[train_mask]
test_data, test_segs = data[test_mask], segs[test_mask]

X_train = jnp.array(train_data[:, 0])[:, None]
Y_train = jnp.array(train_data[:, 1])[:, None]
X_test = jnp.array(test_data[:, 0])[:, None]
Y_test = jnp.array(test_data[:, 1])[:, None]

dm = SimpleDataHandler(X_train, Y_train, CONFIG["batch_size"])

#%%
#%%
# --- 4. MODEL DEFINITIONS ---
inner_gd_steps = CONFIG["inner_gd_steps"]
width_size = CONFIG["width_size"]
inner_gd_lr = CONFIG["inner_gd_lr"]

# --- HELPER FOR TAYLOR EXPANSION WRT Y ---
def taylor_energy_y(energy_fn, y_target, y_anchor, order):
    """
    Expands energy_fn(y) around y_anchor to specified order, evaluated at y_target.
    energy_fn: callable taking (y) -> scalar
    """
    if order == 0:
        return energy_fn(y_target)
    
    tangent = y_target - y_anchor
    
    if order == 1:
        e0, jvp = jax.jvp(energy_fn, (y_anchor,), (tangent,))
        return e0 + jvp
        
    elif order == 2:
        def grad_energy(y):
            return jax.jvp(energy_fn, (y,), (tangent,))[1]
        
        e0, term1 = jax.jvp(energy_fn, (y_anchor,), (tangent,))
        _, term2 = jax.jvp(grad_energy, (y_anchor,), (tangent,))
        return e0 + term1 + 0.5 * term2
    
    return energy_fn(y_target)

# --- STANDARD MODELS ---
class LinearModel(eqx.Module):
    layer: eqx.nn.Linear
    def __init__(self, key, context_dim=None): 
        self.layer = eqx.nn.Linear(1, 1, key=key)
    def __call__(self, x, c=None, key=None):
        return jax.vmap(self.layer)(x), None 
    def predict(self, x):
        pred, _ = self(x)
        return pred

width_size = CONFIG["width_size"]
class MLPModel(eqx.Module):
    layers: list
    def __init__(self, key, context_dim=None):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(1, width_size, key=k1), jax.nn.relu,
                       eqx.nn.Linear(width_size, width_size, key=k2), jax.nn.relu,
                       eqx.nn.Linear(width_size, width_size, key=k2), jax.nn.relu,
                       eqx.nn.Linear(width_size, width_size, key=k2), jax.nn.relu,
                    #    eqx.nn.Linear(width_size, width_size, key=k2), jax.nn.relu,
                       eqx.nn.Linear(width_size, 1, key=k3)]
    def _forward(self, x):
        for l in self.layers: x = l(x)
        return x
    def __call__(self, x, c=None, key=None):
        return jax.vmap(self._forward)(x), None
    def predict(self, x):
        pred, _ = self(x)
        return pred

class TaylorMLP(eqx.Module):
    layers: list
    radius: float
    order: int
    
    def __init__(self, key, radius=0.2, context_dim=None, order=1):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [eqx.nn.Linear(1, 64, key=k1), jax.nn.relu,
                       eqx.nn.Linear(64, 64, key=k2), jax.nn.relu,
                       eqx.nn.Linear(64, 1, key=k3)]
        self.radius = radius
        self.order = order
        
    def _forward(self, x):
        for l in self.layers: x = l(x)
        return x

    def __call__(self, x, c=None, key=None):
        if key is None or self.order == 0: 
            return self.predict(x), None
        
        epsilon = jax.random.uniform(key, x.shape, minval=-self.radius, maxval=self.radius)
        x0 = x + epsilon
        
        def approx(xi, x0i):
            tangent = xi - x0i
            if self.order == 1:
                y0, jvp = jax.jvp(self._forward, (x0i,), (tangent,))
                return y0 + jvp
            elif self.order == 2:
                def forward_jvp(primals):
                    return jax.jvp(self._forward, (primals,), (tangent,))[1]
                y0, term1 = jax.jvp(self._forward, (x0i,), (tangent,))
                _, term2 = jax.jvp(forward_jvp, (x0i,), (tangent,))
                return y0 + term1 + 0.5 * term2
            return self._forward(xi) 

        return jax.vmap(approx)(x, x0), None

    def predict(self, x):
        return jax.vmap(self._forward)(x)

class BaseEBM(eqx.Module):
    neuralnet: eqx.nn.MLP
    y_mean: float
    order_y: int
    radius: float 

    def __init__(self, key, y_mean=0.0, context_dim=None, order_y=0, radius=0.2):
        self.neuralnet = eqx.nn.MLP(in_size=2, out_size=1, width_size=width_size, depth=3, 
                                    activation=jax.nn.swish, key=key)
        self.y_mean = y_mean
        self.order_y = order_y
        self.radius = radius 

    def energy(self, x, y):
        inp = jnp.concatenate([x, y], axis=0)
        return jnp.squeeze(self.neuralnet(inp))

    def _solve_y(self, x, key=None):
        y_init = jnp.array([self.y_mean])
        opt = optax.adam(inner_gd_lr)
        opt_state = opt.init(y_init)
        
        if key is not None:
             step_keys = jax.random.split(key, inner_gd_steps)
        else:
             step_keys = [None] * inner_gd_steps

        def step(state, i):
            y_curr, opt_state = state
            
            # Y Expansion Point
            if self.order_y > 0 and key is not None:
                step_key = step_keys[i] # Single key (2,)
                # Generate noise for this specific sample
                epsilon = jax.random.uniform(step_key, y_curr.shape, minval=-self.radius, maxval=self.radius)
                y0 = y_curr + epsilon
            else:
                y0 = y_curr

            def loss_fn(y_arg):
                def energy_wrt_y(yi): return self.energy(x, yi)
                return taylor_energy_y(energy_wrt_y, y_arg, y0, self.order_y)

            grads = jax.grad(loss_fn)(y_curr)
            updates, opt_state = opt.update(grads, opt_state, y_curr)
            y_next = optax.apply_updates(y_curr, updates)
            return (y_next, opt_state), None
            
        (y_final, _), _ = jax.lax.scan(step, (y_init, opt_state), jnp.arange(inner_gd_steps))
        return y_final

    def __call__(self, x, c=None, key=None):
        if key is not None:
            # BaseEBM uses vmap over _solve_y, so we need one key per sample
            keys = jax.random.split(key, x.shape[0])
            return jax.vmap(self._solve_y)(x, keys), None
        else:
            return jax.vmap(self._solve_y)(x, None), None

    def predict(self, x):
        pred, _ = self(x)
        return pred

class TaylorBaseEBM(eqx.Module):
    neuralnet: eqx.nn.MLP
    y_mean: float
    radius: float
    order_x: int
    order_y: int
    
    def __init__(self, key, y_mean=0.0, radius=0.2, order=1, order_y=0, context_dim=None):
        self.neuralnet = eqx.nn.MLP(in_size=2, out_size=1, width_size=width_size, depth=3, 
                                    activation=jax.nn.swish, key=key)
        self.y_mean = y_mean
        self.radius = radius
        self.order_x = order
        self.order_y = order_y

    def energy(self, x, y):
        inp = jnp.concatenate([x, y], axis=0)
        return jnp.squeeze(self.neuralnet(inp))

    def _solve_y(self, x, key=None):
        y_init = jnp.array([self.y_mean])
        opt = optax.adam(inner_gd_lr)
        opt_state = opt.init(y_init)

        if key is not None:
            k1, k2 = jax.random.split(key)
            eps_x = jax.random.uniform(k1, x.shape, minval=-self.radius, maxval=self.radius)
            x0 = x + eps_x
            step_keys = jax.random.split(k2, inner_gd_steps)
        else:
            x0 = x
            step_keys = [None] * inner_gd_steps

        def step(state, i):
            y_curr, opt_state = state
            
            if self.order_y > 0 and key is not None:
                step_key = step_keys[i]
                eps_y = jax.random.uniform(step_key, y_curr.shape, minval=-self.radius, maxval=self.radius)
                y0 = y_curr + eps_y
            else:
                y0 = y_curr

            def loss_fn(y_arg):
                def energy_expanded_x(yi):
                    def energy_wrt_x(xi): return self.energy(xi, yi)
                    
                    if self.order_x == 0 or key is None:
                        return self.energy(x, yi)
                    
                    tangent_x = x - x0
                    if self.order_x == 1:
                        e0, jvp = jax.jvp(energy_wrt_x, (x0,), (tangent_x,))
                        return e0 + jvp
                    elif self.order_x == 2:
                        def e_jvp(p): return jax.jvp(energy_wrt_x, (p,), (tangent_x,))[1]
                        e0, t1 = jax.jvp(energy_wrt_x, (x0,), (tangent_x,))
                        _, t2 = jax.jvp(e_jvp, (x0,), (tangent_x,))
                        return e0 + t1 + 0.5 * t2
                    return self.energy(x, yi)

                return taylor_energy_y(energy_expanded_x, y_arg, y0, self.order_y)

            grads = jax.grad(loss_fn)(y_curr)
            updates, opt_state = opt.update(grads, opt_state, y_curr)
            y_next = optax.apply_updates(y_curr, updates)
            return (y_next, opt_state), None

        (y_final, _), _ = jax.lax.scan(step, (y_init, opt_state), jnp.arange(inner_gd_steps))
        return y_final

    def __call__(self, x, c=None, key=None):
        if key is not None:
            keys = jax.random.split(key, x.shape[0])
            return jax.vmap(self._solve_y)(x, keys), None
        else:
            return jax.vmap(self._solve_y)(x, None), None

    def predict(self, x):
        pred, _ = self(x, key=None)
        return pred

# --- PAM COMPATIBLE CLASSES ---

class Contexts(eqx.Module):
    params: jnp.ndarray
    def __init__(self, key, num_samples, dim, init_strategy="zero"):
        if init_strategy == "zero":
            self.params = jnp.zeros((num_samples, dim))
        elif init_strategy == "random":
            self.params = jax.random.normal(key, (num_samples, dim)) * 0.1
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")
    def __call__(self, indices):
        return self.params[indices]

class EBMNet(eqx.Module):
    hypernet: eqx.nn.MLP
    target_static: Any = eqx.field(static=True)
    target_shapes: List[Tuple] = eqx.field(static=True)
    target_treedef: Any = eqx.field(static=True)
    y_mean: float
    c_dim: int
    order_y: int
    radius: float

    def __init__(self, key, context_dim, y_mean=0.0, order_y=0, radius=0.2):
        k1, k2 = jax.random.split(key)
        self.y_mean = y_mean
        self.c_dim = context_dim
        self.order_y = order_y
        self.radius = radius
        
        target_template = eqx.nn.MLP(in_size=1, out_size=1, width_size=width_size, depth=3, 
                                     activation=jax.nn.swish, key=k1)
        target_flat, self.target_shapes, self.target_treedef = flatten_pytree(
            eqx.filter(target_template, eqx.is_array)
        )
        target_param_size = target_flat.shape[0]
        _, self.target_static = eqx.partition(target_template, eqx.is_array)
        self.hypernet = eqx.nn.MLP(in_size=context_dim + 1, out_size=target_param_size,
                                   width_size=width_size * 2, depth=1, 
                                   activation=jax.nn.relu, key=k2)

    def _get_target_net(self, flat_params):
        target_params = unflatten_pytree(flat_params, self.target_shapes, self.target_treedef)
        return eqx.combine(target_params, self.target_static)

    def __call__(self, x, c, key=None):
        return self._solve_training(x, c, key)

    def _solve_training(self, x_batch, c_batch, key=None):
        hyper_input = jnp.concatenate([x_batch, c_batch], axis=1)
        flat_params_batch = jax.vmap(self.hypernet)(hyper_input)
        y_init = jnp.full((x_batch.shape[0], 1), self.y_mean)
        opt = optax.adam(inner_gd_lr)
        opt_state = opt.init(y_init)
        
        # --- FIX: Split keys for batched noise generation ---
        if key is not None:
            # We only need (Steps, 2) to generate noise for the full batch at each step
            # because uniform(key, shape=(Batch, 1)) uses one key to fill the shape.
            step_keys = jax.random.split(key, inner_gd_steps) 
        else:
            step_keys = [None] * inner_gd_steps

        def step(state, inputs):
            y_curr, opt_state = state
            key_step = inputs # Single key (2,) or None
            
            # Y Anchor Sampling
            if self.order_y > 0 and key is not None:
                # Generate (Batch, 1) noise using SINGLE key_step
                eps = jax.random.uniform(key_step, y_curr.shape, minval=-self.radius, maxval=self.radius)
                y0 = y_curr + eps
            else:
                y0 = y_curr

            def energy_fn(y, flat_params, y_anchor):
                model = self._get_target_net(flat_params)
                def e_y(yi): return jnp.squeeze(model(yi))
                return taylor_energy_y(e_y, y, y_anchor, self.order_y)
                
            grads = jax.vmap(jax.grad(energy_fn))(y_curr, flat_params_batch, y0)
            updates, opt_state = opt.update(grads, opt_state, y_curr)
            y_next = optax.apply_updates(y_curr, updates)
            return (y_next, opt_state), None
            
        (y_final, _), _ = jax.lax.scan(step, (y_init, opt_state), step_keys, length=inner_gd_steps)
        return y_final

    def predict(self, x_batch):
        return self._solve_inference(x_batch)

    def _solve_inference(self, x_batch):
        B = x_batch.shape[0]
        y_init = jnp.full((B, 1), self.y_mean)
        c_init = jnp.zeros((B, self.c_dim))
        opt = optax.adam(inner_gd_lr)
        opt_state = opt.init((y_init, c_init))
        
        def step(state, _):
            y_curr, c_curr, opt_state = state
            def energy_c(y, c, x):
                hyper_in = jnp.concatenate([x, c], axis=0)
                flat_params = self.hypernet(hyper_in)
                model = self._get_target_net(flat_params)
                return jnp.squeeze(model(y))
            grads = jax.vmap(jax.grad(energy_c, argnums=(0, 1)))(y_curr, c_curr, x_batch)
            updates, opt_state = opt.update(grads, opt_state, (y_curr, c_curr))
            y_next, c_next = optax.apply_updates((y_curr, c_curr), updates)
            return (y_next, c_next, opt_state), None
        (y_final, c_final, _), _ = jax.lax.scan(step, (y_init, c_init, opt_state), None, length=inner_gd_steps)
        return y_final

class TaylorEBMNet(EBMNet):
    radius: float
    order_x: int
    order_c: int
    
    def __init__(self, key, context_dim, y_mean=0.0, radius=0.2, order_x=1, order_c=1, order_y=0):
        super().__init__(key, context_dim, y_mean, order_y, radius)
        self.radius = radius
        self.order_x = order_x
        self.order_c = order_c

    def _solve_training(self, x_batch, c_batch, key):
        if key is None or (self.order_x == 0 and self.order_c == 0 and self.order_y == 0):
            return super()._solve_training(x_batch, c_batch, key)

        # A. Context Expansion Neighbors
        dists = jnp.linalg.norm(c_batch[:, None, :] - c_batch[None, :, :], axis=-1)
        dists = dists + jnp.eye(dists.shape[0]) * 1e9
        nearest_idx = jnp.argmin(dists, axis=1)
        c0_batch = c_batch[nearest_idx]
        
        # B. Input Expansion Setup (Need split key)
        if key is not None:
            k_x, k_y = jax.random.split(key)
            epsilon = jax.random.uniform(k_x, x_batch.shape, minval=-self.radius, maxval=self.radius)
            x0_batch = x_batch + epsilon
            
            # --- FIX: Generate single keys per step for batch noise ---
            step_keys = jax.random.split(k_y, inner_gd_steps)
        else:
            x0_batch = x_batch
            step_keys = [None] * inner_gd_steps

        # C. Taylor Weights Helper
        def get_weights_ctx_taylor(x, c, c0):
             def hyper_closure(c_arg): return self.hypernet(jnp.concatenate([x, c_arg], axis=0))
             
             if self.order_c == 0: return hyper_closure(c)
             tangent = c - c0
             if self.order_c == 1:
                 w0, jvp = jax.jvp(hyper_closure, (c0,), (tangent,))
                 return w0 + jvp
             elif self.order_c == 2:
                 def h_jvp(p): return jax.jvp(hyper_closure, (p,), (tangent,))[1]
                 w0, t1 = jax.jvp(hyper_closure, (c0,), (tangent,))
                 _, t2 = jax.jvp(h_jvp, (c0,), (tangent,))
                 return w0 + t1 + 0.5 * t2
             return hyper_closure(c)

        # D. Optimization Loop
        y_init = jnp.full((x_batch.shape[0], 1), self.y_mean)
        opt = optax.adam(inner_gd_lr)
        opt_state = opt.init(y_init)

        def step(state, inputs):
            y_curr, opt_state = state
            key_step = inputs
            
            # Y Anchor
            if self.order_y > 0 and key is not None:
                # Generate (Batch, 1) noise using single step_key
                eps = jax.random.uniform(key_step, y_curr.shape, minval=-self.radius, maxval=self.radius)
                y0 = y_curr + eps
            else:
                y0 = y_curr
            
            def energy_final(y, x, x0, c, c0, y_anchor):
                def energy_taylor_x(yi):
                    def energy_wrt_x(x_arg):
                        weights = get_weights_ctx_taylor(x_arg, c, c0)
                        model = self._get_target_net(weights)
                        return jnp.squeeze(model(yi))
                    
                    if self.order_x == 0: return energy_wrt_x(x)
                    
                    tangent = x - x0
                    if self.order_x == 1:
                        e0, jvp = jax.jvp(energy_wrt_x, (x0,), (tangent,))
                        return e0 + jvp
                    elif self.order_x == 2:
                        def e_jvp(p): return jax.jvp(energy_wrt_x, (p,), (tangent,))[1]
                        e0, t1 = jax.jvp(energy_wrt_x, (x0,), (tangent,))
                        _, t2 = jax.jvp(e_jvp, (x0,), (tangent,))
                        return e0 + t1 + 0.5 * t2
                    return energy_wrt_x(x)

                return taylor_energy_y(energy_taylor_x, y, y_anchor, self.order_y)

            grads = jax.vmap(jax.grad(energy_final, argnums=0))(
                y_curr, x_batch, x0_batch, c_batch, c0_batch, y0
            )
            
            updates, opt_state = opt.update(grads, opt_state, y_curr)
            y_next = optax.apply_updates(y_curr, updates)
            return (y_next, opt_state), None

        (y_final, _), _ = jax.lax.scan(step, (y_init, opt_state), step_keys, length=inner_gd_steps)
        return y_final

# --- CONTAINERS ---
class ContextEBM(eqx.Module):
    nn: EBMNet
    ctx: Contexts
    def __init__(self, key, num_train_samples, y_mean=0.0, context_dim=1):
        k1, k2 = jax.random.split(key)
        self.nn = EBMNet(k1, context_dim, y_mean, 
                         order_y=CONFIG["taylor_order_ebm_y"], 
                         radius=CONFIG["taylor_radius"])
        self.ctx = Contexts(k2, num_train_samples, context_dim, CONFIG["context_init"])
    def __call__(self, x, c, key=None):
        return self.nn(x, c, key)
    def predict(self, x):
        return self.nn.predict(x)

class TaylorContextEBM(eqx.Module):
    nn: TaylorEBMNet
    ctx: Contexts
    def __init__(self, key, num_train_samples, y_mean=0.0, context_dim=1):
        k1, k2 = jax.random.split(key)
        self.nn = TaylorEBMNet(k1, context_dim, y_mean, 
                               radius=CONFIG["taylor_radius"], 
                               order_x=CONFIG["taylor_order_ebm_x"],
                               order_c=CONFIG["taylor_order_ebm_c"],
                               order_y=CONFIG["taylor_order_ebm_y"])
        self.ctx = Contexts(k2, num_train_samples, context_dim, CONFIG["context_init"])
    def __call__(self, x, c, key=None): 
        return self.nn(x, c, key)
    def predict(self, x):
        return self.nn.predict(x)

#%%
# --- 5. TRAINING LOOP ---

if TRAIN:
    run_dir = get_run_path()
    print(f"🚀 Starting Training. Run ID: {run_dir.name}")
    trained_models = [] 
    
    key = jax.random.PRNGKey(CONFIG["seed"])
    y_mean = jnp.mean(Y_train)
    num_train = len(X_train)
    c_dim = CONFIG["context_dim"]
    t_radius = CONFIG["taylor_radius"]
    
    # Orders from config
    t_order_mlp = CONFIG["taylor_order_mlp"]
    t_order_ebm_x = CONFIG["taylor_order_ebm_x"]
    t_order_ebm_c = CONFIG["taylor_order_ebm_c"]
    t_order_ebm_y = CONFIG["taylor_order_ebm_y"]

    models_config = [
        ("Linear", LinearModel(key)),
        ("MLP", MLPModel(key)),
        ("TaylorMLP", TaylorMLP(key, radius=t_radius, order=t_order_mlp)),
        ("BaseEBM", BaseEBM(key, y_mean, order_y=0, radius=0)),
        ("TaylorBaseEBM", TaylorBaseEBM(key, y_mean, radius=t_radius, order=t_order_ebm_x, order_y=t_order_ebm_y)),
        ("ContextEBM", ContextEBM(key, num_train, y_mean, c_dim)),
        ("TaylorContextEBM", TaylorContextEBM(key, num_train, y_mean, c_dim))
    ]

    for name, model in models_config:
        print(f"\n=== Training {name} ===")
        start_t = time.time()
        
        is_pam_model = name in ["ContextEBM", "TaylorContextEBM"]
        
        if not is_pam_model:
            # === STANDARD TRAINING LOOP ===
            optimizer = optax.adabelief(CONFIG["lr_nn"]) 
            opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

            @eqx.filter_value_and_grad(has_aux=True)
            def compute_loss(model, x, y, key):
                pred, _ = model(x, key=key) 
                loss = jnp.mean((pred - y) ** 2)
                return loss, None

            @eqx.filter_jit
            def make_step(model, opt_state, x, y, key):
                (loss, _), grads = compute_loss(model, x, y, key)
                updates, opt_state = optimizer.update(grads, opt_state, model)
                model = eqx.apply_updates(model, updates)
                return model, opt_state, loss

            loss_history = []
            train_key = jax.random.PRNGKey(CONFIG["seed"])
            
            for epoch in range(CONFIG["epochs"]):
                batch_losses = []
                for bx, by, _ in dm.get_iterator():
                    train_key, step_key = jax.random.split(train_key)
                    model, opt_state, loss = make_step(model, opt_state, bx, by, step_key)
                    batch_losses.append(loss)
                loss_history.append(np.mean(batch_losses))
                
                if (epoch+1) % 250 == 0:
                    print(f"Ep {epoch+1} | Loss: {loss_history[-1]:.4f}")
            
            history_dict = {"train_loss": loss_history}
            save_run(run_dir, model, name, CONFIG, history_dict)
            trained_models.append((name, model, history_dict))

        else:
            # === PROXIMAL ALTERNATING MINIMIZATION (PAM) LOOP ===
            
            opt_nn = optax.adam(CONFIG["lr_nn"])
            opt_state_nn = opt_nn.init(eqx.filter(model.nn, eqx.is_inexact_array))
            
            opt_ctx = optax.adam(CONFIG["lr_ctx"])
            opt_state_ctx = opt_ctx.init(eqx.filter(model.ctx, eqx.is_inexact_array))

            proximal_reg = CONFIG["pam_proximal_reg"]
            
            # --- STEP 1: Update Neural Net (Fix Contexts) ---
            @eqx.filter_jit
            def train_step_nn(nn, nn_old, ctx, opt_state, x, y, indices, key):
                
                def prox_loss_nn(nn_params, ctx_static):
                    c_batch = ctx_static(indices)
                    # Pass key for stochastic Taylor expansion
                    y_pred = nn_params(x, c_batch, key=key) 
                    task_loss = jnp.mean((y_pred - y) ** 2)
                    
                    diff_norm = params_diff_norm_squared(nn_params, nn_old)
                    return task_loss + proximal_reg * diff_norm / 2.0, (task_loss, diff_norm)

                (loss, (task_loss, diff_norm)), grads = eqx.filter_value_and_grad(prox_loss_nn, has_aux=True)(nn, ctx)
                
                updates, opt_state = opt_nn.update(grads, opt_state, nn)
                nn = eqx.apply_updates(nn, updates)
                return nn, opt_state, task_loss, diff_norm

            # --- STEP 2: Update Contexts (Fix Neural Net) ---
            @eqx.filter_jit
            def train_step_ctx(ctx, ctx_old, nn, opt_state, x, y, indices, key):
                
                def prox_loss_ctx(ctx_params, nn_static):
                    c_batch = ctx_params(indices)
                    y_pred = nn_static(x, c_batch, key=key)
                    task_loss = jnp.mean((y_pred - y) ** 2)
                    
                    diff_norm = params_diff_norm_squared(ctx_params, ctx_old)
                    return task_loss + proximal_reg * diff_norm / 2.0, (task_loss, diff_norm)

                (loss, (task_loss, diff_norm)), grads = eqx.filter_value_and_grad(prox_loss_ctx, has_aux=True)(ctx, nn)
                
                updates, opt_state = opt_ctx.update(grads, opt_state, ctx)
                ctx = eqx.apply_updates(ctx, updates)
                return ctx, opt_state, task_loss, diff_norm

            losses_nn_history = []
            losses_ctx_history = []
            pam_key = jax.random.PRNGKey(CONFIG["seed"])
            
            # PAM Outer Loop
            for out_step in range(CONFIG["pam_outer_steps_max"]):
                
                nn_old = jax.tree_util.tree_map(lambda x: x, model.nn)
                ctx_old = jax.tree_util.tree_map(lambda x: x, model.ctx)

                nn_curr = model.nn
                
                # 1. Optimize Neural Net
                for in_step_nn in range(CONFIG["pam_inner_steps_nn"]):
                    batch_losses = []
                    for bx, by, b_idx in dm.get_iterator():
                        pam_key, step_key = jax.random.split(pam_key)
                        nn_curr, opt_state_nn, loss, diff = train_step_nn(
                            nn_curr, nn_old, model.ctx, opt_state_nn, bx, by, b_idx, step_key
                        )
                        batch_losses.append(loss)
                    
                    losses_nn_history.append(np.mean(batch_losses))

                model = eqx.tree_at(lambda m: m.nn, model, nn_curr)

                ctx_curr = model.ctx

                # 2. Optimize Contexts
                for in_step_ctx in range(CONFIG["pam_inner_steps_ctx"]):
                    batch_losses_ctx = []
                    for bx, by, b_idx in dm.get_iterator():
                        pam_key, step_key = jax.random.split(pam_key)
                        ctx_curr, opt_state_ctx, loss, diff = train_step_ctx(
                            ctx_curr, ctx_old, model.nn, opt_state_ctx, bx, by, b_idx, step_key
                        )
                        batch_losses_ctx.append(loss)
                    
                    losses_ctx_history.append(np.mean(batch_losses_ctx))
                
                model = eqx.tree_at(lambda m: m.ctx, model, ctx_curr)

                if (out_step+1) % CONFIG["print_every"] == 0:
                    print(f"PAM Step {out_step+1} | LossNN: {losses_nn_history[-1]:.4f} | LossCtx: {losses_ctx_history[-1]:.4f}")

            print(f"✅ {name} finished")
            history_dict = {"train_loss": losses_nn_history, "ctx_loss": losses_ctx_history}
            save_run(run_dir, model, name, CONFIG, history_dict)
            trained_models.append((name, model, history_dict))

else:
    if RUN_DIR == "": raise ValueError("Provide RUN_DIR")
    run_dir = Path(RUN_DIR)
    trained_models = []

#%%
# --- 6. ANALYSIS & PLOTTING ---
colors = ['crimson', 'black', 'green', 'blue', 'orange', 'red', 'purple', 'brown']

# 1. Loss Curves
plt.figure(figsize=(10, 5))
for i, (name, _, history) in enumerate(trained_models):
    loss_data = history['train_loss']
    plt.plot(loss_data, label=f"{name} (NN)", linewidth=1.5, color=colors[i % len(colors)])
    if 'ctx_loss' in history:
        plt.plot(history['ctx_loss'], label=f"{name} (Ctx)", linewidth=1.5, linestyle="--", color=colors[i % len(colors)])

plt.yscale('log')
plt.title("Training Loss")
plt.legend()
plt.savefig(run_dir / "loss.png")
plt.show()

# 2. Predictions
plt.figure(figsize=(12, 8))

# Data Background
for seg in np.unique(train_segs):
    mask = train_segs == seg
    plt.scatter(train_data[mask, 0], train_data[mask, 1], c='lightblue', s=20, alpha=0.5)
for seg in np.unique(test_segs):
    mask = test_segs == seg
    plt.scatter(test_data[mask, 0], test_data[mask, 1], c='wheat', s=20, alpha=0.5)

x_grid = jnp.linspace(-1.5, 1.5, 300)[:, None]

for i, (name, model, _) in enumerate(trained_models):
    y_grid = model.predict(x_grid)
    y_test = model.predict(X_test)
    mse = jnp.mean((y_test - Y_test)**2)
    plt.plot(x_grid, y_grid, color=colors[i % len(colors)], linewidth=2.5, label=f"{name} ({mse:.2e})")

plt.legend()
plt.title("Model Predictions (Test MSE)")
plt.savefig(run_dir / "predictions.png")
plt.show()

#%%
# 3. Segmented Plot For the Embeddings
## For the context-based models only
context_models = [m for m in trained_models if hasattr(m[1], 'ctx')]

if len(context_models) > 0:
    plt.figure(figsize=(12, 8))
    for i, (name, model, _) in enumerate(context_models):
        c_embeddings = model.ctx.params
        
        plt.subplot(1, len(context_models), i + 1)
        for seg in np.unique(train_segs):
            mask = train_segs == seg
            plt.scatter(X_train[mask, 0], c_embeddings[mask, 0], 
                        color=plt.cm.Blues(0.5 + seg*0.1), alpha=0.6, label=f"Train Seg {int(seg)}", s=10)
            plt.xlabel("X")
            plt.ylabel("C1")

        plt.title(f"{name} Context Embeddings")
        plt.legend()
    plt.savefig(run_dir / "context_embeddings.png")
    plt.show()


#%% Plot the x-y energy landscape for the Base EBM model, as a 3D plot
base_ebm_models = [m for m in trained_models if m[0] in ["BaseEBM", "TaylorBaseEBM"]]

if len(base_ebm_models) > 0:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    x_max = 0.8
    y_max = 0.8

    x_vals = jnp.linspace(-x_max, x_max, 200)
    y_vals = jnp.linspace(-y_max, y_max, 200)
    X_grid, Y_grid = jnp.meshgrid(x_vals, y_vals)
    grid_points = jnp.stack([X_grid.ravel(), Y_grid.ravel()], axis=-1)

    for name, model, _ in base_ebm_models:
        Z_grid = jax.vmap(lambda xy: model.energy(xy[0:1], xy[1:2]))(grid_points)
        Z_grid = Z_grid.reshape(X_grid.shape)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='coolwarm', alpha=0.8)
        ax.set_title(f"Energy Landscape of {name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Energy")
        plt.savefig(run_dir / f"{name}_energy_landscape.png")
        plt.show()

#%% Plot the x-y energy landscape for the Base EBM model, as a 2D colormap
base_ebm_models = [m for m in trained_models if m[0] in ["BaseEBM", "TaylorBaseEBM"]]

if len(base_ebm_models) > 0:
    x_max = 0.8
    y_max = 0.8

    x_vals = jnp.linspace(-x_max, x_max, 200)
    y_vals = jnp.linspace(-y_max, y_max, 200)
    X_grid, Y_grid = jnp.meshgrid(x_vals, y_vals)
    grid_points = jnp.stack([X_grid.ravel(), Y_grid.ravel()], axis=-1)

    for name, model, _ in base_ebm_models:
        Z_grid = jax.vmap(lambda xy: model.energy(xy[0:1], xy[1:2]))(grid_points)
        Z_grid = Z_grid.reshape(X_grid.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(X_grid, Y_grid, Z_grid, levels=50, cmap='viridis')
        plt.colorbar(label='Energy')
        plt.title(f"Energy Landscape of {name}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(run_dir / f"{name}_energy_landscape_2D.png")
        plt.show()
