#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax

import warnings
warnings.filterwarnings("ignore")

import jax.numpy as jnp
import equinox as eqx
import optax
import time
from jax.flatten_util import ravel_pytree
from torch.utils.data import Dataset, DataLoader

jax.print_environment_info()

# --- Configuration ---
CONFIG = {
    "seed": 2026,
    "n_epochs": 50,
    "print_every": 10,
    "num_seqs": 12000,
    "batch_size": 512,
    "seq_len": 32,
    "seq_dim": 8,
    "root_width": 32,
    "root_depth": 5,
    "lr": 1e-5, # Increased slightly for faster convergence with static theta_0
}

key = jax.random.PRNGKey(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
sns.set(style="white", context="talk")

#---- Generate ICL Cumulative Sum Data ---
def generate_icl_data(num_seqs, seq_len, x_dim):
    """ Matches the logic of ICLDataset in loaders.py """
    # Generate Xs uniformly
    Xs = np.random.uniform(low=-1, high=1, size=(num_seqs, seq_len, x_dim)).astype(np.float32)
    
    # Generate output weights uniformly
    output_matrices = np.random.uniform(low=-1, high=1, size=(num_seqs, x_dim)).astype(np.float32)
    
    # ys = Xs @ output_matrix
    ys = np.einsum('nsi,ni->ns', Xs, output_matrices)
    
    ys_zero = ys.copy()
    ys_zero[:, -1] = 0.0 # Set the last output to 0
    
    # Cumulative sum
    ys = np.cumsum(ys, axis=-1)
    ys_zero = np.cumsum(ys_zero, axis=-1)
    # Xs = np.cumsum(Xs, axis=-1)
    Xs = np.cumsum(Xs, axis=1)
    
    # Concatenate inputs and zeroed targets
    inputs = np.concatenate((Xs, ys_zero[..., None]), axis=-1)
    
    # Target is strictly the 1D y
    outputs_1D = ys[..., None]
    
    # Create normalized time array
    t_eval = np.linspace(0., 1., seq_len, dtype=np.float32)[:, None]
    t_eval = np.repeat(t_eval[None, ...], num_seqs, axis=0)

    split_idx = int(num_seqs * 0.8)
    
    train_data = {"x": inputs[:split_idx], "t": t_eval[:split_idx], "y": outputs_1D[:split_idx]}
    test_data = {"x": inputs[split_idx:], "t": t_eval[split_idx:], "y": outputs_1D[split_idx:]}
    
    return train_data, test_data

train_data, test_data = generate_icl_data(CONFIG["num_seqs"], CONFIG["seq_len"], CONFIG["seq_dim"])
print("Data shape (X):", train_data["x"].shape)

#%%
# --- WARP Model ---

class RootMLP(eqx.Module):
    layers: list

    def __init__(self, in_size, out_size, width, depth, key):
        keys = jax.random.split(key, depth + 1)
        self.layers = []
        self.layers.append(eqx.nn.Linear(in_size, width, key=keys[0]))
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i+1]))
        self.layers.append(eqx.nn.Linear(width, out_size, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

class WARP(eqx.Module):
    A: jax.Array
    B: jax.Array
    theta_0: jax.Array # Static learned initialization
    
    root_structure: RootMLP = eqx.field(static=True)
    unravel_fn: callable = eqx.field(static=True)
    
    d_theta: int = eqx.field(static=True)

    hypernet: any

    def __init__(self, data_size, root_width, root_depth, key):
        k_root, k_A, k_B, k_theta = jax.random.split(key, 4)
        
        # Root takes (time + input_data)
        root_input_dim = 1 + (data_size + 1)
        output_dim = 1 
        
        template_root = RootMLP(root_input_dim, output_dim, root_width, root_depth, k_root)
        flat_params, self.unravel_fn = ravel_pytree(template_root)
        self.d_theta = flat_params.shape[0]
        self.root_structure = template_root 
        
        # Initialization matching models.py
        self.theta_0 = flat_params # Initialize with the random weights

        ## Define a hypernetwork-like initialization for theta_0 to encourage better starting performance
        # self.hypernet = eqx.nn.Linear(root_input_dim, self.d_theta, key=k_theta)


        self.A = jnp.eye(self.d_theta)
        
        B_out_dim = 1 + (data_size + 1) # time + data
        self.B = jnp.zeros((self.d_theta, B_out_dim))

    def __call__(self, xs, ts):
        def scan_step(carry, input_signal):
            thet, x_prev, t_prev = carry
            x_true, t_curr = input_signal

            x_t = jnp.concatenate([t_curr[:1], x_true], axis=-1)
            x_p = jnp.concatenate([t_prev[:1], x_prev], axis=-1)

            # x_t = x_true
            # x_p = x_prev

            # Core Recurrence
            thet_next = self.A @ thet + self.B @ (x_t - x_p)
            
            return (thet_next, x_true, t_curr), thet_next

        # Scan over sequence
        # theta_0 = self.hypernet(jnp.concatenate([ts[0][:1], xs[0]], axis=-1)) # Learn an initialization based on the first input 
        # initial_carry = (theta_0, xs[0], ts[0])

        initial_carry = (self.theta_0, xs[0], ts[0])
        _, theta_outs = jax.lax.scan(scan_step, initial_carry, (xs, ts))

        @eqx.filter_vmap
        def apply_theta(theta, t_curr, x_curr):
            root_fun = self.unravel_fn(theta)
            root_in = jnp.concatenate([t_curr, x_curr], axis=-1)
            return root_fun(root_in)

        # Decode sequence
        ys_hat = apply_theta(theta_outs, ts, xs)
        return ys_hat

#%%
# --- Training Setup ---

def numpy_collate(batch):
    return {key: np.stack([b[key] for b in batch]) for key in batch[0]}

class DictDataset(Dataset):
    def __init__(self, data):
        self.data = {k: v.astype(np.float32) for k, v in data.items()}
        self.length = len(list(data.values())[0])
    def __len__(self): return self.length
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

train_loader = DataLoader(DictDataset(train_data), batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=numpy_collate)
test_loader = DataLoader(DictDataset(test_data), batch_size=len(test_data["x"]), shuffle=False, collate_fn=numpy_collate)

key, k1 = jax.random.split(key, 2)
model = WARP(data_size=CONFIG["seq_dim"], root_width=CONFIG["root_width"], root_depth=CONFIG["root_depth"], key=k1)

optimizer = optax.adamw(CONFIG["lr"])
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

@eqx.filter_jit
def train_step(model, opt_state, batch):
    Xs, Ts, Ys = batch["x"], batch["t"], batch["y"]
    
    def loss_fn(m, x, t, y):
        y_hat = jax.vmap(m)(x, t)
        return jnp.mean((y_hat - y) ** 2)

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, Xs, Ts, Ys)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

# Training Loop
loss_history = []
print("Starting Training...")
for epoch in range(CONFIG["n_epochs"]):
    batch_losses = []
    for batch in train_loader:
        batch = jax.device_put(batch)
        model, opt_state, loss = train_step(model, opt_state, batch)
        batch_losses.append(loss)
    
    avg_loss = np.mean(batch_losses)
    loss_history.append(avg_loss)
    
    if epoch == 0 or (epoch + 1) % CONFIG["print_every"] == 0:
        print(f"{time.strftime('%H:%M:%S')} - Epoch {epoch+1}: MSE = {avg_loss:.5f}")

plt.figure(figsize=(8, 4))
plt.plot(loss_history, label="Train MSE", color='blue')
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

#%%
# --- Testing ---

test_batch = jax.device_put(next(iter(test_loader)))
Ys_hat = jax.vmap(model)(test_batch["x"], test_batch["t"])

last_Y = test_batch["y"][:, -1, 0]
last_Y_hat = Ys_hat[:, -1, 0]

mse = jnp.mean((last_Y_hat - last_Y) ** 2)
print(f"Final Test MSE on Query Target: {mse:.5f}")

plt.figure(figsize=(6, 6))
plt.scatter(last_Y, last_Y_hat, alpha=0.5, color='crimson')
plt.plot([last_Y.min(), last_Y.max()], [last_Y.min(), last_Y.max()], 'k--')
plt.xlabel("True Last Y")
plt.ylabel("Predicted Last Y")
plt.title("Query Point Predictions")
plt.show()