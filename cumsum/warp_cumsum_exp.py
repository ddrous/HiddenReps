#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax

# jax.config.update("jax_debug_nans", True)
## Disable JIT as well
# jax.config.update("jax_disable_jit", True)

## Diable python warnings
import warnings
warnings.filterwarnings("ignore")

import jax.numpy as jnp
import equinox as eqx
import optax
# import torch
# from torch.utils import data
import math
import time
from jax.flatten_util import ravel_pytree

## Print JAX info
jax.print_environment_info()

## JAX debug NaNs


# --- Configuration ---
CONFIG = {
    "seed": 2026,
    "n_epochs": 1000,
    "print_every": 100,
    "num_seqs": 12000,
    "batch_size": 512,
    "seq_len": 32,
    "seq_dim": 8,
    "cum_sum": True,
    "ar_train_mode": False,
    "n_refine_steps": 10,
    "d_model": 128*1,
    "lr": 1e-5
}

# Set seeds
key = jax.random.PRNGKey(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
# torch.manual_seed(CONFIG["seed"])

sns.set(style="white", context="talk")

#---- Generate Cumulative Sum Data ---
def generate_cumsum_data(num_seqs, seq_len, seq_dim):
    # np.random.seed(seed)

    ## For each traj, we want y_t = w x_t. First we sample x_t
    # xs = np.random.randn(num_traj, seq_len, seq_dim-1)
    # xs = np.random.normal(size=(num_seqs, seq_len, seq_dim-1), scale=1)
    xs = np.random.uniform(low=-1, high=1, size=(num_seqs, seq_len, seq_dim-1))

    ## Sample a linear transformation vector w
    # w = np.random.randn(num_seqs, seq_dim-1)
    w = np.random.uniform(low=-1, high=1, size=(num_seqs, seq_dim-1))
    
    ## Compute y_t = w x_t (dot product along last dim)
    ys = np.einsum('ijk,ik->ij', xs, w)
    
    ## Now we concatenate xs and ys to have shape (num_traj, seq_len, seq_dim)
    data = np.concatenate([xs, ys[..., None]], axis=-1)

    ## Save data as numpy arrays, proportion 0.8 train, 0.2 test
    split_idx = int(num_seqs * 0.8)

    return data[:split_idx], data[split_idx:]

train_data, test_data = generate_cumsum_data(num_seqs=CONFIG["num_seqs"],
                                            seq_len=CONFIG["seq_len"], 
                                            seq_dim=CONFIG["seq_dim"])

print("Data shape:", train_data.shape)
# Plot a few trajectories
for i in range(1):
    plt.plot(train_data[i, :, 0], label=f'Traj {i} (x1)')
    plt.plot(train_data[i, :, -1], label=f'Traj {i} (y)')
    # plt.plot(data[i, :, 1], label=f'Traj {i} (x2)')

    ## DO a cumsum before plotting
    plt.plot(np.cumsum(train_data[i, :, 0]), "--", label=f'Traj {i} (cumsum x1)')
    plt.plot(np.cumsum(train_data[i, :, -1]), "--", label=f'Traj {i} (cumsum y)')

plt.legend()
plt.title("Sample Trajectorie (Cum Sum Data)")
plt.show()



#%%

# Using dummy data for the example (List of arrays)
# train_data = np.random.randn(10, 5)

# 2. Transformation Logic
def process_sample(sample):
    """
    sample: numpy array of shape (seq_len, seq_dim)
    Returns: dict with 'x' and 'y', each of shape (seq_len, seq_dim)
    """
    if CONFIG["cum_sum"]:
        sample = np.cumsum(sample, axis=0)
    else:
        sample = sample.copy()   # avoid modifying original
    
    # Target: the last feature column (y_t) for all time steps
    y = sample[:, -1:].copy()    # shape (seq_len, 1)
    
    # Input: mask the last value of the target column
    x = sample.copy()
    x[-1, -1] = 0.0              # zero out the final target

    return {"x": x, "y": y}



#%%
# --- 2. WARP Model (Weight-space Adaptive Recurrent Prediction) ---

class RootMLP(eqx.Module):
    """
    The auxiliary 'root' network that gets updated at every time step.
    It maps Input -> Scalar Output.
    """
    layers: list

    def __init__(self, in_size, out_size, width, depth, key):
        keys = jax.random.split(key, depth + 1)
        self.layers = []
        
        # Input layer
        self.layers.append(eqx.nn.Linear(in_size, width, key=keys[0]))
        
        # Hidden layers
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i+1]))
            
        # Output layer
        self.layers.append(eqx.nn.Linear(width, out_size, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

class WARP(eqx.Module):
    # Learnable parameters for the weight-space recurrence
    A: jax.Array  # State transition (Weights -> Weights)
    B: jax.Array  # Input transition (Data -> Weights)
    
    # Hypernetwork to initialize theta_0
    hypernet: eqx.nn.MLP
    
    # Static definition of the root network structure (to use for unraveling)
    root_structure: RootMLP = eqx.field(static=True)
    unravel_fn: callable = eqx.field(static=True)
    
    # Configuration
    d_theta: int = eqx.field(static=True)
    use_tau: bool = eqx.field(static=True)
    seq_len: int = eqx.field(static=True)

    def __init__(self, input_dim, seq_len, root_width, root_depth, use_tau, key):
        k_root, k_hyper, k_A, k_B = jax.random.split(key, 4)
        self.use_tau = use_tau
        self.seq_len = seq_len

        # 1. Define the Root Network Structure
        # If use_tau is True, input dim increases by 1
        root_input_dim = input_dim + 1 if use_tau else input_dim
        
        # Initialize a template root network to get its parameter size and structure
        template_root = RootMLP(root_input_dim, 1, root_width, root_depth, k_root)
        
        # Extract flat parameter vector size using JAX's flatten_util
        flat_params, self.unravel_fn = ravel_pytree(template_root)
        self.d_theta = flat_params.shape[0]
        self.root_structure = template_root # Store for structure reference if needed
        
        print(f"WARP Initialized: Root Network has {self.d_theta} parameters (State Size).")

        # 2. Initialize WARP Dynamics (A, B)
        # A: Identity initialization (as per paper Section 2.2) to emulate gradient descent/residual flow
        self.A = jnp.eye(self.d_theta)
        
        # B: Zero initialization (as per paper Section 2.2)
        # Maps input difference (dim=input_dim) to weight space (dim=d_theta)
        self.B = jnp.zeros((self.d_theta, input_dim))

        # 3. Initialize Hypernetwork (phi)
        # Maps initial input x_0 -> initial weights theta_0
        # Paper suggests gradually increasing width, here using standard MLP for simplicity
        self.hypernet = eqx.nn.MLP(
            in_size=input_dim,
            out_size=self.d_theta,
            width_size=128,
            depth=2,
            activation=jax.nn.relu,
            key=k_hyper
        )

    def __call__(self, input_seq, key=None):
        """
        input_seq: (Seq_Len, Input_Dim)
        Returns: (Seq_Len, 1) - The predicted y values
        """
        
        # 1. Precompute Input Differences (Delta x)
        # Delta x_t = x_t - x_{t-1}
        # For t=0, we can define Delta x_0 = 0 or learnable. 
        # The paper uses a hypernet for theta_0 based on x_0, so recurrence starts effectively at t=1 updates.
        
        # Helper to get x_{t} and x_{t-1}. 
        # We pad the beginning to compute diffs easily.
        x_pad = jnp.concatenate([input_seq[:1], input_seq[:-1]], axis=0)
        delta_x = input_seq - x_pad # Shape (L, D_x)
        
        # 2. Initialize Weight State theta_0
        # theta_0 = phi(x_0)
        theta_0 = self.hypernet(input_seq[0])
        
        # 3. Recurrence Scan: theta_t = A * theta_{t-1} + B * Delta x_t
        def step(theta_prev, dx_t):
            # Eq (1) from paper
            theta_next = self.A @ theta_prev + self.B @ dx_t
            return theta_next, theta_next

        # Scan over sequence
        _, theta_seq = jax.lax.scan(step, theta_0, delta_x)
        
        # Prepend theta_0 to align time steps (theta_seq contains theta_1 ... theta_T)
        # If we strictly follow paper, theta_t is used to predict y_t.
        # However, theta_0 is the state *before* seeing delta_x_1? 
        # The paper says: theta_t = A theta_{t-1} + B Delta x_t. y_t = MLP_{theta_t}.
        # This implies theta_0 is used for y_0, theta_1 (updated with dx1) used for y_1.
        # But scan returns theta_1...theta_T. We need to construct the full sequence [theta_0, theta_1, ... theta_{T-1}].
        
        # Shift: The state used for time t is result of update at time t (current input diff).
        # Actually, usually theta_0 is prepared from x_0, and immediately used to predict y_0?
        # Let's assume the scan output theta_seq corresponds to indices 0..T-1 effectively if we treat the update as happening 'instantaneously' or if theta_0 is the starting state.
        # Let's stick to: We have theta_0. The scan produces theta_1 to theta_{L}.
        # We want outputs for 0 to L-1.
        # So we use [theta_0, theta_1, ... theta_{L-1}].
        
        all_thetas = jnp.concatenate([theta_0[None, :], theta_seq[:-1]], axis=0)
        
        # 4. Decode (Apply Root Network)
        # y_t = Root(x_t; weights=theta_t)
        
        # Generate normalized time tau if needed
        L = input_seq.shape[0]
        taus = jnp.linspace(0, 1, L)[:, None] # (L, 1)

        def apply_root_at_t(theta, x, tau):
            # Reconstruct the root network from the flat weight vector
            root = self.unravel_fn(theta)
            
            # Prepare input
            if self.use_tau:
                root_in = jnp.concatenate([x, tau], axis=-1)
            else:
                root_in = x
                
            return root(root_in)

        # Vectorize the application of the root network over the sequence
        ys = jax.vmap(apply_root_at_t)(all_thetas, input_seq, taus)
        
        return ys

    def predict(self, input_seq, key=None):
        """ Predict and remove the cumsum on the prediction if needed """
        output_seq = self(input_seq, key)

        if CONFIG["cum_sum"]:
            diff_seq = output_seq[1:, :] - output_seq[:-1, :]
            first_val = output_seq[:1, :]
            return jnp.concatenate([first_val, diff_seq], axis=0)
        else:
            return output_seq


#%%
# --- 4. Training Setup & Comparison ---

# Initialize Models
key, k1 = jax.random.split(key, 2)

# Replacing S4 with WARP
# Note: Root width/depth controls d_theta. 
# Width 16, Depth 1 -> ~50-100 params. Matrix A ~ 10k params. Very fast.
# Width 32, Depth 2 -> ~1000 params. Matrix A ~ 1M params. Slower but more expressive.
models = {
    "WARP": WARP(
        input_dim=CONFIG["seq_dim"], 
        seq_len=CONFIG["seq_len"],
        root_width=32,       # Keep small for efficiency in this demo
        root_depth=2,        # Depth of the meta-learned network
        use_tau=True,        # Optional flag
        key=k1
    )
}

# Train Step
@eqx.filter_jit
def train_step(model, optimizer, opt_state, batch, key):
    Xs, Ys = batch["x"], batch["y"]
    
    def loss_fn(m, x, y, k):
        # Model forward
        # WARP expects (Batch, Seq, Dim) -> vmap over batch
        # y_hat = jax.vmap(m)(x, key=k)
        y_hat = jax.vmap(m, in_axes=(0))(x)

        indices = jnp.arange(x.shape[1])
        # indices = jnp.array([-1])

        return jnp.mean((y_hat[:, indices] - y[:, indices]) ** 2)

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, Xs, Ys, key)

    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss_val

import torch
from torch.utils.data import Dataset, DataLoader

def numpy_collate(batch):
    """Collate function to convert PyTorch tensors to numpy/JAX arrays"""
    if isinstance(batch[0], dict):
        return {key: np.stack([b[key] for b in batch]) for key in batch[0]}
    return np.stack(batch)

def train(models, train_data, key):

    # Create a data source (no batching/shuffling here)
    class ICLRegressionDataset(Dataset):
        def __init__(self, data):
            self._data = data.astype(np.float32)        
        def __len__(self):
            return len(self._data)
        def __getitem__(self, index):
            return process_sample(self._data[index])

    train_dataset = ICLRegressionDataset(train_data)
    # Create PyTorch DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0, # JAX/Torch multiprocessing conflict avoidance
        collate_fn=numpy_collate,
        drop_last=False,
    )

    batch = next(iter(train_loader))
    print(f"Example X shape: {batch['x'].shape}, type: {batch['x'].dtype}")

    # Optimizers (One per model)
    optimizers = {name: optax.adamw(CONFIG["lr"]) for name in models}
    opt_states = {name: opt.init(eqx.filter(model, eqx.is_inexact_array)) 
                for name, (model, opt) in zip(models.keys(), zip(models.values(), optimizers.values()))}

    # Training Loop
    loss_history = {name: [] for name in models}

    print(f"Training 1 models: {list(models.keys())}")

    for epoch in range(CONFIG["n_epochs"]):
        batch_losses = {name: [] for name in models}

        # for _ in range(train_loader.steps_per_epoch):
        #     batch, state, mask = iterate(state)

        for batch in train_loader:
            key, subkey = jax.random.split(key)
            
            batch = jax.device_put(batch)

            for name in models:
                models[name], opt_states[name], loss = train_step(
                    models[name], optimizers[name], opt_states[name], batch, subkey
                )
                batch_losses[name].append(loss)

                # print(f"    Loss for batch: {name}: {loss:.5f}", flush=True)

        for name in models:
            avg = np.mean(batch_losses[name])
            loss_history[name].append(avg)

        # if (epoch+1) % CONFIG["print_every"] == 0:
            # print(f"Epoch {epoch+1}: " + ", ".join([f"{n}: {l:.5f}" for n, l in zip(loss_history.keys(), [h[-1] for h in loss_history.values()])]), flush=True)
        ## Print current time HH:MM:SS as well
        if epoch==0 or (epoch+1) % CONFIG["print_every"] == 0:
            print(f"{time.strftime('%H:%M:%S')} - Epoch {epoch+1}: " + ", ".join([f"{n}: {h[-1]:.5f}" for n, h in loss_history.items()]), flush=True)

    colors = ['r', 'b', 'g']
    plt.figure(figsize=(10, 4))
    for name, losses in loss_history.items():
        plt.plot(losses, label=name, color=colors.pop(0))
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale('log')
    # plt.title("Training Loss Comparison")
    plt.legend()
    plt.show()

    return models, loss_history


#%%
def test(models, test_data):
    # Plot Training Curves

    # 1. Define the vectorized version of the function
    vectorized_process = np.vectorize(
        process_sample, 
        signature='(m,n)->()' # Tells numpy to treat (seq_len, seq_dim) as one core unit
    )

    # 2. Apply it to the data
    # Since process_sample returns a dict, numpy will create an object array of dicts
    batch_array = vectorized_process(test_data)

    # 3. Restructure the list of dicts into a dict of arrays (what JAX needs)
    # np.vectorize returns an array of dictionaries, which JAX cannot put on device directly.
    batch = {
        "x": jnp.stack([d["x"] for d in batch_array]),
        "y": jnp.stack([d["y"] for d in batch_array])
    }

    # Now it is a single JAX-compatible object
    batch = jax.device_put(batch)

    def compute_metrics(model, key):
        # Standard evaluation params
        # test_data of shape (num_traj, seq_len, seq_dim)
        # test_data = jnp.asarray(test_data)
        # batch = next(iter(test_loader))

        Xs, Ys = batch["x"], batch["y"]
        
        # WARP requires vmap over batch dimension
        Ys_hat = jax.vmap(model)(Xs)

        last_Y = Ys[:, -1, :]  # Shape (batch_size, 1)
        last_Y_hat = Ys_hat[:, -1, :]  # Shape (batch_size, 1)

        # JAX Metrics
        mse = jnp.mean((last_Y_hat - last_Y) ** 2)
        mae = jnp.mean(jnp.abs(last_Y_hat - last_Y))
        
        return mse, mae

    print(f"{'Model':<25} | {'MSE':<10} | {'MAE':<10} | {'Wass.':<10} | {'FD':<10}")
    print("-" * 75)

    metrics_results = {}
    for name, model in models.items():
        mse, mae = compute_metrics(model = model, key=key)
        metrics_results[name] = (mse, mae)
        print(f"{name:<25} | {mse:.5f}    | {mae:.5f}     ")

    for name, model in models.items():
        Xs, Ys = batch["x"], batch["y"]

        if CONFIG["cum_sum"]:
            ## Revert Ys back to normal values for plotting
            Ys_diff = Ys[:, 1:, :] - Ys[:, :-1, :]
            first_val = Ys[:, :1, :]
            Ys = jnp.concatenate([first_val, Ys_diff], axis=1)

        # We need to manually vectorise the predict method since it's an instance method
        # Ys_hat = jax.vmap(model)(Xs)
        Ys_hat = jax.vmap(model.predict)(Xs)

        last_Y = Ys[:, -1, :]       # Shape (batch_size, 1)
        last_Y_hat = Ys_hat[:, -1, :]

        plt.figure(figsize=(6, 6))
        plt.scatter(last_Y, last_Y_hat, alpha=0.5)
        plt.xlabel("True Last Y")
        plt.ylabel("Predicted Last Y")
        plt.title(f"{name} - Last Y Prediction")
        plt.plot([last_Y.min(), last_Y.max()], [last_Y.min(), last_Y.max()], 'k--')  # Diagonal line
        plt.show()

        # ## PLot x (1D) agianst pred y (1D) for all steps in a random sequence
        # seq_idx = np.random.randint(0, Xs.shape[0])
        # x = Xs[seq_idx, :, 0]  # Shape (seq_len,)
        # ## reorder x
        # x_ids = np.argsort(x)
        # x_sorted = x[x_ids]
        # y_sorted = Ys_hat[seq_idx, :, 0][x_ids]
        # plt.figure(figsize=(6, 6))
        # plt.scatter(x_sorted, y_sorted, alpha=0.5)
        # plt.xlabel("Input X (sorted)")
        # plt.ylabel("Predicted Y")
        # plt.title(f"{name} - y_t vs x_t for seq {seq_idx}")
        # plt.show()




#%%
if __name__ == "__main__":
    models, loss_history = train(models, train_data, key)
    test(models, test_data)