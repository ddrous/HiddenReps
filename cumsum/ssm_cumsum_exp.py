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

## Print JAX info
jax.print_environment_info()

## JAX debug NaNs


# --- Configuration ---
CONFIG = {
    "seed": 2026,
    "n_epochs": 2000,
    "print_every": 200,
    "num_seqs": 1200,
    "batch_size": 128,
    "seq_len": 32,
    "seq_dim": 8,
    "cum_sum": False,
    "ar_train_mode": False,
    "n_refine_steps": 10,
    "d_model": 128*1,
    "lr": 3e-5
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
    xs = np.random.normal(size=(num_seqs, seq_len, seq_dim-1), scale=1)
    # xs = np.random.uniform(low=-0.1, high=0.1, size=(num_traj, seq_len, seq_dim-1))

    ## Sample a linear transformation vector w
    w = np.random.randn(num_seqs, seq_dim-1)
    
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
# --- 2. S4 Model (State Space Sequence Model) ---

class S4Kernel(eqx.Module):
    """
    A minimal implementation of the S4D (Diagonal) kernel.
    Computes the convolution kernel K based on diagonal state matrix A.
    """
    log_dt: jax.Array
    A_real: jax.Array
    A_imag: jax.Array
    C: jax.Array
    
    def __init__(self, d_model, d_state, key):
        k_dt, k_A, k_C = jax.random.split(key, 3)
        
        # 1. Initialize dt (step size) - standard random log initialization
        # Drawn from U[0.001, 0.1] in log space
        log_dt_min, log_dt_max = math.log(0.001), math.log(0.1)
        self.log_dt = jax.random.uniform(k_dt, (d_model,), minval=log_dt_min, maxval=log_dt_max)
        
        # 2. Initialize A (Diagonal State Matrix)
        # Using HiPPO-LegS approximation for diagonal A: Real part -0.5, Imag part Fourier-like
        # A = -0.5 + i * n * pi
        self.A_real = -0.5 * jnp.ones((d_model, d_state))
        self.A_imag = jnp.repeat(jnp.arange(d_state)[None, :], d_model, axis=0) * math.pi
        
        # 3. Initialize C (Output projection)
        # Standard complex normal initialization
        self.C = jax.random.normal(k_C, (d_model, d_state, 2)) * 0.5 # stored as (real, imag)
    
    def __call__(self, L):
        """ Computes the convolution kernel of length L """
        # Materialize parameters
        dt = jnp.exp(self.log_dt)[:, None] # (H, 1)
        A = jax.lax.complex(self.A_real, self.A_imag) # (H, N)
        C = jax.lax.complex(self.C[..., 0], self.C[..., 1]) # (H, N)
        
        # Discretize A using Bilinear (Tustin) transform
        # A_bar = (I + dt*A/2) / (I - dt*A/2)  <-- Simplified diagonal update
        # However, for the kernel computation in frequency domain, we use the continuous form:
        # K_bar(z) = C (I - A_bar z^-1)^-1 B_bar
        # The closed form computation for S4D kernel over L steps:
        
        # 1. Compute Power of A terms (Vandermonde generator)
        # We need the roots of unity for the FFT
        omega = jnp.exp(-2j * jnp.pi * jnp.arange(L) / L) # (L,)
        
        # The transfer function at frequencies omega: H(z) = 2 * dt * C * (1 - Omega)^-1 ...
        # A more stable approach for S4D is computing the kernel via the Cauchy structure directly
        # K = 2 * dt * C * ( (1+omega) - (1-omega) A dt / 2 )^-1
        
        # Vectorized computation broadcasted over H (channels) and N (state)
        # z = (1 + omega) / (1 - omega) * (2 / dt)  <-- Bilinear mapping to continuous domain
        
        # Standard S4D Kernel (Frequency Domain):
        # K[k] = \sum_n ( C_n * 2*dt / (1 - A_n * dt * (1+omega)/(1-omega)) ) * ...
        # Let's use the explicit "Cauchy" calculation which is numerically safe
        
        z = jnp.exp(2j * jnp.pi * jnp.arange(L) / L) # Roots of unity
        
        # Bilinear transform of z back to continuous s-plane
        # s = 2/dt * (z-1)/(z+1)
        # But we can just compute the term: 2*dt / ( (1-z) - A*dt*(1+z)/2 ) * ...
        # Simplified: K_bar = 2 * dt * C / ( 1 - A_bar * z_inv ) is for recurrence.
        
        # Using the standard minimal kernel formula for S4D:
        # K_hat = 2 * C * (1 - A)^-1 B (approx)
        # Exact discrete time kernel for Diagonal A:
        # k = C * A_bar^t * B_bar
        
        # We compute this efficiently in frequency domain:
        # K_f = 2 * dt * C / ( (1 - A_real*dt) - z * (1 + A_real*dt) - i*(...) )
        # Let's do the rigorous calculation:
        
        dt_A = A * dt  # (H, N)
        
        # Denominator for bilinear transform
        # We want to evaluate at z = exp(i 2pi k / L)
        # Term = (2/dt) * (1-zinv)/(1+zinv) - A
        # Reshaped for broadcasting:
        # (1-z^-1)/(1+z^-1) = i * tan(pi k / L) ... or just compute raw
        
        z_inv = jnp.exp(-2j * jnp.pi * jnp.arange(L) / L) # (L,)
        
        # This term maps z-domain to s-domain
        # s = 2/dt * (1 - z^-1) / (1 + z^-1)
        # Denom = s - A
        
        # Trick to avoid division by zero at k=0 if A is purely imaginary (it's not, real part is -0.5)
        # H (batch), N (state), L (seq)
        
        # Expand dims
        z_inv = z_inv[None, None, :] # (1, 1, L)
        dt_exp = dt[:, :, None]      # (H, 1, 1)
        A_exp = A[:, :, None]        # (H, N, 1)
        
        # Bilinear transform s-vals
        # Handle the singularity at z=-1 (Nyquist) by using a small epsilon or logic, 
        # but standard complex float usually handles it okay-ish or we assume L is even.
        
        # A simpler way often used in S4D codebases:
        # K_hat = 2 * dt * C / ( 1 + dt*A - z_inv * (1 - dt*A) ) * B
        # Assume B = 1 for S4D (absorbed into C)
        
        denom = (1 - z_inv) - A_exp * (dt_exp / 2.0) * (1 + z_inv)
        # Actually standard bilinear discretization:
        # A_bar = (1 + A dt/2) / (1 - A dt/2)
        # B_bar = dt / (1 - A dt/2) * B
        # Transfer func H(z) = C (I - A_bar z^-1)^-1 B_bar
        
        # We calculate H(z) directly.
        # H(z) = C * B_bar / (1 - A_bar z^-1)
        
        A_bar = (1 + dt_A/2) / (1 - dt_A/2) # (H, N)
        B_bar = dt / (1 - dt_A/2)           # (H, N) assuming B=ones
        
        # H(z) = \sum C_n * B_bar_n / (1 - A_bar_n z^-1)
        # This is a sum of simple poles.
        
        C_exp = C[:, :, None]          # (H, N, 1)
        B_bar_exp = B_bar[:, :, None]  # (H, N, 1)
        A_bar_exp = A_bar[:, :, None]  # (H, N, 1)
        
        # Sum over state dimension N
        # Result shape (H, L)
        H_z = jnp.sum( C_exp * B_bar_exp / (1 - A_bar_exp * z_inv), axis=1 )
        
        return H_z # Frequency response

class S4Block(eqx.Module):
    layer_norm: eqx.nn.LayerNorm
    s4_kernel: S4Kernel
    out_proj: eqx.nn.Linear
    skip_proj: eqx.nn.Linear
    
    mlp: eqx.nn.MLP
    norm2: eqx.nn.LayerNorm
    
    d_model: int = eqx.field(static=True)
    
    def __init__(self, d_model, d_state, dropout, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.d_model = d_model
        
        self.layer_norm = eqx.nn.LayerNorm(d_model)
        self.s4_kernel = S4Kernel(d_model, d_state, k1)
        
        # O = S4(x)
        self.out_proj = eqx.nn.Linear(d_model, d_model, key=k2)
        self.skip_proj = eqx.nn.Linear(d_model, d_model, key=k3) # D connection
        
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.mlp = eqx.nn.MLP(
            in_size=d_model, out_size=d_model, width_size=d_model*4, depth=1, activation=jax.nn.gelu, key=k4
        )

    def __call__(self, x, key=None):
        # x: (L, H)
        L, H = x.shape
        
        shortcut = x
        x = jax.vmap(self.layer_norm)(x)
        
        # 1. S4 Convolution
        # Compute Kernel in freq domain
        K_f = self.s4_kernel(L) # (H, L)
        
        # FFT of input
        U_f = jnp.fft.rfft(x, axis=0) # (L//2+1, H)
        
        # We need to handle the FFT lengths matching
        # K_f is full length L, U_f is rfft length
        # Typically we use full FFT for convolution or adapt K
        # Let's use full FFT to be safe and clear
        U_f_full = jnp.fft.fft(x, axis=0) # (L, H)
        
        # Convolution y = k * u  <=> Y = K * U
        Y_f = K_f.T * U_f_full # (L, H) * (L, H) elementwise
        
        y = jnp.fft.ifft(Y_f, axis=0).real # (L, H)
        
        # 2. Skip (D) connection
        # Standard S4 has y = S4(x) + D*x
        # We model D as a linear mixing for simplicity or elementwise parameter
        y = y + jax.vmap(self.skip_proj)(x)
        
        # 3. Output projection + GLU (often used in S4, here simple Linear+GELU for compat)
        y = jax.nn.gelu(y)
        y = jax.vmap(self.out_proj)(y)
        
        # Residual
        x = shortcut + y
        
        # MLP Block
        shortcut = x
        x = jax.vmap(self.norm2)(x)
        x = jax.vmap(self.mlp)(x)
        x = shortcut + x
        
        return x

class S4Model(eqx.Module):
    embedding: eqx.nn.Linear
    blocks: list
    output_projection: eqx.nn.Linear

    d_model: int = eqx.field(static=True)

    def __init__(self, input_dim, d_model, n_layers, d_state, key):
        self.d_model = d_model
        
        keys = jax.random.split(key, 5)
        self.embedding = eqx.nn.Linear(input_dim, d_model, key=keys[0])
        
        # S4 generally doesn't need Positional Embeddings!
        
        self.blocks = [
            S4Block(d_model, d_state, dropout=0.0, key=k)
            for k in jax.random.split(keys[1], n_layers)
        ]
        
        self.output_projection = eqx.nn.Linear(d_model, 1, key=keys[2])
        # Zero Init Output for stability
        self.output_projection = eqx.tree_at(
            lambda l: (l.weight, l.bias), 
            self.output_projection, 
            (jnp.zeros_like(self.output_projection.weight), jnp.zeros_like(self.output_projection.bias))
        )

    def __call__(self, input_seqs, key=None):
        def single_forward(input_seq, key):
            # Input (L, dim)
            x = jax.vmap(self.embedding)(input_seq)
            
            # S4 Blocks
            for block in self.blocks:
                x = block(x, key=key)
            
            # Output Head
            y = jax.vmap(self.output_projection)(x)
            return y
        
        keys = jax.random.split(key, input_seqs.shape[0])
        return jax.vmap(single_forward)(input_seqs, keys)

    def predict(self, input_seq, key=None):
        """ Predict and remove the cumsum on the prediction if needed """
        output_seq = self(input_seq, key)

        if CONFIG["cum_sum"]:
            diff_seq = output_seq[:, 1:, :] - output_seq[:, :-1, :]
            first_val = output_seq[:, :1, :]
            return jnp.concatenate([first_val, diff_seq], axis=1)
        else:
            return output_seq


#%%
# --- 4. Training Setup & Comparison ---

# Initialize Models
key, k1 = jax.random.split(key, 2)

# Replacing Transformer with S4
models = {
    "S4": S4Model(
        input_dim=CONFIG["seq_dim"], 
        d_model=CONFIG["d_model"], 
        n_layers=2,           # S4 is deep usually, but 2 layers is fine here
        d_state=64,           # State dimension for the ODE
        key=k1
    )
}

# Train Step
@eqx.filter_jit
def train_step(model, optimizer, opt_state, batch, key):
    Xs, Ys = batch["x"], batch["y"]
    
    def loss_fn(m, x, y, k):
        # Model forward
        y_hat = m(x, key=k)

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
        Ys_hat = model(Xs, key=key)

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

        # Ys_hat = model(Xs, key=key)
        Ys_hat = model.predict(Xs, key=key)

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