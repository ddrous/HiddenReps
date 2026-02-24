#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax
# jax.config.update("jax_debug_nans", True)
# ## Disable JIT as well
# jax.config.update("jax_disable_jit", True)

import jax.numpy as jnp
import equinox as eqx
import optax
# import torch
# from torch.utils import data
import math
import grain.python as grain


## Print JAX info
jax.print_environment_info()

## JAX debug NaNs
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

# --- Configuration ---
CONFIG = {
    "seed": 2028,
    "n_epochs": 100,
    "print_every": 10,
    "num_seqs": 64*4,
    "batch_size": 64,
    "seq_len": 16,
    "seq_dim": 2,
    "cum_sum": False,
    "ar_train_mode": False,
    "n_refine_steps": 10,
    "d_model": 128*1,
    "lr": 3e-4
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

   # Convert to JAX arrays and place on GPU
    x = jax.device_put(jnp.array(x))
    y = jax.device_put(jnp.array(y))

    return {"x": x, "y": y}



#%%
# --- 2. Transformer Models (Decoder-Only) ---
class PositionalEncoding(eqx.Module):
    embedding: jax.Array
    def __init__(self, d_model: int, max_len: int = 2000):
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.embedding = pe
    def __call__(self, x, start_idx=0):
        return x + self.embedding[start_idx : start_idx + x.shape[0], :]
 
class TransformerBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    norm2: eqx.nn.LayerNorm
    
    def __init__(self, d_model, n_heads, d_ff, dropout, key):
        k1, k2 = jax.random.split(key)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=n_heads, query_size=d_model, dropout_p=dropout, key=k1
        )
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.mlp = eqx.nn.MLP(
            in_size=d_model, out_size=d_model, width_size=d_ff, depth=1, activation=jax.nn.gelu, key=k2
        )
        self.norm2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, mask=None, key=None):
        attn_out = self.attention(x, x, x, mask=mask, key=key)
        x = jax.vmap(self.norm1)(x + attn_out)
        mlp_out = jax.vmap(self.mlp)(x)
        x = jax.vmap(self.norm2)(x + mlp_out)
        return x

class DecoderOnlyTransformer(eqx.Module):
    embedding: eqx.nn.Linear
    pos_encoder: PositionalEncoding
    blocks: list
    output_projection: eqx.nn.Linear
    refinement_mlp: eqx.nn.MLP

    d_model: int = eqx.field(static=True)
    n_substeps: int = eqx.field(static=True)
    use_refinement: bool = eqx.field(static=True)

    def __init__(self, input_dim, d_model, n_heads, n_layers, d_ff, n_substeps, max_len, use_refinement, key):
        self.d_model = d_model
        self.n_substeps = n_substeps
        self.use_refinement = use_refinement
        
        keys = jax.random.split(key, 5)
        self.embedding = eqx.nn.Linear(input_dim, d_model, key=keys[0])
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout=0.0, key=k)
            for k in jax.random.split(keys[1], n_layers)
        ]
        
        self.output_projection = eqx.nn.Linear(d_model, 1, key=keys[2])
        # Zero Init Output
        self.output_projection = eqx.tree_at(
            lambda l: (l.weight, l.bias), 
            self.output_projection, 
            (jnp.zeros_like(self.output_projection.weight), jnp.zeros_like(self.output_projection.bias))
        )

        self.refinement_mlp = eqx.nn.MLP(
            in_size=d_model * 2, out_size=input_dim, width_size=d_ff, depth=1, activation=jax.nn.gelu, key=keys[3]
        )
        # Zero Init Refinement
        self.refinement_mlp = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias),
            self.refinement_mlp,
            (jnp.zeros_like(self.refinement_mlp.layers[-1].weight), jnp.zeros_like(self.refinement_mlp.layers[-1].bias))
        )

    def make_causal_mask(self, seq_len):
        idx = jnp.arange(seq_len)
        return idx[:, None] < idx[None, :]

    def refine_step(self, start_val, context_h):
        def loop_fn(i, curr_val):
            w_emb = self.embedding(curr_val) * jnp.sqrt(self.d_model)
            combined = jnp.concatenate([w_emb, context_h])
            delta = self.refinement_mlp(combined)
            return curr_val + delta
        return jax.lax.fori_loop(0, self.n_substeps, loop_fn, start_val)

    def __call__(self, input_seqs, key=None):
        def single_forward(input_seq, key):
            x = jax.vmap(self.embedding)(input_seq) * jnp.sqrt(self.d_model)
            x = self.pos_encoder(x)
            mask = self.make_causal_mask(x.shape[0])
            for block in self.blocks:
                x = block(x, mask=mask, key=key)
            
            if not self.use_refinement:
                y = jax.vmap(self.output_projection)(x)
            else:
                y = jax.vmap(self.refine_step)(input_seq, x)

            return y
        
        keys = jax.random.split(key, input_seqs.shape[0])
        return jax.vmap(single_forward)(input_seqs, keys)



#%%
# --- 4. Training Setup & Comparison ---

# Initialize Models
key, k1, k2, k3 = jax.random.split(key, 4)

max_seq_len = max(train_data.shape[1], test_data.shape[1])
models = {
    "Transformer": DecoderOnlyTransformer(
        input_dim=CONFIG["seq_dim"], 
        d_model=CONFIG["d_model"], 
        n_heads=1, 
        n_layers=1, 
        d_ff=128,
        n_substeps=CONFIG["n_refine_steps"], 
        max_len=CONFIG["seq_len"], 
        use_refinement=False, 
        key=k1
    )
}

# Train Step
# @eqx.filter_jit
def train_step(model, optimizer, opt_state, batch, key):
    Xs, Ys = batch["x"], batch["y"]
    
    def loss_fn(m, x, y, k):
        # Transformer
        y_hat = m(x, key=k)

        indices = jnp.arange(x.shape[1])
        # indices = jnp.array([-1])

        return jnp.mean((y_hat[:, indices] - y[:, indices]) ** 2)

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, Xs, Ys, key)

    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss_val


def train(models, train_data, key):

    # Create a data source (no batching/shuffling here)
    class SimpleDataSource(grain.RandomAccessDataSource):
        def __init__(self, data):
            self._data = data        
        def __len__(self):
            return len(self._data)
        def __getitem__(self, index):
            return process_sample(self._data[index])

    data_source = SimpleDataSource(train_data)

    # Sampler handles shuffling
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shuffle=True,
        seed=CONFIG["seed"],
        shard_options=grain.NoSharding(),
        num_epochs=1,
    )

    # DataLoader handles batching
    loader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=[
            grain.Batch(batch_size=CONFIG["batch_size"], 
                        drop_remainder=False),
        ],
        worker_count=0,
    )

    # 4. The JAX Bridge
    def jax_device_loader(grain_loader):
        """Simple generator to move batches to device."""
        for batch in grain_loader:
            yield jax.device_put(batch)

    # 5. Usage; # Grab one batch to test
    train_loader = jax_device_loader(loader)
    batch = next(train_loader)
    print(f"Example X shape: {batch['x'].shape}, Device: {batch['x'].devices()}")

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

        epoch_iterator = jax_device_loader(loader)  # Create a new iterator for each epoch

        for batch in loader:
            key, subkey = jax.random.split(key)
            
            for name in models:
                models[name], opt_states[name], loss = train_step(
                    models[name], optimizers[name], opt_states[name], batch, subkey
                )
                batch_losses[name].append(loss)

                # print(f"    Loss for batch: {name}: {loss:.5f}", flush=True)

        for name in models:
            avg = np.mean(batch_losses[name])
            loss_history[name].append(avg)

        if (epoch+1) % CONFIG["print_every"] == 0:
            print(f"Epoch {epoch+1}: " + ", ".join([f"{n}: {l:.5f}" for n, l in zip(loss_history.keys(), [h[-1] for h in loss_history.values()])]), flush=True)


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
        Ys_hat = model(Xs, key=key)
        last_Y = Ys[:, -1, :]       # Shape (batch_size, 1)
        last_Y_hat = Ys_hat[:, -1, :]
        plt.figure(figsize=(6, 6))
        plt.scatter(last_Y, last_Y_hat, alpha=0.5)
        plt.xlabel("True Last Y")
        plt.ylabel("Predicted Last Y")
        plt.title(f"{name} - Last Y Prediction")
        plt.plot([last_Y.min(), last_Y.max()], [last_Y.min(), last_Y.max()], 'k--')  # Diagonal line
        plt.show()

        ## PLot x (1D) agianst pred y (1D) for all steps in a random sequence
        seq_idx = np.random.randint(0, Xs.shape[0])
        x = Xs[seq_idx, :, 0]  # Shape (seq_len,)
        ## reorder x
        x_ids = np.argsort(x)
        x_sorted = x[x_ids]
        y_sorted = Ys_hat[seq_idx, :, 0][x_ids]
        plt.figure(figsize=(6, 6))
        plt.scatter(x_sorted, y_sorted, alpha=0.5)
        plt.xlabel("Input X (sorted)")
        plt.ylabel("Predicted Y")
        plt.title(f"{name} - y_t vs x_t for seq {seq_idx}")
        plt.show()




#%%
if __name__ == "__main__":
    models, loss_history = train(models, train_data, key)
    test(models, test_data)
