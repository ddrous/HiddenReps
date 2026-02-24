#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax
jax.config.update("jax_debug_nans", True)

import jax.numpy as jnp
import equinox as eqx
import optax
# import torch
# from torch.utils import data
import math


## Print JAX info
jax.print_environment_info()

## JAX debug NaNs


# --- Configuration ---
CONFIG = {
    "seed": 2028,
    "n_epochs": 2,
    "print_every": 1,
    "num_seqs": 10000,
    "batch_size": 32*160,
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

# def get_dataset_on_device(config):
#   datset = get_dataset(config)
#   sharding = jax.P(config.mesh_axis_names)
#   return map(ft.partial(jax.make_array_from_process_local_data, sharding), datset)

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

# class RandomNormalDataset(data.Dataset):
#     def __init__(self, data, cum_sum, device="cpu"):
#         self.data = torch.from_numpy(data).float().to(device)
#         # self.data = data
#         self.cum_sum = cum_sum

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, idx):
#         traj = self.data[idx]       ## Shape (seq_len, seq_dim)
#         if self.cum_sum:
#             traj = torch.cumsum(traj, dim=0)
#             # traj = np.cumsum(traj, axis=0)
        
#         # X is the entire traj (except last row last col, Y is the last dim)
#         y = traj[:, -1:].copy()        # Last dim is target
#         traj[-1, -1] = 0.0              # Zero out last point of target
#         x = traj

#         return x, y


# # def numpy_collate(batch):
# # #   return tree_map(np.asarray, data.default_collate(batch))
# #   return tree_map(np.asarray, batch)

def numpy_collate(batch):
    # 'batch' is a list of tuples: [(x1, y1), (x2, y2), ..., (x25, y25)]
    
    # 1. Unzip the batch into separate lists for x and y
    # elements becomes: [(x1, x2, ...), (y1, y2, ...)]
    elements = list(zip(*batch))
    
    # 2. Stack them into single numpy arrays
    # xs becomes shape (batch_size, seq_len, seq_dim)
    # ys becomes shape (batch_size, seq_len, 1)
    xs = np.stack(elements[0])
    ys = np.stack(elements[1])
    
    return xs, ys

def torch_to_jax(batch):
    elems = list(zip(*batch))
    xs = torch.stack(elems[0], dim=0)
    ys = torch.stack(elems[1], dim=0)
    xs_jax = jax.device_put(xs.cpu().numpy())
    ys_jax = jax.device_put(ys.cpu().numpy())
    return xs_jax, ys_jax

# def jax_collate_gpu(batch):
#     """
#     Unzips, stacks on GPU, and converts to JAX via zero-copy DLPack.
#     Assumes inputs are already Torch tensors (e.g. from a GPU-based Dataset).
#     """
#     # 1. Unzip the batch (x1, y1), (x2, y2) -> [x1, x2...], [y1, y2...]
#     xs_list, ys_list = zip(*batch)
    
#     # 2. Stack directly on the current Torch device (GPU)
#     # Using torch.stack is faster than numpy.stack for GPU tensors
#     xs_torch = torch.stack(xs_list)
#     ys_torch = torch.stack(ys_list)
    
#     # 3. Zero-copy conversion to JAX
#     # to_dlpack/from_dlpack handles the memory pointer handoff
#     xs_jax = jax_dlpack.from_dlpack(to_dlpack(xs_torch))
#     ys_jax = jax_dlpack.from_dlpack(to_dlpack(ys_torch))
    
#     return xs_jax, ys_jax

# train_dataset = RandomNormalDataset(train_data, cum_sum=CONFIG["cum_sum"])
# train_loader = data.DataLoader(train_dataset, 
#                                batch_size=CONFIG["batch_size"], 
#                                shuffle=True, 
#                             #    collate_fn=numpy_collate, 
#                                collate_fn=jax_collate_gpu, 
#                             #    pin_memory=True,
#                                num_workers=24)

#%%

# # --- 1. Dataset: Keep on CPU, Fix Corruption ---
# class RandomNormalDataset(data.Dataset):
#     def __init__(self, data_array, cum_sum=False):
#         # Keep data on CPU (float32). Do NOT move to GPU here if using num_workers > 0.
#         if isinstance(data_array, np.ndarray):
#             self.data = torch.from_numpy(data_array).float()
#         else:
#             self.data = data_array.float()
            
#         self.cum_sum = cum_sum

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, idx):
#         # 1. Get data
#         traj = self.data[idx] 

#         # 2. Apply ops (and ensure we have a copy, not a view)
#         if self.cum_sum:
#             # cumsum creates a new tensor, so it's safe
#             traj = torch.cumsum(traj, dim=0)
#             # traj = np.cumsum(traj, axis=0)
#         else:
#             # IMPORTANT: Clone to avoid modifying the original self.data
#             traj = traj.clone()
        
#         # 3. Prepare X and Y
#         # Clone target to ensure memory safety
#         y = traj[:, -1:].clone()
        
#         # In-place modification is now safe because we own 'traj'
#         traj[-1, -1] = 0.0
#         x = traj

#         # Return torch CPU tensors
#         return x, y


# # --- 2. Efficient Data Loading Logic ---

# # def torch_to_jax(tensor):
# #     """Torch CPU to JAX GPU """
# #     return jax.device_put(tensor.cpu().numpy())

# # def jax_dataloader_iterator(dataloader):
# #     """
# #     Generator that iterates over a PyTorch DataLoader, 
# #     moves batches to GPU asynchronously, and converts to JAX.
# #     """
# #     for batch in dataloader:
# #         x_torch, y_torch = batch

# #         # Move to JAX (and implicitly to GPU)
# #         x_jax = torch_to_jax(x_torch)
# #         y_jax = torch_to_jax(y_torch)
        
# #         yield x_jax, y_jax

# # --- 3. Usage ---
# # A. Initialize Dataset (CPU)
# train_dataset = RandomNormalDataset(train_data, cum_sum=CONFIG["cum_sum"])


# # # B. Initialize Loader
# # # - num_workers > 0 is now safe because dataset is CPU.
# # # - pin_memory=True is CRITICAL. It allows fast async transfer from CPU RAM to GPU RAM.
# # train_loader_torch = data.DataLoader(
# #     train_dataset, 
# #     batch_size=CONFIG["batch_size"], 
# #     shuffle=True, 
# #     num_workers=24,        # Adjust based on CPU cores
# #     # pin_memory=True,      # Required for fast CPU->GPU transfer
# #     drop_last=True
# # )

# import jax_dataloader as jdl
# # jdl.manual_seed(1234) # Set the global seed to 1234 for reproducibility

# train_loader_torch = jdl.DataLoader(
#     train_dataset, # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
#     backend='jax', # Use 'jax' backend for loading data
#     batch_size=32, # Batch size 
#     shuffle=True, # Shuffle the dataloader every iteration or not
#     drop_last=False, # Drop the last batch or not
#     generator=jdl.Generator() # Control the randomness of this dataloader 
# )



# # C. Training Loop
# print("Starting JAX training loop...")

# # Wrap loader with the JAX converter
# train_loader = train_loader_torch
# # train_loader = jax_dataloader_iterator(train_loader_torch)



#%%
# # --- 1. Dataset: Keep on CPU, Fix Corruption ---
# from cyreal import datasets
# class RandomNormalDataset(datasets.dataset_protocol.DatasetProtocol):
#     def __init__(self, data_array, cum_sum=False):
#         self.data = data_array.astype(np.float32)  # Keep as NumPy array on CPU
#         self.cum_sum = cum_sum

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, idx):
#         # 1. Get data
#         seq = self.data[idx] 

#         # 2. Apply ops (and ensure we have a copy, not a view)
#         if self.cum_sum:
#             seq = np.cumsum(seq, axis=0)
#         else:
#             seq = seq.copy()
        
#         # 3. Prepare X and Y
#         y = seq[:, -2:].copy()
        
#         # In-place modification is now safe because we own 'traj'
#         seq[-1, -1] = 0.0
#         x = seq

#         # Return torch CPU tensors
#         return x, y


# # --- 3. Usage ---
# # A. Initialize Dataset (CPU)
# train_dataset = RandomNormalDataset(train_data, cum_sum=CONFIG["cum_sum"])


# # # B. Initialize Loader
# from cyreal.transforms import BatchTransform, DevicePutTransform
# from cyreal.loader import DataLoader
# from cyreal.sources import ArraySource

# pipeline = [
#   # Load dataset into memory-backed array
#   ArraySource(train_dataset, ordering="shuffle"),
#   # Batch it
#   BatchTransform(batch_size=CONFIG["batch_size"], drop_last=True),
#   # Move the batch to the GPU
#   DevicePutTransform(),
# ]
# train_loader = DataLoader(pipeline)
# state = train_loader.init_state(key)
# iterate = jax.jit(train_loader.next)

# ## Test loader
# batch, state, _ = iterate(state)
# print("Batch :", type(batch))  # Should be (batch_size, seq_len, seq_dim)
# print("Batch X shape:", batch.shape)  # Should be (batch_size, seq_len, seq_dim)
# print("Batch Y shape:", state.shape)  # Should be on GPU


# import grain
# train_dataset_jax = (
#     grain.MapDataset.source(train_dataset)
#     .shuffle(seed=10)       # Shuffles globally.
#     .map(lambda x: x + 1)   # Maps each element.
#     .batch(batch_size=2)    # Batches consecutive elements.
# )

# print(train_dataset_jax)
# print(list(train_dataset_jax))

#%%
import grain.python as grain
import jax
import numpy as np


# Using dummy data for the example (List of arrays)
train_data = np.random.randn(10, 5)

# 2. Transformation Logic
def process_batch(traj):
    """Operates on a single element before batching."""
    if CONFIG["cum_sum"]:
        traj = np.cumsum(traj, axis=0)
    
    # Create x, y logic
    y = traj[-1:, :].copy()  # Example: take the last row as target
    x = traj.copy()
    x[-1, :] = 0.0          # Mask out the target in the input
    
    return {"x": x, "y": y}

# if __name__ == "__main__":
    
# 3. The Grain Pipeline
# We wrap our list in a MapDataset source
dataset = grain.MapDataset.source(train_data).map(process_batch).shuffle(seed=42).batch(batch_size=CONFIG["batch_size"]) 

# The DataLoader handles the multiprocessing
loader = grain.DataLoader(
    data_source=dataset,
    sampler=grain.IndexSampler(
        num_records=len(dataset), 
        shard_options=grain.ShardOptions(
                shard_index=0,     # This is the first (and only) worker group
                shard_count=1      # Total number of worker groups
            ),
        shuffle=True, 
        seed=42
    ),
    worker_count=1, # Start low (e.g., 4) and increase if CPU bottlenecks
    worker_buffer_size=2,
)

# 4. The JAX Bridge
def jax_device_loader(grain_loader):
    """Simple generator to move batches to device."""
    for batch in grain_loader:
        # device_put converts the dict of numpy arrays to JAX arrays on GPU
        yield jax.device_put(batch)

# 5. Usage
train_loader = jax_device_loader(loader)

# Grab one batch to test
batch = next(train_loader)
print(f"X shape: {batch['x'].shape}, Device: {batch['x'].devices()}")



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

# Optimizers (One per model)
optimizers = {name: optax.adamw(CONFIG["lr"]) for name in models}
opt_states = {name: opt.init(eqx.filter(model, eqx.is_inexact_array)) 
              for name, (model, opt) in zip(models.keys(), zip(models.values(), optimizers.values()))}

# Train Step
@eqx.filter_jit
def train_step(model, optimizer, opt_state, batch, key):
    # batch = jax.device_put(batch)
    Xs, Ys = batch
    
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

# Training Loop
loss_history = {name: [] for name in models}

print(f"Training 1 models: {list(models.keys())}")

for epoch in range(CONFIG["n_epochs"]):
    batch_losses = {name: [] for name in models}
    
    # for _ in range(train_loader.steps_per_epoch):
    #     batch, state, mask = iterate(state)

    for batch in train_loader:
        # batch = jnp.array(batch)
        key, subkey = jax.random.split(key)
        # batch = jax.device_put(batch)
        # batch = torch_to_jax(batch)
        
        for name in models:
            models[name], opt_states[name], loss = train_step(
                models[name], optimizers[name], opt_states[name], batch, subkey
            )
            batch_losses[name].append(loss)

    for name in models:
        avg = np.mean(batch_losses[name])
        loss_history[name].append(avg)
        
    if (epoch+1) % CONFIG["print_every"] == 0:
        print(f"Epoch {epoch+1}: " + ", ".join([f"{n}: {l:.5f}" for n, l in zip(loss_history.keys(), [h[-1] for h in loss_history.values()])]), flush=True)

#%%
# Plot Training Curves
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

#%%
# --- 5. Metrics & Comparison ---

test_dataset = RandomNormalDataset(test_data, cum_sum=CONFIG["cum_sum"])
test_loader = data.DataLoader(test_dataset, batch_size=test_data.shape[0], shuffle=False, collate_fn=numpy_collate, num_workers=24)

def compute_metrics(model, test_loader, key):
    # Standard evaluation params
    # test_data of shape (num_traj, seq_len, seq_dim)
    # test_data = jnp.asarray(test_data)
    batch = next(iter(test_loader))

    Xs, Ys = batch
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
    mse, mae = compute_metrics(model = model, test_loader=test_loader, key=key)
    metrics_results[name] = (mse, mae)
    print(f"{name:<25} | {mse:.5f}    | {mae:.5f}     ")

#%%


## We want to plot the last y hat against the true last y for the test set
batch = next(iter(test_loader))
Xs, Ys = batch

for name, model in models.items():
    Ys_hat = model(Xs, key=key)
    last_Y = Ys[:, -1, :]  # Shape (batch_size, 1)
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
