#%%
import grain.python as grain
import jax
import numpy as np

import seaborn as sns
sns.set(style="white", context="talk")



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


def main():

    train_data, test_data = generate_cumsum_data(num_seqs=CONFIG["num_seqs"],
                                                seq_len=CONFIG["seq_len"], 
                                                seq_dim=CONFIG["seq_dim"])

    # --- Use the existing cumsum data (already 3D: [num_samples, seq_len, features]) ---
    # train_data is defined earlier from generate_cumsum_data() as (8000, 16, 2)
    print("Original train_data shape:", train_data.shape)  # Should be (8000, 16, 2)

    # 1. Transformation function (works on a single sample of shape (seq_len, features))
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

    # 2. Build the Grain pipeline
    dataset = grain.MapDataset.source(train_data)   # each element is (16,2)
    dataset = dataset.map(process_sample)           # now each element is a dict with x,y
    # Batch (do NOT shuffle here – let the sampler handle it)
    dataset = dataset.batch(batch_size=CONFIG["batch_size"], drop_remainder=True)

    # 3. Sampler (handles shuffling)
    sampler = grain.IndexSampler(
        num_records=len(dataset),
        shuffle=True,
        seed=CONFIG["seed"],
        shard_options=grain.ShardOptions(shard_index=0, shard_count=1)
    )

    # 4. DataLoader (start with worker_count=0 for debugging)
    loader = grain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        worker_count=24,
        worker_buffer_size=2,
    )

    # 5. JAX bridge (moves batches to GPU)
    def jax_device_loader(grain_loader):
        for batch in grain_loader:
            yield jax.device_put(batch)

    train_loader = jax_device_loader(loader)

    # Test one batch
    batch = next(train_loader)
    print(f"X shape: {batch['x'].shape}, device: {batch['x'].devices()}")
    print(f"Y shape: {batch['y'].shape}, device: {batch['y'].devices()}")
