import numpy as np
import jax.numpy as jnp
import torch.utils.data as data
from jax.tree_util import tree_map

def numpy_collate(batch):
  """Collate function to convert a batch of PyTorch data into NumPy arrays."""
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  """Custom DataLoader to return NumPy arrays from a PyTorch Dataset."""
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
      super(self.__class__, self).__init__(dataset,
          batch_size=batch_size,
          shuffle=shuffle,
          sampler=sampler,
          batch_sampler=batch_sampler,
          num_workers=num_workers,
          collate_fn=numpy_collate,
          pin_memory=pin_memory,
          drop_last=drop_last,
          timeout=timeout,
          worker_init_fn=worker_init_fn)

class NumpyDataset(data.Dataset):
    """Simple Dataset wrapping numpy arrays."""
    def __init__(self, *arrays):
        self.arrays = arrays

    def __getitem__(self, index):
        return tuple(arr[index] for arr in self.arrays)

    def __len__(self):
        return self.arrays[0].shape[0]

class SpiralDataHandler:
    def __init__(self, batch_size=64, num_points=10000, num_clusters=5, 
                 val_split=0.1, cluster_spread=0.5, noise_level=0.05, seed=42, num_workers=0):
        self.batch_size = batch_size
        self.num_points = num_points
        self.num_clusters = num_clusters
        self.val_split = val_split
        self.cluster_spread = cluster_spread
        self.noise_level = noise_level
        self.seed = seed
        self.num_workers = num_workers
        self.rng = np.random.default_rng(seed)
        
        self.means = None
        self.stds = None
        
        # Prepare data immediately
        self._prepare_data()

    def _generate_data(self):
        # Exact replication of the logic from original data.py
        points_per_cluster_base = self.num_points // self.num_clusters
        print(f"Generating {self.num_points} points across {self.num_clusters} clusters.")

        # Unequal cluster sizes
        random_sizes = self.rng.integers(-points_per_cluster_base//2, points_per_cluster_base//2, size=self.num_clusters)
        random_sizes -= int(random_sizes.mean())
        points_per_cluster = points_per_cluster_base + random_sizes
        points_per_cluster[0] += self.num_points - points_per_cluster.sum()

        all_data, all_labels = [], []
        center_thetas = self.rng.uniform(0.5 * np.pi, 3.5 * np.pi, self.num_clusters)
        center_heights = self.rng.uniform(-2.0, 2.0, self.num_clusters)

        for i in range(self.num_clusters):
            theta = self.rng.normal(loc=center_thetas[i], scale=self.cluster_spread, size=points_per_cluster[i])
            height = self.rng.normal(loc=center_heights[i], scale=self.cluster_spread, size=points_per_cluster[i])
            theta = np.clip(theta, 0, 4 * np.pi)
            height = np.clip(height, -2, 2)

            x = theta * np.cos(theta) + self.rng.normal(0, self.noise_level, points_per_cluster[i])
            y = height + self.rng.normal(0, self.noise_level, points_per_cluster[i])
            z = theta * np.sin(theta) + self.rng.normal(0, self.noise_level, points_per_cluster[i])

            all_data.append(np.vstack([x, y, z]).T)
            all_labels.append(np.full(points_per_cluster[i], i))

        data = np.vstack(all_data)
        labels = np.concatenate(all_labels)
        
        # Shuffle
        perm = self.rng.permutation(len(data))
        return data[perm], labels[perm]

    def _prepare_data(self):
        full_data, full_labels = self._generate_data()
        
        # Split
        val_len = int(self.val_split * len(full_data))
        train_len = len(full_data) - val_len
        
        train_data_raw = full_data[:train_len]
        self.train_labels = full_labels[:train_len]
        val_data_raw = full_data[train_len:]
        self.val_labels = full_labels[train_len:]
        
        # Calculate Stats on Train only
        self.means = train_data_raw.mean(axis=0)
        self.stds = train_data_raw.std(axis=0)
        self.stds[self.stds < 1e-6] = 1.0 # Safety
        
        # Normalize and cast to float32
        self.train_data = ((train_data_raw - self.means) / self.stds).astype(np.float32)
        self.val_data = ((val_data_raw - self.means) / self.stds).astype(np.float32)

        # Wrap in Datasets
        self.train_ds = NumpyDataset(self.train_data, self.train_labels)
        self.val_ds = NumpyDataset(self.val_data, self.val_labels)

    def get_iterator(self, split='train', shuffle=True):
        dataset = self.train_ds if split == 'train' else self.val_ds
        
        # JAX often prefers fixed batch sizes (drop_last=True) to avoid recompilation
        drop_last = True if split == 'train' else False
        
        return NumpyLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            num_workers=self.num_workers,
            drop_last=drop_last
        )
            
    def get_full_data(self, split='train'):
        if split == 'train':
            return jnp.array(self.train_data), jnp.array(self.train_labels)
        return jnp.array(self.val_data), jnp.array(self.val_labels)

    def compute_true_latents(self, split='train', normalize=True):
        """
        Compute the intrinsic 2D coordinates (theta, height)
        underlying the spiral structure.
        
        Returns:
            latents: jnp.array of shape (N, 2)
        """
        # Get normalized data
        if split == 'train':
            data = self.train_data
        else:
            data = self.val_data

        # Un-normalize
        data = data * self.stds + self.means
        x, y, z = data[:, 0], data[:, 1], data[:, 2]

        # Intrinsic coordinates
        theta = np.sqrt(x**2 + z**2)
        height = y

        latents = np.stack([theta, height], axis=1)

        if normalize:
            latents = (latents - latents.mean(axis=0)) / (latents.std(axis=0) + 1e-6)

        return jnp.array(latents, dtype=jnp.float32)
