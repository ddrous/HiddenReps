#%%
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torchvision import datasets, transforms

class SineWaveDataModule(pl.LightningDataModule):
    def __init__(self, seq_len=50, batch_size=32, num_samples=1000):
        super().__init__()
        self.save_hyperparameters()

    def generate_data(self):
        # Generate mixed sine waves: sin(x) + sin(2.3x)
        x = np.linspace(0, 100, self.hparams.num_samples + self.hparams.seq_len)
        # data = np.sin(x) + np.sin(2.3 * x)
        # data = np.sin(2.0*x)

        ## Data is just noise !
        data = np.random.randn(*x.shape)

        X, y = [], []
        for i in range(len(data) - self.hparams.seq_len):
            X.append(data[i:i+self.hparams.seq_len])
            # y.append(data[i+self.hparams.seq_len]) # Predict next step

            ## y is a single value that needs to account for every step in the sequence.
            ## Sum of all input arrs
            # y.append(data[i:i+self.hparams.seq_len].mean())

            # ## the sign of the sequence is how often its values are positive vs negative
            # seq = data[i:i+self.hparams.seq_len]
            # sign = 1.0 if (seq >= 0).sum() >= (seq < 0).sum() else -1.0
            # y.append(sign*data[i+self.hparams.seq_len])

            ## Return the maximum along the sequence
            # y.append(data[i:i+self.hparams.seq_len].max())

            # ## The output is all the input, but transposed
            # y.append(data[i:i+self.hparams.seq_len])
            seq_out = data[i:i+self.hparams.seq_len]
            y_dat = np.concatenate([seq_out, -seq_out[::-1]])  # Mirror and invert
            # print("SHape is", y_dat.shape, flush=True)
            y.append(y_dat)

        # return torch.FloatTensor(np.array(X)).unsqueeze(-1), torch.FloatTensor(np.array(y)).unsqueeze(-1)
        return torch.FloatTensor(np.array(X)).unsqueeze(-1), torch.FloatTensor(np.array(y))

    def setup(self, stage=None):
        X, y = self.generate_data()
        split = int(0.8 * len(X))
        self.train_ds = TensorDataset(X[:split], y[:split])
        self.val_ds = TensorDataset(X[split:], y[split:])

    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True)
    def val_dataloader(self):   return DataLoader(self.val_ds, batch_size=self.hparams.batch_size)



#%% SpiralDataModule
# ----------------------------------------------
# PYTORCH DATASET (Kept external for cleaner separation)
# ----------------------------------------------

class ManifoldDataset(Dataset):
    """Dataset for 3D spiral reconstruction: x = y."""
    def __init__(self, data: np.ndarray):
        self.data = torch.from_numpy(data).float()
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.data[idx] 
        return x, y

# ----------------------------------------------
# COMPACT PYTORCH LIGHTNING DATAMODULE
# ----------------------------------------------

class SpiralDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_points=10000, revolutions=10, 
                 radius_decay=0.85, noise_level=0.07, val_split=0.1):
        super().__init__()
        # Use save_hyperparameters to store all args and access via self.hparams
        self.save_hyperparameters()

    def _generate_data(self):
        """Generates a rolled plane (Swiss roll) embedded in 3D space."""
        hp = self.hparams
        
        # Two parameters: angle and height
        theta = np.random.uniform(0, 4 * np.pi, hp.num_points)  # Wrapping angle
        height = np.random.uniform(-2, 2, hp.num_points)         # Vertical extent
        
        # Roll the plane into a spiral
        x = theta * np.cos(theta) + np.random.normal(0, hp.noise_level, hp.num_points)
        y = height + np.random.normal(0, hp.noise_level, hp.num_points)
        z = theta * np.sin(theta) + np.random.normal(0, hp.noise_level, hp.num_points)
        
        data = np.vstack([x, y, z]).T
        data -= data.mean(axis=0)
        return data

    def setup(self, stage=None):
        # Generate data and ensure it's a new copy each time setup is run
        full_data = self._generate_data()
        
        # Determine split sizes
        val_len = int(self.hparams.val_split * len(full_data))
        train_len = len(full_data) - val_len
        
        # Instantiate ManifoldDataset once for the full data
        full_dataset = ManifoldDataset(full_data)
        
        # Use random_split for a clean, shufflied split
        self.train_ds, self.val_ds = random_split(full_dataset, [train_len, val_len])

    def train_dataloader(self): 
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True)
    
    def val_dataloader(self):   
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size)

# --- Example Usage ---
if __name__ == '__main__':
    dm = SpiralDataModule(num_points=5000, batch_size=128)
    dm.prepare_data() # Not strictly necessary here, but good practice
    dm.setup()
    
    train_loader = dm.train_dataloader()
    print(f"Total Train Samples: {len(dm.train_ds)}")
    
    # Check one batch
    x, y = next(iter(train_loader))
    print(f"Batch X shape: {x.shape}")
    print(f"Batch Y shape: {y.shape}")

# %%


# The data manipulation happens outside, in the DataModule.
class ClusteredManifoldDataset(Dataset):
    """Dataset returning the 3D data point (x) and its cluster label (y)."""
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- PyTorch Lightning DataModule (Updated) ---
class SpiralClusteredDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_points=10000, num_clusters=5, 
                 val_split=0.1, cluster_spread=0.5, noise_level=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.means = None # Will store the mean
        self.stds = None  # Will store the standard deviation

    def _generate_data(self):
        # ... (Same data generation logic as before) ...
        hp = self.hparams
        
        points_per_cluster_base = hp.num_points // hp.num_clusters
        ## We want unequal cluster sizes, adding to the base number a random amount (their sum of this random array should be 0)
        random_sizes = np.random.randint(-points_per_cluster_base//2, points_per_cluster_base//2, size=hp.num_clusters)
        random_sizes -= random_sizes.mean().astype(np.int64)  # Center
        points_per_cluster = points_per_cluster_base + random_sizes
        ## Add any remainder due to float mean calculation to the first cluster
        points_per_cluster[0] += hp.num_points - points_per_cluster.sum()

        all_data, all_labels = [], []
        center_thetas = np.random.uniform(0.5 * np.pi, 3.5 * np.pi, hp.num_clusters)
        center_heights = np.random.uniform(-2.0, 2.0, hp.num_clusters)

        for i in range(hp.num_clusters):
            theta = np.random.normal(loc=center_thetas[i], scale=hp.cluster_spread, size=points_per_cluster[i])
            height = np.random.normal(loc=center_heights[i], scale=hp.cluster_spread, size=points_per_cluster[i])
            theta = np.clip(theta, 0, 4 * np.pi)
            height = np.clip(height, -2, 2)

            x = theta * np.cos(theta) + np.random.normal(0, hp.noise_level, points_per_cluster[i])
            y = height + np.random.normal(0, hp.noise_level, points_per_cluster[i])
            z = theta * np.sin(theta) + np.random.normal(0, hp.noise_level, points_per_cluster[i])

            all_data.append(np.vstack([x, y, z]).T)
            all_labels.append(np.full(points_per_cluster[i], i))

        data = np.vstack(all_data)
        labels = np.concatenate(all_labels)
        perm = np.random.permutation(len(data))
        return data[perm], labels[perm]

    def setup(self, stage=None):
        full_data, full_labels = self._generate_data()
        
        # 1. Split data (BEFORE NORMALIZATION)
        val_len = int(self.hparams.val_split * len(full_data))
        train_len = len(full_data) - val_len
        
        # Determine train and validation data arrays
        train_data = full_data[:train_len]
        train_labels = full_labels[:train_len]
        val_data = full_data[train_len:]
        val_labels = full_labels[train_len:]
        
        # 2. FIT: Calculate statistics ONLY on the training data
        self.means = torch.from_numpy(train_data.mean(axis=0)).float()
        self.stds = torch.from_numpy(train_data.std(axis=0)).float()
        
        # Ensure minimum std is used to prevent division by zero for constant features
        # self.stds[self.stds < 1e-6] = 1.0 
        
        # 3. TRANSFORM: Apply normalization to both splits
        
        # Convert to torch tensor *before* normalization
        train_data_t = torch.from_numpy(train_data).float()
        val_data_t = torch.from_numpy(val_data).float()

        # Apply standard scaling: (X - mean) / std
        train_data_normalized = (train_data_t - self.means) / self.stds
        val_data_normalized = (val_data_t - self.means) / self.stds
        
        # 4. Create Datasets (passing normalized data)
        self.train_ds = ClusteredManifoldDataset(
            train_data_normalized.numpy(), train_labels
        )
        self.val_ds = ClusteredManifoldDataset(
            val_data_normalized.numpy(), val_labels
        )

    # Dataloader methods remain unchanged
    def train_dataloader(self): 
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True)
    
    def val_dataloader(self):   
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size)


# --- Example Usage ---
if __name__ == '__main__':
    dm = SpiralClusteredDataModule(num_points=5000, batch_size=128)
    dm.prepare_data() # Not strictly necessary here, but good practice
    dm.setup()
    
    train_loader = dm.train_dataloader()
    print(f"Total Train Samples: {len(dm.train_ds)}")
    
    # Check one batch
    x, y = next(iter(train_loader))
    print(f"Batch X shape: {x.shape}")
    print(f"Batch Y (Labels) shape: {y.shape}")





#%%

# --- 1. Data Module ---
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # [cite_start]Paper uses 32x32 images [cite: 438] and Tanh activation (requires [-1, 1] norm)
        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ])

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        full_train = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
        # [cite_start]Split train/val as per standard practice, though paper uses 60k train [cite: 451]
        self.train_ds, self.val_ds = random_split(full_train, [55000, 5000])
        self.test_ds = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=2)
