#%%
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split

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



#%%
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

    # def _generate_data(self):
    #     """Generates the 3D spiral data based on hparams."""
    #     hp = self.hparams
    #     t = np.linspace(0, 2 * np.pi * hp.revolutions, hp.num_points)
    #     R = 1.0 * np.exp(-t / (2 * np.pi * hp.revolutions / (1 - hp.radius_decay)))
        
    #     # Parametric equations
    #     x = R * np.cos(t) + np.random.normal(0, hp.noise_level, hp.num_points)
    #     y = R * np.sin(t) + np.random.normal(0, hp.noise_level, hp.num_points)
    #     z = (t / (2 * np.pi * hp.revolutions) * 4.0) + np.random.normal(0, hp.noise_level, hp.num_points)
        
    #     data = np.vstack([x, y, z]).T
    #     data -= data.mean(axis=0) # Center the data
    #     return data

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
