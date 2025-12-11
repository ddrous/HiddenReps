import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

class SineWaveDataModule(pl.LightningDataModule):
    def __init__(self, seq_len=50, batch_size=32, num_samples=1000):
        super().__init__()
        self.save_hyperparameters()

    def generate_data(self):
        # Generate mixed sine waves: sin(x) + sin(2.3x)
        x = np.linspace(0, 100, self.hparams.num_samples + self.hparams.seq_len)
        # data = np.sin(x) + np.sin(2.3 * x)
        data = np.sin(2.0*x)
        
        X, y = [], []
        for i in range(len(data) - self.hparams.seq_len):
            X.append(data[i:i+self.hparams.seq_len])
            y.append(data[i+self.hparams.seq_len]) # Predict next step
            
        return torch.FloatTensor(np.array(X)).unsqueeze(-1), torch.FloatTensor(np.array(y)).unsqueeze(-1)

    def setup(self, stage=None):
        X, y = self.generate_data()
        split = int(0.8 * len(X))
        self.train_ds = TensorDataset(X[:split], y[:split])
        self.val_ds = TensorDataset(X[split:], y[split:])

    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True)
    def val_dataloader(self):   return DataLoader(self.val_ds, batch_size=self.hparams.batch_size)