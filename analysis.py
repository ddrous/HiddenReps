import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def analyze_hidden_states(model, dataloader, title_suffix=""):
    model.eval()
    x_batch, _ = next(iter(dataloader))
    
    with torch.no_grad():
        _, hidden_seq = model(x_batch) 
        # hidden_seq shape: (Batch, Seq, Hidden)
    
    # Flatten for global analysis: (Batch*Seq, Hidden)
    h_flat = hidden_seq.reshape(-1, hidden_seq.shape[-1]).cpu().numpy()

    # 1. Singular Value Spectrum (Covariance Analysis)
    # If curve is flat = High Rank (Good usage or too small). 
    # If curve drops fast = Low Rank (Model is too big/sparse).
    # NaN to zero handling
    _, s, _ = np.linalg.svd(h_flat - h_flat.mean(axis=0), full_matrices=False)
    s = s / s.max() # Normalize

    # 2. PCA Trajectory of the first sample in batch
    pca = PCA(n_components=2)
    h_sample = hidden_seq[0].cpu().numpy() # (Seq, Hidden)
    h_pca = pca.fit_transform(h_sample)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Spectrum Plot
    ax[0].plot(s, marker='o')
    ax[0].set_title(f'Singular Value Spectrum {title_suffix}')
    ax[0].set_xlabel('Dimension Index'); ax[0].set_ylabel('Normalized Singular Value')
    ax[0].grid(True, alpha=0.3)

    # Trajectory Plot
    ax[1].plot(h_pca[:, 0], h_pca[:, 1], marker='.', alpha=0.6)
    ax[1].scatter(h_pca[0, 0], h_pca[0, 1], c='g', label='Start')
    ax[1].scatter(h_pca[-1, 0], h_pca[-1, 1], c='r', label='End')
    ax[1].set_title(f'Hidden State Trajectory (PCA) {title_suffix}')
    ax[1].legend()

    plt.tight_layout()
    plt.show()


class LossHistory(pl.Callback):
    """Callback to track training and validation loss in memory."""
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Retrieve 'train_loss' from the logged metrics
        loss = trainer.callback_metrics.get("train_loss")
        if loss: self.train_loss.append(loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Retrieve 'val_loss' from the logged metrics
        loss = trainer.callback_metrics.get("val_loss")
        if loss: self.val_loss.append(loss.item())

def plot_loss_curves(history, title="Loss Curve"):
    """Plots the training and validation loss from the History object."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.train_loss, label='Train Loss', marker='.')
    # Val loss is often logged fewer times, so we stretch it or plot on secondary axis if needed
    # For simplicity here, assuming 1:1 epoch mapping
    if history.val_loss:
        plt.plot(history.val_loss, label='Val Loss', marker='.')

    plt.title(title)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()