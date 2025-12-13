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


def plot_sigmoid_mask(sigparam, h_size, title="Learned Sigmoid Mask"):
    sigpos = np.linspace(0, h_size-1, h_size)
    sigdrop = sigparam[0] * h_size
    # print("Shapes are", sigpos.shape, sigdrop.shape, flush=True)
    sig_fun = 1.0 -  1/(1 + np.exp(10 * (-sigpos + sigdrop)))  # Steepness fixed at 100 for now
    plt.figure(figsize=(6, 4))
    plt.plot(sigpos, sig_fun, label=f'Sigmoid Mask (Drop at {sigdrop:.1f})')
    plt.title(title)
    plt.xlabel('Hidden Unit Index')
    plt.ylabel('Mask Value')
    plt.ylim([-0.1, 1.1])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_sigmoid_mask_final(sigparam, latent_dim, title="Learned Sigmoid Mask"):
    sigpos = np.linspace(0, latent_dim - 1, latent_dim)
    sigdrop_index = sigparam * latent_dim
    # Reverse-engineer the sigmoid function used in the model
    sig_mask = 1.0 -  1/(1 + np.exp(10 * (-sigpos + sigdrop_index)))
    
    plt.figure(figsize=(8, 4))
    plt.plot(sigpos, sig_mask, label=f'Sigmoid Mask (Drop at {sigdrop_index:.2f} of {latent_dim} dim)')
    plt.axvline(sigdrop_index, color='r', linestyle='--', label=f'Learned Cut-off $\sigma$={sigdrop_index:.2f}')
    
    # Plot the expected ID=2 position
    expected_id = 2.0
    expected_index = expected_id
    if expected_index < latent_dim:
        plt.axvline(expected_index, color='g', linestyle=':', label=f'Expected ID=2')
        
    plt.title(title)
    plt.xlabel('Latent Unit Index')
    plt.ylabel('Mask Value')
    plt.ylim([-0.1, 1.1])
    plt.xticks(np.arange(0, latent_dim, 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def analyze_latent_space(model, dataloader, title_suffix=""):
    """
    1. Collect Latent Codes (both masked and unmasked) and Original Data
    2. Run PCA on all three
    3. Visualize side by side
    """
    model.eval()
    all_z = []
    all_z_masked = []
    all_x = []
    
    with torch.no_grad():
        for x, _ in dataloader:
            x_hat, z, z_masked = model(x.to(model.device))
            all_z.append(z.cpu().numpy())
            all_z_masked.append(z_masked.cpu().numpy())
            all_x.append(x.cpu().numpy())
    
    data_z = np.concatenate(all_z, axis=0)
    data_z_masked = np.concatenate(all_z_masked, axis=0)
    data_x = np.concatenate(all_x, axis=0)
    
    print(f"Unmasked latent codes: {data_z.shape}")
    print(f"Masked latent codes: {data_z_masked.shape}")
    print(f"Original data: {data_x.shape}")
    
    # --- PCA Analysis ---
    pca_z = PCA(n_components=min(data_z.shape[1], data_z.shape[0]))
    pca_z_masked = PCA(n_components=min(data_z_masked.shape[1], data_z_masked.shape[0]))
    pca_x = PCA(n_components=min(data_x.shape[1], data_x.shape[0]))
    
    pca_z.fit(data_z)
    pca_z_masked.fit(data_z_masked)
    pca_x.fit(data_x)
    
    # --- Scree Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    # Unmasked latent codes
    axes[0].plot(np.arange(1, len(pca_z.explained_variance_ratio_) + 1), 
                 np.cumsum(pca_z.explained_variance_ratio_), marker='o', linestyle='-')
    axes[0].axhline(0.95, color='r', linestyle='--', label='95% Explained Variance')
    axes[0].set_title(f'PCA: Unmasked Latent Codes {title_suffix}')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Cumulative Explained Variance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Masked latent codes
    axes[1].plot(np.arange(1, len(pca_z_masked.explained_variance_ratio_) + 1), 
                 np.cumsum(pca_z_masked.explained_variance_ratio_), marker='o', linestyle='-')
    axes[1].axhline(0.95, color='r', linestyle='--', label='95% Explained Variance')
    axes[1].set_title(f'PCA: Masked Latent Codes {title_suffix}')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Original data
    axes[2].plot(np.arange(1, len(pca_x.explained_variance_ratio_) + 1), 
                 np.cumsum(pca_x.explained_variance_ratio_), marker='o', linestyle='-')
    axes[2].axhline(0.95, color='r', linestyle='--', label='95% Explained Variance')
    axes[2].set_title(f'PCA: Original Data {title_suffix}')
    axes[2].set_xlabel('Number of Components')
    axes[2].set_ylabel('Cumulative Explained Variance')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # --- 2D Projections ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Unmasked latent codes
    if data_z.shape[1] >= 2:
        z_2d = pca_z.transform(data_z)[:, :2]
        scatter = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], s=5, alpha=0.5, 
                                  c=data_z[:, 0], cmap='viridis')
        axes[0].set_title(f'2D PCA: Unmasked Latent Codes {title_suffix}')
        axes[0].set_xlabel('PC 1')
        axes[0].set_ylabel('PC 2')
        plt.colorbar(scatter, ax=axes[0], label='Latent Code Z[0]')
    
    # Masked latent codes
    if data_z_masked.shape[1] >= 2:
        z_masked_2d = pca_z_masked.transform(data_z_masked)[:, :2]
        scatter = axes[1].scatter(z_masked_2d[:, 0], z_masked_2d[:, 1], s=5, alpha=0.5, 
                                  c=data_z_masked[:, 0], cmap='viridis')
        axes[1].set_title(f'2D PCA: Masked Latent Codes {title_suffix}')
        axes[1].set_xlabel('PC 1')
        axes[1].set_ylabel('PC 2')
        plt.colorbar(scatter, ax=axes[1], label='Masked Latent Code Z[0]')
    
    # Original data
    if data_x.shape[1] >= 2:
        x_2d = pca_x.transform(data_x)[:, :2]
        scatter = axes[2].scatter(x_2d[:, 0], x_2d[:, 1], s=5, alpha=0.5, 
                                  c=np.arange(len(x_2d)), cmap='viridis')
        axes[2].set_title(f'2D PCA: Original Data {title_suffix}')
        axes[2].set_xlabel('PC 1')
        axes[2].set_ylabel('PC 2')
        plt.colorbar(scatter, ax=axes[2], label='Sample Index')
    
    plt.tight_layout()
    plt.show()