import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from umap import UMAP
from sklearn.metrics import normalized_mutual_info_score as NMI
import equinox as eqx

def plot_loss_curves(train_loss, val_loss, title="Loss Curve"):
    plt.figure(figsize=(10, 4))
    plt.plot(train_loss, label='Train Loss', marker='.', alpha=0.6)
    if len(val_loss) > 0:
        # Interpolate val loss to match train x-axis if lengths differ
        x_val = np.linspace(0, len(train_loss), len(val_loss))
        plt.plot(x_val, val_loss, label='Val Loss', marker='.', linewidth=2)

    plt.title(title)
    plt.yscale('log')
    plt.xlabel('Steps')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def latent_space_analysis(model, data_handler, title_suffix=""):
    """
    Plots AE Latent, PCA, t-SNE, and UMAP embeddings side-by-side.
    Calculates NMI and Trustworthiness.
    """
    print("Collecting data for analysis...")
    # Get full validation data (on CPU for sklearn)
    X_jax, Y_jax = data_handler.get_full_data(split='val') # or train
    
    # Run inference in batches or full batch if it fits
    # Using vmap for efficiency
    @eqx.filter_jit
    def get_latent(m, x):
        return eqx.filter_vmap(m.encode)(x)
        
    Z_jax = get_latent(model, X_jax)
    
    # Convert to numpy for sklearn/plotting
    data_x = np.array(X_jax)
    data_z = np.array(Z_jax)
    data_y = np.array(Y_jax)

    # 2. Calculate Embeddings
    print(f"Computing embeddings for {len(data_x)} points...")
    
    # PCA on X (Reference)
    x_pca = PCA(n_components=2).fit_transform(data_z)
    
    # t-SNE on Z
    print("Running t-SNE...")
    x_tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42).fit_transform(data_z)
    
    # UMAP on Z
    print("Running UMAP...")
    x_umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(data_z)

    embeddings = {
        "Autoencoder": data_z, # Assumes latent_dim=2 for direct plot, otherwise takes first 2
        "PCA": x_pca,
        "t-SNE": x_tsne,
        "UMAP": x_umap
    }

    # 3. Calculate Metrics
    def get_nmi(embedding, labels):
        # Discretize 1st dimension into 10 bins
        bins = np.digitize(embedding[:, 0], np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), 10))
        return NMI(labels, bins)

    print("\n" + "="*60)
    print(f"⚡️ Embedding Quality Metrics {title_suffix}")
    print(f"{'Embedding':<20} | {'NMI (Struct)':<12} | {'Trustworthiness':<15}")
    print("-" * 60)

    for name, emb in embeddings.items():
        # If embedding > 2D, take first 2 for plotting, but use full for metrics if appropriate
        plot_emb = emb[:, :2]
        
        nmi = get_nmi(plot_emb, data_y)
        # Trustworthiness is heavy, calc only for Z
        tw = trustworthiness(data_x, plot_emb, n_neighbors=15) if "PCA" not in name else 1.0
        print(f"{name:<20} | {nmi:.4f}       | {tw:.4f}")
    print("="*60 + "\n")

    # 4. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    cmap = plt.cm.get_cmap('tab10', len(np.unique(data_y)))

    for i, (name, emb) in enumerate(embeddings.items()):
        ax = axes[i]
        plot_emb = emb[:, :2]
        sc = ax.scatter(plot_emb[:, 0], plot_emb[:, 1], c=data_y, cmap=cmap, s=8, alpha=0.6)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([]) 
        
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    fig.colorbar(sc, cax=cbar_ax, label="True Cluster Label")
    plt.suptitle(f"Latent Space Analysis {title_suffix}", fontsize=16)
    plt.show()


def ambiant_space_analysis(model, data_handler, title_suffix=""):
    """
    Plots PCA, t-SNE, and UMAP of the original data side-by-side. For the autoencoder only, use the latent space.
    Calculates NMI and Trustworthiness.
    """
    print("Collecting data for analysis...")
    # Get full validation data (on CPU for sklearn)
    X_jax, Y_jax = data_handler.get_full_data(split='val') # or train
    
    # Run inference in batches or full batch if it fits
    # Using vmap for efficiency
    @eqx.filter_jit
    def get_latent(m, x):
        return eqx.filter_vmap(m.encode)(x)
        
    Z_jax = get_latent(model, X_jax)
    
    # Convert to numpy for sklearn/plotting
    data_x = np.array(X_jax)
    data_z = np.array(Z_jax)
    data_y = np.array(Y_jax)

    # 2. Calculate Embeddings
    print(f"Computing embeddings for {len(data_x)} points...")
    
    # PCA on X (Reference)
    x_pca = PCA(n_components=2).fit_transform(data_x)
    
    # t-SNE on Z
    print("Running t-SNE...")
    x_tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42).fit_transform(data_x)
    
    # UMAP on Z
    print("Running UMAP...")
    x_umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(data_x)

    embeddings = {
        "Autoencoder (Latent*)": data_z, # Assumes latent_dim=2 for direct plot, otherwise takes first 2
        "PCA": x_pca,
        "t-SNE": x_tsne,
        "UMAP": x_umap
    }

    # 3. Calculate Metrics
    def get_nmi(embedding, labels):
        # Discretize 1st dimension into 10 bins
        bins = np.digitize(embedding[:, 0], np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), 10))
        return NMI(labels, bins)

    print("\n" + "="*60)
    print(f"⚡️ Embedding Quality Metrics {title_suffix}")
    print(f"{'Embedding':<20} | {'NMI (Struct)':<12} | {'Trustworthiness':<15}")
    print("-" * 60)

    for name, emb in embeddings.items():
        # If embedding > 2D, take first 2 for plotting, but use full for metrics if appropriate
        plot_emb = emb[:, :2]
        
        nmi = get_nmi(plot_emb, data_y)
        # Trustworthiness is heavy, calc only for Z
        tw = trustworthiness(data_x, plot_emb, n_neighbors=15) if "PCA" not in name else 1.0
        print(f"{name:<20} | {nmi:.4f}       | {tw:.4f}")
    print("="*60 + "\n")

    # 4. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    cmap = plt.cm.get_cmap('tab10', len(np.unique(data_y)))

    for i, (name, emb) in enumerate(embeddings.items()):
        ax = axes[i]
        plot_emb = emb[:, :2]
        sc = ax.scatter(plot_emb[:, 0], plot_emb[:, 1], c=data_y, cmap=cmap, s=8, alpha=0.6)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([]) 
        
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    fig.colorbar(sc, cax=cbar_ax, label="True Cluster Label")
    plt.suptitle(f"Ambient Space Analysis {title_suffix}", fontsize=16)
    plt.show()
