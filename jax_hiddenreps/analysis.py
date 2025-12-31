import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, trustworthiness
from umap import UMAP
from sklearn.metrics import normalized_mutual_info_score as NMI
import equinox as eqx
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.spatial import procrustes

def plot_loss_curves(train_loss, val_loss, run_folder, title="Loss Curve"):
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
    plt.draw()
    plot_dir = run_folder / "plots"
    plt.savefig(plot_dir / "loss_curve.png", bbox_inches='tight', dpi=100)

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
        # return eqx.filter_vmap(m.encode)(x)
        return m.encode(x)
        
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



def ambiant_space_analysis(model, data_handler, run_folder, title_suffix=""):
    """
    Plots PCA, Kernel PCA, t-SNE, and UMAP of the original data side-by-side.
    For the autoencoder, uses the latent space. Also plots True Latents if available.

    Layout (2x3):
    [Autoencoder] [PCA]       [Kernel PCA]
    [True Latents][t-SNE]     [UMAP]
    """
    print("Collecting data for analysis...")
    X_jax, Y_jax = data_handler.get_full_data(split="train")

    @eqx.filter_jit
    def get_latent(m, x):
        return m.encode(x)

    Z_jax = get_latent(model, X_jax)

    data_x = np.array(X_jax)
    data_z = np.array(Z_jax)
    data_y = np.array(Y_jax)

    # Check for true latents early so we can use them in plots
    has_true_latents = hasattr(data_handler, "compute_true_latents")
    Z_true = None
    if has_true_latents:
        Z_true = np.array(data_handler.compute_true_latents(split="train"))

    print(f"Computing embeddings for {len(data_x)} points...")

    # ---- Embeddings ----
    x_pca = PCA(n_components=2).fit_transform(data_x)

    # NEW: Kernel PCA (Non-linear)
    print("Running Kernel PCA...")
    try:
        # Using RBF kernel for non-linearity. 
        # Note: This can be slow on very large datasets (>20k samples)
        x_kpca = KernelPCA(n_components=2, kernel='rbf').fit_transform(data_x)
    except Exception as e:
        print(f"Skipping Kernel PCA due to error: {e}")
        x_kpca = np.zeros_like(x_pca)

    print("Running t-SNE...")
    x_tsne = TSNE(
        n_components=2, perplexity=30, init="pca",
        learning_rate="auto", random_state=42
    ).fit_transform(data_x)

    print("Running UMAP...")
    x_umap = UMAP(
        n_components=2, n_neighbors=15,
        min_dist=0.1, random_state=42
    ).fit_transform(data_x)

    # Add KPCA to embeddings dict so metrics are calculated for it too
    embeddings = {
        "AE (Latents)": data_z,
        "PCA": x_pca,
        "Kernel PCA": x_kpca, # Added here
        "t-SNE": x_tsne,
        "UMAP": x_umap,
    }

    artefacts_dir = run_folder / "artefacts"
    artefacts_dir.mkdir(exist_ok=True, parents=True)
    np.save(
        artefacts_dir / f"ambient_space_embeddings{title_suffix.replace(' ', '_')}.npy",
        embeddings
    )

    # ---- Structural Metrics (always computed) ----
    def get_nmi(embedding, labels):
        bins = np.digitize(
            embedding[:, 0],
            np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), 10)
        )
        return NMI(labels, bins)

    print("\n" + "=" * 80)
    print(f"⚡️ Embedding Quality Metrics {title_suffix}")
    header = f"{'Embedding':<22} | {'NMI':<8} | {'Trust':<8}"
    print(header, end="")

    if has_true_latents:
        print(" | Proc-MSE ↓ | Max|ρ| ↑ | Mean|ρ| ↑ | R² ↑")
    else:
        print()

    print("-" * 80)

    # ---- Latent-aware helpers ----
    def standardize(Z):
        return StandardScaler().fit_transform(Z)

    def latent_metrics(Z, Zt):
        Zs = standardize(Z)
        Ts = standardize(Zt)

        _, Zp, Tp = procrustes(Ts, Zs)
        mse = np.mean((Zp - Tp) ** 2)

        corr = np.abs(np.corrcoef(Zs.T, Ts.T)[:2, 2:])
        max_corr = corr.max()
        mean_corr = corr.mean()

        r2 = LinearRegression().fit(Zs, Ts).score(Zs, Ts)

        return mse, max_corr, mean_corr, r2

    results = {}

    for name, emb in embeddings.items():
        plot_emb = emb[:, :2]

        nmi = get_nmi(plot_emb, data_y)
        # Trustworthiness is slow; skip for linear PCA to save time, or keep as is
        tw = trustworthiness(data_x, plot_emb, n_neighbors=15) if name != "PCA" else 1.0

        line = f"{name:<22} | {nmi:8.4f} | {tw:8.4f}"

        if has_true_latents and Z_true is not None:
            mse, max_corr, mean_corr, r2 = latent_metrics(plot_emb, Z_true)
            line += (
                f" | {mse:10.3e}"
                f" | {max_corr:8.3f}"
                f" | {mean_corr:9.3f}"
                f" | {r2:6.3f}"
            )

            results[name] = dict(
                nmi=nmi,
                trust=tw,
                procrustes_mse=mse,
                max_corr=max_corr,
                mean_corr=mean_corr,
                r2=r2,
            )
        else:
            results[name] = dict(nmi=nmi, trust=tw)

        print(line)

    print("=" * 80 + "\n")

    np.save(
        artefacts_dir / f"ambient_space_metrics{title_suffix.replace(' ', '_')}.npy",
        results
    )

    # ---- Visualization (Updated to 2x3 Grid) ----
    # 2 rows, 3 columns. Wider figure.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10)) 
    
    # We map specific data to specific axes to satisfy the layout constraints
    # Layout:
    # (0,0) AE       (0,1) PCA       (0,2) Kernel PCA
    # (1,0) True Z   (1,1) t-SNE     (1,2) UMAP
    
    # Define the plot list: (Row, Col, Name, Data)
    plots_config = [
        (0, 0, "AE (Latents)", data_z),
        (0, 1, "PCA", x_pca),
        (0, 2, "Non-linear PCA (RBF)", x_kpca),
        (1, 0, "True Latents", Z_true if Z_true is not None else np.zeros((1,2))), # Handle missing Z_true safely
        (1, 1, "t-SNE", x_tsne),
        (1, 2, "UMAP", x_umap)
    ]

    cmap = plt.cm.get_cmap("tab10", len(np.unique(data_y)))

    for row, col, name, emb in plots_config:
        ax = axes[row, col]
        
        # If True Latents are missing, just hide axis or show text
        if name == "True Latents" and (Z_true is None):
            ax.text(0.5, 0.5, "True Latents Not Available", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        sc = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=data_y, cmap=cmap, s=8, alpha=0.6
        )
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    # Colorbar positioning adjusted for 2x3
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="True Cluster Label")

    plt.suptitle(f"Ambient Space Analysis {title_suffix}", fontsize=16)

    plt.draw()
    plot_dir = run_folder / "plots"
    fig.savefig(
        plot_dir / f"ambient_space_analysis{title_suffix.replace(' ', '_')}.png",
        bbox_inches="tight", dpi=100
    )

    return results
