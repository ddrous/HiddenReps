#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Import your local modules
from _utils import *
from data import SpiralDataHandler
from models import Autoencoder, TaylorAutoencoder
from analysis import ambiant_space_analysis

sns.set(style="whitegrid") # Changed to whitegrid for better chart readability

# --- CONFIGURATION ---
SAMPLE_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] # Define your range of data sizes
MODEL_TYPES = [
    ("Standard AE", Autoencoder),
    ("Taylor AE", TaylorAutoencoder)
]

# Fixed Hyperparameters
CONFIG = {
    "input_dim": 3,
    "latent_dim": 2,
    "lr": 1e-3,
    "batch_size": 64,
    "epochs": 1500,
    "val_every": 200,
    "seed": 2024
}

# ---------------------

def train_and_evaluate(num_points, ModelClass, run_dir, model_name):
    """
    Trains a specific model on a specific dataset size and returns the metrics.
    """
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {model_name} | Data Size: {num_points}")
    print(f"{'='*60}")

    key = jax.random.PRNGKey(CONFIG["seed"])
    
    # 1. Init Data
    dm = SpiralDataHandler(
        batch_size=CONFIG["batch_size"],
        num_points=num_points,
        val_split=0.2,
        num_clusters=10,
        cluster_spread=0.3,
        seed=CONFIG["seed"]
    )
    
    # 2. Init Model
    model = ModelClass(CONFIG["input_dim"], CONFIG["latent_dim"], key)
    
    # 3. Setup Optimizer & Loss
    optimizer = optax.adam(CONFIG["lr"])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def loss_fn(m, x, y):
        x_hat, z = m(x)
        return jnp.mean((x_hat - x) ** 2)

    @eqx.filter_jit
    def make_step(m, opt_state, x, y):
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(m, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, m)
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss_value

    # 4. Train
    start_time = time.time()
    for epoch in range(CONFIG["epochs"]):
        batch_losses = []
        for x, y in dm.get_iterator('train'):
            model, opt_state, loss = make_step(model, opt_state, x, y)
            batch_losses.append(loss)
        
        # Optional: Print less frequently to keep output clean
        if (epoch % CONFIG["val_every"] == 0) or (epoch == CONFIG["epochs"] - 1):
             avg_loss = np.mean(batch_losses)
             # print(f"   Epoch {epoch:04d} | Loss: {avg_loss:.5f}")

    print(f"   ⏱ Training finished in {time.time() - start_time:.2f}s")

    ## Save the run, overwritting what was there
    save_run(run_dir, model, None, None)

    # 5. Analyze
    # We pass a title_suffix to identify this run in the saved artifacts
    suffix = f"({model_name}_{num_points})"
    
    # !! Important: ambiant_space_analysis must return the 'results' dictionary !!
    # This assumes the modification from the previous step was applied.
    results = ambiant_space_analysis(model, dm, run_folder=run_dir, title_suffix=suffix)
    
    # Extract the specific metrics for the Autoencoder's Latent Space
    # The dictionary key for the AE latents is typically "AE (Latents)"
    ae_metrics = results.get("AE (Latents)", {})

    return ae_metrics


# --- MAIN EXECUTION ---

# 1. Create Shared Directory
run_dir = get_run_path()
print(f"📂 Global Experiment Directory: {run_dir}")

# Storage for plotting later
# Structure: { "Standard AE": { "nmi": [v1, v2..], "max_corr": [v1, v2..] }, ... }
experiment_data = {name: {"sizes": [], "nmi": [], "max_corr": [], "mean_corr": [], "r2": []} 
                   for name, _ in MODEL_TYPES}

# 2. Run Loops
for num_points in SAMPLE_SIZES:
    for model_name, ModelClass in MODEL_TYPES:
        
        # Run the experiment
        metrics = train_and_evaluate(num_points, ModelClass, run_dir, model_name)
        
        # Store results
        data_bucket = experiment_data[model_name]
        data_bucket["sizes"].append(num_points)
        
        # Safely get metrics (default to 0 if missing)
        data_bucket["nmi"].append(metrics.get("nmi", 0))
        data_bucket["max_corr"].append(metrics.get("max_corr", 0))
        data_bucket["mean_corr"].append(metrics.get("mean_corr", 0))
        data_bucket["r2"].append(metrics.get("r2", 0))

# 3. Plotting the Comparative Results
print("\n--- Generating Comparative Plots ---")

metrics_to_plot = [
    ("nmi", "NMI Score", "Normalized Mutual Information (Higher is Better)"),
    ("max_corr", "Max Correlation |ρ|", "Max Absolute Correlation (Higher is Better)"),
    ("mean_corr", "Mean Correlation |ρ|", "Mean Absolute Correlation (Higher is Better)"),
    ("r2", "Linear R²", "R² Score to True Latents (Higher is Better)"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (metric_key, ylabel, title) in enumerate(metrics_to_plot):
    ax = axes[i]
    
    for model_name, data in experiment_data.items():
        if len(data["sizes"]) > 0:
            ax.plot(data["sizes"], data[metric_key], marker='o', linewidth=2, label=model_name)
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Training Points")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

plt.suptitle(f"Model Performance vs Data Size", fontsize=16)
plt.tight_layout()

# Save final comparison
save_path = run_dir / "experiment_comparison.png"
plt.draw()
plt.savefig(save_path, dpi=120)
print(f"✅ Comparison plot saved to: {save_path}")
