import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import json
import shutil
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

# jax.print_environment_info()

from data import *
from models import *
from analysis import *

# --- CONFIGURATION ---
TRAIN = True  # Toggle this to False to load instead of train
RUN_DIR = "" if TRAIN else "./"  # Specify a run directory to load, or leave empty to find latest
# ---------------------

def get_run_path():
    """Generates a run path based on timestamp inside ../runs"""
    # Assuming script is in `project/src`, runs will be in `project/runs`
    timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    base_dir = Path("../runs")
    run_dir = base_dir / timestamp
    return run_dir

def find_latest_run():
    """Finds the most recent folder in ../runs"""
    base_dir = Path("../runs")
    if not base_dir.exists():
        raise FileNotFoundError("No runs folder found.")
    runs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError("No run directories found.")
    return max(runs, key=os.path.getmtime)

def save_experiment(run_dir, model, config, history):
    """Saves model, config, and history."""
    run_dir.mkdir(parents=True, exist_ok=True)
    

    # 0. Save data.py, models.py, analysis.py, and main.py for reproducibility
    shutil.copy(__file__, run_dir / "main.py")
    shutil.copy(Path(__file__).parent / "data.py", run_dir / "data.py")
    shutil.copy(Path(__file__).parent / "models.py", run_dir / "models.py")
    shutil.copy(Path(__file__).parent / "analysis.py", run_dir / "analysis.py")

    # 1. Save Model (Equinox way)
    eqx.tree_serialise_leaves(run_dir / "model.eqx", model)
    
    # 2. Save Config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # 3. Save History
    np.save(run_dir / "loss_history.npy", history)
    print(f"✅ Experiment saved to {run_dir}")

def load_experiment(run_dir, model_skeleton):
    """Loads model and history."""
    print(f"📂 Loading experiment from {run_dir}...")
    
    # 1. Load Model
    model = eqx.tree_deserialise_leaves(run_dir / "model.eqx", model_skeleton)
    
    # 2. Load History
    history = np.load(run_dir / "loss_history.npy", allow_pickle=True).item()
    
    return model, history

def main():
    # Hyperparameters
    config = {
        "input_dim": 3,
        "latent_dim": 2,
        "lr": 1e-3,
        "batch_size": 2048,
        "epochs": 10,
        "num_points": 15000,
        "seed": 2027
    }
    
    # JAX Key
    key = jax.random.PRNGKey(config["seed"])
    
    # Init Data
    print("--- Initializing Data ---")
    dm = SpiralDataHandler(
        batch_size=config["batch_size"],
        num_points=config["num_points"],
        val_split=0.2,
        num_clusters=10,
        cluster_spread=0.3,
        seed=config["seed"]
    )
    
    # Init Model (Skeleton needed for loading too)
    model = Autoencoder(config["input_dim"], config["latent_dim"], key)

    if TRAIN:
        run_dir = get_run_path()
        print(f"🚀 Starting Training. Run ID: {run_dir.name}")
        
        # Optimizer
        optimizer = optax.adam(config["lr"])
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        # Loss Function
        def loss_fn(model, x, y):
            x_hat, z = eqx.filter_vmap(model)(x)
            return jnp.mean((x_hat - x) ** 2)

        # Update Step
        @eqx.filter_jit
        def make_step(model, opt_state, x, y):
            loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value

        # Training Loop
        train_loss_history = []
        val_loss_history = []
        
        for epoch in range(config["epochs"]):
            batch_losses = []
            for x, y in dm.get_iterator('train'):
                model, opt_state, loss = make_step(model, opt_state, x, y)
                batch_losses.append(loss)
            
            avg_train_loss = np.mean(batch_losses)
            train_loss_history.append(avg_train_loss)
            
            # Simple Validation Log
            if epoch % 10 == 0 or epoch == config["epochs"] - 1:
                # Full batch validation for simplicity
                x_val, y_val = dm.get_full_data('val')
                val_loss = loss_fn(model, x_val, y_val)
                val_loss_history.append(val_loss)
                print(f"Epoch {epoch:04d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save
        history = {"train": train_loss_history, "val": val_loss_history}
        save_experiment(run_dir, model, config, history)
        
    else:
        # LOAD MODE
        try:
            ## Only find the latest run if run_dir empty
            if RUN_DIR == "":
                print("🔍 No run directory provided. Finding latest run...")
                run_dir = find_latest_run()
            else:
                run_dir = Path(RUN_DIR)
            model, history = load_experiment(run_dir, model)
            train_loss_history = history['train']
            val_loss_history = history['val']
        except FileNotFoundError as e:
            print(e)
            return

    # --- Analysis (Runs for both Train and Load) ---
    print("\n--- Running Analysis ---")
    plot_loss_curves(train_loss_history, val_loss_history)
    
    # 3D Plot of Data (Subset)
    print("Plotting Data Distribution...")
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    X_sample, Y_sample = dm.get_full_data('train')
    # Subsample for plotting speed
    idx = np.random.choice(len(X_sample), 5000, replace=False)
    scatter = ax.scatter(X_sample[idx, 0], X_sample[idx, 1], X_sample[idx, 2], 
                        c=Y_sample[idx], cmap='tab10', s=2, alpha=0.6)
    ax.legend(*scatter.legend_elements(), title="Cluster ID")
    plt.title("Input Data Distribution")
    plt.show()

    # Ambient Space Analysis
    ambiant_space_analysis(model, dm)

    # Latent Analysis
    # latent_space_analysis(model, dm, title_suffix=f"(Latent Dim {config['latent_dim']})")

if __name__ == "__main__":
    main()