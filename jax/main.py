#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

# jax.print_environment_info()
import time
from _utils import *
from data import SpiralDataHandler
from models import *
from analysis import *

# ## Stop on warnings
# import warnings
# warnings.filterwarnings("error")


# --- CONFIGURATION ---
TRAIN = True  # Toggle this to False to load instead of train
RUN_DIR = "" if TRAIN else "./"  # Specify a run directory to load, or leave empty to find latest
# ---------------------


#%%
# Hyperparameters
config = {
    "input_dim": 3,
    "latent_dim": 2,
    "lr": 1e-3,
    "batch_size": 64,
    "epochs": 1500,
    "val_every": 100,
    "num_points": 1500,
    "seed": 2018
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
# model = Autoencoder(config["input_dim"], config["latent_dim"], key)
model = TaylorAutoencoder(config["input_dim"], config["latent_dim"], key)

if TRAIN:
    run_dir = get_run_path()
    print(f"🚀 Starting Training. Run ID: {run_dir.name}")
    
    # Optimizer
    optimizer = optax.adam(config["lr"])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Loss Function
    def loss_fn(model, x, y):
        # x_hat, z = eqx.filter_vmap(model)(x)
        x_hat, z = model(x)
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
    
    start_time = time.time()
    for epoch in range(config["epochs"]):
        batch_losses = []
        for x, y in dm.get_iterator('train'):
            model, opt_state, loss = make_step(model, opt_state, x, y)
            batch_losses.append(loss)
        
        avg_train_loss = np.mean(batch_losses)
        train_loss_history.append(avg_train_loss)
        
        # Simple Validation Log
        if epoch % config["val_every"] == 0 or epoch == config["epochs"] - 1:
            # Full batch validation for simplicity
            x_val, y_val = dm.get_full_data('val')
            val_loss = loss_fn(model, x_val, y_val)
            val_loss_history.append(val_loss)
            print(f"Epoch {epoch:04d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print(f"⏱ Training completed in HH:MM:SS -> {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

    # Save
    history = {"train": train_loss_history, "val": val_loss_history}
    save_run(run_dir, model, config, history)
    
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


#%%
# --- Analysis (Runs for both Train and Load) ---
print("\n--- Running Analysis ---")
plot_loss_curves(train_loss_history, val_loss_history, run_dir, title="Autoencoder Loss Curve")

# 3D Plot of Data (Subset)
print("Plotting Data Distribution...")
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
X_sample, Y_sample = dm.get_full_data('train')
# Subsample for plotting speed
num_plot_points = min(5000, len(X_sample))
idx = np.random.choice(len(X_sample), num_plot_points, replace=False)
scatter = ax.scatter(X_sample[idx, 0], X_sample[idx, 1], X_sample[idx, 2], 
                    c=Y_sample[idx], cmap='tab10', s=2, alpha=0.6)
ax.legend(*scatter.legend_elements(), title="Cluster ID")
plt.title("Input Data Distribution")
plt.draw()
plot_dir = run_dir / "plots"
fig.savefig(plot_dir / "data_distribution.png", bbox_inches='tight', dpi=100)

# Ambient Space Analysis
ambiant_space_analysis(model, dm, run_folder=run_dir)

# Latent Analysis
# latent_space_analysis(model, dm, title_suffix=f"(Latent Dim {config['latent_dim']})")
