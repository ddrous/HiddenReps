#%%
import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import equinox as eqx
import diffrax
import optax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import datetime
import shutil
import sys
import typing
from typing import Optional, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set plotting style
sns.set(style="whitegrid", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

## Do not plot the grid for cleanliness
plt.rcParams['axes.grid'] = False

## Jax config stop on NaNs
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)


# --- 1. CONFIGURATION ---
TRAIN = True  
RUN_DIR = "."

CONFIG = {
    # "seed": time.time_ns() % (2**32 - 1),
    "seed": 2028,

    "x_range": [-4.0, 4.0],  # Wider range to see the sine wave repeat
    "segments": 5,           # Split into 5 distinct vertical strips
    "train_seg_ids": [1, 2, 3], # Train on the middle

    # Data & MLP Hyperparameters
    "data_samples": 2000,
    "noise_std": 0.005,
    "segments": 11,
    "x_range": [0.175, 1.5],
    "train_seg_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8], 
    "width_size": 24,
    "mlp_batch_size": 64,

    # Expansion Hyperparameters
    "n_circles": 11,           
    "warmup_steps": 0,
    
    # --- TRANSFORMER HYPERPARAMETERS ---
    "lr": 5e-5,      
    "transformer_epochs": 70000,
    "print_every": 2500,
    "transformer_batch_size": 1,      

    # New Params
    "transformer_d_model": 128*4,    # Embedding Dimension
    "transformer_n_heads": 1,      # Number of Heads
    "transformer_n_layers": 1,     # Number of Transformer Blocks
    "transformer_d_ff": 1//1,       # Feedforward dimension inside block: TODO: not needed atm, see forward pass.
    "transformer_substeps": 50,     # Number of micro-steps per macro step
    "kl_weight": 1e-1,          # Weight on KL divergence loss

    "transformer_target_step": 60*2,    # Total steps to unroll
    "scheduled_loss_weight": False,

    ## Consistency Loss Config
    "n_synthetic_points": 512,
    "consistency_loss_weight": 0.0,

    # Regularization Config
    "regularization_step": 40*2,     
    "regularization_weight": 0.0,  

    # Data Selection Mode
    "data_selection": "annulus",        ## "annulus" or "full_disk"
    "final_step_mode": "none",          ## "full" or "circle_only"
}

print("Config seed is:", CONFIG["seed"])

#%%
# --- 2. UTILITY FUNCTIONS ---

#%%
# --- 3. DATA GENERATION ---

def gen_data(seed, n_samples, n_segments=3, x_range=[-3.0, 3.0], noise_std=0.1):
    """
    Generates a classical 1D-1D regression dataset: y = sin(3x) + 0.5x
    Segments are assigned based on spatial position (x-value), allowing
    for easy OOD splitting (e.g., train on segments [0,1], test on [2]).
    """
    np.random.seed(seed)
    
    # 1. Generate X uniformly across the full range
    x_min, x_max = x_range
    X = np.random.uniform(x_min, x_max, n_samples)
    
    # 2. Define the unchanging relation P(Y|X) (Concept)
    # y = sin(3x) + 0.5x is classic because it has both trend and periodicity
    Y = np.sin(10 * X) + 0.5 * X
    
    # Add noise
    noise = np.random.normal(0, noise_std, n_samples)
    Y += noise
    
    # 3. Create Segments spatially
    # We divide the x_range into n_segments equal distinct regions
    bins = np.linspace(x_min, x_max, n_segments + 1)
    
    # np.digitize returns indices 1..N, we want 0..N-1
    segs = np.digitize(X, bins) - 1
    
    # Clip to ensure bounds (in case of float precision issues at max edge)
    segs = np.clip(segs, 0, n_segments - 1)
    
    # 4. Format Output
    # data shape: (N, 2) -> [x, y]
    # segs shape: (N,)
    data = np.column_stack((X, Y))
    
    return data, segs


## Generate the dataset
data, segs = gen_data(
    seed=CONFIG["seed"],
    n_samples=CONFIG["data_samples"],
    n_segments=CONFIG["segments"],
    x_range=CONFIG["x_range"],
    noise_std=CONFIG["noise_std"]
)

#%%
# --- 4. DATA VISUALIZATION ---


#%%
## We want to plot all traning data in blue, and all OOD data in orange, to show the distribution shift.

## Increase x and y labels for the whole plot

plt.rcParams['axes.labelsize'] = 22

# plt.figure(figsize=(10, 4)) 
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharey=False)

plt.figure(figsize=(4, 10)) 
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(24, 5), sharey=False)

## Please do not plot the grid for cleanliness
plt.grid(False)



# Create a boolean mask for training segments
train_mask = np.isin(segs, CONFIG["train_seg_ids"])
train_data = data[train_mask]
test_data = data[~train_mask]

# Plot training data in blue
ax1.scatter(train_data[:, 0], train_data[:, 1], c='blue', alpha=0.7, edgecolor='k', label='Train')
# Plot OOD data in orange
ax1.scatter(test_data[:, 0], test_data[:, 1], c='orange', alpha=0.7, edgecolor='k', label='Test')

anchor_x = np.min(train_data[:, 0])
# dists = np.linalg.norm(test_data[:, 0] - anchor_x)
dists = np.abs(test_data[:, 0] - anchor_x)

# nb_circles = CONFIG["n_circles"]
nb_circles = 11
circle_radii = np.linspace(0, np.max(dists), nb_circles)

## Overlay the vertical bars correspoding to the radii of the circles in the expansion space, to show how the data is split into segments.
for r in circle_radii:
    ax1.axvline(x=anchor_x + r, color='gray', linestyle='--', alpha=0.5)


## At the top of the plot, we can anotate the rings R1, ... R_ncircles, to show how the data is split into segments.
for i, r in enumerate(circle_radii[:-1]):
    ax1.text(anchor_x + r +0.065, ax1.get_ylim()[1]+0.12, r'$R_{' + str(i+1) + '}$', color='gray', fontsize=14, ha='center', va='top')


## Mark the anchor point in huge red cross. Plot it exactly on the x axis, and annotate it as "Anchor Point"

## it has to appear on top of the horizontal x bar
# ax1.scatter(anchor_x, -1.25, c='red', marker='X', s=200, label='Anchor x={:.2f}'.format(anchor_x), edgecolor='k', zorder=5)

## The whole marker isn't visible because we are in sns talk mode, so we will just plot a vertical line to mark the anchor point, and annotate it as "Anchor Point"
ax1.axvline(x=anchor_x, color='red', linestyle='-', label='Anchor'+r' $x={:.2f}$'.format(anchor_x), zorder=5)



# ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
ax1.set_title("(a) Domain Decomposition", fontsize=26, loc='left')
ax1.set_xlabel(r"$x$", fontsize=18)
ax1.set_ylabel(r"$y$", fontsize=18)
ax1.legend(loc='lower right')  # Place legend above the plot
ax1.set_ylim(-1.25, 2)


## Below this, I want to plot the tangent of the sine wave at each point x, (plot the intercept of that tangent as well)
true_func = lambda x: np.sin(10 * x) + 0.5 * x
true_func_deriv = lambda x: 10 * np.cos(10 * x) + 0.5
x_dense = np.linspace(CONFIG["x_range"][0], CONFIG["x_range"][1], 1000)

y_dense = true_func(x_dense)
slope = true_func_deriv(x_dense)
## I want the slope of the line that goes from the anchor point to the test point, which is (y_test - y_anchor) / (x_test - x_anchor)
slope_anchor_to_test = (true_func(x_dense) - true_func(anchor_x)) / (x_dense - anchor_x)
## I want its intercept as well, which is y_anchor - slope * x_anchor
intercept_anchor_to_test = true_func(anchor_x) - slope_anchor_to_test * anchor_x

ax2.plot(x_dense, slope_anchor_to_test, label='Slope to Anchor', color='purple', linewidth=6)
ax2.plot(x_dense, intercept_anchor_to_test, label='Intercept to Anchor', color='brown', linewidth=6)
ax2.set_title("(b) Weight-Space Sequence Modelling", fontsize=26, loc='left')
ax2.set_xlabel("Time Step")
# ax2.set_ylabel(r"$\theta^d$")
ax2.set_ylabel("Weight Value")
ax2.legend(loc='lower right')  # Place legend above the plot
# ax2.set_ylim(-1.25, 3)

## Replace the x ticks with the indices of the corresponsing rings
ring_indices = np.arange(1, nb_circles + 1)
ring_positions = anchor_x + circle_radii
ax2.set_xticks(ring_positions)
# ax2.set_xticklabels([r'$t_{' + str(i) + '}$' for i in ring_indices])
ax2.set_xticklabels([r'$' + str(i) + '$' for i in ring_indices])

### Add sticks dashes on the x bar, which were removed becuase we are in sns talk mode, to show the positions of the rings more clearly
ax2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)


## Plot a vertical to mark the end of the training data (which corresponds approximately to x=1.225

# ax2.axvline(x=1.26, color='k', linestyle='-', label="End of Training Data")
ax2.legend(loc='lower right')  # Place legend above the plot

## Share the entire area after x=1.26 in grey to show that it is OOD test region
# ax2.axvspan(1.26, CONFIG["x_range"][1], color='gray', alpha=0.2, label="OoS Region")
ax2.legend(loc='lower right')  # Place legend above the plot

plt.tight_layout()



## Save as PDF
plots_path = Path(RUN_DIR) / "plots"
plots_path.mkdir(parents=True, exist_ok=True)
# plt.savefig(plots_path / "data_and_weight_space_for_paper.png", dpi=300, bbox_inches='tight')
plt.savefig(plots_path / "data_and_weight_space_for_paper_new.pdf", dpi=300, bbox_inches='tight')


