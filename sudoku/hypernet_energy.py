#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import seaborn as sns
from jax.flatten_util import ravel_pytree
import time

# Visualization Settings
sns.set_theme(style="white", context="paper")
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

# --- 1. Domain Logic: Sudoku Constraints ---

def get_coords():
    """Generates normalized grid coordinates (81, 2)."""
    ticks = jnp.linspace(-1, 1, 9)
    y, x = jnp.meshgrid(ticks, ticks)
    return jnp.stack([x.ravel(), y.ravel()], axis=-1)

def generate_random_batch(key, batch_size=1):
    """
    Generates a random valid sudoku board, then masks it.
    Returns: puzzle (masked board), mask (1 where clue exists).
    """
    # A base solved board
    base = jnp.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [3, 4, 5, 6, 7, 8, 0, 1, 2],
        [6, 7, 8, 0, 1, 2, 3, 4, 5],
        [1, 2, 0, 4, 5, 3, 7, 8, 6],
        [4, 5, 3, 7, 8, 6, 1, 2, 0],
        [7, 8, 6, 1, 2, 0, 4, 5, 3],
        [2, 0, 1, 5, 3, 4, 8, 6, 7],
        [5, 3, 4, 8, 6, 7, 2, 0, 1],
        [8, 6, 7, 2, 0, 1, 5, 3, 4]
    ])

    def get_single(k):
        k1, k2 = jax.random.split(k)
        # Random permutation (shift) to create variety
        shift = jax.random.randint(k1, (), 0, 9)
        board = (base + shift) % 9
        
        # Create mask (keep ~30-40 clues)
        mask_prob = jax.random.uniform(k2, (9, 9))
        mask = jnp.where(mask_prob > 0.6, 1, 0)
        
        # 9 represents 'Empty'
        puzzle = jnp.where(mask, board, 9)
        return puzzle, mask

    keys = jax.random.split(key, batch_size)
    return jax.vmap(get_single)(keys)

def compute_sudoku_validity(logits):
    """
    The Unsupervised Loss. Checks if the board satisfies Sudoku rules.
    Input: Logits (9, 9, 10)
    """
    probs = jax.nn.softmax(logits, axis=-1)
    
    # 1. Number Probabilities (indices 0-8)
    num_probs = probs[..., :9] 
    
    # Constraint A: Row/Col Uniqueness (Sum of prob for digit k in row/col == 1)
    row_loss = jnp.mean((jnp.sum(num_probs, axis=1) - 1.0)**2)
    col_loss = jnp.mean((jnp.sum(num_probs, axis=0) - 1.0)**2)
    
    # Constraint B: Box Uniqueness
    # Reshape (9,9) -> (3,3,3,3) -> transpose -> (9 blocks, 9 cells)
    blocks = num_probs.reshape(3, 3, 3, 3, 9).transpose(0, 2, 1, 3, 4).reshape(9, 9, 9)
    box_loss = jnp.mean((jnp.sum(blocks, axis=1) - 1.0)**2)
    
    # Constraint C: Emptiness (Index 9 should have 0 prob)
    empty_loss = jnp.mean(probs[..., 9]**2)
    
    return row_loss + col_loss + box_loss + empty_loss

# --- 2. The Models ---

class SudokuINR(eqx.Module):
    layers: list

    def __init__(self, key):
        # Keeping it small so the HyperNetwork can handle the parameter vector
        # Input (2) -> Hidden (32) -> Hidden (32) -> Output (10)
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(2, 32, key=k1),
            jax.nn.gelu,
            eqx.nn.Linear(32, 32, key=k2),
            jax.nn.gelu,
            eqx.nn.Linear(32, 10, key=k3) 
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EnergyHyperNet(eqx.Module):
    """
    Takes the flattened parameters of the INR and outputs a scalar energy.
    Input Size: Calculated dynamically based on INR size.
    """
    net: eqx.nn.MLP

    def __init__(self, in_size, key):
        self.net = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=128,
            depth=3,
            activation=jax.nn.softplus, # Smooth energy surface
            key=key
        )

    def __call__(self, flat_params):
        return jnp.squeeze(self.net(flat_params))

# --- 3. Phase 1: Initialization (Overfitting Clues) ---

@eqx.filter_jit
def init_loss(inr, coords, target_board):
    logits = jax.vmap(inr)(coords)
    # Standard Classification on the initial state (learns numbers + empty)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target_board.ravel()))

def get_initial_inr(key, puzzle_board):
    """
    Returns an INR overfitted to the initial puzzle state (clues + holes).
    """
    inr = SudokuINR(key)
    coords = get_coords()
    
    # --- FIX START: Partition the model ---
    # We split the model into learnable arrays (params) and static structure (static)
    # 'static' contains the 'gelu' function causing the error.
    params, static = eqx.partition(inr, eqx.is_array)
    
    # Optimizer only works on params
    optim = optax.adam(1e-2)
    opt_state = optim.init(params)
    
    def step(carry, _):
        params_curr, state = carry
        
        # Recombine to form the full model for the forward pass
        model = eqx.combine(params_curr, static)
        
        # Calculate loss & grads
        loss, grads = eqx.filter_value_and_grad(init_loss)(model, coords, puzzle_board)
        
        # Update only the parameters
        updates, state = optim.update(grads, state, params_curr)
        new_params = eqx.apply_updates(params_curr, updates)
        
        return (new_params, state), loss

    # We scan ONLY over the params and opt_state (which are arrays)
    (final_params, _), _ = jax.lax.scan(step, (params, opt_state), None, length=500)
    
    # Recombine one last time to return the full model object
    return eqx.combine(final_params, static)

# --- 4. Phase 2: Bi-Level Optimization ---

def flatten_inr(inr):
    params, static = eqx.partition(inr, eqx.is_array)
    flat, unflatten_fn = ravel_pytree(params)
    return flat, unflatten_fn, static

@eqx.filter_jit
def inner_loop_inference(inr_init, energy_model, coords, puzzle_indices, mask, steps=20):
    """
    The 'Solver'. 
    Minimizes E(theta) starting from theta_0.
    """
    # We must differentiate wrt the parameters of INR, but use EnergyModel fixed.
    
    # 1. Flatten the initial INR to get our starting point in parameter space
    params_init, static = eqx.partition(inr_init, eqx.is_array)
    flat_init, unflatten_fn = ravel_pytree(params_init)
    
    # Hyperparameters for the inner optimization
    inner_lr = 0.01

    def solve_step(flat_curr, _):
        # Reconstruct INR to compute Clue Loss (Anchoring)
        # We need the Energy Model to solve the logic, but we anchor 
        # the model to the clues using standard Cross Entropy so it doesn't drift.
        
        def loss_fn(f):
            # 1. Energy from HyperNet
            energy = energy_model(f)
            
            # 2. Clue Anchor (Hard constraint helper)
            # Re-hydrate model to run forward pass
            p = unflatten_fn(f)
            model = eqx.combine(p, static)
            logits = jax.vmap(model)(coords)
            
            # Calculate Cross Entropy on Clues only
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, puzzle_indices.ravel())
            clue_loss = jnp.mean(ce * mask.ravel())
            
            # Total Inner Loss
            return energy + 50.0 * clue_loss

        grads = jax.grad(loss_fn)(flat_curr)
        # Gradient Descent on the parameters
        flat_new = flat_curr - inner_lr * grads
        return flat_new, None

    # Run the inner loop (differentiable scan)
    flat_final, _ = jax.lax.scan(solve_step, flat_init, None, length=steps)
    
    # Reconstruct the final model
    params_final = unflatten_fn(flat_final)
    inr_final = eqx.combine(params_final, static)
    
    return inr_final

@eqx.filter_jit
def outer_loop_loss(energy_model, inr_init, coords, puzzle_indices, mask):
    """
    The 'Teacher'.
    1. Runs the solver (inner loop).
    2. Checks if the resulting board is a valid Sudoku.
    """
    # 1. Solve using current Energy Model
    inr_solved = inner_loop_inference(inr_init, energy_model, coords, puzzle_indices, mask)
    
    # 2. Decode the result
    logits = jax.vmap(inr_solved)(coords).reshape(9, 9, 10)
    
    # 3. Compute Unsupervised Constraint Violation
    validity_loss = compute_sudoku_validity(logits)
    
    return validity_loss

def train_hypernetwork(steps=1000):
    key = jax.random.PRNGKey(4205)
    m_key, t_key = jax.random.split(key)
    
    # 1. Setup Models
    dummy_inr = SudokuINR(key)
    flat, _, _ = flatten_inr(dummy_inr)
    in_size = flat.shape[0]
    print(f"INR Parameter Size: {in_size}")
    
    energy_model = EnergyHyperNet(in_size, m_key)
    
    # 2. Optimizer for the Hypernetwork
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(energy_model, eqx.is_array))
    
    coords = get_coords()
    history = []
    
    print("Training HyperNetwork (Energy Model)...")
    for i in range(steps):
        t_key, subkey = jax.random.split(t_key)
        
        # A. Sample Random Batch
        puzzles, masks = generate_random_batch(subkey, batch_size=1)
        puzzle, mask = puzzles[0], masks[0]
        
        # B. Initialize INR (Phase 1 - Memorization)
        # We stop gradients here; we only want to optimize the path FROM init TO solution
        subkey, init_key = jax.random.split(subkey)
        inr_init = get_initial_inr(init_key, puzzle)
        
        # C. Update Energy Model (Phase 2 - Meta Update)
        loss, grads = eqx.filter_value_and_grad(outer_loop_loss)(
            energy_model, inr_init, coords, puzzle, mask
        )
        
        updates, opt_state = optimizer.update(grads, opt_state, energy_model)
        energy_model = eqx.apply_updates(energy_model, updates)
        
        history.append(loss)
        if i % 500 == 0:
            # print(f"Step {i:4d} | Validation Loss: {loss:.5f}")
            print(f"Time {time.strftime('%H:%M:%S')} | Step {i:4d} | Validation Loss: {loss:.5f}")
            
    return energy_model, history

# --- 5. Execution ---

energy_model, train_loss = train_hypernetwork(steps=5000)


#%%

# --- 6. Inference Visualization ---

def run_inference_demo(energy_model):
    # 1. New Board
    key = jax.random.PRNGKey(99)
    puzzle, mask = generate_random_batch(key)[0], generate_random_batch(key)[1][0]
    coords = get_coords()
    
    # 2. Init INR
    print("\nInference: Initializing INR on new board...")
    inr = get_initial_inr(jax.random.PRNGKey(123), puzzle)
    
    # 3. Iterative Solving (Visualizing steps)
    params, static = eqx.partition(inr, eqx.is_array)
    flat, unflatten_fn = ravel_pytree(params)
    
    snapshots = []
    energy_vals = []
    
    # Record Initial State
    logits = jax.vmap(inr)(coords)
    snapshots.append(jnp.argmax(logits, axis=-1).reshape(9,9))
    
    # Optimization loop (Manual SGD using the trained Energy Model)
    lr = 0.01
    current_flat = flat
    
    print("Inference: Minimizing Energy...")
    for k in range(200): # More steps for inference
        
        # Calculate Energy and Gradients
        def energy_fn(f):
            # Energy
            E = energy_model(f)
            # Anchor
            p = unflatten_fn(f)
            m = eqx.combine(p, static)
            l = jax.vmap(m)(coords)
            ce = optax.softmax_cross_entropy_with_integer_labels(l, puzzle.ravel())
            anchor = jnp.mean(ce * mask.ravel())
            return E + 50.0 * anchor # High anchor weight to keep clues locked

        val, grads = jax.value_and_grad(energy_fn)(current_flat)
        current_flat = current_flat - lr * grads
        energy_vals.append(val)
        
        # Snapshot every 50 steps
        if k % 50 == 0:
            p = unflatten_fn(current_flat)
            m = eqx.combine(p, static)
            l = jax.vmap(m)(coords)
            snapshots.append(jnp.argmax(l, axis=-1).reshape(9,9))
            
    # Final Snapshot
    p = unflatten_fn(current_flat)
    m = eqx.combine(p, static)
    l = jax.vmap(m)(coords)
    snapshots.append(jnp.argmax(l, axis=-1).reshape(9,9))
    
    return train_loss, energy_vals, snapshots, puzzle

t_loss, i_energy, snaps, puzzle = run_inference_demo(energy_model)

# --- 7. Plotting Utilities ---

def plot_losses(train_loss, infer_energy):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Training
    ax1.plot(train_loss, color='black', alpha=0.7)
    ax1.set_title("Training: Constraint Violation (Outer Loop)")
    ax1.set_xlabel("Meta-Steps")
    ax1.set_ylabel("Sudoku Invalidity")
    ax1.grid(True, alpha=0.3)
    
    # Inference
    ax2.plot(infer_energy, color='#e74c3c')
    ax2.set_title("Inference: Energy Descent (Inner Loop)")
    ax2.set_xlabel("Optimization Steps")
    ax2.set_ylabel("Predicted Energy")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_snapshots(snapshots, puzzle):
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3.5))
    
    for i, ax in enumerate(axes):
        board = snapshots[i]
        
        # Prepare for display: 
        # Model uses 0-8 for numbers, 9 for Empty.
        # We want to display 1-9 for numbers, 0 for Empty.
        display = jnp.where(board == 9, 0, board + 1)
        
        # Color map setup
        sns.heatmap(display, annot=True, fmt="d", cbar=False, ax=ax,
                    cmap="Blues", square=True, linewidths=1, linecolor='black',
                    vmin=0, vmax=9, annot_kws={"size": 9})
        
        if i == 0: title = "Start (Overfitted)"
        elif i == n-1: title = "Final (Solved)"
        else: title = f"Step {i*50}"
        
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

plot_losses(t_loss, i_energy)
plot_snapshots(snaps, puzzle)
