import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set visual style
sns.set_theme(style="white", context="paper")
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

# --- 1. Sudoku Logic & Constraints (The "Ground Truth") ---

def get_coords():
    """Generates normalized grid coordinates (81, 2)."""
    ticks = jnp.linspace(-1, 1, 9)
    y, x = jnp.meshgrid(ticks, ticks)
    return jnp.stack([x.ravel(), y.ravel()], axis=-1)

def generate_random_board_batch(batch_size, key):
    """
    Generates a batch of solved boards, then masks them.
    We return the MASKED version. The solution is discarded (Unsupervised).
    """
    # Note: Generating valid Sudokus from scratch in pure JAX is complex.
    # For this demo, we permute a base solution to get random valid starting points.
    base_solution = jnp.array([
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
    
    def get_one(k):
        # Random permutations to create variety
        k1, k2, k3 = jax.random.split(k, 3)
        board = base_solution
        # Shift numbers
        board = (board + jax.random.randint(k1, (), 0, 9)) % 9
        
        # Create Mask (Approx 40 clues)
        mask = jax.random.bernoulli(k2, 0.45, (9, 9))
        
        # 10th class (index 9) is 'Empty'
        puzzle = jnp.where(mask, board, 9) 
        return puzzle, mask

    keys = jax.random.split(key, batch_size)
    return jax.vmap(get_one)(keys)

def compute_constraint_violation(board_probs):
    """
    The 'Physical Laws' of Sudoku. 
    Returns 0.0 if valid, >0.0 if broken.
    Input: (9, 9, 10) probabilities
    """
    # 1. Slice out the number probs (0-8)
    # We ignore the 'Empty' class (index 9) for row/col sums
    # because a solved board has no empty cells.
    num_probs = board_probs[..., :9] # (9, 9, 9)

    # Constraint A: Row Uniqueness (Sum of each digit 1-9 in a row must be 1)
    row_sums = jnp.sum(num_probs, axis=1) # (9 rows, 9 digits)
    loss_rows = jnp.mean((row_sums - 1.0)**2)

    # Constraint B: Col Uniqueness
    col_sums = jnp.sum(num_probs, axis=0) # (9 cols, 9 digits)
    loss_cols = jnp.mean((col_sums - 1.0)**2)

    # Constraint C: Box Uniqueness
    # Reshape to (3, 3, 3, 3, 9) -> (9 blocks, 9 cells, 9 digits)
    blocks = num_probs.reshape(3, 3, 3, 3, 9).transpose(0, 2, 1, 3, 4).reshape(9, 9, 9)
    box_sums = jnp.sum(blocks, axis=1)
    loss_boxes = jnp.mean((box_sums - 1.0)**2)

    # Constraint D: No Empty Cells Allowed in Final State
    # The probability of class 9 must be 0
    loss_empty = jnp.mean(board_probs[..., 9]**2)
    
    return loss_rows + loss_cols + loss_boxes + loss_empty

# --- 2. The Models ---

class BoardINR(eqx.Module):
    layers: list

    def __init__(self, key):
        # A simple MLP to represent the board
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(2, 64, key=k1),
            jax.nn.gelu,
            eqx.nn.Linear(64, 64, key=k2),
            jax.nn.gelu,
            eqx.nn.Linear(64, 10, key=k3) # 10 outputs (9 digits + 1 empty)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EnergyModel(eqx.Module):
    """
    The Learned Energy Function. 
    Input: The entire board state (probs).
    Output: Scalar Energy.
    """
    net: eqx.nn.MLP

    def __init__(self, key):
        self.net = eqx.nn.MLP(
            in_size=81 * 10, # Flattened board
            out_size=1,
            width_size=256,
            depth=2,
            activation=jax.nn.softplus, # Smooth, positive activation
            key=key
        )

    def __call__(self, board_probs):
        # board_probs: (9, 9, 10)
        flat = board_probs.ravel()
        return jnp.squeeze(self.net(flat))

# --- 3. Phase 1: Overfitting Init (The "Initialization") ---

@eqx.filter_jit
def overfit_loss(inr, coords, target_board_indices):
    logits = jax.vmap(inr)(coords)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target_board_indices))

def get_overfitted_inr(key, target_board, steps=100):
    """
    Creates an INR and trains it to memorize the 'target_board' (with empty cells).
    This runs 'outside' the main gradient tape for speed, usually.
    But JAX can differentiate through this if needed (we won't here).
    """
    inr = BoardINR(key)
    coords = get_coords()
    target_flat = target_board.ravel() # (81,)
    
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(eqx.filter(inr, eqx.is_array))

    def step(carry, _):
        inr, opt_state = carry
        loss, grads = eqx.filter_value_and_grad(overfit_loss)(inr, coords, target_flat)
        updates, opt_state = optimizer.update(grads, opt_state, inr)
        inr = eqx.apply_updates(inr, updates)
        return (inr, opt_state), loss

    (inr, _), _ = jax.lax.scan(step, (inr, opt_state), None, length=steps)
    return inr

# --- 4. Phase 2: Bi-Level Training (The "Meta-Loop") ---

@eqx.filter_jit
def inner_loop_solve(inr, energy_model, coords, clue_mask, puzzle_indices, steps=10):
    """
    Descend the learned Energy Landscape.
    This creates the path: theta_0 -> theta_1 -> ... -> theta_k
    """
    
    # We only want to update the INR here, Energy Model is frozen
    # Use simple SGD for the inner loop to maintain stable gradients for meta-opt
    lr = 0.05 
    
    def loss_fn(inr_curr):
        logits = jax.vmap(inr_curr)(coords)
        probs = jax.nn.softmax(logits, axis=-1).reshape(9, 9, 10)
        
        # 1. The Learned Energy
        energy = energy_model(probs)
        
        # 2. Hard Clue Constraint (Optional: Help the Energy Model)
        # We enforce that the INR shouldn't deviate from known clues.
        # Ideally the Energy Model learns this, but adding it explicitly
        # stabilizes training massively.
        # We mask the loss to only affect clue cells.
        ce_matrix = optax.softmax_cross_entropy_with_integer_labels(logits, puzzle_indices.ravel()).reshape(9,9)
        clue_loss = jnp.sum(ce_matrix * clue_mask)
        
        return energy + 10.0 * clue_loss

    def step(inr_curr, _):
        grads = eqx.filter_grad(loss_fn)(inr_curr)
        # Manual SGD update
        new_inr = jax.tree_map(lambda p, g: p - lr * g, inr_curr, grads)
        return new_inr, None

    final_inr, _ = jax.lax.scan(step, inr, None, length=steps)
    return final_inr

@eqx.filter_jit
def outer_loss_fn(energy_model, inr_init, coords, clue_mask, puzzle_indices):
    """
    1. Run Inner Loop (Solve using Energy Model)
    2. Check if the result satisfies Mathematical Constraints
    """
    # 1. Solve (Differentiable Unrolling)
    inr_final = inner_loop_solve(inr_init, energy_model, coords, clue_mask, puzzle_indices)
    
    # 2. Decode Final State
    logits = jax.vmap(inr_final)(coords)
    probs = jax.nn.softmax(logits, axis=-1).reshape(9, 9, 10)
    
    # 3. Compute Ground Truth Constraint Violation
    validity_loss = compute_constraint_violation(probs)
    
    return validity_loss

def train_energy_model(steps=2000):
    key = jax.random.PRNGKey(42)
    e_key, d_key = jax.random.split(key)
    
    # Initialize Energy Model
    energy_model = EnergyModel(e_key)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(energy_model, eqx.is_array))
    
    coords = get_coords()
    loss_history = []

    print("Training Energy Model (Unsupervised)...")
    
    for i in range(steps):
        # 1. Sample a random board
        d_key, subkey = jax.random.split(d_key)
        # Shape: (9,9)
        puzzle, mask = generate_random_board_batch(1, subkey)
        puzzle = puzzle[0] # remove batch dim
        mask = mask[0]
        
        # 2. Prepare Init INR (Phase 1) - No grad tracking needed here
        # We just need a starting point theta_0
        subkey, init_key = jax.random.split(subkey)
        inr_init = get_overfitted_inr(init_key, puzzle)
        
        # 3. Update Energy Model (Phase 2)
        # We want Energy Model to guide inr_init -> valid_solution
        grads = eqx.filter_grad(outer_loss_fn)(energy_model, inr_init, coords, mask, puzzle)
        updates, opt_state = optimizer.update(grads, opt_state, energy_model)
        energy_model = eqx.apply_updates(energy_model, updates)
        
        # Logging
        if i % 100 == 0:
            loss = outer_loss_fn(energy_model, inr_init, coords, mask, puzzle)
            loss_history.append(loss)
            print(f"Step {i:4d} | Constraint Violation: {loss:.5f}")

    return energy_model, loss_history

# --- 5. Execution ---
# Train the Hyper-Energy-Network
energy_model, training_losses = train_energy_model(steps=500) # Short run for demo

# --- 6. Inference Visualization ---

def inference_on_new_board(energy_model):
    # 1. Generate new random board
    key = jax.random.PRNGKey(999)
    puzzle, mask = generate_random_board_batch(1, key)
    puzzle = puzzle[0]
    mask = mask[0]
    coords = get_coords()
    
    # 2. Initialize INR (Overfit theta_0)
    print("Inference: Initializing INR...")
    inr = get_overfitted_inr(jax.random.PRNGKey(55), puzzle, steps=200)
    
    # 3. Run Optimization using the Trained Energy Model
    # We use more steps than training to fully converge
    print("Inference: Solving...")
    
    snapshots = []
    energy_vals = []
    
    optimizer = optax.adam(0.01) # Solver optimizer
    opt_state = optimizer.init(eqx.filter(inr, eqx.is_array))

    @eqx.filter_jit
    def solve_step(inr, opt_state):
        logits = jax.vmap(inr)(coords)
        probs = jax.nn.softmax(logits, axis=-1).reshape(9, 9, 10)
        
        # Loss = Learned Energy + Clue Anchoring
        def objective(m):
            l = jax.vmap(m)(coords)
            p = jax.nn.softmax(l, axis=-1).reshape(9, 9, 10)
            e = energy_model(p)
            
            # Keep clues fixed (Weighted penalty)
            ce = optax.softmax_cross_entropy_with_integer_labels(l, puzzle.ravel()).reshape(9,9)
            clue_p = jnp.sum(ce * mask)
            return e + 10.0 * clue_p

        loss, grads = eqx.filter_value_and_grad(objective)(inr)
        updates, opt_state = optimizer.update(grads, opt_state, inr)
        inr = eqx.apply_updates(inr, updates)
        return inr, opt_state, loss

    # Capture loop
    for k in range(300):
        inr, opt_state, loss = solve_step(inr, opt_state)
        energy_vals.append(loss)
        
        if k % 50 == 0 or k == 299:
            logits = jax.vmap(inr)(coords)
            # Take argmax to see the board state
            # 0-8 are numbers, 9 is Empty
            board_state = jnp.argmax(logits, axis=-1).reshape(9, 9)
            snapshots.append(board_state)

    return training_losses, energy_vals, snapshots, puzzle

# Run Inference
t_loss, i_energy, snaps, puzzle = inference_on_new_board(energy_model)

# --- 7. Plotting ---

def plot_training_loss(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, color='black', lw=2)
    plt.title("Meta-Training: Constraint Violation over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Violation Loss (0 = Valid Sudoku)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_inference(energy_trace, snapshots, initial_puzzle):
    # Plot 1: Energy Minimization
    plt.figure(figsize=(8, 4))
    plt.plot(energy_trace, color='#2980b9', lw=2)
    plt.title("Inference: Energy Minimization")
    plt.xlabel("Optimization Steps")
    plt.ylabel("Learned Energy")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot 2: Snapshots
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    
    for i, ax in enumerate(axes):
        board = snapshots[i]
        
        # Convert to display format: 
        # Internal 0-8 -> Display 1-9
        # Internal 9 (Empty) -> Display 0
        display = jnp.where(board == 9, 0, board + 1)
        
        # Highlight original clues in Bold logic (using mask logic visually)
        # For simplicity here we just show the numbers
        sns.heatmap(display, annot=True, fmt="d", cbar=False, ax=ax,
                    cmap="Blues", square=True, linewidths=1, linecolor='black',
                    vmin=0, vmax=9)
        
        title = "Initial State" if i == 0 else "Final State" if i == n-1 else f"Step {i*50}"
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

plot_training_loss(t_loss)
plot_inference(i_energy, snaps, puzzle)