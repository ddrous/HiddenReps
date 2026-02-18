#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set visual style
sns.set_theme(style="white")
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

# --- 1. Utilities for Sudoku Logic ---

def get_sudoku_coords():
    """Generates normalized grid coordinates."""
    ticks = jnp.linspace(-1, 1, 9)
    y, x = jnp.meshgrid(ticks, ticks)
    # Shape: (81, 2)
    return jnp.stack([x.ravel(), y.ravel()], axis=-1)

def block_reshape(board_probs):
    """
    Reshapes (9, 9, ...) grid into (9 blocks, 9 cells, ...).
    Used to calculate sub-grid constraints efficiently.
    """
    # 1. Split into (3, 3) blocks of (3, 3) cells
    # Shape becomes: (3 rows of blocks, 3 cols of blocks, 3 rows of cells, 3 cols of cells, classes)
    sh = board_probs.shape
    reshaped = board_probs.reshape(3, 3, 3, 3, sh[-1])
    # 2. Transpose to align blocks: (3 block_rows, 3 block_cols, 3 cell_rows, 3 cell_cols)
    transposed = reshaped.transpose(0, 2, 1, 3, 4)
    # 3. Flatten to (9 blocks, 9 cells in block, classes)
    return transposed.reshape(9, 9, sh[-1])

def generate_valid_puzzle():
    """
    Returns a valid solved board and a masked puzzle version.
    Hardcoded sample to ensure solvability for the demo.
    """
    # A valid solved board
    solution = jnp.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ]) - 1 # Adjust to 0-8 internally, we will shift back for display

    # Create a mask (0 = empty, 1 = clue)
    # Let's keep ~40 clues to make it an "Easy/Medium" puzzle for the INR
    key = jax.random.PRNGKey(402)
    mask_prob = jax.random.uniform(key, (9, 9))
    mask = jnp.where(mask_prob > 0.5, 1, 0)
    
    # The initial puzzle state. Unmasked cells are set to a distinct "Empty" value.
    # We use value 9 to represent "Empty" in the raw data, 
    # but the model will output 10 logits (0-8 are numbers, 9 is empty).
    puzzle = jnp.where(mask, solution, 9) 
    
    return solution, puzzle, mask

# --- 2. The Model ---

class SudokuINR(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=2,      
            out_size=10,    # Output: Digits 0-8 (indices) + 1 Empty Class (index 9)
            width_size=256, # Wider to handle the logic
            depth=4,        # Deeper for reasoning
            activation=jax.nn.gelu,
            key=key
        )

    def __call__(self, x):
        return self.mlp(x)

# --- 3. Phase 1: Initialization (Overfitting) ---

@eqx.filter_jit
def pretrain_loss(model, coords, target_labels):
    # Standard Classification: Match the board provided (including Empty cells)
    logits = jax.vmap(model)(coords)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target_labels))

def train_initial_state(model, coords, puzzle_flat, steps=2000, lr=1e-3):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(pretrain_loss)(model, coords, puzzle_flat)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    print("Phase 1: Memorizing Initial Board...")
    for i in range(steps):
        model, opt_state, loss = step(model, opt_state)
        if i % 500 == 0:
            print(f"  Iter {i} | Init Loss: {loss:.4f}")
    return model

# --- 4. Phase 2: Energy-Based Constraint Satisfaction ---

@eqx.filter_jit
def energy_function(model, coords, puzzle_clues, clue_mask):
    """
    Calculates the 'Energy' of the board. 0 Energy = Solved Sudoku.
    """
    logits = jax.vmap(model)(coords)
    # Shape: (81, 10) -> (9, 9, 10)
    # We treat index 9 as "Empty". Indices 0-8 are the numbers 1-9.
    probs = jax.nn.softmax(logits, axis=-1).reshape(9, 9, 10)
    
    # 1. Separate "Number" probabilities from "Empty" probabilities
    number_probs = probs[..., :9] # The probability of being 1-9
    empty_probs = probs[..., 9]   # The probability of being Empty
    
    # --- Constraint A: Clue Preservation ---
    # If a cell is a clue, the probability of that number must be 1.
    # We use cross-entropy against the fixed clues.
    # Clues are 0-8. 
    clue_indices = puzzle_clues.reshape(9, 9)
    # We only care where mask == 1
    # Gather the log-prob of the correct digit at clue locations
    log_probs = jax.nn.log_softmax(logits).reshape(9, 9, 10)
    clue_loss = -jnp.sum(
        jax.vmap(lambda lp, t, m: jnp.where(m, lp[t], 0.0))(
            log_probs.reshape(-1, 10), 
            puzzle_clues, 
            clue_mask.ravel()
        )
    ) / (jnp.sum(clue_mask) + 1e-6)

    # --- Constraint B: Must not be empty ---
    # We want empty_probs to go to 0 everywhere.
    emptiness_energy = jnp.mean(empty_probs)

    # --- Constraint C: Logical Constraints (Rows, Cols, Boxes) ---
    # For a valid Sudoku, the sum of probabilities for a specific digit k 
    # across a whole row/col/box must equal 1.
    
    # Sum of probabilities for each digit 0-8 across rows (axis 1)
    # row_sums shape: (9 rows, 9 digits)
    row_sums = jnp.sum(number_probs, axis=1) 
    row_energy = jnp.mean((row_sums - 1.0)**2)

    # Sum across cols (axis 0)
    col_sums = jnp.sum(number_probs, axis=0)
    col_energy = jnp.mean((col_sums - 1.0)**2)

    # Sum across boxes
    blocks = block_reshape(number_probs) # (9 blocks, 9 cells, 9 digits)
    box_sums = jnp.sum(blocks, axis=1)
    box_energy = jnp.mean((box_sums - 1.0)**2)
    
    # --- Constraint D: Discretization / Entropy ---
    # We want the model to be confident (probabilities close to 0 or 1), not fuzzy.
    entropy = -jnp.mean(jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1))

    # Weighted Sum of Energies
    total_energy = (
        10.0 * clue_loss +       # Don't forget the clues!
        5.0 * emptiness_energy + # Fill the empty spots!
        2.0 * (row_energy + col_energy + box_energy) + # Follow rules!
        0.05 * entropy           # Be decisive!
    )
    
    return total_energy

def solve_with_energy(model, coords, puzzle_flat, mask, steps=15000, lr=1e-4):
    # We use a lower LR here to gently slide down the energy manifold
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    history = []
    snapshots = []
    
    # Save Initial State
    logits = jax.vmap(model)(coords)
    snapshots.append(jnp.argmax(logits, axis=-1).reshape(9, 9))

    @eqx.filter_jit
    def step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(energy_function)(model, coords, puzzle_flat, mask)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    print("Phase 2: Solving via Energy Minimization...")
    for i in range(steps):
        model, opt_state, loss = step(model, opt_state)
        history.append(loss)
        
        if i % 5000 == 0:
            print(f"  Iter {i} | Energy: {loss:.6f}")
            # Capture intermediate state
            logits = jax.vmap(model)(coords)
            snapshots.append(jnp.argmax(logits, axis=-1).reshape(9, 9))

    # Final state
    logits = jax.vmap(model)(coords)
    snapshots.append(jnp.argmax(logits, axis=-1).reshape(9, 9))
    
    return model, history, snapshots

#%%

# --- 5. Main Execution ---

# Setup
key = jax.random.PRNGKey(101)
coords = get_sudoku_coords()
solution, puzzle, mask = generate_valid_puzzle()
puzzle_flat = puzzle.ravel().astype(jnp.int32)

# Initialize Model
model = SudokuINR(key)

# 1. Overfit to initial state (with empty cells)
model = train_initial_state(model, coords, puzzle_flat)

# 2. Solve using Energy
model, history, snapshots = solve_with_energy(model, coords, puzzle_flat, mask)


#%%
# --- 6. Visualization ---

def plot_solution_evolution(snapshots, history, mask):
    n_snaps = len(snapshots)
    fig = plt.figure(figsize=(20, 8))
    
    # Plot Energy Loss
    ax_loss = plt.subplot2grid((2, n_snaps), (1, 0), colspan=n_snaps)
    ax_loss.plot(history, color='#e74c3c', lw=2)
    ax_loss.set_yscale('log')
    ax_loss.set_title("System Energy (Loss) over Time")
    ax_loss.set_xlabel("Optimization Steps")
    ax_loss.set_ylabel("Energy")
    ax_loss.grid(True, alpha=0.3)

    # Plot Snapshots
    for i, board in enumerate(snapshots):
        ax = plt.subplot2grid((2, n_snaps), (0, i))
        
        # Prepare display board: 
        # Indices 0-8 -> Digits 1-9
        # Index 9 -> Empty (display as 0 or empty)
        display_board = jnp.where(board == 9, 0, board + 1)
        
        # Visual trick: Dim the clues, highlight the AI's guesses
        # We can use the mask to color code.
        
        sns.heatmap(display_board, annot=True, fmt="d", cbar=False, ax=ax,
                    cmap="Blues", square=True, linewidths=1, linecolor='black',
                    annot_kws={"size": 10})
        
        step_label = "Initialization" if i == 0 else f"Step {(i-1)*1000}" if i < n_snaps-1 else "Final"
        ax.set_title(step_label)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

plot_solution_evolution(snapshots, history, mask)