#%%

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="white")
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

# 1. Define the INR Model (MLP as a function of coordinates)
class SudokuINR(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=2,      # Input: (x, y) coordinates
            out_size=9,     # Output: Logits for digits 1-9
            width_size=128, # Slightly wider for cleaner convergence
            depth=3,
            activation=jax.nn.relu,
            key=key
        )

    def __call__(self, x):
        return self.mlp(x)

# 2. Setup Data
def get_sudoku_data(key):
    # Generating a random target board (values 0-8 for indexing)
    target_board = jax.random.randint(key, (9, 9), 0, 9)
    # Normalized coordinates from -1 to 1
    ticks = jnp.linspace(-1, 1, 9)
    # Note: meshgrid 'xy' indexing to match row/col logic
    y, x = jnp.meshgrid(ticks, ticks)
    coords = jnp.stack([x.ravel(), y.ravel()], axis=-1)
    targets = target_board.ravel()
    return target_board, coords, targets

# 3. Training Logic
@eqx.filter_jit
def loss_fn(model, coords, targets):
    logits = jax.vmap(model)(coords)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))

def train_inr(steps=10000, lr=1e-3):
    key = jax.random.PRNGKey(0)
    m_key, d_key = jax.random.split(key)
    
    model = SudokuINR(m_key)
    target_board, coords, targets = get_sudoku_data(d_key)
    
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    history = []

    @eqx.filter_jit
    def step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, coords, targets)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    print("Training INR...")
    for i in range(steps):
        model, opt_state, loss = step(model, opt_state)
        history.append(loss)
        if i % 2000 == 0:
            print(f"Iteration {i:4d} | Loss: {loss:.6f}")
            
    return model, target_board, coords, history

# Execute Training
model, target_board, coords, history = train_inr()




#%%
# 4. Comprehensive Visualization
def visualize_results(model, target_board, coords, history):
    # Get predictions
    logits = jax.vmap(model)(coords)
    probs = jax.nn.softmax(logits, axis=-1)
    preds = jnp.argmax(logits, axis=-1).reshape(9, 9)
    
    # Calculate confidence (max probability per cell)
    confidence = jnp.max(probs, axis=-1).reshape(9, 9)

    fig = plt.figure(figsize=(18, 5))
    
    # Plot 1: Training Loss
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(history, color='#2ecc71', lw=2)
    ax1.set_yscale('log')
    ax1.set_title("Optimization Convergence (Log Loss)")
    ax1.set_xlabel("Iterations")
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Plot 2: Target Board
    ax2 = fig.add_subplot(1, 3, 2)
    sns.heatmap(target_board, annot=target_board + 1, fmt="d", cmap="Greys", 
                cbar=False, ax=ax2, square=True, linewidths=.5, linecolor='black')
    ax2.set_title("Desired State (Ground Truth)")

    # Plot 3: INR Output + Confidence
    # We use the confidence as the background color to see where the MLP is "unsure"
    ax3 = fig.add_subplot(1, 3, 3)
    # sns.heatmap(confidence, annot=preds + 1, fmt="d", cmap="YlGnBu", 
    #             ax=ax3, square=True, linewidths=.5, linecolor='black')
    sns.heatmap(preds, annot=preds + 1, fmt="d", cmap="YlGnBu", 
                ax=ax3, square=True, linewidths=.5, linecolor='black')
    ax3.set_title("INR Predicted Digits")

    plt.tight_layout()
    plt.show()

visualize_results(model, target_board, coords, history)