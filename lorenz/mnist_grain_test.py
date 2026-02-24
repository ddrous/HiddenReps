#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import grain.python as grain
import numpy as np
from typing import Iterator, Dict, Any

# Set JAX to use GPU
jax.config.update('jax_platform_name', 'gpu')
print(f"Using device: {jax.devices()[0]}")

# ==================== GRAIN DATA PIPELINE ====================

class SineDataSource(grain.RandomAccessDataSource):
    """Custom Grain data source that generates sine wave data on the fly"""
    
    def __init__(self, n_samples: int, noise_scale: float = 0.1, seed: int = 42):
        self.n_samples = n_samples
        self.noise_scale = noise_scale
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Generate a single sample with numpy"""
        # Use a deterministic pattern based on idx
        x = (idx / self.n_samples) * 6 - 3  # Range [-3, 3]
        
        # Add some randomness per sample
        x += self.rng.uniform(-0.1, 0.1)
        
        # Generate y = sin(x) + noise
        y = np.sin(x)
        y += self.rng.normal(0, self.noise_scale)
        
        return {
            'x': np.array([x], dtype=np.float32),
            'y': np.array([y], dtype=np.float32)
        }

# Define a transformation function instead of a class
def numpy_to_jax(data):
    """Convert numpy arrays to JAX arrays"""
    if isinstance(data, dict):
        return {
            'x': jnp.array(data['x']),
            'y': jnp.array(data['y'])
        }
    return data

def create_grain_dataloader(
    n_samples: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42
) -> grain.DataLoader:
    """Create a Grain dataloader with proper GPU transfer pipeline"""
    
    # 1. Create data source
    source = SineDataSource(n_samples, seed=seed)
    
    # 2. Create sampler
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=1,
        shuffle=shuffle,
        seed=seed
    )
    
    # 3. Create transformations - using functions directly
    transformations = [
        grain.Batch(batch_size=batch_size, drop_remainder=True),
        numpy_to_jax,  # Just pass the function directly
    ]
    
    # 4. Create and return the dataloader
    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=transformations,
        worker_count=0  # Start with 0 workers for debugging
    )
    
    return dataloader

# ==================== MODEL DEFINITION ====================

class SineRegressor(eqx.Module):
    """Simple MLP for sine regression"""
    layers: list
    
    def __init__(self, key):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(1, 64, key=keys[0]),
            eqx.nn.Linear(64, 64, key=keys[1]),
            eqx.nn.Linear(64, 1, key=keys[2])
        ]
    
    def __call__(self, x):
        x = self.layers[0](x)
        x = jnp.tanh(x)
        x = self.layers[1](x)
        x = jnp.tanh(x)
        x = self.layers[2](x)
        return x

# ==================== TRAINING LOOP ====================

@eqx.filter_jit
def train_step(model, batch, optimizer, opt_state):
    """Single training step with JIT compilation"""
    
    def loss_fn(model):
        pred = jax.vmap(model)(batch['x'])
        return jnp.mean((pred - batch['y']) ** 2)
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss

def main():
    # Hyperparameters
    train_samples = 100000
    batch_size = 6400
    n_epochs = 3
    learning_rate = 0.001
    
    print("Creating Grain dataloader...")
    train_loader = create_grain_dataloader(
        n_samples=train_samples,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    # Test the dataloader first
    print("Testing dataloader...")
    test_batch = next(iter(train_loader))
    print(f"Test batch type: {type(test_batch)}")
    print(f"Test batch keys: {test_batch.keys() if isinstance(test_batch, dict) else 'Not a dict'}")
    print(f"Test batch x shape: {test_batch['x'].shape if isinstance(test_batch, dict) else 'N/A'}")
    
    # Initialize model
    key = jax.random.PRNGKey(42)
    model = SineRegressor(key)
    
    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Training metrics
    losses = []
    
    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"Total batches per epoch: {train_samples // batch_size}")
    
    # Training loop
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # # Recreate dataloader for each epoch
        # train_loader = create_grain_dataloader(
        #     n_samples=train_samples,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     seed=epoch + 42
        # )
        
        for step, batch in enumerate(train_loader):
            # Convert batch to JAX if it's not already (safety check)
            if not isinstance(batch['x'], jnp.ndarray):
                batch = {
                    'x': jnp.array(batch['x']),
                    'y': jnp.array(batch['y'])
                }
            
            model, opt_state, loss = train_step(model, batch, optimizer, opt_state)
            
            epoch_losses.append(float(loss))
            
            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss:.6f}")
            
            # Limit steps for faster testing
            if step >= 100:
                break
        
        avg_epoch_loss = np.mean(epoch_losses)
        losses.extend(epoch_losses)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.6f}")
    
    # Evaluation
    print("\nEvaluating model...")
    
    # Generate test data
    x_test = jnp.linspace(-3, 3, 500).reshape(-1, 1)
    y_pred = jax.vmap(model)(x_test)
    y_true = jnp.sin(x_test)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Learned function
    plt.subplot(1, 2, 1)
    plt.plot(x_test, y_true, 'g-', label='True sine', linewidth=2)
    plt.plot(x_test, y_pred, 'r--', label='Model prediction', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sine Regression Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training loss
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Model device: {model.layers[0].weight.devices()}")

if __name__ == "__main__":
    main()