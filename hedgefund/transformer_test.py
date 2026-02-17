#%%
## Open and visualize the sine tiny dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

train_data = np.load('sine_data/small/train.npy')
test_data = np.load('sine_data/small/test.npy')

print(train_data.shape)

plt.plot(train_data[0])
plt.plot(train_data[1])
plt.title("Time series in sine dataset")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.show()

# %%


class PositionalEncoding(eqx.Module):
    pe: jax.Array

    def __init__(self, d_model: int, max_len: int = 500):
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / d_model))
        
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe

    def __call__(self, x):
        return x + self.pe[:x.shape[0], :]

class TransformerBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    norm2: eqx.nn.LayerNorm
    
    def __init__(self, d_model, n_heads, d_ff, dropout, key):
        k1, k2 = jax.random.split(key)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=n_heads, 
            query_size=d_model, 
            use_query_bias=True, 
            use_key_bias=True, 
            use_value_bias=True, 
            use_output_bias=True, 
            dropout_p=dropout, 
            key=k1
        )
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.mlp = eqx.nn.MLP(
            in_size=d_model, 
            out_size=d_model, 
            width_size=d_ff, 
            depth=1, 
            activation=jax.nn.gelu, 
            key=k2
        )
        self.norm2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, mask=None, key=None):
        attn_out = self.attention(x, x, x, mask=mask, key=key)
        x = eqx.filter_vmap(self.norm1)(x + attn_out)
        mlp_out = jax.vmap(self.mlp)(x)                   ## TODO: we don't need MLPs at all ?
        x = eqx.filter_vmap(self.norm2)(x + mlp_out)
        return x


## TODO: This is the weighttranformer we want. Traditional attention and residual stream !
class Transformer(eqx.Module):
    embedding: eqx.nn.Linear
    pos_encoder: PositionalEncoding
    blocks: list
    norm_final: eqx.nn.LayerNorm  # Added final normalization
    output_projection: eqx.nn.Linear
    
    d_model: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    
    def __init__(self, input_dim, d_model, n_heads, n_layers, d_ff, max_len, key):
        self.d_model = d_model
        self.n_layers = n_layers
        
        k_emb, k_out = jax.random.split(key)
        k_layers = jax.random.split(key, n_layers)
        
        self.embedding = eqx.nn.Linear(input_dim, d_model, key=k_emb)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout=0.0, key=k)
            for k in k_layers
        ]
        
        self.norm_final = eqx.nn.LayerNorm(d_model)
        
        # --- CRITICAL FIX: Zero Initialization for Output Projection ---
        # This ensures the model initially predicts delta approx 0.
        self.output_projection = eqx.nn.Linear(d_model, input_dim, key=k_out)
        
        # Manually set weights and bias to zero
        zeros_w = jnp.zeros_like(self.output_projection.weight)
        zeros_b = jnp.zeros_like(self.output_projection.bias)
        self.output_projection = eqx.tree_at(
            lambda l: (l.weight, l.bias), 
            self.output_projection, 
            (zeros_w, zeros_b)
        )

    def make_causal_mask(self, seq_len):
        idx = jnp.arange(seq_len)
        mask = idx[:, None] >= idx[None, :]
        return mask

    def __call__(self, x0, steps, key=None):
        traj_buffer = jnp.zeros((steps, x0.shape[0]))
        traj_buffer = traj_buffer.at[0].set(x0)
        full_mask = self.make_causal_mask(steps)
        
        def scan_step(carry, step_idx):
            current_traj = carry
            
            # 1. Embedding scaled by sqrt(d_model) for stability
            x = jax.vmap(self.embedding)(current_traj) * jnp.sqrt(self.d_model)
            x = self.pos_encoder(x)
            
            # 2. Transformer Blocks
            for block in self.blocks:
                x = block(x, mask=full_mask)
            
            # 3. Final Norm (Standard in Pre-Norm architectures)
            x = jax.vmap(self.norm_final)(x)
                
            # 4. Extract last valid hidden state
            last_hidden = x[step_idx]
            
            # 5. Predict Delta
            delta = self.output_projection(last_hidden)
            
            # 6. Update
            next_weight = current_traj[step_idx] + delta
            # next_weight = delta
            
            write_idx = jnp.minimum(step_idx + 1, steps - 1)
            new_traj = current_traj.at[write_idx].set(next_weight)
            return new_traj, next_weight

        step_indices = jnp.arange(steps)
        final_traj, _ = jax.lax.scan(scan_step, traj_buffer, step_indices)
        return final_traj


# ## TODO: This is the new Transformer with micro-steps, like a Neural ODE. We repeatedly refine updates of the state
# class Transformer(eqx.Module):
#     embedding: eqx.nn.Linear
#     pos_encoder: PositionalEncoding
#     blocks: list
#     norm_final: eqx.nn.LayerNorm
    
#     # Replaces simple output_projection
#     refinement_mlp: eqx.nn.MLP
    
#     d_model: int = eqx.field(static=True)
#     n_layers: int = eqx.field(static=True)
#     n_substeps: int = eqx.field(static=True)  # New Hyperparam
    
#     def __init__(self, input_dim, d_model, n_heads, n_layers, d_ff, max_len, n_substeps, key):
#         self.d_model = d_model
#         self.n_layers = n_layers
#         self.n_substeps = n_substeps
        
#         k_emb, k_refine = jax.random.split(key)
#         k_layers = jax.random.split(key, n_layers)
        
#         # 1. Embedding
#         self.embedding = eqx.nn.Linear(input_dim, d_model, key=k_emb)
#         self.pos_encoder = PositionalEncoding(d_model, max_len)
        
#         # 2. Transformer Blocks
#         self.blocks = [
#             TransformerBlock(d_model, n_heads, d_ff, dropout=0.0, key=k)
#             for k in k_layers
#         ]
        
#         self.norm_final = eqx.nn.LayerNorm(d_model)
        
#         # 3. Refinement MLP (The "Solver")
#         # Input: d_model (Context) + d_model (Current Weight Embedding)
#         # We perform a simple addition of context + state, so input size is d_model.
#         self.refinement_mlp = eqx.nn.MLP(
#             in_size=d_model*2,
#             out_size=input_dim,
#             width_size=d_model * 2, # Slightly wider hidden layer
#             # width_size=(2*d_model + input_dim)//2, # Slightly wider hidden layer
#             depth=1,                # 1 hidden layer is usually enough for local steps
#             activation=jax.nn.gelu,
#             key=k_refine
#         )
        
#         # --- STABILITY FIX: Zero Init Last Layer of MLP ---
#         # This ensures the refinement starts as "Identity" (delta=0)
#         zeros_w = jnp.zeros_like(self.refinement_mlp.layers[-1].weight)
#         zeros_b = jnp.zeros_like(self.refinement_mlp.layers[-1].bias)
#         self.refinement_mlp = eqx.tree_at(
#             lambda m: (m.layers[-1].weight, m.layers[-1].bias),
#             self.refinement_mlp,
#             (zeros_w, zeros_b)
#         )

#     def make_causal_mask(self, seq_len):
#         idx = jnp.arange(seq_len)
#         mask = idx[:, None] >= idx[None, :]
#         return mask

#     def make_onestep_mask(self, seq_len):
#         """ Mask that allows each position to see only itself and the previous position"""
#         idx = jnp.arange(seq_len)
#         mask = (idx[:, None] - idx[None, :]) <= 1
#         return mask
#     def make_nsteps_mask(self, seq_len, n_steps):
#         idx = jnp.arange(seq_len)
#         mask = (idx[:, None] - idx[None, :]) <= n_steps
#         return mask

#     def __call__(self, x0, steps, key=None):
#         traj_buffer = jnp.zeros((steps, x0.shape[0]))
#         traj_buffer = traj_buffer.at[0].set(x0)
#         # full_mask = self.make_causal_mask(steps)
#         # full_mask = self.make_onestep_mask(steps)
#         full_mask = self.make_nsteps_mask(steps, n_steps=2)
#         # full_mask = self.make_nsteps_mask(steps, n_steps=20)
#         # full_mask = self.make_nsteps_mask(steps, n_steps=CONFIG["transformer_target_step"])

#         def scan_step(carry, step_idx):
#             current_traj = carry
            
#             # --- PHASE 1: MACRO STEP (Transformer) ---
#             # Look at history x_0 ... x_t to decide "Global Direction"
            
#             # Embed & Scale
#             x = jax.vmap(self.embedding)(current_traj) * jnp.sqrt(self.d_model)
#             x = self.pos_encoder(x)
            
#             # Apply Blocks
#             for block in self.blocks:
#                 x = block(x, mask=full_mask)
#             x = jax.vmap(self.norm_final)(x)
            
#             # Get the "Context/Instruction" for the current step
#             context_h = x[step_idx] # Shape (d_model,)
            
#             # --- PHASE 2: MICRO STEPS (Iterative Refinement) ---
#             # Evolve x_t -> x_t+1 using the context as a guide
            
#             start_weight = current_traj[step_idx]
            
#             def refinement_loop(i, curr_weight):
#                 # 1. Embed the intermediate weight to latent space
#                 # We reuse the main embedding to share the "coordinate system"
#                 w_emb = self.embedding(curr_weight) * jnp.sqrt(self.d_model)
                
#                 # 2. Condition: Combine "Where we are" (w_emb) + "Where to go" (context_h)
#                 # Adding them is a standard ResNet-like conditioning strategy
#                 # combined_input = w_emb + context_h
#                 combined_input = jnp.concatenate([w_emb, context_h])
                
#                 # 3. Predict Micro-Delta
#                 micro_delta = self.refinement_mlp(combined_input)
                
#                 # 4. Update
#                 return curr_weight + micro_delta

#             # Run the loop n_substeps times
#             next_weight = jax.lax.fori_loop(0, self.n_substeps, refinement_loop, start_weight)
            
#             # --- WRITE BACK ---
#             write_idx = jnp.minimum(step_idx + 1, steps - 1)
#             new_traj = current_traj.at[write_idx].set(next_weight)
            
#             # We return next_weight just for scan compliance, though we use new_traj
#             return new_traj, next_weight

#         step_indices = jnp.arange(steps)
#         final_traj, _ = jax.lax.scan(scan_step, traj_buffer, step_indices)
        
#         return final_traj


#%% Train the model, and predict and visualize

# The transformer takes a sequence of shape (16, 1) and predicts the same shape. But this is a initial value problem, so the input is only the first element, with the remaining 15 steps filled with zeros. We predict the rest autoregressively.

model = Transformer(
    input_dim=1,
    d_model=32,
    n_heads=4,
    n_layers=2,
    d_ff=128,
    max_len=100,
    key=jax.random.PRNGKey(0)
)


# Test forward pass
# x0 = pad_with_zeros(train_data[0], total_length=16)  # Shape (90, 1)
x0 = train_data[0,0]  # (1,)
print("Input shape:", x0.shape)
predicted_traj = model(x0, steps=16, key=jax.random.PRNGKey(1))  # Shape (90, 1)
print("Predicted trajectory shape:", predicted_traj.shape)

## Now train the model to minimize MSE over the predicted trajectory

## Fully functional training loop
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

def train_step(model, x0, true_traj, opt_state, key):
    def loss_fn(model, x0, true_traj, key):
        pred_traj = eqx.filter_vmap(model, in_axes=(0, None, None))(x0, true_traj.shape[1], key)
        # pred_traj = model(x0, steps=true_traj.shape[0], key=key)
        return jnp.mean((pred_traj - true_traj) ** 2)
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x0, true_traj, key)
    # model = optimizer.update(grads, model)
    updaes, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updaes)
    return model, opt_state, loss


num_epochs = 200
batch_size = 1024
losses = []

for epoch in range(num_epochs):
    # Simple batching
    perm = np.random.permutation(train_data.shape[0])
    for i in range(0, train_data.shape[0], batch_size):
        batch_indices = perm[i:i+batch_size]
        batch_x0 = train_data[batch_indices, 0]  # Initial values
        batch_true_traj = train_data[batch_indices]  # Full trajectories
        
        # for j in range(batch_size):
        #     key, subkey = jax.random.split(jax.random.PRNGKey(epoch * batch_size + i + j))
        #     model, opt_state, loss = train_step(model, batch_x0[j], batch_true_traj[j], opt_state, subkey)
        #     losses.append(loss)

        key, subkey = jax.random.split(jax.random.PRNGKey(epoch*batch_size + i))
        model, opt_state, loss = train_step(model, batch_x0, batch_true_traj, opt_state, subkey)
        losses.append(loss)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

#%%
# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title("Training Loss over Steps")
plt.xlabel("Training Steps")
plt.ylabel("MSE Loss")
plt.yscale("log")
plt.show()


#%%
# Visualize test set predictions after training
for i in range(5):
    x0_test = test_data[i, 0]  # Initial value
    true_traj = test_data[i]   # True trajectory
    
    pred_traj = model(x0_test, steps=true_traj.shape[0], key=jax.random.PRNGKey(42))
    
    plt.figure(figsize=(10, 6))
    plt.plot(true_traj, label="True", marker='o')
    plt.plot(pred_traj, label="Predicted", marker='x')
    plt.title(f"Test Sample {i} Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()