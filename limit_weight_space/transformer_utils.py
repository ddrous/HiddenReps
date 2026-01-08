import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional

class PositionalEncoding(eqx.Module):
    pe: jax.Array

    def __init__(self, d_model: int, max_len: int = 500):
        # Standard Sinusoidal Positional Encoding
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / d_model))
        
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe

    def __call__(self, x):
        # x shape: (Seq_Len, D_Model)
        # Returns: (Seq_Len, D_Model)
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
        # x: (Seq_Len, D_Model)
        # Attention needs inputs (Q, K, V). For self-attention, all are x.
        
        # 1. Self Attention with Residual + Norm
        # EQX MHA expects query shape (Seq, D) if batch_first=False? 
        # Actually EQX MHA by default treats inputs as (Seq_Len, ...).
        attn_out = self.attention(x, x, x, mask=mask, key=key)
        x = self.norm1(x + attn_out)
        
        # 2. MLP with Residual + Norm
        mlp_out = jax.vmap(self.mlp)(x)
        x = self.norm2(x + mlp_out)
        return x

class WeightTransformer(eqx.Module):
    embedding: eqx.nn.Linear
    pos_encoder: PositionalEncoding
    blocks: list
    output_projection: eqx.nn.Linear
    
    d_model: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    
    def __init__(self, input_dim, d_model, n_heads, n_layers, d_ff, max_len, key):
        self.d_model = d_model
        self.n_layers = n_layers
        
        k_emb, k_out = jax.random.split(key)
        k_layers = jax.random.split(key, n_layers)
        
        # Project raw weights (Dimension D) to Model Dimension (D_Model)
        self.embedding = eqx.nn.Linear(input_dim, d_model, key=k_emb)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout=0.0, key=k)
            for k in k_layers
        ]
        
        # Project back to Weight Space
        self.output_projection = eqx.nn.Linear(d_model, input_dim, key=k_out)

    def make_causal_mask(self, seq_len):
        # Returns mask shape (Seq, Seq)
        # 1s where attention is allowed, 0s where blocked.
        # But Equinox MHA mask expects True/False or +/- infinity logits.
        # We generally use a mask where True = Keep, False = Drop.
        idx = jnp.arange(seq_len)
        mask = idx[:, None] >= idx[None, :]
        return mask

    def __call__(self, x0, steps, key=None):
        # x0: Initial weight vector (Input_Dim,)
        # steps: Total steps to generate
        
        # Initialize trajectory buffer
        # We start with [x0, 0, 0, ...]
        # We maintain the buffer to re-run the transformer at every step.
        
        # Shape: (Steps, Input_Dim)
        # We pre-allocate the buffer.
        traj_buffer = jnp.zeros((steps, x0.shape[0]))
        traj_buffer = traj_buffer.at[0].set(x0)
        
        def scan_step(carry, step_idx):
            # carry is the current full trajectory buffer
            current_traj = carry
            
            # 1. Prepare inputs for Transformer
            # We take the *entire* buffer, but we mask out the future positions effectively
            # by slicing or masking. Here we slice for efficiency.
            # We only really need inputs 0...step_idx
            
            # Slice: (Step_Idx + 1, Input_Dim)
            valid_inputs = jax.lax.dynamic_slice_in_dim(current_traj, 0, step_idx + 1, axis=0)
            
            # 2. Embedding & Positional Encoding
            x = jax.vmap(self.embedding)(valid_inputs) # (T, D_Model)
            x = self.pos_encoder(x)
            
            # 3. Create Causal Mask for this slice
            seq_len = x.shape[0]
            mask = self.make_causal_mask(seq_len)
            
            # 4. Pass through blocks
            for block in self.blocks:
                x = block(x, mask=mask)
                
            # 5. Get prediction for the NEXT step
            # The Transformer outputs a sequence. The output at position `t` 
            # corresponds to the prediction for `t+1`.
            # We want the LAST output from the current sequence.
            last_hidden = x[-1]
            
            # Residual connection: The Transformer predicts the *delta* or the next state?
            # Let's assume it predicts the *delta* for stability, or the next raw state.
            # Here, let's predict the delta to add to the current state.
            delta = self.output_projection(last_hidden)
            next_weight = valid_inputs[-1] + delta
            # OR direct prediction:
            # next_weight = self.output_projection(last_hidden)
            
            # Update buffer
            # We place next_weight at index `step_idx + 1`
            # Note: We must use a safe index to avoid OOB in the last scan iteration
            write_idx = jnp.minimum(step_idx + 1, steps - 1)
            new_traj = current_traj.at[write_idx].set(next_weight)
            
            return new_traj, next_weight

        # We run from t=0 to t=steps-1
        # scan returns: (final_carry, stacked_outputs)
        step_indices = jnp.arange(steps)
        final_traj, _ = jax.lax.scan(scan_step, traj_buffer, step_indices)
        
        return final_traj