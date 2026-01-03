#%%

import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import seaborn as sns
sns.set(style="white", context="talk")
import time

# Use double precision for stability analysis
jax.config.update("jax_enable_x64", True)

# ==========================================
# 1. Data Generation (Damped Oscillator)
# ==========================================
def get_data(t_end=15, steps=100):
    # System: x'' + 0.5x' + x = 0 (Spirals to 0,0)
    # This is the "Ground Truth" physics we want to learn
    t_eval = jnp.linspace(0, t_end, steps)
    y0 = jnp.array([2.0, 1.0]) 
    
    def true_vector_field(t, y, args):
        pos, vel = y
        return jnp.array([vel, -0.5 * vel - pos])

    term = diffrax.ODETerm(true_vector_field)
    solver = diffrax.Tsit5()
    sol = diffrax.diffeqsolve(term, solver, t0=0, t1=t_end, dt0=0.01, y0=y0, saveat=diffrax.SaveAt(ts=t_eval))
    
    # Return shape: (1, Time, Feat)
    return jnp.expand_dims(sol.ys, 0), t_eval

# Helper for Temporal Batching
def get_batch(data, ts, batch_size=32, window_size=10):
    traj = data[0]
    total_len = traj.shape[0]
    starts = np.random.randint(0, total_len - window_size, size=batch_size)
    batch_y = []
    batch_t = ts[:window_size] # Relative time for ODEs
    
    for start in starts:
        batch_y.append(traj[start : start + window_size])
        
    return jnp.stack(batch_y), batch_t

# ==========================================
# 2. Model Architectures
# ==========================================

# --- A. Continuous Model (Neural ODE) ---
class VectorField(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, hidden_size, key):
        self.mlp = eqx.nn.MLP(hidden_size, hidden_size, 32, 2, jax.nn.tanh, key=key)
    def __call__(self, t, y, args):
        return self.mlp(y)

class NeuralODE(eqx.Module):
    encoder: eqx.nn.Linear
    vector_field: VectorField
    decoder: eqx.nn.Linear
    
    def __init__(self, data_size, hidden_size, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.encoder = eqx.nn.Linear(data_size, hidden_size, key=k1)
        self.vector_field = VectorField(hidden_size, key=k2)
        self.decoder = eqx.nn.Linear(hidden_size, data_size, key=k3)

    def __call__(self, ts, y0_batch):
        # y0_batch shape: (Batch, Feat) -> Just the start points
        # Encode
        z0 = jax.vmap(self.encoder)(y0_batch)
        
        # Integrate
        def integrate(z_start):
            term = diffrax.ODETerm(self.vector_field)
            sol = diffrax.diffeqsolve(
                term, diffrax.Tsit5(), t0=ts[0], t1=ts[-1], dt0=0.1, y0=z_start, 
                saveat=diffrax.SaveAt(ts=ts)
            )
            return sol.ys
            
        zs = jax.vmap(integrate)(z0)
        return jax.vmap(jax.vmap(self.decoder))(zs), zs

    def extrapolate(self, start_obs, steps):
        # For visualization: Integrate far into the future
        z = self.encoder(start_obs)
        term = diffrax.ODETerm(self.vector_field)
        # Solve for 10x the duration
        t_long = jnp.linspace(0, steps*0.2, steps) 
        sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0=0, t1=t_long[-1], dt0=0.1, y0=z, 
            saveat=diffrax.SaveAt(ts=t_long)
        )
        return jax.vmap(self.decoder)(sol.ys), sol.ys[-1]

# --- B. Discrete Models (RNN/GRU/LSTM) ---
class DiscreteSequenceModel(eqx.Module):
    cell: eqx.Module
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear
    cell_type: str
    hidden_size: int

    def __init__(self, type_name, data_size, hidden_size, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.cell_type = type_name
        self.hidden_size = hidden_size
        self.encoder = eqx.nn.Linear(data_size, hidden_size, key=k1)
        self.decoder = eqx.nn.Linear(hidden_size, data_size, key=k3)
        
        if type_name == "RNN":
            self.cell = eqx.nn.RNNCell(hidden_size, hidden_size, key=k2)
        elif type_name == "GRU":
            self.cell = eqx.nn.GRUCell(hidden_size, hidden_size, key=k2)
        elif type_name == "LSTM":
            self.cell = eqx.nn.LSTMCell(hidden_size, hidden_size, key=k2)

    def forward_step(self, state, x_input):
        # Autoregressive step: The model predicts the NEXT state from CURRENT state
        # In this setup, we treat the latent state as the dynamic object
        if self.cell_type == "LSTM":
            h, c = state
            h_new, c_new = self.cell(h, h) # Input is previous hidden (autonomous)
            return (h_new, c_new), h_new
        else:
            h = state
            h_new = self.cell(h, h) 
            return h_new, h_new

    def __call__(self, ts, y0_batch):
        # Autoregressive unrolling for 'len(ts)' steps
        # y0_batch: (Batch, Feat)
        
        def scan_fn(carry, t):
            state = carry
            # Decode current state to get prediction
            if self.cell_type == "LSTM":
                h, c = state
                pred = self.decoder(h)
            else:
                pred = self.decoder(state)
            
            # Evolve state
            new_state, _ = self.forward_step(state, None) # Input is ignored in autonomous mode
            return new_state, pred

        def unroll(start_y):
            # Init state
            h0 = self.encoder(start_y)
            if self.cell_type == "LSTM":
                state = (h0, jnp.zeros_like(h0))
            else:
                state = h0
            
            # Scan over time
            _, preds = jax.lax.scan(scan_fn, state, ts)
            return preds

        return jax.vmap(unroll)(y0_batch), None

    def extrapolate(self, start_obs, steps):
        # Init
        h0 = self.encoder(start_obs)
        if self.cell_type == "LSTM":
            state = (h0, jnp.zeros_like(h0))
        else:
            state = h0
            
        preds = []
        current_state = state
        
        for _ in range(steps):
            if self.cell_type == "LSTM":
                h, c = current_state
                preds.append(self.decoder(h))
            else:
                preds.append(self.decoder(current_state))
            
            current_state, _ = self.forward_step(current_state, None)
            
        return jnp.stack(preds), current_state

# ==========================================
# 3. Training Logic
# ==========================================
def train_model(model_name, model, data, ts):
    optimizer = optax.adam(0.005)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def step(model, opt_state, batch_t, batch_y):
        def loss_fn(m):
            # batch_y shape: (Batch, Time, Feat)
            start_pts = batch_y[:, 0, :]
            targets = batch_y # We want to reconstruct the trajectory
            
            preds, _ = m(batch_t, start_pts)
            return jnp.mean((preds - targets)**2)
            
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    losses = []
    # Train for 1000 steps
    for i in range(1000):
        batch_y, batch_t = get_batch(data, ts, batch_size=32, window_size=20)
        model, opt_state, loss = step(model, opt_state, batch_t, batch_y)
        losses.append(loss.item())
        
    return model, losses

# ==========================================
# 4. Analysis & Dashboard
# ==========================================
def analyze_stability(model, limit_state):
    # Calculates eigenvalues of the Jacobian at the limit
    if isinstance(model, NeuralODE):
        # Jacobian of Vector Field f(z)
        J_fn = jax.jacfwd(lambda z: model.vector_field(0, z, None))
        J = J_fn(limit_state) # limit_state is just z vector
        return jnp.linalg.eigvals(J), "Continuous"
    else:
        # Jacobian of Transition Map f(h) -> h_next
        if model.cell_type == "LSTM":
            # For LSTM, we look at Jacobian w.r.t hidden state h 
            # (Assuming c is stable or looking at effective map)
            h, c = limit_state
            def transition(h_in):
                (h_new, c_new), _ = model.cell(h_in, h_in)
                return h_new
            J = jax.jacfwd(transition)(h)
        else:
            def transition(h_in):
                return model.cell(h_in, h_in)
            J = jax.jacfwd(transition)(limit_state)
            
        return jnp.linalg.eigvals(J), "Discrete"

def create_dashboard(results, data, ts):
    true_y = data[0]
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Phase Portrait
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(true_y[:, 0], true_y[:, 1], 'k--', linewidth=2, label='Ground Truth', alpha=0.5)
    ax1.scatter([0], [0], c='k', marker='*', s=100)
    
    # 2. Time Series
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(true_y[:, 0], 'k--', linewidth=2, label='Ground Truth', alpha=0.5)
    
    # 3. Loss
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_yscale('log')
    
    # 4. Eigenvalues
    ax4 = fig.add_subplot(gs[1, 1])
    # Draw Unit Circle (Stability boundary for Discrete)
    circle = Circle((0, 0), 1, color='gray', fill=False, linestyle='--', label='Unit Circle (Discrete Bound)')
    ax4.add_patch(circle)
    ax4.axvline(0, color='gray', linestyle=':', label='Imag Axis (Continuous Bound)')
    
    colors = {'NeuralODE': 'red', 'RNN': 'blue', 'GRU': 'green', 'LSTM': 'purple'}
    
    for name, res in results.items():
        c = colors[name]
        model = res['model']
        losses = res['losses']
        
        # Extrapolate
        start_obs = true_y[0]
        traj, limit_state = model.extrapolate(start_obs, steps=300)
        
        # Plot Phase
        ax1.plot(traj[:, 0], traj[:, 1], color=c, label=name)
        ax1.scatter(traj[-1, 0], traj[-1, 1], color=c, marker='x')
        
        # Plot Time Series
        ax2.plot(traj[:, 0], color=c, label=name, alpha=0.8)
        
        # Plot Loss
        ax3.plot(losses, color=c, label=name, alpha=0.8)
        
        # Plot Eigenvalues
        eigs, mode = analyze_stability(model, limit_state)
        re, im = jnp.real(eigs), jnp.imag(eigs)
        
        if mode == "Continuous":
            # Plot as diamonds
            ax4.scatter(re, im, color=c, marker='D', s=40, label=f"{name} (Cont)")
        else:
            # Plot as circles
            ax4.scatter(re, im, color=c, marker='o', s=40, label=f"{name} (Disc)")

    ax1.set_title("Phase Space Extrapolation (Target: 0,0)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Time Series Extrapolation (t -> infinity)")
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title("Training Loss")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title("Eigenvalues at Limit State")
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.legend(loc='lower left', fontsize='small')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    print("Generating Data...")
    data, ts = get_data() # (1, 100, 2)
    hidden_dim = 16
    key = jax.random.PRNGKey(0)
    
    results = {}
    
    # --- Train Neural ODE ---
    print("Training Neural ODE...")
    node = NeuralODE(2, hidden_dim, key)
    node_trained, node_loss = train_model("NeuralODE", node, data, ts)
    results['NeuralODE'] = {'model': node_trained, 'losses': node_loss}
    
    # # --- Train RNN ---
    # print("Training RNN...")
    # rnn = DiscreteSequenceModel("RNN", 2, hidden_dim, key)
    # rnn_trained, rnn_loss = train_model("RNN", rnn, data, ts)
    # results['RNN'] = {'model': rnn_trained, 'losses': rnn_loss}
    
    # --- Train GRU ---
    print("Training GRU...")
    gru = DiscreteSequenceModel("GRU", 2, hidden_dim, key)
    gru_trained, gru_loss = train_model("GRU", gru, data, ts)
    results['GRU'] = {'model': gru_trained, 'losses': gru_loss}
    
    # # --- Train LSTM ---
    # print("Training LSTM...")
    # lstm = DiscreteSequenceModel("LSTM", 2, hidden_dim, key)
    # lstm_trained, lstm_loss = train_model("LSTM", lstm, data, ts)
    # results['LSTM'] = {'model': lstm_trained, 'losses': lstm_loss}
    
    print("Visualizing...")
    create_dashboard(results, data, ts)

if __name__ == "__main__":
    main()