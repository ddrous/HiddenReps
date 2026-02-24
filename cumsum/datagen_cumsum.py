#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", context="notebook")

#---- Generate Cumulative Sum Data (for testing) ---
def generate_cumsum_data(num_traj=1000, seq_len=50, seq_dim=3, seed=2026):
    # np.random.seed(seed)

    ## For each traj, we want y_t = w x_t. First we sample x_t
    # xs = np.random.randn(num_traj, seq_len, seq_dim-1)
    xs = np.random.normal(size=(num_traj, seq_len, seq_dim-1), scale=1)
    # xs = np.random.uniform(low=-0.1, high=0.1, size=(num_traj, seq_len, seq_dim-1))

    ## Sample a linear transformation vector w
    w = np.random.randn(num_traj, seq_dim-1)
    
    ## Compute y_t = w x_t (dot product along last dim)
    ys = np.einsum('ijk,ik->ij', xs, w)
    
    ## Now we concatenate xs and ys to have shape (num_traj, seq_len, seq_dim)
    data = np.concatenate([xs, ys[..., None]], axis=-1)

    ## Save data as numpy arrays, proportion 0.8 train, 0.2 test
    split_idx = int(num_traj * 0.8)
    np.save('cumsum/train.npy', data[:split_idx])
    np.save('cumsum/test.npy', data[split_idx:])

    return data

if __name__ == "__main__":
    data = generate_cumsum_data(num_traj=1, seq_len=64, seq_dim=2)

    print("Data shape:", data.shape)
    # Plot a few trajectories
    for i in range(1):
        plt.plot(data[i, :, -1], label=f'Traj {i} (y)')
        plt.plot(data[i, :, 0], label=f'Traj {i} (x1)')
        # plt.plot(data[i, :, 1], label=f'Traj {i} (x2)')

        ## DO a cumsum before plotting
        plt.plot(np.cumsum(data[i, :, -1]), "--", label=f'Traj {i} (cumsum y)')
        plt.plot(np.cumsum(data[i, :, 0]), "--", label=f'Traj {i} (cumsum x1)')

    plt.legend()
    plt.title("Sample Trajectories (Cumulative Sum Data)")
    plt.show()