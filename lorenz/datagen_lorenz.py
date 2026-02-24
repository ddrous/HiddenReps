#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", context="notebook")

## Fix the seed for reproducibility
np.random.seed(42)

## We want to generate synthetic data from ODEs of the form:
# dx/dt = f(x, t; theta)
# where x is the state vector, t is time, and theta are parameters. 
# The ODE must be hard.
# The test horizon should be longer than the trianng horizon.
# Each data point should have some noise.
# Save the trajectories as a numpy array of shape (num_trajectories, num_timepoints, state_dimension), both for train.npy and test.npy

# %%

def generate_ode_data(ode_func, theta, x0, t_span, num_trajectories, noise_std=0.1):
    from scipy.integrate import solve_ivp

    # t_eval = np.linspace(t_span[0], t_span[1], num=500)
    t_eval = np.arange(t_span[0], t_span[1], 0.02)

    data = []
    x0_orig = np.array(x0)

    for _ in range(num_trajectories):
        sol = solve_ivp(ode_func, t_span, x0, args=(theta,), t_eval=t_eval)
        trajectory = sol.y.T  # Shape (num_timepoints, state_dimension)
        noise = np.random.normal(0, noise_std, trajectory.shape)
        noisy_trajectory = trajectory + noise
        data.append(noisy_trajectory)

        ## Change the initial condition slightly for the next trajectory
        x0 = x0 + np.random.normal(0, 0.8, size=len(x0))

    return np.array(data)  # Shape (num_trajectories, num_timepoints, state_dimension)

# Example ODE function: Simple harmonic oscillator
def harmonic_oscillator(t, x, theta):
    k, m = theta
    dxdt = [x[1], - (k/m) * x[0]]
    return dxdt

# Exaample ODE function: highly non-linear system (Lorenz system)
def lorenz_system(t, x, theta):
    sigma, rho, beta = theta
    dxdt = [sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]]
    return dxdt

# Parameters for Lorenz system
theta_lorenz = (10.0, 28.0, 8/3)
x0_lorenz = [1.0, 1.0, 1.0]
t_span_train = (0, 10)
t_span_test = (0, 15)
num_trajectories_train = 5000
num_trajectories_test = 100

# Generate training and testing data
train_data = generate_ode_data(lorenz_system, theta_lorenz, x0_lorenz, t_span_train, num_trajectories_train, noise_std=0.1)
test_data = generate_ode_data(lorenz_system, theta_lorenz, x0_lorenz, t_span_test, num_trajectories_test, noise_std=0.0)
# Save the data
np.save('lorenz/train.npy', train_data)
np.save('lorenz/test.npy', test_data)


# %%
## PLot the first trajecgory, in 3D

from mpl_toolkits.mplot3d import Axes3D
plot_id = np.random.randint(0, test_data.shape[0])

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(train_data[plot_id,:,0], train_data[plot_id,:,1], train_data[plot_id,:,2], color='g')
ax.plot(test_data[plot_id,:,0], test_data[plot_id,:,1], test_data[plot_id,:,2], color='m')

## Put a dot at the start
ax.scatter(train_data[plot_id,0,0], train_data[plot_id,0,1], train_data[plot_id,0,2], color='g', s=100, label='Start Point (Train)')
ax.scatter(test_data[plot_id,0,0], test_data[plot_id,0,1], test_data[plot_id,0,2], color='m', s=100, label='Start Point (Test)')


ax.legend()

ax.set_title(f'3D Trajectory of Two Samples (Lorenz System)')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()

# %%
train_data.shape, test_data.shape