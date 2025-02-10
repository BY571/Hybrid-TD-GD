import numpy as np
import matplotlib.pyplot as plt

# Environment setup: MRP with 3 states
P = np.array([[0.5, 0.5, 0.0],  # Transition probabilities
              [0.0, 0.5, 0.5],
              [0.5, 0.0, 0.5]])

rewards = np.array([1.0, 0.0, -1.0])  # Reward for each state
gamma = 0.9  # Discount factor

# Feature matrix: simple identity mapping (each state is a feature)
phi = np.eye(3)

# Initialize parameters
theta_td = np.random.randn(3)  # TD learning parameters
theta_gd = np.random.randn(3)  # GD parameters
theta_hybrid = np.random.randn(3)  # Hybrid approach parameters

alpha = 0.1  # Learning rate
lambda_hybrid = 0.3  # Mixing coefficient for Hybrid TD-GD

# Number of iterations
num_iters = 200
errors_td, errors_gd, errors_hybrid = [], [], []

# True value function (computed from Bellman equation)
true_values = np.linalg.inv(np.eye(3) - gamma * P) @ rewards

# Training loop
for _ in range(num_iters):
    # Sample a state randomly
    s = np.random.choice(3)
    next_s = np.random.choice(3, p=P[s])  # Sample next state from transition probabilities
    r = rewards[s]  # Get immediate reward

    # TD(0) update
    delta_td = r + gamma * np.dot(theta_td, phi[next_s]) - np.dot(theta_td, phi[s])
    theta_td += alpha * delta_td * phi[s]

    # Gradient Descent update (directly minimizing MSE)
    target = r + gamma * np.dot(true_values, phi[next_s])  # True target
    gradient = (np.dot(theta_gd, phi[s]) - target) * phi[s]  # MSE gradient
    theta_gd -= alpha * gradient

    # Hybrid TD-GD update
    delta_hybrid = r + gamma * np.dot(theta_hybrid, phi[next_s]) - np.dot(theta_hybrid, phi[s])
    theta_hybrid += alpha * (delta_hybrid * phi[s] + lambda_hybrid * gradient)

    # Compute errors
    errors_td.append(np.linalg.norm(theta_td - true_values))
    errors_gd.append(np.linalg.norm(theta_gd - true_values))
    errors_hybrid.append(np.linalg.norm(theta_hybrid - true_values))

# Plot the results
plt.plot(errors_td, label="TD(0)", linestyle="dashed")
plt.plot(errors_gd, label="Gradient Descent", linestyle="dotted")
plt.plot(errors_hybrid, label="Hybrid TD-GD", linewidth=2)
plt.xlabel("Iterations")
plt.ylabel("Error ||θ - θ*||")
plt.legend()
plt.title("Comparison of TD, GD, and Hybrid TD-GD")
plt.show()
