# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Generate synthetic dataset
# -----------------------------
rng = np.random.default_rng(42)
n = 200
x = rng.uniform(0.0, 5.0, size=n)
epsilon = rng.normal(0.0, 1.0, size=n)
y = 3.0 + 4.0 * x + epsilon
X = np.column_stack([np.ones_like(x), x])  # add bias

# -----------------------------
# Step 2: Closed-form solution
# -----------------------------
theta_closed = np.linalg.inv(X.T @ X) @ X.T @ y
b0_closed, b1_closed = theta_closed

# -----------------------------
# Step 3: Gradient Descent
# -----------------------------
def mse(theta, X, y):
    return np.mean((y - X @ theta) ** 2)

def grad_mse(theta, X, y):
    n = X.shape[0]
    return (-2.0 / n) * (X.T @ (y - X @ theta))

theta_gd = np.array([0.0, 0.0])
eta, num_iters = 0.05, 1000
loss_history = []
for _ in range(num_iters):
    loss_history.append(mse(theta_gd, X, y))
    theta_gd -= eta * grad_mse(theta_gd, X, y)
b0_gd, b1_gd = theta_gd

# -----------------------------
# Step 4: Print results
# -----------------------------
print("Closed-form solution:")
print(f"  Intercept (b0): {b0_closed:.6f}")
print(f"  Slope     (b1): {b1_closed:.6f}")
print("\nGradient Descent solution:")
print(f"  Intercept (b0): {b0_gd:.6f}")
print(f"  Slope     (b1): {b1_gd:.6f}")

# -----------------------------
# Step 5: Plots
# -----------------------------
x_line = np.linspace(x.min(), x.max(), 200)
X_line = np.column_stack([np.ones_like(x_line), x_line])

plt.figure()
plt.scatter(x, y, s=12, alpha=0.7, label="Data")
plt.plot(x_line, X_line @ theta_closed, label="Closed-form fit", linewidth=2)
plt.plot(x_line, X_line @ theta_gd, label="GD fit", linestyle="--", linewidth=2)
plt.xlabel("x"); plt.ylabel("y")
plt.title("Linear Regression: Closed-form vs Gradient Descent")
plt.legend(); plt.show()

plt.figure()
plt.plot(loss_history)
plt.xlabel("Iteration"); plt.ylabel("MSE")
plt.title("Gradient Descent Loss Curve")
plt.show()