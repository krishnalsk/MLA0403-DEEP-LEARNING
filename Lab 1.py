import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
Y = np.array([2,4,5,4,5,7,8,9,10,12], dtype=float)

# Initialize parameters
m = 0
c = 0

learning_rate = 0.01
iterations = 1000
n = len(X)

loss_history = []

# Gradient Descent
for i in range(iterations):

    y_pred = m * X + c

    loss = np.mean((Y - y_pred) ** 2)
    loss_history.append(loss)

    dm = (-2/n) * np.sum(X * (Y - y_pred))
    dc = (-2/n) * np.sum(Y - y_pred)

    m = m - learning_rate * dm
    c = c - learning_rate * dc

print("Slope (m):", round(m,3))
print("Intercept (c):", round(c,3))
print("Final Loss:", round(loss,4))

# Regression Plot
plt.figure(figsize=(6,4))
plt.scatter(X,Y,color='blue',label="Actual Data")
plt.plot(X,m*X+c,color='red',label="Regression Line")
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Learning Curve
plt.figure(figsize=(6,4))
plt.plot(loss_history)
plt.title("Learning Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
