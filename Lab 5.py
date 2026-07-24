import numpy as np

# Input values
X = np.array([2, 3, 4])

# Weights
W = np.array([0.5, 0.4, 0.3])

# Bias
b = 1


# Calculate weighted sum
Z = np.dot(X, W) + b


# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU Activation Function
def relu(x):
    return max(0, x)


# Calculate outputs
sigmoid_output = sigmoid(Z)
relu_output = relu(Z)


# Display results

print("Weighted Sum:", Z)

print("Sigmoid Output:", round(sigmoid_output,4))

print("ReLU Output:", relu_output)
