import numpy as np

# Linearly separable dataset (AND gate)

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

Y = np.array([0,0,0,1])


# Initialize weights and bias

weights = np.zeros(2)
bias = 0

learning_rate = 0.1
epochs = 10


# Step activation function

def activation(x):
    if x >= 0:
        return 1
    else:
        return 0


# Training Perceptron

for epoch in range(epochs):

    for i in range(len(X)):

        linear_output = np.dot(X[i], weights) + bias

        prediction = activation(linear_output)

        error = Y[i] - prediction

        weights = weights + learning_rate * error * X[i]

        bias = bias + learning_rate * error


print("Final Weights:", weights)
print("Final Bias:", bias)


# Testing

for x in X:
    result = activation(np.dot(x, weights) + bias)
    print(x, "=>", result)
