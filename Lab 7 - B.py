import math

# Inputs
x1 = 0.5
x2 = 0.8

# Initial Weights
w1 = 0.4
w2 = 0.7

# Bias
b = 0.2

# Target Output
target = 1

# Learning Rate
lr = 0.1

# ---------- Forward Propagation ----------
z = x1*w1 + x2*w2 + b
output = 1/(1+math.exp(-z))

# ---------- Backpropagation ----------
error = target - output

gradient = error * output * (1-output)

# Update weights
w1 = w1 + lr * gradient * x1
w2 = w2 + lr * gradient * x2
b = b + lr * gradient

# Display Results
print("Output =", round(output,4))
print("Error =", round(error,4))

print("Updated Weight w1 =", round(w1,4))
print("Updated Weight w2 =", round(w2,4))
print("Updated Bias =", round(b,4))
