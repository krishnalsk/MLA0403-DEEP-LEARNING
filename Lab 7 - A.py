import math

# Inputs
x1 = 0.5
x2 = 0.8

# Weights
w1 = 0.4
w2 = 0.7

# Bias
b = 0.2

# Forward Propagation
z = (x1 * w1) + (x2 * w2) + b

output = 1 / (1 + math.exp(-z))

print("Weighted Sum =", round(z, 4))
print("Output =", round(output, 4))
