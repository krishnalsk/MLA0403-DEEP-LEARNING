import numpy as np
import matplotlib.pyplot as plt

# Actual parameters
actual_mean = 50
actual_std = 10
actual_variance = actual_std ** 2

# Generate synthetic data
np.random.seed(42)
data = np.random.normal(actual_mean, actual_std, 1000)

# Maximum Likelihood Estimation (MLE)
estimated_mean = np.mean(data)
estimated_variance = np.mean((data - estimated_mean) ** 2)

# Display results
print("Actual Mean:", actual_mean)
print("Estimated Mean:", round(estimated_mean, 2))

print("Actual Variance:", actual_variance)
print("Estimated Variance:", round(estimated_variance, 2))

# Plot Histogram
plt.hist(data, bins=30, edgecolor='black')
plt.title("Histogram of Synthetic Data")
plt.xlabel("Data Values")
plt.ylabel("Frequency")
plt.show()
