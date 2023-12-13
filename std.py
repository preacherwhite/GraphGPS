import numpy as np

# Data
data = np.array([84.26, 80.56, 78.70])

# Calculating the mean
mean = np.mean(data)

# Calculating the standard error of the mean (SEM)
sem = np.std(data, ddof=1) / np.sqrt(len(data))

print(mean,sem)