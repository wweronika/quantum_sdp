import numpy as np
import matplotlib.pyplot as plt

# Read data from file
d = 4
bin_size = 0.002
with open(f'results_d{d}.csv', 'r') as f:
    data = [float(line.strip()) for line in f]

# Determine bin edges using the specified width of 0.01
min_val = min(data)
max_val = max(data)
bin_edges = np.arange(min_val, max_val + bin_size, bin_size)

# Plotting the histogram
plt.hist(data, bins=bin_edges, edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title(f'Values of convergence for  d={d}')
plt.tight_layout()
plt.show()