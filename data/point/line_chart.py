import numpy as np
import matplotlib.pyplot as plt

# Define the epsilon values (in log scale)
epsilon = np.logspace(-7, 0, 400)

# Define the gamma values
gamma_values = [0.2, 0.4, 0.6, 0.8]

# Simulate the number of edges for each gamma value (dummy data for demonstration)
edges = {
    0.2: 5000 * np.exp(-epsilon * 10**4),
    0.4: 5000 * np.exp(-epsilon * 8**4),
    0.6: 5000 * np.exp(-epsilon * 6**4),
    0.8: 5000 * np.exp(-epsilon * 4**4),
}

# Plot the data
plt.figure(figsize=(10, 6))

for gamma in gamma_values:
    plt.plot(epsilon, edges[gamma], label=f'γ={gamma}')

plt.xscale('log')
plt.yscale('linear')
plt.xlabel(r'$\epsilon$ (in log scale)')
plt.ylabel('number of edges')
plt.legend()
plt.tight_layout()

# Save the plot to a file
output_path_gamma_plot = 'gamma_plot.png'
plt.savefig(output_path_gamma_plot)
