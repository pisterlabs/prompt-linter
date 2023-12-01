"""
This script generates and plots the Fock distribution of three different quantum states:
a coherent state, a thermal state, and a Fock state. It uses the QuTiP (Quantum Toolbox in Python)
library to generate the states and plot the distributions.

Author: Chen Huang
Date: 20 Nov 2023
"""

import matplotlib.pyplot as plt
import numpy as np
from qutip import coherent_dm, thermal_dm, fock_dm, plot_fock_distribution

N = 20  # Define the dimension of the Fock space

# Create different density matrices: coherent state, thermal state, and Fock state
rho_coherent = coherent_dm(N, np.sqrt(2))  # Coherent state with amplitude sqrt(2)
rho_thermal = thermal_dm(N, 2)  # Thermal state with average occupation number 2
rho_fock = fock_dm(N, 2)  # Fock state with 2 photons

# Initialize a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# Plot the Fock distribution of each state
plot_fock_distribution(rho_coherent, fig=fig, ax=axes[0], title='Coherent State')
plot_fock_distribution(rho_thermal, fig=fig, ax=axes[1], title='Thermal State')
plot_fock_distribution(rho_fock, fig=fig, ax=axes[2], title='Fock State')

fig.tight_layout()  # Automatically adjust subplot parameters to give specified padding

plt.savefig('plot-fock-distribution.png')
plt.show()
