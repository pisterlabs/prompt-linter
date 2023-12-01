# --------------------------------------------------
#      OpenAi NES Algorithm Minimizing a Sphere
# --------------------------------------------------
# file: fig_openai_nes_sphere.py
# --------------------------------------------------
# Import the necessary libraries and functions
import numpy as np
from es_rep.algs.openai_nes import openai_es
from test_functions_rep.so.sphere import sphere

# 0 - Repeatability
np.random.seed(123)

# 1 - Define the upper and lower bounds of the search space
ndim = 20               # Number of dimensions of the problem
lb = -5*np.ones(ndim)   # Bounds of the search space
ub = 5*np.ones(ndim)

# 2 - Define the parameters for the optimization
maxIterations = 300  # maximum number of iterations
nMC = 10             # number of MC runs
# 3 - Parameters for the algorithm
npop = 25       # population size
sigma = 0.1     # noise standard deviation
alpha = 0.01    # learning rate
theta_0 = np.random.rand(ndim,nMC)  # random initial solution

# 4 - Define the cost function
fcost = sphere

# 5 - Run the NES algorithm
best_theta  = np.zeros((ndim,nMC))
best_scores = np.zeros((maxIterations,nMC))
for id_run in range(nMC):
    np.random.seed(123)
    [a, b] = openai_es(fcost, theta_0[:, id_run], maxIterations, lb, ub, npop, alpha, sigma)
    best_theta[:, id_run] = a
    best_scores[:, id_run] = np.squeeze(b) # remove an unsed dimension in b


# 6 - Plot the convergence curves
from auxiliary.plot_convergence_curve import plot_convergence_curve
fig_file = 'figure_5_7.eps'
title_str = \
    "OpenAI NES Minimizing a " + str(ndim) + "-dimensional Sphere"
plot_convergence_curve(fig_file,title_str,best_scores)


