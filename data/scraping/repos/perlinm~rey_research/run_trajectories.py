#!/usr/bin/env python3

import sys

assert(len(sys.argv[1:]) in [ 3, 4 ])
method = sys.argv[1]
dec_idx = int(sys.argv[2])
seed = int(sys.argv[3])

OAT, TAT, TNT = "OAT", "TAT", "TNT"
assert(method in [ OAT, TAT, TNT ])

log10_N = 2
N = 10**log10_N

trajectories = 100
max_tau = 2
time_steps = 1000
default_savepoints = True

log_dir = f"./logs/"
data_dir = f"../data/squeezing/jumps/"
job_name = f"sqz_D_exact_logN{log10_N}_{method}_d{dec_idx:02d}_s{seed:03d}"

log_text = f"""#!/bin/sh

#SBATCH --partition=nistq,jila
#SBATCH --mem=5G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}{job_name}.o
#SBATCH --error={log_dir}{job_name}.e
#SBATCH --time=10-00

module load python3

python3 {sys.argv[0]} {method} {dec_idx} {seed}
"""

if len(sys.argv[1:]) == 4:
    with open(log_dir + job_name + ".sh", "w") as f:
        f.write(log_text)
    exit()

######################################################################

import numpy as np

from dicke_methods import coherent_spin_state
from correlator_methods import mat_zxy_to_pzm, vec_zxy_to_pzm
from jump_methods import correlators_from_trajectories

dec_rates = np.logspace(-3,1,13)
dec_rates = [ (dec_rates[dec_idx],)*3, (0,0,0) ]

init_state = "-Z"
h_vec = {}
h_vec[OAT] = { (0,2,0) : 1 }
h_vec[TAT] = { (0,2,0) : +1/3,
               (0,0,2) : -1/3 }
h_vec[TNT] = { (0,2,0) : 1,
               (1,0,0) : -N/2 }
for model, vec in h_vec.items():
    h_vec[model] = vec_zxy_to_pzm(vec)

dec_mat_zxy = np.array([ [ 0, -1, 0 ],
                         [ 1,  0, 0 ],
                         [ 0,  0, 1 ]])
dec_mat = mat_zxy_to_pzm(dec_mat_zxy)

max_time = max_tau * N**(-2/3)
times = np.linspace(0, max_time, time_steps)

init_state_vec = coherent_spin_state(init_state, N)
def jump_args(hamiltonian):
    return [ N, trajectories, times, init_state_vec, hamiltonian, dec_rates, dec_mat ]

correlators = correlators_from_trajectories(*jump_args(h_vec[method]), seed = seed,
                                            default_savepoints = default_savepoints)
with open(data_dir + job_name + ".txt", "w") as f:
    f.write(f"# trajectories: {trajectories}\n")
    f.write(f"# max_tau: {max_tau}\n")
    f.write(f"# time_steps: {time_steps}\n")
    ops = list(correlators.keys())
    f.write("# ops: " + " ".join([ str(op) for op in ops]) + "\n")
    for op in ops:
        f.write(",".join([ str(val) for val in correlators[op] ]) + "\n")

