#!/usr/bin/env python3

import numpy as np
import itertools as it
import scipy.sparse
import os, sys, functools
import time, glob

from squeezing_methods import squeezing_from_correlators
from dicke_methods import coherent_spin_state as coherent_state_PS
from multibody_methods import dist_method, get_multibody_states
from operator_product_methods import build_shell_operator

np.set_printoptions(linewidth = 200)
cutoff = 1e-10

if len(sys.argv) < 4:
    print(f"usage: {sys.argv[0]} [test?] [lattice_shape] [alpha] [max_manifold]")
    exit()

# determine whether this is a test run
if "test" in sys.argv:
    test_run = True
    sys.argv.remove("test")
else:
    test_run = False

lattice_shape = tuple(map(int, sys.argv[1].split("x")))
alpha = float(sys.argv[2]) # power-law couplings ~ 1 / r^\alpha
max_manifold = int(sys.argv[3])

# values of the ZZ coupling to simulate in an XXZ model
zz_couplings = np.arange(-3, +4.01, 0.1)

# values of the ZZ coupling to inspect more closely: half-integer values
inspect_zz_couplings = [ zz_coupling for zz_coupling in zz_couplings
                         if np.allclose(zz_coupling % 0.5, 0)
                         or np.allclose(zz_coupling % 0.5, 0.5) ]
inspect_sim_time = 2

max_time = 10 # in units of J_\perp
ivp_tolerance = 1e-10 # error tolerance in the numerical integrator

data_dir = "../data/shells/"
partial_dir = "../data/shells/partial/" # directory for storing partial results

if np.allclose(alpha, int(alpha)): alpha = int(alpha)
lattice_name = "x".join([ str(size) for size in lattice_shape ])
name_tag = f"L{lattice_name}_a{alpha}_M{max_manifold}"

##################################################


lattice_dim = len(lattice_shape)
spin_num = np.product(lattice_shape)
if np.allclose(alpha, int(alpha)): alpha = int(alpha)

zz_couplings = [ zz for zz in zz_couplings if not np.allclose(zz,1) ]
inspect_zz_couplings = [ zz for zz in inspect_zz_couplings if not np.allclose(zz,1) ]

for directory in [ data_dir, partial_dir ]:
    if not os.path.isdir(directory):
        os.makedirs(directory)

start_time = time.time()
def runtime():
    return f"[{int(time.time() - start_time)} sec]"

print("lattice shape:",lattice_shape)
##########################################################################################
# build SU(n) interaction matrix, and build generators of interaction eigenstates
##########################################################################################

dist = dist_method(lattice_shape)

sunc = {}
sunc["mat"] = np.zeros((spin_num,spin_num))
for pp, qq in it.combinations(range(spin_num),2):
    sunc["mat"][pp,qq] = sunc["mat"][qq,pp] = -1/dist(pp,qq)**alpha
sunc["TI"] = True

# build generators of interaction eigenstates (and associated data)
shell_energy_file = partial_dir + f"sunc_{name_tag}_shells_energies.txt"
def tensor_file(idx): return partial_dir + f"sunc_{name_tag}_tensor_{idx}.txt"
try:
    print("attempting to import eigenstates... ", end = "")
    sys.stdout.flush()

    sunc["shells"] = {}
    with open(shell_energy_file, "r") as file:
        for line in file:
            values = line.split()
            if ":" in line:
                manifold = int(values[0])
                shells = np.array([ int(val) for val in values[2:] ], dtype = int)
                sunc["shells"][manifold] = shells
            else:
                sunc["energies"] = np.array(values, dtype = float)

    for shell in range(len(sunc["energies"])):
        sunc[shell] = np.loadtxt(tensor_file(shell))

    for manifold, shells in sunc["shells"].items():
        for shell in shells:
            # if we imported the entire tensor, reshape it accordingly
            if sunc[shell].size == spin_num**manifold:
                sunc[shell].shape = (spin_num,) * manifold

            # if we only imported an "off-diagonal symmetric block" of the tensor,
            #   then construct the full tensor appropriately
            elif sunc[shell].size == np.math.comb(spin_num,manifold):
                tensor = np.zeros( (spin_num,) * manifold )
                for idx, comb in enumerate(it.combinations(range(spin_num),manifold)):
                    for perm in it.permutations(comb):
                        tensor[perm] = sunc[shell][idx]
                sunc[shell] = tensor

            else:
                print()
                print(f"could not make sense of a tensor with size {sunc[shell].size}")
                print(f"manifold, spin_num: {manifold,spin_num}")
                exit()

    print("success!")

    total_shells = 0
    for manifold, shells in sunc["shells"].items():
        total_shells += len(shells)
        print(f"manifold, size: {manifold}, {total_shells}")

except:
    print("failed!")
    print("building eigenstates")
    sys.stdout.flush()

    sunc["shells"], sunc["energies"], sunc_tensors \
        = get_multibody_states(lattice_shape, sunc["mat"], max_manifold, sunc["TI"])
    sunc.update(sunc_tensors)

    with open(shell_energy_file, "w") as file:
        for manifold, shells in sunc["shells"].items():
            shell_text = " ".join([ str(shell) for shell in shells ])
            file.write(f"{manifold} : {shell_text}\n")
        energy_text = " ".join([ str(energy) for energy in sunc["energies"] ])
        file.write(f"{energy_text}")

    for manifold, shells in sunc["shells"].items():
        for shell in shells:
            tensor = np.array(sunc_tensors[shell])
            if sunc["TI"]: # save the full tensor
                np.savetxt(tensor_file(shell), tensor.flatten())
            else: # save only an "off-diagonal symmetric block" of the tensor
                with open(tensor_file(shell), "w") as file:
                    for comb in it.combinations(range(spin_num),manifold):
                        file.write(f"{tensor[comb]}" + "\n")

shell_num = len(sunc["energies"])

for manifold, shells in list(sunc["shells"].items()):
    if len(shells) == 0:
        del sunc["shells"][manifold]

##########################################################################################
# compute states and operators in the shell / Z-projection basis
##########################################################################################
print(runtime(), "building operators in the shell / Z-projection basis")
sys.stdout.flush()

# spin basis
dn = np.array([1,0])
up = np.array([0,1])

# 1-local Pauli operators
local_ops = { "Z" : np.outer(up,up) - np.outer(dn,dn),
              "+" : np.outer(up,dn),
              "-" : np.outer(dn,up) }

# 2-local tensor products of Pauli operators
for op_lft, op_rht in it.product(local_ops.keys(), repeat = 2):
    mat_lft = local_ops[op_lft]
    mat_rht = local_ops[op_rht]
    local_ops[op_lft + op_rht] = np.kron(mat_lft,mat_rht).reshape((2,)*4)

# build the ZZ perturbation operator in the shell / Z-projection basis
coupling_file = partial_dir + f"coupling_{name_tag}.txt"
try:
    print(runtime(), "attempting to import perturbation operator... ", end = "")
    sys.stdout.flush()
    shell_coupling_op = np.loadtxt(coupling_file)
    shell_coupling_op.shape = (shell_num, shell_num, spin_num+1)
    print("success!")

except:
    print("failed!")
    print(runtime(), "building perturbation operator")
    sys.stdout.flush()
    shell_coupling_op \
        = build_shell_operator([sunc["mat"]], [local_ops["ZZ"]/4], sunc, sunc["TI"])
    shell_coupling_op = np.einsum("ikjk->ijk", shell_coupling_op)
    np.savetxt(coupling_file, shell_coupling_op.flatten())

# build collective spin operators
def _pauli_op(pauli):
    tensors = [np.ones(spin_num)]
    operators = [local_ops[pauli]]
    diagonal = ( pauli in [ "Z", "ZZ" ] )
    return build_shell_operator(tensors, operators, sunc, sunc["TI"],
                                collective = True, shell_diagonal = diagonal)
collective_shape = ( shell_num*(spin_num+1), )*2

collective_ops = {}
for op in [ "Z", "+" ]:
    collective_file = partial_dir + f"collective_{name_tag}_{op}.txt"
    try:
        print(runtime(), f"attempting to import collective operator : {op} ... ", end = "")
        sys.stdout.flush()
        collective_ops[op] = scipy.sparse.dok_matrix(collective_shape, dtype = float)
        with open(collective_file, "r") as file:
            for line in file:
                idx_lft, idx_rht, val = line.split()
                collective_ops[op][int(idx_lft),int(idx_rht)] = float(val)
        collective_ops[op] = collective_ops[op].tocsr()
        print("success!")

    except:
        print("failed!")
        print(runtime(), f"building collective operator : {op}")
        sys.stdout.flush()
        pauli_op = _pauli_op(op)
        pauli_op.shape = collective_shape
        with open(collective_file, "w") as file:
            for idx_lft, idx_rht in zip(*np.nonzero(pauli_op)):
                op_val = pauli_op[idx_lft, idx_rht]
                file.write(f"{idx_lft} {idx_rht} {op_val}\n")

        collective_ops[op] = scipy.sparse.csr_matrix(pauli_op)
        del pauli_op

collective_ops["ZZ"] = collective_ops["Z"] @ collective_ops["Z"]
collective_ops["++"] = collective_ops["+"] @ collective_ops["+"]
collective_ops["+Z"] = collective_ops["+"] @ collective_ops["Z"]
collective_ops["+-"] = collective_ops["+"] @ collective_ops["+"].conj().T

# if this is a test run, we can exit now
if test_run: exit()

##########################################################################################
# methods to build states (and energies)
##########################################################################################

# energies and energy eigenstates within each sector of fixed spin projection
def energies_states(zz_sun_ratio):
    energies = np.zeros( ( shell_num, spin_num+1 ) )
    eig_states = np.zeros( ( shell_num, shell_num, spin_num+1 ), dtype = complex )

    for spins_up in range(spin_num+1):
        # construct the Hamiltonian at this Z projection, from SU(n) + ZZ couplings
        _proj_hamiltonian = np.diag(sunc["energies"]) \
                          + zz_sun_ratio * shell_coupling_op[:,:,spins_up]

        # diagonalize the net Hamiltonian at this Z projection
        energies[:,spins_up], eig_states[:,:,spins_up] = np.linalg.eigh(_proj_hamiltonian)

        spins_dn = spin_num - spins_up
        if spins_up == spins_dn: break
        energies[:,spins_dn], eig_states[:,:,spins_dn] \
            = energies[:,spins_up], eig_states[:,:,spins_up]

    return energies, eig_states

# coherent spin state
def coherent_spin_state(vec):
    zero_shell = np.zeros(shell_num)
    zero_shell[0] = 1
    return np.outer(zero_shell, coherent_state_PS(vec,spin_num))

##########################################################################################
# simulate!
##########################################################################################

chi_eff_bare = sunc["mat"].sum() / (spin_num * (spin_num-1))
state_X = coherent_spin_state([0,1,0])

def _states(initial_state, zz_sun_ratio, times):
    energies, eig_states = energies_states(zz_sun_ratio)
    init_state_eig = np.einsum("Ssz,Sz->sz", eig_states, initial_state)
    phases = np.exp(-1j * np.tensordot(times, energies, axes = 0))
    evolved_eig_states = phases * init_state_eig[None,:,:]
    return np.einsum("sSz,tSz->tsz", eig_states, evolved_eig_states)

def simulate(zz_coupling, sim_time = None, max_tau = 2, points = 500):
    print("zz_coupling:", zz_coupling)
    sys.stdout.flush()
    zz_sun_ratio = zz_coupling - 1
    assert(zz_sun_ratio != 0)

    # determine how long to simulate
    if sim_time is None:
        chi_eff = abs(zz_sun_ratio * chi_eff_bare)
        sim_time = min(max_time, max_tau * spin_num**(-2/3) / chi_eff)

    times = np.linspace(0, sim_time, points)

    # compute states at all times of interest
    states = _states(state_X, zz_sun_ratio, times)

    # compute collective spin correlators
    def val(mat, state):
        return state.conj() @ ( mat @ state ) / ( state.conj() @ state )
    correlators = { op : np.array([ val(mat,state.flatten()) for state in states ])
                    for op, mat in collective_ops.items() }

    # compute populations
    pops = np.einsum("tsz->ts", abs(states)**2)

    return times, correlators, pops

##########################################################################################
print(runtime(), "running inspection simulations")
sys.stdout.flush()

str_ops = [ "Z", "+", "ZZ", "++", "+Z", "+-" ]
tup_ops = [ (0,1,0), (1,0,0), (0,2,0), (2,0,0), (1,1,0), (1,0,1) ]
def relabel(correlators):
    return { tup_op : correlators[str_op] for tup_op, str_op in zip(tup_ops, str_ops) }

str_op_list = ", ".join(str_ops)

for zz_coupling in inspect_zz_couplings:
    times, correlators, pops = simulate(zz_coupling, sim_time = inspect_sim_time)
    sqz = squeezing_from_correlators(spin_num, relabel(correlators), pauli_ops = True)

    with open(data_dir + f"inspect_{name_tag}_z{zz_coupling:.1f}.txt", "w") as file:
        file.write(f"# times, {str_op_list}, sqz, populations (within each shell)\n")
        for manifold, shells in sunc["shells"].items():
            file.write(f"# manifold {manifold} : ")
            file.write(" ".join([ str(shell) for shell in shells ]))
            file.write("\n")
        for tt in range(len(times)):
            file.write(f"{times[tt]} ")
            file.write(" ".join([ str(correlators[op][tt]) for op in str_ops ]))
            file.write(f" {sqz[tt]} ")
            file.write(" ".join([ str(pop) for pop in pops[tt,:] ]))
            file.write("\n")

##########################################################################################
if len(zz_couplings) == 0: exit()
print(runtime(), "running sweep simulations")
sys.stdout.flush()

results = [ simulate(zz_coupling) for zz_coupling in zz_couplings ]
sweep_times, sweep_correlators, sweep_pops = zip(*results)

print(runtime(), "computing squeezing values")
sys.stdout.flush()

sweep_sqz = [ squeezing_from_correlators(spin_num, relabel(correlators), pauli_ops = True)
              for correlators in sweep_correlators ]
min_sqz_vals = [ min(sqz) for sqz in sweep_sqz ]
min_sqz_idx = [ max(1,np.argmin(sqz)) for sqz in sweep_sqz ]
opt_time_vals = [ sweep_times[zz][tt] for zz, tt in enumerate(min_sqz_idx) ]

min_SS_vals = [ min( + correlators["ZZ"][:tt].real/4
                     + correlators["+-"][:tt].real
                     - correlators["Z"][:tt].real/2 )
                for correlators, tt in zip(sweep_correlators, min_sqz_idx) ]

pop_vals = [ np.array([ pops[:tt,shells].sum(axis = 1)
                        for shells in sunc["shells"].values() ]).T
             for pops, tt in zip(sweep_pops, min_sqz_idx) ]
min_pop_vals = np.array([ pops.min(axis = 0) for pops in pop_vals ])
max_pop_vals = np.array([ pops.max(axis = 0) for pops in pop_vals ])

str_op_opt_list = ", ".join([ op + "_opt" for op in str_ops ])
opt_correlators = [ { op : correlators[op][tt] for op in str_ops }
                    for correlators, tt in zip(sweep_correlators, min_sqz_idx) ]

print(runtime(), "saving results")
sys.stdout.flush()

with open(data_dir + f"sweep_{name_tag}.txt", "w") as file:
    file.write(f"# zz_coupling, time_opt, sqz_min, {str_op_opt_list}, SS_min, "
               + "min_pop_0, max_pop (for manifolds > 0)\n")
    file.write("# manifolds : ")
    file.write(" ".join([ str(manifold) for manifold in sunc["shells"].keys() ]))
    file.write("\n")
    for zz in range(len(zz_couplings)):
        file.write(f"{zz_couplings[zz]} {opt_time_vals[zz]} ")
        file.write(f"{min_sqz_vals[zz]} {min_SS_vals[zz]} ")
        file.write(" ".join([ str(opt_correlators[zz][op]) for op in str_ops ]))
        file.write(f" {min_pop_vals[zz,0]} ")
        file.write(" ".join([ str(val) for val in max_pop_vals[zz,1:] ]))
        file.write("\n")

print("completed", runtime())
