#!/usr/bin/env python3

import sys, scipy
import numpy as np
import matplotlib.pyplot as plt

from dicke_methods import coherent_spin_state


figsize = (4,3)
params = { "font.size" : 12,
           "text.usetex" : True,
           "text.latex.preamble" : r"\usepackage{braket,bm}" }
plt.rcParams.update(params)

if len(sys.argv) != 5:
    print(f"usage: {sys.argv[0]} [lattice_shape] [radius] [alpha] [max_manifold]")
    exit()

lattice_shape = tuple(map(int, sys.argv[1].split("x")))
radius_text = sys.argv[2] # blockade radius
alpha_text = sys.argv[3] # power-law coefficient
max_manifold = int(sys.argv[4])

radius = float(radius_text)
if alpha_text == "inf":
    alpha = np.inf
else:
    alpha = int(alpha_text)

data_dir = "../data/shells_XX/"
partial_dir = data_dir + "partial/"
fig_dir = "../figures/shells_XX/"

lattice_name = "x".join([ str(size) for size in lattice_shape ])
name_tag = f"L{lattice_name}_r{radius_text}_a{alpha_text}_M{max_manifold}"

lattice_dim = len(lattice_shape)
spin_num = np.product(lattice_shape)

energies_file = data_dir + f"energies_{name_tag}.txt"
eig_states_file = data_dir + f"eig_states_{name_tag}.txt"

with open(energies_file,"r") as file:
    for line in file:
        energies_shape = list(map(int,line.split()[1:]))
        break
with open(eig_states_file,"r") as file:
    for line in file:
        eig_states_shape = list(map(int,line.split()[1:]))
        break

##################################################

# energies and energy eigenstates within each sector of fixed spin projection
energies = np.loadtxt(energies_file, dtype = float).reshape(energies_shape)
eig_states = np.loadtxt(eig_states_file, dtype = complex).reshape(eig_states_shape)

shell_num = energies.shape[0]

collective_shape = ( shell_num*(spin_num+1), )*2
collective_ops = {}
for op in [ "Z", "+" ]:
    collective_file = partial_dir + f"collective_{name_tag}_{op}.txt"
    collective_ops[op] = scipy.sparse.dok_matrix(collective_shape, dtype = float)
    with open(collective_file, "r") as file:
        for line in file:
            idx_lft, idx_rht, val = line.split()
            collective_ops[op][int(idx_lft),int(idx_rht)] = float(val)
    collective_ops[op] = collective_ops[op].tocsr()
collective_ops["ZZ"] = collective_ops["Z"] @ collective_ops["Z"]
collective_ops["+-"] = collective_ops["+"] @ collective_ops["+"].conj().T
SS = collective_ops["ZZ"]/4 + collective_ops["+-"].real
del collective_ops

##################################################

state_X = coherent_spin_state([0,1,0],spin_num)
eig_pops = abs(np.einsum("ez,z->ez", eig_states[0,:,:], state_X))**2
eig_pops /= eig_pops.sum()
max_pop = eig_pops.max()
scar_pop = eig_pops[0,:].sum()
with open(data_dir + f"scar_pop_X_{name_tag}.txt","w") as file:
    print(scar_pop, file = file)

def add_to_plots(name, ylabel, spin_proj, color_vals, yvals):
    _spin_proj = np.full(yvals.size, spin_proj)

    plt.figure(name, figsize = figsize)
    plt.plot(+_spin_proj, yvals, "k.")
    plt.plot(-_spin_proj, yvals, "k.")
    plt.xlabel(r"$S_{\mathrm{z}}$")
    plt.ylabel(ylabel)

    # sort data by increasing value,
    #   so that larger populations (darker dots) are
    #   not masked by smaller ones (lighter dots)
    data = sorted(zip( color_vals, _spin_proj, yvals ))
    _color_vals, _spin_proj, _yvals = map(np.array,zip(*data))
    plt.figure(name + "_X", figsize = figsize)
    kwargs = dict( marker = ".", c = _color_vals, cmap = "binary",
                   vmin = 0, vmax = 1 )
    plt.scatter(+_spin_proj, _yvals, **kwargs)
    plt.scatter(-_spin_proj, _yvals, **kwargs)
    plt.xlabel(r"$S_{\mathrm{z}}$")
    plt.ylabel(ylabel)

for spins_up in range(spin_num//2+1):
    spins_dn = spin_num - spins_up
    spin_proj = spins_up - spins_dn

    _eig_pops = eig_pops[:,spins_up] / max_pop
    _energies = energies[:,spins_up]
    add_to_plots("spect", r"$E/J_\perp$", spin_proj, _eig_pops, _energies)

    _gaps = np.array([_energies[1] - _energies[0]])
    add_to_plots("gaps", r"$\Delta/J_\perp$", spin_proj, _eig_pops, _gaps)

    _SS = SS[spins_up::spin_num+1,spins_up::spin_num+1].toarray()
    _eig_states = eig_states[:,:,spins_up]
    _SS_args = ( _eig_states.conj(), _SS, _eig_states )
    _SS_vals = np.einsum("se,sS,Se->e", *_SS_args).real
    _SS_max = spin_num/2 * (spin_num/2 + 1)
    _SS_text = r"\braket{\bm S\cdot\bm S}"
    _eig_pops = _eig_pops[_SS_vals != 0]
    _SS_vals = _SS_vals[_SS_vals != 0]
    add_to_plots("SS", f"${_SS_text}/{_SS_text}" + r"_{\mathrm{max}}$",
                 spin_proj, _eig_pops, _SS_vals/_SS_max)

plt.figure("spect")
plt.tight_layout()
plt.savefig(fig_dir + f"spect_{name_tag}.pdf")

plt.figure("spect_X")
plt.tight_layout()
plt.savefig(fig_dir + f"spect_X_{name_tag}.pdf")

plt.figure("gaps")
plt.tight_layout()
plt.savefig(fig_dir + f"gaps_{name_tag}.pdf")

plt.figure("gaps_X")
plt.tight_layout()
plt.savefig(fig_dir + f"gaps_X_{name_tag}.pdf")

plt.figure("SS")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(fig_dir + f"SS_{name_tag}.pdf")

plt.figure("SS_X")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(fig_dir + f"SS_X_{name_tag}.pdf")
