# Checking influence of absorption as position of channels with loses.

# A coherent state and a single photon with two beam splitters and phase modul.
# Absorptions comute!!!

import sys
try:
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
except: pass

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
from qutip import (wigner, super_tensor, Qobj)
from time import gmtime, strftime

from customutils.utils import *
from core.basic import *
from core.state_configurations import coherent_state, single_photon, squeezed_vacuum, squeezed_coherent_state
from setup_parameters import *


sess = tf.Session()

# Parameters for states
series_length = 4
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length


# INPUT
input_st = single_photon(input_series_length)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY
auxiliary_st = single_photon(auxiliary_series_length)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))


in_state_tf = tf.constant(input_st, tf.complex128)
aux_state_tf = tf.constant(auxiliary_st, tf.complex128)

# A tensor product, returns numpy array
mut_state_unappl = tf.tensordot(
    in_state_tf,
    aux_state_tf,
    axes=0,
    name=None
).eval(session=sess)

# First, channels with loses are located before BS

t1 = sqrt(0.6)
r1 = sqrt(1 - pow(t1, 2))

t2 = sqrt(0.5)
r2 = sqrt(1 - pow(t2, 2))

t3 = sqrt(0.5)
r3 = sqrt(1 - pow(t3, 2))

state_aft2bs_unappl = two_bs2x4_transform(t1, r1, t2, r2, mut_state_unappl)

# Make state appl
state_aft2bs_appl = make_state_appliable_4ch(state_aft2bs_unappl)

# Form density matrix and trace
dm = dens_matrix_4ch(state_aft2bs_appl)

# Trace loosy channels 1st and 3rd
size = len(dm)
dm_aft_trace_appl = np.zeros((size,) * 4, dtype=complex)

for p2 in range(size):
    for p2_ in range(size):
        for p4 in range(size):
            for p4_ in range(size):
                matrix_sum = 0
                for k1 in range(size):
                    for k3 in range(size):
                        matrix_sum = matrix_sum + dm[k1, p2, k3, p4, k1, p2_, k3, p4_]
                dm_aft_trace_appl[p2, p4, p2_, p4_] = matrix_sum


# last BS transformation
final_dens_matrix = bs_densmatrix_transform(dm_aft_trace_appl, t3, r3)


# Second method.
# First, channels with loses are located after BS
state_aft_1st_bs_unappl = bs2x2_transform(t3, r3, mut_state_unappl)

# r1, t1, r2, t2
state_aft2bs_unappl_2 = two_bs2x4_transform(t1, r1, t2, r2, state_aft_1st_bs_unappl)

# Make state appl
state_aft2bs_appl_2 = make_state_appliable_4ch(state_aft2bs_unappl_2)

# Form density matrix and trace
dm_2 = dens_matrix_4ch(state_aft2bs_appl_2)

# Trace loosy channels
size = len(dm_2)
dm_aft_trace_appl_2 = np.zeros((size,) * 4, dtype=complex)

# trace 1st and 3rd channels
for p2 in range(size):
    for p2_ in range(size):
        for p4 in range(size):
            for p4_ in range(size):
                matrix_sum = 0
                for k1 in range(size):
                    for k3 in range(size):
                        matrix_sum = matrix_sum + dm_2[k1, p2_, k3, p4, k1, p2_, k3, p4_]
                dm_aft_trace_appl_2[p2, p4, p2_, p4_] = matrix_sum


matr_diff = dm_aft_trace_appl_2 - final_dens_matrix[:7, :7, :7, :7]

densm_diff = np.sum(np.abs(matr_diff))

# prob distr diff
pd1 = prob_distr(final_dens_matrix[:7, :7, :7, :7])

pd2 = prob_distr(dm_aft_trace_appl_2)

pd_diff = pd1 - pd2

prob_diff = np.sum(np.abs(pd_diff))
print('Prob diff:', prob_diff)

# Dens matr abs diff.
plt.matshow(np.abs(pd_diff))
plt.title('Abs(diff_dens)')
plt.colorbar()
plt.show()


