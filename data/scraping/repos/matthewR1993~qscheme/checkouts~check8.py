# Full scheme with detection of 'FIRST' for two photons,
# doesn't work, gives different entropy

import sys
try:
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
except: pass

import numpy as np
import tensorflow as tf
from qutip import (wigner, super_tensor, Qobj)
from time import gmtime, strftime

from customutils.utils import *
from core.basic import *
from core.state_configurations import coherent_state, single_photon
from setup_parameters import *


sess = tf.Session()

# Parameters for states
series_length = 3
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length

# Set up input and auxiliary states as a Taylor series
# input_st[n] = state with 'n' photons !!!a

# INPUT
input_st = single_photon(series_length)
# input_st = coherent_state(input_series_length, alpha=1)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY
auxiliary_st = single_photon(series_length)
# auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

# Measurement event, detectors configuration:
# DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
DET_CONF = 'FIRST'  # 1st detector clicked
# DET_CONF = 'THIRD'  # 3rd detector clicked
# DET_CONF = 'NONE'  # None of detectors were clicked

in_state_tf = tf.constant(input_st, tf.float64)
aux_state_tf = tf.constant(auxiliary_st, tf.float64)

# tensor product, returns numpy array
mut_state_unappl = tf.tensordot(
    in_state_tf,
    aux_state_tf,
    axes=0,
    name=None
).eval(session=sess)

# First BS
state_after_bs_unappl = bs2x2_transform(t1, r1, mut_state_unappl)

# 2d and 3rd BS
state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs_unappl)

# Detection
# Gives not normalised state
state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event=DET_CONF)
# Calculating the norm
norm_after_det = state_norm(state_after_dett_unappl)
print('Norm after det.:', norm_after_det)
# normalised state
state_after_dett_unappl_norm = state_after_dett_unappl / norm_after_det


# Build dens matrix and trace
dens_matrix_2channels = dens_matrix_with_trace(state_after_dett_unappl_norm, state_after_dett_unappl_norm)

# The new method, works
# dens_matrix_2channels = dens_matrix_with_trace_new(state_after_dett_unappl_norm, state_after_dett_unappl_norm)

# Disable a phase addition.
dens_matrix_2channels_withph = dens_matrix_2channels


# log_entropy_array = np.zeros((r4_grid), dtype=complex)
# log_negativity = np.zeros((r4_grid), dtype=complex)


t4 = 0.82
r4 = sqrt(1 - t4**2)


# Transformation at last BS
# Trim for better performance,
# trim_size=10 for series_len=10
# trim_size=4 for series_len=3
trim_size = 4
final_dens_matrix = bs_densmatrix_transform(dens_matrix_2channels_withph[:trim_size, :trim_size, :trim_size, :trim_size], t4, r4)

# Trace one channel out of final state
final_traced = trace_channel(final_dens_matrix, channel=4)
print('trace of final reduced matrix 2nd channel:', np.trace(final_traced))

# Other channel traced
final_traced_4th = trace_channel(final_dens_matrix, channel=2)
print('trace of final reduced matrix 4th channel:', np.trace(final_traced_4th))


# TODO Gives different entropy for different reduced density matrices

log_entropy(final_traced)
log_entropy(final_traced_4th)

entropy = 0
w, v = np.linalg.eig(final_traced)
# for n in range(len(final_traced)):
#     if w[n] != 0:
#         entropy = entropy - w[n] * np.log2(w[n])


entr1 = - (1 - 2*(t4**2)/3) * np.log2(1 - 2*t4**2/3) - 2*(t4**2)/3 * np.log2(2*(t4**2)/3)
entr2 = - ((1 + 2*(t4**2))/3) * np.log2((1 + 2*(t4**2))/3) - (2/3)*(1 - t4**2) * np.log2((2/3)*(1 - t4**2))


# Calculate entropy
# log_entanglement = log_entropy(final_traced)
log_entanglement = log_entropy(final_traced_4th)  # other channel traced matrix
print('FN entropy: ', np.real(log_entanglement))
# log_entropy_array[i, j] = log_entanglement

# Logarithmic entropy difference
print('FN entropy difference: ', log_entanglement - log_entropy(final_traced_4th))

# lin_entropy[i, j] = np.real(linear_entropy(final_traced))
#lin_entropy[i, j] = np.real(linear_entropy(final_traced_4th))  # other channel traced matrix
#print('Lin. entropy: ', lin_entropy[i, j])

# Linear entropy difference
#print('Linear entropy difference: ', lin_entropy[i, j] - linear_entropy(final_traced_4th))

log_negativity = negativity(final_dens_matrix, neg_type='logarithmic')
print('Log. negativity: ', log_negativity)
