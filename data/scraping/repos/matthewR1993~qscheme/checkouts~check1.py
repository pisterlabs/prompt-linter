# Checking out properties for two coherent states, entanglement should be zero.

import sys
try:
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
except: pass

import tensorflow as tf

from customutils.utils import *
from core.basic import *
from core.state_configurations import coherent_state, single_photon
from setup_parameters import *


sess = tf.Session()

# Parameters for states
series_length = 15
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length

# INPUT
# input_st = single_photon(series_length)
input_st = coherent_state(input_series_length, alpha=1)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY
# auxiliary_st = single_photon(series_length)
auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))


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

state_after_bs_appl = make_state_appliable(state_after_bs_unappl)

# dens_matrix_2channels = dens_matrix(state_after_bs_unappl)
dens_matrix_2channels = dens_matrix(state_after_bs_appl)


dens_matrix = trace_channel(dens_matrix_2channels, channel=2)

# Entropy
log_fn_entropy = log_entropy(dens_matrix)
print('FN log. entropy:', log_fn_entropy)

print('Lin. entropy', linear_entropy(dens_matrix))

log_negativity = negativity(dens_matrix_2channels, neg_type='logarithmic')
print('Log. negativity', log_negativity)

# Fuckin works, entanglement is 0!
