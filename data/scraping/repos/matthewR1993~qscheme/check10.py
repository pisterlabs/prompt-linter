# Check squeezing right before the detection.

import sys
try:
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
except: pass

import numpy as np
import tensorflow as tf

from customutils.utils import *
from core.basic import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon, fock_state
from setup_parameters import *


sess = tf.Session()

# Parameters for states
series_length = 10
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length

# Set up input and auxiliary states as a Taylor series
# input_st[n] = state with 'n' photons !!!a

# INPUT
# input_st = single_photon(series_length)
input_st = coherent_state(input_series_length, alpha=1)
# input_st = fock_state(n=2, series_length=input_series_length)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY
auxiliary_st = single_photon(series_length)
# auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
# auxiliary_st = fock_state(n=2, series_length=auxiliary_series_length)
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

dm1 = dens_matrix(make_state_appliable(mut_state_unappl))

dX_1, dP_1 = squeezing_quadratures(dm1, channel=1)
# Works, both are 0.5

# First BS
state_after_bs_unappl = bs2x2_transform(t1, r1, mut_state_unappl)

dm2 = dens_matrix(make_state_appliable(state_after_bs_unappl))

dX_2, dP_2 = squeezing_quadratures(dm2, channel=1)
# The variance here is different because of the mixing at BS.
