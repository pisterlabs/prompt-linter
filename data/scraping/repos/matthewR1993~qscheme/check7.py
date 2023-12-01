# Check last BS transformation
# 2 photons described by density matrix go to detector.

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
# DET_CONF = 'FIRST'  # 1st detector clicked
DET_CONF = 'THIRD'  # 3rd detector clicked
# DET_CONF = 'NONE'  # None of detectors were clicked

in_state_tf = tf.constant(input_st, tf.float64)
aux_state_tf = tf.constant(auxiliary_st, tf.float64)


# Tensor product, returns numpy array
mut_state_unappl = tf.tensordot(
    in_state_tf,
    aux_state_tf,
    axes=0,
    name=None
).eval(session=sess)

# Building density matrix

mut_state_appl = make_state_appliable(mut_state_unappl)
from time import gmtime, strftime
dm_in = dens_matrix(mut_state_appl)

# The transformation at BS

t4 = 0.4
r4 = sqrt(1 - t4**2)

dm_out = bs_densmatrix_transform(dm_in, t4, r4)

# (t4**2 - r4**2)**2    // 0.4623
#
# (t4**2 - r4**2)*t4*r4 * sqrt(2)  // -0.352
#
# t4**2 * r4**2 * 2     // 0.268

# Works:
dm_out[1, 1, 1, 1]
dm_out[1, 1, 2, 0]
dm_out[1, 1, 0, 2]

dm_out[2, 0, 1, 1]
dm_out[2, 0, 2, 0]
dm_out[2, 0, 0, 2]

dm_out[0, 2, 1, 1]
dm_out[0, 2, 2, 0]
dm_out[0, 2, 0, 2]
