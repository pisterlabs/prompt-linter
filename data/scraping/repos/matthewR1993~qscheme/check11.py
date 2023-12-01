# EPR normalisation for different combinations.
# Check the squeezing right before the detection.

import sys
try:
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
except: pass

import tensorflow as tf

from customutils.utils import *
from core.basic import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon, fock_state


sess = tf.Session()

# Parameters for states
series_length = 10
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length

# INPUT
input_st = single_photon(series_length)
# input_st = coherent_state(input_series_length, alpha=1)
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

dm = dens_matrix(make_state_appliable(mut_state_unappl))

# ERP correlations
erp_x, erp_p = erp_squeezing_correlations(dm)

# For coherent + single:
# erp_x, erp_p = 1, 1

# For single + single:
# erp_x, erp_p = 1.2247, 1.2247 <=> sqrt(3/2), sqrt(3/2)

# For coherent + coherent:
# erp_x, erp_p = sqrt(1/2), sqrt(1/2)

# For vac + vac:
# erp_x, erp_p = sqrt(1/2), sqrt(1/2)
