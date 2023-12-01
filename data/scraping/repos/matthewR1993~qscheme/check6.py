# Two photons go directly to detector

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


# tensor product, returns numpy array
mut_state_unappl = tf.tensordot(
    in_state_tf,
    aux_state_tf,
    axes=0,
    name=None
).eval(session=sess)

# random parameters
t2 = 0.4
r2 = sqrt(1 - t2**2)

t3 = 0.8
r3 = sqrt(1 - t3**2)


# Two bs
state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, mut_state_unappl)

# Detection
state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event=DET_CONF)
# works
