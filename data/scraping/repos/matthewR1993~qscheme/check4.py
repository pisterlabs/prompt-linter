# Two beam splitters with loses, INCORRECT description of losses.

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
series_length = 14
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length


# INPUT
# input_st = single_photon(series_length)
input_st = coherent_state(input_series_length, alpha=1)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY
auxiliary_st = single_photon(series_length)
# auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))


in_state_tf = tf.constant(input_st, tf.float64)
aux_state_tf = tf.constant(auxiliary_st, tf.float64)

# A tensor product, returns numpy array
mut_state_unappl = tf.tensordot(
    in_state_tf,
    aux_state_tf,
    axes=0,
    name=None
).eval(session=sess)

# Adding absorption
a1 = 0.8
T1_max = 1 - a1**2
t1 = sqrt(T1_max/2)
r1 = sqrt(1 - pow(t1, 2) - pow(a1, 2))

# The first BS
state_after_bs1_unappl = bs2x2_transform(t1, r1, mut_state_unappl)

grd = 30

# Varying BS2
a2 = 0.0
t2_max = sqrt(1 - a2**2)
t2_arr = np.linspace(0, t2_max, grd)
r2_arr = np.zeros(grd)
for i in range(grd):
    r2_arr[i] = sqrt(1 - pow(t2_arr[i], 2) - pow(a2, 2))


ph_inpi = 0.0
phase_mod = ph_inpi * np.pi

log_entr_arr = np.zeros(grd)
log_neg_arr = np.zeros(grd)

for i in range(grd):
    print('step:', i)
    t2 = t2_arr[i]
    r2 = r2_arr[i]
    # The phase modulation
    state_after_phmod_unappl = phase_modulation_state(state_after_bs1_unappl, phase_mod)

    # The second BS
    state_after_bs2_unappl = bs2x2_transform(t2, r2, state_after_phmod_unappl)

    state_after_bs2_appl = make_state_appliable(state_after_bs2_unappl)

    dens_matrix_2channels = dens_matrix(state_after_bs2_appl)

    reduced_dens_matrix = trace_channel(dens_matrix_2channels, channel=2)

    # Entanglement
    log_fn_entropy = log_entropy(reduced_dens_matrix)
    log_entr_arr[i] = log_fn_entropy
    # print('FN log. entropy:', log_fn_entropy)

    log_negativity = negativity(dens_matrix_2channels, neg_type='logarithmic')
    log_neg_arr[i] = log_negativity
    # print('Log. negativity', log_negativity)


fig, ax = plt.subplots()
ax.plot(np.square(t2_arr), log_entr_arr, label=r'$Log. FN \ entropy$')
ax.plot(np.square(t2_arr), log_neg_arr, label=r'$Log. negativity$')
plt.title('Phase = {0}pi, a1 = {1}, a2 = {2}'.format(ph_inpi, a1, a2))
plt.xlabel('$T_{2}$')
plt.ylabel('$Entanglement$')
plt.legend()
plt.grid(True)
plt.show()
