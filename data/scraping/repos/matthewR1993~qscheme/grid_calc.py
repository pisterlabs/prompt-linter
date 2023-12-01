import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/usr/local/lib/python3.5/dist-packages')

import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
import argparse

from customutils.utils import *
from core.basic import *
from core.sytem_setup import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon, fock_state


# Parameters for states
series_length = 10
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length


# INPUT - the state in the first(at the bottom) channel
# input_st = single_photon(series_length)
input_st = coherent_state(input_series_length, alpha=1.0)
# input_st = fock_state(n=2, series_length=input_series_length)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY - the state in the second(on top) channel
auxiliary_st = single_photon(series_length)
# auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
# auxiliary_st = fock_state(n=2, series_length=auxiliary_series_length)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

# Measurement event, detectors configuration:
# DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
DET_CONF = 'FIRST'  # 1st detector is clicked
# DET_CONF = 'THIRD'  # 3rd detector is clicked
# DET_CONF = 'NONE'  # None of detectors were clicked

# Building a mutual state via tensor product, that returns numpy array.
mut_state_unappl = np.tensordot(input_st, auxiliary_st, axes=0)

phi_inpi_arr = np.linspace(1.2, 1.8, 31)
epr_X_phi_arr = np.zeros(len(phi_inpi_arr), dtype=complex)
probab_phi_arr = np.zeros(len(phi_inpi_arr), dtype=complex)

# The phase difference before last BS
for k in range(len(phi_inpi_arr)):
    ph_inpi = phi_inpi_arr[k]
    # ph_inpi = 1.5
    phase_diff = ph_inpi * np.pi

    # BS grids.
    r1_grid = 1
    r4_grid = 1

    r2_grid = 1
    r3_grid = 1

    # BS values range.
    T1_min = 0.78
    T1_max = 0.78
    T4_min = 1.0
    T4_max = 1.0

    T2_min = 0.84
    T2_max = 0.84
    T3_min = 0.12
    T3_max = 0.12

    # Varying BSs.
    t1_array, r1_array = bs_parameters(T1_min, T1_max, r4_grid)
    t4_array, r4_array = bs_parameters(T4_min, T4_max, r4_grid)
    t2_array, r2_array = bs_parameters(T2_min, T2_max, r2_grid)
    t3_array, r3_array = bs_parameters(T3_min, T3_max, r3_grid)

    sz = (r1_grid, r4_grid, r2_grid, r3_grid)
    det_prob_array = np.zeros(sz, dtype=complex)
    log_entropy_subs1_array = np.zeros(sz, dtype=complex)
    log_entropy_subs2_array = np.zeros(sz, dtype=complex)
    lin_entropy_subs1 = np.zeros(sz, dtype=complex)
    lin_entropy_subs2 = np.zeros(sz, dtype=complex)
    log_negativity = np.zeros(sz, dtype=complex)
    mut_information = np.zeros(sz, dtype=complex)
    full_fn_entropy = np.zeros(sz, dtype=complex)
    sqeez_dX = np.zeros(sz, dtype=complex)
    sqeez_dP = np.zeros(sz, dtype=complex)
    epr_correl_x = np.zeros(sz, dtype=complex)
    epr_correl_p = np.zeros(sz, dtype=complex)
    norm_after_det_arr = np.zeros(sz, dtype=complex)


    # Start time.
    print('Started at:', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    for n1 in range(r1_grid):
        for n4 in range(r4_grid):
            for n2 in range(r2_grid):
                for n3 in range(r3_grid):
                    print('Steps [n1, n4, n2, n3]:', n1, n4, n2, n3)
                    bs_params = {
                        't1': t1_array[n1],
                        't4': t4_array[n4],
                        't2': t2_array[n2],
                        't3': t3_array[n3],
                    }
                    final_dens_matrix, det_prob, norm = process_all(mut_state_unappl, bs_params, phase_diff=phase_diff, det_event=DET_CONF)
                    det_prob_array[n1, n4, n2, n3] = det_prob
                    norm_after_det_arr[n1, n4, n2, n3] = norm

                    # Trace one channel out of final state
                    final_traced_subs1 = trace_channel(final_dens_matrix, channel=4)
                    # print('trace of final reduced matrix 2nd channel:', np.trace(final_traced_subs1))

                    # Other channel traced
                    final_traced_subs2 = trace_channel(final_dens_matrix, channel=2)
                    # print('trace of final reduced matrix 4th channel:', np.trace(final_traced_subs2))

                    # Calculate entropy
                    # log_entanglement_subs1 = log_entropy(final_traced_subs1)
                    # log_entanglement_subs2 = log_entropy(final_traced_subs2)
                    # log_entropy_subs1_array[n1, n4, n2, n3] = log_entanglement_subs1
                    # log_entropy_subs2_array[n1, n4, n2, n3] = log_entanglement_subs2

                    # Full entropy and the mutual information
                    # final_reorg_matr = reorganise_dens_matrix(final_dens_matrix)
                    # full_entr = log_entropy(final_reorg_matr)

                    # mut_information[n1, n4, n2, n3] = log_entanglement_subs1 + log_entanglement_subs2 - full_entr
                    # full_fn_entropy[n1, n4, n2, n3] = full_entr

                    log_negativity[n1, n4, n2, n3] = negativity(final_dens_matrix, neg_type='logarithmic')
                    # print('Log. negativity: ', log_negativity[n1, n4, n2, n3])

                    # Squeezing quadratures.
                    dX, dP = squeezing_quadratures(final_dens_matrix, channel=1)
                    # print('dX:', dX, ' dP:', dP)
                    sqeez_dX[n1, n4, n2, n3] = dX
                    sqeez_dP[n1, n4, n2, n3] = dP

                    # ERP correlations.
                    epr_x, epr_p = erp_squeezing_correlations(final_dens_matrix)
                    epr_correl_x[n1, n4, n2, n3] = epr_x
                    epr_correl_p[n1, n4, n2, n3] = epr_p
                    # print('epr_X:', epr_x, ' epr_P:', epr_p)

    # print('dXdP:', sqeez_dX[0, 0, 0, 0] * sqeez_dP[0, 0, 0, 0])
    # print('EPR dXdP:', epr_correl_x[0, 0, 0, 0] * epr_correl_p[0, 0, 0, 0])
    # print('EPR X:', np.sqrt(2) * epr_correl_x[0, 0, 0, 0])
    # print('EPR P:', epr_correl_p[0, 0, 0, 0])
    # print('Prob of det:', det_prob_array[0, 0, 0, 0])
    # print('Norm after det:', norm_after_det_arr[0, 0, 0, 0])

    epr_X_phi_arr[k] = np.sqrt(2) * epr_correl_x[0, 0, 0, 0]
    probab_phi_arr[k] = det_prob_array[0, 0, 0, 0]

plt.plot(phi_inpi_arr, epr_X_phi_arr)
plt.xlabel('$Phase, [\pi]$')
plt.ylabel(r'$\sqrt{2} \ \Delta[X^{(1)} - X^{(2)}]^{(out)}$')
plt.title('T1: {}, T4: {}, T2: {}, T3: {}'.format(T1_max, T4_max, T2_max, T3_max))
plt.grid(True)
plt.show()

# plt.plot(phi_inpi_arr, probab_phi_arr)
# plt.xlabel('$Phase, [\pi]$')
# plt.xlabel('$Probability$')
# plt.show()
