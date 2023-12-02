# A solution with a scaled grid.

import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/usr/local/lib/python3.5/dist-packages')
elif platform.system() == 'Darwin':
    sys.path.append('/Users/matvei/PycharmProjects/qscheme')

from time import gmtime, strftime
import numpy as np
import argparse

from customutils.utils import *
from core.basic import *
from core.sytem_setup import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon, fock_state


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--det", help="Detection", type=str, required=True)
parser.add_argument("-p", "--phase", help="Phase in pi", type=float, required=True)
args = parser.parse_args()

save_root = '/Users/matvei/PycharmProjects/qscheme/results/res16/'
# save_root = '/home/matthew/qscheme/results/res16/'
fname = 'coh(chan-1)_single(chan-2)_phase-{}pi_det-{}.npy'.format(args.phase, args.det)
print('Saving path:', save_root + fname)

# Parameters for states
series_length = 10
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length


# INPUT - the state in the first(at the bottom) channel
# input_st = single_photon(series_length)
input_st = coherent_state(input_series_length, alpha=1)
# input_st = fock_state(n=2, series_length=input_series_length)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY - the state in the second(on top) channel
auxiliary_st = single_photon(series_length)
# auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
# auxiliary_st = fock_state(n=2, series_length=auxiliary_series_length)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

# Measurement event, detectors configuration:
# DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
# DET_CONF = 'FIRST'  # 1st detector is clicked
# DET_CONF = 'THIRD'  # 3rd detector is clicked
# DET_CONF = 'NONE'  # None of detectors were clicked
DET_CONF = args.det


# Building a mutual state via tensor product.
mut_state_unappl = np.tensordot(input_st, auxiliary_st, axes=0)

QUANT_T0_MINIMIZE = 'EPR_P'
SCALING_DEPTH = 3

# The phase difference before last BS
# ph_inpi = 0.0
ph_inpi = args.phase
phase_diff = ph_inpi * np.pi

# BS grids.
r1_grid = 5
r4_grid = 5

r2_grid = 5
r3_grid = 5


T1_min_arr = np.zeros(SCALING_DEPTH + 1, dtype=float)
T1_max_arr = np.zeros(SCALING_DEPTH + 1, dtype=float)
T4_min_arr = np.zeros(SCALING_DEPTH + 1, dtype=float)
T4_max_arr = np.zeros(SCALING_DEPTH + 1, dtype=float)

T2_min_arr = np.zeros(SCALING_DEPTH + 1, dtype=float)
T2_max_arr = np.zeros(SCALING_DEPTH + 1, dtype=float)
T3_min_arr = np.zeros(SCALING_DEPTH + 1, dtype=float)
T3_max_arr = np.zeros(SCALING_DEPTH + 1, dtype=float)

min_bound = 1e-5
max_bound = 1 - 1e-5

# Starting BS parameters grid range.
T1_min_arr[0] = 0.0
T1_max_arr[0] = 1.0
T4_min_arr[0] = 0.0
T4_max_arr[0] = 1.0

T2_min_arr[0] = min_bound
T2_max_arr[0] = max_bound
T3_min_arr[0] = min_bound
T3_max_arr[0] = max_bound


if __name__ == "__main__":
    print('Started at:', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    # Scaling step loop.
    for p in range(SCALING_DEPTH):
        print('Depth:', p)
        # BS values range.
        T1_min = T1_min_arr[p]
        T1_max = T1_max_arr[p]
        T4_min = T4_min_arr[p]
        T4_max = T4_max_arr[p]

        T2_min = T2_min_arr[p]
        T2_max = T2_max_arr[p]
        T3_min = T3_min_arr[p]
        T3_max = T3_max_arr[p]

        T1_step = abs(T1_max - T1_min) / (r1_grid - 1)
        T4_step = abs(T4_max - T4_min) / (r4_grid - 1)
        T2_step = abs(T2_max - T2_min) / (r2_grid - 1)
        T3_step = abs(T3_max - T3_min) / (r3_grid - 1)

        # Varying BSs.
        t1_array, r1_array = bs_parameters(T1_min, T1_max, r1_grid)
        t4_array, r4_array = bs_parameters(T4_min, T4_max, r4_grid)
        t2_array, r2_array = bs_parameters(T2_min, T2_max, r2_grid)
        t3_array, r3_array = bs_parameters(T3_min, T3_max, r3_grid)

        det_prob_array = np.full((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex, fill_value=None)
        log_entropy_subs1_array = np.full((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex, fill_value=None)
        log_entropy_subs2_array = np.full((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex, fill_value=None)
        lin_entropy_subs1 = np.full((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex, fill_value=None)
        lin_entropy_subs2 = np.full((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex, fill_value=None)
        log_negativity = np.full((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex, fill_value=None)
        mut_information = np.full((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex, fill_value=None)
        full_fn_entropy = np.full((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex, fill_value=None)
        sqeez_dX = np.full((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex, fill_value=None)
        sqeez_dP = np.full((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex, fill_value=None)
        epr_correl_x = np.full((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex, fill_value=None)
        epr_correl_p = np.full((r1_grid, r4_grid, r2_grid, r3_grid), dtype=complex, fill_value=None)

        for n1 in range(r1_grid):
            for n4 in range(r4_grid):
                for n2 in range(r2_grid):
                    for n3 in range(r3_grid):
                        print('Steps [n1, n4, n2, n3]:', n1, n4, n2, n3)
                        bs_params = {
                            't1': t1_array[n1],
                            'r1': r1_array[n1],
                            't4': t4_array[n4],
                            'r4': r4_array[n4],
                            't2': t2_array[n2],
                            'r2': r2_array[n2],
                            't3': t3_array[n3],
                            'r3': r3_array[n3],
                        }
                        final_dens_matrix, det_prob = process_all(mut_state_unappl, bs_params, phase_diff=phase_diff, det_event=DET_CONF)
                        if final_dens_matrix is None or det_prob is None:
                            print('Warning: the norm is zero.')
                            pass

                        det_prob_array[n1, n4, n2, n3] = det_prob

                        # Trace one channel out of final state
                        final_traced_subs1 = trace_channel(final_dens_matrix, channel=4)

                        # Other channel traced
                        final_traced_subs2 = trace_channel(final_dens_matrix, channel=2)

                        # Calculate entropy
                        log_entanglement_subs1 = log_entropy(final_traced_subs1)
                        log_entanglement_subs2 = log_entropy(final_traced_subs2)
                        log_entropy_subs1_array[n1, n4, n2, n3] = log_entanglement_subs1
                        log_entropy_subs2_array[n1, n4, n2, n3] = log_entanglement_subs2

                        # Full entropy and the mutual information
                        final_reorg_matr = reorganise_dens_matrix(final_dens_matrix)
                        full_entr = log_entropy(final_reorg_matr)

                        mut_information[n1, n4, n2, n3] = log_entanglement_subs1 + log_entanglement_subs2 - full_entr
                        full_fn_entropy[n1, n4, n2, n3] = full_entr

                        log_negativity[n1, n4, n2, n3] = negativity(final_dens_matrix, neg_type='logarithmic')

                        # Squeezing quadratures.
                        dX, dP = squeezing_quadratures(final_dens_matrix, channel=1)
                        sqeez_dX[n1, n4, n2, n3] = dX
                        sqeez_dP[n1, n4, n2, n3] = dP

                        # ERP correlations.
                        epr_x, epr_p = erp_squeezing_correlations(final_dens_matrix)
                        epr_correl_x[n1, n4, n2, n3] = epr_x
                        epr_correl_p[n1, n4, n2, n3] = epr_p

        epr_x_min = np.nanmin(epr_correl_x)
        epr_p_min = np.nanmin(epr_correl_p)
        dX_min = np.nanmin(sqeez_dX)
        dP_min = np.nanmin(sqeez_dP)

        uncert = np.multiply(sqeez_dX, sqeez_dP)

        dX_min_ind = list(np.unravel_index(np.nanargmin(sqeez_dX, axis=None), sqeez_dX.shape))
        dP_min_ind = list(np.unravel_index(np.nanargmin(sqeez_dP, axis=None), sqeez_dP.shape))
        epr_x_min_ind = list(np.unravel_index(np.nanargmin(epr_correl_x, axis=None), epr_correl_x.shape))
        epr_p_min_ind = list(np.unravel_index(np.nanargmin(epr_correl_p, axis=None), epr_correl_p.shape))

        # Calculate the minimun.
        if QUANT_T0_MINIMIZE is 'EPR_X':
            ind = epr_x_min_ind
            print('EPR_X min value:', epr_x_min)
            print('EPR_X min value:', epr_correl_x[tuple(epr_x_min_ind)])
        elif QUANT_T0_MINIMIZE is 'EPR_P':
            ind = epr_p_min_ind
            print('EPR_P min value:', epr_p_min)
            print('EPR_P min value:', epr_correl_p[tuple(epr_p_min_ind)])
        elif QUANT_T0_MINIMIZE is 'dX':
            ind = dX_min_ind
            print('dX min value:', dX_min)
            print('dX min value:', sqeez_dX[tuple(dX_min_ind)])
        elif QUANT_T0_MINIMIZE is 'dP':
            ind = dP_min_ind
            print('dP min value:', dP_min)
            print('dP min value:', sqeez_dP[tuple(dP_min_ind)])
        else:
            raise ValueError

        # Minimizing set of parameters T1, T2, T3, T4:
        T1_mid = t1_array[ind[0]]
        T4_mid = t4_array[ind[1]]
        T2_mid = t2_array[ind[2]]
        T3_mid = t3_array[ind[3]]

        print('T1_mid:', T1_mid)
        print('T4_mid:', T4_mid)
        print('T2_mid:', T2_mid)
        print('T3_mid:', T3_mid)

        # Building a T grid, for a new scale step.
        T1_min_arr[p + 1] = T1_mid - T1_step
        T1_max_arr[p + 1] = T1_mid + T1_step
        T4_min_arr[p + 1] = T4_mid - T4_step
        T4_max_arr[p + 1] = T4_mid + T4_step

        T2_min_arr[p + 1] = T2_mid - T2_step
        T2_max_arr[p + 1] = T2_mid + T2_step
        T3_min_arr[p + 1] = T3_mid - T3_step
        T3_max_arr[p + 1] = T3_mid + T3_step

        # Check boundaries.
        if T1_min_arr[p + 1] < 0:
            T1_min_arr[p + 1] = 0
        if T1_max_arr[p + 1] > 1:
            T1_max_arr[p + 1] = 1
        if T4_min_arr[p + 1] < 0:
            T4_min_arr[p + 1] = 0
        if T4_max_arr[p + 1] > 1:
            T4_max_arr[p + 1] = 1
        if T2_min_arr[p + 1] < min_bound:
            T2_min_arr[p + 1] = min_bound
        if T2_max_arr[p + 1] > max_bound:
            T2_max_arr[p + 1] = max_bound
        if T3_min_arr[p + 1] < min_bound:
            T3_min_arr[p + 1] = min_bound
        if T3_max_arr[p + 1] > max_bound:
            T3_max_arr[p + 1] = max_bound

        print('T1_min next:', T1_min_arr[p + 1])
        print('T1_max next:', T1_max_arr[p + 1])

        print('T4_min next:', T4_min_arr[p + 1])
        print('T4_max next:', T4_max_arr[p + 1])

        print('T2_min next:', T2_min_arr[p + 1])
        print('T2_max next:', T2_max_arr[p + 1])

        print('T3_min next:', T3_min_arr[p + 1])
        print('T3_max next:', T3_max_arr[p + 1])

        # Save it.
        if p == SCALING_DEPTH - 1:
            data = {
                'det_prob': det_prob_array,
                'log_negativity': log_negativity,
                'mut_inform': mut_information,
                'squeez_dx': sqeez_dX,
                'squeez_dp': sqeez_dP,
                'epr_correl_x': epr_correl_x,
                'epr_correl_p': epr_correl_p,
                'det_conf': DET_CONF,
                'phase': phase_diff,
                't1_arr': t1_array,
                't4_arr': t4_array,
                't2_arr': t2_array,
                't3_arr': t3_array,
                'states_config': 'coh(chan-1)_single(chan-2)'
            }
            np.save(save_root + fname, data)
