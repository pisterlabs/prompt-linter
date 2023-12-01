import numpy as np
from time import gmtime, strftime

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


# INPUT - the state in the first(at the bottom) channel. (chan-1).
input_st = single_photon(series_length)
# input_st = coherent_state(input_series_length, alpha=1)
# input_st = fock_state(n=2, series_length=input_series_length)
print('Input state norm:', get_state_norm(input_st))

# AUXILIARY - the state in the second(on top) channel. (chan-2).
# auxiliary_st = single_photon(series_length)
auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)
# auxiliary_st = fock_state(n=2, series_length=auxiliary_series_length)
print('Auxiliary state norm:', get_state_norm(auxiliary_st))

states_config = 'single(chan-1)_coher(chan-2)'
# states_config = 'coher(chan-1)_single(chan-2)'
# states_config = 'single(chan-1)_single(chan-2)'

# Measurement event, detectors configuration:
# DET_CONF = 'BOTH'  # both 1st and 3rd detectors clicked
DET_CONF = 'FIRST'  # 1st detector is clicked
# DET_CONF = 'THIRD'  # 3rd detector is clicked
# DET_CONF = 'NONE'  # None of detectors were clicked
# DET_CONF = args.det

mut_state_unappl = np.tensordot(input_st, auxiliary_st, axes=0)

# The phase difference before last BS
ph_inpi = 0.0
# ph_inpi = args.phase
phase_diff = ph_inpi * np.pi

phase_mod_channel = 1

save_root = '/Users/matvei/PycharmProjects/qscheme/results/res30/'
# save_root = '/home/matvei/qscheme/results/res27/'
fname = 'disabled_det_{}_phase-{:.4f}pi_det-{}_phase_chan-{}.npy'.format(states_config, ph_inpi, DET_CONF, phase_mod_channel)
print('Saving path:', save_root + fname)

# BS grids.
r1_grid = 1
r4_grid = 41

r2_grid = 41
r3_grid = 1

min_bound = 1e-5
max_bound = 1 - 1e-5

# BS values range.
T1_min = 0.7
T1_max = 0.7
T4_min = 0.0
T4_max = 1.0

T2_min = min_bound
T2_max = max_bound
T3_min = 1
T3_max = 1

# Varying BSs. Small t, r parameters, with step regarding to big "T".
t1_array, r1_array = bs_parameters(T1_min, T1_max, r1_grid)
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
final_dens_matrix_list = []


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

                final_dens_matrix, det_prob, norm = process_all(
                    mut_state_unappl,
                    bs_params,
                    phase_diff=phase_diff,
                    phase_mod_channel=phase_mod_channel,
                    det_event=DET_CONF
                )

                if final_dens_matrix is None or det_prob is None:
                    print('Warning: the norm is zero.')
                    pass

                det_prob_array[n1, n4, n2, n3] = det_prob
                norm_after_det_arr[n1, n4, n2, n3] = norm
                # final_dens_matrix_list.append({'dm': final_dens_matrix, 'keys': [n1, n4, n2, n3]})

                # Trace one channel out of final state
                # final_traced_subs1 = trace_channel(final_dens_matrix, channel=4)
                # print('trace of final reduced matrix 2nd channel:', np.trace(final_traced_subs1))

                # Other channel traced
                # final_traced_subs2 = trace_channel(final_dens_matrix, channel=2)
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
                # print('erp_X:', erp_x, ' erp_P:', erp_p)

# Save it.
fl = {
    'det_prob': det_prob_array,
    'norm_aft_det': norm_after_det_arr,
    # 'final_dens_matrix': final_dens_matrix_list,
    'log_negativity': log_negativity,
    # 'mut_inform': mut_information,
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
    'states_config': states_config
}
np.save(save_root + fname, fl)
