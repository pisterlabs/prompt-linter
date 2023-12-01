# A coherent state plus a single photon, EPR variance.
# 1) alpha=1, phase=?, vary: t1, t2
# 2) alpha=1, t2=1, vary: t1, phase
# 3) alpha=1, t2=1/sqrt(2), vary: t1, phase
# 4) alpha=1, t1=1, vary: t2, phase
# 5) alpha=1, t1=1/sqrt(2), vary: t2, phase

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from customutils.utils import *
from core.basic import *
from core.sytem_setup import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon
from core.optimized import transformations as trans


# Parameters for states
series_length = 8
input_series_length = series_length
auxiliary_series_length = series_length
max_power = input_series_length + auxiliary_series_length


# INPUT - the state in the first(at the bottom) channel
input_st = single_photon(series_length)
# AUXILIARY - the state in the second(on top) channel
auxiliary_st = coherent_state(auxiliary_series_length, alpha=1)

input_state = np.tensordot(input_st, auxiliary_st, axes=0)


# 1) Works!
# phase = 0.0 * np.pi
#
# t1_grid = 40
# t2_grid = 40
#
# t1_arr = np.linspace(0, 1, t1_grid)
# t2_arr = np.linspace(0, 1, t2_grid)
#
# sz = (t1_grid, t2_grid)
# epr_correl_x = np.zeros(sz, dtype=complex)
#
#
# for n1 in range(len(t1_arr)):
#     for n2 in range(len(t2_arr)):
#         print('n1, n2:', n1, n2)
#         t1 = t1_arr[n1]
#         r1 = sqrt(1 - t1**2)
#         t2 = t2_arr[n2]
#         r2 = sqrt(1 - t2**2)
#         # BS1.
#         state1_unappl = bs2x2_transform(t1, r1, input_state)
#         # Phase modulation.
#         state2_unappl = phase_modulation_state(state1_unappl, phase)
#         # BS2.
#         state3_unappl = bs2x2_transform(t2, r2, state2_unappl)
#
#         # Form a density matrix. It is applied.
#         dm = dens_matrix(make_state_appliable(state3_unappl))
#
#         epr_x, epr_p = erp_squeezing_correlations(dm)
#         epr_correl_x[n1, n2] = epr_x
#
#
# print('A real part:', np.sum(np.real(epr_correl_x)))
# print('An image part:', np.sum(np.imag(epr_correl_x)))
#
# print('Minimum:', np.amin(np.real(epr_correl_x)))
# print('Maximum:', np.amax(np.real(epr_correl_x)))
# # Minimum: 0.9999087524316295
# # Maximum: 1.00012260380289
#
#
# plt.imshow(np.real(epr_correl_x), origin='lower', cmap=cm.GnBu_r)
# plt.colorbar()
# plt.xlabel('T2')
# plt.ylabel('T1')
# plt.show()


# 2,3) Works!

# phase_grid = 40
# t1_grid = 40
#
# phase_arr = np.linspace(0, 2 * np.pi, phase_grid)
# t1_arr = np.linspace(0, 1, t1_grid)
# t2_arr = np.array([1])
#
# t2 = t2_arr[0]
# r2 = np.sqrt(1 - t2_arr[0]**2)
#
# sz = (t1_grid, phase_grid)
# epr_correl_x = np.zeros(sz, dtype=complex)
#
# for n1 in range(len(t1_arr)):
#     for p in range(len(phase_arr)):
#         print('n1, p:', n1, p)
#         t1 = t1_arr[n1]
#         r1 = sqrt(1 - t1**2)
#         phase = phase_arr[p]
#         # BS1.
#         state1_unappl = bs2x2_transform(t1, r1, input_state)
#         # Phase modulation.
#         state2_unappl = phase_modulation_state(state1_unappl, phase)
#         # BS2.
#         state3_unappl = bs2x2_transform(t2, r2, state2_unappl)
#
#         # Form a density matrix. It is applied.
#         dm = dens_matrix(make_state_appliable(state3_unappl))
#
#         epr_x, epr_p = erp_squeezing_correlations(dm)
#         epr_correl_x[n1, p] = epr_x
#
#
# print('A real part:', np.sum(np.real(epr_correl_x)))
# print('An image part:', np.sum(np.imag(epr_correl_x)))
#
# print('Minimum:', np.amin(np.real(epr_correl_x)))
# print('Maximum:', np.amax(np.real(epr_correl_x)))
# # Minimum: 0.5007051822120813
# # Maximum: 1.4993403397160232
#
#
# plt.imshow(np.real(epr_correl_x), origin='lower', cmap=cm.GnBu_r)
# plt.colorbar()
# plt.xlabel('phase')
# plt.ylabel('T1')
# plt.show()


# 4, 5) Works
# phase_grid = 40
# t2_grid = 40
#
# phase_arr = np.linspace(0, 2 * np.pi, phase_grid)
# t2_arr = np.linspace(0, 1, t2_grid)
# t1_arr = np.array([1/sqrt(2)])
#
# t1 = t1_arr[0]
# r1 = np.sqrt(1 - t1_arr[0]**2)
#
# sz = (t2_grid, phase_grid)
# epr_correl_x = np.zeros(sz, dtype=complex)
#
# for n2 in range(len(t2_arr)):
#     for p in range(len(phase_arr)):
#         print('n2, p:', n2, p)
#         t2 = t2_arr[n2]
#         r2 = sqrt(1 - t2**2)
#         phase = phase_arr[p]
#         # BS1.
#         state1_unappl = bs2x2_transform(t1, r1, input_state)
#         # Phase modulation.
#         state2_unappl = phase_modulation_state(state1_unappl, phase)
#         # BS2.
#         state3_unappl = bs2x2_transform(t2, r2, state2_unappl)
#
#         # Form a density matrix. It is applied.
#         dm = dens_matrix(make_state_appliable(state3_unappl))
#
#         epr_x, epr_p = erp_squeezing_correlations(dm)
#         epr_correl_x[n2, p] = epr_x
#
#
# print('A real part:', np.sum(np.real(epr_correl_x)))
# print('An image part:', np.sum(np.imag(epr_correl_x)))
#
# print('Minimum:', np.amin(np.real(epr_correl_x)))
# print('Maximum:', np.amax(np.real(epr_correl_x)))
#
#
# plt.imshow(np.real(epr_correl_x), origin='lower', cmap=cm.GnBu_r)
# plt.colorbar()
# plt.xlabel('phase')
# plt.ylabel('T2')
# plt.show()


# With density matrices.

# 1) Works.
# phase = 0.0 * np.pi
#
# t1_grid = 20
# t2_grid = 20
#
# t1_arr = np.linspace(0, 1, t1_grid)
# t2_arr = np.linspace(0, 1, t2_grid)
#
# sz = (t1_grid, t2_grid)
# epr_correl_x = np.zeros(sz, dtype=complex)
#
#
# for n1 in range(len(t1_arr)):
#     for n2 in range(len(t2_arr)):
#         print('n1, n2:', n1, n2)
#         t1 = t1_arr[n1]
#         r1 = sqrt(1 - t1**2)
#         t2 = t2_arr[n2]
#         r2 = sqrt(1 - t2**2)
#         # BS1.
#         state1_unappl = bs2x2_transform(t1, r1, input_state)
#         # Density matrix
#         dm1 = dens_matrix(make_state_appliable(state1_unappl))
#         # Phase modulation.
#         dm2 = phase_modulation(dm1, phase, channel=1)
#         # BS2.
#         trim_dm = 8
#         dm_final = trans.bs_matrix_transform_copt(dm2[:trim_dm, :trim_dm, :trim_dm, :trim_dm].copy(order='C'), t2, r2)
#
#         epr_x, epr_p = erp_squeezing_correlations(dm_final)
#         epr_correl_x[n1, n2] = epr_x
#
#
# print('A real part:', np.sum(np.real(epr_correl_x)))
# print('An image part:', np.sum(np.imag(epr_correl_x)))
#
# print('Minimum:', np.amin(np.real(epr_correl_x)))
# print('Maximum:', np.amax(np.real(epr_correl_x)))
#
# # trim dm = 7
# # Minimum: 0.997450983060972
# # Maximum: 1.0035315367702382
#
# # trim dm = 8
# # Minimum: 0.9994530876721822
# # Maximum: 1.0007523140729127
#
# plt.imshow(np.real(epr_correl_x), origin='lower', cmap=cm.GnBu_r)
# plt.colorbar()
# plt.xlabel('T2')
# plt.ylabel('T1')
# plt.show()


# 2, 3) Works.
#
# phase_grid = 20
# t1_grid = 20
#
# phase_arr = np.linspace(0, 2 * np.pi, phase_grid)
# t1_arr = np.linspace(0, 1, t1_grid)
# t2_arr = np.array([1/sqrt(2)])
#
# t2 = t2_arr[0]
# r2 = np.sqrt(1 - t2_arr[0]**2)
#
# sz = (t1_grid, phase_grid)
# epr_correl_x = np.zeros(sz, dtype=complex)
#
#
# for n1 in range(len(t1_arr)):
#     for p in range(len(phase_arr)):
#         print('n1, p:', n1, p)
#         t1 = t1_arr[n1]
#         r1 = sqrt(1 - t1**2)
#         phase = phase_arr[p]
#         # BS1.
#         state1_unappl = bs2x2_transform(t1, r1, input_state)
#         # Density matrix
#         dm1 = dens_matrix(make_state_appliable(state1_unappl))
#         # Phase modulation.
#         dm2 = phase_modulation(dm1, phase, channel=1)
#         # BS2.
#         trim_dm = 8
#         dm_final = trans.bs_matrix_transform_copt(dm2[:trim_dm, :trim_dm, :trim_dm, :trim_dm].copy(order='C'), t2, r2)
#
#         epr_x, epr_p = erp_squeezing_correlations(dm_final)
#         epr_correl_x[n1, p] = epr_x
#
#
# print('A real part:', np.sum(np.real(epr_correl_x)))
# print('An image part:', np.sum(np.imag(epr_correl_x)))
#
# print('Minimum:', np.amin(np.real(epr_correl_x)))
# print('Maximum:', np.amax(np.real(epr_correl_x)))
#
#
# # series_length = 8, t2 = 1
# # trim dm = 7
# # Minimum: 0.5009536207958811
# # Maximum: 1.4951683245564293
#
# # trim dm = 8
# # Minimum: 0.5022594814429775
# # Maximum: 1.4970759777014475
#
# # series_length = 10, t2 = 1
# # Minimum: 0.5025054547010559
# # Maximum: 1.4970246832205738
#
# # trim dm = 7, grid = 30
# # Minimum: 0.499551541849767
# # Maximum: 1.4965610780599536
#
# # trim dm = 8, grid = 40. Works
# # Minimum: 0.5001827221667734
# # Maximum: 1.4991547456845964
#
#
# plt.imshow(np.real(epr_correl_x), origin='lower', cmap=cm.GnBu_r)
# plt.colorbar()
# plt.xlabel('phase')
# plt.ylabel('T1')
# plt.show()
#


# 4, 5) Works.

# phase_grid = 20
# t2_grid = 20
#
# phase_arr = np.linspace(0, 2 * np.pi, phase_grid)
# t2_arr = np.linspace(0, 1, t2_grid)
# t1_arr = np.array([1/sqrt(2)])
#
# t1 = t1_arr[0]
# r1 = np.sqrt(1 - t1_arr[0]**2)
#
# sz = (t2_grid, phase_grid)
# epr_correl_x = np.zeros(sz, dtype=complex)
#
# for n2 in range(len(t2_arr)):
#     for p in range(len(phase_arr)):
#         print('n2, p:', n2, p)
#         t2 = t2_arr[n2]
#         r2 = sqrt(1 - t2**2)
#         phase = phase_arr[p]
#         # BS1.
#         state1_unappl = bs2x2_transform(t1, r1, input_state)
#         # Density matrix
#         dm1 = dens_matrix(make_state_appliable(state1_unappl))
#         # Phase modulation.
#         dm2 = phase_modulation(dm1, phase, channel=1)
#         # BS2.
#         trim_dm = 7
#         dm_final = trans.bs_matrix_transform_copt(dm2[:trim_dm, :trim_dm, :trim_dm, :trim_dm].copy(order='C'), t2, r2)
#
#         epr_x, epr_p = erp_squeezing_correlations(dm_final)
#         epr_correl_x[n2, p] = epr_x
#
#
# print('A real part:', np.sum(np.real(epr_correl_x)))
# print('An image part:', np.sum(np.imag(epr_correl_x)))
#
# print('Minimum:', np.amin(np.real(epr_correl_x)))
# print('Maximum:', np.amax(np.real(epr_correl_x)))
#
# # t1 = 1
# # Minimum: 0.9989777596078151
# # Maximum: 1.0014495898555928
#
# # t1 = 1/sqrt(2)
# # Minimum: 0.4999459102944174
# # Maximum: 1.500491214140784
#
#
# plt.imshow(np.real(epr_correl_x), origin='lower', cmap=cm.GnBu_r)
# plt.colorbar()
# plt.xlabel('phase')
# plt.ylabel('T2')
# plt.show()


# With traces of two channels.

# 1) Works

# phase = 0.5 * np.pi
#
# t1_grid = 20
# t4_grid = 20
#
# t1_arr = np.linspace(0, 1, t1_grid)
# t4_arr = np.linspace(0, 1, t4_grid)
#
# sz = (t1_grid, t4_grid)
# epr_correl_x = np.zeros(sz, dtype=complex)
#
# t2, t3 = 1, 1
# r2, r3 = 0, 0
#
#
# for n1 in range(len(t1_arr)):
#     for n4 in range(len(t4_arr)):
#         print('n1, n4:', n1, n4)
#         t1 = t1_arr[n1]
#         r1 = sqrt(1 - t1**2)
#         t4 = t4_arr[n4]
#         r4 = sqrt(1 - t4**2)
#         # BS1.
#         state1_unappl = bs2x2_transform(t1, r1, input_state)
#
#         # BS2 and BS3.
#         state_aft2bs_unappl = trans.two_bs2x4_transform_copt(t2, r2, t3, r3, state1_unappl)
#
#         # The detection event.
#         # Gives non-normalised state.
#         state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event='NONE')
#
#         # Calculating the norm.
#         norm_after_det = state_norm_opt(state_after_dett_unappl)
#         print('Norm after det.:', norm_after_det)
#
#         # The normalised state.
#         state_after_dett_unappl_norm = state_after_dett_unappl / norm_after_det
#
#         # Trim the state, 8 is min.
#         trim_state = 8
#         state_after_dett_unappl_norm_tr = state_after_dett_unappl_norm[:trim_state, :trim_state, :trim_state,
#                                           :trim_state]
#         # sm_state = np.sum(np.abs(state_after_dett_unappl_norm)) - np.sum(np.abs(state_after_dett_unappl_norm[:trim_state, :trim_state, :trim_state, :trim_state]))
#         # print('State trim norm:', sm_state)
#
#         # Building dens. matrix and trace.
#         dens_matrix_2ch = dens_matrix_with_trace_opt(state_after_dett_unappl_norm_tr, state_after_dett_unappl_norm_tr)
#
#         # Phase modulation.
#         dm2 = phase_modulation(dens_matrix_2ch, phase, channel=1)
#
#         # The transformation at last BS, 7 is min.
#         trim_dm = 7
#         dm_final = trans.bs_matrix_transform_copt(
#             dm2[:trim_dm, :trim_dm, :trim_dm, :trim_dm].copy(order='C'), t4, r4)
#
#         epr_x, epr_p = erp_squeezing_correlations(dm_final)
#         epr_correl_x[n1, n4] = epr_x
#         print('EPR_X:', epr_x)
#
#
# print('A real part:', np.sum(np.real(epr_correl_x)))
# print('An image part:', np.sum(np.imag(epr_correl_x)))
#
# print('Minimum:', np.amin(np.real(epr_correl_x)))
# print('Maximum:', np.amax(np.real(epr_correl_x)))
#
#
# # phase = 0.0
# # Max = 1
# # Min = 1
#
# # phase = 0.25
# # Max = 1
# # Min = 0.65
#
# # phase = 0.5
# # Minimum: 0.4992167994992098
# # Maximum: 1.0014598540145987
#
# # phase = 1.5
# # Minimum: 0.998977761201982
# # Maximum: 1.5012310931085122
#
#
# plt.imshow(np.real(epr_correl_x), origin='lower', cmap=cm.GnBu_r)
# plt.colorbar()
# plt.xlabel('T4')
# plt.ylabel('T1')
# plt.show()


# 2, 3)

# phase_grid = 20
# t1_grid = 20
#
# phase_arr = np.linspace(0, 2 * np.pi, phase_grid)
# t1_arr = np.linspace(0, 1, t1_grid)
# t4_arr = np.array([1/sqrt(2)])
#
# t4 = t4_arr[0]
# r4 = np.sqrt(1 - t4_arr[0]**2)
#
# t2, t3 = 1, 1
# r2, r3 = 0, 0
#
# sz = (t1_grid, phase_grid)
# epr_correl_x = np.zeros(sz, dtype=complex)
#
#
# for n1 in range(len(t1_arr)):
#     for p in range(len(phase_arr)):
#         print('n1, p:', n1, p)
#         t1 = t1_arr[n1]
#         r1 = sqrt(1 - t1**2)
#         phase = phase_arr[p]
#         # BS1.
#         state1_unappl = bs2x2_transform(t1, r1, input_state)
#
#         # BS2 and BS3.
#         state_aft2bs_unappl = trans.two_bs2x4_transform_copt(t2, r2, t3, r3, state1_unappl)
#
#         # The detection event.
#         # Gives non-normalised state.
#         state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event='NONE')
#
#         # Calculating the norm.
#         norm_after_det = state_norm_opt(state_after_dett_unappl)
#         print('Norm after det.:', norm_after_det)
#
#         # The normalised state.
#         state_after_dett_unappl_norm = state_after_dett_unappl / norm_after_det
#
#         # Trim the state, 8 is min.
#         trim_state = 8
#         state_after_dett_unappl_norm_tr = state_after_dett_unappl_norm[:trim_state, :trim_state, :trim_state,
#                                           :trim_state]
#         # sm_state = np.sum(np.abs(state_after_dett_unappl_norm)) - np.sum(np.abs(state_after_dett_unappl_norm[:trim_state, :trim_state, :trim_state, :trim_state]))
#         # print('State trim norm:', sm_state)
#
#         # Building dens. matrix and trace.
#         dens_matrix_2ch = dens_matrix_with_trace_opt(state_after_dett_unappl_norm_tr, state_after_dett_unappl_norm_tr)
#
#         # Phase modulation.
#         dm2 = phase_modulation(dens_matrix_2ch, phase, channel=1)
#
#         # The transformation at last BS, 7 is min.
#         trim_dm = 7
#         dm_final = trans.bs_matrix_transform_copt(
#             dm2[:trim_dm, :trim_dm, :trim_dm, :trim_dm].copy(order='C'), t4, r4)
#
#         epr_x, epr_p = erp_squeezing_correlations(dm_final)
#         epr_correl_x[n1, p] = epr_x
#
#
# print('A real part:', np.sum(np.real(epr_correl_x)))
# print('An image part:', np.sum(np.imag(epr_correl_x)))
#
# print('Minimum:', np.amin(np.real(epr_correl_x)))
# print('Maximum:', np.amax(np.real(epr_correl_x)))
#
# # t4 = 1
# # Minimum: 0.5009383740280302
# # Maximum: 1.495183630554327
#
# # t4 = 1/sqrt(2)
# # Minimum: 0.5028703337076682
# # Maximum: 1.4975623994401666
#
# plt.imshow(np.real(epr_correl_x), origin='lower', cmap=cm.GnBu_r)
# plt.colorbar()
# plt.xlabel('phase')
# plt.ylabel('T1')
# plt.show()


# 4, 5)

phase_grid = 20
t4_grid = 20

phase_arr = np.linspace(0, 2 * np.pi, phase_grid)
t4_arr = np.linspace(0, 1, t4_grid)
t1_arr = np.array([1/sqrt(2)])

t1 = t1_arr[0]
r1 = np.sqrt(1 - t1_arr[0]**2)

t2, t3 = 1, 1
r2, r3 = 0, 0

det = 'FIRST'

sz = (t4_grid, phase_grid)
epr_correl_x = np.zeros(sz, dtype=complex)


for n4 in range(len(t4_arr)):
    for p in range(len(phase_arr)):
        print('n4, p:', n4, p)
        t4 = t4_arr[n4]
        r4 = sqrt(1 - t4**2)
        phase = phase_arr[p]
        # BS1.
        state1_unappl = bs2x2_transform(t1, r1, input_state)

        # BS2 and BS3.
        state_aft2bs_unappl = trans.two_bs2x4_transform_copt(t2, r2, t3, r3, state1_unappl)

        # The detection event.
        # Gives non-normalised state.
        state_after_dett_unappl = detection(state_aft2bs_unappl, detection_event=det)

        # Calculating the norm.
        norm_after_det = state_norm_opt(state_after_dett_unappl)
        print('Norm after det.:', norm_after_det)

        # The normalised state.
        state_after_dett_unappl_norm = state_after_dett_unappl / norm_after_det

        # Trim the state, 8 is min.
        trim_state = 8
        state_after_dett_unappl_norm_tr = state_after_dett_unappl_norm[:trim_state, :trim_state, :trim_state,
                                          :trim_state]
        # sm_state = np.sum(np.abs(state_after_dett_unappl_norm)) - np.sum(np.abs(state_after_dett_unappl_norm[:trim_state, :trim_state, :trim_state, :trim_state]))
        # print('State trim norm:', sm_state)

        # Building dens. matrix and trace.
        dens_matrix_2ch = dens_matrix_with_trace_opt(state_after_dett_unappl_norm_tr, state_after_dett_unappl_norm_tr)

        # Phase modulation.
        dm2 = phase_modulation(dens_matrix_2ch, phase, channel=1)

        # The transformation at last BS, 7 is min.
        trim_dm = 7
        dm_final = trans.bs_matrix_transform_copt(
            dm2[:trim_dm, :trim_dm, :trim_dm, :trim_dm].copy(order='C'), t4, r4)

        epr_x, epr_p = erp_squeezing_correlations(dm_final)
        epr_correl_x[n4, p] = epr_x
        print('EPR_X:', epr_x)


print('A real part:', np.sum(np.real(epr_correl_x)))
print('An image part:', np.sum(np.imag(epr_correl_x)))

print('Minimum:', np.amin(np.real(epr_correl_x)))
print('Maximum:', np.amax(np.real(epr_correl_x)))


# det=NONE
# t1 = 1
# Minimum: 0.998977761201982
# Maximum: 1.0014598540145987

# t1 = 1/sqrt(2)
# Minimum: 0.49993062980123876
# Maximum: 1.5005065581671158


plt.imshow(np.real(epr_correl_x), origin='lower', cmap=cm.GnBu_r)
plt.colorbar()
plt.xlabel('phase')
plt.ylabel('T4')
plt.show()

