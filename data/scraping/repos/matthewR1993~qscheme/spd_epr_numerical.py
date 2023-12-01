import numpy as np
from time import gmtime, strftime
import matplotlib.pyplot as plt
import sys
import platform
import argparse

from customutils.utils import *
from core.basic import *
from core.sytem_setup import *
from core.squeezing import *
from core.state_configurations import coherent_state, single_photon
from core.optimized import transformations as trans


series_length = 10

st1 = single_photon(series_length)

alpha = 1.0
st2 = coherent_state(series_length, alpha=alpha)


# The phase difference before last BS
phase_arr = np.linspace(0, 2*np.pi, 25)

# Bottom channel = 1.
phase_mod_channel = 1


# BS grids.
r1_grid = 13
r4_grid = 13

r2_grid = 13
r3_grid = 13


min_bound = 1e-5
max_bound = 1 - 1e-5

# BS values range.
T1_min = 0.0
T1_max = 1.0
T4_min = 0.0
T4_max = 1.0

T2_min = min_bound
T2_max = max_bound
T3_min = min_bound
T3_max = max_bound


# Varying BSs. Small t, r parameters, with step regarding to big "T".
t1_array, r1_array = bs_parameters(T1_min, T1_max, r1_grid)
t4_array, r4_array = bs_parameters(T4_min, T4_max, r4_grid)
t2_array, r2_array = bs_parameters(T2_min, T2_max, r2_grid)
t3_array, r3_array = bs_parameters(T3_min, T3_max, r3_grid)

sz = (r1_grid, r4_grid, r2_grid, r3_grid)
epr_correl_x = np.zeros(sz, dtype=complex)
epr_min_vs_phase = np.zeros(len(phase_arr), dtype=complex)


mut_state_unappl = np.tensordot(st1, st2, axes=0)


def state_norm_opt(state):
    fact_arr = np.array([factorial(x) for x in range(len(state))])
    tf2 = np.tensordot(fact_arr, fact_arr, axes=0)
    st_abs_quad = np.power(np.abs(state), 2)
    mult = np.multiply(st_abs_quad, tf2)
    return np.sqrt(np.sum(mult))


def make_state_appliable(state):
    fact_arr = np.array([sqrt(factorial(x)) for x in range(len(state))])
    tf2 = np.tensordot(fact_arr, fact_arr, axes=0)
    return np.multiply(state, tf2)


for p, phase in enumerate(phase_arr):
    print('phase step:', p)
    for n1 in range(r1_grid):
        for n4 in range(r4_grid):
            for n2 in range(r2_grid):
                for n3 in range(r3_grid):
                    # print('Steps [n1, n4, n2, n3]:', n1, n4, n2, n3)
                    t1 = t1_array[n1]
                    t2 = t2_array[n2]
                    t3 = t3_array[n3]
                    t4 = t4_array[n4]

                    r1 = r1_array[n1]
                    r2 = r2_array[n2]
                    r3 = r3_array[n3]
                    r4 = r4_array[n4]


                    # First BS.
                    state_after_bs_unappl = bs2x2_transform(t1, r1, mut_state_unappl)

                    # 2d and 3rd BS.
                    # state_aft2bs_unappl = two_bs2x4_transform(t2, r2, t3, r3, state_after_bs_unappl)
                    state_aft2bs_unappl = trans.two_bs2x4_transform_copt(t2, r2, t3, r3, state_after_bs_unappl[:9, :9].copy(order='C'))

                    state_aft2bs_unappl = state_aft2bs_unappl[:9, :9, :9, :9]
                    # state after det.
                    # FIRST bottom single photon det clicks.
                    sz = len(state_aft2bs_unappl)
                    state_aft_det_unnorm = np.zeros((sz,) * 2, dtype=complex)
                    for p2 in range(sz):
                        for p4 in range(sz):
                            state_aft_det_unnorm[p2, p4] += state_aft2bs_unappl[1, p2, 0, p4]

                    norm = state_norm_opt(state_aft_det_unnorm)

                    state_aft_det_norm_unappl = state_aft_det_unnorm / norm

                    # phase
                    st1_unappl = phase_modulation_state(state_aft_det_norm_unappl, phase)

                    # last BS
                    state_unapll_final = bs2x2_transform(t4, r4, st1_unappl)

                    state_final = make_state_appliable(state_unapll_final)
                    state_final = state_final[:9, :9]

                    # form Dens matrix
                    dm_final = np.einsum('ij,kl->ijkl', state_final, np.conj(state_final))

                    # ERP correlation's variance.
                    epr_x, epr_p = erp_squeezing_correlations(dm_final)
                    epr_correl_x[n1, n4, n2, n3] = epr_x

    epr_min = np.amin(epr_correl_x)
    epr_min_vs_phase[p] = epr_min

    print('EPR min:', epr_min)


np.save('data2.npy', {
    'epr_min': epr_min_vs_phase,
    'phases': phase_arr
})


fl = np.load('data2.npy')

epr_min_arr = np.real(fl.item().get('epr_min'))
phase_arr = fl.item().get('phases')

# epr_min_arr[19] = 0.37899992

plt.plot(phase_arr / np.pi, np.real(epr_min_arr))
plt.xlabel('$Phase$')
plt.ylabel('$VAR[X_{1} - X_{2}]$')
plt.grid(True)
plt.show()


