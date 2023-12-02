import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import factorial, sqrt

from core.state_configurations import coherent_state
from core.squeezing import erp_squeezing_correlations


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


# def make_state_appliable(state):
#     """
#     Apply operators to the state in 2 channels.
#     :param state: Unapplied state in 2 channels.
#     :return: Applied state in 2 channels.
#     """
#     size = len(state)
#     st_appl = np.zeros((size, size), dtype=complex)
#     for p1 in range(size):
#         for p2 in range(size):
#             st_appl[p1, p2] = state[p1, p2] * np.sqrt(factorial(p1) * factorial(p2))
#     return st_appl
#
#
def dens_matrix(state):
    """
    Build a density matrix in 2 channels.
    :param state: Applied state in 2 channels.
    :return: Applied density matrix for 2 channels.
    """
    size = len(state)
    state_conj = np.conj(state)
    dm = np.zeros((size,) * 4, dtype=complex)

    for p1 in range(size):
        for p2 in range(size):
            for p1_ in range(size):
                for p2_ in range(size):
                    dm[p1, p2, p1_, p2_] = state[p1, p2] * state_conj[p1_, p2_]

    return dm


t_grd = 10
phase_grd = 9

t1_arr = np.linspace(0, 1, t_grd)
t2_arr = np.linspace(1e-4, 1 - 1e-4, t_grd)
t3_arr = np.linspace(1e-4, 1 - 1e-4, t_grd)
t4_arr = np.linspace(0, 1, t_grd)
phase_arr = np.linspace(0, 2*np.pi, phase_grd)


L = 10

alpha = 0.1

epr_arr = np.zeros((phase_grd, t_grd, t_grd, t_grd, t_grd), dtype=complex)


for p in range(phase_grd):
    print('Phase step:', p)
    for n1 in range(t_grd):
        print('n1 step:', n1)
        for n2 in range(t_grd):
            for n3 in range(t_grd):
                for n4 in range(t_grd):
                    t1 = t1_arr[n1]
                    t2 = t2_arr[n2]
                    t3 = t3_arr[n3]
                    t4 = t4_arr[n4]
                    phase = phase_arr[p]

                    r1 = sqrt(1 - t1**2)
                    r2 = sqrt(1 - t2**2)
                    r3 = sqrt(1 - t3**2)
                    r4 = sqrt(1 - t4**2)
                    alpha_1_f = (alpha * t1 * t2) * 1j * r4 + (1j * alpha * r1 * t3) * np.exp(1j * phase) * t4
                    alpha_2_f = (alpha * t1 * t2) * t4 + (1j * alpha * r1 * t3) * np.exp(1j * phase) * 1j * r4

                    ksi1 = 1j * r4 * (1j * r1 * t2) * ((-alpha) * r1 * r3)
                    ksi2 = t4 * (1j * r1 * t2) * ((-alpha) * r1 * r3)
                    ksi0 = 1j * t1 * r3 + t1 * t3 * ((-alpha) * r1 * r3)

                    # print('ksi 0:', ksi0)
                    # print('ksi 1:', ksi1)
                    # print('ksi 2:', ksi2)

                    # Unapplied.
                    cs1 = coherent_state(L, alpha=alpha_1_f)
                    cs2 = coherent_state(L, alpha=alpha_2_f)

                    # a1_conj * cs1
                    a1_cs1 = np.roll(cs1, 1)
                    a1_cs1[0] = 0

                    # a2_conj * cs2
                    a2_cs2 = np.roll(cs2, 1)
                    a2_cs2[0] = 0

                    # Unapplied, unnormalised, output state.
                    state = (ksi1 * np.tensordot(a1_cs1, cs2, axes=0) +
                             ksi2 * np.tensordot(cs1, a2_cs2, axes=0) +
                             ksi0 * np.tensordot(cs1, cs2, axes=0)
                             )

                    state_unappl_norm = state / state_norm_opt(state)

                    # print(state_norm_opt(state_unappl_norm))

                    state_appl = make_state_appliable(state_unappl_norm)

                    dm = np.einsum('ij,kl->ijkl', state_appl, np.conj(state_appl))
                    # dm1 = dens_matrix(state_appl)

                    # print(np.sum(np.einsum('ij,kl->ijkl', state_appl, np.conj(state_appl)) - dens_matrix(state_appl)))
                    #print(np.sum(dm1 - dm))

                    epr_x, _ = erp_squeezing_correlations(dm)

                    # print('EPR_X:', epr_x)

                    epr_arr[p, n1, n2, n3, n4] = epr_x


epr_min_vs_phase = np.zeros(phase_grd, dtype=complex)

for i in range(phase_grd):
    epr_min_vs_phase[i] = np.amin(epr_arr[i, :, :, :, :])

# plt.plot(phase_arr, np.real(epr_min_vs_phase))
# plt.show()

# epr=0.43764574, alpha = 1
# epr=0.43807441, alpha=0.5
# epr=0.43763398, alpha=0.1


# epr_min_vs_t4 = np.zeros(t_grd, dtype=complex)
# for n4 in range(t_grd):
#     epr_min_vs_t4[n4] = np.amin(epr_arr[:, :, :, :, n4])
#
#
# plt.plot(t4_arr, np.real(epr_min_vs_t4))
# plt.show()
#
#
# #
# epr_min_vs_t1 = np.zeros(t_grd, dtype=complex)
# for n1 in range(t_grd):
#     epr_min_vs_t1[n1] = np.amin(epr_arr[:, n1, :, :, :])
#
#
# plt.plot(t1_arr, np.real(epr_min_vs_t1))
# plt.show()
#

# np.min(epr_arr[5, :, :, :, :])

