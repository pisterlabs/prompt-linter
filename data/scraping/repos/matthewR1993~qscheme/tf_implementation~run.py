import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from customutils.utils import *
from core.basic import *
from core.state_configurations import coherent_state, single_photon, squeezed_vacuum

from tf_implementation.core.squeezing import *
from tf_implementation.core.common import *


L = 10


# state1 = single_photon(L)
state1 = coherent_state(L, alpha=1.0)

# state2 = coherent_state(L, alpha=1.0)
state2 = single_photon(L)

st1_tf = tf.constant(state1, tf.complex128)
st2_tf = tf.constant(state2, tf.complex128)


def t_constr(x):
    return tf.clip_by_value(x, 1e-5, 1 - 1.e-5)


with tf.name_scope('system') as scope:
    # Unapplied input state:
    mut_state = tf.tensordot(st1_tf, st2_tf, axes=0, name='input_state')

    # Trainable parameters.
    phase = tf.Variable(1.47 * np.pi, trainable=True, dtype=tf.float64, name='phase')
    T1 = tf.Variable(0.5, trainable=True, dtype=tf.float64, name='T1', constraint=t_constr)
    T2 = tf.Variable(0.1, trainable=True, dtype=tf.float64, name='T2', constraint=t_constr)

    s1 = bs_transformation_tf(mut_state, T1)
    s2 = phase_mod(phase, s1[:L, :L], input_type='state', channel=1)
    # Unapplied output state:
    state_out = bs_transformation_tf(s2, T2)
    # Applied output state:
    state = make_state_applicable(state_out)
    dm_out = tf.einsum('kl,mn->klmn', state, tf.conj(state))

    # Cost function.
    cor_x, _ = erp_squeezing_correlations_tf(dm_out)
    cost = tf.cast(cor_x, tf.float64)

    # Register summaries.
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('T1', T1)
    tf.summary.scalar('T2', T2)
    tf.summary.scalar('phase', phase)


optimizer = tf.train.AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam'
)
minimize_op = optimizer.minimize(loss=cost, var_list=[T1, T2, phase])


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# tensorboard --logdir=/home/matvei/qscheme/tf_implementation/logs/summaries/log
# http://localhost:6006
sum_path = '/Users/matvei/PycharmProjects/qscheme'
# sum_path = '/home/matvei/qscheme'
summaries_dir = sum_path + '/tf_implementation/logs/summaries'
# summaries_dir = '/home/matvei/qscheme/tf_implementation/logs/summaries'
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(summaries_dir + '/log', sess.graph)

max_steps = 1200
display_step = 20
summarize_step = 10

cost_progress = []


for i in range(max_steps):
    [_, summary, cost_val, T1_val, T2_val, phase_val] = sess.run([minimize_op, merged, cost, T1, T2, phase])
    cost_progress.append(
        {
            'cost': cost_val,
            'T1': T1_val,
            'T2': T2_val,
            'phase': phase_val
        })

    # Prints progress.
    if i % display_step == 0:
        print("Rep: {} Cost: {} T1: {} T2: {} phase: {}".format(i, cost_val, T1_val, T2_val, phase_val))
    if i % summarize_step == 0:
        writer.add_summary(summary, i)


plt.plot([c['cost'] for c in cost_progress])
plt.xlabel('cost')
plt.xlabel('step')
plt.show()


# plt.plot([c['par_value'] for c in cost_progress])
# plt.show()

# pd.DataFrame(cost_progress).plot()



# from core.squeezing import *
#
# L = 10
# state = tf.constant(np.random.rand(L, L), tf.complex128)
#
# state_val = state.eval(session=sess)
# dm_val = np.outer(state_val, state_val.conj())
#
#
# dm = tf.einsum('kl,mn->klmn', state, tf.conj(state))
#
# sess = tf.Session()
#
# res = erp_squeezing_correlations_tf(dm)
# print(res[0].eval(session=sess), res[1].eval(session=sess))
#
# print(erp_squeezing_correlations(dm.eval(session=sess)))
#
# print(erp_squeezing_correlations(dm_val))
#
#
#
#
# cor_x, _ = erp_squeezing_correlations_tf(dm_out)
# cost = tf.cast(cor_x, tf.float64)
#
# erp_squeezing_correlations(dm_out)


# Build density matrix from state


# def two_bs2x4_transform(t1, r1, t2, r2, input_state):
#     """
#     Transformation at 2 beam splitters.
#     Two input channels and four output channles - 2x4 transformation.
#     Creation operators transformation:
#     a1 => t1 a2 + i r1 a1.
#     a2 => t2 a4 + i r2 a3.
#     With transmission and reflection coefficients:
#     t1^2 + r1^2 = 1.
#     t2^2 + r2^2 = 1.
#     :param t1: BS1 transmission.
#     :param r1: BS1 reflection.
#     :param t2: BS2 transmission.
#     :param r2: BS2 reflection.
#     :param input_state: Two channels(modes) unapllied state.
#     :return: Four channels(modes) unapllied state.
#     """
#     size = len(input_state)
#     output_state = np.zeros((size,) * 4, dtype=complex)
#     for m in range(size):
#         for n in range(size):
#
#             for k in range(m + 1):
#                 for l in range(n + 1):
#                     # channels indexes
#                     ind1 = k
#                     ind2 = m - k
#                     ind3 = l
#                     ind4 = n - l
#                     coeff = input_state[m, n] * t1**(m - k) * (1j*r1)**k * t2**(n - l) * (1j*r2)**l * factorial(m) * factorial(n) / (factorial(k) * factorial(m - k) * factorial(l) * factorial(n - l))
#                     output_state[ind1, ind2, ind3, ind4] = output_state[ind1, ind2, ind3, ind4] + coeff
#
#     return output_state

# L = 10
# dm = tf.constant(np.random.rand(L, L, L, L), tf.complex128)
#
# sess = tf.Session()
#
# res = erp_squeezing_correlations_tf(dm)
# print(res[0].eval(session=sess), res[1].eval(session=sess))
#
# print(erp_squeezing_correlations(dm.eval(session=sess)))


# Detection:
# def detection(state, type):
#     return 0


# TODO.
# def bs_2x4_transform_tf(T1, T2, input_state):
#     """
#     Transformation at 2 beam splitters.
#     Two input channels and four output channles - 2x4 transformation.
#     Creation operators transformation:
#     a1 => t1 a2 + i r1 a1.
#     a2 => t2 a4 + i r2 a3.
#     With transmission and reflection coefficients:
#     T1 + R1 = 1.
#     T2 + R2 = 1.
#     :param T1: BS1 transmission.
#     :param T2: BS2 transmission.
#     :param input_state: Two channels unapllied state.
#     :return: Four channels(modes) unapllied state.
#     """
#     return 0
#
#
# def two_bs2x4_transform(t1, r1, t2, r2, input_state):
#     """
#     Transformation at 2 beam splitters.
#     Two input channels and four output channles - 2x4 transformation.
#     Creation operators transformation:
#     a1 => t1 a2 + i r1 a1.
#     a2 => t2 a4 + i r2 a3.
#     With transmission and reflection coefficients:
#     t1^2 + r1^2 = 1.
#     t2^2 + r2^2 = 1.
#     :param t1: BS1 transmission.
#     :param r1: BS1 reflection.
#     :param t2: BS2 transmission.
#     :param r2: BS2 reflection.
#     :param input_state: Two channels(modes) unapllied state.
#     :return: Four channels(modes) unapllied state.
#     """
#     size = len(input_state)
#     output_state = np.zeros((size,) * 4, dtype=complex)
#     for m in range(size):
#         for n in range(size):
#
#             for k in range(m + 1):
#                 for l in range(n + 1):
#                     # channels indexes
#                     ind1 = k
#                     ind2 = m - k
#                     ind3 = l
#                     ind4 = n - l
#                     coeff = input_state[m, n] * t1**(m - k) * (1j*r1)**k * t2**(n - l) * (1j*r2)**l * factorial(m) * factorial(n) / (factorial(k) * factorial(m - k) * factorial(l) * factorial(n - l))
#                     output_state[ind1, ind2, ind3, ind4] = output_state[ind1, ind2, ind3, ind4] + coeff
#
#     return output_state
#
#
# def two_bs2x4_transform_opt(t1, r1, t2, r2, input_state):
#     """
#     Transformation at 2 beam splitters. Optimised version
#     Two input channels and four output channles - 2x4 transformation.
#     Creation operators transformation:
#     a1 => t1 a2 + i r1 a1.
#     a2 => t2 a4 + i r2 a3.
#     With transmission and reflection coefficients:
#     t1^2 + r1^2 = 1.
#     t2^2 + r2^2 = 1.
#     :param t1: BS1 transmission.
#     :param r1: BS1 reflection.
#     :param t2: BS2 transmission.
#     :param r2: BS2 reflection.
#     :param input_state: Two channels(modes) unapllied state.
#     :return: Four channels(modes) unapllied state.
#     """
#     size = len(input_state)
#     out = np.zeros((size,) * 4, dtype=complex)
#
#     def coef(k1, k2, k3, k4):
#         return t1 ** k2 * (1j * r1) ** k1 * t2 ** k4 * (1j * r2) ** k3 / (factorial(k1) * factorial(k2) * factorial(k3) * factorial(k4))
#
#     # index 'i' = (m,n,k,l)
#     for i in np.ndindex(size, size, size, size):
#         if i[2] <= i[0] and i[3] <= i[1] and i[0] + i[1] < size:
#             out[i[2], i[0] - i[2], i[3], i[1] - i[3]] = coef(i[2], i[0] - i[2], i[3], i[1] - i[3]) * input_state[i[0], i[1]] * factorial(i[0]) * factorial(i[1])
#
#     return out

