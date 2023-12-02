# code in this file adapted and borrowed from OpenAI's SpinningUp repository

from mpi4py import MPI
import tensorflow as tf
import numpy as np
import os
import sys
import subprocess


def broadcast(x, root=0):
    MPI.COMM_WORLD.Bcast(x, root=root)


def flat_concatenation(vals):
    return tf.concat([tf.reshape(x, (-1,)) for x in vals], axis=0)


def assign_parameters_from_flattened(x, params):
    def flat_size(p): return int(np.prod(p.shape.as_list()))
    splitters = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape)
                  for p, p_new in zip(params, splitters)]
    return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])


def sync_parameters(params):
    gather_params = flat_concatenation(params)

    def _broadcast(x):
        broadcast(x)
        return x
    synced_parameters = tf.py_func(_broadcast, [gather_params], tf.float32)
    return assign_parameters_from_flattened(synced_parameters, params)


def sync_all_parameters():
    return sync_parameters(tf.global_variables())


class MPIAdamOptimizer(tf.train.AdamOptimizer):
    def __init__(self, **kwargs):
        self.communication = MPI.COMM_WORLD
        tf.compat.v1.train.AdamOptimizer.__init__(self, **kwargs)

    def calc_grads(self, loss, variable_list, **kwargs):
        grads_and_vars = super().compute_gradients(loss, variable_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_gradients = flat_concatenation([g for g, v in grads_and_vars])
        grad_shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in grad_shapes]

        num_runs = self.comm.Get_size()

        mpibuf = np.zeros(flat_gradients.shape, np.float32)

        def _collect_grads(flat_gradients):
            self.comm.Allreduce(flat_gradients, mpibuf, op=MPI.SUM)
            np.divide(mpibuf, float(num_runs), out=mpibuf)
            return mpibuf

        avg_flattened_grad = tf.py_func(
            _collect_grads, [flat_gradients], tf.float32)
        avg_flattened_grad.set_shape(flat_gradients.shape)
        avg_grads = tf.split(avg_flattened_grad, sizes, axis=0)

        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                              for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        optimize = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([optimize]):
            sync = sync_parameters([v for g, v in grads_and_vars])
        return tf.group([optimize, sync])


def mpi_fork(n, bind_to_core=False):
    """Re-launches the current script with workers linked by MPI. Also,
    terminates the original process that launched it. Taken almost without
    modification from the Baselines function of the `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py
    Args:
        n (int): Number of process to split into.
        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n <= 1:
        return
    if os.getenv('IN_MPI') is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS='1',
            OMP_NUM_THREADS='1',
            IN_MPI='1'
        )
        args = ['mpirun', '-np', str(n)]
        if bind_to_core:
            args += ['-bind-to', 'core']
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def process_id():
    return MPI.COMM_WORLD.Get_rank()


def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def num_processes():
    return MPI.COMM_WORLD.Get_size()


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def mpi_avg(x):
    return mpi_sum(x) / num_processes()


def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_op([np.sum(x), len(x)], MPI.SUM)
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std
