'''
Taken with minor modification from OpenAI Spinning Up's github
Ref:
[1] https://github.com/openai/spinningup/blob/master/spinup/utils/mpi_tools.py
[2] https://github.com/openai/spinningup/blob/master/spinup/utils/mpi_pytorch.py
'''
import os, sys, subprocess
from mpi4py import MPI
import torch
import numpy as np


comm = MPI.COMM_WORLD


def mpi_fork(n, bind_to_core=False):
    '''
    Re-launches the current script with workers linked by MPI.
    Also, terminates the original process that launched it.

    :param n: (int) Number of processes to split into
    :param bind_to_core: (bool) Bind each MPI process to a core
    '''
    if n <= 1: 
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-n", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def proc_rank():
    '''
    Get process's rank/id
    '''
    return comm.Get_rank()


def n_procs():
    '''
    Get number of processes
    '''
    return comm.Get_size()


def mpi_op(x, op):
    '''
    Do :param op: with :param x: and distribute the result to all processes
    '''
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    comm.Allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def broadcast(x, root=0):
    '''
    Broadcast `x` from process `root` to all other MPI processes
    '''
    comm.Bcast(x, root=root)


def mpi_sum(x):
    '''
    Do a summation over MPI processes and distribute the result to all of them
    '''
    return mpi_op(x, MPI.SUM)


def mpi_avg(x):
    '''
    Get an average a over MPI processes and distribute the result to all of them
    '''
    return mpi_sum(x) / n_procs()


def mpi_max(x):
    '''
    Get the maximal value over MPI processes
    '''
    return mpi_op(x, MPI.MAX)


def mpi_min(x):
    '''
    Get the minimal value over MPI processes
    '''
    return mpi_op(x, MPI.MIN)


def mpi_mean(x):
    mean = mpi_sum(np.sum(x)) / mpi_sum(x.size)
    return mean


def mpi_get_statistics(x, need_optima=False):
    '''
    Get mean, standard deviation, max, min over `x` collected over MPI processes
    '''
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), x.size])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)

    if need_optima:
        max_ = mpi_max(np.max(x) if x.size > 0 else -np.inf)
        min_ = mpi_min(np.min(x) if x.size > 0 else np.inf)
        return mean, std, max_, min_
    return mean, std


def setup_pytorch_for_mpi():
    '''
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    '''
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / n_procs()), 1)
    torch.set_num_threads(fair_num_threads)


def mpi_avg_grads(module):
    '''
    Average contents of gradient buffers across all MPI processes
    '''
    if n_procs() == 1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()   # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def mpi_print(msg, rank=0):
    '''
    :param msg: (str) Messege to print
    :param rank: (int) Rank of the process that is proceeded to print the messege
    '''
    if proc_rank() == rank:
        print(msg)


def sync_params(module):
    '''
    Sync all parameters of module across all MPI processes
    '''
    if n_procs() == 1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy)
