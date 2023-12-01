import numpy as np
import torch
from mpi4py import MPI

#########
# Code from openai mpi tools
# mpi utils + pytorch mpi utils
#########


def proc_id():
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()


def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()


def broadcast(x, root=0):
    MPI.COMM_WORLD.Bcast(x, root=root)


def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()


def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    # print('Proc %d: Reporting original number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)

    # print('Proc %d: Reporting new number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)


def mpi_avg_grads(module):
    """Average contents of gradient buffers across MPI processes."""
    if num_procs() == 1:
        return
    for p in module.parameters():
        try:
            p_grad_numpy = p.grad.numpy()  # numpy view of tensor data
            avg_p_grad = mpi_avg(p.grad)
            if avg_p_grad.dtype is np.dtype(np.float64):
                p_grad_numpy = np.array(avg_p_grad)
            else:
                p_grad_numpy[:] = avg_p_grad[:]
        except AttributeError:
            # print("AttributeError detected in grad averaging. Normal once every init")
            pass


def sync_params(module):
    """Sync all parameters of module across all MPI processes."""
    if num_procs() == 1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy)


def create_comm_correspondences(rank, num_workers, num_splits=None):
    comm = MPI.COMM_WORLD
    if num_splits is None:
        return comm
    comm_correspondences = np.array_split(list(range(num_workers)), num_splits)
    color = int(np.where(np.array(comm_correspondences) == rank)[0])
    splitted_comm = comm.Split(color, rank)
    return splitted_comm
