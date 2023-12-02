"""
Adapted from OpenAI baselines.
"""

from mpi4py import MPI
import deeplearning.tf_util as U
import tensorflow as tf
import numpy as np

class MpiSGD(object):
    def __init__(self, var_list, *, momentum=0.0, scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.momentum = momentum
        self.scale_grad_by_procs = scale_grad_by_procs
        self.size = sum(U.numel(v) for v in var_list)
        self.m = np.zeros(self.size, 'float32')
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)
        self.comm = MPI.COMM_WORLD if comm is None else comm

    def update(self, localg, stepsize):
        if self.t % 1000 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        globalg = np.zeros_like(localg)
        self.comm.Allreduce(localg, globalg, op=MPI.SUM)
        if self.scale_grad_by_procs:
            globalg /= self.comm.Get_size()

        self.t += 1
        self.m = self.momentum * self.m + (1 - self.momentum) * globalg
        a = stepsize / (1 - self.momentum**self.t)
        self.setfromflat(self.getflat() - a * self.m)

    def sync(self):
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)

    def check_synced(self):
        if self.comm.Get_rank() == 0: # this is root
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)

    def reset(self):
        self.m = np.zeros(self.size, 'float32')
        self.t = 0

    def save(self, filename):
        np.savez(filename, m=self.m, t=self.t)

    def load(self, filename):
        state = np.load(filename)
        self.m = state['m']
        self.t = state['t']

def test_MpiSGD():
    np.random.seed(0)
    tf.set_random_seed(0)

    a = tf.Variable(np.random.randn(3).astype('float32'))
    b = tf.Variable(np.random.randn(2,5).astype('float32'))
    loss = tf.reduce_sum(tf.square(a)) + tf.reduce_sum(tf.sin(b))

    stepsize = 1e-2
    update_op = tf.train.MomentumOptimizer(stepsize, 0).minimize(loss)
    do_update = U.function([], loss, updates=[update_op])

    U.single_threaded_session().__enter__()
    tf.get_default_session().run(tf.global_variables_initializer())
    for i in range(10):
        print(i,do_update())

    tf.set_random_seed(0)
    tf.get_default_session().run(tf.global_variables_initializer())

    var_list = [a,b]
    lossandgrad = U.function([], [loss, U.flatgrad(loss, var_list)], updates=[update_op])
    sgd = MpiSGD(var_list)

    for i in range(10):
        l,g = lossandgrad()
        sgd.update(g, stepsize)
        print(i,l)

    if sgd.comm.Get_rank() == 0:
        sgd2 = MpiSGD(var_list)
        sgd.save('./.sgd_state.npz')
        sgd2.load('./.sgd_state.npz')
        assert np.allclose(sgd.m, sgd2.m)
        assert np.allclose(sgd.t, sgd2.t)
        import os
        os.remove('./.sgd_state.npz')

if __name__=='__main__':
    test_MpiSGD()
