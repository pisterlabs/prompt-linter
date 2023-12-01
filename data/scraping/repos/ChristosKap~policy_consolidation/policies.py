# Edited MLP policy from OpenAI Baselines

import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
        
class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False,scope="model",obs=None): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        if obs is None:
            X = tf.placeholder(tf.float32, ob_shape, name='Ob')
        else:
            X = obs
        with tf.variable_scope(scope, reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)
        self.pdparam = pdparam
        
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.a = a0
        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.neglogp = neglogp0
        self.v = vf
        self.state = self.initial_state
        self.M = False
