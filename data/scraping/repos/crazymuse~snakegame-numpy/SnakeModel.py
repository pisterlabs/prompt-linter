import os
import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, ortho_init 
from baselines.common.distributions import make_pdtype, CategoricalPdType,CategoricalPd


# Some of the code blocks have been taken from openAI baselines repo 
# Coding style such as closure and naming conventions of Baselines have been followed to maintain high code quality standards
# and code reusability
# Baselines Github link : https://github.com/openai/baseline

# nf means number of features , rf means region of filter, nin means number of input,

class CNNPolicy(object):
    def __init__(self,sess,p, train_phase=True,has_state = False):
        with tf.variable_scope("model",reuse = train_phase) as scope: # Reuse = true for training phase
            # Initialization of placeholders
            X = tf.placeholder(tf.uint8, p.OBS_SHAPE) #obs
            S = tf.placeholder(tf.float32,p.STATE_SHAPE)
            scaled_x = tf.cast(X, tf.float32) / 255.

            # Additional Functions which may be needed
            relu_activ = tf.nn.relu #Relu Activation
            normalize = lambda layer,phase :  tf.layers.batch_normalization(layer, center=True,scale=True, training=train_phase) # Batch Normalization
            # Model Details
            #h1 = relu_activ(conv(scaled_x,scope = 'conv1', nf = 10, rf = 5, stride = 1,init_scale=np.sqrt(2)))
            #h2 = relu_activ(conv(h1,scope = 'conv2', nf = 10, rf = 3, stride = 1))
            flattened_x = conv_to_fc(scaled_x)
            h1 = relu_activ(fc(flattened_x,scope = 'fc1', nh = 20,init_scale=np.sqrt(2)))
            h2 = relu_activ(fc(h1,scope = 'fc2', nh = 15,init_scale=np.sqrt(2)))
            hconcat = tf.concat([h2,S],axis=1)
            h3 = relu_activ(fc(hconcat,scope = 'fc3', nh = 10,init_scale=np.sqrt(2)))
            hcommon = relu_activ(fc(h3,scope = 'fcommon', nh = 10,init_scale=np.sqrt(2)))
            pi = fc(hcommon, scope = "policy" , nh = 3,init_scale=0.01)
            vf = fc(hcommon, scope = "value"  , nh = 1)

        self.pd_type = CategoricalPdType(p.NUM_ACTIONS)
        self.pd = self.pd_type.pdfromflat(pi) # Sampling from action distribution as per baselines

        # Sample from the distribution
        v0 = vf[:, 0] # To remove extra dimension
        a0 = self.pd.sample() # Sample from distribution
        neglogp0 = self.pd.neglogp(a0) #Self entropy of selected action
        self.initial_state = None # Not required for CNN (only for RNN Models)

        # Interfaces to the outer world
        def step(ob, state, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob,S:state})
            return a, v, neglogp

        def value(ob,state, *_args, **_kwargs):
            return sess.run(v0, {X:ob,S:state})

        def hidden_value(ob,state,*_args, **_kwargs):
            """
            Created for debugging purposes
            """
            #amodel = np.argmax(np.array(sess.run([pi], {X:ob,S:state})).flatten())
            #a =  sess.run([a0], {X:ob,S:state})
            #adict = {"amodel":amodel,"asampler":a}
            
            return sess.run([hcommon], {X:ob,S:state})


        self.pi = pi
        self.vf = vf
        self.X = X
        self.S = S
        self.step = step
        self.value = value
        self.hidden_value = hidden_value # Required for debugging purpose
