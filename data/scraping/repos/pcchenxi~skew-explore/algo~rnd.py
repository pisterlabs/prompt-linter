"""
This code is inspired by the implementation of "random-network-distillation" 
from OpenAI (https://github.com/openai/random-network-distillation) 
"""

import time
import numpy as np
import tensorflow as tf
import logging

class RandomNetworkDistilation(object):
    def __init__(self, input_dim = 2, learning_rate=1e-4):
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.encoded_f_size = 1024
        self.proportion_of_exp_used_for_predictor_update = 1

        tf.reset_default_graph()
        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        logging.info('init rnd network with input_dim %f' % (input_dim))

    # Build the netowrk and the loss functions
    def build(self):
        self.obs = tf.placeholder(name='obs', dtype=tf.float32, shape=[None, self.input_dim])

        # Random target network.
        logging.info("MlpTarget: using '%s' shape %s as input" % (self.obs.name, str(self.obs.shape)))
        xr = tf.cast(self.obs, tf.float32)
        xr = tf.layers.dense(inputs=xr, units=32 * 1, activation=tf.nn.leaky_relu)
        xr = tf.layers.dense(inputs=xr, units=32 * 2 * 1, activation=tf.nn.leaky_relu)
        xr = tf.layers.dense(inputs=xr, units=32 * 2 * 1, activation=tf.nn.leaky_relu)
        X_r = tf.layers.dense(inputs=xr, units=self.encoded_f_size, activation=None)

        # Predictor network.
        xrp = tf.cast(self.obs, tf.float32)
        xrp = tf.layers.dense(inputs=xrp, units=32 * 1, activation=tf.nn.leaky_relu)
        xrp = tf.layers.dense(inputs=xrp, units=32 * 2 * 1, activation=tf.nn.leaky_relu)
        xrp = tf.layers.dense(inputs=xrp, units=32 * 2 * 1, activation=tf.nn.leaky_relu)

        X_r_hat = tf.layers.dense(inputs=xrp, units=128, activation=tf.nn.relu)
        X_r_hat = tf.layers.dense(inputs=X_r_hat, units=self.encoded_f_size, activation=None)

        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)

        targets = tf.stop_gradient(X_r)
        # self.aux_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
        self.aux_loss = tf.reduce_mean(tf.square(targets - X_r_hat), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.aux_loss)

    def get_intrinsic_reward(self, x):
        reward = self.sess.run(self.int_rew,feed_dict={self.obs: x})
        weight = 1000
        return reward * weight

    def train_single_step(self, x):
        _, losses = self.sess.run(
            [self.train_op, self.aux_loss],
            feed_dict={self.obs: x}
        )
        return losses

def rnd_trainer(model, data, num_epoch=15, batch_size = 500):
    data_size = len(data)
    data_index = np.arange(data_size)
    
    for epoch in range(num_epoch):
        np.random.shuffle(data_index)
        for i in range(0, data_size, batch_size):
            # Get a batch
            inds = data_index[i:i+batch_size]
            batch = data[inds]
            losses = model.train_single_step(batch)

        logging_info = 'epoch: ' + str(epoch) + ' loss: ' + str(losses)
        logging.info(logging_info)
            
