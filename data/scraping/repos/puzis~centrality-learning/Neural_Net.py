import numpy as np
import tensorflow as tf
import time, random, threading
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layer
import multiprocessing
from Batch import *
import Constants

"""
Copied from the Universe starter agent from OpenAI. In its description, it says:
'Used to initialize weights for policy and value output layers'
"""
def normalized_columns_initializer(std=1.0):
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer

#*******************************************************************************
#*******************************************************************************
#*******************************************************************************

class Neural_Net():
	def __init__(self, scope, session, learning_rate):
		self.session = session
		self.scope = scope
		self.learning_rate = learning_rate

		if Constants.USE_GPU:
			self.device = '/gpu:0'
		else:
			self.device = '/cpu:0'

		self._build_network()
		self._prepare_loss_function()

	#-------------------------------------------------------------------
	def _build_network(self):
		self._node_embed()

	#-------------------------------------------------------------------
	def _node_embed(self):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device):
		#with tf.variable_scope(self.scope), tf.device(self.device):
			self.node_degree = tf.placeholder("float", [None, 1], name="degree")
			self.node_eigen = tf.placeholder("float", [None, 1], name="eigen")
			#weight_init = normalized_columns_initializer(Constants.INIT_WEIGHT)
			weight_init = tf.contrib.layers.xavier_initializer(uniform=True)
			input = tf.concat(values=[self.node_degree, self.node_eigen], axis=1)

			h 				= layer.fully_connected(input,
													Constants.UNITS_H1,
													activation_fn=tf.nn.tanh,#tf.nn.relu,
													weights_initializer=weight_init, 
													biases_initializer=tf.constant_initializer(Constants.INIT_WEIGHT),
													scope='FC_1')
			h 				= layer.fully_connected(h,
													Constants.UNITS_H2,
													activation_fn=tf.nn.tanh,#tf.nn.relu,
													weights_initializer=weight_init, 
													biases_initializer=tf.constant_initializer(Constants.INIT_WEIGHT),
													scope='FC_2')
			h 				= layer.fully_connected(h,
													Constants.UNITS_H3,
													activation_fn=tf.nn.tanh,#tf.nn.relu,
													weights_initializer=weight_init, 
													biases_initializer=tf.constant_initializer(Constants.INIT_WEIGHT),
													scope='FC_3')
			self.centrality	= layer.fully_connected(h,
													1,
													activation_fn=None,
													weights_initializer=weight_init, 
													biases_initializer=tf.constant_initializer(Constants.INIT_WEIGHT),
													scope='FC_4')

	#-------------------------------------------------------------------
	def _prepare_loss_function(self):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device):
		#with tf.variable_scope(self.scope), tf.device(self.device):
			"""
			This region builds the operations for updating the trainable
			variables (weights) of the Neural Network.
			"""
			self.true_centrality = tf.placeholder("float", [None, 1], name="true_state")
			params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

			#self.main_loss		= tf.nn.l2_loss(self.centrality - self.true_centrality, name='l2')
			#self.main_loss 		= tf.reduce_mean(tf.squared_difference(self.centrality, self.true_centrality))
			self.main_loss 		= tf.losses.mean_squared_error(self.true_centrality, self.centrality)
			self.regularization	= tf.reduce_sum(tf.stack([tf.nn.l2_loss(var) for var in params]), name='regularization')

			self.total_loss = self.main_loss + Constants.LAMBDA_REGUL*self.regularization

			#self.learning_rate = tf.placeholder(tf.float32,shape=(),name="learing_rate")
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate, use_locking=True)
			#self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9)
			#self.optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9, use_locking=True)

			"""
			Compute gradients of the loss function with respect to the
			variables of the network. Then, apply gradients to update the weights
			of the network
			"""
			gvs = self.optimizer.compute_gradients(self.total_loss)
			capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
			self.gradients = capped_gvs
			self.minimize = self.optimizer.apply_gradients(capped_gvs)
			#self.minimize = self.optimizer.minimize(self.total_loss)

	#-------------------------------------------------------------------
	def get_node_embed(self, session, batch):
		e	= session.run(	[self.node_embed],
							feed_dict={	self.node_degree	: batch.degree,
										self.node_eigen		: batch.eigen})

		return e[0]

	#-------------------------------------------------------------------
	def train_network(self, session, batch):
		_, tl, ml, rl, c	= session.run(	[self.minimize,
											self.total_loss,
											self.main_loss,
											self.regularization,
											self.centrality],
											feed_dict={	self.node_degree	: batch.degree,
														self.node_eigen		: batch.eigen,
														self.true_centrality: batch.centrality})

		#print c

		return tl, ml, rl, c

	#-------------------------------------------------------------------
	def test_network(self, session, batch):
		tl, ml, rl, c		= session.run(	[self.total_loss,
											self.main_loss,
											self.regularization,
											self.centrality],
											feed_dict={	self.node_degree	: batch.degree,
														self.node_eigen	: batch.eigen,
														self.true_centrality: batch.centrality})

		"""for c_t, c_p in zip(batch.centrality, c):
			if c_t > c_p:
				print "(", c_t, ",", c_p, ") --> ", c_t-c_p
			else:
				print "(", c_t, ",", c_p, ") --> ", c_p-c_t"""

		return tl, ml, rl, c
