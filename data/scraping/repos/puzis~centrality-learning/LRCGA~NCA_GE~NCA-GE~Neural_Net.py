import numpy as np
import tensorflow as tf
import time, random, threading
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layer
import multiprocessing
from Batch import *
import Constants
import math

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
		self.learning_rate = Constants.LEARNING_RATE

		if Constants.USE_GPU:
			self.device = '/gpu:0'
		else:
			self.device = '/cpu:0'

		self._build_network()
		self._prepare_loss_function()

		self.cont = 0

	#-------------------------------------------------------------------
	def _build_network(self):
		if Constants.EMBED_METHOD == Constants.GCN:
			self._node_embed_GCN()
		else:
			self._node_embed_struc2vec()
		self._centrality_decoder()

	#-------------------------------------------------------------------
	def _node_embed_GCN(self):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device):
		#with tf.variable_scope(self.scope), tf.device(self.device):
			#initializer = normalized_columns_initializer(Constants.INIT_WEIGHT)
			#initializer = tf.constant_initializer(Constants.INIT_WEIGHT)
			initializer = tf.contrib.layers.xavier_initializer()
			#initializer = normalized_columns_initializer(Constants.INIT_WEIGHT)
			#init_bias = tf.zeros_initializer
			#init_bias = None
			#init_bias = normalized_columns_initializer(Constants.INIT_WEIGHT)
			init_bias = tf.contrib.layers.xavier_initializer()

			self.node_features = tf.placeholder(tf.float32, [None, Constants.NUM_FEATURES], name="features")
			self.node_id_onehot = tf.sparse_placeholder(tf.float32, name="id")
			self.A = tf.sparse_placeholder(tf.float32, name="A")
			self.w = []
			self.b = []
			self.w.append(tf.Variable(initializer((Constants.NUM_FEATURES,Constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w1"))
			self.b.append(tf.get_variable("b1", [Constants.EMBED_SIZE], initializer=init_bias))
			if Constants.LAYERS > 1:
				self.w.append(tf.Variable(initializer((Constants.EMBED_SIZE,Constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w2"))
				self.b.append(tf.get_variable("b2", [Constants.EMBED_SIZE], initializer=init_bias))
			if Constants.LAYERS > 2:
				self.w.append(tf.Variable(initializer((Constants.EMBED_SIZE,Constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w3"))
				self.b.append(tf.get_variable("b3", [Constants.EMBED_SIZE], initializer=init_bias))
			if Constants.LAYERS > 3:
				self.w.append(tf.Variable(initializer((Constants.EMBED_SIZE,Constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w4"))
				self.b.append(tf.get_variable("b4", [Constants.EMBED_SIZE], initializer=init_bias))

			self.n_nodes = tf.shape(self.A)[1]
			self.n_nodes = tf.dtypes.cast(self.n_nodes, tf.float32)
			H = self.node_features

			for i in range(Constants.LAYERS):
				h = tf.sparse_tensor_dense_matmul(self.A, H)
				if Constants.USE_NORMALIZE:
					h = tf.math.l2_normalize(h, axis=0)
					#h = tf.div(h, self.n_nodes)
				h = tf.add(tf.matmul(h, self.w[i]), self.b[i])
				H = tf.nn.relu(h)

			self.mu = tf.sparse_tensor_dense_matmul(self.node_id_onehot, H)

	#-------------------------------------------------------------------
	def _node_embed_struc2vec(self):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device): 
			#initializer = normalized_columns_initializer(Constants.INIT_WEIGHT)
			initializer = tf.contrib.layers.xavier_initializer()

			self.node_features = tf.placeholder(tf.float32, [None, Constants.NUM_FEATURES], name="degree")
			#self.node_id_onehot = tf.placeholder(tf.float32, [None, None], name="id")
			self.node_id_onehot = tf.sparse_placeholder(tf.float32, name="id")
			self.A = tf.sparse_placeholder(tf.float32, name="A")
			self.w1 = tf.Variable(initializer((Constants.NUM_FEATURES,Constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w1")
			self.w2 = tf.Variable(initializer((Constants.EMBED_SIZE,Constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w2")
			self.w3 = tf.Variable(initializer((1,Constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w3")
			self.w4 = tf.Variable(initializer([]), trainable=True, dtype=tf.float32, name="w4")

			self.n_nodes = tf.shape(self.A)[1]

			wx_all = tf.matmul(self.node_features, self.w1)						#NxE

			weight_sum_init = tf.sparse_reduce_sum(self.A, axis=1, keepdims=True)#Nx1
			weight_sum_init = tf.reshape(weight_sum_init, [self.n_nodes, 1])	#Reshape, since the sparse_reduce_sum has unknown dim 0
			weight_sum = tf.multiply(weight_sum_init, self.w4)
			weight_sum = tf.nn.relu(weight_sum)									#Nx1
			weight_sum = tf.matmul(weight_sum, self.w3)							#NxE

			weight_wx = tf.add(wx_all, weight_sum)
			current_mu = tf.nn.relu(weight_wx)									#NxE

			for i in range(0, Constants.LAYERS):
				#neighbor_sum = tf.reshape(matmul_special(self.adj, current_mu), [-1, Constants.EMBED_SIZE])
				neighbor_sum = tf.sparse_tensor_dense_matmul(self.A, current_mu)
				neighbor_linear = tf.matmul(neighbor_sum, self.w2)				#NxE

				current_mu = tf.nn.relu(tf.add(neighbor_linear, weight_wx))		#NxE

			mu_all = current_mu

			self.mu = tf.sparse_tensor_dense_matmul(self.node_id_onehot, mu_all)
			#self.mu = tf.reshape(matmul_special(self.node_id_onehot, mu_all), [-1, Constants.EMBED_SIZE])

	#-------------------------------------------------------------------
	def _centrality_decoder(self):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device):
			#initializer = normalized_columns_initializer(Constants.INIT_WEIGHT)
			#initializer = tf.constant_initializer(Constants.INIT_WEIGHT)
			initializer = tf.contrib.layers.xavier_initializer()
			#init_bias = tf.constant_initializer(Constants.INIT_WEIGHT)
			#init_bias = normalized_columns_initializer(Constants.INIT_WEIGHT)
			init_bias = tf.contrib.layers.xavier_initializer()
			#init_bias = tf.zeros_initializer
			#init_bias = None
			h = self.mu
			h 					= layer.fully_connected(h,
													Constants.UNITS_H1,
													activation_fn=tf.nn.relu,
													weights_initializer=initializer,
													biases_initializer=init_bias,
													scope='FC_1')

			h 					= layer.fully_connected(h,
													Constants.UNITS_H2,
													activation_fn=tf.nn.relu,
													weights_initializer=initializer,
													biases_initializer=init_bias,
													scope='FC_2')

			h 					= layer.fully_connected(h,
													Constants.UNITS_H2,
													activation_fn=tf.nn.relu,
													weights_initializer=initializer,
													biases_initializer=init_bias,
													scope='FC_3')

			self.centrality 	= layer.fully_connected(h,
													1,
													activation_fn=None,
													weights_initializer=initializer,
													biases_initializer=init_bias,
													scope='FC_4')

			if Constants.EMBED_METHOD == Constants.GCN:
				if Constants.RANGE == Constants.RANGE_01:
					self.centrality = tf.nn.sigmoid(self.centrality)
				else:
					self.centrality = tf.nn.tanh(self.centrality)

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

			self.main_loss 		= tf.losses.mean_squared_error(self.true_centrality, self.centrality)
			self.regularization	= tf.reduce_sum(tf.stack([tf.nn.l2_loss(var) for var in params]), name='regularization')

			self.total_loss = self.main_loss + Constants.LAMBDA_REGUL*self.regularization

			self.lr_in = tf.placeholder(tf.float32, shape=[])
			self.optimizer = tf.train.AdamOptimizer(self.lr_in, beta1=0.9, beta2=0.99, use_locking=True)
			#self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9)
			#self.optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9, use_locking=True)

			"""
			Compute gradients of the loss function with respect to the
			variables of the network. Then, apply gradients to update the weights
			of the network
			"""
			gvs = self.optimizer.compute_gradients(self.total_loss)
			gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
			self.gradients = gvs
			self.minimize = self.optimizer.apply_gradients(gvs)
			#self.minimize = self.optimizer.minimize(self.total_loss)

	#-------------------------------------------------------------------
	def _learning_rate_decay(self):
		self.learning_rate = self.learning_rate * Constants.DECAY
		if self.learning_rate < Constants.MIN_LEARNING_RT:
			self.learning_rate = Constants.MIN_LEARNING_RT
		#print('learning rate = ', self.learning_rate)

	#-------------------------------------------------------------------
	def get_sparse_tensor(self, batch):
		num_nodes = batch.num_nodes
		data_A = []
		ind_A = []

		A = batch.adj
		if len(A.row) != len(A.col) or len(A.row) != len(A.data):
			print("ERROR: A.row != A.col or A.row != A.data")
			exit(0)
		for j in range(len(A.row)):
			ind_A.append([A.row[j], A.col[j]])
			data_A.append(A.data[j])

		tf_sp_A = tf.SparseTensorValue(indices=ind_A,
									values=data_A,
									dense_shape=[num_nodes, num_nodes])

		ind = []
		for i in range(0, batch.size):
			ind.append([i, batch.id[i]])
		ind = np.array(ind)
		data = np.ones(batch.size, dtype=np.float32)
		tf_sp_onehot = tf.SparseTensorValue(indices=ind,
									values=data,
									dense_shape=[batch.size, batch.num_nodes])

		return tf_sp_A, tf_sp_onehot

	#-------------------------------------------------------------------
	def run_network(self, session, batch):
		tf_sp_A, tf_sp_onehot = self.get_sparse_tensor(batch)
		c	= session.run(	[self.centrality],
							feed_dict={	self.node_features	: batch.features,
										self.A				: tf_sp_A,
										self.node_id_onehot	: tf_sp_onehot})

		return c

	#-------------------------------------------------------------------
	def train_network(self, session, batch):
		self.cont += 1
		tf_sp_A, tf_sp_onehot = self.get_sparse_tensor(batch)
		_, ml, rl, g, c	= session.run(	[self.minimize,
											self.main_loss,
											self.regularization,
											self.gradients,
											self.centrality],
											feed_dict={	self.node_features	: batch.features,
														self.A				: tf_sp_A,
														self.node_id_onehot	: tf_sp_onehot,
														self.true_centrality: batch.centrality,
														self.lr_in			: self.learning_rate})
		self._learning_rate_decay()

		#for c_t, c_p in zip(batch.centrality, c):
		#	if abs(c_t - c_p) > 0.4:
		#			print("(", c_t, ",", c_p, ") --> ", abs(c_t-c_p))
		#for i in range(0, batch.size):
		#	if abs(batch.centrality[i] - c[i]) > 0.4:
		#			print("[", batch.id[i], "] (", batch.centrality[i], ",", c[i], ") --> ", abs(batch.centrality[i]-c[i]))
			

		return ml, rl, c

	#-------------------------------------------------------------------
	def test_network(self, session, batch):
		tf_sp_A, tf_sp_onehot = self.get_sparse_tensor(batch)
		ml, rl, c		= session.run(	[self.main_loss,
											self.regularization,
											self.centrality],
											feed_dict={	self.node_features	: batch.features,
														self.A				: tf_sp_A,
														self.node_id_onehot	: tf_sp_onehot,
														self.true_centrality: batch.centrality})

		#for c_t, c_p in zip(batch.centrality, c):
		#		print("(", c_t, ",", c_p, ") --> ", abs(c_t-c_p))

		return ml, rl, c
