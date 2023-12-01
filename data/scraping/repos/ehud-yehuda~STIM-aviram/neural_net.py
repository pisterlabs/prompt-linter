import numpy as np
import tensorflow as tf
import scipy as sp
import time, random, threading
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layer
#import keras.backend as K
from tensorflow.keras.layers import *
import multiprocessing
from batch import *
import constants
import math
from loss_fn import *
import os
from scipy.special import softmax

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def sparse_dense_matmult_batch(sp_a, b):
	def map_function(x):
		i, dense_slice = x[0], x[1]
		sparse_slice = tf.sparse.reshape(tf.sparse.slice(
			sp_a, [i, 0, 0], [1, sp_a.dense_shape[1], sp_a.dense_shape[2]]),
			[sp_a.dense_shape[1], sp_a.dense_shape[2]])
		mult_slice = tf.sparse.matmul(sparse_slice, dense_slice)
		return mult_slice

	elems = (tf.range(0, sp_a.dense_shape[0], delta=1, dtype=tf.int64), b)
	return tf.map_fn(map_function, elems, dtype=tf.float32, back_prop=True)

#-------------------------------------------------------------------------------
def matmul_batchdim_3dx2d(a, b, bias, a_first_dim, a_last_dim, b_last_dim):
	a = tf.reshape(a, [-1, a_last_dim])
	h = tf.matmul(a, b)
	if not bias is None:
		h = tf.add(h, bias)
	h = tf.reshape(h, [-1, a_first_dim, b_last_dim])
	#h = tf.reshape(h, [-1, -1, b_last_dim])
	return h

#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
def discount(r, bootstrap, size, gamma):
	print(r)
	if constants.DISCOUNT_REWARD:
		R_batch = np.zeros([size], np.float64)
		R = bootstrap
		for i in reversed(range(0, size)):
			R = r[i] + gamma*R
			R_batch[i] = R
		return R_batch
	else:
		return r

#-------------------------------------------------------------------------------
def get_m(r, bootstrap, best_q_dist, size, gamma):
	final_dist = np.zeros((constants.N_ATOM))
	final_dist[0] = 1.0
	best_q_dist.append(final_dist)
	best_q_dist = best_q_dist[1:]

	Tz = np.zeros([size, constants.N_ATOM], np.float64)
	vmin = constants.VMIN
	vmax = constants.VMAX
	for i in range(size):
		for j in range(constants.N_ATOM):
			Tz[i, j] = r[i] + gamma * constants.ATOMS[j]
		
	Tz = np.clip(Tz, vmin, vmax)
	b = (Tz - vmin) / constants.DELTA_Z
	b = np.clip(b, 0, constants.N_ATOM-1)
	lower = np.floor(b).astype(int)
	upper = np.ceil(b).astype(int)
	
	m_mat = []
	for i in range(size):
		m = np.zeros((constants.N_ATOM))
		for j in range(constants.N_ATOM):
			m[lower[i,j]] += best_q_dist[i][j] * ( float(upper[i,j]) - b[i,j] )
			m[upper[i,j]] += best_q_dist[i][j] * ( b[i,j] - float(lower[i,j]) )
		m_mat.append(m)
	#m_mat = softmax(np.array(m_mat), axis=1)

	'''print("\n\nTz= ", Tz, "\n")
	print("r = ", r, "\n")
	print("Q = ", best_q_dist, "\n")
	print("b = ", b, "\n")
	print("l = ", lower, "\n")
	print("u = ", upper, "\n")
	print("m = ", m_mat, "\n\n")
	print("\n\nm_sum = ", np.sum(m_mat, axis=1))'''
			
	return m_mat
	

#*******************************************************************************
#*******************************************************************************
#*******************************************************************************

class Neural_Net():
	def __init__(self, scope, session, learning_rate, lr_decay, gamma, n_feat):
		self.session = session
		self.scope = scope
		#self.learning_rate = tf.train.polynomial_decay(	learning_rate,
		#												train_nodes,
		#												MAX_STEPS//2,
		#												LEARNING_RATE*0.1)
		self.learning_rate = learning_rate
		self.lr_decay = lr_decay
		self.gamma = gamma
		self.n_feat = n_feat

		if constants.USE_GPU:
			self.device = '/gpu:0'
		else:
			self.device = '/cpu:0'

		self._build_network()

	#-------------------------------------------------------------------
	def _build_network(self):
		if constants.EMBED_METHOD == constants.GCN:
			self._node_embed_GCN()
		else:
			self._node_embed_struc2vec()
		self._LSTM_embed()
		self._state_encoder()
		self._prepare_q_loss()

	#-------------------------------------------------------------------
	def _node_embed_struc2vec(self):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device): 
			#initializer = normalized_columns_initializer(constants.INIT_WEIGHT)
			#initializer = tf.constant_initializer(constants.INIT_WEIGHT)
			initializer = tf.contrib.layers.xavier_initializer()
			#init_bias = tf.zeros_initializer
			init_bias = tf.contrib.layers.xavier_initializer()

			self.node_features = tf.placeholder(tf.float32, [None, None, self.n_feat], name="features")
			self.A = tf.sparse_placeholder(tf.float32, name="A")
			self.w1 = tf.Variable(initializer((self.n_feat, constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w1")
			self.w2 = tf.Variable(initializer((constants.EMBED_SIZE, constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w2")
			self.w3 = tf.Variable(initializer((1,constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w3")
			self.w4 = tf.Variable(initializer([]), trainable=True, dtype=tf.float32, name="w4")

			self.n_nodes = tf.shape(self.A)[1]

			wx_all = matmul_batchdim_3dx2d(	a=self.node_features, 
											b=self.w1, 
											bias=None, 	
											a_first_dim=self.n_nodes, 
											a_last_dim=self.n_feat, 
											b_last_dim=constants.EMBED_SIZE)		#BxNxE

			weight_sum_init = tf.sparse_reduce_sum(self.A, axis=2, keepdims=True)	#BxNx1
			weight_sum_init = tf.reshape(weight_sum_init, [-1, self.n_nodes, 1])	#Reshape, since the sparse_reduce_sum has unknown dim 0
			weight_sum = tf.multiply(weight_sum_init, self.w4)						#BxNx1
			weight_sum = tf.nn.relu(weight_sum)										#BxNx1
			weight_sum = matmul_batchdim_3dx2d(	a=weight_sum, 
												b=self.w3, 
												bias=None, 	
												a_first_dim=self.n_nodes, 
												a_last_dim=1, 
												b_last_dim=constants.EMBED_SIZE)	#BxNxE

			weight_wx = tf.add(wx_all, weight_sum)
			current_mu = tf.nn.relu(weight_wx)										#BxNxE

			for i in range(0, constants.LAYERS):
				neighbor_sum = sparse_dense_matmult_batch(self.A, current_mu)
				neighbor_linear = matmul_batchdim_3dx2d(a=neighbor_sum, 
														b=self.w2, 
														bias=None, 	
														a_first_dim=self.n_nodes, 
														a_last_dim=constants.EMBED_SIZE, 
														b_last_dim=constants.EMBED_SIZE)
				current_mu = tf.nn.relu(tf.add(wx_all, neighbor_linear))

			self.embed_mat = current_mu

	#-------------------------------------------------------------------
	def _node_embed_GCN(self):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device): 
			#initializer = normalized_columns_initializer(constants.INIT_WEIGHT)
			#initializer = tf.constant_initializer(constants.INIT_WEIGHT)
			initializer = tf.contrib.layers.xavier_initializer()
			#init_bias = tf.zeros_initializer
			init_bias = tf.contrib.layers.xavier_initializer()

			self.node_features = tf.placeholder(tf.float32, [None, None, self.n_feat], name="features")
			self.A = tf.sparse_placeholder(tf.float32, name="A")
			self.w = []
			self.b = []
			self.w.append(tf.Variable(initializer((constants.NUM_FEATURES,constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w1"))
			self.b.append(tf.get_variable("b1", [constants.EMBED_SIZE], initializer=init_bias))
			if constants.LAYERS > 1:
				self.w.append(tf.Variable(initializer((constants.EMBED_SIZE,constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w2"))
				self.b.append(tf.get_variable("b2", [constants.EMBED_SIZE], initializer=init_bias))
			elif constants.LAYERS > 2:
				self.w.append(tf.Variable(initializer((constants.EMBED_SIZE,constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w3"))
				self.b.append(tf.get_variable("b3", [constants.EMBED_SIZE], initializer=init_bias))
			elif constants.LAYERS > 3:
				self.w.append(tf.Variable(initializer((constants.EMBED_SIZE,constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w4"))
				self.b.append(tf.get_variable("b4", [constants.EMBED_SIZE], initializer=init_bias))

			self.n_nodes = tf.shape(self.A)[1]
			H = self.node_features

			for i in range(constants.LAYERS):
				dim_size = constants.EMBED_SIZE
				if i == 0:
					dim_size = self.n_feat

				h = sparse_dense_matmult_batch(self.A, H)
				h = matmul_batchdim_3dx2d(a=h, b=self.w[i], bias=self.b[i],	a_first_dim=self.n_nodes, 
																						a_last_dim=dim_size, 
																						b_last_dim=constants.EMBED_SIZE)
				H = tf.nn.relu(h)


			self.embed_mat = H

	#-------------------------------------------------------------------
	def _LSTM_layer(self, lstm_input, c_in, h_in, units, name):
		#with tf.variable_scope(self.scope), tf.device(self.device):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device): 

			lstm_cell = tf.contrib.rnn.BasicLSTMCell(units, state_is_tuple=True)
			lstm_state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
			lstm_out_full, lstm_state_out = tf.nn.dynamic_rnn(	lstm_cell, lstm_input, initial_state=lstm_state_in,
																sequence_length=None, time_major=False, scope=name)

			return lstm_state_out, lstm_out_full

	#-------------------------------------------------------------------
	def _LSTM_create_layers(self, lstm_input, c_in, h_in, n_layers, units, name_prefix):
		#with tf.variable_scope(self.scope), tf.device(self.device):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device): 

			out_full = lstm_input
			state_out = []
			for i in range(n_layers):
				sout, out_full = self._LSTM_layer(out_full, c_in[i], h_in[i], units, name_prefix + str(i))
				state_out.append(sout)

			return state_out, out_full

	#-------------------------------------------------------------------
	def _LSTM_flow_1(self, n_layers, units):
		self.lstm_c_in_f1 = tf.placeholder(tf.float32, [n_layers, None, units])
		self.lstm_h_in_f1 = tf.placeholder(tf.float32, [n_layers, None, units])

		lstm_input = tf.transpose(self.node_features, perm=[1, 0, 2])		#lstm = [N, B, C]

		state_out, out_full = self._LSTM_create_layers(lstm_input, self.lstm_c_in_f1, self.lstm_h_in_f1, n_layers, units, 'lstm_f1_')

		out_full = tf.transpose(out_full, perm=[1, 0, 2])

		self.lstm_sout_f1 = state_out
		self.lstm_embed_f1 = out_full

	#-------------------------------------------------------------------
	def _LSTM_flow_2(self, n_layers, units):
		self.lstm_c_in_f2 = tf.placeholder(tf.float32, [n_layers, None, units])
		self.lstm_h_in_f2 = tf.placeholder(tf.float32, [n_layers, None, units])

		lstm_input = tf.transpose(self.embed_mat, perm=[1, 0, 2])		#lstm = [N, B, E]

		state_out, out_full = self._LSTM_create_layers(lstm_input, self.lstm_c_in_f2, self.lstm_h_in_f2, n_layers, units, 'lstm_f2_')

		out_full = tf.transpose(out_full, perm=[1, 0, 2])

		self.lstm_sout_f2 = state_out
		self.lstm_embed_f2 = out_full

	#-------------------------------------------------------------------
	def _LSTM_embed(self):

		self._LSTM_flow_2(constants.LSTM_LAYERS_F2, constants.LSTM_SIZE_F2)
		lstm_embed_out = self.lstm_embed_f2

		new_size = constants.LSTM_SIZE_F2
		if constants.USE_DOUBLE_FLOW:
			self._LSTM_flow_1(constants.LSTM_LAYERS_F1, constants.LSTM_SIZE_F1)
			new_size = constants.LSTM_SIZE_F1 + constants.LSTM_SIZE_F2
			lstm_embed_out = tf.concat([self.lstm_embed_f1, self.lstm_embed_f2], axis=2)

		initializer = tf.contrib.layers.xavier_initializer()
		self.w3 = tf.Variable(initializer((new_size, constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w3")
		self.w4 = tf.Variable(initializer((constants.EMBED_SIZE, constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="w4")

		wx_all = matmul_batchdim_3dx2d(	a=lstm_embed_out, 
										b=self.w3, 
										bias=None, 	
										a_first_dim=self.n_nodes, 
										a_last_dim=new_size, 
										b_last_dim=constants.EMBED_SIZE)

		current_mu = tf.nn.relu(wx_all)

		neighbor_sum = sparse_dense_matmult_batch(self.A, current_mu)
		neighbor_linear = matmul_batchdim_3dx2d(a=neighbor_sum, 
												b=self.w4, 
												bias=None, 	
												a_first_dim=self.n_nodes, 
												a_last_dim=constants.EMBED_SIZE, 
												b_last_dim=constants.EMBED_SIZE)
		current_mu = tf.nn.relu(tf.add(wx_all, neighbor_linear))

		self.lstm_embed = current_mu
			
	#-------------------------------------------------------------------
	def _state_encoder(self):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device):
			#initializer = normalized_columns_initializer(constants.INIT_WEIGHT)
			#initializer = tf.constant_initializer(constants.INIT_WEIGHT)
			initializer = tf.contrib.layers.xavier_initializer()
			#init_bias = tf.constant_initializer(constants.INIT_WEIGHT)
			#init_bias = normalized_columns_initializer(constants.INIT_WEIGHT)
			init_bias = tf.contrib.layers.xavier_initializer()
			#init_bias = tf.zeros_initializer
			#init_bias = None

			self.wa = tf.Variable(initializer((constants.EMBED_SIZE,constants.EMBED_SIZE)), trainable=True, dtype=tf.float32, name="wa")
			self.ba = tf.get_variable("ba", [constants.EMBED_SIZE], initializer=init_bias)
			self.wf = tf.Variable(initializer((constants.EMBED_SIZE,constants.N_ATOM)), trainable=True, dtype=tf.float32, name="wf")
			self.bf = tf.get_variable("bf", [constants.N_ATOM], initializer=init_bias)

			final_embed = self.embed_mat
			if constants.USE_LSTM:
				final_embed = self.lstm_embed

			self.action = matmul_batchdim_3dx2d(a=final_embed, 	b=self.wa, 
															bias=self.ba,	
															a_first_dim=self.n_nodes, 
															a_last_dim=constants.EMBED_SIZE, 
															b_last_dim=constants.EMBED_SIZE)#BxNxF
			
			
			self.Qs_a 	= tf.nn.relu(self.action)
			self.Qs_a = matmul_batchdim_3dx2d(a=self.Qs_a, 	b=self.wf, 
															bias=self.bf,	
															a_first_dim=self.n_nodes, 
															a_last_dim=constants.EMBED_SIZE, 
															b_last_dim=constants.N_ATOM)	#BxNxA
			self.Q_activation = tf.nn.softmax(self.Qs_a)
			print('ACTIVATION = ', self.Q_activation.get_shape())

	#-------------------------------------------------------------------
	def _prepare_q_loss(self, simple_loss=True):
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device):
			"""
			This region builds the operations for updating the trainable
			variables (weights) of the Neural Network.
			"""
			self.action_id_onehot = tf.sparse_placeholder(tf.float32, name="action_id")		#Bx1xN
			self.lr_in = tf.placeholder(tf.float32, shape=[])
			self.m =  tf.placeholder(tf.float32, shape=[None, constants.N_ATOM], name='m')

			action = sparse_dense_matmult_batch(self.action_id_onehot, self.Qs_a)			#Bx1xA
			self.Q_selected = tf.reshape(action, [-1, constants.N_ATOM])					#BxA		

			self.loss_rl = bce(self.m, self.Q_selected)
			#self.loss_rl = cross_entropy_loss(self.m, self.Q_activation)

			if simple_loss:
				self.optimizer = tf.train.AdamOptimizer(self.lr_in, beta1=0.9, beta2=0.99, use_locking=True)

				"""
				Compute gradients of the loss function with respect to the
				variables of the local network. We then clip the gradients to
				avoid updating the network with high gradient values
				"""
				self.local_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

				gradients = tf.gradients(self.loss_rl, self.local_params)
				self.gradients_rl = [tf.clip_by_value(grad, -1., 1.) for grad in gradients]
				master_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'worker_global')
				self.apply_grads_rl = self.optimizer.apply_gradients(list(zip(self.gradients_rl, master_net_params)))

	#-------------------------------------------------------------------
	def _learning_rate_decay(self):
		self.learning_rate = self.learning_rate * self.lr_decay
		if self.learning_rate < constants.MIN_LEARNING_RT:
			self.learning_rate = constants.MIN_LEARNING_RT
		print('learning rate = ', self.learning_rate)

	#-------------------------------------------------------------------
	def _get_sparse_tensor(self, batch):
		num_nodes = batch.num_nodes
		data_A = []
		ind_A = []
		for i in range(batch.size):
			A = batch.adj[i]
			if len(A.row) != len(A.col) or len(A.row) != len(A.data):
				print("ERROR: A.row != A.col or A.row != A.data")
				exit(0)
			for j in range(len(A.row)):
				ind_A.append([i, A.row[j], A.col[j]])
				data_A.append(A.data[j])

		tf_sp_A = tf.SparseTensorValue(indices=ind_A,
									values=data_A,
									dense_shape=[batch.size, num_nodes, num_nodes])

		return tf_sp_A

	#-------------------------------------------------------------------
	def _get_sparse_id_onehot(self, id_vec, size, num_nodes):
		if id_vec is None or len(id_vec) == 0 or id_vec[0] is None:
			return None
		else:
			ind = []
			n_data = 0
			for i in range(0, size):
				if id_vec[i] != constants.NO_ACTION:
					ind.append([i, 0, id_vec[i]])
					n_data += 1
			if ind == []:
				ind = np.empty((0, 3), dtype=np.int64)
			else:
				ind = np.array(ind)
			data = np.ones(n_data, dtype=np.float32)
			tf_sp_onehot = tf.SparseTensorValue(indices=ind,
										values=data,
										dense_shape=[size, 1, num_nodes])
			
			return tf_sp_onehot

	#-------------------------------------------------------------------
	def _unpack_batch(self, batch):
		tf_sp_A = self._get_sparse_tensor(batch)
		tf_sp_onehot = self._get_sparse_id_onehot(batch.node, batch.size, batch.num_nodes)
		tf_sp_onehot_sup = self._get_sparse_id_onehot(batch.node_sup, batch.size, batch.num_nodes)

		return tf_sp_A, tf_sp_onehot, tf_sp_onehot_sup

	#-------------------------------------------------------------------
	def get_Qs_a(self, session, batch):
		tf_sp_A, _, _ = self._unpack_batch(batch)
		Q_mat	= session.run(	self.Q_activation,
								feed_dict={	self.A				: tf_sp_A,
											self.node_features 	: batch.features,
											self.lstm_c_in_f1	: batch.hidden_states_f1[0],
											self.lstm_h_in_f1	: batch.hidden_states_f1[1],
											self.lstm_c_in_f2	: batch.hidden_states_f2[0],
											self.lstm_h_in_f2	: batch.hidden_states_f2[1]})
		Q_mat = np.array(Q_mat)					#BxNxA
		Q_mat = Q_mat[-1,:,:]					#NxA
		Q_vec = np.zeros((Q_mat.shape[0]))		#N
		for i in range(Q_mat.shape[0]):
			q_vec = Q_mat[i,:]
			Q_vec[i] = np.sum(q_vec * constants.ATOMS)

		return Q_mat, Q_vec

	#-------------------------------------------------------------------
	def get_hidden_state(self, session, batch):
		tf_sp_A, tf_sp_onehot, _ = self._unpack_batch(batch)
		f1, f2	= session.run(	[self.lstm_sout_f1,
								self.lstm_sout_f2],
								feed_dict={	self.A				: tf_sp_A,
											self.node_features 	: batch.features,
											self.lstm_c_in_f1	: batch.hidden_states_f1[0],
											self.lstm_h_in_f1	: batch.hidden_states_f1[1],
											self.lstm_c_in_f2	: batch.hidden_states_f2[0],
											self.lstm_h_in_f2	: batch.hidden_states_f2[1]})
		
		return f1, f2

	#-------------------------------------------------------------------
	def train_network_rl(self, session, batch):
		tf_sp_A, tf_sp_onehot, _ = self._unpack_batch(batch)
		R = discount(batch.r, 0.0, batch.size, self.gamma)
		m_mat = get_m(R, batch.bootstrap, batch.best_q_dist, batch.size, self.gamma)

		_,lrl,f1,f2  = session.run([self.apply_grads_rl,
									self.loss_rl,
									self.lstm_sout_f1,
									self.lstm_sout_f2],
									feed_dict={	self.A					: tf_sp_A,
												self.node_features 		: batch.features,
												self.action_id_onehot	: tf_sp_onehot,
												self.lr_in				: self.learning_rate,
												self.m					: m_mat,
												self.lstm_c_in_f1	: batch.hidden_states_f1[0],
												self.lstm_h_in_f1	: batch.hidden_states_f1[1],
												self.lstm_c_in_f2	: batch.hidden_states_f2[0],
												self.lstm_h_in_f2	: batch.hidden_states_f2[1]})

		self._learning_rate_decay()
		
		return lrl, f1, f2

	#-------------------------------------------------------------------
	def test_network_rl(self, session, batch):
		tf_sp_A, tf_sp_onehot, _ = self._unpack_batch(batch)
		R = discount(batch.r, 0.0, batch.size, self.gamma)
		m_mat = get_m(R, batch.bootstrap, batch.best_q_dist, batch.size, self.gamma)

		lrl,f1,f2 = session.run([self.loss_rl, 
								self.lstm_sout_f1,
								self.lstm_sout_f2],
								feed_dict={	self.A					: tf_sp_A,
											self.node_features 		: batch.features,
											self.action_id_onehot	: tf_sp_onehot,
											self.lr_in				: self.learning_rate,
											self.m					: m_mat,
											self.lstm_c_in_f1	: batch.hidden_states_f1[0],
											self.lstm_h_in_f1	: batch.hidden_states_f1[1],
											self.lstm_c_in_f2	: batch.hidden_states_f2[0],
											self.lstm_h_in_f2	: batch.hidden_states_f2[1]})



		return lrl, f1, f2

	#-------------------------------------------------------------------
	def update_network_op(self, from_scope):
		with tf.variable_scope(self.scope):
			to_scope = self.scope
			from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
			to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

			ops = []
			for from_var,to_var in zip(from_vars,to_vars):
				ops.append(to_var.assign(from_var))

			return tf.group(*ops)

