import numpy as np
import tensorflow as tf
import time, random, threading
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layer
import multiprocessing

FRAMES_STACKED = 4

"""
Values used in
	-"Playing FPS Games with Deep Reinforcement Learning":
		- UNITS_H1 = 4608
		- UNITS_LSTM = 512
	- "Asynchronous Methods for Deep Reinforcement Learning"
		- UNITS_H1 = 256
		- UNITS_LSTM = 256
"""
UNITS_H1 = 256
UNITS_LSTM = 256

BETA 			= 0.01
GAMMA 			= 0.99
LEARNING_RATE 	= 7e-4
DECAY 			= 0.99
EPSILON 		= 0.1
NORM_CLIP		= 40.0

BATCH_SIZE = 20

SMALL_VALUE = 1e-20

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
class Network():
	def __init__(self, scope, num_actions, width, height, channels, gamma, learning_rate):
		self.scope = scope
		self.num_actions = num_actions
		self.width = width
		self.height = height
		self.channels = channels
		self.gamma = gamma

		"""
		Choose the optimier to be used
		"""
		self.optimizer = tf.train.RMSPropOptimizer(	learning_rate=learning_rate,
													decay=DECAY,
													epsilon=EPSILON)
		#self.optimizer = tf.train.AdadeltaOptimizer(LEARNING_RATE, rho=DECAY)
		#self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

		self.build_network()
		self.prepare_loss_function()

	#-------------------------------------------------------------------
	def build_network(self):
		with tf.variable_scope(self.scope):
			self.state = tf.placeholder("float", [None, self.height, self.width, FRAMES_STACKED], name="state")
			"""
			Creates a series of convolutional layers
			"""
			conv = tf.layers.conv2d(inputs=self.state,
									activation=tf.nn.relu,
									bias_initializer=tf.constant_initializer(0.0),
									filters=16, kernel_size=8, strides=4, padding='valid')
			conv = tf.layers.conv2d(inputs=conv,
									activation=tf.nn.relu,
									bias_initializer=tf.constant_initializer(0.0),
									filters=32, kernel_size=4, strides=2, padding='valid')
			# conv = tf.layers.conv2d(inputs=conv,
			# 						activation=tf.nn.elu,
			# 						bias_initializer=tf.constant_initializer(0.0),
			# 						filters=32, kernel_size=3, strides=2, padding='same')
			# conv = tf.layers.conv2d(inputs=conv,
			# 						activation=tf.nn.elu,
			# 						bias_initializer=tf.constant_initializer(0.0),
			# 						filters=32, kernel_size=3, strides=2, padding='same')

			h = slim.flatten(conv)
			"""
			To use a fully connected layer, just uncomment the following line
				- h.get_shape() ===> (?, UNITS_H1)
			"""
			h = layer.fully_connected(h, UNITS_H1, activation_fn=tf.nn.relu, biases_initializer=tf.constant_initializer(0.0))

			"""
			Create new dimension, so that the input for the LSTM is (if using the
			fully connected layer):
			[batch_size=1, time_step=Conv_layer_batch_size, input_dim=UNITS_H1]
				- lstm_input.get_shape() ===> (1, ?, UNITS_H1)
			or, if not using the fully connected layer, it is:
			[batch_size=1, time_step=Conv_layer_batch_size, input_dim=SIZE_CONV_FLAT]
				- lstm_input.get_shape() ===> (1, ?, SIZE_CONV_FLAT)
			"""
			lstm_input = tf.expand_dims(h, [0])

			"""
			Retrives the original batch size (used in the Convolutional layer).
			This value (represented by the ? in 'lstm_input.get_shape()') indicates
			how many steps the LSTM layer will unroll. This makes sense, since the
			current batch carries an ordered sequence of events, which is required
			by the LSTM layer.
			"""
			step_size = tf.shape(self.state)[:1]

			"""
			Create a LSTM cell with UNITS_LSTM units. We then define an initial
			state for the LSTM cell. The initial state is divided into 'c' and 'h',
			each with dimension [LSTM_BATCH_SIZE, LSTM_CELL_STATE_SIZE], where
			LSTM_BATCH_SIZE=1. We use 'self.lstm_state_init' only when we want
			te reset the LSTM layer's state (between episodes, for example)
			"""
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(UNITS_LSTM,state_is_tuple=True)
			c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
			h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
			self.lstm_state_init = [c_init, h_init]
			"""
			Creates a placeholder that receives the current internal state of the
			LSTM layer. These values must be fed from the outside.
			"""
			c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
			h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
			self.state_in = [c_in, h_in]
			lstm_state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
			"""
			Creates the LSTM layer. 'time_major=False' guarantees that the output
			is shaped: [batch_size=1, time_step=Conv_layer_batch_size, input_dim=UNITS_LSTM]
			It also returns the final state of the LSTM layer, with the following
			shape: [batch_size=1, LSTM_CELL_STATE_SIZE]
				- lstm_out.get_shape() ===> (1, ?, UNITS_LSTM)
				- lstm_state.get_shape() ===> (1, ?, UNITS_LSTM)
			"""
			lstm_out, lstm_state_out = tf.nn.dynamic_rnn(	lstm_cell, lstm_input,
															initial_state=lstm_state_in,
			 												sequence_length=step_size,
															time_major=False)
			"""
			Breaks the final state into 'c' and 'h'
			"""
			lstm_c, lstm_h = lstm_state_out
			self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
			"""
			Remove the extra fake batch_size=1 dimension, such that the
			output of the LSTM is [Conv_layer_batch_size, UNITS_LSTM]
				- lstm_out.get_shape() ===> (?, UNITS_LSTM)
			"""
			lstm_out = tf.reshape(lstm_out, [-1, UNITS_LSTM])


			"""
			Separate the LSTM layer output into two different streams:
			the value function estimator and the action policy estimator
				- self.policy.get_shape() ===> (?, NUM_ACTIONS)
				- self.value.get_shape() ===> (?, 1)
			"""
			self.policy = layer.fully_connected(lstm_out,
												self.num_actions,
												activation_fn=tf.nn.softmax,
												weights_initializer=normalized_columns_initializer(0.01),
               									biases_initializer=tf.constant_initializer(0))
			self.value = layer.fully_connected(	lstm_out,
												1,
												activation_fn=None,
												weights_initializer=normalized_columns_initializer(1.0),
               									biases_initializer=tf.constant_initializer(0))

	#-------------------------------------------------------------------
	def prepare_loss_function(self):
		with tf.variable_scope(self.scope):
			"""
			This region builds the operations for updating the trainable
			variables (weights) of the Neural Network.
			"""
			self.actions = 	tf.placeholder(shape=[None], dtype=tf.int32, 	name="action")
			self.R 		 = 	tf.placeholder(shape=[None], dtype=tf.float32, 	name="discounted_reward")
			self.A 		 = 	tf.placeholder(shape=[None], dtype=tf.float32, 	name="advantage")
			actions_onehot = tf.one_hot(self.actions,self.num_actions,dtype=tf.float32)
			"""
			v.get_shape() ===> (?)
			"""
			v = tf.reshape(self.value, [-1])

			log_policy = tf.log(tf.clip_by_value(self.policy, SMALL_VALUE, 1.0))
			responsible_outputs = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), reduction_indices=1)
			"""
			Loss funtions:
			  - Value Loss retrieved from "Asynchronous Methods for Deep
				Reinforcement Learning"
			  - Entropy function presented in Eq 21 from "Function optimization
			  using connectionist reinforcement learning algorithms"
			"""
			policy_loss = - tf.reduce_sum(responsible_outputs*self.A)
			value_loss = 0.5 * tf.reduce_sum(tf.square(v - self.R))
			entropy = - tf.reduce_sum(self.policy * log_policy)
			self.total_loss = 0.5 * value_loss + policy_loss - entropy * BETA

			"""
			Compute gradients of the loss function with respect to the
			variables of the local network. We then clip the gradients to
			avoid updating the network with high gradient values
			"""
			local_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
			gradients = tf.gradients(self.total_loss, local_params)
			grads, grad_norms = tf.clip_by_global_norm(gradients, NORM_CLIP)

			"""
			Apply gradients w.r.t. the variables of the local network into the
			global network
			"""
			master_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'worker_global')
			self.apply_grads = self.optimizer.apply_gradients(list(zip(grads,master_net_params)))

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
