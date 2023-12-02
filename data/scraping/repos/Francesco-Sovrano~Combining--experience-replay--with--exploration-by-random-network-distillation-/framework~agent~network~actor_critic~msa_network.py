# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from agent.network.actor_critic.base_network import Base_Network, is_continuous_control
import utils.tensorflow_utils as tf_utils
from utils.rnn import RNN
from agent.network.actor_critic.openai_small_network import OpenAISmall_Network
import options
flags = options.get()

# N.B. tf.initializers.orthogonal is broken with tensorflow 1.10 and GPU, use OpenAI implementation
class MSA_Network(OpenAISmall_Network):

	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'CNN'
		_, input_height, input_width, input_channel = input.get_shape().as_list()
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.conv2d(name='CNN_Conv1', inputs=input, filters=16, kernel_size=(input_height,1), dilation_rate=(1,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			input = tf.layers.conv2d(name='CNN_Conv2', inputs=input, filters=16, kernel_size=(1,input_width), dilation_rate=(3,1), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			input = tf.layers.conv2d(name='CNN_Conv3', inputs=input, filters=32, kernel_size=3, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return input
		
	def _concat_layer(self, input, concat, scope, name="", share_trainables=True):
		layer_type = 'Concat'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.flatten(input)
			concat = tf.layers.flatten(concat)
			input = tf.concat([input, concat], -1) # shape: (batch, concat_size+input_size)
			input = tf.layers.dense(name='Concat_Dense1', inputs=input, units=256, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			# Return result
			return input
		
	def _rnn_layer(self, input, scope, name="", share_trainables=True):
		rnn = RNN(type='LSTM', direction=2, units=128, batch_size=1, stack_size=1, training=self.training, dtype=flags.parameters_type)
		internal_initial_state = rnn.state_placeholder(name="initial_lstm_state") # for stateful lstm
		internal_default_state = rnn.default_state()
		layer_type = rnn.type
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			output, internal_final_state = rnn.process_batches(
				input=input, 
				initial_states=internal_initial_state, 
				sizes=self.size_batch
			)
			output = tf.layers.dropout(output, rate=0.75, training=self.training)
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			return output, ([internal_initial_state],[internal_default_state],[internal_final_state])
		