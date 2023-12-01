import numpy as np
import tensorflow as tf
from utils import *
from config import *
from model.nn_ops import *
import sys

class CriticAgent(object):
    def __init__(self, sess,
                 input_dim,
                 hid_dims=args.hid_dims, output_dim=1,
                 eps=args.entropy_eps, act_fn=leaky_relu,
                 optimizer=tf.train.AdamOptimizer,
                 scope='critic_agent'):

        self.sess = sess
        self.scope = scope

        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer

        self.input_dim = input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim

        # input dimension: [batch_size, num_workers + 1]
        self.inputs = tf.placeholder(tf.float32, [None, self.input_dim])

        # initialize nn parameters
        self.weights, self.bias = self.nn_init(
            self.input_dim, self.hid_dims, self.output_dim)

        # actor network
        self.values = self.critic_network(
            self.inputs, self.weights, self.bias)

        # groundtruth for training
        self.actual_values = tf.placeholder(tf.float32, [None, 1])

        # define loss
        self.loss = tf.reduce_sum(tf.square(self.actual_values - self.values))

        # get training parameters
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        # operations for setting parameters
        self.input_params, self.set_params_op = \
            self.define_params_op()

        # actor gradients
        self.critic_gradients = tf.gradients(self.loss, self.params)

        # adaptive learning rate
        self.lr_rate = tf.placeholder(tf.float32, shape=[])

        # actor optimizer
        self.critic_opt = self.optimizer(self.lr_rate).minimize(self.loss)

        # apply gradient directly to update parameters
        self.apply_grads = self.optimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.params))

    def nn_init(self, input_dim, hid_dims, output_dim):
        weights = []
        bias = []

        curr_in_dim = input_dim

        # hidden layers
        for hid_dim in hid_dims:
            weights.append(
                glorot([curr_in_dim, hid_dim], scope=self.scope))
            bias.append(
                zeros([hid_dim], scope=self.scope))
            curr_in_dim = hid_dim

        # output layer
        weights.append(glorot([curr_in_dim, output_dim], scope=self.scope))
        bias.append(zeros([output_dim], scope=self.scope))

        return weights, bias

    def critic_network(self, inputs, weights, bias):

        # non-linear feed forward
        x = inputs

        for (w, b) in zip(weights[:-1], bias[:-1]):
            x = tf.matmul(x, w)
            x += b
            x = self.act_fn(x)

        # final linear output layer
        x = tf.matmul(x, weights[-1])
        x += bias[-1]

        return x

    def apply_gradients(self, gradients, lr_rate):
        self.sess.run(self.apply_grads, feed_dict={
            i: d for i, d in zip(
                self.critic_gradients + [self.lr_rate],
                gradients + [lr_rate])
        })

    def define_params_op(self):
        # define operations for setting network parameters
        input_params = []
        for param in self.params:
            input_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op

    def get_params(self):
        return self.sess.run(self.params)

    def set_params(self, input_params):
        self.sess.run(self.set_params_op, feed_dict={
            i: d for i, d in zip(self.input_params, input_params)
        })

    def predict(self, inputs):
        return self.sess.run(self.values, feed_dict={
            self.inputs: inputs
        })

    def get_gradients(self, inputs, actual_values):
        return self.sess.run(
            [self.critic_gradients, self.loss],
            feed_dict={
                self.inputs: inputs,
                self.actual_values: actual_values
        })

    def compute_gradients(self, batch_inputs, batch_actual_values):
        # sopt into batch format
        inputs = np.vstack(batch_inputs)
        actual_values = np.vstack(batch_actual_values)

        # invoke learning model
        gradients, loss = self.get_gradients(
            inputs, actual_values)

        return gradients, loss

class ActorAgent(object):
    def __init__(self, sess, eps=args.entropy_eps, act_fn=leaky_relu,
                 optimizer=tf.train.AdamOptimizer,
                 scope='actor_agent'):

        self.sess = sess
        self.scope = scope

        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer

        self.input_dim = args.state_dim
        self.hid_dims = args.hid_dims
        self.output_dim = args.action_dim  # priority (to softmax over)

        # input dimension: [batch_size, num_workers + 1]
        self.inputs = tf.placeholder(tf.float32, [None, self.input_dim])

        # initialize nn parameters
        self.weights, self.bias = self.nn_init(
            self.input_dim, self.hid_dims, self.output_dim)

        # actor network
        self.act_probs = self.actor_network(
            self.inputs, self.weights, self.bias)

        # sample an action (from OpenAI baselines)
        logits = tf.log(self.act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.act = tf.argmax(logits - tf.log(-tf.log(noise)), 1)

        # selected action: [batch_size, num_workers]
        self.act_vec = tf.placeholder(tf.float32, [None, self.output_dim])

        # advantage term
        self.adv = tf.placeholder(tf.float32, [None, 1])

        # use entropy to promote exploration, this term decays over time
        self.entropy_weight = tf.placeholder(tf.float32, ())

        # select action probability
        self.selected_act_prob = tf.reduce_sum(tf.multiply(
            self.act_probs, self.act_vec),
            reduction_indices=1, keep_dims=True)

        # actor loss due to advantge (negated)
        self.adv_loss = tf.reduce_sum(tf.multiply(
            tf.log(self.selected_act_prob + \
            self.eps), -self.adv))

        # entropy loss (normalized)
        self.entropy_loss = tf.reduce_sum(tf.multiply(
            self.act_probs, tf.log(self.act_probs + self.eps))) / \
            np.log(args.action_dim)

        # define combined loss
        self.loss = self.adv_loss + self.entropy_weight * self.entropy_loss

        # get training parameters
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        # operations for setting parameters
        self.input_params, self.set_params_op = \
            self.define_params_op()

        # actor gradients
        self.act_gradients = tf.gradients(self.loss, self.params)

        # adaptive learning rate
        self.lr_rate = tf.placeholder(tf.float32, shape=[])

        # actor optimizer
        self.act_opt = self.optimizer(self.lr_rate).minimize(self.loss)

        # apply gradient directly to update parameters
        self.apply_grads = self.optimizer(self.lr_rate).\
            apply_gradients(zip(self.act_gradients, self.params))

    def nn_init(self, input_dim, hid_dims, output_dim):
        weights = []
        bias = []

        curr_in_dim = input_dim

        # hidden layers
        for hid_dim in hid_dims:
            weights.append(
                glorot([curr_in_dim, hid_dim], scope=self.scope))
            bias.append(
                zeros([hid_dim], scope=self.scope))
            curr_in_dim = hid_dim

        # output layer
        weights.append(glorot([curr_in_dim, output_dim], scope=self.scope))
        bias.append(zeros([output_dim], scope=self.scope))

        return weights, bias

    def actor_network(self, inputs, weights, bias):

        # non-linear feed forward
        x = inputs

        for (w, b) in zip(weights[:-1], bias[:-1]):
            x = tf.matmul(x, w)
            x += b
            x = self.act_fn(x)

        # final linear output layer
        x = tf.matmul(x, weights[-1])
        x += bias[-1]

        # softmax
        x = tf.nn.softmax(x, dim=-1)

        return x

    def apply_gradients(self, gradients, lr_rate):
        self.sess.run(self.apply_grads, feed_dict={
            i: d for i, d in zip(
                self.act_gradients + [self.lr_rate],
                gradients + [lr_rate])
        })

    def define_params_op(self):
        # define operations for setting network parameters
        input_params = []
        for param in self.params:
            input_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op

    def get_params(self):
        return self.sess.run(self.params)

    def set_params(self, input_params):
        self.sess.run(self.set_params_op, feed_dict={
            i: d for i, d in zip(self.input_params, input_params)
        })

    def predict(self, inputs):
        return self.sess.run(self.act, feed_dict={
            self.inputs: inputs
        })

    def get_gradients(self, inputs, act_vec, adv, entropy_weight):
        return self.sess.run(
            [self.act_gradients, [self.adv_loss, self.entropy_loss]],
            feed_dict={
                self.inputs: inputs,
                self.act_vec: act_vec,
                self.adv: adv,
                self.entropy_weight: entropy_weight
        })

    def compute_gradients(self, batch_inputs, batch_act_vec, \
                          batch_adv, entropy_weight):
        # sopt into batch format
        inputs = np.vstack(batch_inputs)
        act_vec = np.vstack(batch_act_vec)
        # invoke learning model
        gradients, loss = self.get_gradients(
            inputs, act_vec, batch_adv, entropy_weight)
        # append baseline loss
        loss.append(np.mean(batch_adv ** 2))

        return gradients, loss

    # def get_action(self, state):
    #
    #     workers, job, _ = state
    #
    #     inputs = np.zeros([1, args.action_dim + 1])
    #
    #     for worker in workers:
    #         inputs[0, worker.worker_id] = \
    #             min(sum(j.size for j in worker.queue) / \
    #             args.job_size_norm_factor / 5.0,  # normalization
    #             20.0)
    #     inputs[0, -1] = min(job.size / args.job_size_norm_factor, 10.0)  # normalization
    #
    #     action = self.predict(inputs)
    #
    #     return action[0]

    def get_action_feasibility(self, observe):

        temp = observe['state']
        feasibility = observe['feasibility']
        curr_time = observe['time']
        state = np.zeros([1, len(temp)])
        state[0] = temp
        prob = self.sess.run(self.act_probs, feed_dict={self.inputs: state})[0]
        max_logit = -sys.maxsize
        action = 0
        for a in feasibility:
            if prob[a]>=max_logit:
                max_logit = prob[a]
                action = a
        return state, action, curr_time
