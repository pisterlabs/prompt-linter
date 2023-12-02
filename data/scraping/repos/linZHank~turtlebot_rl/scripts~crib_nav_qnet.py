#! /usr/bin/env python

"""

QNet example using turtlebot crib environment
Navigate towards preset goal

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

"""
from __future__ import absolute_import, division, print_function

import gym
from gym import wrappers
import tensorflow as tf
import rospy
import numpy as np
import random
import math
import matplotlib.pyplot as plt

import openai_ros_envs.crib_task_env

class Model:
  def __init__(self, num_states, num_actions, batch_size):
    self._num_states = num_states
    self._num_actions = num_actions
    self._batch_size = batch_size
    # define the placeholders
    self._states = None
    self._actions = None
    # the output operations
    self._logits = None
    self._optimizer = None
    self._var_init = None
    # now setup the model
    self._define_model()

  def _define_model(self):
    self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
    self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
    # create a couple of fully connected hidden layers
    fc1 = tf.layers.dense(self._states, 128, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 32, activation=tf.nn.relu)
    self._logits = tf.layers.dense(fc2, self._num_actions)
    loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
    self._optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss)
    self._var_init = tf.global_variables_initializer()

  def predict_one(self, state, sess):
    return sess.run(
      self._logits,
      feed_dict={self._states: state.reshape(1, self._num_states)}
    )
  
  def predict_batch(self, states, sess):
    return sess.run(self._logits, feed_dict={self._states: states})

  def train_batch(self, sess, x_batch, y_batch):
    sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

class Memory:
  def __init__(self, max_memory):
    self._max_memory = max_memory
    self._samples = []

  def add_sample(self, sample):
    self._samples.append(sample)
    if len(self._samples) > self._max_memory:
      self._samples.pop(0)

  def sample(self, no_samples):
    if no_samples > len(self._samples):
      return random.sample(self._samples, len(self._samples))
    else:
      return random.sample(self._samples, no_samples)

class GameRunner:
  def __init__(self, sess, model, env, memory, max_epsilon, min_epsilon, decay, gamma):
    self._sess = sess
    self._env = env
    self._model = model
    self._memory = memory
    self._max_epsilon = max_epsilon
    self._min_epsilon = min_epsilon
    self._decay = decay
    self._gamma = gamma
    self._epsilon = self._max_epsilon
    self._steps = 0
    self._reward_store = []

  def run(self, num_episodes, num_steps):
    for ep in range(num_episodes):
      state = self._env.reset()
      tot_reward = 0
      done = False
      for st in range(num_steps):
        action = self._choose_action(state)
        next_state, reward, done, info = self._env.step(action)
      
        self._memory.add_sample((state, action, reward, next_state))
        self._replay()

        # exponentially decay the eps value
        self._steps += 1
        self._epsilon = min_epsilon \
                      + (max_epsilon - min_epsilon) * math.exp(-self._decay * ep)

        # move the agent to the next state and accumulate the reward
        state = next_state
        tot_reward += reward

        # if the game is done, break the loop
        if done:
          break

      self._reward_store.append(tot_reward)
      print("Episode: {}, Total reward: {}, Epsilon: {}".format(ep, tot_reward, self._epsilon))

  def _choose_action(self, state):
    if random.random() < self._epsilon:
      return random.randint(0, self._model._num_actions - 1)
    else:
      return np.argmax(self._model.predict_one(state, self._sess))

  def _replay(self):
    batch = self._memory.sample(self._model._batch_size)
    states = np.array([val[0] for val in batch])
    next_states = np.array([val[3] for val in batch])
    # predict Q(s,a) given the batch of states
    q_s_a = self._model.predict_batch(states, self._sess)
    # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
    q_s_a_d = self._model.predict_batch(next_states, self._sess)
    # setup training arrays
    x = np.zeros((len(batch), self._model._num_states))
    y = np.zeros((len(batch), self._model._num_actions))
    for i, b in enumerate(batch):
      state, action, reward, next_state = b[0], b[1], b[2], b[3]
      # get the current q values for all actions in state
      current_q = q_s_a[i]
      current_q[action] = reward
      # update the q value for action
      if next_state is None:
        # in this case, the game completed after action, so there is no max Q(s',a')
        # prediction possible
        current_q[action] = reward
      else:
        current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])
      x[i] = state
      y[i] = current_q
      
    self._model.train_batch(self._sess, x, y)


if __name__ == "__main__":
  rospy.init_node("turtlebot2_crib_qlearn", anonymous=True, log_level=rospy.INFO)
  env_name = "TurtlebotCrib-v0"
  env = gym.make(env_name)
  # env.seed(0)
  rospy.loginfo("Gazebo gym environment set")
  # Set parameters
  num_episodes = 128
  num_steps = 512
  num_states = env.env.observation_space.shape[0]
  num_actions = env.env.action_space.n
  batch_size = 16
  max_memory = 80000
  max_epsilon = 0.999
  min_epsilon = 0.05
  gamma = 0.95
  decay = 0.9
  model = Model(num_states, num_actions, batch_size)
  mem = Memory(max_memory)

  with tf.Session() as sess:
    model.var_init = tf.global_variables_initializer()
    sess.run(model.var_init)
    gr = GameRunner(
      sess=sess,
      model=model,
      env=env,
      memory=mem,
      max_epsilon=max_epsilon,
      min_epsilon=min_epsilon,
      decay=decay,
      gamma=gamma
    )
    gr.run(
      num_episodes=num_episodes,
      num_steps=num_steps
    )
    plt.plot(gr._reward_store)
    plt.show()

