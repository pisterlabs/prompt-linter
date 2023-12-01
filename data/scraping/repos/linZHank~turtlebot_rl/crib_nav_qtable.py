#! /usr/bin/env python

"""

Q-Learning example using turtlebot crib environment
Navigate towards preset goal

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

"""
from __future__ import print_function

import gym
from gym import wrappers
import rospy
import numpy as np
import matplotlib.pyplot as plt
import utils

import openai_ros_envs.crib_task_env

if __name__ == "__main__":
  rospy.init_node("turtlebot2_crib_qlearn", anonymous=True, log_level=rospy.INFO)
  env_name = 'TurtlebotCrib-v0'
  env = gym.make(env_name)
  # env.seed(0)
  rospy.loginfo("Gazebo gym environment set")
  # np.random.seed(0) 
  rospy.loginfo("----- using Q Learning -----")
  # Load parameters
  num_states = 100
  num_actions = 4
  Alpha = 1. # learning rate
  Gamma = 0.95 # reward discount
  num_episodes = 2000
  num_steps = 500
  low = env.observation_space.low
  # Initialize Q table
  Q = np.zeros([num_states, num_actions])
  reward_list = []
  for ep in range(num_episodes):
    # Reset env and get first observation
    obs = env.reset()
    state = utils.obs2state(obs, low)
    total_reward = 0
    done = False
    for st in range(num_steps):
      # Choose action greedily
      action = np.argmax(Q[state,:] + np.random.randn(1, num_actions)*(1./(ep+1)))
      # Get new state and reward
      obs, reward, done, _ = env.step(action)
      state1 = utils.obs2state(obs, low)
      # Update Q table
      Q[state, action] = Q[state, action] + Alpha*(reward + Gamma*np.max(Q[state1,:]) - Q[state, action])
      total_reward += reward
      state = state1
      rospy.loginfo("Total reward = {}".format(total_reward))
      if done:
        break

    reward_list.append(total_reward)

  print("Score over time: {}".format(sum(reward_list)/num_episodes))
  print("Final Q-table: {}".format(Q))

  plt.plot(reward_list)
