#!/usr/bin/env python3.8

"""
A minimal Advantage Actor Critic Implementation
Usage:
python3 minA2C.py
"""

import gym
import tensorflow as tf
import numpy as np
import os

from gym.vector.utils import spaces
from tensorflow import keras

import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
import roslaunch
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from collections import deque
import time
import random

# An episode a full game
train_episodes = 100

def create_actor(state_shape, action_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    # model.add(keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation="tanh", kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation="tanh", kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='softmax', kernel_initializer=init))
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def create_critic(state_shape, output_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(output_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def one_hot_encode_action(action, n_actions):
    encoded = np.zeros(n_actions, np.float32)
    encoded[action] = 1
    return encoded

def main():

    rospy.init_node('turtlebot3_a2c', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param('/turtlebot3/task_and_robot_environment_name')
    print("ENV NAME: " + str(task_and_robot_environment_name))
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('a3c_turtlebot3')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    RANDOM_SEED = 7
    tf.random.set_seed(RANDOM_SEED)

    # env = gym.make('MountainCar-v0')
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    actor_checkpoint_path = "/home/taylor/multidrone_slam/src/2122_MultiDroneIndoorSLAM/frontier_rl/openai_examples_projects/a3c_turtlebot3/training_actor/actor_cp.ckpt" #"/root/catkin_ws/src/frontier_rl/openai_examples_projects/a3c_turtlebot3/training_actor/actor_cp.ckpt"
    critic_checkpoint_path = "/home/taylor/multidrone_slam/src/2122_MultiDroneIndoorSLAM/frontier_rl/openai_examples_projects/a3c_turtlebot3/training_critic/critic_cp.ckpt" #"/root/catkin_ws/src/frontier_rl/openai_examples_projects/a3c_turtlebot3/training_critic/critic_cp.ckpt"

    # actor_checkpoint_path = "training_actor/actor_cp.ckpt"
    # critic_checkpoint_path = "training_critic/critic_cp.ckpt"

    print("ENV SHAPE: " + str(env.observation_space.shape))
    print("ENV SPACE: " + str(env.action_space.n))

    # number_observations = rospy.get_param('/turtlebot3/n_observations')
    # number_actions = rospy.get_param('/turtlebot3/n_actions')
    #
    # actor = create_actor(env.env.env, env.env.get)
    # critic = create_critic(number_observations, 1)

    # print(str(env.env.env._get_obs()))
    # print("OBS SHAPE: " + str(np.array(env.env.env._get_obs()).shape))

    #FIXME: the below lines were working
    # actor = create_actor(np.array(env.env.env._get_obs()).shape, env.action_space.n)
    # critic = create_critic(np.array(env.env.env._get_obs()).shape, 1)

    actor = create_actor(np.array(np.zeros(env.observation_space.n)).shape, env.action_space.n)
    critic = create_critic(np.array(np.zeros(env.observation_space.n)).shape, 1)

    # actor = create_actor(env.observation_space.shape, env.action_space.n)
    # critic = create_critic(env.observation_space.shape, 1)

    # FIXME: add this back in
    if os.path.exists('training_actor'):
        actor.load_weights(actor_checkpoint_path)
        critic.load_weights(critic_checkpoint_path)
    # print(actor)
    # print(critic)

    # X = states, y = actions
    X = []
    y = []

    last_action = 0

    all_taken_actions = list()

    total_training_rewards = 0

    print("getting obs")
    observation = env.env.env.get_obs()
    print("resetting gmapping")
    env.env.env.reset_gmapping()
    print("resetting enviornment")
    env.reset()
    print("starting move base")
    env.env.start_move_base()
    print("setting episode start time")
    env.env.env.set_start_time()

    done = False
    while not done:

        # observation = env.env.env._get_obs()
        observation = np.array(observation)

        is_default = np.all(observation == 100)
        if is_default:
            print("default observation hit")
            observation = np.array(env.env.env.get_obs())

        # 0 stays 0 (best) -1 becomes 1 (second best) 100 becomes 2 (worst)
        observation = np.where(observation == -1, 1, observation)
        observation = np.where(observation == 100, 2, observation)

        observation_reshaped = observation.reshape([1, observation.shape[0]])

        action_probs = actor.predict(observation_reshaped).flatten()
        print("*****PROBABILITY: " + str(action_probs))

        # action = np.argmax(action_probs)
        actions_sorted_indices = action_probs.argsort()
        action = actions_sorted_indices[-1]

        print("ACTION: " + str(action))

        # if int(action) == int(last_action):
        #     print("ACTION CHANGED BECAUSE DUPLICATE OF LAST ACTION OR NOT VALID")
        #     for i in range(-2, -1*action_probs.size, -1):
        #         checking_action = actions_sorted_indices[i]
        #         print("checking action for index i, action: " + str(i) + ", " + str(checking_action))
        #         if checking_action not in all_taken_actions:
        #             print("ACTION CHANGED BECAUSE DUPLICATE OF LAST ACTION.  THIS ACTION HAS NEVER BEEN TAKEN BEFORE AND IS VALID: " + str(checking_action))
        #             action = checking_action
        #             break

        env.gazebo.unpauseSim()

        if int(action) in all_taken_actions or not env.is_valid_action_point(action):
            print("ACTION CHANGED BECAUSE DUPLICATE OF LAST ACTION OR NOT VALID")
            for i in range(-2, -1*action_probs.size, -1):
                checking_action = actions_sorted_indices[i]
                print("checking action for index i, action: " + str(i) + ", " + str(checking_action))
                if checking_action not in all_taken_actions and env.is_valid_action_point(checking_action):
                    print("ACTION CHANGED BECAUSE DUPLICATE OF LAST ACTION.  THIS ACTION HAS NEVER BEEN TAKEN BEFORE AND IS VALID: " + str(checking_action))
                    action = checking_action
                    break

        next_observation = env.pseudo_step(action)

        # if np.array_equal(observation, np.array(next_observation)):
        #     print("OBSERVATION IS THE SAME")
        # else:
        #     print("obs are different :)")

        observation = next_observation

        # print("Sending to trained action point")
        # env.env.env.send_to_trained_point(action)

        # observation = env.env.env._get_obs()

        last_action = action
        print("SETTING LAST ACTION TO: " + str(action))
        all_taken_actions.append(int(action))
        print("ALL TAKEN ACTIONS: " + str(all_taken_actions))


    env.close()

if __name__ == '__main__':
    main()
