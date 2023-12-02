#!/usr/bin/env python

import argparse, math, os
import rospy
# from gym import spaces
import gym
import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from openai_ros.robot_envs import multiagent_turtlebot2_env
from openai_ros.task_envs.turtlebot2 import continous_multiagent_turtlebot2_goal
from geometry_msgs.msg import Point
#import algorithms environment
import numpy as np
import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from collections import namedtuple
from itertools import count
# from environments.agents_landmarks.env import agentslandmarks
import glob

from SAC_DUAL_Q_net import SAC



ARG_LIST = ['tau', 'target_update_interval', 'gradient_steps', 'learning_rate', 'gamma', 'capacity',
            'iteration', 'batch_size', 'seed', 'num_hidden_units_per_layer', 'num_hidden_layers', 'activation', 'sample_frequency']


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


def is_in_desired_position(desired_point, current_position, epsilon=0.2):
    """
    It return True if the current position is similar to the desired poistion
    """

    is_in_desired_pos = False

    x_pos_plus = desired_point.x + epsilon
    x_pos_minus = desired_point.x - epsilon
    y_pos_plus = desired_point.y + epsilon
    y_pos_minus = desired_point.y - epsilon

    x_current = current_position.x
    y_current = current_position.y

    x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
    y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

    is_in_desired_pos = x_pos_are_close and y_pos_are_close

    return is_in_desired_pos



def get_name_rewards(args):

    file_name_str = '_'.join([x for x in ARG_LIST])

    return '/home/guoxiyue/cjf/results_SAC/rewards_files/' + file_name_str + '.csv'


def get_name_timesteps(args):

    file_name_str = '_'.join([x for x in ARG_LIST])

    return '/home/guoxiyue/cjf/results_SAC/timesteps_files/' + file_name_str + '.csv'

def get_name_successrate(args):

    file_name_str = '_'.join([x for x in ARG_LIST])

    return '/home/guoxiyue/cjf/results_SAC/successrate_files/' + file_name_str + '.csv'

def run(env, agents, file1, file2, file3, episodes_number, max_ts,marobot1_desired_point, marobot2_desired_point, marobot3_desired_point, test,log_interval):


    # for experiment_num in range(5):

    if test:
        for i, agent in enumerate(agents):
            agent.load()


    total_step = 0
    rewards_list = []
    timesteps_list = []
    success_list = []
    # max_score = -10000
    max_score = [-10000, -10000, -10000]

    for episode_num in range(episodes_number):

        state = env.reset()
        # print("initial state is:", state)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("epsiode number is:", episode_num)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # actions = []
        # for i, agent in enumerate(agents):
        #     action = env.action_space.sample()
        #     action_list = action.tolist()
        #     action_np = np.array(action_list)
        #     actions.append(action_np)

        #ADD
        # actions.append(np.array([0.0,0.0]))
        # actions.append(np.array([0.0,0.0]))


        # print("initial actions are:",actions)
        # print("initioal actions type is :", type(actions))

        # state, _, _,info = env.step(actions)

        # converting list of positions to an array
        # for a series of data transformation, transfer multi list to single list
        # state includes three parts: position and laser and position vector
        position_vector_1 = [state[0][0][0]-marobot1_desired_point.x,state[0][0][1]-marobot1_desired_point.y]
        position_vector_2 = [state[1][0][0]-marobot2_desired_point.x,state[1][0][1]-marobot2_desired_point.y]
        position_vector_3 = [state[2][0][0]-marobot3_desired_point.x,state[2][0][1]-marobot3_desired_point.y]

        # state_1 = state[0][0]+state[0][1]+position_vector_1
        # state_2 = state[1][0]+state[1][1]+position_vector_2
        # state_3 = state[2][0]+state[2][1]+position_vector_3


        # state_1 = state[0][0]+state[0][1]
        # state_2 = state[1][0]+state[1][1]
        # state_3 = state[2][0]+state[2][1]

        state_1 = state[0][0] +state[0][1]
        state_2 = state[1][0] +state[1][1]
        state_3 = state[2][0] +state[1][1]


        state_1 = np.asarray(state_1)
        state_1 = state_1.ravel()

        state_2 = np.asarray(state_2)
        state_2 = state_2.ravel()

        state_3 = np.asarray(state_3)
        state_3 = state_3.ravel()

        state_all = [state_1, state_2, state_3]
        # state_all_tensor = [torch.Tensor(state_1),torch.Tensor(state_2),torch.Tensor(state_3)]


        dones = False # refer to all the robots reach the desired points and whether end given episode
        # reward_all = 0
        reward_all = [0, 0, 0]
        time_step = 0
        done = [False,False,False]
        sub_episode_done = []
        if_done_index = [False,False,False]
        # label means the dones is whether first or not
        if_done_label = [0, 0, 0]

        while not dones and time_step < max_ts:
        # while  time_step < max_ts:

            print("time step number is:", time_step)
            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(state_all[i])
                print("action is:",action)
                # actions type need to transfer
                # action = np.array(action)
                actions.append(action)

            #add

            # actions.append(np.array([0.0,0.0]))
            # actions.append(np.array([0.0,0.0]))
            print("actions are:", actions)

            # decide each agent whether done, decide whether stop training
            # index = [i for i in range(len(done)) if done[i] == True]
            index = [i for i in range(len(done)) if done[i] == True]
            for i in index:
                actions[i] = np.array([0.0,0.0])


            next_state, reward, done, info = env.step(actions)
            print("Env done are:", done)

            dones = done[0] and done[1] and done[2]

            # record current position so that they can decide whether reach desired position or done
            current_position_marobot1 = next_state[0][0]
            current_position_marobot2 = next_state[1][0]
            current_position_marobot3 = next_state[2][0]

            #state includes three parts: position and laser and position vector
            position_vector_1 = [next_state[0][0][0] - marobot1_desired_point.x, next_state[0][0][1] - marobot1_desired_point.y]
            position_vector_2 = [next_state[1][0][0] - marobot2_desired_point.x, next_state[1][0][1] - marobot2_desired_point.y]
            position_vector_3 = [next_state[2][0][0] - marobot3_desired_point.x, next_state[2][0][1] - marobot3_desired_point.y]

            # next_state_1 = next_state[0][0] + next_state[0][1] + position_vector_1
            # next_state_2 = next_state[1][0] + next_state[1][1] + position_vector_2
            # next_state_3 = next_state[2][0] + next_state[2][1] + position_vector_3

            next_state_1 = next_state[0][0] + next_state[0][1]
            next_state_2 = next_state[1][0] + next_state[1][1]
            next_state_3 = next_state[2][0] + next_state[2][1]

            # next_state_1 = next_state[0][0]
            # next_state_2 = next_state[1][0]
            # next_state_3 = next_state[2][0]

            next_state_1 = np.asarray(next_state_1)
            next_state_1 = next_state_1.ravel()

            next_state_2 = np.asarray(next_state_2)
            next_state_2 = next_state_2.ravel()

            next_state_3 = np.asarray(next_state_3)
            next_state_3 = next_state_3.ravel()

            next_state_all = [next_state_1, next_state_2, next_state_3]
            # next_state_all_tensor = [torch.Tensor(next_state_1), torch.Tensor(next_state_2), torch.Tensor(next_state_3)]
            # print("next_state is:", next_state)



            if not test:
                print("total_step is:",total_step)
                print("filling_steps is",filling_steps)

                for i, agent in enumerate(agents):
                    state = state_all[i]
                    next_state = next_state_all[i]
                    # agent.observe((state, actions, reward[i], next_state, dones))
                    # done_mask = 0.0 if done[i] else 1.0
                    if done[i] is True and if_done_index[i] is False:
                        if_done_index[i] = True
                        if_done_label[i] = time_step
                        # agent.store_transition(state, actions[i],  done[i], next_state, reward[i])
                        agent.store(state, actions[i], reward[i], next_state, done[i])
                        # if agent.max_action > agent.capacity:
                        print("<--------------------------------------------->")
                        print("num_transitio is-------------->>>>>>>>>>>>>>>>>:", agent.num_transition)
                        if agent.num_transition > agent.capacity:
                           agent.update()

                    elif if_done_index[i] is False:
                        # agent.store_transition(state, actions[i],  done[i], next_state, reward[i])
                        agent.store(state, actions[i], reward[i], next_state, done[i])
                        if agent.num_transition > agent.capacity:
                           agent.update()

                for i, agent in enumerate(agents):
                    if if_done_index[i] is True and time_step > if_done_label[i]:
                        reward[i] = 0

                total_step += 1
                time_step += 1
                # state = next_state
                state_1 = next_state_1
                state_2 = next_state_2
                state_3 = next_state_3
                state_all = [state_1, state_2, state_3]

                reward_all_np = np.add(np.array(reward_all), np.array(reward))
                reward_all = reward_all_np.tolist()
            else:
                total_step += 1
                time_step += 1
                # state = next_state
                state_1 = next_state_1
                state_2 = next_state_2
                state_3 = next_state_3
                state_all = [state_1, state_2, state_3]
                # state_all_tensor = [torch.Tensor(state_1), torch.Tensor(state_2), torch.Tensor(state_3)]
                # reward_all += reward
                # reward_all = sum(reward) + reward_all

            # in each episode, we will decide if each agent reach desired point and calculate success rate
            if dones == True or time_step >= max_ts:
                current_position_1 = Point()
                current_position_2 = Point()
                current_position_3 = Point()
                # for marobot1:
                current_position_1.x = current_position_marobot1[0]
                current_position_1.y = current_position_marobot1[1]
                current_position_1.z = 0.0

                # for marobot2:
                current_position_2.x = current_position_marobot2[0]
                current_position_2.y = current_position_marobot2[1]
                current_position_2.z = 0.0

                # for marobot3:
                current_position_3.x = current_position_marobot3[0]
                current_position_3.y = current_position_marobot3[1]
                current_position_3.z = 0.0

                # MAX_X = 10.0
                # MIN_X = -10.0
                # MAX_Y = 10.0
                # MIN_Y = -10.0

                desired_current_position = {str(current_position_1): marobot1_desired_point,
                                            str(current_position_2): marobot2_desired_point,
                                            str(current_position_3): marobot3_desired_point}

                _episode_done = False
                for current_position in [current_position_1, current_position_2, current_position_3]:

                    # We see if it got to the desired point
                    if is_in_desired_position(desired_current_position[str(current_position)],
                                                   current_position):
                       _episode_done = True
                    else:
                        _episode_done = False

                    # sub_episode_done = sub_episode_done.append(self._episode_done)
                    sub_episode_done.append(_episode_done)

                _episode_dones = sub_episode_done[:]


        rewards_list.append(reward_all)
        timesteps_list.append(time_step)

        # we can calculate success rate whether each agent reach desired point
        success_percent = round(_episode_dones.count(True)/3.0,2)
        success_list.append(success_percent)


        for i, agent in enumerate(agents):
            if episode_num % log_interval == 0:
                agent.save()
            agent.writer.add_scalar('reward_episode', reward_all[i], global_step=episode_num)




        print("Episode {p}, Score: {s}, Final Step: {t}, Goal: {g}".format(p=episode_num, s=reward_all,
                                                                           t=time_step, g=done))

        # agent.writer.add_scalar('reward', rewards, i_episode)
        # print("episode:{}, reward:{}, buffer_capacity:{}".format(episode_num, reward_all))


        # if not test:
        if episode_num % 1 == 0:
            df = pd.DataFrame(rewards_list, columns=['score-1','score-2','score-3'])
            print("file1 name is:", file1)
            df.to_csv(file1)

            df = pd.DataFrame(timesteps_list, columns=['steps'])
            df.to_csv(file2)

            if total_step >= filling_steps:
                for i, agent in enumerate(agents):
                    # if reward_all > max_score:
                    #     # for agent in agents:
                    #     #     agent.save_model()
                    #     max_score = reward_all
                    if reward_all[i] > max_score[i]:
                        max_score[i] = reward_all[i]

        # record success rate
        df = pd.DataFrame(success_list, columns=['success_rate'])
        df.to_csv(file3)



if __name__ =="__main__":

    rospy.init_node('sac_training_continous_V1', anonymous=True, log_level=rospy.WARN)

    # use the cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'


    seed = rospy.get_param("turtlebot2/seed")
    test = rospy.get_param("turtlebot2/test")

    # play
    experiment_num = rospy.get_param("/turtlebot2/experiment_num")
    episodes_number = rospy.get_param("/turtlebot2/iteration")
    max_ts = rospy.get_param("/turtlebot2/max_timestep")

    # network
    hidden_size = rospy.get_param("/turtlebot2/hidden_size")

    # replay
    memory_size = rospy.get_param("/turtlebot2/memory_size")
    batch_size = rospy.get_param("/turtlebot2/batch_size")

    # learning
    gamma = rospy.get_param("/turtlebot2/gamma")
    lr = rospy.get_param("/turtlebot2/lr")
    tau = rospy.get_param("/turtlebot2/tau")
    update_iteration = rospy.get_param("/turtlebot2/update_iteration")
    directory = rospy.get_param("/turtlebot2/directory")

    target_update_interval = rospy.get_param("/turtlebot2/target_update_interval")
    gradient_steps = rospy.get_param("/turtlebot2/gradient_steps")
    capacity = rospy.get_param("/turtlebot2/capacity")
    num_hidden_layers = rospy.get_param("/turtlebot2/num_hidden_layers")
    num_hidden_units_per_layer = rospy.get_param("/turtlebot2/num_hidden_units_per_layer")
    sample_frequency = rospy.get_param("/turtlebot2/sample_frequency")
    activation = rospy.get_param("/turtlebot2/activation")
    log_interval = rospy.get_param("/turtlebot2/log_interval")
    load = rospy.get_param("/turtlebot2/load")




    # DQN Parameters


    test = rospy.get_param("/turtlebot2/test")
    filling_steps = rospy.get_param("/turtlebot2/first_step_memory")

    max_random_moves = rospy.get_param("/turtlebot2/max_random_moves")
    num_agents = rospy.get_param("/turtlebot2/agents_number")


    learning_rate = rospy.get_param("/turtlebot2/learning_rate")

    memory_capacity = rospy.get_param("/turtlebot2/memory_capacity")
    prioritization_scale = rospy.get_param("/turtlebot2/prioritization_scale")

    target_frequency = rospy.get_param("/turtlebot2/target_frequency")
    maximum_exploration = rospy.get_param("/turtlebot2/maximum_exploration")

    # self.test = rospy.get_param("/turtlebot2/test")



    memory_size = rospy.get_param("/turtlebot2/memory_size")



    batch_size = rospy.get_param("/turtlebot2/batch_size")



    agent_name = rospy.get_param("/turtlebot2/agent_name")






    marobot1_desired_point = Point()
    marobot1_desired_point.x = rospy.get_param("/turtlebot2/marobot1/desired_pose/x")
    marobot1_desired_point.y = rospy.get_param("/turtlebot2/marobot1/desired_pose/y")
    marobot1_desired_point.z = rospy.get_param("/turtlebot2/marobot1/desired_pose/z")

    marobot2_desired_point = Point()
    marobot2_desired_point.x = rospy.get_param("/turtlebot2/marobot2/desired_pose/x")
    marobot2_desired_point.y = rospy.get_param("/turtlebot2/marobot2/desired_pose/y")
    marobot2_desired_point.z = rospy.get_param("/turtlebot2/marobot2/desired_pose/z")

    marobot3_desired_point = Point()
    marobot3_desired_point.x = rospy.get_param("/turtlebot2/marobot3/desired_pose/x")
    marobot3_desired_point.y = rospy.get_param("/turtlebot2/marobot3/desired_pose/y")
    marobot3_desired_point.z = rospy.get_param("/turtlebot2/marobot3/desired_pose/z")

    # env
    env = gym.make("MultiagentTurtleBot2-v1")
    env = NormalizedActions(env)
    rospy.loginfo("Gym environment done")

    #set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    print("state_dim is:", state_dim)
    action_dim = env.action_space.shape[0]
    print("action_dim is:", action_dim)
    max_action = float(env.action_space.high[0])
    min_Val = torch.tensor(1e-7).float().to(device)
    Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])
    # action_space = rospy.get_param("/turtlebot2/n_actions")
    # state_size = rospy.get_param("/turtlebot2/n_observations")



    all_agents = []
    for i in range(num_agents):
        agent_name = "SAC"+ str(i)
        agent = SAC(state_dim = state_dim, action_dim = action_dim, min_Val=min_Val, Transition=Transition, learning_rate=learning_rate,
                             capacity=capacity, gradient_steps=gradient_steps, batch_size=batch_size, gamma=gamma, tau=tau, max_action=max_action, device=device, agent_id=i)
        all_agents.append(agent)

    # rewards_file = []
    # timesteps_file = []
    # successrate_file = []

    # for i in range(experiment_num):
    #     rewards_file.append(get_name_rewards(ARG_LIST + [str(i)]))
    #     print("ARG_LIST is:", ARG_LIST + [str(i)])
    #     timesteps_file.append(get_name_timesteps(ARG_LIST + [str(i)]))
    #     successrate_file.append(get_name_successrate(ARG_LIST + [str(i)]))

    rewards_file = get_name_rewards(ARG_LIST)
    timesteps_file = get_name_timesteps(ARG_LIST)
    successrate_file = get_name_successrate(ARG_LIST)

    run(env, agents=all_agents, file1=rewards_file, file2=timesteps_file, file3=successrate_file, episodes_number=episodes_number,
        max_ts=max_ts,  marobot1_desired_point=marobot1_desired_point, marobot2_desired_point=marobot2_desired_point, marobot3_desired_point=marobot3_desired_point, test=test, log_interval=log_interval)
    env.close()

