#!/usr/bin/env python3

'''
LAST UPDATE: 2022.04.02

AUTHOR: Neset Unver Akmandor (NUA)
        Eric Dusel (ED)
        Gary Lvov (GML)
        Hongyu Li (LHY)

E-MAIL: akmandor.n@northeastern.edu
        dusel.e@northeastern.edu
        lvov.g@northeastern.edu
	    li.hongyu1@northeastern.edu

DESCRIPTION: TODO...

REFERENCES:

NUA TODO:
'''

import rospy
import numpy as np
import time
import math
import cv2
import os
import csv
import random
import pathlib
import pickle
from matplotlib import pyplot as plt
from PIL import Image
from squaternion import Quaternion

import tf
import tf2_ros
import roslaunch
import rospkg
from std_msgs.msg import Header, Bool, Float32MultiArray
from geometry_msgs.msg import Pose, PoseStamped, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path, OccupancyGrid
from nav_msgs.srv import GetPlan
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelStates

from gym import spaces
from gym.envs.registration import register

from openai_ros.robot_envs import turtlebot3_env
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest

from tentabot_drl.tentabot_drl_config import *
from tentabot.srv import *

#from imitation.data.types import TrajectoryWithRew
#from imitation.data import types

'''
DESCRIPTION: TODO...
'''
class TurtleBot3TentabotDRL(turtlebot3_env.TurtleBot3Env):

    '''
    DESCRIPTION: TODO...This Task Env is designed for having the TurtleBot3 in some kind of maze.
    It will learn how to move around the maze without crashing.
    '''
    def __init__(self, robot_id=0, data_folder_path=""):

        ### Initialize Parameters

        ## General
        self.robot_id = robot_id
        self.previous_robot_id = self.robot_id
        self.robot_namespace = "turtlebot3_" + str(self.robot_id)
        self.data_folder_path = data_folder_path
        self.world_name = rospy.get_param('world_name', "")
        self.next_world_name = self.world_name
        self.init_flag = False
        self.step_num = 0
        self.total_step_num = 0
        self.total_collisions = 0
        self.step_reward = 0.0
        self.episode_reward = 0.0
        self.total_mean_episode_reward = 0.0
        self.goal_reaching_status = Bool()
        self.goal_reaching_status.data = False
        self.action_counter = 0
        self.observation_counter = 0
        self.odom_dict = {}
        self.previous_area_id = 0
        self.obs_data = {}
        self.move_base_goal = PoseStamped()
        self.move_base_flag = False
        self.training_data = []
        self.training_data.append(["episode_reward"])
        self.oar_data = []
        self.episode_oar_data = dict(obs=[], acts=[], infos=None, terminal=[], rews=[])
        #self.time_old = time.time()
        self.validation_ep_num = 0
        self.validation_flag = False

        self.config = Config(data_folder_path=data_folder_path)

        ## Set Observation-Action-Reward data filename
        self.oar_data_file = data_folder_path + "oar_data.csv"

        # Rospack path for pedsim reset
        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()
        rospack_list = rospack.list() 
        self.tentabot_path = rospack.get_path('tentabot')

        # Subscriptions
        rospy.Subscriber("/" + str(self.robot_namespace) + "/scan", LaserScan, self.callback_laser_scan)
        rospy.Subscriber("/" + str(self.robot_namespace) + "/laser_image", OccupancyGrid, self.callback_laser_image)
        rospy.Subscriber("/" + str(self.robot_namespace) + "/laser_rings", Float32MultiArray, self.callback_laser_rings)

        #if self.config.observation_space_type == "laser_image_2DCNN_FC":
        #    rospy.Subscriber("/" + str(self.robot_namespace) + "/laser_image", OccupancyGrid, self.callback_laser_image)

        #if self.config.observation_space_type == "laser_rings_2DCNN_FC":
        #    rospy.Subscriber("/" + str(self.robot_namespace) + "/laser_rings", Float32MultiArray, self.callback_rings_image)

        # Services
        if  self.config.observation_space_type == "Tentabot_FC" or \
            self.config.observation_space_type == "Tentabot_1DCNN_FC" or \
            self.config.observation_space_type == "Tentabot_2DCNN_FC" or \
            self.config.observation_space_type == "Tentabot_2DCNN" or \
            self.config.observation_space_type == "Tentabot_WP_FC":

            rospy.wait_for_service('rl_step')
            self.srv_rl_step = rospy.ServiceProxy('rl_step', rl_step, True)

            rospy.wait_for_service('update_goal')
            self.srv_update_goal = rospy.ServiceProxy('update_goal', update_goal, True)

            #rospy.wait_for_service('reset_map_utility')
            #self.srv_reset_map_utility = rospy.ServiceProxy('reset_map_utility', reset_map_utility)

        if self.config.observation_space_type == "laser_WP_1DCNN_FC" or self.config.observation_space_type == "Tentabot_WP_FC":
            
            rospy.wait_for_service('/move_base/make_plan')
            rospy.wait_for_service('/move_base/clear_costmaps')
            self.srv_move_base_get_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan, True)
            # self.srv_clear_costmap = rospy.ServiceProxy('/move_base/clear_costmaps', Empty, True)

        # Publishers
        self.goal_reaching_status_pub = rospy.Publisher(self.robot_namespace + '/goal_reaching_status', Bool, queue_size=1)
        self.goal_visu_pub = rospy.Publisher(self.robot_namespace + '/nav_goal', MarkerArray, queue_size=1)
        self.filtered_laser_pub = rospy.Publisher(self.robot_namespace + '/laser/scan_filtered', LaserScan, queue_size=1)
        self.debug_visu_pub = rospy.Publisher('/debug_visu', MarkerArray, queue_size=1)

        # Initialize OpenAI Gym Structure
        self.get_init_pose(init_flag=False)

        super(TurtleBot3TentabotDRL, self).__init__(robot_namespace=self.robot_namespace, initial_pose=self.initial_pose, data_folder_path=data_folder_path, velocity_control_msg=self.config.velocity_control_msg)

        self.get_goal_location()
        self.init_observation_action_space()

        #print("turtlebot3_tentabot_drl::__init__ -> obs laser shape: " + str(self.obs["laser"].shape))
        #print("turtlebot3_tentabot_drl::__init__ -> obs target_action shape: " + str(self.obs["target_action"].shape))

        self.reward_range = (-np.inf, np.inf)
        self.init_flag = True

    '''
    DESCRIPTION: TODO...Sets the Robot in its init pose
    '''
    def _set_init_pose(self):

        self.move_base( self.config.init_lateral_speed,
                        self.config.init_angular_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True

    '''
    DESCRIPTION: TODO...Inits variables needed to be initialised each time we reset at the start
    of an episode.
    :return:
    '''
    def _init_env_variables(self):

        #print("turtlebot3_tentabot_drl::_init_env_variables -> self.total_step_num: " + str(self.total_step_num))

        if self.episode_num:
            #self.total_mean_episode_reward = round((self.total_mean_episode_reward * (self.episode_num - 1) + self.episode_reward) / self.episode_num, self.config.mantissa_precision)
            self.total_mean_episode_reward = (self.total_mean_episode_reward * (self.episode_num - 1) + self.episode_reward) / self.episode_num

            ## Add training data
            self.training_data.append([self.episode_reward])

        print("--------------")
        print("turtlebot3_tentabot_drl::_init_env_variables -> robot_id: {}".format(self.robot_id))
        print("turtlebot3_tentabot_drl::_init_env_variables -> step_num: {}".format(self.step_num))
        print("turtlebot3_tentabot_drl::_init_env_variables -> total_step_num: {}".format(self.total_step_num))
        print("turtlebot3_tentabot_drl::_init_env_variables -> episode_num: {}".format(self.episode_num))
        print("turtlebot3_tentabot_drl::_init_env_variables -> total_collisions: {}".format(self.total_collisions))
        print("turtlebot3_tentabot_drl::_init_env_variables -> episode_reward: {}".format(self.episode_reward))
        print("turtlebot3_tentabot_drl::_init_env_variables -> total_mean_episode_reward: {}".format(self.total_mean_episode_reward))
        print("--------------")

        self.previous_robot_id = self.robot_id
        self.episode_reward = 0.0
        self._episode_done = False
        self._reached_goal = False
        self.step_num = 0

        '''
        print("turtlebot3_tentabot_drl::_init_env_variables -> BEFORE client_reset_map_utility")
        # Reset Map
        success_reset_map_utility = self.client_reset_map_utility()
        print("turtlebot3_tentabot_drl::_init_env_variables -> AFTER client_reset_map_utility")
        '''

        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)

        self.previous_distance2goal = self.get_distance2goal()
        
        self.previous_action = np.array([[self.config.init_lateral_speed, self.config.init_angular_speed]]).reshape(self.config.fc_obs_shape)

        #self.update_global_path_length()

        self.reinit_observation()
    
    '''
    DESCRIPTION: TODO...Here we define what sensor data defines our robots observations
    To know which Variables we have acces to, we need to read the
    TurtleBot3Env API DOCS
    :return:
    '''
    def _get_obs(self):

        #print("turtlebot3_tentabot_drl::_get_obs -> self.total_step_num: " + str(self.total_step_num))

        # Update target observation
        self.update_observation()

        # Check if the goal is reached
        self.goal_check()

        return self.obs

    '''
    DESCRIPTION: TODO...
    '''
    def _set_action(self, action):

        #print("turtlebot3_tentabot_drl::_set_action -> self.total_step_num: " + str(self.total_step_num))
        
        linear_speed = self.config.velocity_control_data[action, 0]
        angular_speed = float(self.config.velocity_control_data[action, 1])

        self.previous_action = np.array([[linear_speed, angular_speed]], dtype=np.float32).reshape(self.config.fc_obs_shape)
        self.act = action

        # We tell TurtleBot3 the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)

    '''
    DESCRIPTION: TODO...
    '''
    def _is_done(self, observations):

        #print("turtlebot3_tentabot_drl::_is_done -> self.total_step_num: " + str(self.total_step_num))

        if self._episode_done and (not self._reached_goal):

            rospy.logdebug("turtlebot3_tentabot_drl::_is_done -> Boooo! Episode done but not reached the goal...")
            print("turtlebot3_tentabot_drl::_is_done -> Boooo! Episode done but not reached the goal...")

        elif self._episode_done and self._reached_goal:

            rospy.logdebug("turtlebot3_tentabot_drl::_is_done -> Gotcha! Episode done and reached the goal!")
            print("turtlebot3_tentabot_drl::_is_done -> Gotcha! Episode done and reached the goal!")
            
        else:

            rospy.logdebug("turtlebot3_tentabot_drl::_is_done -> Not yet bro...")
            #print("turtlebot3_tentabot_drl::_is_done -> Not yet bro...")

        return self._episode_done

    '''
    DESCRIPTION: TODO...
    '''
    def _compute_reward(self, observations, done):

        #print("turtlebot3_tentabot_drl::_compute_reward -> self.total_step_num: " + str(self.total_step_num))

        self.total_step_num += 1
        self.step_num += 1

        if self.step_num >= self.config.max_episode_steps:

            self._episode_done = True
            print("turtlebot3_tentabot_drl::_compute_reward -> Too late...")

        if self._episode_done and (not self._reached_goal):

            self.step_reward = self.config.penalty_terminal_fail
            self.goal_reaching_status.data = False
            self.goal_reaching_status_pub.publish(self.goal_reaching_status)

            # Update initial pose and goal for the next episode
            self.get_init_pose()
            self.get_goal_location()

        elif self._episode_done and self._reached_goal:

            #self.step_reward = self.config.reward_terminal_success + self.config.reward_terminal_mintime * (self.config.max_episode_steps - self.step_num) / self.config.max_episode_steps
            self.step_reward = self.config.reward_terminal_success
            self.goal_reaching_status.data = True
            self.goal_reaching_status_pub.publish(self.goal_reaching_status)

            # Update initial pose and goal for the next episode
            self.get_init_pose()
            self.get_goal_location()

        else:

            '''
            if current_distance2goal > self.init_distance2goal:
                #penalty
                self.step_reward = -1 * self.config.reward_cumulative_step * current_distance2goal / (self.config.max_episode_steps * self.init_distance2goal)

            else:
                #reward
                self.step_reward = self.config.reward_cumulative_step * (self.init_distance2goal - current_distance2goal) / (self.config.max_episode_steps * self.init_distance2goal)
            '''

            current_distance2goal = self.get_distance2goal()
            penalty_step = self.config.penalty_cumulative_step / self.config.max_episode_steps
            rp_step = self.config.reward_step_scale * (self.previous_distance2goal - current_distance2goal)
            self.step_reward = penalty_step + rp_step
            self.previous_distance2goal = current_distance2goal

            #print("turtlebot3_tentabot_drl::_compute_reward -> reward_step: " + str(reward_step))

            '''
            penalty_safety = 0
            if self.min_distance2obstacle < self.config.safety_range_threshold:
                
                penalty_safety = self.config.penalty_safety_scale * (self.config.safety_range_threshold / self.min_distance2obstacle)
                #print("turtlebot3_tentabot_drl::_compute_reward -> penalty_safety: {}".format(penalty_safety))
            '''
            #self.step_reward = round(penalty_safety + reward_step, self.config.mantissa_precision)

            '''
            time_now = time.time()
            dt = time_now - self.time_old
            self.time_old = time_now
            
            print("----------------------")
            print("turtlebot3_tentabot_drl::_compute_reward -> current_distance2goal: " + str(current_distance2goal))
            #print("turtlebot3_tentabot_drl::_compute_reward -> init_distance2goal: " + str(self.init_distance2goal))
            print("turtlebot3_tentabot_drl::_compute_reward -> max_episode_steps: " + str(self.config.max_episode_steps))
            print("turtlebot3_tentabot_drl::_compute_reward -> reward_terminal_success: " + str(self.config.reward_terminal_success))
            print("turtlebot3_tentabot_drl::_compute_reward -> reward_step_scale: " + str(self.config.reward_step_scale))
            print("turtlebot3_tentabot_drl::_compute_reward -> penalty_terminal_fail: " + str(self.config.penalty_terminal_fail))
            print("turtlebot3_tentabot_drl::_compute_reward -> penalty_cumulative_step: " + str(self.config.penalty_cumulative_step))
            print("turtlebot3_tentabot_drl::_compute_reward -> penalty_step: " + str(penalty_step))
            print("turtlebot3_tentabot_drl::_compute_reward -> rp_step: " + str(rp_step))
            print("turtlebot3_tentabot_drl::_compute_reward -> step_reward: " + str(self.step_reward))
            #print("turtlebot3_tentabot_drl::_compute_reward -> dt: " + str(dt))
            #print("turtlebot3_tentabot_drl::_compute_reward -> max_lateral_speed: " + str(self.config.max_lateral_speed))
            #print("turtlebot3_tentabot_drl::_compute_reward -> max_step_reward: " + str(round(self.config.max_lateral_speed * dt, self.config.mantissa_precision)))
            print("----------------------")
            '''
            
        self.episode_reward += self.step_reward

        rospy.logdebug("turtlebot3_tentabot_drl::_compute_reward -> step_reward: " + str(self.step_reward))
        rospy.logdebug("turtlebot3_tentabot_drl::_compute_reward -> episode_reward: " + str(self.episode_reward))
        rospy.logdebug("turtlebot3_tentabot_drl::_compute_reward -> total_step_num: " + str(self.total_step_num))

        '''
        print("**********************")
        print("turtlebot3_tentabot_drl::_compute_reward -> self.step_reward: " + str(self.step_reward))
        print("----------------------")
        '''

        '''
        # Save Observation-Action-Reward data into a file
        self.save_oar_data()

        if self._episode_done and (len(self.episode_oar_data['obs']) > 1):

            #print("turtlebot3_tentabot_drl::save_oar_data -> episode_oar_data obs len: " + str(len(self.episode_oar_data['obs'])))
            #print("turtlebot3_tentabot_drl::save_oar_data -> episode_oar_data acts len: " + str(len(self.episode_oar_data['acts'])))

            if self.goal_reaching_status.data:
                info_data = np.ones(len(self.episode_oar_data['acts']))
            else:
                info_data = np.zeros(len(self.episode_oar_data['acts']))

            self.oar_data.append(TrajectoryWithRew( obs=np.array(self.episode_oar_data['obs']), 
                                                    acts=np.array(self.episode_oar_data['acts']),
                                                    infos=np.array(info_data),
                                                    terminal=True,
                                                    rews=np.array(self.episode_oar_data['rews']),))
        '''

        if self.total_step_num == self.config.training_timesteps:
            
            # Write Observation-Action-Reward data into a file
            #self.write_oar_data()

            ## Write training data
            write_data(self.data_folder_path + "training_data.csv", self.training_data)

        return self.step_reward

    # Internal TaskEnv Methods

    '''
    DESCRIPTION: TODO...
    '''
    def write_data(file, data):
        file_status = open(file, 'a')
        with file_status:
            write = csv.writer(file_status)
            write.writerows(data)
            print("tentabot_drl_training::write_data -> Data is written in " + str(file))

    '''
    DESCRIPTION: TODO...
    '''
    def save_oar_data(self):

        if  self.config.observation_space_type == "laser_FC" or \
            self.config.observation_space_type == "Tentabot_FC":
        
                #print("----------------------------------")
                #print("turtlebot3_tentabot_drl::save_oar_data -> self.obs shape: " + str(self.obs.shape))
                #print("turtlebot3_tentabot_drl::save_oar_data -> self.previous_action shape: " + str(self.previous_action.shape))
                #print("")

                obs_data = self.obs.reshape((-1))

                #print("turtlebot3_tentabot_drl::save_oar_data -> obs_data shape: " + str(obs_data.shape))
                #print("----------------------------------")

                # Save Observation-Action-Reward Data
                self.episode_oar_data['obs'].append(obs_data)

                if not self._episode_done:

                    self.episode_oar_data['acts'].append(self.act)
                    #self.episode_oar_data['infos'].append()
                    #self.episode_oar_data['terminal'].append(self._episode_done)
                    self.episode_oar_data['rews'].append(self.step_reward)   

                '''
                print("----------------------------------")
                print("turtlebot3_tentabot_drl::save_oar_data -> episode_oar_data obs type: " + str(type(self.episode_oar_data['obs'])))
                print("turtlebot3_tentabot_drl::save_oar_data -> episode_oar_data obs len: " + str(len(self.episode_oar_data['obs'])))
                print("turtlebot3_tentabot_drl::save_oar_data -> episode_oar_data acts len: " + str(len(self.episode_oar_data['acts'])))
                print("turtlebot3_tentabot_drl::save_oar_data -> episode_oar_data: " + str(self.episode_oar_data))
                #print("turtlebot3_tentabot_drl::save_oar_data -> episode_oar_data obs: " + str(self.episode_oar_data.obs))
                print("turtlebot3_tentabot_drl::save_oar_data -> episode_oar_data obs shape: " + str(self.episode_oar_data.obs.shape))
                #print("turtlebot3_tentabot_drl::save_oar_data -> oar_data: " + str(self.oar_data))
                print("----------------------------------")
                '''

    '''
    DESCRIPTION: TODO...Save a sequence of Trajectories to disk.

        Args:
            path: Trajectories are saved to this path.
            trajectories: The trajectories to save.
    '''
    def write_oar_data(self) -> None:

        path = self.data_folder_path + "oar_data.pkl"
        trajectories = self.oar_data
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = f"{path}.tmp"
        
        with open(tmp_path, "wb") as f:
            pickle.dump(trajectories, f)

        # Ensure atomic write
        os.replace(tmp_path, path)

        print("turtlebot3_tentabot_drl::write_oar_data -> Written Observation-Action-Reward data!")

    '''
    DESCRIPTION: TODO...
    '''
    def callback_laser_scan(self, data):

        if self.init_flag:
            self.check_collision(data)
            self.filter_laser_scan(data)

        else:
            self.config.set_laser_data(data)

    '''
    DESCRIPTION: TODO...
    '''
    def callback_laser_image(self, data):

        self.config.laser_image_width = data.info.width
        self.config.laser_image_height = data.info.height

        laser_image = np.array(data.data[0:self.config.laser_image_width])
        for i in range(1, self.config.laser_image_height):

            idx_from = i*self.config.laser_image_width
            idx_to = idx_from + self.config.laser_image_width
            laser_image_row = np.array(data.data[idx_from:idx_to])
            laser_image = np.vstack([laser_image, laser_image_row])

        max_scale = 1 / np.max(laser_image)
        self.laser_image = max_scale * laser_image

        '''
        if self.step_num == 50:

            imi = (self.laser_image * 255).astype(np.uint8)
            im = Image.fromarray(imi)
            im = im.convert("L")
            im.save(self.data_folder_path + "laser_image.jpeg")
            np.savetxt(self.data_folder_path + "laser_image.txt", self.laser_image)
        '''

        '''
        print("----------------------------------")
        #print("turtlebot3_tentabot_drl::callback_laser_image -> laser_image_width: " + str(self.config.laser_image_width))
        #print("turtlebot3_tentabot_drl::callback_laser_image -> laser_image_height: " + str(self.config.laser_image_height))
        #print("turtlebot3_tentabot_drl::callback_laser_image -> data.info.resolution: " + str(data.info.resolution))
        #print("turtlebot3_tentabot_drl::callback_laser_image -> data.info.width: " + str(data.info.width))
        #print("turtlebot3_tentabot_drl::callback_laser_image -> data.info.height: " + str(data.info.height))
        #print("turtlebot3_tentabot_drl::callback_laser_image -> data type: " + str(type(data.data)))
        #print("turtlebot3_tentabot_drl::callback_laser_image -> data len: " + str(len(data.data)))
        #print("turtlebot3_tentabot_drl::callback_laser_image -> laser_image len: " + str(len(laser_image)))
        #print("turtlebot3_tentabot_drl::callback_laser_image -> self.laser_image shape: " + str(self.laser_image.shape))
        print("turtlebot3_tentabot_drl::callback_laser_image -> max_scale: " + str(max_scale))
        print("turtlebot3_tentabot_drl::callback_laser_image -> maxi: " + str(maxi))
        print("----------------------------------")
        '''
    
    '''
    DESCRIPTION: TODO...
    '''
    def callback_laser_rings(self, data):

        #self.config.laser_image_width = data.info.width
        #self.config.laser_image_height = data.info.height

        laser_image = np.array(data.data[0:self.config.laser_image_width])
        for i in range(1, self.config.laser_image_height):

            idx_from = i*self.config.laser_image_width
            idx_to = idx_from + self.config.laser_image_width
            laser_image_row = np.array(data.data[idx_from:idx_to])
            laser_image = np.vstack([laser_image, laser_image_row])

        max_scale = 1 / np.max(laser_image)
        self.laser_image = max_scale * laser_image

        '''
        if self.step_num == 50:

            imi = (self.laser_image * 255).astype(np.uint8)
            im = Image.fromarray(imi)
            im = im.convert("L")
            im.save(self.data_folder_path + "laser_image.jpeg")
            np.savetxt(self.data_folder_path + "laser_image.txt", self.laser_image)
        '''

        '''
        print("----------------------------------")
        print("turtlebot3_tentabot_drl::callback_laser_rings -> layout dim type: " + str(type(data.layout.dim)))
        print("turtlebot3_tentabot_drl::callback_laser_rings -> layout dim size: " + str(len(data.layout.dim)))
        print("turtlebot3_tentabot_drl::callback_laser_rings -> data type: " + str(type(data.data)))
        print("turtlebot3_tentabot_drl::callback_laser_rings -> data len: " + str(len(data.data)))
        print("turtlebot3_tentabot_drl::callback_laser_rings -> laser_image shape: " + str(self.laser_image.shape))
        print("turtlebot3_tentabot_drl::callback_laser_rings -> laser_image: ")
        print(self.laser_image)
        print("----------------------------------")
        '''
        
    '''
    DESCRIPTION: TODO...
    '''
    def callback_move_base_global_plan(self, data):

        self.move_base_global_plan = data.poses
        self.move_base_flag = True

    '''
    DESCRIPTION: TODO... Update the odometry data
    '''
    def update_odom(self):

        self.odom_data = self.get_odom()
        
        q = Quaternion( self.odom_data.pose.pose.orientation.w,
                        self.odom_data.pose.pose.orientation.x,
                        self.odom_data.pose.pose.orientation.y,
                        self.odom_data.pose.pose.orientation.z)
        e = q.to_euler(degrees=False)

        self.odom_dict["x"] = self.odom_data.pose.pose.position.x
        self.odom_dict["y"] = self.odom_data.pose.pose.position.y
        self.odom_dict["z"] = self.odom_data.pose.pose.position.z
        self.odom_dict["theta"] = e[2]
        self.odom_dict["u"] = self.odom_data.twist.twist.linear.x
        self.odom_dict["omega"] = self.odom_data.twist.twist.angular.z
        self.config.set_odom(self.odom_dict)

    '''
    DESCRIPTION: TODO... Check if the goal is reached
    '''
    def goal_check(self):

        current_distance2goal = self.get_distance2goal()
        if (current_distance2goal < self.config.goal_close_threshold):
            self._episode_done = True
            self._reached_goal = True

    '''
    DESCRIPTION: TODO...Gets the initial location of the robot to reset
    '''
    def get_init_pose(self, init_flag=True):

        self.initial_pose = {}
        robot0_init_yaw = 0.0
        self.world_name = self.next_world_name

        if self.world_name == "training_garden_static_0":

            initial_pose_areas_x = []
            initial_pose_areas_x.extend(([-0.5,0.5], [1.5,2.5], [-2.5,-1], [-2.5,0.5]))

            initial_pose_areas_y = []
            initial_pose_areas_y.extend(([-0.5,0.5], [0.5,1.5], [1,1.5], [-1.5,-1]))

            area_idx = random.randint(0, len(initial_pose_areas_x)-1)
            self.robot_init_area_id = area_idx

            self.initial_pose["x_init"] = random.uniform(initial_pose_areas_x[area_idx][0], initial_pose_areas_x[area_idx][1])
            self.initial_pose["y_init"] = random.uniform(initial_pose_areas_y[area_idx][0], initial_pose_areas_y[area_idx][1])
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = random.uniform(0.0, 2*math.pi)

        elif self.world_name == "training_garden_static_1":

            initial_pose_areas_x = []
            #initial_pose_areas_x.extend(([-1.5,1.5], [4.5,5.5], [-5.5,-2], [1,5], [-5.5,-3.5]))
            initial_pose_areas_x.extend(([-1.5,0.5], [1.0,5.5], [-5.5,-3], [4.0,5.5], [-5.5,-4.5]))

            initial_pose_areas_y = []
            #initial_pose_areas_y.extend(([-1,1], [1,4.5], [3.5, 4.5], [-4,-2], [-4.5,-2]))
            initial_pose_areas_y.extend(([0,2], [4,4.5], [4.0,4.5], [-4.5,-3.0], [-4.5,-3.5]))

            area_idx = random.randint(0, len(initial_pose_areas_x)-1)
            self.robot_init_area_id = area_idx

            self.initial_pose["x_init"] = random.uniform(initial_pose_areas_x[area_idx][0], initial_pose_areas_x[area_idx][1])
            self.initial_pose["y_init"] = random.uniform(initial_pose_areas_y[area_idx][0], initial_pose_areas_y[area_idx][1])
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = random.uniform(0.0, 2*math.pi)

        elif self.world_name == "corridor":
            self.initial_pose["x_init"] = 0.0
            self.initial_pose["y_init"] = 0.0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = random.uniform(0.0, 2*math.pi)

        elif self.world_name == "training_garden_dynamic_0":

            initial_pose_areas_x = []
            initial_pose_areas_x.extend(([2,2.5], [-2,-2.5]))

            initial_pose_areas_y = []
            initial_pose_areas_y.extend(([-0.5,0.5], [-0.5,0.5]))

            area_idx = random.randint(0, len(initial_pose_areas_x)-1)
            self.robot_init_area_id = area_idx

            self.initial_pose["x_init"] = random.uniform(initial_pose_areas_x[area_idx][0], initial_pose_areas_x[area_idx][1])
            self.initial_pose["y_init"] = random.uniform(initial_pose_areas_y[area_idx][0], initial_pose_areas_y[area_idx][1])
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = random.uniform(0.0, 2*math.pi)

        elif self.world_name == "training_garden_dynamic_1":

            initial_pose_areas_x = []
            initial_pose_areas_x.extend(([-2.0,-0.5], [4.5,5.5], [-5.5,-4.5], [3,5], [-5.5,-4.5]))

            initial_pose_areas_y = []
            initial_pose_areas_y.extend(([-4.0,0.0], [3.5,4.5], [4, 4.5], [-4,-1], [-4.5,-4]))

            area_idx = random.randint(0, len(initial_pose_areas_x)-1)
            self.robot_init_area_id = area_idx

            self.initial_pose["x_init"] = random.uniform(initial_pose_areas_x[area_idx][0], initial_pose_areas_x[area_idx][1])
            self.initial_pose["y_init"] = random.uniform(initial_pose_areas_y[area_idx][0], initial_pose_areas_y[area_idx][1])
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = random.uniform(0.0, 2*math.pi)
        
        elif self.world_name == "training_garden_dynamic_2":

            initial_pose_areas_x = []
            initial_pose_areas_x.extend(([4.5,5.5], [-5.5,-3.5], [3.0,5.0], [-5.5,-5.0], [-2.0,0.5]))

            initial_pose_areas_y = []
            initial_pose_areas_y.extend(([3.5,4.5], [3.0,4.5], [-4.5,-2], [-4.5,-2], [0.0,-0.5]))

            area_idx = random.randint(0, len(initial_pose_areas_x)-1)
            self.robot_init_area_id = area_idx

            self.initial_pose["x_init"] = random.uniform(initial_pose_areas_x[area_idx][0], initial_pose_areas_x[area_idx][1])
            self.initial_pose["y_init"] = random.uniform(initial_pose_areas_y[area_idx][0], initial_pose_areas_y[area_idx][1])
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = random.uniform(0.0, 2*math.pi)

        elif self.world_name == "testing_dwarl_zigzag_static":

            self.initial_pose["x_init"] = 5.0
            self.initial_pose["y_init"] = -8.0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = 0.0

        elif self.world_name == "testing_lvl_1":

            self.initial_pose["x_init"] = 4.0
            self.initial_pose["y_init"] = 3.0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = 0.0

        elif self.world_name == "testing_lvl_2":

            self.initial_pose["x_init"] = 4.0
            self.initial_pose["y_init"] = 3.0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = 0.0

        elif self.world_name == "testing_lvl_3":

            self.initial_pose["x_init"] = 4.0
            self.initial_pose["y_init"] = 3.0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = 0.0

        elif self.world_name == "testing_lvl_4":

            self.initial_pose["x_init"] = 4.0
            self.initial_pose["y_init"] = -4.0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = 0.0

        elif self.world_name == "testing_lvl_5":

            self.initial_pose["x_init"] = 4.0
            self.initial_pose["y_init"] = -4.0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = 0.0

        elif self.world_name == "testing_lvl_6":

            self.initial_pose["x_init"] = 4.0
            self.initial_pose["y_init"] = -4.0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = 0.0

        elif self.world_name == "testing_lvl_7":

            self.initial_pose["x_init"] = 1.0
            self.initial_pose["y_init"] = 3.0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = -160*math.pi/180

        elif self.world_name == "testing_lvl_8":

            self.initial_pose["x_init"] = 7.0
            self.initial_pose["y_init"] = -3.0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = 0.0

        elif self.world_name == "testing_lvl_9":

            self.initial_pose["x_init"] = 7.0
            self.initial_pose["y_init"] = -3.0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = 0.0

        elif self.world_name == "testing_lvl_10":

            self.initial_pose["x_init"] = 7.0
            self.initial_pose["y_init"] = -3.0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = 0.0
        
        elif self.world_name == "validation" or \
             self.world_name == "validation_overtaking":

            if self.world_name == "validation":
                self.validation_flag = True

            #self.reset_pedsim()
            
            self.initial_pose["x_init"] = -.25
            self.initial_pose["y_init"] = 0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = 0

            self.validation_ep_num += 1

            if self.validation_flag and self.validation_ep_num >= self.config.max_testing_episodes:

                self.next_world_name = "validation_passing"
                self.validation_ep_num = 0
        
        elif self.world_name == "validation_passing":

            #self.reset_pedsim()

            self.initial_pose["x_init"] = 0.0
            self.initial_pose["y_init"] = -4.0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = 0

            self.validation_ep_num += 1

            if self.validation_flag and self.validation_ep_num >= self.config.max_testing_episodes:

                self.next_world_name = "validation_crossing"
                self.validation_ep_num = 0

        elif self.world_name == "validation_crossing":
            
            #self.reset_pedsim()

            self.initial_pose["x_init"] = -4.0
            self.initial_pose["y_init"] = 0
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = 3.1415

            self.validation_ep_num += 1

            if self.validation_flag and self.validation_ep_num >= self.config.max_testing_episodes:

                self.next_world_name = "museum_static"
                self.validation_ep_num = 0

        elif self.world_name == "museum_static":

            self.initial_pose["x_init"] = 8
            self.initial_pose["y_init"] = 5
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = -.785

            self.validation_ep_num += 1

            if self.validation_flag and self.validation_ep_num >= self.config.max_testing_episodes:

                self.next_world_name = "museum_static_and_dynamic"
                self.validation_ep_num = 0

            else:
                self.validation_ep_num += 1

        elif self.world_name == "museum_static_and_dynamic":
            
            #self.reset_pedsim()

            self.initial_pose["x_init"] = 17
            self.initial_pose["y_init"] = 5
            self.initial_pose["z_init"] = 0.0
            robot0_init_yaw = -.785

        robot0_init_quat = Quaternion.from_euler(0, 0, robot0_init_yaw)
        self.initial_pose["x_rot_init"] = robot0_init_quat.x
        self.initial_pose["y_rot_init"] = robot0_init_quat.y
        self.initial_pose["z_rot_init"] = robot0_init_quat.z
        self.initial_pose["w_rot_init"] = robot0_init_quat.w

        #print("turtlebot3_tentabot_drl::get_init_pose -> Updated initial_pose x: " + str(self.initial_pose["x_init"]) + ", y: " + str(self.initial_pose["y_init"]))
        rospy.logdebug("turtlebot3_tentabot_drl::get_init_pose -> Updated initial_pose x: " + str(self.initial_pose["x_init"]) + ", y: " + str(self.initial_pose["y_init"]))

        if init_flag:
            super(TurtleBot3TentabotDRL, self).update_initial_pose(self.initial_pose)

        return self.initial_pose

    '''
    DESCRIPTION: TODO...Gets the goal location for each robot
    '''
    def get_goal_location(self):

        self.goal_pose = {}

        if self.world_name == "training_garden_static_0":

            goal_areas_x = []
            goal_areas_x.extend(([-0.5,0.5], [1.5,2.5], [-2.5,-1], [-2.5,0.5]))

            goal_areas_y = []
            goal_areas_y.extend(([-0.5,0.5], [0.5,1.5], [1,1.5], [-1.5,-1]))

            area_idx = random.randint(0, len(goal_areas_x)-1)
            while self.robot_init_area_id == area_idx:
                area_idx = random.randint(0, len(goal_areas_x)-1)

            self.goal_pose["x"] = random.uniform(goal_areas_x[area_idx][0], goal_areas_x[area_idx][1])
            self.goal_pose["y"] = random.uniform(goal_areas_y[area_idx][0], goal_areas_y[area_idx][1])
            self.goal_pose["z"] = 0.0

        elif self.world_name == "training_garden_static_1":

            goal_areas_x = []
            #goal_areas_x.extend(([-1.5,1.5], [4.5,5.5], [-5.5,-2], [1,5], [-5.5,-3.5]))
            goal_areas_x.extend(([-1.5,0.5], [1.0,5.5], [-5.5,-3], [4.0,5.5], [-5.5,-4.5]))

            goal_areas_y = []
            #goal_areas_y.extend(([-1,1], [1,4.5], [3.5, 4.5], [-4,-2], [-4.5,-2]))
            goal_areas_y.extend(([0,2], [4,4.5], [4.0,4.5], [-4.5,-3.0], [-4.5,-3.5]))

            area_idx = random.randint(0, len(goal_areas_x)-1)
            while self.robot_init_area_id == area_idx:
                area_idx = random.randint(0, len(goal_areas_x)-1)

            self.goal_pose["x"] = random.uniform(goal_areas_x[area_idx][0], goal_areas_x[area_idx][1])
            self.goal_pose["y"] = random.uniform(goal_areas_y[area_idx][0], goal_areas_y[area_idx][1])
            self.goal_pose["z"] = 0.0

        elif self.world_name == "corridor":

            self.goal_pose["x"] = 0.0
            self.goal_pose["y"] = 7.0
            self.goal_pose["z"] = 0.0

        elif self.world_name == "training_garden_dynamic_0":

            goal_areas_x = []
            goal_areas_x.extend(([2,2.5], [-2,-2.5]))

            goal_areas_y = []
            goal_areas_y.extend(([-0.5,0.5], [-0.5,0.5]))

            area_idx = random.randint(0, len(goal_areas_x)-1)
            while self.robot_init_area_id == area_idx:
                area_idx = random.randint(0, len(goal_areas_x)-1)

            self.goal_pose["x"] = random.uniform(goal_areas_x[area_idx][0], goal_areas_x[area_idx][1])
            self.goal_pose["y"] = random.uniform(goal_areas_y[area_idx][0], goal_areas_y[area_idx][1])
            self.goal_pose["z"] = 0.0

        elif self.world_name == "training_garden_dynamic_1":

            goal_areas_x = []
            goal_areas_x.extend(([-2.0,-0.5], [4.5,5.5], [-5.5,-4.5], [3,5], [-5.5,-4.5]))

            goal_areas_y = []
            goal_areas_y.extend(([-4.0,0.0], [3.5,4.5], [4, 4.5], [-4,-1], [-4.5,-4]))

            area_idx = random.randint(0, len(goal_areas_x)-1)
            while self.robot_init_area_id == area_idx:
                area_idx = random.randint(0, len(goal_areas_x)-1)

            self.goal_pose["x"] = random.uniform(goal_areas_x[area_idx][0], goal_areas_x[area_idx][1])
            self.goal_pose["y"] = random.uniform(goal_areas_y[area_idx][0], goal_areas_y[area_idx][1])
            self.goal_pose["z"] = 0.0

        elif self.world_name == "training_garden_dynamic_2":

            goal_areas_x = []
            goal_areas_x.extend(([4.5,5.5], [-5.5,-3.5], [3.0,5.0], [-5.5,-5.0], [-2.0,0.5]))

            goal_areas_y = []
            goal_areas_y.extend(([3.5,4.5], [3.0,4.5], [-4.5,-2], [-4.5,-2], [0.0,-0.5]))

            area_idx = random.randint(0, len(goal_areas_x)-1)
            while self.robot_init_area_id == area_idx:
                area_idx = random.randint(0, len(goal_areas_x)-1)

            self.goal_pose["x"] = random.uniform(goal_areas_x[area_idx][0], goal_areas_x[area_idx][1])
            self.goal_pose["y"] = random.uniform(goal_areas_y[area_idx][0], goal_areas_y[area_idx][1])
            self.goal_pose["z"] = 0.0

        elif self.world_name == "testing_dwarl_zigzag_static":

            self.goal_pose["x"] = -11.0
            self.goal_pose["y"] = 8.0
            self.goal_pose["z"] = 0.0

        elif self.world_name == "testing_lvl_1":

            self.goal_pose["x"] = -4.0
            self.goal_pose["y"] = -1.0
            self.goal_pose["z"] = 0.0

        elif self.world_name == "testing_lvl_2":

            self.goal_pose["x"] = -4.0
            self.goal_pose["y"] = -1.0
            self.goal_pose["z"] = 0.0

        elif self.world_name == "testing_lvl_3":

            self.goal_pose["x"] = -4.0
            self.goal_pose["y"] = -1.0
            self.goal_pose["z"] = 0.0

        elif self.world_name == "testing_lvl_4":

            self.goal_pose["x"] = -4.0
            self.goal_pose["y"] = 2.0
            self.goal_pose["z"] = 0.0

        elif self.world_name == "testing_lvl_5":

            self.goal_pose["x"] = -4.0
            self.goal_pose["y"] = 2.0
            self.goal_pose["z"] = 0.0

        elif self.world_name == "testing_lvl_6":

            self.goal_pose["x"] = -4.0
            self.goal_pose["y"] = 2.0
            self.goal_pose["z"] = 0.0

        elif self.world_name == "testing_lvl_7":

            self.goal_pose["x"] = -4.0
            self.goal_pose["y"] = 2.0
            self.goal_pose["z"] = 0.0

        elif self.world_name == "testing_lvl_8":

            self.goal_pose["x"] = -4.0
            self.goal_pose["y"] = 3.0
            self.goal_pose["z"] = 0.0

        elif self.world_name == "testing_lvl_9":

            self.goal_pose["x"] = -4.0
            self.goal_pose["y"] = 3.0
            self.goal_pose["z"] = 0.0

        elif self.world_name == "testing_lvl_10":

            self.goal_pose["x"] = -4.0
            self.goal_pose["y"] = 3.0
            self.goal_pose["z"] = 0.0

        elif self.world_name == "validation" or \
             self.world_name == "validation_overtaking":

            self.goal_pose["x"] = 5.5
            self.goal_pose["y"] = 0
            self.goal_pose["z"] = 0.0
        
        elif self.world_name == "validation_passing":

            self.goal_pose["x"] = 5.5
            self.goal_pose["y"] = -4
            self.goal_pose["z"] = 0.0
        
        elif self.world_name == "validation_crossing":

            self.goal_pose["x"] = -9.5
            self.goal_pose["y"] = 0.0
            self.goal_pose["z"] = 0.0
        
        elif self.world_name == "museum_static":

            self.goal_pose["x"] = 14
            self.goal_pose["y"] = -5
            self.goal_pose["z"] = 0.0

        elif self.world_name == "museum_static_and_dynamic":

            self.goal_pose["x"] = 24
            self.goal_pose["y"] = -6
            self.goal_pose["z"] = 0.0

        if  self.config.observation_space_type == "Tentabot_FC" or \
            self.config.observation_space_type == "Tentabot_1DCNN_FC" or \
            self.config.observation_space_type == "Tentabot_2DCNN_FC" or \
            self.config.observation_space_type == "Tentabot_2DCNN" or \
            self.config.observation_space_type == "Tentabot_WP_FC":
        
            self.client_update_goal()

        self.config.set_goal(self.goal_pose)
        self.publish_goal()
        self.init_distance2goal = self.get_initdistance2goal()

    '''
    DESCRIPTION: TODO...
    '''
    def calculate_euclidean_distance(self, p1, p2):

        return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2 + (p1["z"] - p2["z"])**2)

    '''
    DESCRIPTION: TODO...Get the initial distance to the goal
    '''
    def get_initdistance2goal(self):

        return math.sqrt((self.goal_pose["x"] - self.initial_pose["x_init"])**2 + (self.goal_pose["y"] - self.initial_pose["y_init"])**2)

    '''
    DESCRIPTION: TODO...Gets the distance to the goal
    '''
    def get_distance2goal(self):

        self.update_odom()
        return math.sqrt((self.goal_pose["x"] - self.odom_dict["x"])**2 + (self.goal_pose["y"] - self.odom_dict["y"])**2)

    '''
    DESCRIPTION: TODO...
    value.
    '''
    def check_collision(self, laser_scan):

        self.min_distance2obstacle = min(laser_scan.ranges)

        for scan_range in laser_scan.ranges:

            if (self.config.obs_min_range > scan_range > 0):

                if not self._episode_done:

                    self.total_collisions += 1
                    #rospy.logdebug("turtlebot3_tentabot_drl::check_collision -> Hit me baby one more time!")
                    print("turtlebot3_tentabot_drl::check_collision -> Hit me baby one more time!")

                self._episode_done = True
                return True
        
        return False

    '''
    DESCRIPTION: TODO...Discards all the laser readings that are not multiple in index of laser_downsampling_scale
    value.
    '''
    def filter_laser_scan(self, data):

        #data = self.get_laser_scan()

        data_size = len(data.ranges)
        filtered_laser_scan = []
        normalized_laser_scan = []

        if 0 < self.config.laser_size_downsampled < data_size:
            max_n_laser_ranges = self.config.laser_size_downsampled
        
        else:
            max_n_laser_ranges = data_size

        mod = self.config.laser_downsample_scale

        for i, item in enumerate(data.ranges):

            if (i % mod == 0):
                
                if item == float ('Inf') or np.isinf(item) or item > self.config.laser_range_max:

                    if len(filtered_laser_scan) < max_n_laser_ranges:
                        
                        #filtered_laser_scan.append(round(self.config.laser_range_max, self.config.mantissa_precision))
                        filtered_laser_scan.append(self.config.laser_range_max)
                        normalized_laser_scan.append(1.0)
                        
                        # NUA DEBUG:
                        #filtered_laser_scan.append(round(self.step_num, self.config.mantissa_precision))
                        #filtered_laser_scan.append(round(i, self.config.mantissa_precision))
                
                elif np.isnan(item) or item < self.config.laser_range_min:

                    if len(filtered_laser_scan) < max_n_laser_ranges:
                        
                        #filtered_laser_scan.append(round(self.config.laser_range_min, self.config.mantissa_precision))
                        filtered_laser_scan.append(self.config.laser_range_min)
                        #normalized_laser_scan.append(round(self.config.laser_range_min / self.config.laser_range_max, self.config.mantissa_precision))
                        normalized_laser_scan.append(self.config.laser_range_min / self.config.laser_range_max)

                        
                        # NUA DEBUG:
                        #filtered_laser_scan.append(round(self.step_num, self.config.mantissa_precision))
                        #filtered_laser_scan.append(round(i, self.config.mantissa_precision))
                
                else:

                    if len(filtered_laser_scan) < max_n_laser_ranges:
                        
                        #filtered_laser_scan.append(round(item, self.config.mantissa_precision))
                        filtered_laser_scan.append(item)
                        #normalized_laser_scan.append(round(item / self.config.laser_range_max, self.config.mantissa_precision))
                        normalized_laser_scan.append(item / self.config.laser_range_max)
                        
                        # NUA DEBUG:
                        #filtered_laser_scan.append(round(self.step_num, self.config.mantissa_precision))
                        #filtered_laser_scan.append(round(i, self.config.mantissa_precision))

        if  self.config.observation_space_type == "laser_FC" or \
            self.config.observation_space_type == "Tentabot_FC" or \
            self.config.observation_space_type == "Tentabot_WP_FC":

            self.filtered_laser_ranges = np.array(filtered_laser_scan).reshape(self.config.fc_obs_shape)
            self.normalized_laser_ranges = np.array(normalized_laser_scan).reshape(self.config.fc_obs_shape)
        
        else:
            self.filtered_laser_ranges = np.array(filtered_laser_scan).reshape(self.config.cnn_obs_shape)
            self.normalized_laser_ranges = np.array(normalized_laser_scan).reshape(self.config.cnn_obs_shape)

        self.publish_filtered_laser_scan()

    '''
    DESCRIPTION: TODO...
    '''
    def publish_filtered_laser_scan(self):
        
        filtered_laser_ranges = self.filtered_laser_ranges.reshape(self.config.fc_obs_shape)

        laser_scan_msg = LaserScan()
        
        laser_scan_msg.angle_min = self.config.laser_angle_min
        laser_scan_msg.angle_max = self.config.laser_angle_max
        laser_scan_msg.angle_increment = self.config.laser_angle_increment
        laser_scan_msg.time_increment = self.config.laser_time_increment
        laser_scan_msg.scan_time = self.config.laser_scan_time
        laser_scan_msg.range_min = self.config.laser_range_min
        laser_scan_msg.range_max = self.config.laser_range_max
        laser_scan_msg.ranges = tuple(filtered_laser_ranges)
        
        laser_scan_msg.header.frame_id = self.config.laser_frame_id
        laser_scan_msg.header.stamp = rospy.Time.now()

        self.filtered_laser_pub.publish(laser_scan_msg)

    '''
    DESCRIPTION: TODO...
    '''
    def update_obs_target(self):

        # Update the odometry data
        self.update_odom()

        # Update "distance to target" and "angle to target"
        translation_robot_wrt_world = tf.transformations.translation_matrix((self.odom_data.pose.pose.position.x, 
                                                                                self.odom_data.pose.pose.position.y, 
                                                                                self.odom_data.pose.pose.position.z))
        rotation_robot_wrt_world = tf.transformations.quaternion_matrix((self.odom_data.pose.pose.orientation.x, 
                                                                            self.odom_data.pose.pose.orientation.y, 
                                                                            self.odom_data.pose.pose.orientation.z, 
                                                                            self.odom_data.pose.pose.orientation.w))
        transform_robot_wrt_world = np.matmul(translation_robot_wrt_world, rotation_robot_wrt_world)
        transform_world_wrt_robot = tf.transformations.inverse_matrix(transform_robot_wrt_world)

        translation_goal_wrt_world = np.array([ [self.goal_pose["x"]], [self.goal_pose["y"]], [0.0], [1.0] ])

        translation_goal_wrt_robot = np.dot(transform_world_wrt_robot, translation_goal_wrt_world)
        current_angle2goal = math.atan2(translation_goal_wrt_robot[1], translation_goal_wrt_robot[0])

        current_distance2goal = self.get_distance2goal()
        self.obs_target = np.array([[current_distance2goal, current_angle2goal]]).reshape(self.config.fc_obs_shape)
        #self.obs_target = np.array([[self.step_num, self.step_num]]).reshape(self.config.fc_obs_shape)

    '''
    DESCRIPTION: TODO... Merge with update_obs_target2 which is implemented for Tentabot_WP
    '''
    def update_obs_target2(self):
        self.obs_wp = np.zeros(self.config.n_wp * 2)

        if self.client_move_base_get_plan():
            translation_robot_wrt_world = tf.transformations.translation_matrix((self.odom_data.pose.pose.position.x, 
                                                                    self.odom_data.pose.pose.position.y, 
                                                                    self.odom_data.pose.pose.position.z))
            rotation_robot_wrt_world = tf.transformations.quaternion_matrix((self.odom_data.pose.pose.orientation.x, 
                                                                                self.odom_data.pose.pose.orientation.y, 
                                                                                self.odom_data.pose.pose.orientation.z, 
                                                                                self.odom_data.pose.pose.orientation.w))
            transform_robot_wrt_world = np.matmul(translation_robot_wrt_world, rotation_robot_wrt_world)
            transform_world_wrt_robot = tf.transformations.inverse_matrix(transform_robot_wrt_world)

            wp_skip = int(self.config.look_ahead / self.config.wp_global_dist)

            # downsample global plan
            self.full_wp = self.move_base_global_plan[wp_skip::wp_skip]

            # add last waypoint (goal pos)
            if (len(self.move_base_global_plan) - 1)%wp_skip > 0:
                self.full_wp.append(self.move_base_global_plan[-1])

            # print("turtlebot3_tentabot_drl::init_obs_waypoints -> self.full_wp length: " + str(len(self.full_wp)))

            try:
                tmp_wp = self.full_wp[0]
                translation_wp_wrt_world = np.array([ [tmp_wp.pose.position.x], [tmp_wp.pose.position.y], [0.0], [1.0] ])
                translation_wp_wrt_robot = np.dot(transform_world_wrt_robot, translation_wp_wrt_world)

                self.obs_wp[i*2] = translation_wp_wrt_robot[0]
                self.obs_wp[i*2+1] = translation_wp_wrt_robot[1]
            except:
                pass

            # self.publish_debug_visu(self.move_base_global_plan)
            wp_obs = list(self.full_wp[0:self.config.n_wp])
            self.publish_wp_visu(self.full_wp,wp_obs)

        translation_goal_wrt_robot = np.dot(transform_world_wrt_robot, translation_goal_wrt_world)
        current_angle2goal = math.atan2(translation_goal_wrt_robot[1], translation_goal_wrt_robot[0])

        current_distance2goal = self.get_distance2goal()
        self.obs_target = np.array([[current_distance2goal, current_angle2goal]]).reshape(self.config.fc_obs_shape)
        #self.obs_target = np.array([[self.step_num, self.step_num]]).reshape(self.config.fc_obs_shape)

    '''
    DESCRIPTION: TODO...
    '''
    def init_obs_waypoints(self):

        self.obs_wp = np.zeros(self.config.n_wp * 2)

        if self.client_move_base_get_plan():
            translation_robot_wrt_world = tf.transformations.translation_matrix((self.odom_data.pose.pose.position.x, 
                                                                                    self.odom_data.pose.pose.position.y, 
                                                                                    self.odom_data.pose.pose.position.z))
            rotation_robot_wrt_world = tf.transformations.quaternion_matrix((self.odom_data.pose.pose.orientation.x, 
                                                                                self.odom_data.pose.pose.orientation.y, 
                                                                                self.odom_data.pose.pose.orientation.z, 
                                                                                self.odom_data.pose.pose.orientation.w))
            transform_robot_wrt_world = np.matmul(translation_robot_wrt_world, rotation_robot_wrt_world)
            transform_world_wrt_robot = tf.transformations.inverse_matrix(transform_robot_wrt_world)

            wp_skip = int(self.config.look_ahead / self.config.wp_global_dist)

            # downsample global plan
            self.full_wp = self.move_base_global_plan[wp_skip::wp_skip]

            # add last waypoint (goal pos)
            if (len(self.move_base_global_plan) - 1)%wp_skip > 0:
                self.full_wp.append(self.move_base_global_plan[-1])

            # print("turtlebot3_tentabot_drl::init_obs_waypoints -> self.full_wp length: " + str(len(self.full_wp)))

            for i in range(self.config.n_wp):
                try:
                    tmp_wp = self.full_wp[i]
                    translation_wp_wrt_world = np.array([ [tmp_wp.pose.position.x], [tmp_wp.pose.position.y], [0.0], [1.0] ])
                    translation_wp_wrt_robot = np.dot(transform_world_wrt_robot, translation_wp_wrt_world)

                    self.obs_wp[i*2] = translation_wp_wrt_robot[0]
                    self.obs_wp[i*2+1] = translation_wp_wrt_robot[1]
                except:
                    pass

            # self.publish_debug_visu(self.move_base_global_plan)
            wp_obs = list(self.full_wp[0:self.config.n_wp])
            self.publish_wp_visu(self.full_wp,wp_obs)
    
    '''
    DESCRIPTION: TODO...
    '''
    def update_obs_waypoints(self):

        self.update_odom()
        translation_robot_wrt_world = tf.transformations.translation_matrix((self.odom_data.pose.pose.position.x, 
                                                                self.odom_data.pose.pose.position.y, 
                                                                self.odom_data.pose.pose.position.z))
        rotation_robot_wrt_world = tf.transformations.quaternion_matrix((self.odom_data.pose.pose.orientation.x, 
                                                                            self.odom_data.pose.pose.orientation.y, 
                                                                            self.odom_data.pose.pose.orientation.z, 
                                                                            self.odom_data.pose.pose.orientation.w))
        transform_robot_wrt_world = np.matmul(translation_robot_wrt_world, rotation_robot_wrt_world)
        transform_world_wrt_robot = tf.transformations.inverse_matrix(transform_robot_wrt_world)


        trunc_list = False
        self.obs_wp = np.zeros(self.config.n_wp * 2)

        tmp_wp = self.full_wp[0]

        dist  = math.sqrt( (self.odom_data.pose.pose.position.x - tmp_wp.pose.position.x)**2 + (self.odom_data.pose.pose.position.y - tmp_wp.pose.position.y)**2 )

        for s in range(1, len(self.full_wp)):
            
            tmp_wp = self.full_wp[s]
            tmp_dist  = math.sqrt( (self.odom_data.pose.pose.position.x - tmp_wp.pose.position.x)**2 + (self.odom_data.pose.pose.position.y - tmp_wp.pose.position.y)**2 )

            # compare to previous on every point but last point
            if tmp_dist < dist and s < len(self.full_wp)-1:
                dist = tmp_dist
            else: # found closest waypoint
                start_idx = s-1
                # is closest wp reached in a certain radius?
                if dist < self.config.wp_reached_dist:
                    trunc_list = True

                # is closest waypoint already overtaken? Then take next one.
                if dist + self.config.look_ahead > tmp_dist:
                    start_idx = s

                # get all sequencing waypoints
                for i in range(start_idx, start_idx+self.config.n_wp):
                    try:
                        tmp_wp = self.full_wp[i]
                        translation_wp_wrt_world = np.array([ [tmp_wp.pose.position.x], [tmp_wp.pose.position.y], [0.0], [1.0] ])
                        translation_wp_wrt_robot = np.dot(transform_world_wrt_robot, translation_wp_wrt_world)

                        self.obs_wp[i*2] = translation_wp_wrt_robot[0]
                        self.obs_wp[i*2+1] = translation_wp_wrt_robot[1]
                    except:
                        pass

                break

        # print(self.obs_wp)
        # print(self.obs_wp[0], self.obs_wp[1] )
        wp_obs = list(self.full_wp[start_idx:start_idx+self.config.n_wp])
        
        if trunc_list:
            self.full_wp = self.full_wp[s-1:]

        self.publish_wp_visu(self.full_wp,wp_obs)

    '''
    DESCRIPTION: TODO...
    '''
    def init_observation_action_space(self):

        self.episode_oar_data = dict(obs=[], acts=[], infos=None, terminal=[], rews=[])

        if self.config.observation_space_type == "laser_FC":
            
            if self.config.laser_normalize_flag:
                
                obs_laser_low = np.full((1, self.config.laser_n_range), 0.0).reshape(self.config.fc_obs_shape)
                obs_laser_high = np.full((1, self.config.laser_n_range), 1.0).reshape(self.config.fc_obs_shape)
            
            else:
                obs_laser_low = np.full((1, self.config.laser_n_range), self.config.laser_range_min).reshape(self.config.fc_obs_shape)
                obs_laser_high = np.full((1, self.config.laser_n_range), self.config.laser_range_max).reshape(self.config.fc_obs_shape)

            obs_target_low = np.array([[0.0, -math.pi]]).reshape(self.config.fc_obs_shape)
            obs_target_high = np.array([[np.inf, math.pi]]).reshape(self.config.fc_obs_shape)

            obs_action_low = np.array([[self.config.min_lateral_speed, self.config.min_angular_speed]]).reshape(self.config.fc_obs_shape)
            obs_action_high = np.array([[self.config.max_lateral_speed, self.config.max_angular_speed]]).reshape(self.config.fc_obs_shape)

            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_laser_low shape: " + str(obs_laser_low.shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_target_low shape: " + str(obs_target_low.shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_action_low shape: " + str(obs_action_low.shape))

            self.obs_data = {   "laser": np.vstack([obs_laser_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "target": np.vstack([obs_target_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "action": np.vstack([obs_action_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data laser shape: " + str(self.obs_data["laser"].shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data action shape: " + str(self.obs_data["action"].shape))

            obs_stacked_laser_low = np.hstack([obs_laser_low] * self.config.n_obs_stack)
            obs_stacked_laser_high = np.hstack([obs_laser_high] * self.config.n_obs_stack)

            obs_space_low = np.concatenate((obs_stacked_laser_low, obs_target_low, obs_action_low), axis=0)
            obs_space_high = np.concatenate((obs_stacked_laser_high, obs_target_high, obs_action_high), axis=0)

            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_stacked_laser_low shape: " + str(obs_stacked_laser_low.shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_space_low shape: " + str(obs_space_low.shape))

            self.obs = obs_space_low
            self.observation_space = spaces.Box(obs_space_low, obs_space_high)
            self.action_space = spaces.Discrete(self.config.n_actions)

        elif self.config.observation_space_type == "Tentabot_FC":
            
            obs_occupancy_low = np.full((1, self.config.n_observations), 0.0).reshape(self.config.fc_obs_shape)
            obs_occupancy_high = np.full((1, self.config.n_observations), 1.0).reshape(self.config.fc_obs_shape)

            obs_target_low = np.array([[0.0, -math.pi]]).reshape(self.config.fc_obs_shape)
            obs_target_high = np.array([[np.inf, math.pi]]).reshape(self.config.fc_obs_shape)

            obs_action_low = np.array([[self.config.min_lateral_speed, self.config.min_angular_speed]]).reshape(self.config.fc_obs_shape)
            obs_action_high = np.array([[self.config.max_lateral_speed, self.config.max_angular_speed]]).reshape(self.config.fc_obs_shape)

            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_occupancy_low shape: " + str(obs_occupancy_low.shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_target_low shape: " + str(obs_target_low.shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_action_low shape: " + str(obs_action_low.shape))

            self.obs_data = {   "occupancy": np.vstack([obs_occupancy_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "target": np.vstack([obs_target_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "action": np.vstack([obs_action_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data occupancy shape: " + str(self.obs_data["occupancy"].shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data action shape: " + str(self.obs_data["action"].shape))

            obs_stacked_occupancy_low = np.hstack([obs_occupancy_low] * self.config.n_obs_stack)
            obs_stacked_occupancy_high = np.hstack([obs_occupancy_high] * self.config.n_obs_stack)

            obs_space_low = np.concatenate((obs_stacked_occupancy_low, obs_target_low, obs_action_low), axis=0)
            obs_space_high = np.concatenate((obs_stacked_occupancy_high, obs_target_high, obs_action_high), axis=0)

            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_stacked_occupancy_low shape: " + str(obs_stacked_occupancy_low.shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_space_low shape: " + str(obs_space_low.shape))

            self.obs = obs_space_low
            self.observation_space = spaces.Box(obs_space_low, obs_space_high)
            self.action_space = spaces.Discrete(self.config.n_actions)

        elif self.config.observation_space_type == "Tentabot_1DCNN_FC" or \
             self.config.observation_space_type == "Tentabot_2DCNN_FC":

            obs_occupancy_low = np.full((1, self.config.n_observations), 0.0).reshape(self.config.cnn_obs_shape)
            obs_occupancy_high = np.full((1, self.config.n_observations), 1.0).reshape(self.config.cnn_obs_shape)

            obs_target_low = np.array([[0.0, -math.pi]]).reshape(self.config.fc_obs_shape)
            obs_target_high = np.array([[np.inf, math.pi]]).reshape(self.config.fc_obs_shape)

            obs_action_low = np.array([[self.config.min_lateral_speed, self.config.min_angular_speed]]).reshape(self.config.fc_obs_shape)
            obs_action_high = np.array([[self.config.max_lateral_speed, self.config.max_angular_speed]]).reshape(self.config.fc_obs_shape)

            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_occupancy_low shape: " + str(obs_occupancy_low.shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_target_low shape: " + str(obs_target_low.shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_action_low shape: " + str(obs_action_low.shape))

            if self.config.cit_flag:
                
                self.obs_data = {   "occupancy": np.vstack([obs_occupancy_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                    "target": np.vstack([obs_target_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                    "action": np.vstack([obs_action_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

                obs_space_occupancy_low = np.vstack([obs_occupancy_low] * self.config.n_obs_stack)
                obs_space_occupancy_high = np.vstack([obs_occupancy_high] * self.config.n_obs_stack)       

            else:
                
                self.obs_data = {   "occupancy": np.hstack([obs_occupancy_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                    "target": np.vstack([obs_target_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                    "action": np.vstack([obs_action_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

                obs_space_occupancy_low = np.hstack([obs_occupancy_low] * self.config.n_obs_stack)
                obs_space_occupancy_high = np.hstack([obs_occupancy_high] * self.config.n_obs_stack)

            obs_space_target_action_low = np.concatenate((obs_target_low, obs_action_low), axis=0)
            obs_space_target_action_high = np.concatenate((obs_target_high, obs_action_high), axis=0)

            if self.config.observation_space_type == "Tentabot_2DCNN_FC":

                obs_space_occupancy_low = np.expand_dims(obs_space_occupancy_low, axis=0)
                obs_space_occupancy_high = np.expand_dims(obs_space_occupancy_high, axis=0)

            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data occupancy shape: " + str(self.obs_data["occupancy"].shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data action shape: " + str(self.obs_data["action"].shape))

            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_space_occupancy_low shape: " + str(obs_space_occupancy_low.shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_space_target_action_low shape: " + str(obs_space_target_action_low.shape))

            self.obs = {"occupancy": obs_space_occupancy_low, 
                        "target_action": obs_space_target_action_low}

            self.observation_space = spaces.Dict({  "occupancy": spaces.Box(obs_space_occupancy_low, obs_space_occupancy_high), 
                                                    "target_action": spaces.Box(obs_space_target_action_low, obs_space_target_action_high)})

            self.action_space = spaces.Discrete(self.config.n_actions)

        elif self.config.observation_space_type == "laser_1DCNN_FC":

            if self.config.laser_normalize_flag:
                
                obs_laser_low = np.full((1, self.config.laser_n_range), 0.0).reshape(self.config.cnn_obs_shape)
                obs_laser_high = np.full((1, self.config.laser_n_range), 1.0).reshape(self.config.cnn_obs_shape)
            
            else:
                obs_laser_low = np.full((1, self.config.laser_n_range), self.config.laser_range_min).reshape(self.config.cnn_obs_shape)
                obs_laser_high = np.full((1, self.config.laser_n_range), self.config.laser_range_max).reshape(self.config.cnn_obs_shape)

            obs_target_low = np.array([[0.0, -math.pi]]).reshape(self.config.fc_obs_shape)
            obs_target_high = np.array([[np.inf, math.pi]]).reshape(self.config.fc_obs_shape)

            action_space_low = np.array([[self.config.min_lateral_speed, self.config.min_angular_speed]]).reshape(self.config.fc_obs_shape)
            action_space_high = np.array([[self.config.max_lateral_speed, self.config.max_angular_speed]]).reshape(self.config.fc_obs_shape)

            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_laser_low shape: " + str(obs_laser_low.shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_target_low shape: " + str(obs_target_low.shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> action_space_low shape: " + str(action_space_low.shape))

            self.obs_data = {   "laser": np.vstack([obs_laser_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "target": np.vstack([obs_target_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "action": np.vstack([action_space_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}
            
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data laser shape: " + str(self.obs_data["laser"].shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data action shape: " + str(self.obs_data["action"].shape))

            obs_space_laser_low = np.vstack([obs_laser_low] * self.config.n_obs_stack)
            obs_space_laser_high = np.vstack([obs_laser_high] * self.config.n_obs_stack)

            obs_space_target_action_low = np.concatenate((obs_target_low, action_space_low.reshape(self.config.fc_obs_shape)), axis=0)
            obs_space_target_action_high = np.concatenate((obs_target_high, action_space_high.reshape(self.config.fc_obs_shape)), axis=0)

            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_space_laser_low shape: " + str(obs_space_laser_low.shape))
            #print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_space_target_action_low shape: " + str(obs_space_target_action_low.shape))

            self.obs = {"laser": obs_space_laser_low, 
                        "target_action": obs_space_target_action_low}

            self.observation_space = spaces.Dict({  "laser": spaces.Box(obs_space_laser_low, obs_space_laser_high), 
                                                    "target_action": spaces.Box(obs_space_target_action_low, obs_space_target_action_high)})
            
            self.action_space = spaces.Discrete(self.config.n_actions)
            #self.action_space = spaces.Box(action_space_low, action_space_high)
        
        elif self.config.observation_space_type == "laser_WP_1DCNN_FC":

            if self.config.laser_normalize_flag:
                
                obs_laser_low = np.full((1, self.config.laser_n_range), 0.0).reshape(self.config.cnn_obs_shape)
                obs_laser_high = np.full((1, self.config.laser_n_range), 1.0).reshape(self.config.cnn_obs_shape)
            
            else:
                obs_laser_low = np.full((1, self.config.laser_n_range), self.config.laser_range_min).reshape(self.config.cnn_obs_shape)
                obs_laser_high = np.full((1, self.config.laser_n_range), self.config.laser_range_max).reshape(self.config.cnn_obs_shape)

            obs_space_waypoints_low = np.full((1, 2*self.config.n_wp), -np.inf).reshape(self.config.fc_obs_shape)
            obs_space_waypoints_high = np.full((1, 2*self.config.n_wp), np.inf).reshape(self.config.fc_obs_shape)

            obs_action_low = np.array([[self.config.min_lateral_speed, self.config.min_angular_speed]]).reshape(self.config.fc_obs_shape)
            obs_action_high = np.array([[self.config.max_lateral_speed, self.config.max_angular_speed]]).reshape(self.config.fc_obs_shape)

            # print("turtlebot3_tentabot_rl::init_observation_action_space -> obs_laser_low shape: " + str(obs_laser_low.shape))
            # print("turtlebot3_tentabot_rl::init_observation_action_space -> obs_space_waypoints_low shape: " + str(obs_space_waypoints_low.shape))
            # print("turtlebot3_tentabot_rl::init_observation_action_space -> obs_action_low shape: " + str(obs_action_low.shape))

            self.obs_data = {   "laser": np.vstack([obs_laser_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "waypoints": np.vstack([obs_space_waypoints_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "action": np.vstack([obs_action_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}
            
            # print("turtlebot3_tentabot_rl::init_observation_action_space -> obs_data laser shape: " + str(self.obs_data["laser"].shape))
            # print("turtlebot3_tentabot_rl::init_observation_action_space -> obs_data waypoints shape: " + str(self.obs_data["waypoints"].shape))
            # print("turtlebot3_tentabot_rl::init_observation_action_space -> obs_data action shape: " + str(self.obs_data["action"].shape))

            obs_space_laser_low = np.vstack([obs_laser_low] * self.config.n_obs_stack)
            obs_space_laser_high = np.vstack([obs_laser_high] * self.config.n_obs_stack)

            obs_space_wp_action_low = np.concatenate((obs_space_waypoints_low, obs_action_low.reshape(self.config.fc_obs_shape)), axis=0)
            obs_space_wp_action_high = np.concatenate((obs_space_waypoints_high, obs_action_high.reshape(self.config.fc_obs_shape)), axis=0)

            # print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_space_laser_low shape: " + str(obs_space_laser_low.shape))

            self.obs = {"laser": obs_space_laser_low, 
                        "waypoints_action ": obs_space_wp_action_low}

            self.observation_space = spaces.Dict({  "laser": spaces.Box(obs_space_laser_low, obs_space_laser_high), 
                                                    "waypoints_action": spaces.Box(obs_space_wp_action_low, obs_space_wp_action_high)})
            
            self.action_space = spaces.Discrete(self.config.n_actions)

        elif self.config.observation_space_type == "Tentabot_WP_FC":
            
            obs_occupancy_low = np.full((1, self.config.n_observations), 0.0).reshape(self.config.fc_obs_shape)
            obs_occupancy_high = np.full((1, self.config.n_observations), 1.0).reshape(self.config.fc_obs_shape)

            obs_space_waypoints_low = np.full((1, 2*self.config.n_wp), -np.inf).reshape(self.config.fc_obs_shape)
            obs_space_waypoints_high = np.full((1, 2*self.config.n_wp), np.inf).reshape(self.config.fc_obs_shape)

            obs_action_low = np.array([[self.config.min_lateral_speed, self.config.min_angular_speed]]).reshape(self.config.fc_obs_shape)
            obs_action_high = np.array([[self.config.max_lateral_speed, self.config.max_angular_speed]]).reshape(self.config.fc_obs_shape)

            # print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_occupancy_low shape: " + str(obs_occupancy_low.shape))
            # print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_space_waypoints_low shape: " + str(obs_space_waypoints_low.shape))
            # print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_action_low shape: " + str(obs_action_low.shape))

            self.obs_data = {   "occupancy": np.vstack([obs_occupancy_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "waypoints": np.vstack([obs_space_waypoints_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "action": np.vstack([obs_action_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

            # print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data occupancy shape: " + str(self.obs_data["occupancy"].shape))
            # print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data waypoints shape: " + str(self.obs_data["waypoints"].shape))
            # print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data action shape: " + str(self.obs_data["action"].shape))

            obs_stacked_occupancy_low = np.hstack([obs_occupancy_low] * self.config.n_obs_stack)
            obs_stacked_occupancy_high = np.hstack([obs_occupancy_high] * self.config.n_obs_stack)

            obs_space_low = np.concatenate((obs_stacked_occupancy_low, obs_space_waypoints_low, obs_action_low), axis=0)
            obs_space_high = np.concatenate((obs_stacked_occupancy_high, obs_space_waypoints_high, obs_action_high), axis=0)

            # print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_stacked_occupancy_low shape: " + str(obs_stacked_occupancy_low.shape))
            # print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_space_low shape: " + str(obs_space_low.shape))

            self.obs = obs_space_low
            self.observation_space = spaces.Box(obs_space_low, obs_space_high)
            self.action_space = spaces.Discrete(self.config.n_actions)

        elif self.config.observation_space_type == "laser_image_2DCNN_FC" or \
             self.config.observation_space_type == "laser_rings_2DCNN_FC":

            obs_laser_image_low = np.full((1, self.config.laser_image_width), 0.0)
            obs_laser_image_low = np.vstack([obs_laser_image_low] * self.config.laser_image_height)
            obs_laser_image_low = np.expand_dims(obs_laser_image_low, axis=0)

            obs_laser_image_high = np.full((1, self.config.laser_image_width), 1.0)
            obs_laser_image_high = np.vstack([obs_laser_image_high] * self.config.laser_image_height)
            obs_laser_image_high = np.expand_dims(obs_laser_image_high, axis=0)

            obs_target_low = np.array([[0.0, -math.pi]]).reshape(self.config.fc_obs_shape)
            obs_target_high = np.array([[np.inf, math.pi]]).reshape(self.config.fc_obs_shape)

            obs_action_low = np.array([[self.config.min_lateral_speed, self.config.min_angular_speed]]).reshape(self.config.fc_obs_shape)
            obs_action_high = np.array([[self.config.max_lateral_speed, self.config.max_angular_speed]]).reshape(self.config.fc_obs_shape)
            
            self.obs_data = {   "laser_image": np.vstack([obs_laser_image_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "target": np.vstack([obs_target_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "action": np.vstack([obs_action_low] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

            obs_space_laser_image_low = np.vstack([obs_laser_image_low] * self.config.n_obs_stack)
            obs_space_laser_image_high = np.vstack([obs_laser_image_high] * self.config.n_obs_stack)

            obs_space_target_action_low = np.concatenate((obs_target_low, obs_action_low), axis=0)
            obs_space_target_action_high = np.concatenate((obs_target_high, obs_action_high), axis=0)

            self.obs = {"laser_image": obs_space_laser_image_low, 
                        "target_action": obs_space_target_action_low}

            self.observation_space = spaces.Dict({  "laser_image": spaces.Box(obs_space_laser_image_low, obs_space_laser_image_high), 
                                                    "target_action": spaces.Box(obs_space_target_action_low, obs_space_target_action_high)})

            self.action_space = spaces.Discrete(self.config.n_actions)

            '''
            print("---------------------")
            print("turtlebot3_tentabot_drl::init_observation_action_space -> laser_image_width: " + str(self.config.laser_image_width))
            print("turtlebot3_tentabot_drl::init_observation_action_space -> laser_image_height: " + str(self.config.laser_image_height))
            print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data laser_image shape: " + str(self.obs_data["laser_image"].shape))
            print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data target shape: " + str(self.obs_data["target"].shape))
            print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_data action shape: " + str(self.obs_data["action"].shape))

            print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_space_laser_image_low shape: " + str(obs_space_laser_image_low.shape))
            print("turtlebot3_tentabot_drl::init_observation_action_space -> obs_space_target_action_low shape: " + str(obs_space_target_action_low.shape))
            print("---------------------")
            '''

        #print("turtlebot3_tentabot_drl::init_observation_action_space -> observation_space: " + str(self.observation_space))
        #print("turtlebot3_tentabot_drl::init_observation_action_space -> action_space: " + str(self.action_space))

    '''
    DESCRIPTION: TODO...
    '''
    def reinit_observation(self):

        self.episode_oar_data = dict(obs=[], acts=[], infos=None, terminal=[], rews=[])

        if self.config.observation_space_type == "laser_FC":

            # Update laser scan
            #self.filter_laser_scan()

            if self.config.laser_normalize_flag:
                obs_laser = self.normalized_laser_ranges
            
            else:
                obs_laser = self.filtered_laser_ranges

            # Update target observation
            self.update_obs_target()

            # Stack observation data
            self.obs_data = {   "laser": np.vstack([obs_laser] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "target": np.vstack([self.obs_target] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "action": np.vstack([self.previous_action] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

            #print("turtlebot3_tentabot_drl::reinit_observation -> filtered_laser_ranges shape: " + str(self.filtered_laser_ranges.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_target shape: " + str(self.obs_target.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> previous_action shape: " + str(self.previous_action.shape))

            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data laser shape: " + str(self.obs_data["laser"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))

            # Initialize observation
            obs_stacked_laser = np.hstack([obs_laser] * self.config.n_obs_stack)

            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_stacked_laser shape: " + str(obs_stacked_laser.shape))

            self.obs = np.concatenate((obs_stacked_laser, self.obs_target, self.previous_action), axis=0)

            #print("turtlebot3_tentabot_drl::reinit_observation -> obs: " + str(self.obs.shape))

        elif self.config.observation_space_type == "Tentabot_FC":

            # Update tentabot observation
            success_rl_step = self.client_rl_step(1)
            if not success_rl_step:
                rospy.logerr("turtlebot3_tentabot_drl::reinit_observation -> OBSERVATION FAILURE!")

            # Update target observation
            self.update_obs_target()

            # Stack observation data
            self.obs_data = {   "occupancy": np.vstack([self.occupancy_set] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "target": np.vstack([self.obs_target] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "action": np.vstack([self.previous_action] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

            #print("turtlebot3_tentabot_drl::reinit_observation -> occupancy_set shape: " + str(self.occupancy_set.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_target shape: " + str(self.obs_target.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> previous_action shape: " + str(self.previous_action.shape))

            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data occupancy shape: " + str(self.obs_data["occupancy"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))

            # Initialize observation
            obs_stacked_occupancy = np.hstack([self.occupancy_set] * self.config.n_obs_stack)

            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_stacked_occupancy shape: " + str(obs_stacked_occupancy.shape))

            self.obs = np.concatenate((obs_stacked_occupancy, self.obs_target, self.previous_action), axis=0)

        elif self.config.observation_space_type == "Tentabot_1DCNN_FC" or \
             self.config.observation_space_type == "Tentabot_2DCNN_FC":

            # Update tentabot observation
            success_rl_step = self.client_rl_step(1)
            if not success_rl_step:
                rospy.logerr("turtlebot3_tentabot_drl::reinit_observation -> OBSERVATION FAILURE!")

            # Update target observation
            self.update_obs_target()

            if self.config.cit_flag:
                
                # Stack observation data
                self.obs_data = {   "occupancy": np.vstack([self.occupancy_set] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                    "target": np.vstack([self.obs_target] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                    "action": np.vstack([self.previous_action] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

                # Initialize observation                    
                obs_space_occupancy = np.vstack([self.occupancy_set] * self.config.n_obs_stack)

            else:

                # Stack observation data
                self.obs_data = {   "occupancy": np.hstack([self.occupancy_set] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                    "target": np.vstack([self.obs_target] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                    "action": np.vstack([self.previous_action] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

                # Initialize observation 
                obs_space_occupancy = np.hstack([self.occupancy_set] * self.config.n_obs_stack)

            obs_space_target_action = np.concatenate((self.obs_target, self.previous_action), axis=0)

            #print("turtlebot3_tentabot_drl::reinit_observation -> occupancy_set shape: " + str(self.occupancy_set.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_target shape: " + str(self.obs_target.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> previous_action shape: " + str(self.previous_action.shape))

            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data occupancy shape: " + str(self.obs_data["occupancy"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))

            if self.config.observation_space_type == "Tentabot_2DCNN_FC":

                obs_space_occupancy = np.expand_dims(obs_space_occupancy, axis=0)
                obs_space_target_action = np.expand_dims(obs_space_target_action, axis=0)

            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_space_occupancy shape: " + str(obs_space_occupancy.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_space_target_action shape: " + str(obs_space_target_action.shape))

            self.obs = {"occupancy": obs_space_occupancy,
                        "target_action": obs_space_target_action}

        elif self.config.observation_space_type == "laser_1DCNN_FC":

            # Update laser scan
            #self.filter_laser_scan()

            if self.config.laser_normalize_flag:
                obs_laser = self.normalized_laser_ranges
            
            else:
                obs_laser = self.filtered_laser_ranges

            # Update target observation
            self.update_obs_target()

            # Stack observation data
            self.obs_data = {   "laser": np.vstack([obs_laser] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "target": np.vstack([self.obs_target] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "action": np.vstack([self.previous_action] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

            #print("turtlebot3_tentabot_drl::reinit_observation -> laser shape: " + str(self.filtered_laser_ranges.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_target shape: " + str(self.obs_target.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> previous_action shape: " + str(self.previous_action.shape))

            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data laser shape: " + str(self.obs_data["laser"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))

            # Initialize observation  
            obs_space_laser = np.vstack([obs_laser] * self.config.n_obs_stack)
            obs_space_target_action = np.concatenate((self.obs_target, self.previous_action), axis=0)

            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_space_laser shape: " + str(obs_space_laser.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_space_target_action shape: " + str(obs_space_target_action.shape))
            
            self.obs = {"laser": obs_space_laser,
                        "target_action": obs_space_target_action}

        elif self.config.observation_space_type == "laser_WP_1DCNN_FC":
            
            # Update laser scan
            #self.filter_laser_scan()

            if self.config.laser_normalize_flag:
                obs_laser = self.normalized_laser_ranges
            
            else:
                obs_laser = self.filtered_laser_ranges

            # Update waypoints
            self.init_obs_waypoints()

            # Stack observation data
            self.obs_data = {   "laser": np.vstack([obs_laser] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "waypoints": np.vstack([self.obs_wp] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "action": np.vstack([self.previous_action] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

            #print("turtlebot3_tentabot_drl::reinit_observation -> laser shape: " + str(self.filtered_laser_ranges.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> waypoints shape: " + str(self.obs_wp.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> previous_action shape: " + str(self.previous_action.shape))

            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data laser shape: " + str(self.obs_data["laser"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data waypoints shape: " + str(self.obs_data["waypoints"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))

            # Initialize the observation
            obs_space_laser = np.vstack([obs_laser] * self.config.n_obs_stack)
            obs_space_wp_action = np.concatenate((self.obs_wp, self.previous_action), axis=0)
            
            self.obs = {"laser": obs_space_laser,
                        "waypoints_action": obs_space_wp_action}
        
        elif self.config.observation_space_type == "Tentabot_WP_FC":
            
            # Update waypoints
            self.init_obs_waypoints()

            # Update tentabot observation
            success_rl_step = self.client_rl_step(1)
            if not success_rl_step:
                rospy.logerr("turtlebot3_tentabot_drl::reinit_observation -> OBSERVATION FAILURE!")

            # Stack observation data
            self.obs_data = {   "occupancy": np.vstack([self.occupancy_set] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "waypoints": np.vstack([self.obs_wp] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "action": np.vstack([self.previous_action] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

            #print("turtlebot3_tentabot_drl::reinit_observation -> occupancy_set shape: " + str(self.occupancy_set.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> waypoints shape: " + str(self.obs_wp.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> previous_action shape: " + str(self.previous_action.shape))

            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data occupancy shape: " + str(self.obs_data["occupancy"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data waypoints shape: " + str(self.obs_data["waypoints"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))

            # Initialize observation
            obs_stacked_occupancy = np.hstack([self.occupancy_set] * self.config.n_obs_stack)

            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_stacked_occupancy shape: " + str(obs_stacked_occupancy.shape))

            self.obs = np.concatenate((obs_stacked_occupancy, self.obs_wp, self.previous_action), axis=0)

        elif self.config.observation_space_type == "laser_image_2DCNN_FC" or \
             self.config.observation_space_type == "laser_rings_2DCNN_FC":

            # Update target observation
            self.update_obs_target()

            obs_laser_image = self.laser_image
            obs_laser_image = np.expand_dims(obs_laser_image, axis=0)

            obs_target = self.obs_target

            # Stack observation data
            self.obs_data = {   "laser_image": np.vstack([obs_laser_image] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "target": np.vstack([obs_target] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)),
                                "action": np.vstack([self.previous_action] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))}

            # Initialize observation                    
            obs_space_laser_image = np.vstack([obs_laser_image] * self.config.n_obs_stack)
            obs_space_target_action = np.concatenate((obs_target, self.previous_action), axis=0)

            if self.config.observation_space_type == "Tentabot_2DCNN_FC":

                obs_space_laser_image = np.expand_dims(obs_space_laser_image, axis=0)
                obs_space_target_action = np.expand_dims(obs_space_target_action, axis=0)

            '''
            print("---------------------")
            print("turtlebot3_tentabot_drl::reinit_observation -> obs_laser_image shape: " + str(obs_laser_image.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_target shape: " + str(self.obs_target.shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> previous_action shape: " + str(self.previous_action.shape))

            print("turtlebot3_tentabot_drl::reinit_observation -> obs_data laser_image shape: " + str(self.obs_data["laser_image"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::reinit_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))
            print("turtlebot3_tentabot_drl::reinit_observation -> obs_space_laser_image shape: " + str(obs_space_laser_image.shape))
            print("turtlebot3_tentabot_drl::reinit_observation -> obs_space_target_action shape: " + str(obs_space_target_action.shape))
            print("---------------------")
            '''

            '''
            imi = (self.laser_image * 255).astype(np.uint8)
            im = Image.fromarray(imi)
            im = im.convert("L")
            im.save(self.data_folder_path + "laser_image.jpeg")
            np.savetxt(self.data_folder_path + "laser_image.txt", self.laser_image)
            '''

            self.obs = {"laser_image": obs_space_laser_image,
                        "target_action": obs_space_target_action}
    
    '''
    DESCRIPTION: TODO...
    '''
    def update_observation(self):

        if self.config.observation_space_type == "laser_FC":

            # Update laser scan
            #self.filter_laser_scan()

            if self.config.laser_normalize_flag:
                obs_laser = self.normalized_laser_ranges
            
            else:
                obs_laser = self.filtered_laser_ranges

            # Update target observation
            self.update_obs_target()

            # Update observation data
            self.obs_data["laser"] = np.vstack((self.obs_data["laser"], obs_laser))
            self.obs_data["laser"] = np.delete(self.obs_data["laser"], np.s_[0], axis=0)

            self.obs_data["target"] = np.vstack((self.obs_data["target"], self.obs_target))
            self.obs_data["target"] = np.delete(self.obs_data["target"], np.s_[0], axis=0)

            self.obs_data["action"] = np.vstack((self.obs_data["action"], self.previous_action))
            self.obs_data["action"] = np.delete(self.obs_data["action"], np.s_[0], axis=0)

            #print("turtlebot3_tentabot_drl::update_observation -> obs_data laser shape: " + str(self.obs_data["laser"].shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))

            # Update observation
            obs_stacked_laser = self.obs_data["laser"][-1,:].reshape(self.config.fc_obs_shape)

            #print("turtlebot3_tentabot_drl::update_observation -> obs_stacked_laser shape: " + str(obs_stacked_laser.shape))

            if self.config.n_obs_stack > 1:
                latest_index = (self.config.n_obs_stack * self.config.n_skip_obs_stack) - 1
                j = 0
                for i in range(latest_index-1, -1, -1):
                    j += 1
                    if j % self.config.n_skip_obs_stack == 0:
                        obs_stacked_laser = np.hstack((self.obs_data["laser"][i,:], obs_stacked_laser))

            #print("turtlebot3_tentabot_drl::update_observation -> obs_stacked_laser shape: " + str(obs_stacked_laser.shape))

            self.obs = np.concatenate((obs_stacked_laser, self.obs_target, self.previous_action), axis=0)

            #print("turtlebot3_tentabot_drl::update_observation -> obs: " + str(self.obs.shape))

        elif self.config.observation_space_type == "laser_1DCNN_FC":

            # Update laser scan
            #self.filter_laser_scan()

            if self.config.laser_normalize_flag:
                obs_laser = self.normalized_laser_ranges
            
            else:
                obs_laser = self.filtered_laser_ranges

            # Update target observation
            self.update_obs_target()

            # Update observation data
            self.obs_data["laser"] = np.vstack((self.obs_data["laser"], obs_laser))
            self.obs_data["laser"] = np.delete(self.obs_data["laser"], np.s_[0], axis=0)

            self.obs_data["target"] = np.vstack((self.obs_data["target"], self.obs_target))
            self.obs_data["target"] = np.delete(self.obs_data["target"], np.s_[0], axis=0)

            self.obs_data["action"] = np.vstack((self.obs_data["action"], self.previous_action))
            self.obs_data["action"] = np.delete(self.obs_data["action"], np.s_[0], axis=0)

            #print("turtlebot3_tentabot_drl::update_observation -> obs_data laser shape: " + str(self.obs_data["laser"].shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))

            # Update observation
            obs_space_laser = self.obs_data["laser"][-1,:].reshape(self.config.cnn_obs_shape)

            if self.config.n_obs_stack > 1:
                if(self.config.n_skip_obs_stack > 1):
                    latest_index = (self.config.n_obs_stack * self.config.n_skip_obs_stack) - 1
                    j = 0
                    for i in range(latest_index-1, -1, -1):
                        j += 1
                        if j % self.config.n_skip_obs_stack == 0:
                            obs_space_laser = np.vstack((self.obs_data["laser"][i,:].reshape(self.config.cnn_obs_shape), obs_space_laser))
                
                else:
                    obs_space_laser = self.obs_data["laser"]

            obs_space_target_action = np.concatenate((self.obs_target, self.previous_action), axis=0)

            #print("turtlebot3_tentabot_drl::update_observation -> obs_space_laser: " + str(obs_space_laser.shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_space_target_action: " + str(obs_space_target_action.shape))

            self.obs["laser"] = obs_space_laser
            self.obs["target_action"] = obs_space_target_action

        elif self.config.observation_space_type == "Tentabot_FC":
            
            # Update tentabot observation
            success_rl_step = self.client_rl_step(1)
            if not success_rl_step:
                rospy.logerr("turtlebot3_tentabot_drl::update_observation -> OBSERVATION FAILURE!")

            # Update target observation
            self.update_obs_target()

            # Update observation data
            self.obs_data["occupancy"] = np.vstack((self.obs_data["occupancy"], self.occupancy_set))
            self.obs_data["occupancy"] = np.delete(self.obs_data["occupancy"], np.s_[0], axis=0)

            self.obs_data["target"] = np.vstack((self.obs_data["target"], self.obs_target))
            self.obs_data["target"] = np.delete(self.obs_data["target"], np.s_[0], axis=0)

            self.obs_data["action"] = np.vstack((self.obs_data["action"], self.previous_action))
            self.obs_data["action"] = np.delete(self.obs_data["action"], np.s_[0], axis=0)

            #print("turtlebot3_tentabot_drl::update_observation -> obs_data occupancy shape: " + str(self.obs_data["occupancy"].shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))

            # Update observation
            obs_stacked_occupancy = self.obs_data["occupancy"][-1,:].reshape(self.config.fc_obs_shape)

            if self.config.n_obs_stack > 1:
                latest_index = (self.config.n_obs_stack * self.config.n_skip_obs_stack) - 1
                j = 0
                for i in range(latest_index-1, -1, -1):
                    j += 1
                    if j % self.config.n_skip_obs_stack == 0:
                        obs_stacked_occupancy = np.hstack((self.obs_data["occupancy"][i,:], obs_stacked_occupancy))

            #print("turtlebot3_tentabot_drl::update_observation -> obs_stacked_occupancy shape: " + str(obs_stacked_occupancy.shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_target shape: " + str(self.obs_target.shape))
            #print("turtlebot3_tentabot_drl::update_observation -> previous_action shape: " + str(self.previous_action.shape))

            self.obs = np.concatenate((obs_stacked_occupancy, self.obs_target, self.previous_action), axis=0)

            #print("turtlebot3_tentabot_drl::update_observation -> obs: " + str(self.obs.shape))

        elif self.config.observation_space_type == "Tentabot_1DCNN_FC" or \
             self.config.observation_space_type == "Tentabot_2DCNN_FC":
            
            # Update tentabot observation
            success_rl_step = self.client_rl_step(1)
            if not success_rl_step:
                rospy.logerr("turtlebot3_tentabot_drl::update_observation -> OBSERVATION FAILURE!")

            # Update target observation
            self.update_obs_target()

            if self.config.cit_flag:

                # Update observation data
                self.obs_data["occupancy"] = np.vstack((self.obs_data["occupancy"], self.occupancy_set))
                self.obs_data["occupancy"] = np.delete(self.obs_data["occupancy"], np.s_[0], axis=0)

                # Update observation                    
                obs_space_occupancy = self.obs_data["occupancy"][-1,:].reshape(self.config.cnn_obs_shape)

                if self.config.n_obs_stack > 1:
                    if(self.config.n_skip_obs_stack > 1):
                        latest_index = (self.config.n_obs_stack * self.config.n_skip_obs_stack) - 1
                        j = 0
                        for i in range(latest_index-1, -1, -1):
                            j += 1
                            if j % self.config.n_skip_obs_stack == 0:
                                obs_space_occupancy = np.vstack((self.obs_data["occupancy"][i,:].reshape(self.config.cnn_obs_shape), obs_space_occupancy))
                    
                    else:
                        obs_space_occupancy = self.obs_data["occupancy"]

            else:

                # Update observation data
                self.obs_data["occupancy"] = np.hstack((self.obs_data["occupancy"], self.occupancy_set))
                self.obs_data["occupancy"] = np.delete(self.obs_data["occupancy"], np.s_[0], axis=1)

                # Update observation                    
                obs_space_occupancy = self.obs_data["occupancy"][:,-1].reshape(self.config.cnn_obs_shape)

                #print("turtlebot3_tentabot_drl::update_observation -> obs_space_occupancy: " + str(obs_space_occupancy.shape))

                if self.config.n_obs_stack > 1:
                    if(self.config.n_skip_obs_stack > 1):
                        latest_index = (self.config.n_obs_stack * self.config.n_skip_obs_stack) - 1
                        j = 0
                        for i in range(latest_index-1, -1, -1):
                            j += 1
                            if j % self.config.n_skip_obs_stack == 0:
                                obs_space_occupancy = np.hstack((self.obs_data["occupancy"][:,i].reshape(self.config.cnn_obs_shape), obs_space_occupancy))
                    
                    else:
                        obs_space_occupancy = self.obs_data["occupancy"]

            #print("turtlebot3_tentabot_drl::update_observation -> obs_space_occupancy: " + str(obs_space_occupancy.shape))

            self.obs_data["target"] = np.vstack((self.obs_data["target"], self.obs_target))
            self.obs_data["target"] = np.delete(self.obs_data["target"], np.s_[0], axis=0)

            self.obs_data["action"] = np.vstack((self.obs_data["action"], self.previous_action))
            self.obs_data["action"] = np.delete(self.obs_data["action"], np.s_[0], axis=0)

            obs_space_target_action = np.concatenate((self.obs_target, self.previous_action), axis=0)

            #print("turtlebot3_tentabot_drl::update_observation -> obs_data occupancy shape: " + str(self.obs_data["occupancy"].shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))

            if self.config.observation_space_type == "Tentabot_2DCNN_FC":

                obs_space_occupancy = np.expand_dims(obs_space_occupancy, axis=0)
                obs_space_target_action = np.expand_dims(obs_space_target_action, axis=0)

            #print("**************** " + str(self.step_num))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_space_occupancy: ")
            #print(obs_space_occupancy[0, 65:75])
            #print("turtlebot3_tentabot_drl::update_observation -> obs_target dist: " + str(self.obs_target[0,0]))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_target angle: " + str(self.obs_target[0,1] * 180 / math.pi))
            #print("turtlebot3_tentabot_drl::update_observation -> previous_action: " + str(self.previous_action))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_space_occupancy: " + str(type(obs_space_occupancy)))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_space_occupancy: " + str(obs_space_occupancy.shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_space_target_action: " + str(obs_space_target_action.shape))
            #print("****************")

            #print("turtlebot3_tentabot_drl::update_observation -> obs_space_occupancy: " + str(obs_space_occupancy.shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_space_target_action: " + str(obs_space_target_action.shape))

            self.obs["occupancy"] = obs_space_occupancy
            self.obs["target_action"] = obs_space_target_action

        elif self.config.observation_space_type == "laser_WP_1DCNN_FC":

            # Update laser scan
            #self.filter_laser_scan()

            if self.config.laser_normalize_flag:
                obs_laser = self.normalized_laser_ranges
            
            else:
                obs_laser = self.filtered_laser_ranges

            # Update waypoints 
            if self.config.wp_dynamic:
                self.init_obs_waypoints()
            
            else:
                self.update_obs_waypoints()

            # Update observation data
            self.obs_data["laser"] = np.vstack((self.obs_data["laser"], obs_laser))
            self.obs_data["laser"] = np.delete(self.obs_data["laser"], np.s_[0], axis=0)

            self.obs_data["waypoints"] = np.vstack((self.obs_data["waypoints"], self.obs_wp))
            self.obs_data["waypoints"] = np.delete(self.obs_data["waypoints"], np.s_[0], axis=0)

            self.obs_data["action"] = np.vstack((self.obs_data["action"], self.previous_action))
            self.obs_data["action"] = np.delete(self.obs_data["action"], np.s_[0], axis=0)

            #print("turtlebot3_tentabot_rl::update_observation -> obs_data laser shape: " + str(self.obs_data["laser"].shape))
            #print("turtlebot3_tentabot_rl::update_observation -> obs_data waypoints shape: " + str(self.obs_data["waypoints"].shape))
            #print("turtlebot3_tentabot_rl::update_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))

            # Update observation
            obs_space_laser = self.obs_data["laser"][-1,:].reshape(self.config.cnn_obs_shape)

            if self.config.n_obs_stack > 1:
                if(self.config.n_skip_obs_stack > 1):
                    latest_index = (self.config.n_obs_stack * self.config.n_skip_obs_stack) - 1
                    j = 0
                    for i in range(latest_index-1, -1, -1):
                        j += 1
                        if j % self.config.n_skip_obs_stack == 0:
                            obs_space_laser = np.vstack((self.obs_data["laser"][i,:].reshape(self.config.cnn_obs_shape), obs_space_laser))
                
                else:
                    obs_space_laser = self.obs_data["laser"]

            obs_space_wp_action = np.concatenate((self.obs_wp, self.previous_action), axis=0)

            # print("turtlebot3_tentabot_rl::update_observation -> obs_space_laser shape: " + str(obs_space_laser.shape))
            # print("turtlebot3_tentabot_rl::update_observation -> obs_space_wp_action shape: " + str(obs_space_wp_action.shape))

            self.obs["laser"] = obs_space_laser
            self.obs["waypoints_action"] = obs_space_wp_action

        elif self.config.observation_space_type == "Tentabot_WP_FC":

            # Update waypoints 
            if self.config.wp_dynamic:
                self.init_obs_waypoints()
            else:
                self.update_obs_waypoints()

            # Update tentabot observation
            success_rl_step = self.client_rl_step(1)
            if not success_rl_step:
                rospy.logerr("turtlebot3_tentabot_drl::update_observation -> OBSERVATION FAILURE!")

            # Update observation data
            self.obs_data["occupancy"] = np.vstack((self.obs_data["occupancy"], self.occupancy_set))
            self.obs_data["occupancy"] = np.delete(self.obs_data["occupancy"], np.s_[0], axis=0)

            self.obs_data["waypoints"] = np.vstack((self.obs_data["waypoints"], self.obs_wp))
            self.obs_data["waypoints"] = np.delete(self.obs_data["waypoints"], np.s_[0], axis=0)

            self.obs_data["action"] = np.vstack((self.obs_data["action"], self.previous_action))
            self.obs_data["action"] = np.delete(self.obs_data["action"], np.s_[0], axis=0)

            #print("turtlebot3_tentabot_drl::update_observation -> obs_data occupancy shape: " + str(self.obs_data["occupancy"].shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_data waypoints shape: " + str(self.obs_data["waypoints"].shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))

            # Update observation
            obs_stacked_occupancy = self.obs_data["occupancy"][-1,:].reshape(self.config.fc_obs_shape)

            if self.config.n_obs_stack > 1:
                latest_index = (self.config.n_obs_stack * self.config.n_skip_obs_stack) - 1
                j = 0
                for i in range(latest_index-1, -1, -1):
                    j += 1
                    if j % self.config.n_skip_obs_stack == 0:
                        obs_stacked_occupancy = np.hstack((self.obs_data["occupancy"][i,:], obs_stacked_occupancy))

            #print("turtlebot3_tentabot_drl::update_observation -> obs_stacked_occupancy shape: " + str(obs_stacked_occupancy.shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_wp shape: " + str(self.obs_wp.shape))
            #print("turtlebot3_tentabot_drl::update_observation -> previous_action shape: " + str(self.previous_action.shape))

            self.obs = np.concatenate((obs_stacked_occupancy, self.obs_wp, self.previous_action), axis=0)

            #print("turtlebot3_tentabot_drl::update_observation -> obs: " + str(self.obs.shape))

        elif self.config.observation_space_type == "laser_image_2DCNN_FC" or \
             self.config.observation_space_type == "laser_rings_2DCNN_FC":

            # Update target observation
            self.update_obs_target()

            obs_laser_image = self.laser_image
            obs_laser_image = np.expand_dims(obs_laser_image, axis=0)

            obs_target = self.obs_target

            # Update observation data
            self.obs_data["laser_image"] = np.vstack((self.obs_data["laser_image"], obs_laser_image))
            self.obs_data["laser_image"] = np.delete(self.obs_data["laser_image"], np.s_[0], axis=0)

            # Update observation
            obs_space_laser_image = self.obs_data["laser_image"][-1,:,:]
            obs_space_laser_image = np.expand_dims(obs_space_laser_image, axis=0)

            if self.config.n_obs_stack > 1:
                if(self.config.n_skip_obs_stack > 1):
                    latest_index = (self.config.n_obs_stack * self.config.n_skip_obs_stack) - 1
                    j = 0
                    for i in range(latest_index-1, -1, -1):
                        j += 1
                        if j % self.config.n_skip_obs_stack == 0:

                            obs_space_laser_image_current = self.obs_data["laser_image"][i,:,:]
                            obs_space_laser_image_current = np.expand_dims(obs_space_laser_image_current, axis=0)
                            obs_space_laser_image = np.vstack([obs_space_laser_image_current, obs_space_laser_image])
                
                else:
                    obs_space_laser_image = self.obs_data["laser_image"]

            self.obs_data["target"] = np.vstack((self.obs_data["target"], obs_target))
            self.obs_data["target"] = np.delete(self.obs_data["target"], np.s_[0], axis=0)

            self.obs_data["action"] = np.vstack((self.obs_data["action"], self.previous_action))
            self.obs_data["action"] = np.delete(self.obs_data["action"], np.s_[0], axis=0)

            obs_space_target_action = np.concatenate((obs_target, self.previous_action), axis=0)

            '''
            #print("**************** " + str(self.step_num))
            print("turtlebot3_tentabot_drl::update_observation -> obs_data laser_image shape: " + str(self.obs_data["laser_image"].shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_data target shape: " + str(self.obs_data["target"].shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_data action shape: " + str(self.obs_data["action"].shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_space_laser_image: ")
            #print(obs_space_laser_image[0, 65:75])
            #print("turtlebot3_tentabot_drl::update_observation -> obs_target dist: " + str(obs_target[0,0]))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_target angle: " + str(obs_target[0,1] * 180 / math.pi))
            #print("turtlebot3_tentabot_drl::update_observation -> previous_action: " + str(self.previous_action))
            print("turtlebot3_tentabot_drl::update_observation -> obs_laser_image shape: " + str(obs_laser_image.shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_space_laser_image type: " + str(type(obs_space_laser_image)))
            print("turtlebot3_tentabot_drl::update_observation -> obs_space_laser_image shape: " + str(obs_space_laser_image.shape))
            #print("turtlebot3_tentabot_drl::update_observation -> obs_space_target_action shape: " + str(obs_space_target_action.shape))
            print("****************")
            '''

            '''
            if self.step_num == 50:

                imi = (self.laser_image * 255).astype(np.uint8)
                im = Image.fromarray(imi)
                im = im.convert("L")
                im.save(self.data_folder_path + "laser_image_" + str(self.step_num) + ".jpeg")
                #np.savetxt(self.data_folder_path + "laser_image_" + str(self.step_num) + ".txt", self.laser_image)
            '''

            self.obs["laser_image"] = obs_space_laser_image
            self.obs["target_action"] = obs_space_target_action

    '''
    DESCRIPTION: TODO...
    '''
    def publish_goal(self):
        
        goal_visu = MarkerArray()

        marker = Marker()
        marker.header.frame_id = self.config.world_frame_name
        marker.ns = ""
        marker.id = 1
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.5
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_pose["x"]
        marker.pose.position.y = self.goal_pose["y"]
        marker.pose.position.z = self.goal_pose["z"]

        marker.header.seq += 1
        marker.header.stamp = rospy.Time.now()

        goal_visu.markers.append(marker)

        self.goal_visu_pub.publish(goal_visu);

    '''
    DESCRIPTION: TODO...
    '''
    def publish_move_base_goal(self):

        print("turtlebot3_tentabot_drl::publish_move_base_goal -> x: " + str(self.goal_pose["x"]) + " y: " + str(self.goal_pose["y"]) + " z: " + str(self.goal_pose["z"]))

        self.move_base_goal.pose.position.x = self.goal_pose["x"]
        self.move_base_goal.pose.position.y = self.goal_pose["y"]
        self.move_base_goal.pose.position.z = self.goal_pose["z"]
        self.move_base_goal.pose.orientation.z = 0.0
        self.move_base_goal.pose.orientation.w = 1.0
        self.move_base_goal.header.seq += 1
        self.move_base_goal.header.frame_id = self.config.world_frame_name
        self.move_base_goal.header.stamp = rospy.Time.now()
        self.move_base_goal_pub.publish(self.move_base_goal)

    '''
    DESCRIPTION: TODO...
    '''
    def publish_debug_visu(self, debug_data):

        debug_visu = MarkerArray()

        for i, d in enumerate(debug_data):

            if d[0] < float("inf") or d[1] < float("inf"):
                marker = Marker()
                marker.header.frame_id = self.config.world_frame_name
                marker.ns = str(i)
                marker.id = i
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = d[0]
                marker.pose.position.y = d[1]
                marker.pose.position.z = 0

                debug_visu.markers.append(marker)
        
        if len(debug_visu.markers) > 0:
            
            for m in debug_visu.markers:
                m.header.seq += 1
                m.header.stamp = rospy.Time.now()
            
            self.debug_visu_pub.publish(debug_visu)

    '''
    DESCRIPTION: TODO...
    '''
    def publish_wp_visu(self, full_data, obs_data):

        debug_visu = MarkerArray()

        #delete previous markers
        marker = Marker()
        marker.ns = str(0)
        marker.id = 0
        marker.action = marker.DELETEALL
        debug_visu.markers.append(marker)

        for i, d in enumerate(full_data):

            marker = Marker()
            marker.header.frame_id = self.config.world_frame_name
            marker.ns = str(i+1)
            marker.id = i+1
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = d.pose.position.x
            marker.pose.position.y = d.pose.position.y
            marker.pose.position.z = 0

            debug_visu.markers.append(marker)

        for j, d in enumerate(obs_data):

            marker = Marker()
            marker.header.frame_id = self.config.world_frame_name
            marker.ns = str(i+j+1)
            marker.id = i+j+1
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.1
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = d.pose.position.x
            marker.pose.position.y = d.pose.position.y
            marker.pose.position.z = 0

            debug_visu.markers.append(marker)

        if len(debug_visu.markers) > 0:
            
            for m in debug_visu.markers:
                m.header.seq += 1
                m.header.stamp = rospy.Time.now()
            
            self.debug_visu_pub.publish(debug_visu)

    '''
    DESCRIPTION: TODO...
    '''
    def client_rl_step(self, parity):
        
        #rospy.wait_for_service('rl_step')
        try:
            #srv_rl_step = rospy.ServiceProxy('rl_step', rl_step, True)
            tentabot_client = self.srv_rl_step(parity)
            
            if self.config.observation_space_type == "Tentabot_FC":
                self.occupancy_set = (np.asarray(tentabot_client.occupancy_set)).reshape(self.config.fc_obs_shape)

            else:
                self.occupancy_set = (np.asarray(tentabot_client.occupancy_set)).reshape(self.config.cnn_obs_shape)

            '''
            print("--------------")
            print("turtlebot3_tentabot_drl::client_rl_step -> min id: " + str(np.argmin(self.occupancy_set)) + " val: " + str(np.min(self.occupancy_set)))
            print("turtlebot3_tentabot_drl::client_rl_step -> max id: " + str(np.argmax(self.occupancy_set)) + " val: " + str(np.max(self.occupancy_set)))
            print("turtlebot3_tentabot_drl::client_rl_step -> ")
            for i, val in enumerate(self.occupancy_set[0]):
                if 65 < i < 80:
                    print(str(i) + ": " + str(val))
            print("--------------")
            '''

            #self.tentabot_obs = occupancy_set
            #self.obs = np.stack((clearance_set, clutterness_set, closeness_set), axis=0)
            return True

        except rospy.ServiceException as e:
            print("turtlebot3_tentabot_drl::client_rl_step -> Service call failed: %s"%e)
            return False

    '''
    DESCRIPTION: TODO...
    '''
    def client_update_goal(self):

        #rospy.wait_for_service('update_goal')
        try:
            #srv_update_goal = rospy.ServiceProxy('update_goal', update_goal, True)
            print("turtlebot3_tentabot_drl::get_goal_location -> Updated goal_pose x: " + str(self.goal_pose["x"]) + ", y: " + str(self.goal_pose["y"]))
            
            goalMsg = Pose()
            goalMsg.orientation.z = 0.0
            goalMsg.orientation.w = 1.0
            goalMsg.position.x = self.goal_pose["x"]
            goalMsg.position.y = self.goal_pose["y"]
            goalMsg.position.z = self.goal_pose["z"]

            success = self.srv_update_goal(goalMsg).success

            if(success):
                #print("turtlebot3_tentabot_drl::get_goal_location -> Updated goal_pose x: " + str(self.goal_pose["x"]) + ", y: " + str(self.goal_pose["y"]))
                rospy.logdebug("turtlebot3_tentabot_drl::client_update_goal -> Updated goal_pose x: " + str(self.goal_pose["x"]) + ", y: " + str(self.goal_pose["y"]))
            else:
                #print("turtlebot3_tentabot_drl::client_update_goal -> goal_pose is NOT updated!")
                rospy.logdebug("turtlebot3_tentabot_drl::client_update_goal -> goal_pose is NOT updated!")

            return success

        except rospy.ServiceException as e:  
            print("turtlebot3_tentabot_drl::client_update_goal -> Service call failed: %s"%e)
            return False

    '''
    DESCRIPTION: TODO...
    '''
    def client_reset_map_utility(self, parity):

        #rospy.wait_for_service('reset_map_utility')
        try:
            #srv_reset_map_utility = rospy.ServiceProxy('reset_map_utility', reset_map_utility, True)        
            
            success = self.srv_reset_map_utility(parity).success
            
            if(success):
                print("turtlebot3_tentabot_drl::client_reset_map_utility -> Map is reset!")
                rospy.logdebug("turtlebot3_tentabot_drl::client_reset_map_utility -> Map is reset!")

            else:
                print("turtlebot3_tentabot_drl::client_reset_map_utility -> Map is NOT reset!")
                rospy.logdebug("turtlebot3_tentabot_drl::client_reset_map_utility -> Map is NOT reset!")

            return success

        # Reset Robot Pose and Goal
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    '''
    DESCRIPTION: TODO...
    '''
    def client_move_base_get_plan(self):

        try:
            self.update_odom()

            start = PoseStamped()
            start.pose = self.odom_data.pose.pose
            start.header.seq += 1
            start.header.frame_id = self.config.world_frame_name
            start.header.stamp = rospy.Time.now()

            goal = PoseStamped()
            goal.pose.position.x = self.goal_pose["x"]
            goal.pose.position.y = self.goal_pose["y"]
            goal.pose.position.z = self.goal_pose["z"]
            goal.pose.orientation.z = 0.0
            goal.pose.orientation.w = 1.0
            goal.header.seq += 1
            goal.header.frame_id = self.config.world_frame_name
            goal.header.stamp = rospy.Time.now()

            tolerance = 0.5
            
            #self.srv_clear_costmap()
            self.move_base_global_plan = self.srv_move_base_get_plan(start, goal, tolerance).plan.poses
            
            if(len(self.move_base_global_plan)):
                
                #print("turtlebot3_tentabot_drl::client_move_base_get_plan -> move_base plan is received!")
                rospy.logdebug("turtlebot3_tentabot_drl::client_move_base_get_plan -> move_base plan is received!")
                success = True

            else:
            
                print("turtlebot3_tentabot_drl::client_move_base_get_plan -> move_base plan is received!")
                rospy.logdebug("turtlebot3_tentabot_drl::client_move_base_get_plan -> move_base plan is received!")
                success = False

            return success

        # Reset Robot Pose and Goal
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    '''
    DESCRIPTION: TODO...
    '''
    def update_global_path_length(self):

        if self.client_move_base_get_plan():

            n_points_plan = len(self.move_base_global_plan)
            self.global_plan_length = 0

            p1 = {'x': self.move_base_global_plan[0].pose.position.x, 'y': self.move_base_global_plan[0].pose.position.y, 'z': self.move_base_global_plan[0].pose.position.z}
            for i in range(1, n_points_plan):

                p2 = {'x': self.move_base_global_plan[i].pose.position.x, 'y': self.move_base_global_plan[i].pose.position.y, 'z': self.move_base_global_plan[i].pose.position.z}
                self.global_plan_length += self.calculate_euclidean_distance(p1, p2)
                p1 = p2

            print("turtlebot3_tentabot_drl::update_global_path_length -> global_plan_length: " + str(self.global_plan_length))

    '''
    DESCRIPTION: TODO...
    '''
    def reset_pedsim(self):
        
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [self.tentabot_path + "/launch/others/pedsim_ros/start_pedsim_validation.launch"])
        launch.start()