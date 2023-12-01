#!/usr/bin/env python3

"""
Created on Sun Dec 23 00:47:49 2018

@author: anthony
"""

import time
import math
import random
import numpy as np
from collections import deque
import copy
import rospy
import subprocess
import signal
import os
import sys

#import openai gym and other ml modules
import gym
from gym.utils import seeding
from gym import spaces, logger
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from gym.envs.registration import register

#import ros messages
import std_msgs.msg as stdmsg
import sensor_msgs.msg as sensormsg
import geometry_msgs.msg as geomsg
import syropod_highlevel_controller.msg as shcmsg
import nav_msgs.msg as navmsg
import gazebo_msgs.msg as gzmsg
import gait_adaptation.msg as gamsg

#import ros services
from std_srvs.srv import Empty
import gazebo_msgs.srv as gzsrv


from time import sleep
import controller_manager_msgs.srv as cmmsrv

reg = register(
    id='GaitAdaptationEnv-v1',
    entry_point='gait_selection_ros:GaitAdaptationEnv',
    timestep_limit=100,
)

leg_names = [
    "AR",
    "BR",
    "CR",
    "CL",
    "BL",
    "AL"
]

leg_topics = [
    "/shc/AR/state",
    "/shc/BR/state",
    "/shc/CR/state",
    "/shc/CL/state",
    "/shc/BL/state",
    "/shc/AL/state"
]

param_names = [
    "null",
    "step frequency",
    "swing_height",
    "swing_width",
    "step_depth",
    "stance_span_modifier",
    "virtual_mass",
    "virtual_stiffness",
    "virtual_damping_ratio",
    "force_gain"
]

min_thresholds =[
    0,
    0.2,
    0.02,
    -0.3,
    0.,
    -1.,
    1.,
    5.,
    0.1,
    0.001
]

max_thresholds = [
    0,
    2.0,
    0.12,
    0.3,
    0.,
    1.,
    100.,
    45.,
    100.,
    100.
]

controller_names = [
    "joint_state_controller",
    "AL_coxa_joint",
    "AL_coxat_joint",
    "AL_femur_joint",
    "AL_tibia_joint",
    "AL_tarsus_joint",
    "AR_coxa_joint",
    "AR_coxat_joint",
    "AR_femur_joint",
    "AR_tibia_joint",
    "AR_tarsus_joint",
    "BL_coxa_joint",
    "BL_coxat_joint",
    "BL_femur_joint",
    "BL_tibia_joint",
    "BL_tarsus_joint",
    "BR_coxa_joint",
    "BR_coxat_joint",
    "BR_femur_joint",
    "BR_tibia_joint",
    "BR_tarsus_joint",
    "CL_coxa_joint",
    "CL_femur_joint",
    "CL_tibia_joint",
    "CL_tarsus_joint",
    "CR_coxa_joint",
    "CR_coxat_joint",
    "CR_femur_joint",
    "CR_tibia_joint",
    "CR_tarsus_joint"
]

# http://mathfaculty.fullerton.edu/mathews/n2003/differentiation/numericaldiffproof.pdf
#error is O(h^4)
d_dt = np.array([1./12.0, -8./12.0, 0.0, 8./12.0, -1./12.0])
d_dt2 = np.array([-1./12., 16./12., -30./12., 16./12., -1./12.])

shc_param_setter_topic = "/syropod_remote/parameter_set"
step_complete_topic = "/gait_adaptation/step_complete"
shc_desired_velocity_topic = "/syropod_remote/desired_velocity"
pcl_stats_topic = "/gait_adaptation/pcl_stats"
shc_status_topic = "/shc/status"
joystick_topic = "/joy"

step_complete = 0

joy_pub = rospy.Publisher(joystick_topic, sensormsg.Joy, queue_size = 0)
rate = rospy.Rate(50)

def status_message_callback(msg):
    buttons = [0,0,0,0,0,0,0,0,0,0,0]
    axes = [0., 0., 0., 0., 0., 0., 0., 0.]

    if msg.data == 0:
        button = 8
    elif msg.data == 1:
        button = 7
    
    joy_msg = sensormsg.Joy()
    joy_msg.axes = axes
    joy_msg.buttons = buttons
    joy_msg.buttons[button] = 1
    joy_pub.publish(joy_msg)

    rate.sleep()

    joy_msg.buttons[button] = 0
    joy_pub.publish(joy_msg)

def stop_process_by_name(names):
    processes = subprocess.check_output(['ps']).decode("utf-8")
    processes = processes.split('\n')
    for name in names:
        for process in processes:
            process = [x for x in process.split(" ") if len(x) > 0]
            if name in process:
                try:
                    # pid = int(process[0])
                    subprocess.check_output(["killall -9", process[0]])
                except:
                    continue

def  quaternion_to_euler(w, x, y, z):
    theta_x = math.atan2(2*(w*x + y*z), 1 - 2 * (x**2 + y**2))
    theta_y = math.asin(2 * (w * y - z * x))
    theta_z = math.atan2(2*(w*z + x*y), 1 - 2 * (y**2 + z**2))

    return theta_z, theta_y, theta_x

def euler_to_quaternion(z, y, x):

    cx = math.cos(x * 0.5)
    cy = math.cos(y * 0.5)
    cz = math.cos(z * 0.5)

    sx = math.sin(x * 0.5)
    sy = math.sin(y * 0.5)
    sz = math.sin(z * 0.5)

    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz

    return qw, qx, qy, qz

class GazeboConnection():
    
    def __init__(self):
        
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.switch_controllers_proxy = rospy.ServiceProxy('/syropod/controller_manager/switch_controller', cmmsrv.SwitchController)
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', gzsrv.SetModelState)
    
    def pause_physics(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            print("Pausing Simulation...", file=sys.stderr)
            self.pause()
        except rospy.ServiceException as e:
            print ("/gazebo/pause_physics service call failed", file=sys.stderr)
        
    def unpause_physics(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
            print("Unpausing Simulation", file=sys.stderr)
        except rospy.ServiceException as e:
            print ("/gazebo/unpause_physics service call failed", file=sys.stderr)
        
    def reset_sim(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print ("/gazebo/reset_simulation service call failed", file=sys.stderr)

    def reset_world(self):
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            print ("/gazebo/reset_world service call failed", file=sys.stderr)

    def switch_controllers(self, active):
        if active:
            #If active, deactivate the controllers
            stop_controllers = controller_names
            start_controllers = []
        else:
            #If inactive, restart the controllers
            stop_controllers=[]
            start_controllers = controller_names
        rospy.wait_for_service('/syropod/controller_manager/switch_controller')
        try:
            self.switch_controllers_proxy(start_controllers, stop_controllers, 0)
        except rospy.ServiceException as e:
            print ("/gazebo/reset_world service call failed", file=sys.stderr)

    def set_model_state(self, msg):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_state_proxy(msg)
        except rospy.ServiceException as e:
            print('/gazebo/set_model_state service call failed', file=sys.stderr)



class GaitAdaptationEnv(gym.Env):
    def __init__(self):

        if rospy.is_shutdown():
            print("SHUTDOWN", file=sys.stderr)
        self.num_steps = 0
        self.num_runs = 0

        self.show = False
        self.reset_imu = False
        self.time = rospy.get_time()

        ##Hold subprocess information for gazebo and shc
        self.gazebo = GazeboConnection()

        ##Cost function coefficients
        self.smoothness_coeff = 0.001
        self.distance_coeff = 20
        self.time_coeff = 0.2
        self.vel_coeff = 10

        ###acceleration data to calculate smoothness cost
        self.ax = np.array([])
        self.ay = np.array([])
        self.az = np.array([])

        ###terrain parameters
        self.params = list([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        ###ros settings. init node and set up subscribers and publishers
        # self.leg_param_pub = rospy.Publisher(shc_param_setter_topic, shcmsg.ParamSetting, queue_size=0)
        # self.complete_step_sub = rospy.Subscriber(step_complete_topic, stdmsg.Int8, self.step_complete_callback)
        # self.shc_status_sub = rospy.Subscriber(shc_status_topic, stdmsg.Int8, status_message_callback)
        self.body_vel_pub = rospy.Publisher(shc_desired_velocity_topic, geomsg.Twist, queue_size=0)
        self.param_pub = rospy.Publisher(shc_param_setter_topic, shcmsg.MultiParam, queue_size=0)
        self.reset_pub = rospy.Publisher("/gait_adaptation/reset_all", stdmsg.Int8, queue_size=0)

        self.imu_sub = rospy.Subscriber("/imu/data", sensormsg.Imu, self.imu_callback)
        self.obs_sub = rospy.Subscriber(pcl_stats_topic, gamsg.pcl_stats, self.observation_callback)
        self.odometry_sub = rospy.Subscriber("/gazebo/Odometry", navmsg.Odometry, self.odometry_callback)



        self.action_space = spaces.Box(low=-2, high=2, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(10,), dtype=np.float32)

        self.paused = False
        # self.reset()

    def pause_physics(self):
        self.gazebo.pause_physics()
        self.paused = True

    def unpause_physics(self):
        self.gazebo.unpause_physics()
        self.paused = False

    def randomize_conditions(self):
        # if self.num_runs % 10 == 0:
        #     world = np.random.randint(0, 10)
        # else:
        #     world = self.world

        # if world != self.world:
        #     self.world = world
        #     ##Read in png file for the heightmap
        #     # self.world_im = 
        # size = np.shape(self.world_im)[0]
        # pixel_x = np.floor(size * (x_pos + 50) / 100)
        # pixel_y = np.floor(size * (y_pos + 50) / 100)

        # z_pos = 5 * self.world_im[pixel_y, pixel_x] / 255 + 0.3

        # self.num_runs += 1
        
        x_pos = np.random.rand()* 80. - 40.
        y_pos = np.random.rand()* 80. - 40.
        z_pos = .3

        z_angle = np.random.randint(-180, 180)*np.pi/180
        qw, qx, qy, qz = euler_to_quaternion(z_angle, 0, 0)

        self.publish_model_state(x_pos, y_pos, z_pos, qw, qx, qy, qz)
        
        

    def reset(self):
        if self.paused:
            self.unpause_physics()

        self.publish_params([0., (1.2-0.1)/(2.5-0.1), (0.1-0.02)/(0.12-0.02), (12.-5.)/(45.-5.)])
        # self.gazebo.switch_controllers(True)

        self.pause_physics()
        self.gazebo.reset_world()
        self.gazebo.reset_sim()
        self.randomize_conditions()
        self.unpause_physics()
        sleep(0.5)
        # self.reset_pub.publish(1)
        subprocess.check_output(['rosnode', 'kill', '/syropod_highlevel_controller'])
        rospy.wait_for_message('/shc/ready', stdmsg.Int8)

        
        # for leg in range(6):
        #     self.construct_paramsetter_message(leg, 7, 0.5)
        # rospy.wait_for_message(step_complete_topic, stdmsg.Int8)
        obs = self.params
        self.reset_imu = True
        self.num_steps = 0
        self.step_complete = 0
        self.time = rospy.get_time()

        self.pause_physics()

        return obs

    def seed(self, _):
        return [0]

    def observation_callback(self, pcl_stats):
        self.params[0] = pcl_stats.x_slope
        self.params[1] = pcl_stats.y_slope
        self.params[2] = pcl_stats.variance
        self.params[3] = pcl_stats.shadow_fraction
        self.params[4] = pcl_stats.max_height
        self.params[5] = pcl_stats.centre_line_avg
    
    def odometry_callback(self, odom_msg):
        self.x_pos = odom_msg.pose.pose.position.x
        self.y_pos = odom_msg.pose.pose.position.y
        self.xq = odom_msg.pose.pose.orientation.x
        self.yq = odom_msg.pose.pose.orientation.y
        self.zq = odom_msg.pose.pose.orientation.z
        self.wq = odom_msg.pose.pose.orientation.w
        self.theta,_,_ = quaternion_to_euler(self.wq, self.xq, self.yq, self.zq)

    def publish_params(self, action):
        msg = shcmsg.MultiParam()
        
        
        msg.x_vel = action[0]
        # msg.y_vel = action[1]
        # msg.stride_length = (max_thresholds[1] - min_thresholds[1]) * max(min(action[2], 1.), 0.) + min_thresholds[1]
        msg.freq          = (max_thresholds[1] - min_thresholds[1]) * max(min(action[1], 1.), 0.) + min_thresholds[1]
        msg.step_height   = (max_thresholds[2] - min_thresholds[2]) * max(min(action[2], 1.), 0.) + min_thresholds[2]
        msg.stiffness     = (max_thresholds[7] - min_thresholds[7]) * max(min(action[3], 1.), 0.) + min_thresholds[7]
        
        self.params[6] = msg.x_vel
        self.params[7] = msg.freq
        self.params[8] = msg.step_height
        self.params[9] = msg.stiffness

        self.param_pub.publish(msg)

    def publish_model_state(self, x, y, z, qw, qx, qy, qz):
        msg = gzmsg.ModelState()

        msg.model_name = 'hexapod'

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        msg.pose.orientation.w = qw
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz

        msg.reference_frame = 'world'

        self.gazebo.set_model_state(msg)

    def calculate_total_jerk(self):
        # note: imu data published at 100hz
        dt = 0.01
        length = min(len(self.ax), len(self.ay), len(self.az))
        j = []

        for idx in range(2, length - 3):
            ##Calculate linear jerk in each direction
            jx = sum(np.multiply(self.ax[idx-2:idx+3], d_dt))
            jy = sum(np.multiply(self.ay[idx-2:idx+3], d_dt))
            jz = sum(np.multiply(self.az[idx-2:idx+3], d_dt))
            #Calculate jerk squared
            j.append(jx*jx + jy*jy + jz*jz)
        return sum(j)*dt

    def step_complete_callback(self, msg):
        
        time = rospy.get_time()
        # print("step complete message received at time {}, diff {}, data {}".format(time, time - self.time, msg.data))
        if (time - self.time) > 0.2:
            # print("step complete message received {} -{} = {}".format(time, self.time, time - self.time))
            self.step_complete = msg.data
            self.time = time
        elif time < self.time:
            self.time = time

    def imu_callback(self, msg):
        if self.reset_imu:
            self.ax = np.array([])
            self.ay = np.array([])
            self.az = np.array([])
            self.reset_imu = False
        imu_data = msg
        # print("received imu data {}".format(msg))
        self.ax = np.append(self.ax, imu_data.linear_acceleration.x)
        self.ay = np.append(self.ay, imu_data.linear_acceleration.y)
        self.az = np.append(self.az, imu_data.linear_acceleration.z)


    def step(self, action):
        """
        Args:
        -action: tuple/list of parameters to set:
            (x_vel, y_vel, step_height, step_freq, leg_stiffness)#6x leg stiffnesses in order of leg_names)
        """

        over = False
        ##reset params
        print("Original step parameters: {}".format(action), file=sys.stderr)
        for idx in range(0, len(action)):
            action[idx] = (action[idx] + 2.)/4.
            action[idx] = min(1., max(0, action[idx]))

        x_vel = (1.2 - 0) * max(min(action[0], 1.), 0.05)
        action[0] = x_vel
        self.reset_imu = True

        self.unpause_physics()
        self.publish_params(action)       

        x_pos = self.x_pos
        y_pos = self.y_pos
        x_axis_transformed = [np.cos(self.theta), np.sin(self.theta)]

        print("Taking step {} with parameters {}".format(self.num_steps, action), file=sys.stderr)
        time = rospy.get_time()
        self.time = time
        self.step_complete = 0
        ###wait for step to complete to calculate cost
        while rospy.get_time() - time < 0.2:
            continue
        completed = rospy.wait_for_message(step_complete_topic, stdmsg.Int8)
        if completed.data == 0: #Step failed
            # over = True
            obs = self.params
            cost = -100
            self.pause_physics()
            return obs, cost, over, {}
            

        
        print("Step complete. checking odom", file=sys.stderr)
        time = (rospy.get_time() - time)
        distance_travelled = (self.x_pos - x_pos)*x_axis_transformed[0] + (self.y_pos - y_pos)*x_axis_transformed[1]
        # print("Taking observation", file=sys.stderr)
        obs = self.params
        # if self.pcl_reset_flag == 1:
        #     done = True
        self.pause_physics()


        smoothness_cost = self.calculate_total_jerk()
        # print("waiting for odometry message 2")
        
        ###Seek to  maximize smoothness and distance travelled, and minimize time
        cost = -self.smoothness_coeff * smoothness_cost + self.vel_coeff * distance_travelled /  time
        print("smoothness cost={}, time={}, distance_travelled={}, velocity = {} total={}".format(smoothness_cost, time, distance_travelled, distance_travelled/time, cost), file=sys.stderr)
        

        self.num_steps += 1
        
        if self.num_steps >= 50:
            over = True
        ##returns
        over = False
        return obs, cost, over, {}

    def render(self, show=True):
        self.show = show