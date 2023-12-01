#!/usr/bin/env python
from __future__ import print_function
import os, sys, cv2, math, time
import numpy as np
import msgpack
import msgpack_numpy as m
from collections import deque

#ROS Dependencies
import roslib, rospy
import numpy as np
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from sensor_msgs.msg import Image, LaserScan, Joy
from cv_bridge import CvBridge, CvBridgeError

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class Env(object):
    """
    Stripped down version from OpenaiGym
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None
    ser_msg_length = 0

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def reset(self, **kwargs):
        """Resets the state of the environment and returns an initial observation.
        Returns: 
            observation (object): the initial observation.
        """
        raise NotImplementedError

    def serialize_obs(self):
        """Returns a function that allows you to serialize each observation as a multipart"""
        raise NotImplementedError
    
class f110Env(Env):
    """ Implements a Gym Environment & neccessary funcs for the F110 Autonomous RC Car(similar structure to gym.Env or gym.Wrapper)
    """
    def __init__(self):
        rospy.init_node("Gym_Recorder", anonymous=True, disable_signals=True)
        #At least need LIDAR, IMG & STEER for everything here to work 
        self.obs_info = {
            'lidar': {'topic':'/scan', 'type':LaserScan, 'callback':self.lidar_callback},

            'img': {'topic':'/usb_cam/image_raw', 'type':Image, 'callback':self.img_callback},

            'steer':{'topic':'/vesc/low_level/ackermann_cmd_mux/output', 'type':AckermannDriveStamped, 'callback':self.steer_callback}
        }

        #one observation could be 4 consecutive readings, so init deque for safety
        self.latest_obs = deque(maxlen=4)         
        self.latest_reading_dict = {}
        self.record = False
        self.rev = False
        self.last_step_time = time.time()

        #misc
        self.bridge = CvBridge()
        self.history= deque(maxlen=500) #for reversing during reset

        #GYM Properties (set in subclasses)
        self.observation_space = ['lidar', 'steer', 'img']
        self.action_space = ['angle', 'speed']
        self.ser_msg_length = 4
        self.joy_array = []
        self.setup_subs()

        #Subscribe to joy (to access record_button) & publish to ackermann
        self.joy_sub = rospy.Subscriber('/vesc/joy', Joy, self.joy_callback)        
        self.drive_pub = rospy.Publisher("vesc/high_level/ackermann_cmd_mux/input/nav_0", AckermannDriveStamped, queue_size=20) 

    ############ GYM METHODS ###################################

    def _get_obs(self):
        """
        Returns latest observation 
        """
        while(len(self.latest_obs) == 0):
            rospy.sleep(0.04)
        obs_dict = self.latest_obs[-1]
        return obs_dict
        
    def reset(self, **kwargs):
        """
        Reverse until we're not 'tooclose'
        """
        print("\n RESETTING_ENV")
        self.record = False
        self.rev = True
        obs = self._get_obs()
        while(self.tooclose()):
            self.reverse()
        #Back up a bit more
        for i in range(10):
            dmsg = self.get_drive_msg(0.0, -2.0)
            self.drive_pub.publish(dmsg)
        self.record = True
        self.rev = False
        #TODO: consider sleeping a few milliseconds?
        self.latest_obs.clear()
        return obs

    def get_reward(self):
        """
        TODO:Implement reward functionality
        """
        return 0

    def step(self, action):
        """
        Action should be a steer_dict = {"angle":float, "speed":float}
        """
        #execute action
        drive_msg = self.get_drive_msg(action.get("angle"), action.get("speed"), flip_angle=-1.0)
        self.drive_pub.publish(drive_msg)

        #get reward & check if done & return
        obs = self._get_obs()
        reward = self.get_reward()
        done = self.tooclose()
        info = {'record':self.record, 'buttons':self.joy_array}
        self.latest_obs.clear()
        return obs, reward, done, info
    
    def serialize_obs(self):
        """ Currently assume obs consists of sensor [lidar, steer, img]
        """
        def _ser(obs_dict):
            lidar_dump = msgpack.dumps(obs_dict["lidar"])
            steer_dump = msgpack.dumps(obs_dict["steer"])
            cv_img = obs_dict["img"]
            cv_md = dict(
            dtype=str(cv_img.dtype),
            shape=cv_img.shape,
            )
            cv_md_dump = msgpack.dumps(cv_md)
            multipart_msg = [lidar_dump, steer_dump, cv_md_dump, cv_img]
            return multipart_msg
        return _ser

    ############ GYM METHODS ###################################
    ############ ROS HANDLING METHODS ###################################

    def setup_subs(self):
        """
        Initializes subscribers w/ obs_info & returns a list of subscribers
        """
        obs_info = self.obs_info
        makesub = lambda subdict : rospy.Subscriber(subdict['topic'], subdict['type'], subdict['callback']) 

        sublist = []
        for topic in obs_info:
            sublist.append(makesub(obs_info[topic]))
        self.sublist = sublist

    def add_to_history(self, data):
        if abs(data.drive.steering_angle) > 0.05 and data.drive.steering_angle < -0.05 and data.drive.steering_angle is not None:
            steer_dict = {"angle":data.drive.steering_angle, "speed":data.drive.speed}
            for i in range(40):
                self.history.append(steer_dict) 

    def steer_callback(self, data):
        if data.drive.steering_angle > 0.34:
            data.drive.steering_angle = 0.34
        elif data.drive.steering_angle < -0.34:
            data.drive.steering_angle = -0.34

        steer = dict(
            angle = -1.0 * data.drive.steering_angle, 
            steering_angle_velocity = data.drive.steering_angle_velocity,
            speed = data.drive.speed
        )
        self.latest_reading_dict["steer"] = steer

        self.add_to_history(data) #add steering commands to history

    def lidar_callback(self, data):
        lidar = dict(
            angle_min = data.angle_min,
            angle_increment = data.angle_increment,
            ranges = data.ranges
        )
        self.latest_reading_dict["lidar"] = lidar 
    
    def joy_callback(self, data):
        record_button = data.buttons[1]
        if record_button:
            self.record = True
        else:
            self.record = False
        self.joy_array = list(data.buttons)

    def set_status_str(self, prefix=''):
        status_str = ''
        if self.record:
            status_str = 'True'
        else:
            status_str = 'False'
        sys.stdout.write(prefix + "curr_recording: %s" % status_str)
        
        sys.stdout.flush()
    
    def is_reading_complete(self):
        #checks if all the readings are present in latest_reading_dict
        base_check = "lidar" in self.latest_reading_dict and "steer" in self.latest_reading_dict
        return base_check

    def base_preprocessing(self, cv_img):
        cv_img = cv2.resize(cv_img, None, fx=0.5, fy=0.5)
        return cv_img

    def update_latest_obs(self):
        self.latest_obs.append(self.latest_reading_dict)
        self.latest_reading_dict = {}

    def img_callback(self, data):
        self.set_status_str(prefix='\r')

        #img_callback adds latest_reading to the self.lates_obs
        if self.is_reading_complete():
            try:
                cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e) 
            cv_img  = self.base_preprocessing(cv_img)
            self.latest_reading_dict["img"] = cv_img

            #at this point, the reading must be done
            self.update_latest_obs()

    def get_drive_msg(self, angle, speed, flip_angle=1.0):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "odom" 
        drive_msg.drive.steering_angle = flip_angle * angle
        drive_msg.drive.speed = speed
        return drive_msg

    def reverse(self):
        """
        Uses self.history to back out
        """
        sign = lambda x: (1, -1)[x < 0]
        default_steer_dict = {"angle":0.0, "speed":-1.0}
        try:
            steer_dict = self.history.pop()
        except:
            steer_dict = default_steer_dict

        rev_angle = steer_dict["angle"]
        rev_speed = -2.0
        #print("REVERSE {rev_angle}".format(rev_angle = rev_angle))
        drive_msg = self.get_drive_msg(rev_angle, rev_speed)
        self.drive_pub.publish(drive_msg)
    
    def tooclose(self):
        """
        Uses self.latest_obs to determine if we are too_close (currently uses LIDAR)
        """
        tc = True
        if len(self.latest_obs) > 0:

            reading = self.latest_obs[-1]

            #Use LIDAR Reading to check if we're too close
            lidar = reading["lidar"]
            ranges = lidar.get("ranges")
            angle_min = lidar.get("angle_min")
            angle_incr = lidar.get("angle_incr")
            rfrac = lambda st, en : ranges[int(st*len(ranges)):int(en*len(ranges))]
            mindist = lambda r, min_range : np.nanmin(r[r != -np.inf]) <= min_range
            #ensure that boundaries are met in each region
            r1 = rfrac(0, 1./4.)
            r2 = rfrac(1./4., 3./4.)
            r3 = rfrac(3./4., 1.) 
            if mindist(r1, 0.2) or mindist(r2, 0.4) or mindist(r3, 0.2):
                tc = True
            else:
                tc = False
        else:
            tc = False

        return tc
    ############ ROS HANDLING METHODS ###################################

class f110Wrapper(Env):
    """
    Wraps the f110Env to allow a modular transformation.
    
    This class is the base class for all wrappers. The subclasses can override some methods to change behaviour of the original f110Env w/out touching the original code
    """
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def compute_reward(self, info):
        return self.env.get_reward()

class f110ObservationWrapper(f110Wrapper):
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        raise NotImplementedError
    
    def serialize_obs(self):
        raise NotImplementedError
    

class f110RewardWrapper(f110Wrapper):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info
    
    def reward(self, reward):
        raise NotImplementedError

class f110ActionWrapper(f110Wrapper):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        raise NotImplementedError
    
    def reverse_action(self, action):
        raise NotImplementedError
