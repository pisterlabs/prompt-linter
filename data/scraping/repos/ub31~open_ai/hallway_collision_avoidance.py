from gym import spaces
from openai_ros.robot_envs import hallway_env
from gym.envs.registration import register
from geometry_msgs.msg import Twist
import ros_numpy

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import rospy
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from math import exp
import time
import math
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

# The path is __init__.py of openai_ros, where we import the MovingCubeOneDiskWalkEnv directly
timestep_limit_per_episode = 1000  # Can be any Value

register(
    id='HallwayCollision-v0',
    entry_point='openai_ros.task_envs.hallway.hallway_collision_avoidance:HallwayCollisionAvoidance',
    max_episode_steps=timestep_limit_per_episode,
    # timestep_limit=timestep_limit_per_episode,
)


class HallwayCollisionAvoidance(hallway_env.HallwayEnv):
    def __init__(self):

        #---------CONTINUOUS ACTION SPACE--------------
        #self.action_space = spaces.Box(-1,1,(2,),dtype=np.float32)

        #---------DISCRETE ACTION SPACE--------------
        self.action_space = spaces.Discrete(9)

        self.reward_range = (-np.inf, np.inf)

        # Observation space : 100 laser scan ranges + 3 odometry
        self.n_observations = 100
        self.max_laser_value = 20.0
        self.min_laser_value = 0.02
        high = np.full((self.n_observations), self.max_laser_value)
        low = np.full((self.n_observations), self.min_laser_value)

        # add odometery to observations
        self.n_observations+=9
        self.high = np.append(high,[10,10,1])
        self.low = np.append(low,[-10,-10,-1])

        self.odom_high = np.array([10,10,1])
        self.odom_low = np.array([-10,-10,-1])
        # laser range as observation space
        # self.observation_space = spaces.Box(np.zeros(self.n_observations), np.ones(self.n_observations),dtype=np.float32)

        # kinect depth as observation space
        self.observation_space = spaces.Box(low=0, high=255,shape=(64,128,3),dtype=np.float32)

        self.cumulated_steps = 0.0
        self.last_time = time.time()

        # define rewards
        self.reward_min_dist_to_goal = 5.0
        self.reward_max_dist_to_goal = -5.0
        self.reward_crashing = -1000.0
        self.reward_colliding = -3000.0
        self.reward_time_taken = -1.0
        self.reward_goal_reached = 10000.0
        self.reward_robots_crossed = 3.0
        self.moving_backward = -100.0
        self.crossed = False
        self.dec_obs = 5
        # normalized min_range for detecting collision
        self.min_range = (0.35 - self.min_laser_value) / (self.max_laser_value - self.min_laser_value)
        self.success = False #did both the robots cross over safely ?
        self.tom_success = False
        self.tom_crashed = False
        self.jerry_crashed = False
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(HallwayCollisionAvoidance, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """

        #rospy.logwarn("_set_init_pose")
        self.init_linear_forward_speed = 0
        self.init_linear_turn_speed = 0
        self.move_base(self.init_linear_forward_speed,
                       self.init_linear_turn_speed,
                       epsilon=0.05,
                       update_rate=10,
                       min_laser_distance=-1)
        self.tom_move_base(self.init_linear_forward_speed,
                       self.init_linear_turn_speed,
                       epsilon=0.05,
                       update_rate=10,
                       min_laser_distance=-1)
        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes

        #rospy.logwarn("_init_env_variables")
        self.cumulated_reward = 0.0
        self.tom_cumulated_reward = 0.0

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)

        # TODO: Add reset of published filtered laser readings
        tom_laser_scan = self.get_tom_laser_scan()
        jerry_laser_scan = self.get_jerry_laser_scan()
        self.last_time = time.time()

        # restart tom
        stop_command = Twist()
        stop_command = Twist()
        stop_command.linear.x = 0.5
        self._tom_cmd_vel_pub.publish(stop_command)
        self._jerry_cmd_vel_pub.publish(stop_command)
        # wrt the robot's initial position as origin
        self.jerry_last_position = 0.0
        self.tom_last_position = 0.0

        # reset success reward signal
        self.success = False
        self.tom_success = False
    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        """
        # #rospy.logwarn("_set_action")
        # rospy.logwarn("Start Set Action ==>" + str(action))
        # # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        # np.clip(action,-1.0,1.0 , out = action)
        # linear_speed_tom = 0
        # linear_speed_jerry = 0

        #---------CONTINUOUS ACTION SPACE--------------
        # if -0.5<= action[0]<= 0.5:  # FORWARD
        #     linear_speed_tom = 0.5
        #     angular_speed_tom = 0.0
        #     self.last_action_tom = "TOM_FORWARDS"
        # elif 0.5<action[0]<=1.0:  # LEFT
        #     linear_speed_tom = 0.0
        #     angular_speed_tom = 0.5
        #     self.last_action_tom = "TOM_LEFT"
        # elif -1.0 <= action[0] < -0.5 :  # RIGHT
        #     linear_speed_tom = 0.0
        #     angular_speed_tom = -0.5
        #     self.last_action_tom = "TOM_RIGHT"


        # if -0.5<= action[1]<= 0.5:  # FORWARD
        #     linear_speed_jerry = 0.5
        #     angular_speed_jerry = 0.0
        #     self.last_action_jerry = "JERRY_FORWARDS"
        # elif 0.5<action[1]<=1.0:  # LEFT
        #     linear_speed_jerry = 0.0
        #     angular_speed_jerry = 0.5
        #     self.last_action_jerry = "JERRY_LEFT"
        # elif -1.0 <= action[1] < -0.5 :  # RIGHT
        #     linear_speed_jerry = 0.0
        #     angular_speed_jerry = -0.5
        #     self.last_action_jerry = "JERRY_RIGHT"


        #---------DISCRETE ACTION SPACE--------------
        if action == 0:  # TOM FORWARD, JERRY FORWARD
            linear_speed_tom = 0.5
            angular_speed_tom = 0.0
            linear_speed_jerry = 0.5
            angular_speed_jerry = 0.0
            self.last_action_tom = "TOM_FORWARDS"
            self.last_action_jerry = "JERRY_FORWARDS"
        elif action == 1:  # TOM FORWARD, JERRY LEFT
            linear_speed_tom = 0.5
            angular_speed_tom = 0.0
            linear_speed_jerry = 0.0
            angular_speed_jerry = 0.5
            self.last_action_tom = "TOM_FORWARDS"
            self.last_action_jerry = "JERRY_LEFT"
        elif action == 2:  # TOM FORWARD, JERRY RIGHT
            linear_speed_tom = 0.5
            angular_speed_tom = 0.0
            linear_speed_jerry = 0.0
            angular_speed_jerry = -0.5
            self.last_action_tom = "TOM_FORWARDS"
            self.last_action_jerry = "JERRY_RIGHT"
        elif action == 3:  # TOM LEFT, JERRY FOWARD
            linear_speed_tom = 0.0
            angular_speed_tom = 0.5
            linear_speed_jerry = 0.5
            angular_speed_jerry = 0.0
            self.last_action_tom = "TOM_LEFT"
            self.last_action_jerry = "JERRY_FORWARDS"
        elif action == 4:   # TOM LEFT, JERRY LEFT
            linear_speed_tom = 0.0
            angular_speed_tom = 0.5
            linear_speed_jerry = 0.0
            angular_speed_jerry = 0.5
            self.last_action_tom = "TOM_LEFT"
            self.last_action_jerry = "JERRY_LEFT"
        elif action == 5:   # TOM LEFT, JERRY RIGHT
            linear_speed_tom = 0.0
            angular_speed_tom = 0.5
            linear_speed_jerry = 0.0
            angular_speed_jerry = -0.5
            self.last_action_tom = "TOM_LEFT"
            self.last_action_jerry = "JERRY_RIGHT"
        elif action == 6:  # TOM RIGHT, JERRY FORWARD
            linear_speed_tom = 0.0
            angular_speed_tom = -0.5
            linear_speed_jerry = 0.5
            angular_speed_jerry = 0.0
            self.last_action_tom = "TOM_RIGHT"
            self.last_action_jerry = "JERRY_FORWARDS"
        elif action == 7:  # TOM RIGHT, JERRY LEFT
            linear_speed_tom = 0.0
            angular_speed_tom = -0.5
            linear_speed_jerry = 0.0
            angular_speed_jerry = 0.5
            self.last_action_tom = "TOM_RIGHT"
            self.last_action_jerry = "JERRY_LEFT"
        elif action == 8:  # TOM RIGHT, JERRY RIGHT
            linear_speed_tom = 0.0
            angular_speed_tom = -0.5
            linear_speed_jerry = 0.0
            angular_speed_jerry = -0.5
            self.last_action_tom = "TOM_RIGHT"
            self.last_action_jerry = "JERRY_RIGHT"
        #elif action == 3: # BACKWARD
        #     linear_speed = -1.0
        #     angular_speed = 0.0
        #     self.last_action = "BACKWARDS"
        # elif action == 3: # STOP
        #     linear_speed = 0.0
        #     angular_speed = 0.0
        #     self.last_action = "STOP"

        # We tell Segbot the linear and angular speed to set to execute
        self.move_base(linear_speed_jerry,
                       angular_speed_jerry,
                       epsilon=0.05,
                       update_rate=10,
                       min_laser_distance=self.min_range)
        self.tom_move_base(linear_speed_tom,
                       angular_speed_tom,
                       epsilon=0.05,
                       update_rate=10,
                       min_laser_distance=self.min_range)

        rospy.logdebug("END Set Jerry Action ==>" + str(action) + ", NAME=" + str(self.last_action_jerry))
        rospy.logdebug("END Set Tom Action ==>" + str(action) + ", NAME=" + str(self.last_action_tom))

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        #rospy.logwarn("_get_obs")
        #rospy.logwarn("Start Get Jerry Observation ==>")
        # We get the laser scan data
        jerry_laser_scan = self.get_jerry_laser_scan()

        discretized_observations = self.discretize_observation(jerry_laser_scan,"jerry")

        tom_laser_scan = self.get_tom_laser_scan()
        tom_discretized_observations = self.discretize_observation(tom_laser_scan,"tom")


        jerry_odom = self.get_jerry_odom()
        tom_odom = self.get_tom_odom()

        jerry_odom_x_y_theta = [jerry_odom.pose.pose.position.x, jerry_odom.pose.pose.position.y, jerry_odom.pose.pose.orientation.z]
        tom_odom_x_y_theta = [tom_odom.pose.pose.position.x, tom_odom.pose.pose.position.y, tom_odom.pose.pose.orientation.z]
        
        normalized_jerry_odom = ( 2 * (jerry_odom_x_y_theta - self.odom_low)/(self.odom_high-self.odom_low)) - 1
        normalized_tom_odom = ( 2 * (tom_odom_x_y_theta - self.odom_low)/(self.odom_high-self.odom_low)) - 1

        
        #both tom and jerry success
        if normalized_jerry_odom[0] <= -1.0 and normalized_tom_odom[0] >=1.0:
            self._episode_done = True
            self.success = True
            self.tom_success = True

        # Tom crosses hallway assuming jerry is stuck
        elif normalized_tom_odom[0] >=1.0:
            self.tom_success = True
            self.success = False
            stop_command = Twist()
            stop_command.linear.x=0
            self._tom_cmd_vel_pub.publish(stop_command)
        #Jerry crosses hallway assuming tom is stuck
        elif normalized_jerry_odom[0] <=-1.0:
            self.tom_success = False
            self.success = True
            stop_command = Twist()
            stop_command.linear.x=0
            self._jerry_cmd_vel_pub.publish(stop_command)

        # Jerry leaves hallway assign negative reward to jerry
        if normalized_jerry_odom[0] >=1.0:
            self._episode_done = True

        # Tom leave hallway assign negative reward to tom
        if normalized_tom_odom[0] <=-1.0:
            self._episode_done = True

        # add odometry information to observation
        discretized_observations.append(jerry_odom.pose.pose.position.x)
        discretized_observations.append(jerry_odom.pose.pose.position.y)
        discretized_observations.append(jerry_odom.pose.pose.orientation.z)


        tom_discretized_observations.append(tom_odom.pose.pose.position.x)
        tom_discretized_observations.append(tom_odom.pose.pose.position.y)
        tom_discretized_observations.append(tom_odom.pose.pose.orientation.z)

        jerry_kinect_rgb = self._get_jerry_kinect_rgb()
        jerry_kinect_rgb = ros_numpy.numpify(jerry_kinect_rgb)
        jerry_kinect_rgb = cv2.resize(jerry_kinect_rgb, dsize=(128, 64), interpolation=cv2.INTER_CUBIC)



        jerry_kinect_rgb_normalized = cv2.normalize(jerry_kinect_rgb, None, alpha=0, beta=1,
                                                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        jerry_kinect_depth = np.expand_dims(jerry_kinect_rgb,axis=2)
        return jerry_kinect_rgb


    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        #rospy.logwarn("_is_done")
        if self._episode_done:
            rospy.logdebug("Segbot is Too Close to wall==>"+str(self._episode_done))
        else:
            rospy.logerr("Segbot is Ok ==>")

        return self._episode_done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        
        jerry_odom = self.get_jerry_odom()
        tom_odom = self.get_tom_odom()

        jerry_odom_x_y_theta = [jerry_odom.pose.pose.position.x, jerry_odom.pose.pose.position.y, jerry_odom.pose.pose.orientation.z]
        tom_odom_x_y_theta = [tom_odom.pose.pose.position.x, tom_odom.pose.pose.position.y, tom_odom.pose.pose.orientation.z]
        
        normalized_jerry_odom = ( 2 * (jerry_odom_x_y_theta - self.odom_low)/(self.odom_high-self.odom_low)) - 1
        normalized_tom_odom = ( 2 * (tom_odom_x_y_theta - self.odom_low)/(self.odom_high-self.odom_low)) - 1

        jerry_current_position = 1.0 - normalized_jerry_odom  #self.get_jerry_odom().pose.pose.position.x
        tom_current_position = -1.0 - normalized_tom_odom  #self.get_tom_odom().pose.pose.position.x
        tom_distance_moved_towards_target = tom_current_position-self.tom_last_position
        jerry_distance_moved_towards_target = jerry_current_position-self.jerry_last_position

        j_x=normalized_jerry_odom[0]
        j_y=normalized_jerry_odom[1]
        t_x=normalized_tom_odom[0]
        t_y=normalized_tom_odom[1]

        self.distance_between_tom_jerry = math.sqrt((t_x-j_x)**2+(t_y-j_y)**2) 

        tom_reward = 0
        jerry_reward = 0
        if not done:
            #reward based on distance
            #Tom moved forward towards goal
            if tom_distance_moved_towards_target[0]<=0 and self.last_action_tom == 'TOM_FORWARDS':
                tom_reward = self.reward_min_dist_to_goal
                self.tom_last_position = tom_current_position
            #Tom moves left or right (to avoid circling)
            elif tom_distance_moved_towards_target[0]<=0 and self.last_action_tom !="TOM_FORWARDS":
                tom_reward = self.reward_max_dist_to_goal
            #Tom moves in opposite direction
            elif tom_distance_moved_towards_target[0] > 0: 
                tom_reward = self.moving_backward
                # tom_reward = self.reward_max_dist_to_goal

            #Jerry moved forward towards goal
            if jerry_distance_moved_towards_target[0]>=0 and self.last_action_jerry == 'JERRY_FORWARDS':
                jerry_reward = self.reward_min_dist_to_goal
                self.jerry_last_position = jerry_current_position
            #Jerry moves left or right (to avoid circling)
            elif jerry_distance_moved_towards_target[0]>=0 and self.last_action_jerry !="JERRY_FORWARDS":
                jerry_reward = self.reward_max_dist_to_goal
            #Jerry moves in opposite direction
            elif jerry_distance_moved_towards_target[0] < 0:
                jerry_reward = self.moving_backward                
                # jerry_reward = self.reward_max_dist_to_goal

            #Jerry and Tom cross each other
            if tom_odom_x_y_theta[0] > jerry_odom_x_y_theta[0] and not self.crossed:
                print("crossed")
                tom_reward = self.reward_robots_crossed
                jerry_reward = self.reward_robots_crossed
                self.crossed = True

        else:
            #Jerry and Tom reach goal without colliding
            if self.success and self.tom_success:
                tom_reward = self.reward_goal_reached
                jerry_reward = self.reward_goal_reached
            #Jerry and Tom crashed
            elif self.tom_crashed and self.jerry_crashed:
                tom_reward = self.reward_colliding
                jerry_reward = self.reward_colliding
            #Tom crashed
            elif self.tom_crashed and not self.tom_success:
                tom_reward = self.reward_crashing 
                # jerry_reward = 0
                jerry_reward = abs(1.0-normalized_jerry_odom[0])
            #Jerry crashed
            elif self.jerry_crashed and not self.success:
                jerry_reward = self.reward_crashing
                # tom_reward = 0
                tom_reward = abs(-1.0-normalized_tom_odom[0])
        #print('Tom reward', tom_reward)
        #print('Jerry reward', jerry_reward)

        # penalize for time taken
        tom_reward+=self.reward_time_taken
        jerry_reward+=self.reward_time_taken
        
        return tom_reward, jerry_reward

    # Internal TaskEnv Methods
    def discretize_observation(self, data, robot_name, new_ranges=1):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """

        discretized_ranges = []
        filtered_range = []
        mod = new_ranges
        max_laser_value = data.range_max
        min_laser_value = data.range_min


        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                if item == float('Inf') or np.isinf(item) or item>max_laser_value:
                    discretized_ranges.append(round(max_laser_value, self.dec_obs))
                    item_val = 1.0
                elif np.isnan(item):
                    discretized_ranges.append(round(min_laser_value, self.dec_obs))
                    item_val = -1.0
                else:
                    item_val = (item - self.min_laser_value) / (self.max_laser_value - self.min_laser_value)                   
                    discretized_ranges.append(round(item_val, self.dec_obs))

                if (self.min_range > item_val > -1.0):
                    self._episode_done = True
                    if self.distance_between_tom_jerry<0.1:
                        rospy.logdebug("Both collided")
                        self.jerry_crashed = True
                        self.tom_crashed = True
                        self.success=False
                        self.tom_success=False
                    elif robot_name == "jerry":
                        rospy.logdebug("Jerry alone crashed")
                        self.jerry_crashed = True
                        self.tom_crashed = False
                        self.success = False
                    elif robot_name == "tom":
                        rospy.logdebug("Tom alone crashed")
                        self.tom_crashed = True
                        self.jerry_crashed = False
                        self.tom_success = False
                else:
                    pass
                # We add last value appended
                filtered_range.append(discretized_ranges[-1])
            else:
                # We add value zero
                filtered_range.append(0.1)

        rospy.logdebug("Size of observations, discretized_ranges==>" + str(len(discretized_ranges)))

        return discretized_ranges
