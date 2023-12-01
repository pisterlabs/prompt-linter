from collections import namedtuple
import os
import sys

import cv2
import numpy as np

# pylint: disable=import-error
import rospy
from gym import spaces

from openai_ros.openai_ros_common import ROSLauncher
from openai_ros.robot_envs import wamv_env
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.task_envs.wamv import utils
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion

sys.path.append(os.path.expandvars('${SIMULATION_DIR}/bachelor_thesis/simulation_ws/src/'))
from wamv_openai_ros2.scripts.enums import Dir
from wamv_openai_ros2.scripts.util import make_dir


class WamvNavTwoSetsBuoysEnv(wamv_env.WamvEnv):

    Point = namedtuple('Point', ['x', 'y'])

    def __init__(self):
        self.ros_ws_abspath = None
        self.rospackage_name = None
        self.launch_file_name = None

        self.propeller_high_speed = 0
        self.propeller_low_speed = 0
        self.max_angular_speed = 0

        self.work_space_x_max = 0
        self.work_space_x_min = 0
        self.work_space_y_max = 0
        self.work_space_y_min = 0

        self.cumulated_reward = 0
        self.done_reward = 0
        self.closer_to_point_reward = 0
        self.checkpoint_reward = 0

        self.desired_point_epsilon = 0
        self.dec_obs = 0
        self.current_position_cntr = 0
        self.cumulated_steps = 0

        self.last_chkpt = (self.Point(np.inf, np.inf),)
        self.desired_point = Point()

        self.is_beyond_track = False

        self.buoys = []
        self.route = []
        self.n_game = 0

        self._load_config()

        if not self.ros_ws_abspath:
            raise ValueError('ros_abspath in the yaml config file not set.')

        if not os.path.exists(os.path.expandvars(self.ros_ws_abspath)):
            raise FileNotFoundError('The Simulation ROS Workspace path'
                                   f'{self.ros_ws_abspath} does not exist')

        ROSLauncher(
            rospackage_name=self.rospackage_name,
            launch_file_name=self.launch_file_name,
            ros_ws_abspath=self.ros_ws_abspath
        )

        LoadYamlFileParamsTest(
            rospackage_name="openai_ros",
            rel_path_from_package_to_file="src/openai_ros/task_envs/wamv/config",
            yaml_file_name=f'wamv_nav_{rospy.get_param("/wamv/environment_type")}.yaml'
        )

        super().__init__()
        self._lazy_load_config()

        rospy.logdebug(f'Start {type(self).__name__} INIT...')
        self.action_space = spaces.Discrete(rospy.get_param('/wamv/n_actions'))
        self.reward_range = (-np.inf, np.inf)

        #Get Desired Point to Get
        self._load_point('desired_point')
        self._load_buoys()

        high = np.array([
            np.ones((720, 1280, 3))*255,
            np.ones((720, 1280, 3))*255
        ])
        low = np.array([
            np.zeros((720, 1280, 3)),
            np.zeros((720, 1280, 3))
        ])
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug(f'END {type(self).__name__} INIT...')

    @make_dir(Dir.TRACK.value)
    def _load_buoys(self):
        n = int(rospy.get_param('/buoys/n'))
        reds = np.zeros((n, 2), dtype=int)
        greens = np.zeros((n, 2), dtype=int)
        for i in range(n):
            x_red = rospy.get_param(f'/buoys/red_{n-i}/x')
            x_green = rospy.get_param(f'/buoys/green_{n-i}/x')
            y_red = rospy.get_param(f'/buoys/red_{n-i}/y')
            y_green = rospy.get_param(f'/buoys/green_{n-i}/y')
            reds[i] = np.array([x_red, y_red])
            greens[i] = np.array([x_green, y_green])
            red = self.Point(x_red, y_red)
            green = self.Point(x_green, y_green)
            self.buoys.append((green, red))
        np.save(f'{Dir.TRACK.value}reds.npy', reds)
        np.save(f'{Dir.TRACK.value}greens.npy', greens)

    def _load_config(self):
        self.ros_ws_abspath = os.path.expandvars(rospy.get_param('/wamv/ros_ws_abspath', None))
        self.rospackage_name = rospy.get_param('/wamv/rospackage_name', None)
        self.launch_file_name = rospy.get_param('/wamv/launch_file_name', None)


    def _lazy_load_config(self):
        self.propeller_high_speed = rospy.get_param('/wamv/propeller_high_speed')
        self.propeller_low_speed = rospy.get_param('/wamv/propeller_low_speed')
        self.max_angular_speed = rospy.get_param('/wamv/max_angular_speed')

        self.work_space_x_max = rospy.get_param("/wamv/work_space/x_max")
        self.work_space_x_min = rospy.get_param("/wamv/work_space/x_min")
        self.work_space_y_max = rospy.get_param("/wamv/work_space/y_max")
        self.work_space_y_min = rospy.get_param("/wamv/work_space/y_min")

        self.desired_point_epsilon = rospy.get_param("/wamv/desired_point_epsilon")
        self.done_reward = rospy.get_param("/wamv/done_reward")
        self.closer_to_point_reward = rospy.get_param("/wamv/closer_to_point_reward")
        self.checkpoint_reward = rospy.get_param('/wamv/checkpoint_reward')

        self.dec_obs = rospy.get_param("/wamv/number_decimals_precision_obs")

    def _load_point(self, name):
        getattr(self, name).x = rospy.get_param(f'/wamv/{name}/x')
        getattr(self, name).y = rospy.get_param(f'/wamv/{name}/y')
        getattr(self, name).z = rospy.get_param(f'/wamv/{name}/z')


    def _set_init_pose(self):
        right_propeller_speed = 0.0
        left_propeller_speed = 0.0
        self.set_propellers_speed(
            right_propeller_speed,
            left_propeller_speed,
            time_sleep=0.25
        )

        return True


    def _init_env_variables(self):
        self.cumulated_reward = 0.0
        # We get the initial pose to mesure the distance from the desired point.
        odom = self.odom()
        current_position = Vector3()
        current_position.x = odom.pose.pose.position.x
        current_position.y = odom.pose.pose.position.y


    def _set_action(self, action):
        """
        It sets the joints of wamv based on the action integer given
        based on the action number given.
        :param action: The action integer that sets what movement to do next.
        """

        rospy.logdebug(f'Start Set Action {action}')

        right_propeller_speed = 0.0
        left_propeller_speed = 0.0

        if action == utils.Actions.FORWARD.value: # Go Forwards
            right_propeller_speed = self.propeller_high_speed
            left_propeller_speed = self.propeller_high_speed
        elif action == utils.Actions.BACKWARD.value: # Go BackWards
            right_propeller_speed = -1*self.propeller_high_speed
            left_propeller_speed = -1*self.propeller_high_speed
        elif action == utils.Actions.LEFT.value: # Turn Left
            right_propeller_speed = self.propeller_high_speed
            left_propeller_speed = -1*self.propeller_high_speed
        elif action == utils.Actions.RIGHT.value: # Turn Right
            right_propeller_speed = -1*self.propeller_high_speed
            left_propeller_speed = self.propeller_high_speed
        elif action == utils.Actions.RIGHT45.value:
            right_propeller_speed = -0.5*self.propeller_high_speed
            left_propeller_speed = 0.5*self.propeller_high_speed
        elif action == utils.Actions.LEFT45.value:
            right_propeller_speed = 0.5*self.propeller_high_speed
            left_propeller_speed = -0.5*self.propeller_high_speed
        elif action == utils.Actions.LEFT30.value:
            right_propeller_speed = 0.3*self.propeller_high_speed
            left_propeller_speed = -0.3*self.propeller_high_speed
        elif action == utils.Actions.RIGHT30.value:
            right_propeller_speed = -0.3*self.propeller_high_speed
            left_propeller_speed = 0.3*self.propeller_high_speed
        else:
            raise ValueError(f'Invalid action: {action}')

        self.set_propellers_speed(
            right_propeller_speed,
            left_propeller_speed,
            time_sleep=0.25
        )

        rospy.logdebug(f'END Set Action {action}')

    def _get_obs(self):
        rospy.logdebug('Start Get Observation')

        def _set_mask(color):
            lower, upper = color
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            return cv2.inRange(image, lower, upper)

        odom = self.odom()
        image_right = self.image_right()
        image_left = self.image_left()

        base_position = odom.pose.pose.position
        observation = []
        observation.append(round(base_position.x, self.dec_obs))
        observation.append(round(base_position.y, self.dec_obs))

        image_right = image_right[:int(np.floor(image_right.shape[0]*2/3)), ...]
        image_left = image_right[:int(np.floor(image_left.shape[0]*2/3)), ...]

        image = np.concatenate((
            image_left[:, :image_left.shape[1]//2],
            image_right[:, image_right.shape[1]//2:]
        ), axis=1)

        mask_green = _set_mask(utils.ColorBoundaries.GREEN)
        mask_red = _set_mask(utils.ColorBoundaries.RED)

        image_red = cv2.bitwise_and(image, image, mask=mask_red)
        image_green = cv2.bitwise_and(image, image, mask=mask_green)

        image = image_red + image_green
        image = cv2.resize(image, (64, 80), interpolation=cv2.INTER_AREA)

        image = np.array(image, dtype=np.int8).reshape(3, image.shape[0], image.shape[1])
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        observation.append(image)

        return image


    def _is_done(self, _):

        current_position = Vector3()
        current_position.x, current_position.y = self._get_current_pos()

        is_inside_corridor = self.is_inside_workspace(current_position)
        has_reached_des_point = self.is_in_desired_position(current_position)

        done = not is_inside_corridor or\
               has_reached_des_point or\
               not self.is_in_track()

        if done:
            self._load_buoys()
            self.is_beyond_track = False
            self.last_chkpt = (self.Point(np.inf, np.inf),)
            self.n_game += 1
            self.route = []
        np.save(f'{Dir.TRACK.value}route_{self.n_game}.npy', np.array(self.route))

        return done


    def is_in_track(self):
        current_position = Point()
        current_position.x, current_position.y = self._get_current_pos()
        if current_position.x - 3 > self.last_chkpt[0].x\
            or self.is_beyond_track:
            return False
        return True


    def _compute_reward(self, _, done):
        current_position = Point()
        current_position.x, current_position.y = self._get_current_pos()
        reward = 0

        if not done:
            try:
                chkpt_green, chkpt_red = self.buoys[-1]
                if chkpt_green.x - self.desired_point_epsilon <=\
                            current_position.x <= chkpt_red.x + self.desired_point_epsilon:
                    self.last_chkpt = self.buoys.pop()
                    if chkpt_green.y + 0.5 <=\
                        current_position.y <= chkpt_red.y - 0.5:
                        rospy.loginfo(f'===== CHECKPOINT CROSSED at '
                                    f'{current_position.x} '
                                    f'{current_position.y} =====')
                        reward += self.checkpoint_reward
                    else:
                        self.is_beyond_track = True
            except IndexError:
                pass

        else:
            if self.is_in_desired_position(current_position):
                reward = self.done_reward
                rospy.loginfo('='*60)
                rospy.loginfo(f'DESIRED POSITION x: {current_position.x} y: {current_position.y}')
                rospy.loginfo('='*60)
            else:
                reward = -self.done_reward

        rospy.logdebug(f'reward={reward}')
        self.cumulated_reward += reward
        rospy.logdebug(f'Cumulated_reward={self.cumulated_reward}')
        self.cumulated_steps += 1
        rospy.logdebug(f'Cumulated_steps={self.cumulated_steps}')

        return reward


    def is_in_desired_position(self, current_position):
        is_in_desired_pos = False

        x_current = current_position.x
        y_current = current_position.y

        if self.desired_point.x + self.desired_point_epsilon>= x_current >=\
            self.desired_point.x - self.desired_point_epsilon\
            and self.desired_point.y - 9 <= y_current <= self.desired_point.y + 9:
            is_in_desired_pos = True

        return is_in_desired_pos


    def _get_current_pos(self):
        odom = self.odom()
        base_position = odom.pose.pose.position
        x = round(base_position.x, self.dec_obs)
        y = round(base_position.y, self.dec_obs)
        self.route.append([x, y])
        return x, y


    def get_distance_from_desired_point(self, current_position):

        distance = self.get_distance_from_point(
            current_position,
            self.desired_point
        )

        return distance


    def get_distance_from_point(self, pstart, p_end):
        a = np.array((pstart.x, pstart.y, pstart.z))
        b = np.array((p_end.x, p_end.y, p_end.z))
        distance = np.linalg.norm(a - b)

        return distance


    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw


    def is_inside_workspace(self, current_position):
        is_inside = False

        current_position_str = str(current_position)
        current_position_str = ' '.join(current_position_str.split())
        if self.current_position_cntr %3 == 0:
            rospy.loginfo(f'Current position: {current_position_str}')
            self.current_position_cntr = 0
        self.current_position_cntr += 1

        if current_position.x > self.work_space_x_min and\
            current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and\
                current_position.y <= self.work_space_y_max:
                is_inside = True

        return is_inside
