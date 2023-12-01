#!/usr/bin/env python

import os
import math
import numpy as np
import rospy
from openai_ros.robot_envs import ur3_env
from gym import spaces
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher

class UR3PositionEnv(ur3_env.UR3Env):

    """ Task enviroment to train Reinforcement Learning algorithms
    on the UR3 robot using ROS. """

    def __init__(self):

        ros_ws_abspath = rospy.get_param("/ur3/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath \
         in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), \
        "The Simulation ROS Workspace path " + ros_ws_abspath + \
        " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
        "/src;cd " + ros_ws_abspath + ";catkin_make"

        # Start ROS launch that creates the world where the robot lives
        ROSLauncher(rospackage_name="my_ur3_description",
                    launch_file_name="start_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load params from YAML file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/ur3/config",
                               yaml_file_name="ur3_position_task.yaml")

        # Get parameters from YAML file
        self.get_params()
        # Define action space
        self.action_space = spaces.Discrete(self.n_actions)
        # Define observation space
        high = np.array([np.pi, np.pi, np.pi])
        low = np.array([-np.pi, -np.pi, -np.pi])
        self.observation_space = spaces.Box(low, high)

        # Add init functions prior to starting the enviroment
        super(UR3PositionEnv, self).__init__(ros_ws_abspath=ros_ws_abspath)


    def get_params(self):

        """ Gets configuration parameters from YAML file.
        Additionally creates a new parameter to track the current
        iteration the robot is at within an episode.
        :return:
        """

        self.n_actions = rospy.get_param('/ur3/n_actions')
        self.n_observations = rospy.get_param('/ur3/n_observations')
        self.max_iterations = rospy.get_param('/ur3/max_iterations')
        self.init_pos = rospy.get_param('/ur3/init_pos')
        self.goal_pos = rospy.get_param('/ur3/goal_pos')
        self.pos_step = rospy.get_param('/ur3/position_delta')
        self.current_iteration = 0

    def _set_action(self, action):

        """ Maps action identifiers into actual
         movements of the robot.
         :return:
         """

        if action == 0:
            rospy.loginfo("SHOULDER PAN => +")
            self.pos['shoulder_pan_joint'] += self.pos_step
            self.move_shoulder_pan_joint(self.pos)

        elif action == 1:
            rospy.loginfo("SHOULDER PAN => -")
            self.pos['shoulder_pan_joint'] -= self.pos_step
            self.move_shoulder_pan_joint(self.pos)

        elif action == 2:
            rospy.loginfo("SHOULDER LIFT => +")
            self.pos['shoulder_lift_joint'] += self.pos_step
            self.move_shoulder_lift_joint(self.pos)

        elif action == 3:
            rospy.loginfo("SHOULDER LIFT => -")
            self.pos['shoulder_lift_joint'] -= self.pos_step
            self.move_shoulder_lift_joint(self.pos)

        elif action == 4:
            rospy.loginfo("ELBOW => +")
            self.pos['elbow_joint'] += self.pos_step
            self.move_elbow_joint(self.pos)

        elif action == 5:
            rospy.loginfo("ELBOW => -")
            self.pos['elbow_joint'] -= self.pos_step
            self.move_elbow_joint(self.pos)

    def _get_obs(self):

        """ Stores the current position of the three moving joints
        in a numpy array.
        :return:obs
         """

        obs = np.array([self.joints['shoulder_pan_joint'],
                        self.joints['shoulder_lift_joint'],
                        self.joints['elbow_joint']
                        ])

        return obs

    def _is_done(self, observations):

        """" The episode is done when the robot achieves the desired pose
        with an absolute tolerance of 0.2 per joint.
        :return:done
        """

        done = False
        tolerance = 0.2

        if abs(observations[0] - self.goal_pos['shoulder_pan_joint']) <= tolerance and \
           abs(observations[1] - self.goal_pos['shoulder_lift_joint']) <= tolerance and \
           abs(observations[2] - self.goal_pos['elbow_joint']) <= tolerance:
            done =  True
            rospy.loginfo("Joints achieved goal position")
        elif self.current_iteration == (self.max_iterations -1):
            # Return done at the end of an episode
            rospy.loginfo("Ending episode...")
            done = True
            self.current_iteration = 0

        return done

    def _compute_reward(self, observations, done):

        """
        Gives more points for staying closer to the goal position.
        A fixed reward of 100 is given when the robot achieves the
        goal position.
        :return:reward
        """

        goal_array = np.array(list(self.goal_pos.values()))

        if not done:
            # If not done, then compute reward
            reward = 1 / np.sqrt(np.sum((observations - goal_array)**2))
        elif done and self.current_iteration == 0:
            # If done due to be the last episode (we just reseted it in _is_done())
            # then compute reward
            reward = 1 / np.sqrt(np.sum((observations - goal_array)**2))
        elif done and self.current_iteration > 0:
            # If done at an iteration greater than 0, then it must has reached the
            # goal position with an absolute error of 0.02
            reward = 100
        else:
            rospy.logdebug("Unknown goal status => Reward?")
            reward = 0

        self.current_iteration += 1

        return reward

    def _init_env_variables(self):

        """
        This variable needs to be implemented but
        it's not used, hence it just passes.
        :return:
        """

        pass

    def _set_init_pose(self):

        """
        Sets joints to initial position [0,0,0,0,0,0]
        :return:
        """

        rospy.logdebug('Checking publishers connection')
        self.check_publishers_connection()

        rospy.logdebug('Reseting to initial robot position')
        # Reset internal position variable
        self.init_internal_vars(self.init_pos)
        # Move joints to origin
        self.move_shoulder_pan_joint(self.init_pos)
        self.move_shoulder_lift_joint(self.init_pos)
        self.move_elbow_joint(self.init_pos)
        self.move_wrist_1_joint(self.init_pos)
        self.move_wrist_2_joint(self.init_pos)
        self.move_wrist_3_joint(self.init_pos)
