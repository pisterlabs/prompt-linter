import rospy
import numpy as np
import time
from math import pi
from gym import spaces
from openai_ros.robot_envs import mycobot_env
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os


class MyCobotWorldEnv(mycobot_env.MyCobotEnv):
    def __init__(self):
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/mycobot/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="mycobot_moveit",
                    launch_file_name="mycobot_moveit_gazebo.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/mycobot/config",
                               yaml_file_name="mycobot_world.yaml")
        
        rospy.logerr('SLEEP IN 30 SECONDS...')
        time.sleep(30)
        rospy.logerr('END SLEEP')

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(MyCobotWorldEnv, self).__init__(ros_ws_abspath)

        # Actions
        self.init_angle_joint1 = rospy.get_param('/mycobot/init_angle_joint1')
        self.init_angle_joint2 = rospy.get_param('/mycobot/init_angle_joint2')
        self.init_angle_joint3 = rospy.get_param('/mycobot/init_angle_joint3')
        self.init_angle_joint4 = rospy.get_param('/mycobot/init_angle_joint4')
        self.init_angle_joint5 = rospy.get_param('/mycobot/init_angle_joint5')
        self.init_angle_joint6 = rospy.get_param('/mycobot/init_angle_joint6')
        self.init_angles = np.array([self.init_angle_joint1, self.init_angle_joint2,
                                     self.init_angle_joint3, self.init_angle_joint4,
                                     self.init_angle_joint5, self.init_angle_joint6])

        # Only variable needed to be set here
        self.n_actions = rospy.get_param('/mycobot/n_actions')
        self.pos_lim = rospy.get_param('/mycobot/position_lim')
        self.ori_lim = np.deg2rad(rospy.get_param('/mycobot/orientation_lim'))
        a_low = np.array([-self.pos_lim, -self.pos_lim, -self.pos_lim, -self.ori_lim, -self.ori_lim, -self.ori_lim])
        a_high = np.array([self.pos_lim, self.pos_lim, self.pos_lim, self.ori_lim, self.ori_lim, self.ori_lim])
        self.action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float64)
        
        o_low = np.full(6, -pi)
        o_high = np.full(6, pi)
        self.observation_space = spaces.Box(low=o_low, high=o_high, dtype=np.float64)
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.reward_range = (-np.inf, np.inf)

        self.n_steps = rospy.get_param("/mycobot/n_steps")
        self.now_step = 0
        self.cumulated_steps = 0


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_angle_goal(self.init_angles)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        self.now_step = 0
        self.info = {}
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self._success = False


    def _set_action(self, action):
        """
        This set action will move the mycobot's end-effector to a pose offset from the
        current pose by the given 'action'.
        :param action: The offset to target pose.
        """

        rospy.logdebug("Start Set Action ==> "+str(action))
        pose_goal = self.get_pose() + action
        self.move_pose_goal(pose_goal)

        rospy.logdebug("END Set Action ==>")

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        MyCobotEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        rgb_img = self.get_rgb_img()
        depth_img = self.get_depth_img()[:, :, np.newaxis]
        imgs = np.concatenate([rgb_img, depth_img], axis=2)
        angles = self.get_angles()
        obs_dict = {'image': imgs, 'angles': angles}
        
        rospy.logdebug("END Get Observation ==>")
        return obs_dict

    def _is_done(self, observations):
        # update this function
        self._success = self.is_success()

        if self.now_step >= self.n_steps - 1:
            self._episode_done = True
        
        return self._success, self._episode_done, self.info
    
    def is_success(self):
        # update this function
        return False

    def _compute_reward(self, observations, success, done):
        # update this function
        reward = 1

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.now_step += 1
        rospy.logdebug("Now_step=" + str(self.now_step))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward
    
    def step(self, action):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, success, done, info
        """

        rospy.logdebug("START STEP OpenAIROS")

        self.gazebo.unpauseSim()
        self._set_action(action)
        self.gazebo.pauseSim()
        obs = self._get_obs()
        success, episode_done, info = self._is_done(obs)
        reward = self._compute_reward(obs, success, episode_done)
        self.cumulated_episode_reward += reward

        rospy.logdebug("END STEP OpenAIROS")

        return obs, reward, success, episode_done, info
