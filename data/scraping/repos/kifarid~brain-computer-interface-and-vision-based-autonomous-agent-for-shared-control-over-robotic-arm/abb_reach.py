from gym import utils
import time
import rospy
from gym import spaces
# from openai_ros.robot_envs import fetch_env
from abb_python.src import abb_rob_env
from gym.envs.registration import register
import numpy as np
from sensor_msgs.msg import JointState
from abb_catkin.srv import EePose, EePoseRequest, EeRpy, EeRpyRequest, EeTraj, EeTrajRequest, JointTraj, \
    JointTrajRequest

register(
    id='ABBReach-v0',
    # entry_point='openai_ros:ABBReachEnv',
    entry_point='abb_python.src.abb_reach:ABBReachEnv',
    timestep_limit=500,
)


class ABBReachEnv(abb_rob_env.Abbenv, utils.EzPickle):
    def __init__(self):

        print("Entered Reach Env")

        self.get_params()

        # intializing robot env
        abb_rob_env.Abbenv.__init__(self)
        utils.EzPickle.__init__(self)
        # calling setup env.
        print("Call env setup")
        self._env_setup(initial_qpos=self.init_pos)

        print("Call get_obs")
        obs = self._get_obs()

        self.action_space = spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs[ 'achieved_goal' ].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs[ 'achieved_goal' ].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs[ 'observation' ].shape, dtype='float32'),
        ))
        print("Exit Reach Env")

    def get_params(self):
        # get configuration parameters
        """
        self.n_actions = rospy.get_param('/fetch/n_actions')
        self.has_object = rospy.get_param('/fetch/has_object')
        self.block_gripper = rospy.get_param('/fetch/block_gripper')
        self.n_substeps = rospy.get_param('/fetch/n_substeps')
        self.gripper_extra_height = rospy.get_param('/fetch/gripper_extra_height')
        self.target_in_the_air = rospy.get_param('/fetch/target_in_the_air')
        self.target_offset = rospy.get_param('/fetch/target_offset')
        self.obj_range = rospy.get_param('/fetch/obj_range')
        self.target_range = rospy.get_param('/fetch/target_range')
        self.distance_threshold = rospy.get_param('/fetch/distance_threshold')
        self.init_pos = rospy.get_param('/fetch/init_pos')
        self.reward_type = rospy.get_param('/fetch/reward_type')
        """
        self.n_actions = 5
        self.has_object = True
        self.block_gripper = False
        self.n_substeps = 20
        self.gripper_extra_height = 0.2
        self.target_in_the_air = True
        self.target_offset = 0.0
        self.obj_range = 0.15
        self.target_range = 0.15
        self.distance_threshold = 0.05
        self.reward_type = "sparse"
        self.init_pos = {
            'joint0': 0.0,
            'joint1': 0.0,
            'joint2': 0.0,
            'joint3': 0.0,
            'joint4': 0.0,
            'joint5': 0.0
        }

    def _set_action(self, action):

        # Take action
        assert action.shape == (6,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        # pos_ctrl, rot_ctrl ,gripper_ctrl = action[ :3 ], action[ 3 ], action[ 4 ]
        #
        # # pos_ctrl *= 0.05  # limit maximum change in position
        # rot_ctrl = [ 1., 0., 1., 0. ]  # fixed rotation of the end effector, expressed as a quaternion
        # gripper_ctrl = np.array([ gripper_ctrl, gripper_ctrl ])
        # assert gripper_ctrl.shape == (2,)
        # if self.block_gripper:
        #     gripper_ctrl = np.zeros_like(gripper_ctrl)
        # action = np.concatenate([ pos_ctrl, rot_ctrl, gripper_ctrl ])
        #
        # Apply action to simulation.
        self.set_trajectory_ee(action)

    def _get_obs(self):

        ###################################################################################################
        #getting the image for the current observation the image should be a numpy array constituted
        #depth 4 and each channel consists of the RGB and D of the image




        ###################################################################################################

        # getting the pose of the end effector the pose mainly consists of the end effector 3D translation
        # and rotation about the Z axis in addition to an indication of the aperature of the gripper and
        # command success

        grip_pose, grip_state = self.get_ee_pose()
        grip_pos_array = np.array([ grip_pose.pose.position.x, grip_pose.pose.position.y, grip_pose.pose.position.z])
        grip_rpy = self.get_ee_rpy()
        grip_rot_array = np.array([grip_rpy.z])

        # need to check wether to add success or if the gripper is opened or closed only
        self.gripper_success_only = True

        if self.gripper_success_only:
            gripper_state = np.array(grip_state[1]) #is task reached?
        else:
            gripper_state = np.array(grip_state[0]) #is gripper open?

        obs = np.concatenate([ grip_pos_array, grip_rot_array, gripper_state ])

        ###################################################################################################
        # getting the object poses while training from the simulator and sampling the achieved goals



        achieved_goal = self._sample_achieved_goal(grip_pos_array, object_pos)

        ###################################################################################################
        return {
            'observation': (image, obs.copy()),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _is_done(self, observations):

        d = self.goal_distance(observations[ 'achieved_goal' ], self.goal)

        return (d < self.distance_threshold).astype(np.float32)

    def _compute_reward(self, observations, done):

        d = self.goal_distance(observations[ 'achieved_goal' ], self.goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def compute_reward(self, achieved_goal, goal, info):

        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _init_env_variables(self):
        """
        Inits variables needed to be initialized each time we reset at the start
        of an episode.
        :return:
        """
        pass

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.gazebo.unpauseSim()
        self.set_trajectory_joints(self.init_pos)

        return True

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _sample_goal(self):
        #this function should be modified to be one of the objects pose in addition to any 3d position in the
        # vicinity with placing position 50% of the times in air

        #need to get target position range

        goal = self.initial_gripper_xpos[ :3 ] + self.np_random.uniform(-self.target_range, self.target_range,
                                                                         size=3)
        goal[ 2 ] = self.height_offset
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[ 2 ] += self.np_random.uniform(0, 0.45)

        # return goal.copy()
        return goal

    def _sample_achieved_goal(self, grip_pos_array, object_pos):


        # this should sample the changed position of any object
        achieved_goal = np.squeeze(object_pos.copy())

        # return achieved_goal.copy()
        return achieved_goal

    def _env_setup(self, initial_qpos):
        # called by intializing of task env in order to 1)unpause sim 2)go to initial position

        print("Init Pos:")
        print(initial_qpos)
        # for name, value in initial_qpos.items():
        #time.sleep(20)
        # called by intializing of task env
        self.gazebo.unpauseSim()
        self.set_trajectory_joints(initial_qpos)

        time.sleep(5)
        # Move end effector into position.

        gripper_target = np.array(
            [ 0.498, 0.005, 0.431 + self.gripper_extra_height ])  # + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([ 1., 0., 1., 0. ])
        action = np.concatenate([ gripper_target, gripper_rotation ])
        self.set_trajectory_ee(action)
        time.sleep(5)

        gripper_pos = self.get_ee_pose()
        gripper_pose_array = np.array(
            [ gripper_pos.pose.position.x, gripper_pos.pose.position.y, gripper_pos.pose.position.z ])
        self.initial_gripper_xpos = gripper_pose_array.copy()
        if self.has_object:
            #this needs to be adjusted to be the centeroid height of the object
            self.height_offset = self.sim.data.get_site_xpos('object0')[ 2 ]

        self.goal = self._sample_goal()
        self._get_obs()
