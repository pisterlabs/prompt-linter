from gym import spaces
from gym.envs.registration import register
from openai_ros.robot_envs import naorobot_env
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
import numpy as np
import rospy
from transformations import euler_from_quaternion

import time

# The path is __init__.py of openai_ros, where we import the MovingCubeOneDiskWalkEnv directly
timestep_limit_per_episode = 1000  # Can be any Value

register(
    id='NaoRobotWalk-v0',
    entry_point='openai_ros:task_envs.naorobot.naorobot_walk.NaoRobotWalkEnv',
    timestep_limit=timestep_limit_per_episode,
)


class NaoRobotWalkEnv(naorobot_env.NaoRobotEnv):
    def __init__(self):

        # This is the most common case of Box observation type
        self.obs_low = np.array(
            [-0.671952, -2.08567, -2.08567, -0.314159, -2.08567, -1.32645, -1.54462, -2.08567, 0.0349066, -2.08567, 0,
             0, -1.82387, -1.82387, -1.53589, -0.379435, -1.14529, -1.53589, -0.79046, -1.14529, -0.0923279, -0.0923279,
             -1.18944, -0.397761, -1.1863, -0.768992,-np.inf,-np.inf,-np.inf,-np.inf])
        self.obs_high = np.array(
            [0.514872, 2.08567, 2.08567, 1.32645, 2.08567, 0.314159, -0.0349066, 2.08567, 1.54462, 2.08567, 1, 1,
             1.82387, 1.82387, 0.48398, 0.79046, 0.740718, 0.48398, 0.379435, 0.740718, 2.11255, 2.11255, 0.922581,
             0.768992, 0.932006, 0.397761,np.inf,np.inf,np.inf,np.inf])

        self.action_low = np.array(
            [-0.671952, -2.08567, -2.08567, -0.314159, -2.08567, -1.32645, -1.54462, -2.08567, 0.0349066, -2.08567, 0,
             0, -1.82387, -1.82387, -1.53589, -0.379435, -1.14529, -1.53589, -0.79046, -1.14529, -0.0923279, -0.0923279,
             -1.18944, -0.397761, -1.1863, -0.768992])
        self.action_high = np.array(
            [0.514872, 2.08567, 2.08567, 1.32645, 2.08567, 0.314159, -0.0349066, 2.08567, 1.54462, 2.08567, 1, 1,
             1.82387, 1.82387, 0.48398, 0.79046, 0.740718, 0.48398, 0.379435, 0.740718, 2.11255, 2.11255, 0.922581,
             0.768992, 0.932006, 0.397761])

        # action space is just the joints below torso - hip, legs and feet
        self.action_space = spaces.Box(low=self.action_low[-12:],high=self.action_high[-12:],dtype=np.float64)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float64)
        self.reward_range = (-np.inf, np.inf)
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(NaoRobotWalkEnv, self).__init__()

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self.previous_pos = 0

        rospy.Subscriber('gazebo/link_states', LinkStates, queue_size=10, callback=self.reward_func)
        rospy.Subscriber('/joint_states', JointState, queue_size=10,callback=self.set_observation)

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.previous_pos=0
        self.init_action = [0.0001040219922128216, -0.0008609154246554951,-0.0010163444196260585,0.00010881459694456197,-0.03489872574628894,
                            -0.0011038560387692797, -0.00012842790996892006,-8.213071259177696e-05,-0.0010169716930903405,-0.0010586932796581294,
                            0.0006866347072733703,-0.0011433401964513479]
        self._move_robot(self.init_action)
        return True

    def set_observation(self,data):
        self.joint_angles = list(data.position)
        self.joint_angles = np.array(self.joint_angles,dtype=np.float64)

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)

    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        """
        # input : action : numpy array
        # Move the robot according to the action received.
        self._move_robot(action)

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        obs = np.append(self.joint_angles,self.orientation_list)
        return obs

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        if self._episode_done:
            rospy.logdebug("NaoRobot is about to fall down==>"+str(self._episode_done))
        else:
            rospy.logerr("NaoRobot is Ok ==>")

        return self._episode_done


    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        return self.reward

    # Internal TaskEnv Methods

    def reward_func(self,data):
        if len(data.pose)>1:
            ori = data.pose[1].orientation
            orientation_list = [ori.x,ori.y,ori.z,ori.w]
            self.orientation_list = np.array(orientation_list,dtype=np.float64)
            euler = euler_from_quaternion(orientation_list)
            self.reward = -10
            # negative reward for falling over
            if abs(euler[1]*57.2958)>30:
                self.reward = -100000
                self._episode_done = True
            else :
                # reward for moving forward
                pos = data.pose[1].position
                if pos.x-self.previous_pos>0:
                    self.reward=100
                    print('moved forward..')
                self.previous_pos = max(self.previous_pos,pos.x)
