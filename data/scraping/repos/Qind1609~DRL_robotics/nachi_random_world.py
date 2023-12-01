from turtle import pos
import rospy
import numpy as np
import copy
from gym import spaces
from gym import utils
from openai_ros.robot_envs import nachi_env
from gym.envs.registration import register
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os


"""
def bound_angle(angle):

    bounded_angle = np.absolute(angle) % (2 * np.pi)
    if angle < 0:
        bounded_angle = -bounded_angle

    return bounded_angle
"""


class NachiRandomWorldEnv(nachi_env.NachiEnv, utils.EzPickle):
    def __init__(self, args=None):
        """
        Make robot learn how to reach a random position of cube
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/Nachi/ros_ws_abspath", None)
        assert (
            ros_ws_abspath is not None
        ), "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: 'YOUR/SIM_WS/PATH'"
        assert os.path.exists(ros_ws_abspath), (
            "The Simulation ROS Workspace path "
            + ros_ws_abspath
            + " DOESNT exist, execute: mkdir -p "
            + ros_ws_abspath
            + "/src;cd "
            + ros_ws_abspath
            + ";catkin_make"
        )

        ROSLauncher(
            rospackage_name="openai_hac",
            launch_file_name="start_random_world.launch",
            ros_ws_abspath=ros_ws_abspath,
        )

        rospy.logdebug("Start Nachi_Random_world INIT...")

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(
            rospackage_name="openai_ros",
            rel_path_from_package_to_file="src/openai_ros/task_envs/nachi/config",
            yaml_file_name="nachi_random_world_HER.yaml",
        )

        self.get_params()

        observations_high_range = np.array(
            [self.position_ee_x_max, self.position_ee_y_max, self.position_ee_z_max],
            dtype=np.float32,
        )
        observations_low_range = np.array(
            [self.position_ee_x_min, self.position_ee_y_min, self.position_ee_z_min],
            dtype=np.float32,
        )

        observations_high_dist = np.array([self.max_distance])
        observations_low_dist = np.array([0.0])

        high = np.concatenate([observations_high_range, observations_high_dist])
        low = np.concatenate([observations_low_range, observations_low_dist])

        # number of independent variables in all observation
        self.observation_space = spaces.Box(low, high)

        # dimesion of output action (number of neurons in output layer of Actor)
        max_action = np.array([1, 1, 1], dtype=np.float32)
        min_action = np.array([-1, -1, -1], dtype=np.float32)
        self.action_space = spaces.Box(min_action, max_action)

        # TaskEnv uses that use variables from the parent class.
        super(NachiRandomWorldEnv, self).__init__(ros_ws_abspath)

        self.reward_range = (-np.inf, np.inf)

    def get_params(self):
        """
        get configuration parameters

        """
        self.sim_time = rospy.get_time()
        # self.n_actions = rospy.get_param("/HER/n_actions")  # discrete
        self.n_observations = rospy.get_param("/HER/n_observations")
        self.position_ee_x_max = rospy.get_param("/HER/position_ee_x_max")
        self.position_ee_x_min = rospy.get_param("/HER/position_ee_x_min")
        self.position_ee_y_max = rospy.get_param("/HER/position_ee_y_max")
        self.position_ee_y_min = rospy.get_param("/HER/position_ee_y_min")
        self.position_ee_z_max = rospy.get_param("/HER/position_ee_z_max")
        self.position_ee_z_min = rospy.get_param("/HER/position_ee_z_min")

        self.init_pos = rospy.get_param("/HER/init_pos")
        self.setup_ee_pos = rospy.get_param("/HER/setup_ee_pos")
        self.goal_ee_pos = rospy.get_param("/HER/goal_ee_pos")

        self.position_delta = rospy.get_param("/HER/position_max_delta")
        # self.rotation_delta = rospy.get_param("/HER/rotation_delta")  # discrete
        self.step_punishment = rospy.get_param("/HER/step_punishment")

        self.outofrange_punishement = rospy.get_param("/HER/outofrange_punishement")
        self.reached_goal_reward = rospy.get_param("/HER/reached_goal_reward")

        """
        This defines the possible range of movement(action input) of EE.
        So xyz coordinates of action value will be clipped by the corresponding values below.

        I have empirically confirmed valid init pose as follows.
        x: not to hit the robot itself
        y: not to far from the table
        z: not to hit the table
        """

        work_space_dict = rospy.get_param("/HER/training_space")
        self._x_min = work_space_dict["x_min"]
        self._x_max = work_space_dict["x_max"]
        self._y_min = work_space_dict["y_min"]
        self._y_max = work_space_dict["y_max"]
        self._z_min = work_space_dict["z_min"]
        self._z_max = work_space_dict["z_max"]

        self.distance_threshold = rospy.get_param(
            "/HER/distance_threshold"
        )  # used to indicate if EE reaches the desired position


        self.max_distance = rospy.get_param(
            "/HER/max_distance"
        )  # used to indicate the failed (out of range case)

        self.goal = np.array(
            [self.goal_ee_pos["x"], self.goal_ee_pos["y"], self.goal_ee_pos["z"]]
        )
        self.rot_ctrl = np.array([1.000, 0.000, 0.000, 0.000])  # TODO

        self.prev_grip_pos = np.zeros(3)  # X,Y,Z of TCP
        self.prev_grip_rpy = np.zeros(4)  # quaternion rot of TCP

        self.prev_object_pos = np.zeros(3)
        self.prev_object_rot = np.zeros(4)

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.

        The simulation will be paused, therefore all the data retrieved has to be
        from a system that doesnt need the simulation running, like variables where the
        callbackas have stored last know sesnor data.
        :return:
        """

        x, y, z = self.random_position()  # create random position of cube
        self.goal = np.array([x, y, z])  # new goal

        self.obj_positions.reset_position(x, y, z)  # set cube to the position
   
    def _set_goal(self, x, y, z, rx, ry, rz, w):
        self.goal = np.array([x,y,z])
        self.obj_positions.reset_position(x,y,z)

    def _set_init_pose(self):
        """
        Sets the Robot in its init pose
        The Simulation will be unpaused for this purpose.
        """
        if not self.set_trajectory_joints(self.init_pos):
            assert False, "Initialisation is failed...."

    def remap(self, x, oMin, oMax, nMin, nMax):

        # range check
        if oMin == oMax:
            print("Warning: Zero input range")
            return None

        if nMin == nMax:
            print("Warning: Zero output range")
            return None

        # check reversed input range
        reverseInput = False
        oldMin = min(oMin, oMax)
        oldMax = max(oMin, oMax)
        if not oldMin == oMin:
            reverseInput = True

        # check reversed output range
        reverseOutput = False
        newMin = min(nMin, nMax)
        newMax = max(nMin, nMax)
        if not newMin == nMin:
            reverseOutput = True

        portion = (x - oldMin) * (newMax - newMin) / (oldMax - oldMin)
        if reverseInput:
            portion = (oldMax - x) * (newMax - newMin) / (oldMax - oldMin)

        result = portion + newMin
        if reverseOutput:
            result = newMax - portion

        return result

    def _set_action(self, action):

        pos_ctrl = action[:3]
        # scaling to [-0.001,0.001]
        pos_ctrl = self.remap(
            pos_ctrl, -1, 1, -self.position_delta, self.position_delta
        )
        """
        current_pos = np.array(
            [
                self.get_ee_pose().pose.position.x,
                self.get_ee_pose().pose.position.y,
                self.get_ee_pose().pose.position.z,
            ]
        )"""

        pos_ctrl = np.around(pos_ctrl, 4)
        """
        pos_ctrl = [
            np.clip(pos_ctrl[0], self.position_ee_x_min, self.position_ee_x_max),
            np.clip(pos_ctrl[1], self.position_ee_y_min, self.position_ee_y_max),
            np.clip(pos_ctrl[2], self.position_ee_z_min, self.position_ee_z_max),
        ]"""
        # TODO:
        rot_ctrl = self.rot_ctrl
        action = np.concatenate([pos_ctrl, rot_ctrl])
        done = self.set_trajectory_ee(action.tolist())
        return done

    def _get_obs(self):
        """
        It returns the Position of the TCP/EndEffector as observation.
        And the distance from the desired point
        Orientation for the moment is not considered

        Note:
            - In original code(https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch_env.py#L91),
              the term (xvelp) is used and it means positional velocity in world frame
        """
        grip_pos_v = self.get_ee_pose()
        grip_pos = np.array(
            [
                grip_pos_v.pose.position.x,
                grip_pos_v.pose.position.y,
                grip_pos_v.pose.position.z,
            ]
        )

        grip_rpy = np.array(
            [
                grip_pos_v.pose.orientation.x,
                grip_pos_v.pose.orientation.y,
                grip_pos_v.pose.orientation.z,
                grip_pos_v.pose.orientation.w,
            ]
        )
        dt = self.get_elapsed_time()

        # ESTIMATE #########         Velocity(position) = Distance/Time
        grip_velp = (grip_pos - self.prev_grip_pos) / dt  # Vgx Vgy Vgz
        grip_velr = (grip_rpy - self.prev_grip_rpy) / dt

        # the pose and rotation of the cube/box on a table

        object_data = self.obj_positions.get_states()

        object_pos = object_data[:3]  # pos of cube

        # rotation of cube
        object_rot = object_data[3:]

        object_velp = (
            object_pos - self.prev_object_pos
        ) / dt  # Velocity(position) = Distance/Time
        object_velr = (
            object_rot - self.prev_object_rot
        ) / dt  # Velocity(rotation) = Rotation/Time

        object_rel_pos = object_pos - grip_pos  # RELATE_DISTANCE ()
        object_rel_rot = object_rot - grip_rpy
        d = self.calculate_distance_between(object_pos, grip_pos)
        # Unknown meaning of this operation(https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch_env.py#L102)

        """
        object_velp -= grip_velp  #RELATE VEL [vgox, vgoy, vgoz]
        object_velr -= grip_velr
        
        gripper_state = self.get_joints().position[-1] #angle position of two finger tip
        
        gripper_vel = (gripper_state - self.prev_gripper_pos) / dt
        """
        achieved_goal = np.squeeze(grip_pos.copy())
        obs = np.concatenate(
            [
                grip_pos,  # absolute position of gripper
                [d],
                # grip_velp,  # absolute velocity position of gripper
                # grip_velr,  # absolute velocity rotation of gripper
                # object_pos.ravel(),  # absolute position of object (goal)
                object_rel_pos.ravel(),  # relative position of object from gripper
                # object_rot.ravel(),  # rotations of object
                # object_rel_rot.ravel(),  # relative rotation of object from gripper
                # object_velp.ravel(),  # positional velocities of object
                # object_velr.ravel()  # rotational velocities of object
                # [gripper_state],          # gripper state --> joint position of finger
                # [gripper_vel],            # velocities of finger joint
            ]
        )

        rospy.logdebug("OBSERVATIONS====>>>>>>>" + str(obs))

        # Update the previous properties
        self.prev_grip_pos = grip_pos
        self.prev_grip_rpy = grip_rpy
        self.prev_object_pos = object_pos
        self.prev_object_rot = object_rot

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),  # TODO: add rotation value
            "desired_goal": self.goal.copy(),  # TODO: add desired rot
        }

    def get_elapsed_time(self):
        """
        Returns the elapsed time since the beginning of the simulation
        Then maintains the current time as "previous time" to calculate the elapsed time again
        """
        current_time = rospy.get_time()
        dt = self.sim_time - current_time
        self.sim_time = current_time
        return dt

    def _is_done(self, observations):
        """
        In HER this is not used API, because we literally execute actions by fixed amount of time(e.g. 50 times)
        Yet, maybe needed in other situation so that we decided to preseve this here.
        """
        current_pos = observations["observation"][:3]

        done = self.calculate_if_done(observations['desired_goal'], current_pos)

        return done

    def _compute_reward(self, observations, done):
        """
        Given a success of the execution of the action
        Calculate the reward: binary => 0 for success, -1 for failure
        """
        current_pos = observations["observation"][:3]
        goal = observations["desired_goal"]
        if (
            (self.position_ee_x_min <= current_pos[0] <= self.position_ee_x_max)
            and (self.position_ee_y_min <= current_pos[1] <= self.position_ee_y_max)
            and (self.position_ee_z_min <= current_pos[2] <= self.position_ee_z_max)
        ):

            distance = self.calculate_distance_between(goal, current_pos)
            # Calculate the reward: 0 for success, -1 for failure
            if not done:  # not reach goal yet
                reward = distance  # try change this for improve performance
            else:  # reach goal
                reward = 0
            rospy.logwarn(">>>REWARD>>>" + str(reward))
        else:
            reward = self.outofrange_punishement

        return reward

    def calculate_if_done(self, goal, current_pos):
        """
        It calculated whather it has finished or not
        """
        done = False
        distance = self.calculate_distance_between(goal, current_pos)

        if distance < self.distance_threshold:
            done = True
            rospy.logwarn("Reached a Desired Position!")

        return done

    def calculate_distance_between(self, v1, v2):
        """
        Calculated the Euclidian distance between two vectors given as python lists.
        """
        dist = np.linalg.norm(np.array(v1) - np.array(v2))  # L2 norm
        return dist

    def random_position(self):
        """
        Randomly output xyz coordinates to initialise the desired position witnin an available region
        """
        x = np.random.uniform(low=self._x_min, high=self._x_max)
        y = np.random.uniform(low=self._y_min, high=self._y_max)
        z = np.random.uniform(low=self._z_min, high=self._z_max)
        rospy.logwarn("============================")
        rospy.logwarn("New Desired Pose: " + str([x, y, z]))
        rospy.logwarn("============================")
        return x, y, z
