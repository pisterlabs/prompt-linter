#!/usr/bin/env python
# ---------------------------------------------------
import numpy as np

from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray, Float32
from geometry_msgs.msg import Pose

import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from std_msgs.msg import Empty
from std_msgs.msg import Float64MultiArray

import roslib

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState

from gym import spaces
from openai_ros.robot_envs import parrotdrone_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler

import transforms
import random

timestep_limit_per_episode = 10000  # Can be any Value

register(
    id='DroneTest-v0',
    entry_point='droneTest:DroneTest',
    max_episode_steps=timestep_limit_per_episode,
)


def initial_linear_angular_velocity():
    t = Vector3()
    t.x = 0
    t.y = 0
    t.z = 0
    return t


#



def initial_random_pose(transformation):
    denormal = Point()
    denormal.x = random.uniform(transformation.workspace_x_min, transformation.workspace_x_max)
    denormal.y = random.uniform(transformation.workspace_y_min, transformation.workspace_y_max)
    denormal.z = random.uniform(transformation.workspace_z_min, transformation.workspace_z_max)
    pose = transformation.normalize_position(denormal)
    return pose


def initial_goal(transformation, goalArray):
    desired_points = []
    for (x, y, z) in goalArray:
        temp = Point()
        temp.x = x
        temp.y = y
        temp.z = z
        desired_points.append(transformation.normalize_position(temp))
    return desired_points


def initial_orientation(transformation):
    denormal_roll = 0
    denormal_pitch = 0
    denormal_yaw = 0
    normalized_roll = transformation.normalize_roll(denormal_roll)
    normalized_pitch = transformation.normalize_pitch(denormal_pitch)
    normalized_yaw = transformation.normalize_yaw(denormal_yaw)
    return [normalized_roll, normalized_pitch, normalized_yaw]


class DroneTest(robot_gazebo_env.RobotGazeboEnv):
    def __init__(self):
        """
        Make parrotdrone learn how to navigate to get to a point
        """
        # Get WorkSpace Cube Dimensions
        self.work_space_x_max = rospy.get_param("/drone/work_space/x_max")
        self.work_space_x_min = rospy.get_param("/drone/work_space/x_min")
        self.work_space_y_max = rospy.get_param("/drone/work_space/y_max")
        self.work_space_y_min = rospy.get_param("/drone/work_space/y_min")
        self.work_space_z_max = rospy.get_param("/drone/work_space/z_max")
        self.work_space_z_min = rospy.get_param("/drone/work_space/z_min")

        self.transformation = transforms.Transform(self.work_space_x_min, self.work_space_x_max, self.work_space_y_min,
                                                   self.work_space_y_max, self.work_space_z_min, self.work_space_z_max)
        # for finding if the quadcopter has flipped
        self.max_roll = self.transformation.normalize_roll(np.radians(rospy.get_param("/drone/max_roll")))
        self.max_pitch = self.transformation.normalize_pitch(np.radians(rospy.get_param("/drone/max_pitch")))
        self.max_yaw = self.transformation.normalize_yaw(np.radians(rospy.get_param("/drone/max_yaw")))

        self.flip_reward = rospy.get_param("/drone/flip_reward")
        self.outside_reward = rospy.get_param("/drone/outside_reward")

        # error acceptable for reaching a goal
        self.desired_point_epsilon = rospy.get_param("/drone/desired_point_epsilon")

        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(DroneTest, self).__init__(controllers_list=self.controllers_list,
                                        robot_name_space=self.robot_name_space,
                                        reset_controls=False,
                                        start_init_physics_parameters=False,
                                        reset_world_or_sim="WORLD")

        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1]), high=np.array([1, 1, 1, 1]))

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, 0)

        # We must read this from ros param server
        self.goal_array = [[0, 0, 3]]

        self.desired_points = initial_goal(self.transformation, self.goal_array)

        self.cumulated_steps = 0
        self.cumulated_reward = 0

        # this is the maximum power or speed of each propeler
        self.max_power = 100
        self.goal_index = 0
        self.current_goal = self.desired_points[self.goal_index]

        self.initial_pose = initial_random_pose(self.transformation)

        self.current_pose = self.initial_pose


        self.current_orientation = initial_orientation(self.transformation)

        self.current_linear_velocity = initial_linear_angular_velocity()

        self.current_angular_velocity = initial_linear_angular_velocity()

        high = numpy.array([1,  # current position
                            1,
                            1,
                            # orientation is normalized for better learning
                            1, 1, 1,  # current orientation(position in degree) roll pitch yaw

                            np.inf, np.inf, np.inf,  # linear velocity

                            np.inf, np.inf, np.inf,  # angular velocity

                            1,  # goal position
                            1,
                            1])

        low = numpy.array([-1,  # current position
                           -1,
                           -1,
                           # orientation is normalized for better learning
                           -1, -1, -1,  # current orientation(position in degree) roll pitch yaw

                           -np.inf, -np.inf, -np.inf,  # linear velocity

                           -np.inf, -np.inf, -np.inf,  # angular velocity

                           -1,  # goal position
                           -1,
                           -1])
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>" + str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" + str(self.observation_space))

        # self.sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, self.func_state)
        self.pub_command = rospy.Publisher("/Kwad/joint_motor_controller/command", Float64MultiArray, queue_size=1)
        self.pub_init_pos = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)

    # def func_state(self, msg):
    #     # print(msg)
    #     None


    def _reset_sim(self):
        """Resets a simulation
        """
        rospy.logdebug("RESET SIM START")
        if self.reset_controls :
            rospy.logdebug("RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self._set_init_pose()
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        else:
            rospy.logdebug("DONT RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self._set_init_pose()
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        rospy.logdebug("RESET SIM END")
        return True

    def _set_init_pose(self):  # done

        state_msg = ModelState()
        state_msg.model_name = "Kwad"

        state_msg.pose.position = self.transformation.denormalize_position(self.initial_pose)
        rospy.logwarn(f"initial pose = {state_msg.pose.position}")

        normal_orientation = initial_orientation(self.transformation)
        denormalized_roll = self.transformation.denormalize_roll(normal_orientation[0])
        denormalized_pitch = self.transformation.denormalize_pitch(normal_orientation[1])
        denormalized_yaw = self.transformation.denormalizing_yaw(normal_orientation[2])

        orientation = quaternion_from_euler(denormalized_roll, denormalized_pitch, denormalized_yaw)

        state_msg.pose.orientation.x = orientation[0]
        state_msg.pose.orientation.y = orientation[1]
        state_msg.pose.orientation.z = orientation[2]
        state_msg.pose.orientation.w = orientation[3]

        state_msg.twist.linear = initial_linear_angular_velocity()
        state_msg.twist.angular = initial_linear_angular_velocity()
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            rospy.logerr(f"status of set init pose: {resp}")
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def _check_all_systems_ready(self):  # done
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True

    def _get_obs(self):  # done
        """Returns the observation.
        """
        rospy.logdebug("Start Getting observation")
        odom = None
        try:
            model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            odom = model_coordinates('Kwad', '')
        except rospy.ServiceException as e:
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))
        rospy.logdebug("odom read finished" + str(odom))
        if odom is not None:
            denormal_pose = odom.pose.position
            normal_pose = self.transformation.normalize_position(denormal_pose)
            self.current_pose = normal_pose
            rospy.logdebug("current pose updated : " + str(self.current_pose))
            self.current_orientation = self.get_orientation_euler(odom.pose.orientation)
            self.current_linear_velocity = odom.twist.linear
            self.current_angular_velocity = odom.twist.angular
        else:
            rospy.logdebug("_get_obs did not work, get_model_state faced problem")
            raise Exception("_get_obs did not work, get_model_state faced problem")
        observation = self._create_observation()
        rospy.logdebug("End Getting observation")
        return observation

    def _create_observation(self):  # done
        """
        here I create the observation using the current position, orientation, linear and angular velocities and current
        goal position
        """
        current_pose = [self.current_pose.x, self.current_pose.y, self.current_pose.z]
        current_orientation = self.current_orientation
        current_linear_velocity = [self.current_linear_velocity.x, self.current_linear_velocity.y,
                                   self.current_linear_velocity.z]
        current_angular_velocity = [self.current_angular_velocity.x, self.current_angular_velocity.y,
                                    self.current_angular_velocity.z]
        current_goal = [self.current_goal.x, self.current_goal.y, self.current_goal.z]
        observation = current_pose + current_orientation + current_linear_velocity + current_angular_velocity + \
                      current_goal
        return observation

    def _init_env_variables(self):  # done
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """

        self.cumulated_reward = 0
        self.cumulated_steps = 0
        self.goal_index = 0
        self.desired_points = initial_goal(self.transformation, self.goal_array)
        self.initial_pose = initial_random_pose(self.transformation)
        self.current_pose = self.initial_pose
        self.current_orientation = initial_orientation(self.transformation)
        self.current_linear_velocity = initial_linear_angular_velocity()
        self.current_angular_velocity = initial_linear_angular_velocity()

    def _set_action(self, action):  # done
        """Applies the given action to the simulation.
        """
        rospy.logdebug("Start Set Action ==>" + str(action))
        command = Float64MultiArray()
        #   since the actions are between -1 and 1, but I need to make them to be between 0 and 1, I used this remapping
        #   then, I must send + - + - actions to the quadcopter because of the way it is simulated.

        plus_minus_numpy = np.array([1, -1, 1, -1])
        command_numpy = np.array(
            [((action[i] + 1) / 2) * self.max_power for i in range(len(action))]) * plus_minus_numpy

        command.data = command_numpy.tolist()

        self.pub_command.publish(command)
        rospy.logdebug("End Set Action ==>" + str(action))

    def _goal_updater(self):  # bayad ba tavajoh be inke gharare yek masir ra chanbar peimayesh konim ya na taghyir
        # daham
        self.goal_index += 1
        rospy.logerr("goal indexing:  " + str(self.goal_index) + str(self.desired_points))
        if self.goal_index == len(self.desired_points):
            rospy.logerr("goal finished")
            return True
        self.current_goal = self.desired_points[self.goal_index]
        return False

    def _is_done(self, observations):  # be joz goal updater albaghi tamam ast, goal updater alan faghad
        # baraye noghati ke dakhel desired points hastand kar mikonad, agar tamam shavad bayad reset konim
        """
        Indicates whether the episode is done ( the robot has fallen for example).
        The done can be done due to three reasons:
        1) It went outside the workspace
        2) It flipped due to a crash or something
        3) the list of goals which are some points to be followed are finished
        """

        goal_finished = False  # here I want to check if the sets of goals are finished or not.
        is_inside_workspace_now = self.is_inside_workspace(self.current_pose)
        drone_flipped = self.drone_has_flipped(self.current_orientation)
        has_reached_des_point = self.is_in_desired_position(self.current_pose, self.desired_point_epsilon)

        # rospy.logwarn(">>>>>> DONE RESULTS <<<<<")
        if not is_inside_workspace_now:
            rospy.logerr("is_inside_workspace_now=" + str(is_inside_workspace_now))
        else:
            rospy.logdebug("is_inside_workspace_now=" + str(is_inside_workspace_now))

        if drone_flipped:
            rospy.logerr("drone_flipped=" + str(drone_flipped))
        else:
            rospy.logdebug("drone_flipped=" + str(drone_flipped))

        if has_reached_des_point:
            rospy.logerr("has_reached_des_point=" + str(has_reached_des_point))
            goal_finished = self._goal_updater()
        else:
            rospy.logdebug("has_reached_des_point=" + str(has_reached_des_point))

        # We see if we are outside the Learning Space
        episode_done = (not is_inside_workspace_now) or drone_flipped or goal_finished

        if episode_done:
            rospy.logerr("episode_done====>" + str(episode_done))
            rospy.logwarn("cumulated reward=====>  " + str(self.cumulated_reward))
        else:
            rospy.logdebug("episode_done====>" + str(episode_done))

        return episode_done

    def is_in_desired_position(self, current_position, alpha=0.05):
        """
        It return True if the current position is similar to the desired poistion
        """

        epsilon = self.transformation.normalize_epsilon(alpha)
        # rospy.logwarn(f"alpha = {alpha}, epsilon.x = {epsilon.x}, epsilon.y = {epsilon.y}, epsilon.z = {epsilon.z}")
        is_in_desired_pos = False

        x_pos_plus = self.current_goal.x + epsilon.x
        x_pos_minus = self.current_goal.x - epsilon.x
        y_pos_plus = self.current_goal.y + epsilon.y
        y_pos_minus = self.current_goal.y - epsilon.y
        z_pos_plus = self.current_goal.z + epsilon.z
        z_pos_minus = self.current_goal.z - epsilon.z

        x_current = current_position.x
        y_current = current_position.y
        z_current = current_position.z

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)
        z_pos_are_close = (z_current <= z_pos_plus) and (z_current > z_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close and z_pos_are_close

        return is_in_desired_pos

    def is_inside_workspace(self, current_position):  # done
        """
        Check if the Drone is inside the Workspace defined
        """
        is_inside = False

        # rospy.logwarn("##### INSIDE WORK SPACE? #######")
        # rospy.logwarn("XYZ current_position" + str(current_position))
        # rospy.logwarn(
        #     "work_space_x_max" + str(self.work_space_x_max) + ",work_space_x_min=" + str(self.work_space_x_min))
        # rospy.logwarn(
        #     "work_space_y_max" + str(self.work_space_y_max) + ",work_space_y_min=" + str(self.work_space_y_min))
        # rospy.logwarn(
        #     "work_space_z_max" + str(self.work_space_z_max) + ",work_space_z_min=" + str(self.work_space_z_min))
        # rospy.logwarn("############")

        if current_position.x > -1 and current_position.x <= 1:
            if current_position.y > -1 and current_position.y <= 1:
                if current_position.z > -1 and current_position.z <= 1:
                    is_inside = True

        return is_inside

    def drone_has_flipped(self, current_orientation):  # this is the same code they had, idk if needs change or not.
        """
        Based on the orientation RPY given states if the drone has flipped
        """
        has_flipped = True

        # rospy.logwarn("#### HAS FLIPPED? ########")
        # rospy.logwarn("RPY current_orientation" + str(current_orientation))
        # rospy.logwarn("max_roll" + str(self.max_roll) + ",min_roll=" + str(-1 * self.max_roll))
        # rospy.logwarn("max_pitch" + str(self.max_pitch) + ",min_pitch=" + str(-1 * self.max_pitch))
        # rospy.logwarn("############")
        if -1 * self.max_roll < current_orientation[0] <= self.max_roll:
            if -1 * self.max_pitch < current_orientation[1] <= self.max_pitch:
                has_flipped = False
        return has_flipped

    def _compute_reward(self, observations, done):
        """
        Calculates the reward to give based on the observations given.
        """

        #  this function will work on finding the projection of current pose on the line including goal pose and
        #  initial pose
        projection = self.projection_on_line()
        distance_of_projection_to_current_pose = self.get_distance_from_point(projection, self.current_pose)
        rospy.logdebug("distance_of_projection_to_current_pose : " + str(distance_of_projection_to_current_pose))
        distance_of_projection_to_goal_pose = self.get_distance_from_point(projection, self.current_goal)
        rospy.logdebug("distance_of_projection_to_goal_pose : " + str(distance_of_projection_to_goal_pose))
        distance_of_initial_pose_to_goal_pose = self.get_distance_from_point(self.initial_pose, self.current_goal)
        rospy.logdebug("distance_of_initial_pose_to_goal_pose  : " + str(distance_of_initial_pose_to_goal_pose))

        rospy.logdebug("current pose : " + str(self.current_pose))
        rospy.logdebug("current goal : " + str(self.current_goal))
        rospy.logdebug("initial pose : " + str(self.initial_pose))
        # since the length of different segments(start = initial pose, end = goal pose) are not same, we must
        # normalize them in order to use them for reward function

        # reward is the negative of the sum of distances from current pose to the projection of current pose on the
        # line which includes initial pose and current goal devided by the distance of initial pose and current goal
        # to be normalized
        reward = -(distance_of_projection_to_goal_pose + distance_of_projection_to_current_pose) / \
                 distance_of_initial_pose_to_goal_pose
        rospy.logdebug("distance Reward : " + str(reward))
        if self.drone_has_flipped(self.current_orientation):
            reward = self.flip_reward
            rospy.logwarn("flip Reward : " + str(reward))
        if not self.is_inside_workspace(self.current_pose):
            reward = self.outside_reward
            rospy.logerr("outside Reward : " + str(reward))

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        # rospy.logwarn("Reward : " + str(reward))
        return reward

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_camera1_image_raw_ready()
        self._check_imu_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_imu_ready(self):
        self.imu = None
        rospy.logdebug("Waiting for /Kwad/imu_data to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message("/Kwad/imu_data", Imu, timeout=5.0)
                rospy.logdebug("Current /Kwad/imu_data READY=>")

            except:
                rospy.logerr("Current /Kwad/imu_data not ready yet, retrying for getting imu")

        return self.imu

    def _check_camera1_image_raw_ready(self):
        self.down_camera_rgb_image_raw = None
        rospy.logdebug("Waiting for /Kwad/Kwad/camera1/image_raw to be READY...")
        while self.down_camera_rgb_image_raw is None and not rospy.is_shutdown():
            try:
                self.down_camera_rgb_image_raw = rospy.wait_for_message("/Kwad/Kwad/camera1/image_raw", Image,
                                                                        timeout=5.0)
                rospy.logdebug("Current /Kwad/Kwad/camera1/image_raw READY=>")
            except:
                rospy.logerr(
                    "Current /Kwad/Kwad/camera1/image_raw not ready yet, retrying for getting camera1 raw img")
        return self.down_camera_rgb_image_raw

    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return [self.transformation.normalize_roll(roll), self.transformation.normalize_pitch(pitch),
                self.transformation.normalize_yaw(yaw)]

    def return_numpy_of_point(self, point):
        return np.array([point.x, point.y, point.z])

    def projection_on_line(self):
        """
        in this function I find the projection point of current pose of drone on the line of initial pose and goal pose
        """
        initial = self.return_numpy_of_point(self.initial_pose)
        current = self.return_numpy_of_point(self.current_pose)
        goal = self.return_numpy_of_point(self.current_goal)

        l2 = np.sum((initial - goal) ** 2)
        if l2 == 0:
            print('self.initial_pose and self.current_goal are the same points')

        # The line extending the segment is parameterized as self.initial_pose + t (self.current_goal - self.initial_pose).
        # The projection falls where t = [(self.current_pose-self.initial_pose) . (self.current_goal-self.initial_pose)] / |self.current_goal-self.initial_pose|^2

        # if you need the point to project on line extention connecting self.initial_pose and self.current_goal
        t = np.sum((current - initial) * (goal - initial)) / l2

        projection_numpy = initial + t * (goal - initial)
        projection = Point()
        projection.x = projection_numpy[0]
        projection.y = projection_numpy[1]
        projection.z = projection_numpy[2]
        return projection

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Point Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance
