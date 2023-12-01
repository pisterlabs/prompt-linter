#!/usr/bin/env python

import time
import copy
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from openai_ros import robot_gazebo_env
from openai_ros.openai_ros_common import ROSLauncher

class UR3Env(robot_gazebo_env.RobotGazeboEnv):

    """ Robot enviroment to train Reinforcement Learning algorithms
        on the UR3 robot using ROS. """

    def __init__(self, ros_ws_abspath):

        # Start ROS launch that spawns the robot into the world
        ROSLauncher(rospackage_name="my_ur3_description",
                    launch_file_name="put_robot_in_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Define ROS publishers
        self._shoulder_pan = rospy.Publisher(
            '/ur3/shoulder_pan_joint_position_controller/command',
            Float64, queue_size=1)
        self._shoulder_lift = rospy.Publisher(
            '/ur3/shoulder_lift_joint_position_controller/command',
            Float64, queue_size=1)
        self._elbow = rospy.Publisher(
            '/ur3/elbow_joint_position_controller/command',
            Float64, queue_size=1)
        self._wrist_1 = rospy.Publisher(
            '/ur3/wrist_1_joint_position_controller/command',
            Float64, queue_size=1)
        self._wrist_2 = rospy.Publisher(
            '/ur3/wrist_2_joint_position_controller/command',
            Float64, queue_size=1)
        self._wrist_3 = rospy.Publisher(
            '/ur3/wrist_3_joint_position_controller/command',
            Float64, queue_size=1)

        # Define ROS subscribers
        rospy.Subscriber("/ur3/joint_states",
                         JointState, self.joints_callback)

        # Define controllers list
        self.controllers_list = ['joint_state_controller',
                                 'shoulder_pan_joint_position_controller',
                                 'shoulder_lift_joint_position_controller',
                                 'elbow_joint_position_controller',
                                 'wrist_1_joint_position_controller',
                                 'wrist_2_joint_position_controller',
                                 'wrist_3_joint_position_controller'
                                ]

        # Define robot robot namespace
        self.robot_name_space = "ur3"
        # Set reset_controls class variable to true
        self.reset_controls = True
        # Set rate
        self.rate = rospy.Rate(10)

        # Add init functions prior to starting the enviroment
        super(UR3Env, self).__init__(controllers_list=self.controllers_list,
                                             robot_name_space=self.robot_name_space,
                                             reset_controls=self.reset_controls,
                                             start_init_physics_parameters=False,
                                             reset_world_or_sim="WORLD")

    def joints_callback(self, data):

        """ Creates a dictionary out of joint_states
        topic to store name of the joint and its
        position.
        :return:
        """

        self.joints = dict(zip(data.name, data.position))

    def _env_setup(self, initial_qpos):

        """ Sets up the enviroment reinitializing
        internal variables and pose of the robot.
        Additionally it checks if publishers and
        subscribers are ready.
        :return:
        """

        self.init_internal_vars(self.init_pos)
        self.set_init_pose()
        self.check_all_systems_ready()

    def init_internal_vars(self, init_pos_value):

        """ Overrides current dictionary of the
        robot's current position with the initial pose.
        :return:
        """

        self.pos = copy.deepcopy(init_pos_value)

    def check_publishers_connection(self):

        """
        Checks that all publishers are on and working.
        It sleeps if connection couldn't be established.
        :return:
        """

        while (self._shoulder_pan.get_num_connections() == 0 \
               and not rospy.is_shutdown()):
            rospy.logerr(
                "No susbribers to _shoulder_pan yet so we wait and try again")
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

        while (self._shoulder_lift.get_num_connections() == 0 \
               and not rospy.is_shutdown()):
            rospy.logdebug(
                "No susbribers to _shoulder_lift yet so we wait and try again")
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

        while (self._elbow.get_num_connections() == 0 \
               and not rospy.is_shutdown()):
            rospy.logdebug(
                "No susbribers to _elbow yet so we wait and try again")
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

        while (self._wrist_1.get_num_connections() == 0 \
               and not rospy.is_shutdown()):
            rospy.logdebug(
                "No susbribers to _wrist_1 yet so we wait and try again")
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

        while (self._wrist_2.get_num_connections() == 0 \
               and not rospy.is_shutdown()):
            rospy.logdebug(
                "No susbribers to _wrist_2 yet so we wait and try again")
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

        while (self._wrist_3.get_num_connections() == 0 \
               and not rospy.is_shutdown()):
            rospy.logdebug(
                "No susbribers to _wrist_3 yet so we wait and try again")
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

        rospy.logdebug("All publishers are ready")

    def _check_all_systems_ready(self, init=True):

        """ Checks connection with joint_states topic.
        It tries to connect until the topic does have data
        or execution is interrupted.
        :return:
         """

        self.base_position = None
        while self.base_position is None and not rospy.is_shutdown():
            try:
                self.base_position = rospy.wait_for_message(
                    "/ur3/joint_states", JointState, timeout=1.0)
                rospy.logdebug(
                    "Current ur3/joint_states READY=>" \
                    + str(self.base_position))
                if init:
                    # Check all joints are at the initial values (close to 0)
                    positions_ok = all(
                        abs(i) <= 1.0e-02 for i in self.base_position.position)
                    rospy.logdebug(
                    "Checking Init Values Ok=>" \
                     + str(positions_ok))
            except:
                rospy.logwarn("Current ur3/joint_states not ready yet, \
                               retrying for getting joint_states")

        rospy.logdebug("Connection to ur3/joint_states established")

    def wait_for_completion(self, joint, target, timeout):

        """ Tracks the execution of a robot's action. It waits
        until the robot succesfully reaches a desired position with 0.01
        tolerance or maximum waiting time is exceeded. If the robot
        didn't get to reach the desired position (it's stuck?) it continues
        rather than reseting the world/simulation or breaking the training loop
        (we expect the RL algorithm to eventually change this pose)
        :return:
        """

        start = time.time()
        elapsed = 0
        # The error has to be lower than the maximum and elapsed time
        # should not exceed the maximum waiting time
        while (abs(self.joints[joint] - target) > 0.01) \
               and not (elapsed > timeout):
            elapsed = time.time() - start
            self.rate.sleep()

        if elapsed > timeout:
            rospy.logwarn('Joint could not reach desired position.')

    def move_shoulder_pan_joint(self, joints_dict):

        """ Moves shoulder pan joint to a given position
        stored in a dictionary. Maximum waiting time until
        the robot achieves the desired pose is 3 seconds.
        :return:
        """

        joint_value = Float64()
        joint_value.data = joints_dict['shoulder_pan_joint']
        # Publish desired pose
        self._shoulder_pan.publish(joint_value)
        # Stop further execution until it reaches the pose
        self.wait_for_completion('shoulder_pan_joint', \
                    joints_dict['shoulder_pan_joint'], 3)

    def move_shoulder_lift_joint(self, joints_dict):

        """ Moves shoulder lift joint to a given position
        stored in a dictionary. Maximum waiting time until
        the robot achieves the desired pose is 3 seconds.
        :return:
        """

        joint_value = Float64()
        joint_value.data = joints_dict['shoulder_lift_joint']
        # Publish desired pose
        self._shoulder_lift.publish(joint_value)
        # Stop further execution until it reaches the pose
        self.wait_for_completion('shoulder_lift_joint', \
                    joints_dict['shoulder_lift_joint'], 3)

    def move_elbow_joint(self, joints_dict):

        """ Moves elbow joint to a given position
        stored in a dictionary. Maximum waiting time until
        the robot achieves the desired pose is 3 seconds.
        :return:
        """

        joint_value = Float64()
        joint_value.data = joints_dict['elbow_joint']
        # Publish desired pose
        self._elbow.publish(joint_value)
        # Stop further execution until it reaches the pose
        self.wait_for_completion('elbow_joint', \
                    joints_dict['elbow_joint'], 3)

    def move_wrist_1_joint(self, joints_dict):

        """ Moves wrist 1 joint to a given position
        stored in a dictionary. Maximum waiting time until
        the robot achieves the desired pose is 3 seconds.
        :return:
        """

        joint_value = Float64()
        joint_value.data = joints_dict['wrist_1_joint']
        # Publish desired pose
        self._wrist_1.publish(joint_value)
        # Stop further execution until it reaches the pose
        self.wait_for_completion('wrist_1_joint', \
                    joints_dict['wrist_1_joint'], 3)

    def move_wrist_2_joint(self, joints_dict):

        """ Moves wrist 2 joint to a given position
        stored in a dictionary. Maximum waiting time until
        the robot achieves the desired pose is 3 seconds.
        :return:
        """

        joint_value = Float64()
        joint_value.data = joints_dict['wrist_2_joint']
        # Publish desired pose
        self._wrist_2.publish(joint_value)
        # Stop further execution until it reaches the pose
        self.wait_for_completion('wrist_2_joint', \
                    joints_dict['wrist_2_joint'], 3)

    def move_wrist_3_joint(self, joints_dict):

        """ Moves wrist 3 joint to a given position
        stored in a dictionary. Maximum waiting time until
        the robot achieves the desired pose is 3 seconds.
        :return:
        """

        joint_value = Float64()
        joint_value.data = joints_dict['wrist_3_joint']
        # Publish desired pose
        self._wrist_3.publish(joint_value)
        # Stop further execution until it reaches the pose
        self.wait_for_completion('wrist_3_joint', \
                    joints_dict['wrist_3_joint'], 3)
