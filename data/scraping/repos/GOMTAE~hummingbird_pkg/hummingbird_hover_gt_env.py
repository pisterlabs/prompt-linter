#! /usr/bin/env python

import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
from mav_msgs.msg import Actuators
from nav_msgs.msg import Odometry



class HummingbirdHoverGtEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all HummingbirdHoverGtEnv environments.
    """

    def __init__(self):
        """
        Initializes a new HummingbirdHoverGtEnv environment.
        The Sensors: The sensors accessible are the ones considered useful for AI learning.

        Sensor Topic List:
        * /hummingbird/ground_truth/imu: IMU of the drone giving acceleration and orientation relative to world.
        * /hummingbird/ground_truth/odometry: ODOM
        * /hummingbird/motor_speed: Sensing the motor speed

        Actuators Topic List:
        * /hummingbird/command/motor_speed: Command the motor speed

        Args:
        """
        rospy.logdebug("Start HummingbirdHoverGtEnv INIT...")

        # Variables that we give through the constructor.
        # None in this case

        # Internal Vars
        # Doesnt have any accessible controllers
        self.controllers_list = []

        self.robot_name_space = "hummingbird"

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(HummingbirdHoverGtEnv, self).__init__(controllers_list=self.controllers_list,
                                                  robot_name_space=self.robot_name_space,
                                                  reset_controls=False,
                                                  start_init_physics_parameters=False,
                                                  reset_world_or_sim="WORLD")


        """
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason.
        2) If the simulation was running already for some reason, we need to reset the controllers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to reset to make it work properly.
        """

        self.gazebo.unpauseSim()

        # self.controllers_object.reset_controllers() # Maybe we don't need this
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        # Switch GT <---> Realistic
        rospy.Subscriber("/hummingbird/ground_truth/imu", Imu, self._imu_callback, queue_size=100)
        # rospy.Subscriber("/hummingbird/imu", Imu, self._imu_callback, queue_size=100)

        rospy.Subscriber("/hummingbird/ground_truth/odometry", Odometry, self._odom_callback , queue_size=100)
        rospy.Subscriber("/hummingbird/motor_speed", Actuators, self._actuators_callback, queue_size=100)

        self._cmd_motor_speed_pub = rospy.Publisher('/hummingbird/command/motor_speed', Actuators, queue_size=100)
        # self._hummingbird_position_pub = rospy.Publisher('/hummingbird/command/motor_speed', JointState, queue_size=1)

        self._check_publishers_connection()
        self.gazebo.pauseSim()

        rospy.logdebug("Finished HummingbirdHoverEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    # HummingbirdHoverGtEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        self._check_imu_ready()
        self._check_odom_ready()
        self._check_actuators_ready() # actuator but here
        rospy.logdebug("ALL SENSORS & ACTUATOR READY")

    def _check_imu_ready(self):
        self.imu = None
        rospy.logdebug("Waiting for /hummingbird/ground_truth/imu to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message(
                    "/hummingbird/ground_truth/imu", Imu, timeout=1.0)
                rospy.logdebug("Current /hummingbird/ground_truth/imu READY=>" + str(self.imu))

            except:
                rospy.logerr(
                    "Current /hummingbird/ground_truth/imu not ready yet, retrying for getting imu")

        return self.imu

    def _check_odom_ready(self):
        self.odom = None
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/hummingbird/ground_truth/odometry", Odometry, timeout=1.0)
                rospy.logdebug("Current /hummingbird/ground_truth/odometry READY=>" + str(self.odom))

            except:
                rospy.logerr("Current /hummingbird/ground_truth/odometry not ready yet, retrying for getting odom")

        return self.odom

    def _check_actuators_ready(self):
        self.actuators = None
        while self.actuators is None and not rospy.is_shutdown():
            try:
                self.actuators = rospy.wait_for_message("/hummingbird/motor_speed", Actuators, timeout=1.0)
                rospy.logdebug("Current /hummingbird/motor_speed READY=>" + str(self.actuators))

            except:
                rospy.logerr("Current /hummingbird/motor_speed not ready yet, retrying for getting odom")

        return self.actuators

    def _imu_callback(self, data):
        self.imu = data

    def _odom_callback(self, data):
        self.odom = data

    def _actuators_callback(self, data):
        self.actuators = data

    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_motor_speed_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No subscribers to _cmd_motor_speed_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_motor_speed_pub Publisher Connected")

        # while self._hummingbird_position_pub.get_num_connections() == 0 and not rospy.is_shutdown():
        #     rospy.logdebug("No subscribers to _hummingbird_position_pub yet so we wait and try again")
        #     try:
        #         rate.sleep()
        #     except rospy.ROSInterruptException:
        #         # This is to avoid error when world is rested, time when backwards.
        #         pass
        # rospy.logdebug("_hummingbird_position_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")

    # Methods that the TaskEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TaskEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TaskEnvironment will need.
    # ----------------------------
    def move_motor(self, motor_speed):
        motor_speed_value = Actuators()
        motor_speed_value.angular_velocities = motor_speed # As list or tuple
        rospy.logdebug("Motor Velocity>>" + str(motor_speed_value))
        self._cmd_motor_speed_pub.publish(motor_speed_value)
        # self.wait_until_motor_is_in_vel(motor_speed_value.angular_velocities)

    # def init_hummingbird_pose(self, init_pose):
    #     init_pose_value = JointState()
    #     init_pose_value.position = init_pose
    #     rospy.logdebug("Init pose>>" + str(init_pose_value))
    #     self._hummingbird_position_pub.publish(init_pose_value)

    def get_imu(self):
        return self.imu

    def get_odom(self):
        return self.odom

    def get_actuators(self):
        return self.actuators

    # def reinit_sensors(self):
    #     """
    #     This method is for the tasks so that when reseting the episode
    #     the sensors values are forced to be updated with the real data and
    #
    #     """