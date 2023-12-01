'''
LAST UPDATE: 2022.XX.XX

AUTHOR:     OPENAI_ROS
            Neset Unver Akmandor (NUA)
            Gary M. Lvov (GML)

E-MAIL: akmandor.n@northeastern.edu
        lvov.g@northeastern.edu

DESCRIPTION: TODO...

REFERENCES:
[1] 

'''

import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from openai_ros import robot_gazebo_env
#from openai_ros.openai_ros_common import ROSLauncher


class HusarionEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, robot_namespace="", initial_pose={}, data_folder_path="", velocity_control_msg=""):
        """
        Initializes a new HusarionEnv environment.
        Husarion doesnt use controller_manager, therefore we wont reset the
        controllers in the standard fashion. For the moment we wont reset them.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered usefull for AI learning.

        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot
        * /camera/depth/image_raw: 2d Depth image of the depth sensor.
        * /camera/depth/points: Pointcloud sensor readings
        * /camera/rgb/image_raw: RGB camera
        * /scan: Laser Readings

        Actuators Topic List: /cmd_vel,

        Args:
        """
        rospy.logdebug("ROSbot_env::__init__ -> START...")
        # Variables that we give through the constructor.
        # None in this case
        # We launch the ROSlaunch that spawns the robot into the world
        """ ROSLauncher(rospackage_name="rosbot_gazebo",
                    launch_file_name="put_robot_in_world.launch",
                    ros_ws_abspath=ros_ws_abspath)
        rospy.logerr(">>>>>>>>>>>ROSLAUCHER DONE HusarionEnv INIT...")"""
        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = ["imu"]
        self.robot_namespace = robot_namespace

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(HusarionEnv, self).__init__(controllers_list=self.controllers_list,
                                          robot_namespace=self.robot_namespace,
                                          reset_controls=False,
                                          start_init_physics_parameters=False,
                                          initial_pose=self.initial_pose)

        self.gazebo.unpauseSim()
        # self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/" + str(self.robot_namespace) + "/odom", Odometry, self._odom_callback)
        rospy.Subscriber("/" + str(self.robot_namespace) + "/imu", Imu, self._imu_callback)
        rospy.Subscriber("/" + str(self.robot_namespace) + "/scan", LaserScan, self._laser_scan_callback)
        #rospy.Subscriber("/camera/depth/image_raw", Image, self._camera_depth_image_raw_callback)
        #rospy.Subscriber("/camera/depth/points", PointCloud2, self._camera_depth_points_callback)
        #rospy.Subscriber("/camera/rgb/image_raw", Image, self._camera_rgb_image_raw_callback)

        if velocity_control_msg:
            self._cmd_vel_pub = rospy.Publisher(velocity_control_msg, Twist, queue_size=1)

        else:    
            self._cmd_vel_pub = rospy.Publisher("/" + str(self.robot_namespace) + '/cmd_vel', Twist, queue_size=1)

        print("ROSbot_env::__init__ -> BEFORE velocity_control_msg: " + velocity_control_msg )
        print("ROSbot_env::__init__ -> BEFORE " + "/" + str(self.robot_namespace) + '/cmd_vel')
        self._check_publishers_connection()

        print("ROSbot_env::__init__ -> AFTER")

        self.gazebo.pauseSim()

        rospy.logdebug("ROSbot_env::__init__ -> END")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True

    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_odom_ready()
        self._check_imu_ready()
        self._check_laser_scan_ready()
        # We dont need to check for the moment, takes too long
        #self._check_camera_depth_image_raw_ready()
        #self._check_camera_depth_points_ready()
        #self._check_camera_rgb_image_raw_ready()
        rospy.logdebug("ROSBot_env::_check_all_sensors_ready -> END")

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("ROSbot_env::_check_odom_ready -> Waiting for /" + str(self.robot_namespace) + "/odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/" + str(self.robot_namespace) + "/odom", Odometry, timeout=1.0)
                rospy.logdebug("ROSbot_env::_check_odom_ready -> Current /" + str(self.robot_namespace) + "/odom READY!")

            except:
                rospy.logerr("ROSbot_env::_check_odom_ready -> Current /" + str(self.robot_namespace) + "/odom not ready yet, retrying...")

        return self.odom

    def _check_imu_ready(self):

        self.imu = None
        rospy.logdebug("ROSBot::_check_imu_ready -> Waiting for /" + str(self.robot_namespace) + "/imu to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message("/" + str(self.robot_namespace) + "/imu", Imu, timeout=1.0)
                rospy.logdebug("ROSBot::_check_imu_ready -> Current /" + str(self.robot_namespace) + "/imu READY!")

            except:
                rospy.logerr("ROSBot::_check_imu_ready -> Current /" + str(self.robot_namespace) + "/imu not ready yet, retrying...")

        return self.imu

    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.logdebug("ROSbot::_check_laser_scan_ready -> Waiting for /" + str(self.robot_namespace) + "/scan to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message("/" + str(self.robot_namespace) + "/scan", LaserScan, timeout=1.0)
                rospy.logdebug("ROSbot::_check_laser_scan_ready -> Current /" + str(self.robot_namespace) + "/scan READY!")

            except:
                rospy.logerr("ROSbot::_check_laser_scan_ready -> Current /" + str(self.robot_namespace) + "/scan not ready yet, retrying...")
        return self.laser_scan

    def _check_camera_depth_image_raw_ready(self):
        self.camera_depth_image_raw = None
        rospy.logdebug("ROSbot_env::_check_camera_depth_image_raw_ready -> Waiting for /" + str(self.robot_namespace) + "/camera/depth/image_raw to be READY...")
        while self.camera_depth_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_depth_image_raw = rospy.wait_for_message("/" + str(self.robot_namespace) + "/camera/depth/image_raw", Image, timeout=5.0)
                rospy.logdebug("ROSbot_env::_check_camera_depth_image_raw_ready -> Current /" + str(self.robot_namespace) + "/camera/depth/image_raw READY!")

            except:
                rospy.logerr("ROSbot_env::_check_camera_depth_image_raw_ready -> Current /" + str(self.robot_namespace) + "/camera/depth/image_raw not ready yet, retrying...")
        return self.camera_depth_image_raw

    def _check_camera_depth_points_ready(self):
        self.camera_depth_points = None
        rospy.logdebug("ROSbot::_check_camera_depth_points_ready -> Waiting for /" + str(self.robot_namespace) + "/camera/depth/points to be READY...")
        while self.camera_depth_points is None and not rospy.is_shutdown():
            try:
                self.camera_depth_points = rospy.wait_for_message("/" + str(self.robot_namespace) + "/camera/depth/points", PointCloud2, timeout=5.0)
                rospy.logdebug("ROSbot_env::_check_camera_depth_points_ready -> Current /" + str(self.robot_namespace) + "/camera/depth/points READY!")

            except:
                rospy.logerr("ROSbot_env::_check_camera_depth_points_ready -> Current /" + str(self.robot_namespace) + "/camera/depth/points not ready yet, retrying...")
        return self.camera_depth_points

    def _check_camera_rgb_image_raw_ready(self):
        self.camera_rgb_image_raw = None
        rospy.logdebug("ROSbot_env::_check_camera_rgb_image_raw_ready -> Waiting for /" + str(self.robot_namespace) + "/camera/rgb/image_raw to be READY...")
        while self.camera_rgb_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_rgb_image_raw = rospy.wait_for_message("/" + str(self.robot_namespace) + "/camera/rgb/image_raw", Image, timeout=5.0)
                rospy.logdebug("ROSbot_env::_check_camera_rgb_image_raw_ready -> Current /" + str(self.robot_namespace) + "/camera/rgb/image_raw READY!")

            except:
                rospy.logerr("ROSbot_env::_check_camera_rgb_image_raw_ready -> Current /" + str(self.robot_namespace) + "/camera/rgb/image_raw not ready yet, retrying...")
        return self.camera_rgb_image_raw


    def _odom_callback(self, data):
        self.odom = data

    def _imu_callback(self, data):
        self.imu = data

    def _camera_depth_image_raw_callback(self, data):
        self.camera_depth_image_raw = data

    def _camera_depth_points_callback(self, data):
        self.camera_depth_points = data

    def _camera_rgb_image_raw_callback(self, data):
        self.camera_rgb_image_raw = data

    def _laser_scan_callback(self, data):
        self.laser_scan = data

    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(50)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug(
                "No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("ROSBot_env::_check_publishers_connection -> cmd_vel_pub Publisher Connected")
        rospy.logdebug("ROSBot_env::_check_publishers_connection -> All Publishers READY")

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
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

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("Husarion Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        self.wait_until_twist_achieved(cmd_vel_value,
                                       epsilon,
                                       update_rate)

    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate, angular_speed_noise=0.005):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        Bare in mind that the angular wont be controled , because its too imprecise.
        We will only consider to check if its moving or not inside the angular_speed_noise fluctiations it has.
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        rospy.logwarn("START wait_until_twist_achieved...")

        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0

        rospy.logdebug("Desired Twist Cmd>>" + str(cmd_vel_value))
        rospy.logdebug("epsilon>>" + str(epsilon))

        linear_speed = cmd_vel_value.linear.x
        angular_speed = cmd_vel_value.angular.z

        angular_speed_is = self.check_angular_speed_dir(
            angular_speed, angular_speed_noise)

        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon

        while not rospy.is_shutdown():
            current_odometry = self._check_odom_ready()
            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = current_odometry.twist.twist.angular.z

            rospy.logdebug("Linear VEL=" + str(odom_linear_vel) +
                           ", ?RANGE=[" + str(linear_speed_minus) + ","+str(linear_speed_plus)+"]")
            rospy.logdebug("Angular VEL=" + str(odom_angular_vel) +
                           ", angular_speed asked=[" + str(angular_speed)+"]")

            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (
                odom_linear_vel > linear_speed_minus)

            odom_angular_speed_is = self.check_angular_speed_dir(
                odom_angular_vel, angular_speed_noise)

            # We check if its turning in the same diretion or has stopped
            angular_vel_are_close = (angular_speed_is == odom_angular_speed_is)

            if linear_vel_are_close and angular_vel_are_close:
                rospy.logwarn("Reached Velocity!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rospy.logwarn("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time - start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time)+"]")

        rospy.logwarn("END wait_until_twist_achieved...")

        return delta_time

    def check_angular_speed_dir(self, angular_speed, angular_speed_noise):
        """
        It States if the speed is zero, posititive or negative
        """
        # We check if odom angular speed is positive or negative or "zero"
        if (-angular_speed_noise < angular_speed <= angular_speed_noise):
            angular_speed_is = 0
        elif angular_speed > angular_speed_noise:
            angular_speed_is = 1
        elif angular_speed <= angular_speed_noise:
            angular_speed_is = -1
        else:
            angular_speed_is = 0
            rospy.logerr("Angular Speed has wrong value=="+str(angular_speed))

    def get_imu(self):
        return self.imu
        
    def get_odom(self):
        return self.odom

    def get_camera_depth_image_raw(self):
        return self.camera_depth_image_raw

    def get_camera_depth_points(self):
        return self.camera_depth_points

    def get_camera_rgb_image_raw(self):
        return self.camera_rgb_image_raw

    def get_laser_scan(self):
        return self.laser_scan
