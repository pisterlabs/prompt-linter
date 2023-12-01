#!/usr/bin/env python3

'''
LAST UPDATE: 2021.12.10

AUTHOR:     OPENAI_ROS
            Neset Unver Akmandor (NUA)

E-MAIL: akmandor.n@northeastern.edu

DESCRIPTION: TODO...

REFERENCES:
[1] 

NUA TODO:
'''

import numpy
import rospy
import time
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from openai_ros import robot_real_env
#from openai_ros.openai_ros_common import ROSLauncher

'''
DESCRIPTION: TODO...Superclass for all CubeSingleDisk environments.
'''
class StretchRealEnv(robot_real_env.RobotRealEnv):

    '''
    DESCRIPTION: TODO...
        Initializes a new TurtleBot3Env environment.
        Turtlebot3 doesnt use controller_manager, therefore we wont reset the 
        controllers in the standard fashion. For the moment we wont reset them.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        
        The Sensors: The sensors accesible are the ones considered usefull for AI learning.
        
        Sensor Topic List:
        * /odom: Odometry readings of the Base of the Robot
        * /camera/depth/image_raw: 2d Depth image of the depth sensor.
        * /camera/depth/points: Pointcloud sensor readings
        * /camera/rgb/image_raw: RGB camera
        * /kobuki/laser/scan: Laser Readings
        
        Actuators Topic List: /cmd_vel
        
        Args:
    '''
    def __init__(self, robot_namespace="", velocity_control_msg=""):

        # NUA TODO: This following required if SubprocVecEnv is used! 
        #rospy.init_node('robot_env_' + str(robot_namespace), anonymous=True, log_level=rospy.ERROR)

        rospy.logdebug("stretchreal_env::__init__ -> START...")

        #self.controllers_list = ["imu"]
        self.controllers_list = []
        self.robot_namespace = robot_namespace

        self.odom_msg_name = "/odom"
        self.imu_msg_name = "/imu_mobile_base"
        self.laser_scan_msg_name = "/scan"
        self.camera_depth_image_msg_name = "/camera/depth/image_rect_raw"
        self.camera_depth_pc2_msg_name = "/camera/depth/color/points"
        self.camera_rgb_image_msg_name = "/camera/color/image_raw"

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        #super(TurtleBot3RealEnv, self).__init__(controllers_list=self.controllers_list,
        #                                        robot_namespace=self.robot_namespace,
        #                                        reset_controls=False,
        #                                        start_init_physics_parameters=False,
        #                                        initial_pose=self.initial_pose)

        super(StretchRealEnv, self).__init__(robot_namespace=self.robot_namespace)

        #self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber(self.odom_msg_name, Odometry, self._odom_callback)
        rospy.Subscriber(self.imu_msg_name, Imu, self._imu_callback)
        rospy.Subscriber(self.laser_scan_msg_name, LaserScan, self._laser_scan_callback)
        #rospy.Subscriber("/" + str(self.robot_namespace) + "/odom", Odometry, self._odom_callback)
        #rospy.Subscriber("/" + str(self.robot_namespace) + "/imu", Imu, self._imu_callback)
        #rospy.Subscriber("/" + str(self.robot_namespace) + "/scan", LaserScan, self._laser_scan_callback)
        #rospy.Subscriber("/camera/depth/image_raw", Image, self._camera_depth_image_raw_callback)
        #rospy.Subscriber("/camera/depth/points", PointCloud2, self._camera_depth_points_callback)
        #rospy.Subscriber("/camera/rgb/image_raw", Image, self._camera_rgb_image_raw_callback)

        self._cmd_vel_pub = rospy.Publisher(velocity_control_msg, Twist, queue_size=1)
        self._check_publishers_connection()

        #self.gazebo.pauseSim()

        rospy.logdebug("stretchreal_env::__init__ -> END")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    '''
    DESCRIPTION: TODO...Checks that all the sensors, publishers and other simulation systems are operational.
    '''
    def _check_all_systems_ready(self):

        self._check_all_sensors_ready()
        return True

    # TurtleBot3 Env virtual methods
    # ----------------------------

    '''
    DESCRIPTION: TODO...
    '''
    def _check_all_sensors_ready(self):
        
        rospy.logdebug("stretchreal_env::_check_all_sensors_ready -> START...")
        self._check_odom_ready()
        #self._check_imu_ready()
        self._check_laser_scan_ready()
        #self._check_camera_depth_image_raw_ready()
        #self._check_camera_depth_points_ready()
        #self._check_camera_rgb_image_raw_ready()
        rospy.logdebug("stretchreal_env::_check_all_sensors_ready -> END")

    '''
    DESCRIPTION: TODO...
    '''
    def _check_odom_ready(self):

        self.odom = None
        rospy.logdebug("stretchreal_env::_check_odom_ready -> Waiting for " + self.odom_msg_name + " to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message(self.odom_msg_name, Odometry, timeout=1.0)
                rospy.logdebug("stretchreal_env::_check_odom_ready -> " + self.odom_msg_name + " is READY!")

            except:
                rospy.logerr("stretchreal_env::_check_odom_ready -> " + self.odom_msg_name + " is not ready yet, retrying...")

        return self.odom

    '''
    DESCRIPTION: TODO...
    '''
    def _check_imu_ready(self):

        self.imu = None
        rospy.logdebug("stretchreal_env::_check_imu_ready -> Waiting for " + self.imu_msg_name + " to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message(self.imu_msg_name, Imu, timeout=1.0)
                rospy.logdebug("stretchreal_env::_check_imu_ready -> " + self.imu_msg_name + " is READY!")

            except:
                rospy.logerr("stretchreal_env::_check_imu_ready -> " + self.imu_msg_name + " is not ready yet, retrying...")

        return self.imu

    '''
    DESCRIPTION: TODO...
    '''
    def _check_laser_scan_ready(self):

        self.laser_scan = None
        rospy.logdebug("stretchreal_env::_check_laser_scan_ready -> Waiting for " + self.laser_scan_msg_name + " to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message(self.laser_scan_msg_name, LaserScan, timeout=1.0)
                rospy.logdebug("stretchreal_env::_check_laser_scan_ready -> " + self.laser_scan_msg_name + " is READY!")

            except:
                rospy.logerr("stretchreal_env::_check_laser_scan_ready -> " + self.laser_scan_msg_name + " is not ready yet, retrying...")
        return self.laser_scan

    '''
    DESCRIPTION: TODO...
    '''
    def _check_camera_depth_image_raw_ready(self):
        self.camera_depth_image_raw = None
        rospy.logdebug("stretchreal_env::_check_camera_depth_image_raw_ready -> Waiting for " + self.camera_depth_image_msg_name + " to be READY...")
        while self.camera_depth_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_depth_image_raw = rospy.wait_for_message(self.camera_depth_image_msg_name, Image, timeout=5.0)
                rospy.logdebug("stretchreal_env::_check_camera_depth_image_raw_ready -> " + self.camera_depth_image_msg_name + " is READY!")

            except:
                rospy.logerr("stretchreal_env::_check_camera_depth_image_raw_ready -> " + self.camera_depth_image_msg_name + " is not ready yet, retrying...")
        return self.camera_depth_image_raw
    
    '''
    DESCRIPTION: TODO...
    '''
    def _check_camera_depth_points_ready(self):
        self.camera_depth_points = None
        rospy.logdebug("stretchreal_env::_check_camera_depth_points_ready -> Waiting for " + self.camera_depth_pc2_msg_name + " to be READY...")
        while self.camera_depth_points is None and not rospy.is_shutdown():
            try:
                self.camera_depth_points = rospy.wait_for_message(self.camera_depth_pc2_msg_name, PointCloud2, timeout=5.0)
                rospy.logdebug("stretchreal_env::_check_camera_depth_points_ready -> " + self.camera_depth_pc2_msg_name + " is READY!")

            except:
                rospy.logerr("stretchreal_env::_check_camera_depth_points_ready -> " + self.camera_depth_pc2_msg_name + " is not ready yet, retrying...")
        return self.camera_depth_points
    
    '''
    DESCRIPTION: TODO...
    '''
    def _check_camera_rgb_image_raw_ready(self):
        self.camera_rgb_image_raw = None
        rospy.logdebug("stretchreal_env::_check_camera_rgb_image_raw_ready -> Waiting for " + self.camera_rgb_image_msg_name + " to be READY...")
        while self.camera_rgb_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_rgb_image_raw = rospy.wait_for_message(self.camera_rgb_image_msg_name, Image, timeout=5.0)
                rospy.logdebug("stretchreal_env::_check_camera_rgb_image_raw_ready -> " + self.camera_rgb_image_msg_name + " is READY!")

            except:
                rospy.logerr("stretchreal_env::_check_camera_rgb_image_raw_ready -> " + self.camera_rgb_image_msg_name + " is not ready yet, retrying...")
        return self.camera_rgb_image_raw

    '''
    DESCRIPTION: TODO...
    '''
    def _odom_callback(self, data):
        self.odom = data

    '''
    DESCRIPTION: TODO...
    '''
    def _imu_callback(self, data):
        self.imu = data

    '''
    DESCRIPTION: TODO...
    '''
    def _laser_scan_callback(self, data):
        self.laser_scan = data

    '''
    DESCRIPTION: TODO...
    '''
    def _camera_depth_image_raw_callback(self, data):
        self.camera_depth_image_raw = data
    
    '''
    DESCRIPTION: TODO...
    '''
    def _camera_depth_points_callback(self, data):
        self.camera_depth_points = data
    
    '''
    DESCRIPTION: TODO...
    '''
    def _camera_rgb_image_raw_callback(self, data):
        self.camera_rgb_image_raw = data

    '''
    DESCRIPTION: TODO...Checks that all the publishers are working
                :return:
    ''' 
    def _check_publishers_connection(self):

        rate = rospy.Rate(50)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("stretchreal_env::_check_publishers_connection -> No subscribers to cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("stretchreal_env::_check_publishers_connection -> cmd_vel_pub Publisher Connected")
        rospy.logdebug("stretchreal_env::_check_publishers_connection -> All Publishers READY")

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    '''
    DESCRIPTION: TODO...Inits variables needed to be initialised each time we reset at the start of an episode.
    ''' 
    def _init_env_variables(self):
        raise NotImplementedError()

    '''
    DESCRIPTION: TODO...Calculates the reward to give based on the observations given.
    ''' 
    def _compute_reward(self, observations, done):
        raise NotImplementedError()

    '''
    DESCRIPTION: TODO...Applies the given action to the simulation.
    ''' 
    def _set_action(self, action):
        raise NotImplementedError()

    '''
    DESCRIPTION: TODO...
    '''
    def _get_obs(self):
        raise NotImplementedError()

    '''
    DESCRIPTION: TODO...Checks if episode done based on observations given.
    ''' 
    def _is_done(self, observations):
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------

    '''
    DESCRIPTION: TODO...
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return: 
    ''' 
    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10):

        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("stretchreal_env::move_base -> cmd_vel_value: " + str(cmd_vel_value))
        
        self._check_publishers_connection()
        #print("stretchreal_env::move_base -> cmd_vel_value: ")
        #print(str(cmd_vel_value))
        self._cmd_vel_pub.publish(cmd_vel_value)
        #self.wait_until_twist_achieved(cmd_vel_value,epsilon,update_rate)

        time.sleep(0.1)

    '''
    DESCRIPTION: TODO...
        We wait for the cmd_vel twist given to be reached by the robot reading
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
    ''' 
    '''
    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate):

        rospy.logdebug("START wait_until_twist_achieved...")

        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0
        epsilon = 0.05

        rospy.logdebug("Desired Twist Cmd>>" + str(cmd_vel_value))
        rospy.logdebug("epsilon>>" + str(epsilon))

        linear_speed = cmd_vel_value.linear.x
        angular_speed = cmd_vel_value.angular.z

        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon
        angular_speed_plus = angular_speed + epsilon
        angular_speed_minus = angular_speed - epsilon

        while not rospy.is_shutdown():
            current_odometry = self._check_odom_ready()
            # IN turtlebot3 the odometry angular readings are inverted, so we have to invert the sign.
            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = -1*current_odometry.twist.twist.angular.z

            rospy.logdebug("Linear VEL=" + str(odom_linear_vel) + ", ?RANGE=[" + str(linear_speed_minus) + ","+str(linear_speed_plus)+"]")
            rospy.logdebug("Angular VEL=" + str(odom_angular_vel) + ", ?RANGE=[" + str(angular_speed_minus) + ","+str(angular_speed_plus)+"]")

            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
            angular_vel_are_close = (odom_angular_vel <= angular_speed_plus) and (odom_angular_vel > angular_speed_minus)

            if linear_vel_are_close and angular_vel_are_close:
                rospy.logdebug("Reached Velocity!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rospy.logdebug("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time- start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time)+"]")

        rospy.logdebug("END wait_until_twist_achieved...")

        return delta_time
    '''

    '''
    DESCRIPTION: TODO...
    ''' 
    def get_odom(self):
        return self.odom

    '''
    DESCRIPTION: TODO...
    ''' 
    def get_imu(self):
        return self.imu

    '''
    DESCRIPTION: TODO...
    ''' 
    def get_laser_scan(self):
        return self.laser_scan

    '''
    DESCRIPTION: TODO...
    ''' 
    def get_camera_depth_image_raw(self):
        return self.camera_depth_image_raw
    
    '''
    DESCRIPTION: TODO...
    '''    
    def get_camera_depth_points(self):
        return self.camera_depth_points
    
    '''
    DESCRIPTION: TODO...
    ''' 
    def get_camera_rgb_image_raw(self):
        return self.camera_rgb_image_raw
