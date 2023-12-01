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
from openai_ros.openai_ros_common import ROSLauncher


class WalrusUprightEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, ros_ws_abspath):
        """
        Initializes a new WalrusUprightEnv environment.
        Walrus doesnt use controller_manager, therefore we wont reset the TODO: check controllers
        controllers in the standard fashion. For the moment we wont reset them.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered usefull for AI learning.

        Sensor Topic List: TODO update with actual sensors from Walrus
        * /odom : Odometry readings of the Base of the Robot
        * /imu: Inertial Mesuring Unit that gives relative accelerations and orientations.
        * /scan: Laser Readings

        Actuators Topic List: /cmd_vel, #TODO update

        Args:
        """
        rospy.logdebug("Start WalrusUprightEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # We launch the ROSlaunch that spawns the robot into the world
        ROSLauncher(rospackage_name="walrus_gazebo",
                    launch_file_name="put_robot_in_world_upright.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = ['diff_vel_controller','joint_state_controller']

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(WalrusUprightEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=True,
                                            start_init_physics_parameters=True)




        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/diff_vel_controller/odom", Odometry, self._odom_callback)
        rospy.Subscriber("/imu/data", Imu, self._imu_callback)
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_l_callback)
        rospy.Subscriber("/scan_1", LaserScan, self._laser_scan_r_callback)

        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self._check_publishers_connection()

        self.gazebo.pauseSim()

        # The odometry from diff_vel_controller doesn't reset after each run.
        # Instead, track elapsed odometry before each run, so that it can be subtracted to give actual relative odometry.
        #self.elapsed_x = 0.0
        #self.elapsed_y = 0.0
        #self.elapsed_z = 0.0
        #self.odom = Odometry() # Blank odometry message


        rospy.logdebug("Finished WalrusUprightEnv INIT...")

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
        self._check_laser_scan_l_ready()
        self._check_laser_scan_r_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /diff_vel_controller/odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/diff_vel_controller/odom", Odometry, timeout=5.0)
                rospy.logdebug("Current /diff_vel_controller/odom READY=>")

            except:
                rospy.logerr("Current /diff_vel_controller/odom not ready yet, retrying for getting odom")
        
        self.elapsed_x = self.odom.pose.pose.position.x
        self.elapsed_y = self.odom.pose.pose.position.y
        self.elapsed_z = self.odom.pose.pose.position.z

        return self.odom


    def _check_imu_ready(self):
        self.imu = None
        rospy.logdebug("Waiting for /imu to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message("/imu/data", Imu, timeout=5.0)
                rospy.logdebug("Current /imu/data READY=>")

            except:
                rospy.logerr("Current /imu/data not ready yet, retrying for getting imu")

        return self.imu


    def _check_laser_scan_l_ready(self):
        self.laser_scan_l = None
        rospy.logdebug("Waiting for /scan to be READY...")
        while self.laser_scan_l is None and not rospy.is_shutdown():
            try:
                self.laser_scan_l = rospy.wait_for_message("/scan", LaserScan, timeout=1.0)
                rospy.logdebug("Current /scan READY=>")

            except:
                rospy.logerr("Current /scan not ready yet, retrying for getting laser_scan")
        return self.laser_scan_l

    def _check_laser_scan_r_ready(self):
        self.laser_scan_r = None
        rospy.logdebug("Waiting for /scan_1 to be READY...")
        while self.laser_scan_r is None and not rospy.is_shutdown():
            try:
                self.laser_scan_r = rospy.wait_for_message("/scan_1", LaserScan, timeout=1.0)
                rospy.logdebug("Current /scan_1 READY=>")

            except:
                rospy.logerr("Current /scan not ready yet, retrying for getting laser_scan")
        return self.laser_scan_r


    def _odom_callback(self, data):
        self.odom = data

    def _imu_callback(self, data):
        self.imu = data

    def _laser_scan_l_callback(self, data):
        self.laser_scan_l = data

    def _laser_scan_r_callback(self, data):
        self.laser_scan_r = data

    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")

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
        
        # Since odometry drifts and cannot be reset between runs, save the elapsed pose for later processing.
        self.elapsed_x = self.odom.pose.pose.position.x
        self.elapsed_y = self.odom.pose.pose.position.y
        self.elapsed_z = self.odom.pose.pose.position.z


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
        rospy.logdebug("Walrus Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        #self.wait_until_twist_achieved(cmd_vel_value,epsilon,update_rate)
        # Weplace a waitof certain amiunt of time, because this twist achived doesnt work properly
        time.sleep(0.2)

    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
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
            # IN Walrus the odometry angular readings are inverted, so we have to invert the sign. TODO check this
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


    def get_odom(self):

        # Uncorrected, drifting odom:
        odom_drift = self.odom

        # Initialize relative odom as equal to drifting odom
        rel_odom = odom_drift

        # Now, subtract elapsed odometry and return corrected relative odometry
        rel_odom.pose.pose.position.x -= self.elapsed_x
        rel_odom.pose.pose.position.y -= self.elapsed_y
        rel_odom.pose.pose.position.z -= self.elapsed_z

        # Print an output for debugging
        rospy.logdebug("Uncorrected odom position: " + str(self.odom.pose.pose.position))
        rospy.logdebug("Corrected odom position: " + str(rel_odom.pose.pose.position))
        return rel_odom

    def get_imu(self):
        return self.imu

    def get_laser_scan_l(self):
        return self.laser_scan_l

    def get_laser_scan_r(self):
        return self.laser_scan_r