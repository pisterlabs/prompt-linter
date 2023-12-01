import numpy
import rospy
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist



class DuckieTownEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self):
        """
        Initializes a new DuckieTown environment.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        
        The Sensors: The sensors accessible are the ones considered usefull for AI learning.
        
        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot
        * /camera/rgb/image_raw: RGB camera
        
        Actuators Topic List: /cmd_vel, 
        
        Args:
        """
        rospy.logdebug("Start DuckieTownEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # Using namespace
        self.robot_name_space = "/robot1"
        
        # Control parameters
        self.odom_topic = self.robot_name_space + "/odom"
        self.camera_topic = self.robot_name_space + "/duckbot/camera1/image_raw"
        self.camera2_topic = self.robot_name_space + "/duckbot/camera2/image_raw"
        
        self.detection_results = None

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(DuckieTownEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")




        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber( self.odom_topic, Odometry, self._odom_callback)
        rospy.Subscriber( self.camera_topic, Image, self._camera_rgb_image_raw_callback)
        rospy.Subscriber( self.camera2_topic, Image, self._camera2_rgb_image_raw_callback)

        self._cmd_vel_pub = rospy.Publisher(self.robot_name_space + '/cmd_vel', Twist, queue_size=1)

        self._check_publishers_connection()

        self.gazebo.pauseSim()
        
        rospy.logdebug("Finished DuckieTownEnv INIT...")

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
        self._check_camera_rgb_image_raw_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for "+ self.odom_topic +" to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message(self.odom_topic, Odometry, timeout=5.0)
                rospy.logdebug("Current "+ self.odom_topic + " READY=>")

            except:
                rospy.logerr("Current "+ self.odom_topic + " not ready yet, retrying to get odom")

        return self.odom
        
        
    def _check_camera_rgb_image_raw_ready(self):
        self.camera_rgb_image_raw = None
        rospy.logdebug("Waiting for "+ self.camera_topic + " to be READY...")
        while self.camera_rgb_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_rgb_image_raw = rospy.wait_for_message(self.camera_topic, Image, timeout=5.0)
                rospy.logdebug("Current "+ self.camera_topic + " READY=>")

            except:
                rospy.logerr("Current "+ self.camera_topic + " not ready yet, retrying for getting camera_rgb_image_raw")
        return self.camera_rgb_image_raw
        

    def _odom_callback(self, data):
        self.odom = data
        
    def _camera_rgb_image_raw_callback(self, data):
        self.camera_rgb_image_raw = data
        
    def _camera2_rgb_image_raw_callback(self, data):
        self.camera2_rgb_image_raw = data
        
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
    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10, use_offset=True):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achieved reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return: 
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("DuckBot Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        
    def get_odom(self):
        return self.odom
        
    def get_camera_rgb_image_raw(self):
        return self.camera_rgb_image_raw
        
    def get_camera2_rgb_image_raw(self):
        return self.camera2_rgb_image_raw
