import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Int16
from std_msgs.msg import Float32
# from sensor_msgs.msg import JointState
# from sensor_msgs.msg import Image
import cv2

from nav_msgs.msg import Odometry
# from mav_msgs.msg import Actuators
# from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Vector3
# from sensor_msgs.msg import Range
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped,Pose
from std_msgs.msg import Empty
# from trajectory_msgs.msg import MultiDOFJointTrajectory
from openai_ros.openai_ros_common import ROSLauncher
from rotors_control.srv import *
from visualization_msgs.msg import Marker
from tf import TransformListener
from geometry_msgs.msg import Point
import tf.transformations as transformations
from tf.transformations import euler_from_quaternion
from scipy.io import savemat
import numba as nb
import math
import os
from numba.typed import List
import tf
from gazebo_msgs.srv import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from gazebo_msgs.msg import ModelState 


@nb.jit(nopython=True)
def parallel_process_point_cloud(trans, rot, data):
    EPS = 2.220446049250313e-16 * 4.0
    new_points = []
    for i in range(data.shape[0]):
        pt = [data[i][0],data[i][1],data[i][2]]

        ##########################
        # adapt from https://answers.ros.org/question/249433/tf2_ros-buffer-transform-pointstamped/
        quat = [
            rot[0],
            rot[1],
            rot[2],
            rot[3]
        ]

        ##########################
        # Return homogeneous rotation matrix from quaternion from https://github.com/davheld/tf/blob/master/src/tf/transformations.py
        q = numpy.array(quat[:4], dtype=numpy.float64)
        nq = numpy.dot(q, q)
        if nq < EPS:
            mat = numpy.identity(4)
        else:
            q *= math.sqrt(2.0 / nq)
            q = numpy.outer(q, q)
            mat =  numpy.array((
                (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
                (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
                (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
                (                0.0,                 0.0,                 0.0, 1.0)
                ), dtype=numpy.float64)
        ##########################
        pt_np = [pt[0], pt[1], pt[2], 1.0]
        pt_in_map_np = numpy.dot(mat, numpy.array(pt_np))

        pt_in_map_x = pt_in_map_np[0] + trans[0]
        pt_in_map_y = pt_in_map_np[1] + trans[1]
        pt_in_map_z = pt_in_map_np[2] + trans[2]

        new_pt = [pt_in_map_x,pt_in_map_y,pt_in_map_z]
        ##########################
        # new_pt = transform_point(trans,rot, pt)
        new_points.append(new_pt)
    
    return new_points


class FireflyDroneEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, ros_ws_abspath):
        """
        Initializes a new FireflyDroneEnv environment.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered usefull for AI learning.

        Sensor Topic List:
        * /Firefly_1/odometry_sensor1/odometry
        * /Firefly_1/command/motor_speed
        * /Firefly_2/odometry_sensor1/odometry
        * /Firefly_2/command/motor_speed

        Args:
        """
        rospy.logdebug("Start FireflyDroneEnv INIT...")

        # Variables that we give through the constructor.
        # None in this case

        # Internal Vars
        # Doesnt have any accesibles
        self.counter = 0
        self.counter1 = 0
        self.controllers_list = []

        self.shutdown_joy = 0

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(FireflyDroneEnv, self).__init__(controllers_list=self.controllers_list,
                                             robot_name_space=self.robot_name_space,
                                             reset_controls=False,
                                             start_init_physics_parameters=False,
                                             reset_world_or_sim="WORLD")

        self.gazebo.unpauseSim()

        # ROSLauncher(rospackage_name="rotors_gazebo",
        #             launch_file_name="crazyflie2_swarm_transport_example_2_agents.launch",
        #             ros_ws_abspath=ros_ws_abspath)

        # self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/firefly_1/ground_truth/imu", Imu, self._imu_callback1)
        rospy.Subscriber("/firefly_1/odometry_sensor1/odometry", Odometry, self._odometry_callback1)
        # self._cmd_motor_pub1 = rospy.Publisher('/firefly_1/command/motor_speed', Actuators, queue_size=1)
        self._cmd_pos_pub1 = rospy.Publisher('/firefly_1/cmd_pos', PoseStamped, queue_size=1)

        rospy.Subscriber("/firefly_2/ground_truth/imu", Imu, self._imu_callback2)
        rospy.Subscriber("/firefly_2/odometry_sensor1/odometry", Odometry, self._odometry_callback2)
        # self._cmd_motor_pub2 = rospy.Publisher('/firefly_2/command/motor_speed', Actuators, queue_size=1)
        self._cmd_pos_pub2 = rospy.Publisher('/firefly_2/cmd_pos', PoseStamped, queue_size=1)

        rospy.Subscriber("/goal_pos", Vector3, self._joy_goal_callback)
        rospy.Subscriber("/shutdown_signal", Int16, self._shutdown_collect_callback)
        rospy.Subscriber('/bar/ground_truth/odometry', Odometry, self._bar_callback)

        # kinect cameras:top and front
        # self.tf = TransformListener()
        # rospy.Subscriber("/camera_ir_top/camera/depth/points", PointCloud2, self._point_callback_top)
        # rospy.wait_for_service('/gazebo/set_model_state')
        # self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        # rospy.Timer(rospy.Duration(0.15), self.set_pos_callback)
        # rospy.Timer(rospy.Duration(0.2), self.set_pos_callback_depth) #0.15
        # self.image_sub = rospy.Subscriber("/camera_ir_top/camera/depth/image_raw",Image,self.depth_callback_realtime)
        # self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("/camera_ir_top/camera/depth/image_raw",Image,self.depth_callback)
        # self.bridge = CvBridge()

        self.goal_pub_makers = rospy.Publisher('/goal_makers', Marker, queue_size=10)
        self.goal_pub_makers_c = rospy.Publisher('/corrective_goal_makers', Marker, queue_size=10)
        self.action_pub_makers_c = rospy.Publisher('/action_maker_c', Marker, queue_size=10)
        self.action_pub_makers = rospy.Publisher('/action_maker', Marker, queue_size=10)
        self.action_sequence_pub_makers = rospy.Publisher('/action_seq_maker', Marker, queue_size=100)
        self.action_sequence_pub_makers1 = rospy.Publisher('/action_seq_maker1', Marker, queue_size=100)

        self.pause_controller = rospy.Publisher('/pause_controller', Int16, queue_size=1)
        self.wind_controller_x = rospy.Publisher('/wind_force_x', Float32, queue_size=1)
        self.wind_controller_y = rospy.Publisher('/wind_force_y', Float32, queue_size=1)

        self._check_all_publishers_ready()

        self.gazebo.pauseSim()

        self.goal_joy = numpy.array([1.0,0.0,1.0])
        # self.space_3d = rospy.get_param("/firefly/3d_space")
        self.xy = rospy.get_param("/firefly/xy")

        rospy.logdebug("Finished FireflyEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def set_pos_callback(self,event):
        data = self.get_bar_odometry()
        b_pos = data.pose.pose.position
        objstate = SetModelStateRequest()  # Create an object of type SetModelStateRequest
        # set red cube pose
        objstate.model_state.model_name = "kinect_ros_3"
        objstate.model_state.pose.position.x = b_pos.x
        objstate.model_state.pose.position.y = b_pos.y
        objstate.model_state.pose.position.z = 3.0
        objstate.model_state.pose.orientation.w = 0.70738827
        objstate.model_state.pose.orientation.x = 0
        objstate.model_state.pose.orientation.y = 0.70682518
        objstate.model_state.pose.orientation.z = 0
        objstate.model_state.twist.linear.x = 0.0
        objstate.model_state.twist.linear.y = 0.0
        objstate.model_state.twist.linear.z = 0.0
        objstate.model_state.twist.angular.x = 0.0
        objstate.model_state.twist.angular.y = 0.0
        objstate.model_state.twist.angular.z = 0.0
        objstate.model_state.reference_frame = "world"
        result = self.set_state_service(objstate)

    #this callback function is for depth camera
    def set_pos_callback_depth(self,event):
        data = self.get_bar_odometry()
        b_pos = data.pose.pose.position

        state_msg = ModelState()
        state_msg.model_name = 'kinect_ros_3'
        state_msg.pose.position.x = b_pos.x
        state_msg.pose.position.y = b_pos.y
        state_msg.pose.position.z = 3.0
        state_msg.pose.orientation.x = -0.5
        state_msg.pose.orientation.y = 0.5
        state_msg.pose.orientation.z = 0.5
        state_msg.pose.orientation.w = 0.5
        state_msg.twist.linear.x = 0.0
        state_msg.twist.linear.y = 0.0
        state_msg.twist.linear.z = 0.0
        state_msg.twist.angular.x = 0.0
        state_msg.twist.angular.y = 0.0
        state_msg.twist.angular.z = 0.0
        state_msg.reference_frame = "world"

        result = self.set_state_service(state_msg)

    def set_pos_callback_depth_loop(self):
        data = self.get_bar_odometry()
        b_pos = data.pose.pose.position

        state_msg = ModelState()
        state_msg.model_name = 'kinect_ros_3'
        state_msg.pose.position.x = b_pos.x
        state_msg.pose.position.y = b_pos.y
        state_msg.pose.position.z = 3.0
        state_msg.pose.orientation.x = -0.5
        state_msg.pose.orientation.y = 0.5
        state_msg.pose.orientation.z = 0.5
        state_msg.pose.orientation.w = 0.5
        state_msg.twist.linear.x = 0.0
        state_msg.twist.linear.y = 0.0
        state_msg.twist.linear.z = 0.0
        state_msg.twist.angular.x = 0.0
        state_msg.twist.angular.y = 0.0
        state_msg.twist.angular.z = 0.0
        state_msg.reference_frame = "world"

        result = self.set_state_service(state_msg)
    
    def set_pos_callback_cloud_loop(self):
        data = self.get_bar_odometry()
        b_pos = data.pose.pose.position

        state_msg = ModelState()
        state_msg.model_name = 'kinect_ros_3'
        state_msg.pose.position.x = b_pos.x
        state_msg.pose.position.y = b_pos.y
        state_msg.pose.position.z = 3.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0.70682518
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0.70738827
        state_msg.twist.linear.x = 0.0
        state_msg.twist.linear.y = 0.0
        state_msg.twist.linear.z = 0.0
        state_msg.twist.angular.x = 0.0
        state_msg.twist.angular.y = 0.0
        state_msg.twist.angular.z = 0.0
        state_msg.reference_frame = "world"

        result = self.set_state_service(state_msg)


    def _bar_callback(self,data):
        self.bar_odometry = data

        b_pos = data.pose.pose.position
        br = tf.TransformBroadcaster()
        # br.sendTransform((b_pos.x, b_pos.y, 3.0),
        #              tf.transformations.quaternion_from_euler(0, 1.57, 0),
        #              rospy.Time.now(),
        #              "kinect_camera",
        #              "world")
        br.sendTransform((b_pos.x, b_pos.y, 3.0),
                     tf.transformations.quaternion_from_euler(0.0, 3.14, 1.57),
                     rospy.Time.now(),
                     "camera_link",
                     "world")


    def _joy_goal_callback(self,data):
        if data.x >0:
            self.goal_joy[0] -= 0.03
        elif data.x < 0: 
            self.goal_joy[0] += 0.03
        else:
            pass
        
        if data.y >0:
            self.goal_joy[1] += 0.03
        elif data.y < 0: 
            self.goal_joy[1] -= 0.03
        else:
            pass
            
        if data.z >0:
            self.goal_joy[2] += 0.03
        elif data.z < 0: 
            self.goal_joy[2] -= 0.03
        else:
            pass
        
    def _shutdown_collect_callback(self,data):
        self.shutdown_joy = data.data

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
        self._check_imu_ready()
        self._check_odometry_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_odometry_ready(self):
        self.odometry1 = None
        rospy.logdebug("Waiting for /firefly_1/odometry_sensor1/odometry")
        self.odometry2 = None
        rospy.logdebug("Waiting for /firefly_2/odometry_sensor1/odometry")
        self.bar_odometry = None 
        rospy.logdebug("Waiting for /bar/ground_truth/odometry")

        while self.odometry1 is None and not rospy.is_shutdown():
            try:
                self.odometry1 = rospy.wait_for_message("/firefly_1/odometry_sensor1/odometry", Odometry, timeout=5.0)
                rospy.logdebug("Current/firefly_1/odometry_sensor1/odometry READY=>")
            except:
                rospy.logerr("Current /firefly_1/odometry_sensor1/odometry not ready yet, retrying for getting later")

        while self.odometry2 is None and not rospy.is_shutdown():
            try:
                self.odometry2 = rospy.wait_for_message("/firefly_2/odometry_sensor1/odometry", Odometry, timeout=5.0)
                rospy.logdebug("Current/firefly_2/odometry_sensor1/odometry READY=>")
            except:
                rospy.logerr("Current /firefly_2/odometry_sensor1/odometry not ready yet, retrying for getting later")
        
        while self.bar_odometry is None and not rospy.is_shutdown():
            try:
                self.bar_odometry = rospy.wait_for_message("/bar/ground_truth/odometry", Odometry, timeout=5.0)
                rospy.logdebug("Current/bar/ground_truth/odometry READY=>")
            except:
                rospy.logerr("Current /bar/ground_truth/odometry not ready yet, retrying for getting later")


    def _check_imu_ready(self):
        self.imu1 = None
        rospy.logdebug("Waiting for /firefly_1/ground_truth/imu to be READY...")
        self.imu2 = None
        rospy.logdebug("Waiting for /firefly_2/ground_truth/imu to be READY...")

        while self.imu1 is None and not rospy.is_shutdown():
            try:
                self.imu1 = rospy.wait_for_message("/firefly_1/ground_truth/imu", Imu, timeout=5.0)
                rospy.logdebug("Current/firefly_1/ground_truth/imu READY=>")

            except:
                rospy.logerr(
                    "Current /firefly_1/ground_truth/imu not ready yet, retrying for getting imu")

        while self.imu2 is None and not rospy.is_shutdown():
            try:
                self.imu2 = rospy.wait_for_message("/firefly_2/ground_truth/imu", Imu, timeout=5.0)
                rospy.logdebug("Current/firefly_2/ground_truth/imu READY=>")

            except:
                rospy.logerr(
                    "Current /firefly_2/ground_truth/imu not ready yet, retrying for getting imu")

    def _imu_callback1(self, data):
        self.imu1 = data
    
    def _imu_callback2(self, data):
        self.imu2 = data

    def _odometry_callback1(self, data):
        self.odometry1 = data
    
    def _odometry_callback2(self, data):
        self.odometry2 = data

    def depth_callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            cv_image_array = np.array(cv_image, dtype = np.dtype('f8'))
            cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)

            # cv2.imshow("Image window", cv_image_norm)
            # cv2.waitKey(3)
            folder_path = "/home/wawa/catkin_meta/src/MBRL_transport/depth_images"
            wind_condition_x = 0.8
            wind_condition_y = 0.0
            L = 1.2
            fileName = folder_path+"/wind"+ "_x"+str(wind_condition_x) + "_y"+str(wind_condition_y)

            fileName += "_" + str(2) + "agents"+"_"+"L"+str(L)
            
            if not os.path.exists(fileName):
                os.makedirs(fileName)

            # print(cv_image_norm.shape)
            dic_d = {"depth":cv_image_norm} 
            savemat(fileName+"/{0}.mat".format(self.counter1), dic_d)  
            self.counter1+=1
        except CvBridgeError as e:
            print(e)  

    def depth_callback_realtime(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            cv_image_array = np.array(cv_image, dtype = np.dtype('f8'))
            cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)

            self.cv_image_norm = cv_image_norm
            # cv2.imshow("Image window", cv_image_norm)
            # cv2.waitKey(3)
           
        except CvBridgeError as e:
            print(e)     

    def get_depth_map(self):
        return self.cv_image_norm 

    def _point_callback_top(self, data):
        # We get the laser scan data
        u1_odm = self.get_odometry1()
        u2_odm = self.get_odometry2()
        bar_odm = self.get_bar_odometry()

        b_roll, b_pitch, b_yaw = self.get_orientation_euler1(bar_odm.pose.pose.orientation)
        b_pos = bar_odm.pose.pose.position
        uav1_pos = u1_odm.pose.pose.position
        uav2_pos = u2_odm.pose.pose.position

        max_x = 4
        max_y = 2
        max_z = 2
        
        #also track the two drones
        observations = [round(uav1_pos.x,8)/max_x,
                        round(uav1_pos.y,8)/max_y,
                        round(uav1_pos.z,8)/max_z,
                        round(uav2_pos.x,8)/max_x,
                        round(uav2_pos.y,8)/max_y,
                        round(uav2_pos.z,8)/max_z,
                    round(b_pos.x,8)/max_x,
                    round(b_pos.y,8)/max_y,
                    round(b_pos.z,8)/max_z,
                    round(b_roll,8),
                    round(b_pitch,8),
                    round(b_yaw,8)]
        
        configuration_sys = numpy.array(observations)

        points_top = data

        # transform points from camera_link to world
        try:
            (trans,rot) = self.tf.lookupTransform("/world", "/camera_link", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("tf transform error!!!")
        
        data_p = list(point_cloud2.read_points(points_top, field_names=('x', 'y', 'z'), skip_nans=True))
        
        # # print(len(list(data_p)))
        # # print(len(list(data_p))>0)
        # data_p = list(data_p)
        new_points = parallel_process_point_cloud(List(trans),List(rot),numpy.array(data_p))

        # save data 
        # fileName = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_and_configurations_additional/firefly_points_3d"
        fileName = "/home/wawa/catkin_meta/src/MBRL_transport/point_clouds_obs/firefly_points_3d"

        #wind speed: 0.0, 0.3, 0.5, 0.8
        wind_condition_x = 0.0
        wind_condition_y = 0.0
        L = 0.6
        fileName += "_wind"+ "_x"+str(wind_condition_x) + "_y"+str(wind_condition_y)

        fileName += "_" + str(2) + "agents"+"_"+"L"+str(L)

        if not os.path.exists(fileName):
            os.makedirs(fileName)

        fileName1 = fileName + "/"+str(self.counter)+".mat"

        # we need to transform the points into the world coordinate before saving it
        # Pxy1 = np.array(Pxy)[:,[1,0,2]]
        # Pxy1[:,1] = -Pxy1[:,1]
    
        mdic = {"configuration": configuration_sys, "top":numpy.array(new_points)}
        
        savemat(fileName1, mdic)
        self.counter+=1
    

    def _check_all_publishers_ready(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rospy.logdebug("START ALL SENSORS READY")
        self._check_cmd_pos_pub_connection()
        rospy.logdebug("ALL SENSORS READY")

    def _check_cmd_pos_pub_connection(self):

        rate1 = rospy.Rate(10)  # 10hz
        rate2 = rospy.Rate(10)  # 10hz

        while self._cmd_pos_pub1.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug(
                "No susbribers to _cmd_pos_pub1 yet so we wait and try again")
            try:
                rate1.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_pos_pub1 Publisher Connected")

        while self._cmd_pos_pub2.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug(
                "No susbribers to _cmd_pos_pub2 yet so we wait and try again")
            try:
                rate2.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_pos_pub2 Publisher Connected")

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
    
    def takeoff(self, L):
        """
        Sends the takeoff command and checks it has taken of
        It unpauses the simulation and pauses again
        to allow it to be a self contained action
        """
        
        self.gazebo.unpauseSim()
        # time.sleep(5.0)

        # create PoseStamped
        pose1 = PoseStamped()
        pose1.header.stamp = rospy.Time.now()
        pose1.pose = Pose()
        # pose1.pose.position.x = 1.3 
        if not self.xy:
            pose1.pose.position.x = 1.0+L/2.0
            pose1.pose.position.y = 0 
            pose1.pose.position.z = 1.6
            pose1.pose.orientation.w = 0.0

            # create PoseStamped
            pose2 = PoseStamped()
            pose2.header.stamp = rospy.Time.now()
            pose2.pose = Pose()
            # pose2.pose.position.x = 0.7 
            pose2.pose.position.x = 1.0-L/2.0 
            pose2.pose.position.y = 0 
            pose2.pose.position.z = 1.6 
            pose2.pose.orientation.w = 0.0
        else:
            pose1.pose.position.x = 1.0+L/2.0
            pose1.pose.position.y = 1.0
            pose1.pose.position.z = 1.6
            pose1.pose.orientation.w = 0.0

            # create PoseStamped
            pose2 = PoseStamped()
            pose2.header.stamp = rospy.Time.now()
            pose2.pose = Pose()
            # pose2.pose.position.x = 0.7 
            pose2.pose.position.x = 1.0-L/2.0
            pose2.pose.position.y = 1.0
            pose2.pose.position.z = 1.6 
            pose2.pose.orientation.w = 0.0

		# send PoseStamped
        self._cmd_pos_pub1.publish(pose1)
        self._cmd_pos_pub2.publish(pose2)

        time.sleep(12.0)

        self.gazebo.pauseSim()
    

    def move_pos_base(self, dp, L):
        """
        accept real dx and dz [-0.5,0.5] [-1.0,1.0] metre
        
        """
        
        self._check_cmd_pos_pub_connection()

        assert(dp.shape[0] == 3)
        uav1_odm = self.get_odometry1()
        uav2_dom = self.get_odometry2()

        uav1_pos = uav1_odm.pose.pose.position
        uav2_pos = uav2_dom.pose.pose.position

        goal = numpy.zeros(3)
        goal[0] = (uav1_pos.x+uav2_pos.x)/2+dp[0]
        goal[1] = (uav1_pos.y+uav2_pos.y)/2+dp[1]
        goal[2] = (uav1_pos.z+uav2_pos.z)/2+dp[2]

        # create PoseStamped
        pose1 = PoseStamped()
        pose1.header.stamp = rospy.Time.now()
        pose1.pose = Pose()
        pose1.pose.position.x = goal[0]+L/2 
        pose1.pose.position.y = goal[1]
        pose1.pose.position.z = goal[2]
        pose1.pose.orientation.w = 0.0

        # create PoseStamped
        pose2 = PoseStamped()
        pose2.header.stamp = rospy.Time.now()
        pose2.pose = Pose()
        pose2.pose.position.x = goal[0]-L/2 
        pose2.pose.position.y = goal[1]
        pose2.pose.position.z = goal[2]
        pose2.pose.orientation.w = 0.0

		# send PoseStamped
        self._cmd_pos_pub1.publish(pose1)
        self._cmd_pos_pub2.publish(pose2)

        self.wait_time_for_execute_movement()

        return goal


    def wait_time_for_execute_movement(self):
        """
        Because this Parrot Drone position is global, we really dont have
        a way to know if its moving in the direction desired, because it would need
        to evaluate the diference in position and speed on the local reference.
        """
        time.sleep(0.15)


    def check_array_similar(self, ref_value_array, check_value_array, epsilon):
        """
        It checks if the check_value id similar to the ref_value
        """
        rospy.logwarn("ref_value_array="+str(ref_value_array))
        rospy.logwarn("check_value_array="+str(check_value_array))
        return numpy.allclose(ref_value_array, check_value_array, atol=epsilon)

    def get_imu1(self):
        return self.imu1

    def get_imu2(self):
        return self.imu2

    def get_odometry1(self):
        return self.odometry1

    def get_odometry2(self):
        return self.odometry2

    def get_bar_odometry(self):
        return self.bar_odometry

    # def get_points_top(self):
    #     gen = point_cloud2.read_points(self.points_top, field_names=("x", "y", "z"), skip_nans=True)
    #     # time.sleep(1)
    #     return list(gen)
    #     # time.sleep(1)

    # def get_points_top_and_configuration(self):
    #     # transform points from camera_link to world
    #     try:
    #         (trans,rot) = self.tf.lookupTransform("/world", "/camera_link", rospy.Time(0))
    #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #         rospy.logerr("tf transform error!!!")
        
    #     new_points = []
    #     for x, y, z in point_cloud2.read_points(self.points_top, field_names=('x', 'y', 'z'), skip_nans=True):
    #         pt = Point()
    #         pt.x, pt.y, pt.z = x, y, z

    #         new_pt = self.transform_point(trans,rot, pt)
    #         new_points.append(new_pt)

    #     return new_points,self.configuration_sys
    #     # time.sleep(1)

    # def get_points_front(self):
    #     # transform points from camera_link1 to world
    #     try:
    #         (trans,rot) = self.tf.lookupTransform("/camera_link1", "/world", rospy.Time(0))
    #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #         rospy.logerr("tf transform error!!!")
        
    #     new_points = []
    #     for x, y, z in point_cloud2.read_points(self.points_front, field_names=('x', 'y', 'z'), skip_nans=True):
    #         pt = Point()
    #         pt.x, pt.y, pt.z = x, y, z

    #         new_pt = self.transform_point(trans,rot, pt)
    #         new_points.append(new_pt)

    #     return numpy.array(new_points)

    @staticmethod
    def get_orientation_euler1(quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw


